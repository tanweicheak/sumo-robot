"""End-to-end orchestration tests: full decision cycles with mock SLM (Phase 2)."""

from __future__ import annotations

import unittest

import numpy as np

from src.agents.actuator_bridge import ActuatorBridge
from src.agents.opponent_analysis_agent import OpponentAnalysisAgent
from src.agents.perception_agent import PerceptionAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.tactical_execution_agent import TacticalExecutionAgent
from src.inference.slm_client import MockSLMClient
from src.orchestration.graph_builder import AgentDeps
from src.orchestration.state import SumoController


def _make_controller(seed=0):
    client = MockSLMClient(seed=seed)
    deps = AgentDeps(
        perception_agent=PerceptionAgent(detect_range_m=1.4),
        oaa=OpponentAnalysisAgent(client),
        sa=StrategyAgent(client),
        tea=TacticalExecutionAgent(client),
        bridge=ActuatorBridge(),
    )
    return SumoController(deps)


class TestOrchestration(unittest.TestCase):
    def setUp(self):
        self.ctrl = _make_controller()
        self.ctrl.reset()
        self.n = 7

    def _sensors(self, min_ray=3, dist=0.15):
        tof = [1.5] * self.n
        tof[min_ray] = dist
        return np.array(tof, np.float32), np.array([0.29, 0.29], np.float32), np.array([0, 0, 0, 0], np.float32)

    def test_single_cycle_returns_valid_pwm(self):
        tof, ir, enc = self._sensors()
        left, right, state = self.ctrl.decide(tof, ir, enc)
        self.assertGreaterEqual(left, -1.0)
        self.assertLessEqual(left, 1.0)
        self.assertGreaterEqual(right, -1.0)
        self.assertLessEqual(right, 1.0)
        # Every stage populated.
        self.assertIsNotNone(state.perception)
        self.assertIsNotNone(state.opponent_analysis)
        self.assertIsNotNone(state.macro_strategy)
        self.assertIsNotNone(state.tactical_command)

    def test_timing_recorded_for_all_nodes(self):
        tof, ir, enc = self._sensors()
        _, _, state = self.ctrl.decide(tof, ir, enc)
        for node in ("pa", "oaa", "sa", "tea", "bridge"):
            self.assertIn(node, state.timing_ms)

    def test_pipelined_staleness_first_frame_has_no_prev_oaa(self):
        tof, ir, enc = self._sensors()
        # On frame 0, SA sees prev_opponent_analysis=None; must not crash.
        _, _, state = self.ctrl.decide(tof, ir, enc)
        self.assertEqual(state.frame_index, 0)
        # OAA produced this frame's classification, stamped frame 0.
        self.assertEqual(state.opponent_analysis.frame_stamp, 0)

    def test_pipelined_staleness_second_frame_reads_prev(self):
        tof, ir, enc = self._sensors()
        self.ctrl.decide(tof, ir, enc)          # frame 0
        _, _, state = self.ctrl.decide(tof, ir, enc)  # frame 1
        self.assertEqual(state.frame_index, 1)
        # SA on frame 1 read the classification stamped at frame 0.
        self.assertIsNotNone(state.prev_opponent_analysis)
        self.assertEqual(state.prev_opponent_analysis.frame_stamp, 0)

    def test_history_accumulates_and_bounds(self):
        tof, ir, enc = self._sensors()
        for _ in range(10):
            self.ctrl.decide(tof, ir, enc)
        # History deque is bounded at history_len (5).
        self.assertLessEqual(len(self.ctrl._history), 5)

    def test_reset_clears_pipeline_state(self):
        tof, ir, enc = self._sensors()
        self.ctrl.decide(tof, ir, enc)
        self.ctrl.reset()
        self.assertEqual(self.ctrl._frame, 0)
        self.assertIsNone(self.ctrl._prev_oaa)
        self.assertEqual(len(self.ctrl._history), 0)


if __name__ == "__main__":
    unittest.main()