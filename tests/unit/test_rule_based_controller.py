"""Unit tests for Baseline 1 - pure logic, no PyBullet needed."""

from __future__ import annotations

import unittest

import numpy as np

from src.baselines.rule_based_controller import (
    RuleBasedController,
    RuleBasedParams,
    RuleState,
)


def _obs(tof, ir, enc=(0, 0, 0, 0)):
    return {
        "tof": np.array(tof, dtype=np.float32),
        "ir": np.array(ir, dtype=np.float32),
        "encoder": np.array(enc, dtype=np.float32),
    }


class TestRuleBasedController(unittest.TestCase):
    def setUp(self):
        self.ctrl = RuleBasedController(RuleBasedParams())
        self.n = 7  # tof ray count
        self.far = [1.5] * self.n  # nothing detected

    def test_search_when_nothing_detected(self):
        left, right = self.ctrl(_obs(self.far, [0.0, 0.0]))
        self.assertEqual(self.ctrl.state, RuleState.SEARCH)
        # Spin in place: wheels counter-rotate.
        self.assertAlmostEqual(left, -right, places=5)
        self.assertNotEqual(left, 0.0)

    def test_attack_charges_when_centered(self):
        tof = list(self.far)
        tof[3] = 0.5  # center ray sees a close opponent
        left, right = self.ctrl(_obs(tof, [0.0, 0.0]))
        self.assertEqual(self.ctrl.state, RuleState.ATTACK)
        self.assertAlmostEqual(left, 1.0, places=5)
        self.assertAlmostEqual(right, 1.0, places=5)

    def test_attack_steers_toward_offset_opponent(self):
        tof = list(self.far)
        tof[0] = 0.6  # opponent toward the left rays
        left, right = self.ctrl(_obs(tof, [0.0, 0.0]))
        self.assertEqual(self.ctrl.state, RuleState.ATTACK)
        # Turning left means the left wheel is slower than the right.
        self.assertLess(left, right)

    def test_edge_avoidance_overrides_attack(self):
        # Opponent dead ahead AND a front IR probe over the edge: edge wins.
        tof = list(self.far)
        tof[3] = 0.3
        left, right = self.ctrl(_obs(tof, [0.95, 0.0]))
        self.assertEqual(self.ctrl.state, RuleState.EDGE_AVOID)
        # First edge phase is reverse: both wheels negative.
        self.assertLess(left, 0.0)
        self.assertLess(right, 0.0)

    def test_edge_maneuver_runs_reverse_then_pivot(self):
        p = RuleBasedParams()
        ctrl = RuleBasedController(p)
        # Trigger edge, then feed safe obs; maneuver should complete on its own.
        ctrl(_obs(self.far, [0.95, 0.0]))  # trigger + first reverse cycle
        phases = []
        for _ in range(p.reverse_cycles + p.pivot_cycles + 1):
            ctrl(_obs(self.far, [0.0, 0.0]))
            phases.append(ctrl._edge_phase)
        # It must return to idle by the end of the scripted maneuver.
        self.assertEqual(ctrl._edge_phase, "idle")

    def test_output_always_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            tof = rng.uniform(0.1, 1.5, self.n).tolist()
            ir = rng.uniform(0.0, 1.0, 2).tolist()
            left, right = self.ctrl(_obs(tof, ir))
            self.assertGreaterEqual(left, -1.0)
            self.assertLessEqual(left, 1.0)
            self.assertGreaterEqual(right, -1.0)
            self.assertLessEqual(right, 1.0)


if __name__ == "__main__":
    unittest.main()