"""Unit tests for the SLM agent layer against the mock client (Phase 2)."""

from __future__ import annotations

import unittest

from src.agents.actuator_bridge import ActuatorBridge
from src.agents.opponent_analysis_agent import OpponentAnalysisAgent
from src.agents.schemas import (
    MacroStrategy,
    MacroStrategyDecision,
    OpponentAnalysis,
    OpponentBehavior,
    PerceptionState,
    TacticalCommand,
    TacticalKeyword,
    DistanceLabel,
    DirectionLabel,
    EdgeLabel,
    MomentumLabel,
)
from src.agents.strategy_agent import StrategyAgent
from src.agents.tactical_execution_agent import TacticalExecutionAgent
from src.inference.slm_client import MockSLMClient, ScriptedSLMClient


def _perception():
    return PerceptionState(
        lssd_text="opp=near,dir=FC;edge=safe;mom=fwd",
        opp_distance=DistanceLabel.NEAR,
        opp_direction=DirectionLabel.FRONT_CENTER,
        edge=EdgeLabel.SAFE,
        momentum=MomentumLabel.FORWARD,
        opp_distance_m=0.15,
    )


class TestAgentsMock(unittest.TestCase):
    def test_oaa_returns_valid_and_stamps_frame(self):
        oaa = OpponentAnalysisAgent(MockSLMClient(seed=1))
        out = oaa.analyze(["opp=far,dir=FC;edge=safe;mom=still"], frame_index=7)
        self.assertIsInstance(out, OpponentAnalysis)
        self.assertEqual(out.frame_stamp, 7)
        self.assertIn(out.behavior, list(OpponentBehavior))

    def test_sa_returns_valid_strategy(self):
        sa = StrategyAgent(MockSLMClient(seed=2))
        prev = OpponentAnalysis(behavior=OpponentBehavior.AGGRESSIVE, frame_stamp=5)
        out = sa.decide(_perception(), prev)
        self.assertIsInstance(out, MacroStrategyDecision)
        self.assertIn(out.strategy, list(MacroStrategy))

    def test_sa_handles_no_prior_oaa(self):
        sa = StrategyAgent(MockSLMClient(seed=3))
        out = sa.decide(_perception(), None)   # first frame, no t-1 classification
        self.assertIsInstance(out, MacroStrategyDecision)

    def test_tea_returns_valid_keyword(self):
        tea = TacticalExecutionAgent(MockSLMClient(seed=4))
        macro = MacroStrategyDecision(strategy=MacroStrategy.CHARGE)
        out = tea.execute(_perception(), macro)
        self.assertIsInstance(out, TacticalCommand)
        self.assertIn(out.keyword, list(TacticalKeyword))

    def test_actuator_bridge_maps_all_keywords_in_range(self):
        bridge = ActuatorBridge()
        for k in TacticalKeyword:
            left, right = bridge.to_pwm(TacticalCommand(keyword=k))
            self.assertGreaterEqual(left, -1.0)
            self.assertLessEqual(left, 1.0)
            self.assertGreaterEqual(right, -1.0)
            self.assertLessEqual(right, 1.0)

    def test_charge_forward_is_full_ahead(self):
        bridge = ActuatorBridge()
        left, right = bridge.to_pwm(TacticalCommand(keyword=TacticalKeyword.CHARGE_FORWARD))
        self.assertEqual((left, right), (1.0, 1.0))

    def test_scripted_client_returns_fixed_output(self):
        scripted = ScriptedSLMClient({
            MacroStrategyDecision: MacroStrategyDecision(strategy=MacroStrategy.FLANK),
        })
        sa = StrategyAgent(scripted)
        out = sa.decide(_perception(), None)
        self.assertEqual(out.strategy, MacroStrategy.FLANK)


if __name__ == "__main__":
    unittest.main()