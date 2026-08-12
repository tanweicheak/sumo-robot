"""Integration test: the LangGraph orchestration path (needs langgraph installed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("langgraph")

from src.agents.actuator_bridge import ActuatorBridge  # noqa: E402
from src.agents.opponent_analysis_agent import OpponentAnalysisAgent  # noqa: E402
from src.agents.perception_agent import PerceptionAgent  # noqa: E402
from src.agents.strategy_agent import StrategyAgent  # noqa: E402
from src.agents.tactical_execution_agent import TacticalExecutionAgent  # noqa: E402
from src.inference.slm_client import MockSLMClient  # noqa: E402
from src.orchestration.graph_builder import AgentDeps  # noqa: E402
from src.orchestration.state import SumoController  # noqa: E402


def test_langgraph_runs_full_cycle():
    client = MockSLMClient(seed=0)
    deps = AgentDeps(
        perception_agent=PerceptionAgent(detect_range_m=1.4),
        oaa=OpponentAnalysisAgent(client),
        sa=StrategyAgent(client),
        tea=TacticalExecutionAgent(client),
        bridge=ActuatorBridge(),
    )
    ctrl = SumoController(deps, use_langgraph=True)
    ctrl.reset()
    tof = np.array([1.5, 1.5, 1.5, 0.15, 1.5, 1.5, 1.5], np.float32)
    ir = np.array([0.29, 0.29], np.float32)
    enc = np.array([0, 0, 0, 0], np.float32)
    left, right, state = ctrl.decide(tof, ir, enc)
    assert -1.0 <= left <= 1.0 and -1.0 <= right <= 1.0
    assert state.tactical_command is not None