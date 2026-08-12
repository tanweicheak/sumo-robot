"""
src.orchestration.graph_builder

Phase: Phase 2
Purpose: The decision-cycle dataflow. Node functions (perception -> oaa -> sa -> tea ->
    bridge) are the single source of truth, each pure over SumoRobotState and timed.
    build_langgraph wires them into a real LangGraph StateGraph (the report's specified
    orchestrator); the SumoController can also run them directly for dev/CI without a
    LangGraph install. The pipelined design lives in the DATA: SA reads
    prev_opponent_analysis (frame t-1), so SA has no data dependency on the OAA node,
    which is what permits true concurrency as a Phase 3 optimization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentDeps:
    """Bundle of the instantiated agents/components a decision cycle needs."""
    perception_agent: object
    oaa: object
    sa: object
    tea: object
    bridge: object
    hcma: Optional[object] = None


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


# --- Pure node functions: (state, deps) -> dict of state updates ---------------

def perception_node(state, deps: AgentDeps) -> dict:
    t0 = time.perf_counter()
    ps = deps.perception_agent.perceive(state.raw_tof, state.raw_ir, state.raw_encoder)
    hist = list(state.lssd_history) + [ps.lssd_text]
    timing = dict(state.timing_ms); timing["pa"] = _ms(t0)
    return {"perception": ps, "lssd_history": hist, "timing_ms": timing}


def oaa_node(state, deps: AgentDeps) -> dict:
    t0 = time.perf_counter()
    result = deps.oaa.analyze(state.lssd_history, state.frame_index)
    timing = dict(state.timing_ms); timing["oaa"] = _ms(t0)
    return {"opponent_analysis": result, "timing_ms": timing}


def sa_node(state, deps: AgentDeps) -> dict:
    t0 = time.perf_counter()
    # Pipelined staleness: SA reads the PREVIOUS frame's OAA classification.
    macro = deps.sa.decide(state.perception, state.prev_opponent_analysis)
    timing = dict(state.timing_ms); timing["sa"] = _ms(t0)
    return {"macro_strategy": macro, "timing_ms": timing}


def tea_node(state, deps: AgentDeps) -> dict:
    t0 = time.perf_counter()
    cmd = deps.tea.execute(state.perception, state.macro_strategy)
    timing = dict(state.timing_ms); timing["tea"] = _ms(t0)
    return {"tactical_command": cmd, "timing_ms": timing}


def bridge_node(state, deps: AgentDeps) -> dict:
    t0 = time.perf_counter()
    left, right = deps.bridge.to_pwm(state.tactical_command)
    timing = dict(state.timing_ms); timing["bridge"] = _ms(t0)
    return {"left_pwm": left, "right_pwm": right, "timing_ms": timing}


NODE_SEQUENCE = (perception_node, oaa_node, sa_node, tea_node, bridge_node)


# --- LangGraph assembly (the report's specified orchestrator) ------------------

def build_langgraph(deps: AgentDeps):
    """Wire the node functions into a compiled LangGraph StateGraph. Requires langgraph;
    the SumoController can instead run NODE_SEQUENCE directly when langgraph is absent."""
    from langgraph.graph import END, StateGraph

    from src.agents.schemas import SumoRobotState

    graph = StateGraph(SumoRobotState)

    # Bind deps into each node (LangGraph nodes take state only).
    graph.add_node("perception", lambda s: perception_node(s, deps))
    graph.add_node("oaa", lambda s: oaa_node(s, deps))
    graph.add_node("sa", lambda s: sa_node(s, deps))
    graph.add_node("tea", lambda s: tea_node(s, deps))
    graph.add_node("bridge", lambda s: bridge_node(s, deps))

    # Linear execution order. The OAA->SA edge is execution ordering only; SA's DATA
    # dependency is on prev_opponent_analysis (frame t-1), not this frame's OAA output.
    graph.set_entry_point("perception")
    graph.add_edge("perception", "oaa")
    graph.add_edge("oaa", "sa")
    graph.add_edge("sa", "tea")
    graph.add_edge("tea", "bridge")
    graph.add_edge("bridge", END)

    return graph.compile()