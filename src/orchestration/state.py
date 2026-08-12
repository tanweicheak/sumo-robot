"""
src.orchestration.state

Phase: Phase 2
Purpose: SumoController - drives the multi-agent decision loop across frames. Holds the
    rolling LSSD history and the previous-frame OAA classification (the pipelined
    staleness store), builds each frame's SumoRobotState, runs the decision cycle (pure
    node sequence by default, or the compiled LangGraph graph), and returns
    (left_pwm, right_pwm). Call reset() at the start of each episode.
"""

from __future__ import annotations

from collections import deque

from src.orchestration.graph_builder import NODE_SEQUENCE, AgentDeps


class SumoController:
    def __init__(self, deps: AgentDeps, history_len: int = 5, use_langgraph: bool = False) -> None:
        self.deps = deps
        self.history_len = history_len
        self.use_langgraph = use_langgraph
        self._graph = None
        self._history: deque[str] = deque(maxlen=history_len)
        self._prev_oaa = None
        self._frame = 0
        if use_langgraph:
            from src.orchestration.graph_builder import build_langgraph
            self._graph = build_langgraph(deps)

    def reset(self) -> None:
        if hasattr(self.deps.perception_agent, "reset"):
            self.deps.perception_agent.reset()
        self._history.clear()
        self._prev_oaa = None
        self._frame = 0

    def decide(self, raw_tof, raw_ir, raw_encoder):
        """Run one decision cycle. Returns (left_pwm, right_pwm, final_state)."""
        from src.agents.schemas import SumoRobotState

        state = SumoRobotState(
            frame_index=self._frame,
            raw_tof=list(raw_tof),
            raw_ir=list(raw_ir),
            raw_encoder=list(raw_encoder),
            prev_opponent_analysis=self._prev_oaa,
            lssd_history=list(self._history),
        )

        if self.use_langgraph:
            result = self._graph.invoke(state)
            # LangGraph may return a dict or a model depending on version; normalize.
            state = result if hasattr(result, "left_pwm") else SumoRobotState(**result)
        else:
            for node in NODE_SEQUENCE:
                state = state.model_copy(update=node(state, self.deps))

        # Roll pipelined state for the next frame.
        self._prev_oaa = state.opponent_analysis           # OAA_t becomes t-1 next frame
        if state.perception is not None:
            self._history.append(state.perception.lssd_text)
        self._frame += 1

        return state.left_pwm, state.right_pwm, state