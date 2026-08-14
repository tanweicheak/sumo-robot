"""
src.agents.monolithic_reasoning_agent

Phase: Phase 1 (Baseline 3 - zero-shot monolithic single-SLM call)
Purpose: MRA - one direct SLM call maps the current perception state straight to a
    TacticalKeyword, with none of the multi-agent decomposition (no separate
    opponent-analysis step, no separate macro-strategy step) - the "everything in one
    shot" baseline the multi-agent architecture (Benchmark 1/Benchmark 2) is compared
    against.

    Deliberately reuses the SAME TacticalKeyword vocabulary and (by extension) the same
    ActuatorBridge as the multi-agent path (tactical_execution_agent.py), so both
    conditions produce PWM through an identical final mapping - the comparison this
    baseline exists for is about the REASONING architecture (one call vs. a pipeline),
    not a different action space. Structurally mirrors tactical_execution_agent.py on
    purpose, for the same reason.

    Scope note: this file implements the agent itself. Wiring it into a standalone
    eval/watch loop (perceive -> decide -> bridge -> env.step(), without the multi-agent
    NODE_SEQUENCE) is separate follow-up work, not included here.
"""

from __future__ import annotations

from src.agents.schemas import PerceptionState, TacticalCommand
from src.inference.slm_client import SLMClient

_MRA_SYSTEM = (
    "You are the sole reasoning agent for an autonomous sumo robot - there is no "
    "separate opponent-analysis step and no separate strategy step; you must decide "
    "the robot's next motor action directly from the current sensor state in one shot. "
    "Given the current semantic state, emit exactly one motor keyword: charge_forward, "
    "arc_left, arc_right, pivot_left, pivot_right, reverse, stop. Choose the keyword "
    "that both assesses the situation and executes the best action for it."
)


def build_mra_prompt(perception: PerceptionState) -> str:
    return (
        f"{_MRA_SYSTEM}\n\n"
        f"Current state: {perception.lssd_text}\n\n"
        f"Emit the motor keyword."
    )


class MonolithicReasoningAgent:
    def __init__(self, client: SLMClient) -> None:
        self.client = client

    def decide(self, perception: PerceptionState) -> TacticalCommand:
        prompt = build_mra_prompt(perception)
        return self.client.generate_structured(prompt, TacticalCommand)