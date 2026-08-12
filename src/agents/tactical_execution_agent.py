"""
src.agents.tactical_execution_agent

Phase: Phase 2
Purpose: Tactical Execution Agent (TEA) - Phi-4-mini SLM node (report Section 3.3.2.1).
    Converts the selected macro-strategy plus current state into a concrete
    TacticalKeyword (charge_forward, arc_left, pivot_right, reverse, stop, ...). The
    Actuator Bridge maps that keyword to PWM. In Phase 3 this call is wrapped by Outlines
    constrained decoding so the keyword is schema-guaranteed on first generation.
"""

from __future__ import annotations

from src.agents.schemas import MacroStrategyDecision, PerceptionState, TacticalCommand
from src.inference.slm_client import SLMClient

_TEA_SYSTEM = (
    "You are the Tactical Execution Agent for an autonomous sumo robot. Given the "
    "current semantic state and the chosen macro strategy, emit exactly one motor "
    "keyword: charge_forward, arc_left, arc_right, pivot_left, pivot_right, reverse, "
    "stop. Choose the keyword that best executes the strategy for the current state."
)


def build_tea_prompt(perception: PerceptionState, macro: MacroStrategyDecision) -> str:
    return (
        f"{_TEA_SYSTEM}\n\n"
        f"Current state: {perception.lssd_text}\n"
        f"Macro strategy: {macro.strategy.value}\n\n"
        f"Emit the motor keyword."
    )


class TacticalExecutionAgent:
    def __init__(self, client: SLMClient) -> None:
        self.client = client

    def execute(
        self,
        perception: PerceptionState,
        macro: MacroStrategyDecision,
    ) -> TacticalCommand:
        prompt = build_tea_prompt(perception, macro)
        return self.client.generate_structured(prompt, TacticalCommand)