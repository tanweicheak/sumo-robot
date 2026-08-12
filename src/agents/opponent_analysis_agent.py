"""
src.agents.opponent_analysis_agent

Phase: Phase 2
Purpose: Opponent Analysis Agent (OAA) - Phi-4-mini SLM node (report Section 3.3.2.1).
    Classifies opponent behavior (OpponentBehavior) from a short history of LSSD states.
    Under the pipelined design, OAA_t runs to produce the classification the Strategy
    Agent will read on the *next* frame; it stamps frame_stamp so staleness is explicit.
"""

from __future__ import annotations

from src.agents.schemas import OpponentAnalysis
from src.inference.slm_client import SLMClient

_OAA_SYSTEM = (
    "You are the Opponent Analysis Agent for an autonomous sumo robot. "
    "Given a short history of semantic state readings, classify the opponent's behavior "
    "as one of: aggressive, evasive, defensive, unknown. Base the call on how the "
    "opponent's distance and direction have changed over the history."
)


def build_oaa_prompt(lssd_history: list[str]) -> str:
    history_block = "\n".join(f"  t-{i}: {s}" for i, s in enumerate(reversed(lssd_history)))
    return (
        f"{_OAA_SYSTEM}\n\n"
        f"State history (most recent last):\n{history_block}\n\n"
        f"Classify the opponent's behavior."
    )


class OpponentAnalysisAgent:
    def __init__(self, client: SLMClient) -> None:
        self.client = client

    def analyze(self, lssd_history: list[str], frame_index: int) -> OpponentAnalysis:
        prompt = build_oaa_prompt(lssd_history)
        result: OpponentAnalysis = self.client.generate_structured(prompt, OpponentAnalysis)
        result.frame_stamp = frame_index   # mark which frame produced this classification
        return result