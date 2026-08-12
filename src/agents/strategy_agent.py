"""
src.agents.strategy_agent

Phase: Phase 2
Purpose: Strategy Agent (SA) - Phi-4-mini SLM node (report Section 3.3.2.1). Selects a
    macro-strategy (MacroStrategy) from the current PA state plus the opponent
    classification. Under the pipelined design SA reads the OAA result from frame t-1
    (never waiting on the fresh OAA_t), so it is decoupled from OAA latency. Absent
    entirely under the No-SA ablation (plan A3), where OAA feeds TEA directly.
"""

from __future__ import annotations

from typing import Optional

from src.agents.schemas import MacroStrategyDecision, OpponentAnalysis, PerceptionState
from src.inference.slm_client import SLMClient

_SA_SYSTEM = (
    "You are the Strategy Agent for an autonomous sumo robot. Given the current semantic "
    "state and the (possibly one-frame-old) opponent classification, choose one macro "
    "strategy: charge, flank, retreat, hold, evade_edge. Prioritize evade_edge whenever "
    "the edge reading is critical, regardless of the opponent."
)


def build_sa_prompt(
    perception: PerceptionState,
    prev_oaa: Optional[OpponentAnalysis],
    prompt_program: Optional[str] = None,
) -> str:
    behavior = prev_oaa.behavior.value if prev_oaa else "unknown"
    staleness = (
        f"(classification from frame {prev_oaa.frame_stamp})" if prev_oaa else "(no prior classification)"
    )
    # D5a fix: prompt_program is DSPy's compiled few-shot block (RealDSPyCompiler.compile()'s
    # return value). None (the default) reproduces the exact old behavior - base prompt
    # only - so Benchmark 1 (zero-shot multi-agent) is unaffected; Benchmark 2 (SBSO) and
    # the ablations pass it through once StrategyAgent is constructed with it.
    learned = f"\n\nLearned examples from training:\n{prompt_program}\n" if prompt_program else ""
    return (
        f"{_SA_SYSTEM}\n"
        f"{learned}"
        f"\nCurrent state: {perception.lssd_text}\n"
        f"Opponent behavior: {behavior} {staleness}\n\n"
        f"Select the macro strategy."
    )


class StrategyAgent:
    def __init__(self, client: SLMClient, prompt_program: Optional[str] = None) -> None:
        self.client = client
        self.prompt_program = prompt_program  # None until the first DSPy recompile lands

    def update_prompt_program(self, prompt_program: Optional[str]) -> None:
        """Hot-swap the compiled prompt program. Call this after RecompilationScheduler
        fires and dspy_compiler.compile() returns a new version - SBSOTrainer itself
        doesn't hold a StrategyAgent reference, so this propagation happens in whatever
        calls SBSOTrainer.run() (the Stage 2/3 entry script / match-runner)."""
        self.prompt_program = prompt_program

    def decide(
        self,
        perception: PerceptionState,
        prev_oaa: Optional[OpponentAnalysis],
    ) -> MacroStrategyDecision:
        prompt = build_sa_prompt(perception, prev_oaa, self.prompt_program)
        return self.client.generate_structured(prompt, MacroStrategyDecision)