"""
src.sbso.ablation_strategies

Phase: Phase 4
Purpose: Ablation configuration (plan Assumption A3). Each variant disables exactly one
    component: No-SA (OAA feeds TEA directly), No-MCTS (single-pass sampling replaces the
    tree search), No-DSPy (prompt frozen at initial state), No-Judge (branches unfiltered).
    Consumed by the SBSO trainer to toggle the corresponding code path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AblationConfig:
    mcts_enabled: bool = True
    dspy_enabled: bool = True
    judge_enabled: bool = True
    sa_enabled: bool = True

    @classmethod
    def for_variant(cls, strategy: str) -> "AblationConfig":
        strategy = (strategy or "none").lower()
        if strategy in ("none", ""):
            return cls()
        if strategy == "no_sa":
            return cls(sa_enabled=False)
        if strategy == "no_mcts":
            return cls(mcts_enabled=False)
        if strategy == "no_dspy":
            return cls(dspy_enabled=False)
        if strategy == "no_judge":
            return cls(judge_enabled=False)
        raise ValueError(f"Unknown ablation strategy: {strategy}")