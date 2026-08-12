"""
src.sbso.self_checkpoint_manager

Phase: Phase 4
Purpose: Self-checkpointing (report Section 3.2.1.3). Snapshots the DSPy-compiled prompt
    program every `interval` episodes (base Phi-4-mini weights are unchanged until the
    final LoRA step, so a self-checkpoint opponent is an in-memory prompt swap, not a
    model reload). Sampling draws uniformly across all past checkpoints.
"""

from __future__ import annotations

import random
from typing import Optional


class SelfCheckpointManager:
    def __init__(self, interval: int = 500, seed: int = 0) -> None:
        self.interval = interval
        self._rng = random.Random(seed)
        self._checkpoints: list[tuple[int, str, Optional[str]]] = []

    def maybe_snapshot(self, episode: int, prompt_program: str, dominant_strategy: Optional[str] = None) -> bool:
        if episode > 0 and episode % self.interval == 0:
            self._checkpoints.append((episode, prompt_program, dominant_strategy))
            return True
        return False

    def has_checkpoint(self) -> bool:
        return len(self._checkpoints) > 0

    def sample(self) -> tuple[int, str, Optional[str]] | None:
        if not self._checkpoints:
            return None
        return self._rng.choice(self._checkpoints)