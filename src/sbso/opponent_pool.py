"""
src.sbso.opponent_pool

Phase: Phase 4 (Stage 3a)
Purpose: Training opponent sampling (report Section 3.2.1.2). Episodes 0..warmup use
    Tier1/Tier2 only (no self-checkpoint exists yet); afterward, sampling includes
    self-checkpoints so the agent stays robust against its own past strategies.

    Stage 3a: sample() used to return a bare string label ("self_checkpoint") with
    nothing real behind it. It now returns an OpponentDescriptor - for a real
    self-checkpoint pick, this actually pulls (episode, prompt_program) from
    SelfCheckpointManager and attaches a ready-to-use rollout_policy. Per D2, that
    rollout_policy (a MacroStrategyExecutorOpponent proxy, never live SLM) is what
    PyBulletSimulationBackend runs during MCTS tree search regardless of opponent kind;
    a live-SLM policy for the real, committed match continuation is a separate,
    not-yet-built hook (see STATE_TRANSFER_SUMMARY D2).
"""

from __future__ import annotations

import random
from typing import Optional

from src.sbso.macro_executor import MacroStrategyExecutorOpponent
from src.sbso.self_checkpoint_manager import SelfCheckpointManager
from src.agents.schemas import MacroStrategy


class OpponentDescriptor(str):
    """A string equal to `kind` ("baseline1" / "baseline2" / "self_checkpoint") - so
    existing code/tests that compare sample()'s return value to a plain string, or put
    it in a set/dict, keep working exactly as before (str is hashable; str.__eq__
    compares by value). PLUS the real Stage 3a payload as extra attributes for new code
    that needs it: `episode` and `prompt_program` identify a sampled self-checkpoint;
    `rollout_policy` is the OpponentPolicy callable (obs -> (left_pwm, right_pwm)) MCTS
    rollouts should actually run for it."""

    def __new__(
        cls,
        kind: str,
        episode: Optional[int] = None,
        prompt_program: Optional[str] = None,
        rollout_policy: Optional[object] = None,
    ) -> "OpponentDescriptor":
        obj = str.__new__(cls, kind)
        obj.kind = kind
        obj.episode = episode
        obj.prompt_program = prompt_program
        obj.rollout_policy = rollout_policy
        return obj

    def __repr__(self) -> str:
        return (
            f"OpponentDescriptor(kind={self.kind!r}, episode={self.episode!r}, "
            f"prompt_program={'<set>' if self.prompt_program else None}, "
            f"rollout_policy={'<set>' if self.rollout_policy else None})"
        )


_STRATEGY_BY_NAME = {s.value: s for s in MacroStrategy}


def _strategy_from_prompt_program(prompt_program: str) -> MacroStrategy:
    """Best-effort read of which macro-strategy a compiled prompt program favored, so the
    rollout proxy isn't always a flat CHARGE regardless of what the checkpoint learned.
    RealDSPyCompiler._extract_prompt_program formats demos as "<lssd> -> <strategy>"
    lines, so counting strategy-name occurrences is a reasonable cheap heuristic. Falls
    back to CHARGE if nothing matches (e.g. MockDSPyCompiler's opaque "#recompiled_vN"
    placeholder string, used in Stage 1)."""
    counts = {name: prompt_program.count(name) for name in _STRATEGY_BY_NAME}
    best_name = max(counts, key=counts.get) if any(counts.values()) else None
    return _STRATEGY_BY_NAME.get(best_name, MacroStrategy.CHARGE)


class OpponentPool:
    def __init__(
        self,
        warmup_episodes: int = 500,
        total_episodes: int = 5000,
        seed: int = 0,
        self_checkpoint_manager: Optional[SelfCheckpointManager] = None,
        target_counts: Optional[dict] = None,
    ) -> None:
        self.warmup_episodes = warmup_episodes
        self.total_episodes = total_episodes
        self._rng = random.Random(seed)
        # Without a manager, self_checkpoint is never offered - there is nothing real to
        # sample. Pass the SAME SelfCheckpointManager instance the trainer uses as
        # checkpoint_mgr, so what this pool offers matches what actually got snapshotted.
        self.self_checkpoint_manager = self_checkpoint_manager
        # Target counts for the FULL run (report Section 3.2.1.2 / _shared_defaults.yaml
        # full_run_targets, e.g. {"baseline1": 1667, "baseline2": 1667, "self_checkpoint": 1666}
        # for total_episodes=5000). Pass the real config values explicitly; the computed
        # default below is only an even 3-way split, a reasonable fallback for ad-hoc runs.
        self.target_counts = target_counts or self._default_targets(total_episodes)
        self._counts = {"baseline1": 0, "baseline2": 0, "self_checkpoint": 0}

    @staticmethod
    def _default_targets(total_episodes: int) -> dict:
        base, remainder = divmod(total_episodes, 3)
        targets = {"baseline1": base, "baseline2": base, "self_checkpoint": base}
        for i, k in enumerate(("baseline1", "baseline2", "self_checkpoint")):
            if i < remainder:
                targets[k] += 1
        return targets

    def _weighted_choice(self, choices: list) -> str:
        """Weight toward whichever eligible kind is furthest BEHIND its target count -
        e.g. self_checkpoint accumulates a deficit during warmup (never eligible, so
        never sampled), so once eligible it gets picked more often until it catches up.
        Falls back to uniform once every eligible kind has met its target."""
        deficits = {k: max(0, self.target_counts.get(k, 0) - self._counts[k]) for k in choices}
        total = sum(deficits.values())
        if total <= 0:
            return self._rng.choice(choices)
        r = self._rng.uniform(0, total)
        upto = 0.0
        for k in choices:
            upto += deficits[k]
            if r <= upto:
                return k
        return choices[-1]   # floating-point edge case fallback

    def sample(self, episode: int, has_self_checkpoint: bool) -> OpponentDescriptor:
        choices = ["baseline1", "baseline2"]
        if (
            episode >= self.warmup_episodes
            and has_self_checkpoint
            and self.self_checkpoint_manager is not None
        ):
            choices.append("self_checkpoint")
        kind = self._weighted_choice(choices)

        if kind != "self_checkpoint":
            self._counts[kind] += 1
            return OpponentDescriptor(kind=kind)

        sampled = self.self_checkpoint_manager.sample()
        if sampled is None:
            # has_self_checkpoint was stale (e.g. race with a concurrent snapshot) -
            # fall back to a baseline instead of returning a checkpoint with no data.
            fallback = self._weighted_choice(["baseline1", "baseline2"])
            self._counts[fallback] += 1
            return OpponentDescriptor(kind=fallback)

        self._counts["self_checkpoint"] += 1
        ckpt_episode, prompt_program, dominant_strategy = sampled
        if dominant_strategy is not None:
            strategy = _STRATEGY_BY_NAME.get(dominant_strategy, MacroStrategy.CHARGE)
        else:
            # No precise value stored (e.g. MockDSPyCompiler, which has no structured
            # demos to compute one from) - fall back to the text heuristic.
            strategy = _strategy_from_prompt_program(prompt_program)
        return OpponentDescriptor(
            kind="self_checkpoint",
            episode=ckpt_episode,
            prompt_program=prompt_program,
            rollout_policy=MacroStrategyExecutorOpponent(strategy=strategy),
        )
