"""
src.sbso.training_loop

Phase: Phase 4
Purpose: SBSO training orchestrator (report Section 3.3.4). Runs episodes: samples an
    opponent, runs MCTS to pick each decision (or single-pass sampling under No-MCTS),
    accumulates (state -> best_strategy) training pairs, triggers DSPy recompilation on
    schedule (unless No-DSPy), and snapshots self-checkpoints. Judge pruning is inside
    MCTS (disabled under No-Judge). Produces the training data the LoRA step consumes.

    Stage 1 runs this against MockSimulationBackend + MockJudge (root_state_builder
    defaults to a mock-shaped dict). Stage 2/3 pass a root_state_builder that calls
    PyBulletSimulationBackend.root_state(...) and an outcome_extractor reading the real
    match result, without changing this class.

    KNOWN SIMPLIFICATION (Stage 1/2 token-run): one MCTS decision per "episode" here,
    not one per control cycle across a full ~300-cycle match. Stage 3's full training
    loop should run MCTS once per decision cycle within a match, accumulating many
    training pairs per episode - this class's per-episode loop still applies, just
    called multiple times within a match by the Stage 3 match-runner.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Callable, Optional

from src.sbso.ablation_strategies import AblationConfig


class SBSOTrainer:
    def __init__(
        self,
        ablation: AblationConfig,
        mcts,
        opponent_pool,
        scheduler,
        checkpoint_mgr,
        dspy_compiler,
        strategies,
        episodes: int = 20,
        seed: int = 0,
        root_state_builder: Optional[Callable[[int, str], Any]] = None,
        outcome_extractor: Optional[Callable[[Any, Any], float]] = None,
        on_recompile: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.ablation = ablation
        self.mcts = mcts
        self.opponent_pool = opponent_pool
        self.scheduler = scheduler
        self.checkpoint_mgr = checkpoint_mgr
        self.dspy_compiler = dspy_compiler
        self.strategies = list(strategies)
        self.episodes = episodes
        self._rng = random.Random(seed)
        self.training_pairs: list = []
        self.prompt_program = "SA_BASE_PROMPT"
        self.dominant_strategy: Optional[str] = None   # set alongside prompt_program on each recompile
        self._win_history: deque[float] = deque(maxlen=scheduler.window_w)
        self.root_state_builder = root_state_builder or self._default_root_state_builder
        self.outcome_extractor = outcome_extractor or self._default_outcome_extractor
        self.on_recompile = on_recompile  # e.g. strategy_agent.update_prompt_program

    def _default_root_state_builder(self, episode: int, opponent: str):
        return {"depth": 0, "progress": 0.0, "lssd": f"ep{episode}", "opponent": opponent}

    def _default_outcome_extractor(self, mcts_result, root_state) -> float:
        # Stage 1 placeholder; real match win/loss lands with the real backend (Stage 3).
        return 1.0 if self._rng.random() > 0.5 else 0.0

    def _rolling_winrate(self):
        if not self._win_history:
            return None
        return sum(self._win_history) / len(self._win_history)

    def run(self) -> dict:
        for ep in range(self.episodes):
            opp = self.opponent_pool.sample(ep, self.checkpoint_mgr.has_checkpoint())
            root_state = self.root_state_builder(ep, opp)

            if self.ablation.mcts_enabled:
                result = self.mcts.search(root_state)
                # If the backend snapshots live physics (PyBulletSimulationBackend),
                # restore to root so tree exploration never leaks into the real env.
                restore = getattr(self.mcts.backend, "_restore", None)
                if restore is not None and hasattr(root_state, "pybullet_state_id"):
                    restore(root_state.pybullet_state_id)
                # Episode-tagged triple (item 1 fix): lets RealDSPyCompiler select
                # examples spread across matches instead of a flat recency tail.
                self.training_pairs.append((ep, *result.training_pair))
                won = self.outcome_extractor(result, root_state)
            else:
                # No-MCTS ablation: single-pass direct sampling, no tree search.
                best = self._rng.choice(self.strategies)
                self.training_pairs.append((ep, root_state, best))
                won = self.outcome_extractor(None, root_state)

            self._win_history.append(won)

            if self.ablation.dspy_enabled:
                do, _why = self.scheduler.should_recompile(ep, self._rolling_winrate())
                if do:
                    self.prompt_program = self.dspy_compiler.compile(
                        self.training_pairs, self.prompt_program)
                    self.dominant_strategy = getattr(self.dspy_compiler, "last_dominant_strategy", None)
                    if self.on_recompile is not None:
                        self.on_recompile(self.prompt_program)

            self.checkpoint_mgr.maybe_snapshot(ep, self.prompt_program, self.dominant_strategy)

        return {
            "episodes": self.episodes,
            "training_pairs": len(self.training_pairs),
            "dspy_recompiles": getattr(self.dspy_compiler, "compile_count", 0),
            "checkpoints": len(self.checkpoint_mgr._checkpoints),
            "final_prompt": self.prompt_program,
        }

    def _mcts_decision(self, root_state_builder) -> tuple:
        """Run one MCTS search; if the backend snapshots live physics (PyBulletSimulationBackend),
        restore the live env to the root AFTER search so tree exploration never leaks into
        the real match."""
        root_state = root_state_builder()
        result = self.mcts.search(root_state)
        restore = getattr(self.mcts.backend, "_restore", None)
        if restore is not None and hasattr(root_state, "pybullet_state_id"):
            restore(root_state.pybullet_state_id)
        return result