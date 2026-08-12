"""
src.sbso.match_trainer

Phase: Phase 4 (Stage 3 - pilot & full run)
Purpose: MatchLevelSBSOTrainer runs FULL matches, not single decisions (report Section
    3.2.1.1: "a single match contributes many state-macro-strategy pairs"). At each
    macro-decision point: run MCTS on the live env's current physics state, execute the
    chosen strategy for real for `decision_cycles` control cycles, then re-decide. Repeats
    until the match terminates (win/loss/draw) or max_decisions_per_match is hit. This
    replaces Stage 1/2's one-decision-per-episode simplification (which existed only to
    prove plumbing fast) with the semantics the report actually specifies.

    Timing is instrumented per-decision and per-episode so a pilot run can extrapolate
    real cost/duration for the full 5x5000-episode run before committing to it.
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Any, Callable, Optional

from src.sbso.ablation_strategies import AblationConfig


class MatchLevelSBSOTrainer:
    def __init__(
        self,
        ablation: AblationConfig,
        mcts,
        opponent_pool,
        scheduler,
        checkpoint_mgr,
        dspy_compiler,
        strategies,
        env,
        executor,
        lssd_encoder,
        opponent_factory: Callable[[str], Any],
        episodes: int = 500,
        decision_cycles: int = 6,
        max_decisions_per_match: int = 50,
        seed: int = 0,
    ) -> None:
        self.ablation = ablation
        self.mcts = mcts
        self.opponent_pool = opponent_pool
        self.scheduler = scheduler
        self.checkpoint_mgr = checkpoint_mgr
        self.dspy_compiler = dspy_compiler
        self.strategies = list(strategies)
        self.env = env
        self.executor = executor
        self.lssd_encoder = lssd_encoder
        self.opponent_factory = opponent_factory   # opponent_type_str -> opponent_policy callable
        self.episodes = episodes
        self.decision_cycles = decision_cycles
        self.max_decisions_per_match = max_decisions_per_match
        self._rng = random.Random(seed)
        self.training_pairs: list = []
        self.prompt_program = "SA_BASE_PROMPT"
        self._win_history: deque[float] = deque(maxlen=scheduler.window_w)
        self.timing = {"episode_seconds": [], "decision_seconds": []}

    def _lssd_from_env(self) -> str:
        obs = self.env.agent_sensors.read()
        ego = {"fwd": 0.0, "turn": 0.0}   # neutral proxy; full PA ego-motion not needed for MCTS state
        enc = self.lssd_encoder.encode(obs["tof"], approach_rate=0.0, ego=ego)
        return enc["lssd_text"]

    def _execute_strategy_live(self, strategy) -> Optional[str]:
        """Actually drive the live (non-branched) match forward for real, using the
        scripted macro-executor - same proxy MCTS used internally, now applied to the
        real match continuation. Returns the match outcome if it terminated, else None."""
        import numpy as np

        for _ in range(self.decision_cycles):
            obs = self.env.agent_sensors.read()
            left, right = self.executor.to_pwm(strategy, obs["tof"], obs["ir"])
            _, _, terminated, truncated, info = self.env.step(np.array([left, right], dtype=np.float32))
            if terminated or truncated:
                return info.get("outcome", "draw")
        return None

    def _rolling_winrate(self):
        if not self._win_history:
            return None
        return sum(self._win_history) / len(self._win_history)

    def run(self) -> dict:
        run_t0 = time.perf_counter()
        for ep in range(self.episodes):
            ep_t0 = time.perf_counter()
            opp_type = self.opponent_pool.sample(ep, self.checkpoint_mgr.has_checkpoint())
            self.env.opponent_policy = self.opponent_factory(opp_type)
            self.env.reset()

            outcome = "draw"
            decisions = 0
            while decisions < self.max_decisions_per_match:
                dec_t0 = time.perf_counter()
                lssd_text = self._lssd_from_env()

                if self.ablation.mcts_enabled:
                    root_state = self.mcts.backend.root_state(lssd_text)
                    result = self.mcts.search(root_state)
                    best = result.best_strategy
                    restore = getattr(self.mcts.backend, "_restore", None)
                    if restore is not None:
                        restore(root_state.pybullet_state_id)   # undo search branching, keep live match intact
                    self.training_pairs.append(result.training_pair)
                else:
                    # No-MCTS ablation: single-pass direct sampling.
                    best = self._rng.choice(self.strategies)
                    self.training_pairs.append(({"lssd": lssd_text}, best))

                match_outcome = self._execute_strategy_live(best)
                decisions += 1
                self.timing["decision_seconds"].append(time.perf_counter() - dec_t0)
                if match_outcome is not None:
                    outcome = match_outcome
                    break

            self._win_history.append(1.0 if outcome == "win" else 0.0)
            self.timing["episode_seconds"].append(time.perf_counter() - ep_t0)

            if self.ablation.dspy_enabled:
                do, _why = self.scheduler.should_recompile(ep, self._rolling_winrate())
                if do:
                    self.prompt_program = self.dspy_compiler.compile(self.training_pairs, self.prompt_program)

            self.checkpoint_mgr.maybe_snapshot(ep, self.prompt_program)

        total_wall_s = time.perf_counter() - run_t0
        n_ep = max(1, len(self.timing["episode_seconds"]))
        n_dec = max(1, len(self.timing["decision_seconds"]))
        return {
            "episodes": self.episodes,
            "training_pairs": len(self.training_pairs),
            "dspy_recompiles": getattr(self.dspy_compiler, "compile_count", 0),
            "checkpoints": len(self.checkpoint_mgr._checkpoints),
            "total_wall_seconds": round(total_wall_s, 2),
            "avg_episode_seconds": round(sum(self.timing["episode_seconds"]) / n_ep, 3),
            "avg_decision_seconds": round(sum(self.timing["decision_seconds"]) / n_dec, 3),
        }