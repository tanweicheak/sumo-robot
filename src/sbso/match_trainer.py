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

    training_pairs are (episode, state, strategy) triples, not bare (state, strategy)
    pairs - RealDSPyCompiler._select_examples() needs the episode tag to spread its
    few-shot selection across multiple matches instead of silently collapsing
    everything into one pseudo-episode.

    dominant_strategy is computed by RealDSPyCompiler at compile time (from structured
    demo objects, not text-parsing) and threaded through to checkpoint_mgr so a sampled
    self-checkpoint opponent's rollout proxy reflects what that checkpoint actually
    favored, instead of always defaulting to CHARGE.

    on_episode_end, if given, fires after every episode with (episode, self) - lets a
    caller (e.g. run_phase4_pilot.py) persist progress incrementally without this class
    needing to know anything about files/paths itself.
"""

from __future__ import annotations

import random
import time
from collections import deque
from typing import Any, Callable, Optional

from src.sbso.ablation_strategies import AblationConfig
from src.preprocessing.ir_gradient import IRGradientFilter


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
        on_episode_end: Optional[Callable[[int, "MatchLevelSBSOTrainer"], None]] = None,
        slm_client: Any = None,
        validate_recompiles: bool = False,
        validation_n_samples: int = 30,
        validation_accept_margin: float = 0.0,
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
        self.prompt_version = 0
        self.recompile_history: list[dict] = []
        self.mcts_calibration_log: list[dict] = []
        self.dominant_strategy: Optional[str] = None
        self.on_episode_end = on_episode_end
        self.slm_client = slm_client
        self.validate_recompiles = validate_recompiles
        self.validation_n_samples = validation_n_samples
        self.validation_accept_margin = validation_accept_margin
        self._win_history: deque[float] = deque(maxlen=scheduler.window_w)
        self.timing = {"episode_seconds": [], "decision_seconds": []}
        # Real edge-approach-rate tracking, matching PerceptionAgent's own filter -
        # previously this was hardcoded to 0.0 (always "edge=safe" regardless of actual
        # proximity to the ring boundary), which meant training data never showed an
        # edge=critical example for EVADE_EDGE to learn from at all.
        self._ir_filter = IRGradientFilter(window_length=5, n_probes=2, dt_s=self.env.cfg.control_dt_s)

    def _lssd_from_env(self) -> str:
        obs = self.env.agent_sensors.read()
        ego = {"fwd": 0.0, "turn": 0.0}   # neutral proxy; full PA ego-motion not needed for MCTS state
        approach_rate = self._ir_filter.update(obs["ir"])
        enc = self.lssd_encoder.encode(obs["tof"], approach_rate=approach_rate, ego=ego)
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
            self.last_opponent_type = opp_type   # exposed for _on_episode_end callbacks -
                                                   # previously local-only, no way to log
                                                   # which opponent a given episode played
            self.env.opponent_policy = self.opponent_factory(opp_type)
            self.env.reset()
            self._ir_filter.reset()

            outcome = "draw"
            decisions = 0
            episode_calibration_buffer: list[dict] = []
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
                    # Episode-tagged triple - RealDSPyCompiler._select_examples() needs
                    # this to spread few-shot selection across matches instead of
                    # collapsing everything into one pseudo-episode.
                    self.training_pairs.append((ep, *result.training_pair))

                    # Shared MCTS-proxy-drift / Judge-calibration diagnostic: root_stats'
                    # mean_value for the CHOSEN strategy is exactly the value the search
                    # computed against the scripted proxy opponent (and, when a rollout
                    # doesn't terminate, is largely the Judge's own score_position output
                    # backpropagated up) - i.e. "how good MCTS predicted this choice would
                    # be." Logged here, matched against the real eventual match outcome
                    # once it's known (below), with NO extra simulation or judge calls -
                    # this is free instrumentation on data already computed.
                    strat_key = best.value if hasattr(best, "value") else str(best)
                    episode_calibration_buffer.append({
                        "decision_index": decisions,
                        "chosen_strategy": strat_key,
                        "proxy_predicted_value": result.root_stats.get(strat_key),
                    })
                else:
                    # No-MCTS ablation: single-pass direct sampling.
                    best = self._rng.choice(self.strategies)
                    self.training_pairs.append((ep, {"lssd": lssd_text}, best))

                match_outcome = self._execute_strategy_live(best)
                decisions += 1
                self.timing["decision_seconds"].append(time.perf_counter() - dec_t0)
                if match_outcome is not None:
                    outcome = match_outcome
                    break

            outcome_value = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(outcome, 0.5)
            for entry in episode_calibration_buffer:
                entry["episode"] = ep
                entry["match_outcome"] = outcome
                entry["match_outcome_value"] = outcome_value
            self.mcts_calibration_log.extend(episode_calibration_buffer)

            self._win_history.append(1.0 if outcome == "win" else 0.0)
            self.timing["episode_seconds"].append(time.perf_counter() - ep_t0)

            if self.ablation.dspy_enabled:
                do, why = self.scheduler.should_recompile(ep, self._rolling_winrate())
                if do:
                    candidate_prompt = self.dspy_compiler.compile(self.training_pairs, self.prompt_program)
                    candidate_dominant_strategy = getattr(self.dspy_compiler, "last_dominant_strategy", None)

                    validation = None
                    accepted = True
                    if self.validate_recompiles and self.slm_client is not None:
                        from src.sbso.dspy_compiler import validate_prompt_candidate
                        validation = validate_prompt_candidate(
                            candidate_prompt=candidate_prompt, incumbent_prompt=self.prompt_program,
                            training_pairs=self.training_pairs, slm_client=self.slm_client,
                            judge=self.mcts.backend.judge, n_samples=self.validation_n_samples,
                            accept_margin=self.validation_accept_margin,
                        )
                        accepted = validation["accept"]

                    if accepted:
                        self.prompt_program = candidate_prompt
                        self.dominant_strategy = candidate_dominant_strategy
                        self.prompt_version += 1
                    # Was previously discarded (`_why`) - no record anywhere of which prompt
                    # version was active over which episode range, or why a given recompile
                    # fired. Joins with training_pairs.jsonl by episode number post-hoc
                    # (see scripts/report_strategy_distribution.py --prompt-history) rather
                    # than changing training_pairs' tuple shape, which multiple consumers
                    # (build_sft_dataset, RealDSPyCompiler._select_examples) already depend
                    # on being exactly (episode, state, strategy).
                    self.recompile_history.append({
                        "episode": ep, "trigger_reason": why,
                        "prompt_version": self.prompt_version if accepted else self.prompt_version + 1,
                        "rolling_winrate": self._rolling_winrate(),
                        "accepted": accepted, "validation": validation,
                    })

            self.checkpoint_mgr.maybe_snapshot(ep, self.prompt_program, self.dominant_strategy)

            if self.on_episode_end is not None:
                self.on_episode_end(ep, self)

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