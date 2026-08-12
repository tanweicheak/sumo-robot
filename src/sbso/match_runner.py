"""
src.sbso.match_runner

Phase: Phase 4 (Stage 3)
Purpose: Plays one full match by repeatedly running MCTS for one decision at a time and
    committing each decision for real, continuing until the match ends (win/loss/draw)
    or a decision cap is hit. This is what makes "episode" mean what it's actually
    supposed to mean here: a full match with many decisions, not a single decision -
    see the scope correction on this ("episodes means full matches"). The prior
    training loop's one-decision-per-"episode" behavior undercounts a real match's
    training signal by roughly however many decisions actually fit in ~15s of match
    time at cycles_per_node granularity - could be dozens.

    Reuses SBSOTrainer._mcts_decision() (a helper that already existed, unused, from an
    earlier session) as the per-decision primitive, and stage2_wiring.commit_strategy_for_real
    for committing each decision - no new "how does one decision work" logic here, only
    "how do decisions chain into a match".

DESIGN DECISION (confirmed default - see D2/match-runner discussion): DSPy
recompile-eligibility and self-checkpoint snapshotting are checked ONCE PER MATCH, not
once per decision. This lives in SBSOTrainer.run_stage3() (training_loop.py), not here -
MatchRunner only plays matches and returns (won, training_pairs); it has no opinion on
recompile/checkpoint cadence. If mid-match recompiles are wanted later, run_stage3() is
the only place that needs to change - play_match()'s per-decision loop already returns
control to the caller after every single decision internally via trainer._mcts_decision(),
so the seam is already there if needed.
"""
from __future__ import annotations

from typing import Any

from src.agents.perception_agent import PerceptionAgent
from src.sbso.simulation_backend import PyBulletMCTSState, PyBulletSimulationBackend
from src.sbso.stage2_wiring import commit_strategy_for_real


class MatchRunner:
    def __init__(
        self,
        backend: PyBulletSimulationBackend,
        perception_agent: PerceptionAgent,
        max_decisions_per_match: int = 200,
    ) -> None:
        self.backend = backend
        self.perception_agent = perception_agent
        # Safety cap in case a match somehow never terminates/truncates at the env level
        # (shouldn't happen - PyBulletSumoEnv.max_steps enforces this - but a runaway
        # match here would otherwise loop forever accumulating training pairs).
        self.max_decisions_per_match = max_decisions_per_match

    def _build_root_state(self, opponent: Any) -> PyBulletMCTSState:
        obs = self.backend.env.agent_sensors.read()
        perception_state = self.perception_agent.perceive(obs["tof"], obs["ir"], obs["encoder"])
        opponent_kind = getattr(opponent, "kind", str(opponent))
        return self.backend.root_state(perception_state.lssd_text, opponent_behavior=opponent_kind)

    def play_match(self, trainer, episode: int, opponent: Any) -> tuple[float, list]:
        """Plays one full match. Returns (won, training_pairs) - won is 1.0/0.5/0.0
        (win/draw-or-truncated/loss), training_pairs is every (state, best_strategy)
        pair collected across every decision in the match. Called once per episode from
        SBSOTrainer.run_stage3()."""
        self.backend.env.reset()
        self.perception_agent.reset()
        pairs: list = []

        for _decision_idx in range(self.max_decisions_per_match):
            root_state = self._build_root_state(opponent)
            result = trainer._mcts_decision(lambda: root_state)
            # Episode-tagged triple (item 1 fix) - same shape training_loop.py's run()
            # produces, so RealDSPyCompiler's match-aware example selection works
            # identically regardless of which path (Stage 1/2's run() or Stage 3's
            # run_stage3()) produced the pairs.
            pairs.append((episode, *result.training_pair))

            committed = commit_strategy_for_real(self.backend, root_state, result.best_strategy)

            # mcts.search() (inside _mcts_decision) already freed everything except
            # root_state's own id (A1/D3 fix). Now that we've committed past root_state,
            # free ITS id too, keeping only committed's - the live position going
            # forward. Without this, a long match leaks one saveState id per decision.
            self.backend.release_search_states(keep={committed.pybullet_state_id})

            if committed.terminated:
                won = self.backend._outcome_value(committed.outcome)
                self.backend.release_search_states(keep=None)   # nothing needs to survive past the match
                return won, pairs

        # Hit max_decisions_per_match without a natural conclusion - treat as a draw.
        self.backend.release_search_states(keep=None)
        return 0.5, pairs
