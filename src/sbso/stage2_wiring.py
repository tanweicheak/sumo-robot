"""
src.sbso.stage2_wiring

Phase: Phase 4 (Stage 2)
Purpose: The two callables SBSOTrainer needs to run against REAL physics + REAL
    perception instead of Stage 1's mock dict/coin-flip placeholders - without changing
    training_loop.py itself (its docstring already documents this seam).

    KNOWN SIMPLIFICATION (unchanged from training_loop.py's own docstring): each
    "episode" here commits exactly ONE MCTS-informed decision for real, then reads
    whatever outcome resulted after `cycles_per_node` real physics steps (win / loss /
    draw / still-ongoing, mapped to 1.0 / 0.0 / 0.5 / 0.5). It does NOT play a full
    ~300-cycle match to a natural conclusion - that loop is the Stage 3 match-runner's
    job, reusing these same two functions per decision within a match.
"""
from __future__ import annotations

from typing import Any

from src.agents.perception_agent import PerceptionAgent
from src.sbso.simulation_backend import PyBulletMCTSState, PyBulletSimulationBackend


def build_real_root_state_builder(backend: PyBulletSimulationBackend, perception_agent: PerceptionAgent):
    """Returns root_state_builder(episode, opponent) -> PyBulletMCTSState, reading the
    CURRENT live env state through the real Perception Agent (Savitzky-Golay + IR
    gradient + LSSD encoding) instead of Stage 1's fake f"ep{episode}" string."""

    def _builder(episode: int, opponent: Any) -> PyBulletMCTSState:
        obs = backend.env.agent_sensors.read()
        perception_state = perception_agent.perceive(obs["tof"], obs["ir"], obs["encoder"])
        opponent_kind = getattr(opponent, "kind", str(opponent))
        return backend.root_state(perception_state.lssd_text, opponent_behavior=opponent_kind)

    return _builder


def commit_strategy_for_real(
    backend: PyBulletSimulationBackend, root_state: PyBulletMCTSState, strategy,
) -> PyBulletMCTSState:
    """Executes `strategy` for real against the REAL opponent_policy already attached
    to backend.env (not the fast rollout proxy MCTS uses internally, per D2) for
    backend.cycles_per_node physics steps. Returns the resulting state - backend.env
    has already been mutated by the real env.step() calls inside backend.step(), so
    this IS the live position going forward. Shared by Stage 2's single-decision
    outcome_extractor and Stage 3's MatchRunner (src.sbso.match_runner)."""
    backend._restore(root_state.pybullet_state_id)
    rollout_proxy = backend.opponent_policy
    backend.opponent_policy = None
    try:
        return backend.step(root_state, strategy)
    finally:
        backend.opponent_policy = rollout_proxy


def build_real_outcome_extractor(backend: PyBulletSimulationBackend):
    """Returns outcome_extractor(mcts_result, root_state) -> float in {0.0, 0.5, 1.0}.
    Commits the decision for real and reads whatever terminal outcome resulted.
    """

    def _extractor(mcts_result, root_state: PyBulletMCTSState) -> float:
        committed_state = commit_strategy_for_real(backend, root_state, mcts_result.best_strategy)
        if committed_state.terminated:
            return backend._outcome_value(committed_state.outcome)
        return 0.5   # still ongoing after this one committed decision - neutral signal

    return _extractor
