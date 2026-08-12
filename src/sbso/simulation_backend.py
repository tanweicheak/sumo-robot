"""
src.sbso.simulation_backend

Phase: Phase 4
Purpose: The interface MCTS uses to evaluate strategies - "given a state and a strategy,
    step forward" and "from a state, roll out to a Judge-scored value". This is the seam
    between the MCTS algorithm (pure) and how simulation actually happens. Stage 1 uses
    MockSimulationBackend (fast fake dynamics) to validate the search logic; Stage 3 adds
    a PyBulletSimulationBackend that clones real physics state (saveState/restoreState),
    runs the agent+opponent for the horizon, and Judge-scores the leaf.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from src.sbso.judge import Judge
from src.sbso.macro_executor import MacroStrategyExecutorOpponent
from src.agents.schemas import MacroStrategy

class SimulationBackend(ABC):
    @abstractmethod
    def step(self, state: Any, strategy) -> Any:
        """Apply one strategy from `state`; return the resulting state."""

    @abstractmethod
    def rollout(self, state: Any, horizon: int) -> float:
        """From `state`, play forward up to `horizon` cycles, return a scalar value
        (Judge-scored leaf, or terminal win/loss). This is MCTS's Simulation step."""

    @abstractmethod
    def judge_branch(self, state: Any, strategy) -> float:
        """Judge's pre-rollout coherence score for taking `strategy` from `state`."""

    @abstractmethod
    def is_terminal(self, state: Any) -> bool:
        ...


class MockSimulationBackend(SimulationBackend):
    """Fake, fast dynamics for validating MCTS + the SBSO loop locally. State is a small
    dict; progress drifts per step; terminal at a fixed depth. Real physics lands in
    PyBulletSimulationBackend (Stage 3)."""

    def __init__(self, judge: Judge, seed: int = 0, terminal_depth: int = 6) -> None:
        self.judge = judge
        self._rng = random.Random(seed)
        self.terminal_depth = terminal_depth

    def step(self, state, strategy):
        depth = state.get("depth", 0) + 1
        progress = state.get("progress", 0.0) + self._rng.uniform(-0.1, 0.2)
        return {"depth": depth, "progress": progress, "lssd": state.get("lssd", "")}

    def rollout(self, state, horizon):
        s = dict(state)
        for _ in range(horizon):
            if self.is_terminal(s):
                break
            s = self.step(s, None)
        return self.judge.score_position(s.get("lssd", ""))

    def judge_branch(self, state, strategy):
        return self.judge.score_branch(state.get("lssd", ""), strategy)

    def is_terminal(self, state):
        return state.get("depth", 0) >= self.terminal_depth

"""
(append to src/sbso/simulation_backend.py)

PyBulletSimulationBackend - Stage 2 real backend. Clones live physics state via
p.saveState()/p.restoreState() so MCTS can branch into many hypothetical futures from
the same root without corrupting the live match. Each branch is scored either by a
genuine terminal outcome (win/loss reached within the rollout) or by the Judge at the
horizon (Branch 3: truncated rollout + Judge-value).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PyBulletMCTSState:
    pybullet_state_id: int
    lssd_text: str
    depth: int = 0
    opponent_behavior: str = "unknown"
    terminated: bool = False
    outcome: Optional[str] = None


class PyBulletSimulationBackend(SimulationBackend):
    def __init__(self, env, executor, judge: Judge, cycles_per_node: int = 3, opponent_policy=None) -> None:
        self.env = env                      # a live, already-reset PyBulletSumoEnv
        self.executor = executor            # MacroStrategyExecutor
        self.judge = judge
        self.cycles_per_node = cycles_per_node
        # D2 fix: default to a fast deterministic proxy, not None. Leaving this None used
        # to mean "env's own opponent_policy runs during rollouts too" - for a self-checkpoint
        # opponent, that opponent_policy may itself be a live-SLM call, which is fatal
        # inside MCTS (dozens of rollouts x horizon steps x real inference latency). The
        # opponent, regardless of its real type, always acts through this proxy during
        # tree search; only the live, committed match continuation (outside MCTS) uses
        # the real opponent_policy attached to the env.
        self.opponent_policy = opponent_policy or MacroStrategyExecutorOpponent()
        # D3 fix: p.saveState() ids are NOT garbage-collected by PyBullet - every id this
        # backend creates is tracked here so it can be explicitly freed via
        # release_search_states() (called once per MCTS.search()) and rollout()'s own
        # per-call cleanup. Without this, a long training run leaks thousands of states.
        self._live_state_ids: set[int] = set()
        # PyBullet's saveState/restoreState only capture physics (rigid body) state, NOT
        # arbitrary Python attributes like env._step_count - PyBulletSumoEnv uses
        # _step_count for ITS OWN truncation check (_step_count >= max_steps), and it is
        # a plain incrementing int, never coupled to the physics engine. Without tracking
        # it here too, every MCTS rollout/expansion step() call inflates _step_count
        # (since backend.step() calls self.env.step() for real, even hypothetically)
        # without restoreState() ever bringing it back down - a single decision's search
        # (sim_budget x horizon step() calls) can burn through most of max_steps before
        # a single REAL decision has been committed, truncating matches far too early.
        self._step_count_by_state_id: dict[int, int] = {}

    def _restore(self, state_id: int) -> None:
        import pybullet as p
        p.restoreState(stateId=state_id, physicsClientId=self.env.client_id)
        if state_id in self._step_count_by_state_id:
            self.env._step_count = self._step_count_by_state_id[state_id]

    def _snapshot(self) -> int:
        import pybullet as p
        state_id = p.saveState(physicsClientId=self.env.client_id)
        self._live_state_ids.add(state_id)
        self._step_count_by_state_id[state_id] = self.env._step_count
        return state_id

    def _free_ids(self, ids) -> None:
        ids = set(ids) & self._live_state_ids
        if not ids:
            return
        import pybullet as p
        for state_id in ids:
            # PyBullet's removeState() keyword arg is `stateUniqueId`, NOT `stateId` -
            # inconsistent with restoreState()'s `stateId`, confirmed at runtime. Pass
            # positionally so it works regardless of which name a given PyBullet build uses.
            p.removeState(state_id, physicsClientId=self.env.client_id)
            self._step_count_by_state_id.pop(state_id, None)
        self._live_state_ids -= ids

    def release_search_states(self, keep: Optional[set[int]] = None) -> None:
        """Free every saveState snapshot this backend has created so far, except ids in
        `keep`. MCTS.search() calls this automatically (via duck-typing) once per search,
        keeping only the caller's original root_state id - the tree itself (and every
        snapshot it created) is discarded once the best strategy has been chosen."""
        self._free_ids(self._live_state_ids - (keep or set()))

    def root_state(self, lssd_text: str, opponent_behavior: str = "unknown") -> PyBulletMCTSState:
        """Call once per real decision, at the CURRENT live state, before MCTS.search()."""
        return PyBulletMCTSState(
            pybullet_state_id=self._snapshot(), lssd_text=lssd_text, depth=0,
            opponent_behavior=opponent_behavior,
        )

    def step(self, state: PyBulletMCTSState, strategy) -> PyBulletMCTSState:
        import numpy as np

        self._restore(state.pybullet_state_id)

        # D2 fix: during MCTS tree search (this method), the opponent - regardless of its
        # real type (baseline / self-checkpoint) - must act through the same fast
        # deterministic proxy as the agent, never live SLM inference. self.opponent_policy
        # is swapped onto the env only for the duration of this call and restored after,
        # so the live match's real opponent_policy is untouched once search() returns.
        original_opponent_policy = self.env.opponent_policy
        if self.opponent_policy is not None:
            self.env.opponent_policy = self.opponent_policy
        try:
            outcome = None
            for _ in range(self.cycles_per_node):
                obs = self.env.agent_sensors.read()
                left, right = self.executor.to_pwm(strategy, obs["tof"], obs["ir"])
                _, _, terminated, truncated, info = self.env.step(np.array([left, right], dtype=np.float32))
                if terminated or truncated:
                    outcome = info.get("outcome")
                    break
        finally:
            self.env.opponent_policy = original_opponent_policy

        new_id = self._snapshot()
        return PyBulletMCTSState(
            pybullet_state_id=new_id, lssd_text=state.lssd_text, depth=state.depth + 1,
            opponent_behavior=state.opponent_behavior,
            terminated=outcome is not None, outcome=outcome,
        )

    def rollout(self, state: PyBulletMCTSState, horizon: int) -> float:
        if state.terminated:
            return self._outcome_value(state.outcome)
        # D3 fix: track ids created during THIS rollout so they can be freed immediately
        # below - none of them are stored in the tree, so none need to survive past
        # this call (unlike the id created by the node's own expand-time step(), which
        # is not touched here).
        ids_before = set(self._live_state_ids)
        s = state
        for _ in range(horizon):
            if s.terminated:
                break
            s = self.step(s, MacroStrategy.HOLD)   # neutral proxy during truncated rollout
        if s.terminated:
            value = self._outcome_value(s.outcome)
        else:
            value = self.judge.score_position(s.lssd_text)
        self._restore(state.pybullet_state_id)   # leave the live sim state untouched
        self._free_ids(self._live_state_ids - ids_before)
        return value

    def judge_branch(self, state: PyBulletMCTSState, strategy) -> float:
        return self.judge.score_branch(state.lssd_text, strategy)

    def is_terminal(self, state: PyBulletMCTSState) -> bool:
        return state.terminated

    @staticmethod
    def _outcome_value(outcome: Optional[str]) -> float:
        if outcome == "win":
            return 1.0
        if outcome == "loss":
            return 0.0
        return 0.5   # draw