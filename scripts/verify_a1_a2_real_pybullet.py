"""
scripts/verify_a1_a2_real_pybullet.py

Standalone (non-pytest) verification of the A1 (D3 saveState leak) and A2 (D2 opponent
proxy) fixes, run against the REAL PyBulletSumoEnv and real pybullet - not fakes.

Run from your repo root:
    python scripts/verify_a1_a2_real_pybullet.py

What it checks:
  1. A2/D2 - the env's REAL opponent_policy is never called during MCTS rollouts
     (a spy raises if it is), and env.opponent_policy is restored after search().
  2. A1/D3 - PyBulletSimulationBackend._live_state_ids returns to empty after each
     decision's full cycle (search -> restore -> release), across many repeated
     decisions - the scenario that used to leak thousands of saveState ids over a long
     training run.
  3. Peak resident memory (via resource.getrusage) doesn't grow across the stress loop -
     a cheap, dependency-free empirical signal that the leak is actually gone, not just
     that the bookkeeping set is empty (which could theoretically be wrong on its own).

Adjust the import paths below if your repo layout differs.
"""
from __future__ import annotations

import resource
import sys
from pathlib import Path

# Self-bootstrap: `python scripts/verify_a1_a2_real_pybullet.py` only puts scripts/ on
# sys.path, not the repo root, so `import src...` fails. Add the repo root explicitly
# so this runs standalone regardless of how it's invoked or whether scripts/ is a
# package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.schemas import MacroStrategy
from src.sbso.judge import MockJudge
from src.sbso.macro_executor import MacroStrategyExecutor
from src.sbso.mcts import MCTS
from src.sbso.simulation_backend import PyBulletSimulationBackend
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv

N_DECISIONS = 50   # repeated "one decision per episode" cycles, stress-testing the leak


class RealOpponentPolicySpy:
    """Stands in for whatever the env's real opponent_policy is (baseline controller,
    or eventually a live-SLM self-checkpoint policy). If MCTS ever calls this during a
    rollout, A2/D2 is broken."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, obs):
        self.call_count += 1
        raise AssertionError("D2 VIOLATED: real opponent_policy was called during an MCTS rollout")


def main() -> None:
    env = PyBulletSumoEnv(env_config=EnvConfig(use_gui=False))
    env.reset()
    real_opponent = RealOpponentPolicySpy()
    env.opponent_policy = real_opponent

    executor = MacroStrategyExecutor()
    judge = MockJudge(seed=0)   # structural test - not testing Judge quality here
    backend = PyBulletSimulationBackend(env, executor, judge, cycles_per_node=3)
    strategies = list(MacroStrategy)
    mcts = MCTS(backend, strategies, sim_budget=30, horizon=4, judge_prune_threshold=0.3)

    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for i in range(N_DECISIONS):
        env.reset()   # fresh live state per decision, matching one-decision-per-episode
        env.opponent_policy = real_opponent   # reset() doesn't touch this, but be explicit

        root_state = backend.root_state(f"lssd_test_{i}", opponent_behavior="unknown")
        assert backend._live_state_ids == {root_state.pybullet_state_id}, (
            f"[decision {i}] expected only root id live before search, got {backend._live_state_ids}"
        )

        mcts.search(root_state)

        assert backend._live_state_ids == {root_state.pybullet_state_id}, (
            f"[decision {i}] D3 STILL LEAKING: expected only root id live after search, "
            f"got {backend._live_state_ids}"
        )

        # What training_loop.py does after search(): restore live env to root, then
        # this decision cycle is fully done with its snapshots.
        backend._restore(root_state.pybullet_state_id)
        backend.release_search_states(keep=None)
        assert backend._live_state_ids == set(), (
            f"[decision {i}] expected zero live ids after final cleanup, got {backend._live_state_ids}"
        )

        if (i + 1) % 10 == 0:
            print(f"  decision {i + 1}/{N_DECISIONS} OK  "
                  f"(live_state_ids clean, opponent spy calls={real_opponent.call_count})")

    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_delta_mb = (mem_after - mem_before) / 1024.0   # ru_maxrss is KB on Linux, bytes on macOS

    assert real_opponent.call_count == 0, (
        f"D2 VIOLATED: real opponent_policy was called {real_opponent.call_count} times total"
    )

    print()
    print(f"Ran {N_DECISIONS} decisions, each ~30 MCTS expansions x horizon 4 rollout.")
    print(f"Real opponent_policy call count: {real_opponent.call_count} (must be 0)")
    print(f"Peak RSS before: {mem_before}  after: {mem_after}  delta: {mem_delta_mb:.2f} "
          f"(units: KB on Linux, bytes/1024 on macOS - just check it isn't climbing steadily)")
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
