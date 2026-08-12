"""Unit tests for the Stage 1 mock SBSO loop (Phase 4)."""

from __future__ import annotations

import unittest

from src.agents.schemas import MacroStrategy
from src.sbso.ablation_strategies import AblationConfig
from src.sbso.dspy_compiler import MockDSPyCompiler
from src.sbso.judge import MockJudge
from src.sbso.mcts import MCTS
from src.sbso.opponent_pool import OpponentPool
from src.sbso.recompilation_scheduler import RecompilationScheduler
from src.sbso.self_checkpoint_manager import SelfCheckpointManager
from src.sbso.simulation_backend import MockSimulationBackend
from src.sbso.training_loop import SBSOTrainer

STRATS = list(MacroStrategy)


def _mcts(prune=0.3):
    judge = MockJudge(seed=1)
    backend = MockSimulationBackend(judge, seed=1)
    return MCTS(backend, STRATS, sim_budget=30, horizon=4, judge_prune_threshold=prune)


class TestMCTS(unittest.TestCase):
    def test_search_returns_a_strategy(self):
        r = _mcts().search({"depth": 0, "progress": 0.0, "lssd": "x"})
        self.assertIn(r.best_strategy, STRATS)

    def test_all_strategies_evaluated(self):
        r = _mcts().search({"depth": 0, "progress": 0.0, "lssd": "x"})
        # With enough budget, every root strategy gets at least one estimate.
        evaluated = [k for k, v in r.root_stats.items() if v is not None]
        self.assertEqual(len(evaluated), len(STRATS))

    def test_training_pair_shape(self):
        r = _mcts().search({"depth": 0, "progress": 0.0, "lssd": "x"})
        state, strat = r.training_pair
        self.assertIn(strat, STRATS)


class TestScheduler(unittest.TestCase):
    def test_k_episode_trigger(self):
        s = RecompilationScheduler(k_episodes=5, window_w=10, delta=0.05)
        fired = [s.should_recompile(ep, 0.5)[0] for ep in range(11)]
        self.assertTrue(fired[5])   # fires at K
        self.assertTrue(fired[10])

    def test_reward_drop_trigger(self):
        s = RecompilationScheduler(k_episodes=1000, window_w=10, delta=0.05)
        s.should_recompile(0, 0.80)        # seed prev window
        did, why = s.should_recompile(1, 0.70)   # dropped 10pp > delta
        self.assertTrue(did)
        self.assertEqual(why, "reward_drop")


class TestOpponentPool(unittest.TestCase):
    def test_warmup_excludes_self_checkpoint(self):
        pool = OpponentPool(warmup_episodes=5, total_episodes=20)
        for ep in range(5):
            self.assertIn(pool.sample(ep, has_self_checkpoint=True), ("baseline1", "baseline2"))

    def test_post_warmup_can_sample_self(self):
        mgr = SelfCheckpointManager(interval=1)
        mgr.maybe_snapshot(1, "SA_BASE_PROMPT")   # episode > 0 required - seeds one real checkpoint
        pool = OpponentPool(warmup_episodes=0, total_episodes=20, seed=3, self_checkpoint_manager=mgr)
        seen = {pool.sample(ep, has_self_checkpoint=True) for ep in range(50)}
        self.assertIn("self_checkpoint", seen)


class TestFullLoop(unittest.TestCase):
    def _trainer(self, ablation):
        return SBSOTrainer(
            ablation=ablation,
            mcts=_mcts(prune=0.3 if ablation.judge_enabled else 0.0),
            opponent_pool=OpponentPool(warmup_episodes=5, total_episodes=20),
            scheduler=RecompilationScheduler(k_episodes=5, window_w=10, delta=0.05),
            checkpoint_mgr=SelfCheckpointManager(interval=5),
            dspy_compiler=MockDSPyCompiler(),
            strategies=STRATS,
            episodes=20,
        )

    def test_full_sbso_runs(self):
        summary = self._trainer(AblationConfig()).run()
        self.assertEqual(summary["episodes"], 20)
        self.assertEqual(summary["training_pairs"], 20)
        self.assertGreater(summary["dspy_recompiles"], 0)
        self.assertGreater(summary["checkpoints"], 0)

    def test_no_mcts_ablation_still_collects_pairs(self):
        summary = self._trainer(AblationConfig(mcts_enabled=False)).run()
        self.assertEqual(summary["training_pairs"], 20)

    def test_no_dspy_ablation_never_recompiles(self):
        summary = self._trainer(AblationConfig(dspy_enabled=False)).run()
        self.assertEqual(summary["dspy_recompiles"], 0)
        self.assertEqual(summary["final_prompt"], "SA_BASE_PROMPT")


if __name__ == "__main__":
    unittest.main()