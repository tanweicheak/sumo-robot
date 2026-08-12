"""
scripts.run_phase4_training

Phase: Phase 4
Purpose: Run the SBSO training loop for a variant config. Stage 1 wires the mock
    backend/judge/DSPy so the full loop runs locally in seconds to validate plumbing;
    Stage 3 swaps in real PyBullet rollouts + Llama-3.1-8B + real DSPy on cloud GPU.
"""

from __future__ import annotations

from scripts._script_common import build_run
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


def main() -> None:
    config, ctx = build_run(phase="phase4", description="Phase 4 SBSO training (Stage 1: mock loop).")

    ablation = AblationConfig.for_variant(config.get("ablation", {}).get("strategy", "none"))
    strategies = list(MacroStrategy)

    judge = MockJudge(seed=0)
    backend = MockSimulationBackend(judge, seed=0)
    mcts = MCTS(
        backend, strategies,
        sim_budget=30, horizon=4,
        judge_prune_threshold=0.3 if ablation.judge_enabled else 0.0,
    )

    # Small local episode count for Stage 1; the full 5000 runs on cloud GPU (Stage 3).
    episodes = int(config.get("training", {}).get("local_stage1_episodes", 20))
    # Stage 1 local run: scale checkpoint interval down so it actually fires in a short run.
    # (Full run on cloud uses self_checkpoint_interval_episodes: 500 with 5000 episodes.)
    interval = int(config.get("training", {}).get("local_stage1_checkpoint_interval", 5))

    # Shared instance: OpponentPool needs the SAME manager the trainer snapshots into,
    # so what it offers as "self_checkpoint" (Stage 3a) matches what actually got saved.
    checkpoint_mgr = SelfCheckpointManager(interval=interval)

    trainer = SBSOTrainer(
        ablation=ablation,
        mcts=mcts,
        opponent_pool=OpponentPool(
            warmup_episodes=min(5, episodes), total_episodes=episodes,
            self_checkpoint_manager=checkpoint_mgr,
        ),
        scheduler=RecompilationScheduler(k_episodes=5, window_w=10, delta=0.05),
        checkpoint_mgr=checkpoint_mgr,
        dspy_compiler=MockDSPyCompiler(),
        strategies=strategies,
        episodes=episodes,
    )
    summary = trainer.run()
    print(f"[phase4] variant={ablation}  summary={summary}")


if __name__ == "__main__":
    main()