"""
scripts.run_mcts_sim_budget_sweep

Phase: pilot-calibration
Purpose: sim_budget=20 (phase4_pilot.yaml) / c_uct=1.41 (mcts.py default) were never
    calibrated - just hardcoded, not even flagged as placeholders the way
    arena_config.yaml's values are. This runs the SAME pilot config at a small set of
    sim_budget values and compares the resulting strategy distribution and win rate,
    so the choice is evidence-based rather than an unexamined default.

    Needs a REAL judge (LlamaCppJudge or SGLangJudge) to be meaningful - MockJudge
    returns uniform random noise, which would make every sim_budget value look
    identically uninformative regardless of whether more search actually helps.
    Uses MockDSPyCompiler by default (--real-dspy to override) so recompilation
    timing doesn't confound the sim_budget comparison - you're isolating one
    variable (search budget), not testing DSPy in the same run.

Usage (local, LlamaCppJudge - no RunPod GPU needed for this sweep specifically):
    python -m scripts.run_mcts_sim_budget_sweep \\
        --config config/training/phase4_pilot.yaml \\
        --sim-budgets 15 30 60 \\
        --episodes 150 \\
        --judge-model-path models/llama-3.1-8b-instruct-Q4_K_M.gguf 

Episode count and where to run: 150 episodes/value is a starting point, not a
    calibrated number - enough for strategy_distribution.py's usage percentages to
    stop swinging wildly between runs, not enough for tight statistical confidence.
    This does NOT need RunPod - it needs a real judge, which LlamaCppJudge gives you
    locally (CPU/MPS, per run_phase4_stage3_local.py's own pattern), so running this
    sweep on your Mac is both sufficient and the cheaper choice - save RunPod time
    for the training runs themselves, not for this calibration pass.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts._script_common import build_run
from src.agents.schemas import MacroStrategy
from src.baselines.rule_based_controller import make_randomized_opponent_factory
from src.common.config_loader import load_config
from src.evaluation.strategy_distribution import compute_distribution, format_report
from src.sbso.ablation_strategies import AblationConfig
from src.sbso.dspy_compiler import MockDSPyCompiler, RealDSPyCompiler
from src.sbso.judge import LlamaCppJudge
from src.sbso.macro_executor import MacroStrategyExecutor
from src.sbso.match_trainer import MatchLevelSBSOTrainer
from src.sbso.mcts import MCTS
from src.sbso.opponent_pool import OpponentPool
from src.sbso.recompilation_scheduler import RecompilationScheduler
from src.sbso.self_checkpoint_manager import SelfCheckpointManager
from src.sbso.simulation_backend import PyBulletSimulationBackend
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv

STRATEGIES = list(MacroStrategy)


def _run_one_sweep_point(config: dict, sim_budget: int, episodes: int, judge_model_path: str,
                          real_dspy: bool, out_root: Path) -> dict:
    out_dir = out_root / f"sim_budget_{sim_budget}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "training_pairs.jsonl"
    pairs_file = pairs_path.open("a")
    pairs_written = 0

    def _on_episode_end(ep, trainer):
        nonlocal pairs_written
        for tagged_ep, state, strategy in trainer.training_pairs[pairs_written:]:
            text = state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", "")
            strat = strategy.value if hasattr(strategy, "value") else str(strategy)
            pairs_file.write(json.dumps({"episode": tagged_ep, "lssd_text": text, "strategy": strat}) + "\n")
        pairs_written = len(trainer.training_pairs)
        pairs_file.flush()

    env_cfg = EnvConfig.from_config(use_gui=False)
    env = PyBulletSumoEnv(env_config=env_cfg)
    executor = MacroStrategyExecutor()
    judge = LlamaCppJudge(model_path=judge_model_path)
    backend = PyBulletSimulationBackend(env=env, executor=executor, judge=judge)
    mcts = MCTS(backend=backend, strategies=STRATEGIES, sim_budget=sim_budget,
                judge_prune_threshold=float(config["mcts"].get("judge_prune_threshold", 0.0)))

    checkpoint_mgr = SelfCheckpointManager(interval=max(1, episodes // 3))
    opponent_pool = OpponentPool(warmup_episodes=min(1, episodes), total_episodes=episodes,
                                  self_checkpoint_manager=checkpoint_mgr)
    dspy_compiler = RealDSPyCompiler() if real_dspy else MockDSPyCompiler()

    trainer = MatchLevelSBSOTrainer(
        ablation=AblationConfig.for_variant("none"),
        mcts=mcts, opponent_pool=opponent_pool,
        scheduler=RecompilationScheduler(k_episodes=max(1, episodes // 3), window_w=10, delta=0.05),
        checkpoint_mgr=checkpoint_mgr, dspy_compiler=dspy_compiler, strategies=STRATEGIES,
        env=env, executor=executor,
        lssd_encoder=__import__("src.data.lssd_encoder", fromlist=["LSSDEncoder"]).LSSDEncoder.from_config(),
        opponent_factory=lambda kind: make_randomized_opponent_factory()(),
        episodes=episodes, decision_cycles=int(config.get("decision_cycles", 3)),
        max_decisions_per_match=int(config.get("max_decisions_per_match", 40)),
        on_episode_end=_on_episode_end,
    )

    print(f"\n[sweep] sim_budget={sim_budget}  episodes={episodes}  starting...")
    t0 = time.perf_counter()
    summary = trainer.run()
    elapsed = time.perf_counter() - t0
    pairs_file.close()
    env.close()

    win_rate = sum(trainer._win_history) / len(trainer._win_history) if trainer._win_history else None
    print(f"[sweep] sim_budget={sim_budget} done in {elapsed:.1f}s  win_rate={win_rate}  judge_calls={judge.call_count}")

    return {
        "sim_budget": sim_budget, "episodes": episodes, "elapsed_seconds": round(elapsed, 1),
        "win_rate": win_rate, "judge_call_count": judge.call_count,
        "avg_decision_seconds": summary.get("avg_decision_seconds"),
        "training_pairs_path": str(pairs_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MCTS sim_budget sensitivity sweep.")
    p.add_argument("--config", default="config/training/phase4_pilot.yaml")
    p.add_argument("--sim-budgets", type=int, nargs="+", default=[15, 30, 60])
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--judge-model-path", required=True)
    p.add_argument("--real-dspy", action="store_true")
    p.add_argument("--output-dir", default="results/mcts_sim_budget_sweep")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for sim_budget in args.sim_budgets:
        result = _run_one_sweep_point(config, sim_budget, args.episodes, args.judge_model_path,
                                       args.real_dspy, out_root)
        results.append(result)

    print("\n" + "=" * 70)
    print("SIM_BUDGET SWEEP SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"sim_budget={r['sim_budget']:<4}  win_rate={r['win_rate']}  "
              f"avg_decision_seconds={r['avg_decision_seconds']}  judge_calls={r['judge_call_count']}")

    print()
    for r in results:
        print(f"\n--- strategy distribution @ sim_budget={r['sim_budget']} ---")
        dist = compute_distribution(r["training_pairs_path"], n_buckets=3)
        print(format_report(dist))

    (out_root / "sweep_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote sweep_summary.json -> {out_root / 'sweep_summary.json'}")
    print(
        "\nRead this alongside cost: does win_rate/strategy diversity actually improve "
        "meaningfully as sim_budget rises, relative to the extra wall-clock and judge-call "
        "cost each step costs? If 30 looks about as good as 60, that's your evidence "
        "for keeping the cheaper value for the full run."
    )


if __name__ == "__main__":
    main()
