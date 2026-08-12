"""
scripts/run_phase4_stage3_cloud.py

Phase: Phase 4 (Stage 3, PRODUCTION/RunPod variant)
Purpose: The actual production training run - Benchmark 2 (full SBSO) or one of the
    four ablations, against real PyBullet physics, a real SGLang-served Judge, and
    (optionally) real DSPy recompilation against the same SGLang server. This is
    NOT run_phase4_stage3_local.py - that script is a GPU-free Mac verification tool
    with fast-smoke-test defaults; this one is meant to run on a RunPod GPU pod for
    the real, full-length, full-fidelity training run and refuses to guess at values
    that determine real training quality.

    Deliberately requires --sim-budget, --horizon, and --cycles-per-node with NO
    defaults - _shared_defaults.yaml doesn't specify these, and reusing
    run_phase4_stage3_local.py's fast-testing defaults (sim_budget=8, horizon=2) here
    would silently produce a real training run at local-smoke-test fidelity. Pass the
    values your local pilot run validated.

    Also requires the DSPy recompilation trigger values (config's dspy_recompilation
    block) to be non-null, UNLESS overridden via --k-episodes/--window-w/--delta -
    _shared_defaults.yaml currently has these as PLACEHOLDER nulls pending pilot
    calibration; this script fails loudly rather than silently falling back to
    unvalidated pilot numbers for a real run.

    Persists progress every --checkpoint-interval episodes to
    <checkpoint_output_dir>/progress.json (prompt_program, dominant_strategy, episode
    index, win history) and appends every episode's training pairs to
    <checkpoint_output_dir>/training_pairs.jsonl - so an interrupted pod loses at most
    one checkpoint-interval's worth of progress, not the whole run. This is
    crash-SAFETY (nothing is lost), not crash-RESUMPTION (the run itself does not
    currently resume from a saved point - that would also need MCTS/env state, not
    just this bookkeeping, and is a further piece of work, not attempted here).

Usage:
    python scripts/run_phase4_stage3_cloud.py \\
        --config config/phase4_full_sbso.yaml \\
        --sglang-url http://127.0.0.1:30000 \\
        --sim-budget 30 --horizon 4 --cycles-per-node 3 \\
        --k-episodes 5 --window-w 10 --delta 0.05    # only if config's values are still null
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.perception_agent import PerceptionAgent
from src.agents.schemas import MacroStrategy
from src.agents.strategy_agent import StrategyAgent
from src.common.config_loader import load_config
from src.finetuning.calibration_texts import write_calibration_file
from src.sbso.ablation_strategies import AblationConfig
from src.sbso.dspy_compiler import MockDSPyCompiler, RealDSPyCompiler
from src.sbso.judge import SGLangJudge
from src.sbso.macro_executor import MacroStrategyExecutor
from src.sbso.match_runner import MatchRunner
from src.sbso.mcts import MCTS
from src.sbso.opponent_pool import OpponentPool
from src.sbso.recompilation_scheduler import RecompilationScheduler
from src.sbso.self_checkpoint_manager import SelfCheckpointManager
from src.sbso.simulation_backend import PyBulletSimulationBackend
from src.sbso.training_loop import SBSOTrainer
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="e.g. config/phase4_full_sbso.yaml")
    ap.add_argument("--sglang-url", required=True, help="e.g. http://127.0.0.1:30000")
    ap.add_argument("--sim-budget", type=int, required=True,
                     help="No default on purpose - use the value your local pilot validated, "
                          "not a guess. run_phase4_stage3_local.py's fast-testing default (8) "
                          "would silently degrade a real training run if reused here.")
    ap.add_argument("--horizon", type=int, required=True, help="Same reasoning as --sim-budget.")
    ap.add_argument("--cycles-per-node", type=int, required=True, help="Same reasoning as --sim-budget.")
    ap.add_argument("--real-dspy", action="store_true", help="Use RealDSPyCompiler against SGLang instead of MockDSPyCompiler")
    ap.add_argument("--k-episodes", type=int, default=None, help="Override if config's dspy_recompilation.k_rollout_batches is null")
    ap.add_argument("--window-w", type=int, default=None, help="Override if config's rolling_window_w is null")
    ap.add_argument("--delta", type=float, default=None, help="Override if config's reward_drop_threshold_delta is null")
    ap.add_argument("--checkpoint-interval-override", type=int, default=None,
                     help="Override config's self_checkpoint_interval_episodes (mainly for a smaller cloud smoke test)")
    ap.add_argument("--episodes-override", type=int, default=None,
                     help="Override config's episodes_total (mainly for a smaller cloud smoke test)")
    args = ap.parse_args()

    config = load_config(args.config)
    ablation_name = config.get("ablation", {}).get("strategy", "none")
    ablation = AblationConfig.for_variant(ablation_name)
    strategies = list(MacroStrategy)

    episodes = args.episodes_override or int(config["episodes_total"])
    checkpoint_interval = args.checkpoint_interval_override or int(config["self_checkpoint_interval_episodes"])

    opp_cfg = config.get("opponent_pool", {})
    warmup_episodes = int(opp_cfg.get("warmup_episodes", episodes // 10))
    target_counts = opp_cfg.get("full_run_targets")   # None -> OpponentPool computes an even 3-way split

    # DSPy trigger values: refuse to silently fall back to unvalidated pilot numbers.
    dspy_cfg = config.get("dspy_recompilation", {})
    k_episodes = args.k_episodes if args.k_episodes is not None else dspy_cfg.get("k_rollout_batches")
    window_w = args.window_w if args.window_w is not None else dspy_cfg.get("rolling_window_w")
    delta = args.delta if args.delta is not None else dspy_cfg.get("reward_drop_threshold_delta")
    if ablation.dspy_enabled and (k_episodes is None or window_w is None or delta is None):
        raise SystemExit(
            "config's dspy_recompilation block still has PLACEHOLDER null values "
            "(pilot calibration pending). Refusing to guess for a real training run - "
            "pass --k-episodes/--window-w/--delta explicitly, or fill in the config."
        )

    out_dir = Path(config.get("checkpoint_output_dir", "checkpoints/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    pairs_path = out_dir / "training_pairs.jsonl"

    print(f"[stage3-cloud] variant={config.get('variant_name', '?')} ablation={ablation_name} "
          f"episodes={episodes} checkpoint_interval={checkpoint_interval}")
    print(f"[stage3-cloud] sim_budget={args.sim_budget} horizon={args.horizon} "
          f"cycles_per_node={args.cycles_per_node}")
    print(f"[stage3-cloud] progress -> {progress_path}   training_pairs -> {pairs_path}")

    env = PyBulletSumoEnv(env_config=EnvConfig(use_gui=False))
    env.reset()
    perception_agent = PerceptionAgent()
    perception_agent.reset()

    executor = MacroStrategyExecutor()
    judge = SGLangJudge(server_url=args.sglang_url)
    backend = PyBulletSimulationBackend(env, executor, judge, cycles_per_node=args.cycles_per_node)

    mcts = MCTS(
        backend, strategies, sim_budget=args.sim_budget, horizon=args.horizon,
        judge_prune_threshold=0.3 if ablation.judge_enabled else 0.0,
    )
    max_decisions = max(1, env.max_steps // args.cycles_per_node)
    match_runner = MatchRunner(backend, perception_agent, max_decisions_per_match=max_decisions)

    strategy_agent = StrategyAgent(client=None)

    if args.real_dspy:
        dspy_compiler = RealDSPyCompiler(sglang_api_base=f"{args.sglang_url}/v1")
    else:
        dspy_compiler = MockDSPyCompiler()

    checkpoint_mgr = SelfCheckpointManager(interval=checkpoint_interval)
    opponent_pool = OpponentPool(
        warmup_episodes=warmup_episodes, total_episodes=episodes,
        self_checkpoint_manager=checkpoint_mgr, target_counts=target_counts,
    )
    scheduler = RecompilationScheduler(
        k_episodes=k_episodes or 10**9,   # effectively "never" if DSPy disabled and no value given
        window_w=window_w or 10, delta=delta or 0.05,
    )

    pairs_file = pairs_path.open("a")   # append: safe to resume writing after an interrupted run

    def _persist_progress(episode: int, trainer: SBSOTrainer) -> None:
        progress_path.write_text(json.dumps({
            "episode": episode,
            "prompt_program": trainer.prompt_program,
            "dominant_strategy": trainer.dominant_strategy,
            "checkpoints_taken": len(trainer.checkpoint_mgr._checkpoints),
            "recent_win_history": list(trainer._win_history),
        }, indent=2))

    def play_match(trainer_, episode, opponent):
        won, pairs = match_runner.play_match(trainer_, episode, opponent)
        for ep, state, strategy in pairs:
            text = state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", "")
            strat = strategy.value if hasattr(strategy, "value") else str(strategy)
            pairs_file.write(json.dumps({"episode": ep, "lssd_text": text, "strategy": strat}) + "\n")
        pairs_file.flush()
        if (episode + 1) % checkpoint_interval == 0 or episode == 0:
            _persist_progress(episode, trainer_)
        return won, pairs

    trainer = SBSOTrainer(
        ablation=ablation, mcts=mcts, opponent_pool=opponent_pool, scheduler=scheduler,
        checkpoint_mgr=checkpoint_mgr, dspy_compiler=dspy_compiler, strategies=strategies,
        episodes=episodes, on_recompile=strategy_agent.update_prompt_program,
    )

    print(f"[stage3-cloud] starting {episodes} full matches...")
    t0 = time.monotonic()
    try:
        summary = trainer.run_stage3(play_match)
    finally:
        pairs_file.close()
    wall_clock_s = time.monotonic() - t0
    summary["wall_clock_s"] = wall_clock_s
    print(f"[stage3-cloud] done in {wall_clock_s:.1f}s")
    print(f"[stage3-cloud] summary={summary}")
    print(f"[stage3-cloud] judge.call_count={judge.call_count}")

    calib_path = write_calibration_file(trainer.training_pairs, out_dir / "gptq_calibration_texts.txt")
    print(f"[stage3-cloud] wrote calibration texts -> {calib_path}")


if __name__ == "__main__":
    main()
