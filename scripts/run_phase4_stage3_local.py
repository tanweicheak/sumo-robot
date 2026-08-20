"""
scripts/run_phase4_stage3_local.py

Phase: Phase 4 (Stage 3, LOCAL variant)
Purpose: Same MatchLevelSBSOTrainer as run_phase4_pilot.py (the canonical Stage 3
    trainer - see src/sbso/match_trainer.py), but wired for your Mac instead of RunPod:
    LlamaCppJudge (in-process, CPU/MPS) instead of SGLangJudge, DSPy defaults to
    MockDSPyCompiler (RealDSPyCompiler needs a local llama.cpp HTTP server - see
    dspy_compiler.py's D5b fix docstring; pass --real-dspy once you have one running).

    This exists ONLY because SGLang does not run on macOS at all - it is not a
    parallel design, it is the same trainer class run_phase4_pilot.py uses, so a
    successful local run here actually validates the real production code path
    (previously, before this rewrite, this script used a separate MatchRunner
    implementation that was NOT what production actually runs - see chat history).

    Reuses _make_opponent_factory() from run_phase4_pilot.py rather than
    reimplementing opponent construction - same rule-based/PPO loading logic works
    identically locally.

Usage:
    python scripts/run_phase4_stage3_local.py --config config/stage3_local.yaml --episodes 3

Keep --episodes small for local testing - each match can be dozens of real decisions,
each with its own MCTS search, so this is much slower per-episode than a single
decision would be. Start with 2-3 to sanity check before scaling up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._script_common import build_run
from scripts.run_phase4_pilot import _make_opponent_factory, STRATEGIES
from src.baselines.rule_based_controller import make_rule_based_policy
from src.data.lssd_encoder import LSSDEncoder
from src.finetuning.calibration_texts import write_calibration_file
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


def _add_stage3_local_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--decision-cycles", type=int, default=3)
    parser.add_argument("--sim-budget", type=int, default=8,
                         help="MCTS expansions per decision. Production/real training wants ~30; "
                              "kept low by default here since each expansion's rollout calls the "
                              "REAL Judge (real LLM inference).")
    parser.add_argument("--horizon", type=int, default=2,
                         help="Rollout depth per expansion. Production wants ~4; lower here for speed.")
    parser.add_argument("--max-decisions-per-match", type=int, default=10)
    parser.add_argument("--real-dspy", action="store_true")


def main() -> None:
    config, ctx, args = build_run(
        phase="phase4_stage3_local", description=__doc__, extra_args=_add_stage3_local_args,
    )
    print(f"[stage3-local] run_id={ctx.run_id}")

    judge_model_path = config["judge_model_path"]
    checkpoint_interval = int(config.get("checkpoint_interval", 1))
    dspy_server_url = config.get("dspy_server_url")

    ablation = AblationConfig.for_variant(args.ablation)

    env_cfg = EnvConfig.from_config(use_gui=False, enable_reward_shaping=False)
    env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=make_rule_based_policy())
    env.reset()

    executor = MacroStrategyExecutor()
    judge = LlamaCppJudge(model_path=judge_model_path)
    backend = PyBulletSimulationBackend(
        env, executor, judge, cycles_per_node=args.decision_cycles,
        judge_enabled=ablation.judge_enabled,
    )

    mcts = MCTS(
        backend, STRATEGIES, sim_budget=args.sim_budget, horizon=args.horizon,
        judge_prune_threshold=0.3 if ablation.judge_enabled else 0.0,
    )

    opponent_factory = _make_opponent_factory(pilot_scope=["baseline1", "baseline2"])

    if args.real_dspy:
        if not dspy_server_url:
            raise SystemExit("--real-dspy requires dspy_server_url set in the config "
                              "(a local llama.cpp server)")
        dspy_compiler = RealDSPyCompiler(llama_cpp_api_base=dspy_server_url)
    else:
        dspy_compiler = MockDSPyCompiler()

    checkpoint_mgr = SelfCheckpointManager(interval=checkpoint_interval)

    out_dir = Path(config.get("checkpoint_output_dir", f"checkpoints/{ctx.run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    pairs_path = out_dir / "training_pairs.jsonl"
    prompt_history_path = out_dir / "prompt_history.jsonl"
    calibration_path = out_dir / "mcts_calibration.jsonl"
    pairs_file = pairs_path.open("a")
    prompt_history_file = prompt_history_path.open("a")
    calibration_file = calibration_path.open("a")
    pairs_written = 0
    recompiles_written = 0
    calibration_written = 0

    def _on_episode_end(ep: int, trainer: MatchLevelSBSOTrainer) -> None:
        nonlocal pairs_written, recompiles_written, calibration_written
        for tagged_ep, state, strategy in trainer.training_pairs[pairs_written:]:
            text = state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", "")
            strat = strategy.value if hasattr(strategy, "value") else str(strategy)
            pairs_file.write(json.dumps({"episode": tagged_ep, "lssd_text": text, "strategy": strat}) + "\n")
        pairs_written = len(trainer.training_pairs)
        pairs_file.flush()
        for event in trainer.recompile_history[recompiles_written:]:
            prompt_history_file.write(json.dumps(event) + "\n")
        recompiles_written = len(trainer.recompile_history)
        prompt_history_file.flush()
        for entry in trainer.mcts_calibration_log[calibration_written:]:
            calibration_file.write(json.dumps(entry) + "\n")
        calibration_written = len(trainer.mcts_calibration_log)
        calibration_file.flush()
        if (ep + 1) % checkpoint_interval == 0 or ep == 0:
            progress_path.write_text(json.dumps({
                "episode": ep,
                "prompt_program": trainer.prompt_program,
                "dominant_strategy": trainer.dominant_strategy,
                "checkpoints_taken": len(trainer.checkpoint_mgr._checkpoints),
                "recent_win_history": list(trainer._win_history),
            }, indent=2))

    trainer = MatchLevelSBSOTrainer(
        ablation=ablation,
        mcts=mcts,
        opponent_pool=OpponentPool(
            warmup_episodes=min(1, args.episodes), total_episodes=args.episodes,
            self_checkpoint_manager=checkpoint_mgr,
        ),
        scheduler=RecompilationScheduler(k_episodes=1, window_w=5, delta=0.05),
        checkpoint_mgr=checkpoint_mgr,
        dspy_compiler=dspy_compiler,
        strategies=STRATEGIES,
        env=env,
        executor=executor,
        lssd_encoder=LSSDEncoder.from_config(),
        opponent_factory=opponent_factory,
        episodes=args.episodes,
        decision_cycles=args.decision_cycles,
        max_decisions_per_match=args.max_decisions_per_match,
        on_episode_end=_on_episode_end,
    )

    print(f"[stage3-local] starting {args.episodes} FULL MATCHES "
          f"(judge={judge_model_path}, real_dspy={args.real_dspy})")
    summary = trainer.run()
    pairs_file.close()
    prompt_history_file.close()
    calibration_file.close()
    env.close()
    print(f"[stage3-local] summary={summary}")
    print(f"[stage3-local] judge.call_count={judge.call_count}")

    calib_path = write_calibration_file(trainer.training_pairs, out_dir / "gptq_calibration_texts.txt")
    print(f"[stage3-local] wrote calibration texts -> {calib_path}")


if __name__ == "__main__":
    main()