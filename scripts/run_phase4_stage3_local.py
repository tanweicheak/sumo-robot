"""
scripts/run_phase4_stage3_local.py

Phase: Phase 4 (Stage 3, LOCAL variant)
Purpose: Same real-physics + real-Judge setup as run_phase4_stage2_local.py, but drives
    FULL MATCHES (many decisions each, via MatchRunner) instead of one decision per
    "episode". This is the local, GPU-free way to test the match-runner before trusting
    it on RunPod - identical code path, just swap LlamaCppJudge for SGLangJudge and
    point RealDSPyCompiler at an sglang_api_base when you move to the cloud. Nothing in
    MatchRunner or SBSOTrainer.run_stage3() is SGLang/GPU-specific - only the Judge/DSPy
    backend you plug in determines that, and this script plugs in the CPU/MPS-friendly
    ones on purpose.

Usage:
    python scripts/run_phase4_stage3_local.py --judge-model-path /path/to/judge.gguf --episodes 3

Keep --episodes small for local testing - each match can be dozens of real decisions,
each with its own MCTS search, so this is much slower per-episode than Stage 2's
single-decision script. Start with 2-3 to sanity check before scaling up.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.agents.perception_agent import PerceptionAgent
from src.agents.schemas import MacroStrategy
from src.agents.strategy_agent import StrategyAgent
from src.finetuning.calibration_texts import write_calibration_file
from src.sbso.ablation_strategies import AblationConfig
from src.sbso.dspy_compiler import MockDSPyCompiler, RealDSPyCompiler
from src.sbso.judge import LlamaCppJudge
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
    ap.add_argument("--judge-model-path", required=True)
    ap.add_argument("--episodes", type=int, default=3, help="Number of full MATCHES, not decisions")
    ap.add_argument("--checkpoint-interval", type=int, default=1)
    ap.add_argument("--ablation", default="none")
    ap.add_argument("--cycles-per-node", type=int, default=3)
    ap.add_argument("--sim-budget", type=int, default=8,
                     help="MCTS expansions per decision. Production/real training wants ~30; "
                          "kept low by default here since each expansion's rollout calls the "
                          "REAL Judge (real LLM inference) - 30 vs 8 is roughly 4x slower for "
                          "a local smoke test with no accuracy benefit to that test's purpose.")
    ap.add_argument("--horizon", type=int, default=2,
                     help="Rollout depth per expansion. Production wants ~4; lower here for speed.")
    ap.add_argument("--max-decisions-per-match", type=int, default=None,
                     help="Override the match length cap directly (default: derived from "
                          "env.max_steps // cycles_per_node, ~100 - far more than needed to "
                          "smoke-test the wiring). Try 5-10 for a fast first run.")
    ap.add_argument("--real-dspy", action="store_true")
    ap.add_argument("--dspy-server-url", default=None, help="e.g. http://127.0.0.1:8080/v1 - a LOCAL llama.cpp server, not SGLang (macOS has no SGLang support)")
    ap.add_argument("--calibration-output", default=None,
                     help="If set, write GPTQ calibration text (real lssd_text from this run's "
                          "training_pairs) to this path - one text per line, ready for "
                          "run_export_pipeline.py's --calibration-texts-file")
    args = ap.parse_args()

    ablation = AblationConfig.for_variant(args.ablation)
    strategies = list(MacroStrategy)

    env = PyBulletSumoEnv(env_config=EnvConfig(use_gui=False))
    env.reset()
    perception_agent = PerceptionAgent()
    perception_agent.reset()

    executor = MacroStrategyExecutor()
    judge = LlamaCppJudge(model_path=args.judge_model_path)
    backend = PyBulletSimulationBackend(env, executor, judge, cycles_per_node=args.cycles_per_node)

    mcts = MCTS(
        backend, strategies, sim_budget=args.sim_budget, horizon=args.horizon,
        judge_prune_threshold=0.3 if ablation.judge_enabled else 0.0,
    )

    # Real decision cap, derived from env config, instead of a flat guess (max_steps /
    # cycles_per_node = how many real committed decisions fit in one match) - unless
    # explicitly overridden for a faster local smoke test.
    max_decisions = args.max_decisions_per_match or max(1, env.max_steps // args.cycles_per_node)
    match_runner = MatchRunner(backend, perception_agent, max_decisions_per_match=max_decisions)
    print(f"[stage3-local] max_decisions_per_match={max_decisions} "
          f"(env.max_steps={env.max_steps}, cycles_per_node={args.cycles_per_node})")

    strategy_agent = StrategyAgent(client=None)   # receives DSPy recompiles; not yet driving live decisions

    if args.real_dspy:
        if not args.dspy_server_url:
            raise SystemExit("--real-dspy requires --dspy-server-url (a local llama.cpp server)")
        dspy_compiler = RealDSPyCompiler(llama_cpp_api_base=args.dspy_server_url)
    else:
        dspy_compiler = MockDSPyCompiler()

    checkpoint_mgr = SelfCheckpointManager(interval=args.checkpoint_interval)
    trainer = SBSOTrainer(
        ablation=ablation,
        mcts=mcts,
        opponent_pool=OpponentPool(
            warmup_episodes=min(1, args.episodes), total_episodes=args.episodes,
            self_checkpoint_manager=checkpoint_mgr,
        ),
        scheduler=RecompilationScheduler(k_episodes=1, window_w=5, delta=0.05),
        checkpoint_mgr=checkpoint_mgr,
        dspy_compiler=dspy_compiler,
        strategies=strategies,
        episodes=args.episodes,
        on_recompile=strategy_agent.update_prompt_program,
    )

    def play_match(trainer_, episode, opponent):
        return match_runner.play_match(trainer_, episode, opponent)

    print(f"[stage3-local] starting {args.episodes} FULL MATCHES "
          f"(judge={args.judge_model_path}, real_dspy={args.real_dspy})")
    t0 = time.monotonic()
    summary = trainer.run_stage3(play_match)
    wall_clock_s = time.monotonic() - t0
    summary["wall_clock_s"] = wall_clock_s
    summary["sec_per_match"] = wall_clock_s / max(1, args.episodes)
    summary["decisions_per_match_avg"] = summary["training_pairs"] / max(1, args.episodes)

    print(f"[stage3-local] done in {wall_clock_s:.1f}s ({summary['sec_per_match']:.1f}s/match, "
          f"~{summary['decisions_per_match_avg']:.1f} decisions/match)")
    print(f"[stage3-local] summary={summary}")
    print(f"[stage3-local] judge.call_count={judge.call_count}")

    if args.calibration_output:
        path = write_calibration_file(trainer.training_pairs, args.calibration_output)
        n_written = len(path.read_text().splitlines())
        print(f"[stage3-local] wrote {n_written} calibration texts -> {path}")


if __name__ == "__main__":
    main()