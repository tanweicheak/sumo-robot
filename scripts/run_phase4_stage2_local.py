"""
scripts/run_phase4_stage2_local.py

Phase: Phase 4 (Stage 2)
Purpose: Run the SBSO training loop against REAL PyBullet physics and a REAL
    (in-process llama.cpp) Judge, on your Mac, to prove the code path works before
    spending RunPod time/money on Stage 3's full-scale cloud run.

    DSPy recompilation defaults to MockDSPyCompiler here (RealDSPyCompiler needs an
    actual llama.cpp HTTP server running separately - see dspy_compiler.py's D5b fix
    docstring). Pass --real-dspy once you have one running (dspy_server_url comes from
    config, not a flag - see config/stage2_local.yaml).

    KNOWN SIMPLIFICATION (see training_loop.py / stage2_wiring.py docstrings): one
    MCTS-informed decision is committed for real per "episode" here, not a full match
    played to a natural conclusion. That's the Stage 3 match-runner's job.

    Stable parameters (judge_model_path, checkpoint_interval, dspy_server_url) live in
    the --config YAML - see config/stage2_local.yaml. Sweep-worthy parameters
    (--episodes, --ablation, --real-dspy) stay as CLI flags.

Usage:
    python scripts/run_phase4_stage2_local.py --config config/stage2_local.yaml --episodes 10
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
from scripts._script_common import build_run
from src.sbso.ablation_strategies import AblationConfig
from src.sbso.dspy_compiler import MockDSPyCompiler, RealDSPyCompiler
from src.sbso.judge import LlamaCppJudge
from src.sbso.macro_executor import MacroStrategyExecutor
from src.sbso.mcts import MCTS
from src.sbso.opponent_pool import OpponentPool
from src.sbso.recompilation_scheduler import RecompilationScheduler
from src.sbso.self_checkpoint_manager import SelfCheckpointManager
from src.sbso.simulation_backend import PyBulletSimulationBackend
from src.sbso.stage2_wiring import build_real_outcome_extractor, build_real_root_state_builder
from src.sbso.training_loop import SBSOTrainer
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


def _add_stage2_local_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--real-dspy", action="store_true")


def main() -> None:
    config, ctx, args = build_run(
        phase="phase4_stage2_local", description=__doc__, extra_args=_add_stage2_local_args,
    )
    print(f"[stage2] run_id={ctx.run_id}")

    judge_model_path = config["judge_model_path"]
    checkpoint_interval = int(config.get("checkpoint_interval", 5))
    dspy_server_url = config.get("dspy_server_url")

    ablation = AblationConfig.for_variant(args.ablation)
    strategies = list(MacroStrategy)

    # --- Real physics + real perception ------------------------------------------------
    env = PyBulletSumoEnv(env_config=EnvConfig(use_gui=False))
    env.reset()
    perception_agent = PerceptionAgent()
    perception_agent.reset()

    executor = MacroStrategyExecutor()
    judge = LlamaCppJudge(model_path=judge_model_path)
    backend = PyBulletSimulationBackend(env, executor, judge, cycles_per_node=3)

    mcts = MCTS(
        backend, strategies, sim_budget=30, horizon=4,
        judge_prune_threshold=0.3 if ablation.judge_enabled else 0.0,
    )

    # --- Real root_state_builder / outcome_extractor (this is what Stage 2 adds) -------
    root_state_builder = build_real_root_state_builder(backend, perception_agent)
    outcome_extractor = build_real_outcome_extractor(backend)

    # --- Optional: a live StrategyAgent that actually receives DSPy recompiles ---------
    strategy_agent = StrategyAgent(client=None)   # wire a real SLMClient once TEA/OAA/SA run live

    if args.real_dspy:
        if not dspy_server_url:
            raise SystemExit("--real-dspy requires dspy_server_url set in the config "
                              "(start a llama.cpp server first)")
        dspy_compiler = RealDSPyCompiler(llama_cpp_api_base=dspy_server_url)
    else:
        dspy_compiler = MockDSPyCompiler()

    checkpoint_mgr = SelfCheckpointManager(interval=checkpoint_interval)
    trainer = SBSOTrainer(
        ablation=ablation,
        mcts=mcts,
        opponent_pool=OpponentPool(
            warmup_episodes=min(5, args.episodes), total_episodes=args.episodes,
            self_checkpoint_manager=checkpoint_mgr,
        ),
        scheduler=RecompilationScheduler(k_episodes=5, window_w=10, delta=0.05),
        checkpoint_mgr=checkpoint_mgr,
        dspy_compiler=dspy_compiler,
        strategies=strategies,
        episodes=args.episodes,
        root_state_builder=root_state_builder,
        outcome_extractor=outcome_extractor,
        on_recompile=strategy_agent.update_prompt_program,
    )

    print(f"[stage2] starting {args.episodes} real-physics episodes "
          f"(judge={args.judge_model_path}, real_dspy={args.real_dspy})")
    t0 = time.monotonic()
    summary = trainer.run()
    wall_clock_s = time.monotonic() - t0
    summary["wall_clock_s"] = wall_clock_s
    summary["sec_per_episode"] = wall_clock_s / max(1, args.episodes)

    print(f"[stage2] done in {wall_clock_s:.1f}s ({summary['sec_per_episode']:.2f}s/episode)")
    print(f"[stage2] summary={summary}")
    print(f"[stage2] judge.call_count={judge.call_count}")


if __name__ == "__main__":
    main()