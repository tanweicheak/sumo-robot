"""
scripts.run_phase4_pilot

Phase: Phase 4 (Stage 3 pilot)
Purpose: Run the SBSO training loop at pilot scale (500 episodes, 1 variant) against the
    REAL SGLang-served models and REAL PyBullet physics, on RunPod. Measures real
    wall-clock time per episode/decision and extrapolates cost for the full 5x5000
    run - the whole point of running this BEFORE committing to the full scale.

    Prerequisites (on the RunPod instance):
        bash scripts/launch_sglang_servers.sh
        (wait for "Both SGLang servers are up")

Usage:
    python -m scripts.run_phase4_pilot --config config/training/phase4_pilot.yaml \
        --results-dir results/phase4_pilot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts._script_common import build_run, poll_gpu_stats, setup_logging
from src.agents.schemas import MacroStrategy
from src.baselines.ppo_controller import PPOController
from src.baselines.rule_based_controller import make_rule_based_policy
from src.common.config_loader import load_config
from src.data.lssd_encoder import LSSDEncoder
from src.finetuning.calibration_texts import write_calibration_file
from src.finetuning.cost_projection import project_full_run
from src.sbso.ablation_strategies import AblationConfig
from src.sbso.dspy_compiler import RealDSPyCompiler
from src.sbso.judge import SGLangJudge
from src.sbso.macro_executor import MacroStrategyExecutor
from src.sbso.match_trainer import MatchLevelSBSOTrainer
from src.sbso.mcts import MCTS
from src.sbso.opponent_pool import OpponentPool
from src.sbso.recompilation_scheduler import RecompilationScheduler
from src.sbso.self_checkpoint_manager import SelfCheckpointManager
from src.sbso.simulation_backend import PyBulletSimulationBackend
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv

STRATEGIES = list(MacroStrategy)


def _make_opponent_factory(pilot_scope: list[str]):
    """Returns a function mapping opponent_type_str -> opponent_policy callable.
    Scoped to Baseline 1/2 only for the pilot (see config header note)."""
    _ppo_cache = {}

    def factory(opp_type: str):
        if opp_type == "baseline1":
            return make_rule_based_policy()
        if opp_type == "baseline2":
            if "ppo" not in _ppo_cache:
                _ppo_cache["ppo"] = PPOController.load(
                    "checkpoints/baseline2_ppo/ppo_baseline2.zip",
                    "checkpoints/baseline2_ppo/vecnormalize.pkl",
                )
            return _ppo_cache["ppo"]
        # self_checkpoint would go here in the full run; pilot excludes it.
        return make_rule_based_policy()

    return factory


def _add_pilot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes-override", type=int, default=None,
                         help="Override config's episodes_total - mainly for a quick smoke test "
                          "against the SAME config you'll use for the real run, without needing "
                          "a second, smaller config file just to change the episode count.")
    parser.add_argument("--checkpoint-interval-override", type=int, default=None,
                         help="Override config's self_checkpoint_interval_episodes, same reasoning.")
    parser.add_argument("--use-wandb", action="store_true", help="Report training metrics to Weights & Biases")


def main() -> None:
    config, ctx, args = build_run(
        phase="phase4_pilot", description="Phase 4 pilot: real timing/cost measurement.",
        extra_args=_add_pilot_args,
    )
    out_dir = Path(config.get("checkpoint_output_dir", f"checkpoints/{ctx.run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir, name="phase4_pilot")
    logger.info(f"run_id={ctx.run_id}")

    use_wandb = args.use_wandb
    if use_wandb:
        import wandb
        wandb.init(project="sumo-sbso", name=f"sbso-{ctx.run_id}", config=config)
        logger.info("wandb enabled - project=sumo-sbso run=sbso-%s", ctx.run_id)

    inference_cfg = load_config("config/inference.yaml")
    sg = inference_cfg["sglang"]

    logger.info("connecting to SGLang servers (must already be running - see docstring)...")
    judge = SGLangJudge(server_url=sg["judge_server_url"], temperature=float(sg.get("temperature", 0.0)))

    dspy_compiler = RealDSPyCompiler(sglang_api_base=f"{sg['agent_server_url']}/v1")

    logger.info("building PyBullet env...")
    env_cfg = EnvConfig.from_config(use_gui=False, enable_reward_shaping=False)
    env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=make_rule_based_policy())
    env.reset()

    ablation = AblationConfig.for_variant(config.get("ablation", {}).get("strategy", "none"))

    executor = MacroStrategyExecutor()
    backend = PyBulletSimulationBackend(
        env=env, executor=executor, judge=judge,
        cycles_per_node=config["mcts"]["decision_cycles"],
        judge_enabled=ablation.judge_enabled,
    )
    mcts = MCTS(
        backend, STRATEGIES,
        sim_budget=int(config["mcts"]["sim_budget"]),
        horizon=int(config["mcts"]["horizon"]),
        # no_judge ablation: force 0.0 regardless of config, so nothing gets pruned -
        # config's value only applies when the Judge is actually enabled.
        judge_prune_threshold=float(config["mcts"]["judge_prune_threshold"]) if ablation.judge_enabled else 0.0,
    )

    opponent_factory = _make_opponent_factory(config["opponent_pool"]["pilot_scope"])

    episodes = args.episodes_override or int(config["episodes_total"])
    checkpoint_interval = args.checkpoint_interval_override or int(config["self_checkpoint_interval_episodes"])
    if args.episodes_override or args.checkpoint_interval_override:
        logger.info(f"override active: episodes={episodes} checkpoint_interval={checkpoint_interval}")

    checkpoint_mgr = SelfCheckpointManager(interval=checkpoint_interval)

    # Crash-safety: persist progress + training pairs incrementally, so an interrupted
    # RunPod pod loses at most one checkpoint-interval's worth of progress, not the
    # whole run. NOT crash-resumption - the run itself does not pick back up from a
    # saved point automatically.
    progress_path = out_dir / "progress.json"
    pairs_path = out_dir / "training_pairs.jsonl"
    prompt_history_path = out_dir / "prompt_history.jsonl"
    calibration_path = out_dir / "mcts_calibration.jsonl"
    pairs_file = pairs_path.open("a")
    prompt_history_file = prompt_history_path.open("a")
    calibration_file = calibration_path.open("a")
    logger.info(f"progress -> {progress_path}   training_pairs -> {pairs_path}   "
          f"prompt_history -> {prompt_history_path}   mcts_calibration -> {calibration_path}")

    pairs_written = 0
    recompiles_written = 0
    calibration_written = 0

    def _on_episode_end(ep: int, trainer: MatchLevelSBSOTrainer) -> None:
        nonlocal pairs_written, recompiles_written, calibration_written
        # Write only the pairs collected since the last call (incremental, crash-safe -
        # training_pairs accumulates for the whole run, so this avoids rewriting
        # everything from scratch every episode).
        for tagged_ep, state, strategy in trainer.training_pairs[pairs_written:]:
            text = state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", "")
            strat = strategy.value if hasattr(strategy, "value") else str(strategy)
            pairs_file.write(json.dumps({"episode": tagged_ep, "lssd_text": text, "strategy": strat}) + "\n")
        pairs_written = len(trainer.training_pairs)
        pairs_file.flush()

        # Record which prompt version was active over which episode range and why each
        # recompile fired - previously computed (should_recompile's reason) then thrown
        # away, so there was no way to answer "which prompt made this decision" or
        # "how many times did trigger (a) vs (b) actually fire" after a run finished.
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

        if use_wandb:
            # trainer._win_history is a deque(maxlen=scheduler.window_w) - it already
            # never holds more than the window size, so no slicing is needed (deques
            # don't support slice indexing like list[-50:] anyway - that's what crashed
            # here). list(...) materializes it for sum()/len() below.
            recent = list(trainer._win_history)
            elapsed_hours = sum(trainer.timing["episode_seconds"]) / 3600.0
            cost_so_far_usd = elapsed_hours * float(config["cost_projection"]["gpu_rate_usd_per_hr"])
            log_payload = {
                "episode": ep,
                "rolling_winrate": sum(recent) / len(recent) if recent else None,
                "training_pairs_collected": len(trainer.training_pairs),
                "prompt_version": trainer.prompt_version,
                "checkpoints_taken": len(trainer.checkpoint_mgr._checkpoints),
                # Live cost accumulation - pure arithmetic on wall-clock already
                # tracked (trainer.timing) x the configured GPU rate. No RunPod-side
                # telemetry needed for this piece - it's the "how many dollars have
                # I actually spent so far" counter, distinct from the GPU
                # utilization poller below, which DOES need the real box.
                "elapsed_hours": round(elapsed_hours, 3),
                "cost_so_far_usd": round(cost_so_far_usd, 2),
            }
            gpu_stats = poll_gpu_stats()
            if gpu_stats is not None:
                log_payload.update(gpu_stats)
            # Only log a recompile event on the episode it actually happened, not
            # every episode - avoids a wandb chart with a misleading step-function
            # repeated at every log call.
            if trainer.recompile_history and trainer.recompile_history[-1]["episode"] == ep:
                event = trainer.recompile_history[-1]
                log_payload["recompile_trigger_reason"] = event["trigger_reason"]
                log_payload["recompile_accepted"] = event.get("accepted")
            wandb.log(log_payload, step=ep)

    trainer = MatchLevelSBSOTrainer(
        ablation=ablation,
        mcts=mcts,
        opponent_pool=OpponentPool(
            warmup_episodes=config["opponent_pool"]["warmup_episodes"],
            total_episodes=episodes,
        ),
        scheduler=RecompilationScheduler(
            k_episodes=int(config["dspy_recompilation"]["k_rollout_batches"]),
            window_w=int(config["dspy_recompilation"]["rolling_window_w"]),
            delta=float(config["dspy_recompilation"]["reward_drop_threshold_delta"]),
        ),
        checkpoint_mgr=checkpoint_mgr,
        dspy_compiler=dspy_compiler,
        strategies=STRATEGIES,
        env=env,
        executor=executor,
        lssd_encoder=LSSDEncoder.from_config(),
        opponent_factory=opponent_factory,
        episodes=episodes,
        decision_cycles=int(config["mcts"]["decision_cycles"]),
        max_decisions_per_match=int(config["mcts"]["max_decisions_per_match"]),
        on_episode_end=_on_episode_end,
    )

    logger.info(f"running {episodes} episodes (real matches, real MCTS, "
          f"real Judge, real DSPy)... this will take a while.")
    summary = trainer.run()
    pairs_file.close()
    prompt_history_file.close()
    calibration_file.close()
    env.close()
    logger.info(f"summary: {summary}")

    calib_path = write_calibration_file(trainer.training_pairs, out_dir / "gptq_calibration_texts.txt")
    logger.info(f"wrote calibration texts -> {calib_path}")

    cp = config["cost_projection"]
    projection = project_full_run(
        pilot_episodes=summary["episodes"],
        pilot_wall_clock_s=summary["total_wall_seconds"],
        full_episodes_per_variant=int(cp["full_episodes_per_variant"]),
        num_variants=int(cp["num_variants"]),
        gpu_rate_usd_per_hr=float(cp["gpu_rate_usd_per_hr"]),
    )
    logger.info(f"PROJECTED FULL RUN (5 variants x {cp['full_episodes_per_variant']} eps): {projection}")

    if use_wandb:
        wandb.log({"final/" + k: v for k, v in summary.items() if isinstance(v, (int, float))})
        wandb.log({"projection/" + k: v for k, v in projection.items() if isinstance(v, (int, float))})
        wandb.finish()


if __name__ == "__main__":
    main()