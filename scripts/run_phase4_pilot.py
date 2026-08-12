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

from scripts._script_common import build_run
from src.agents.schemas import MacroStrategy
from src.baselines.ppo_controller import PPOController
from src.baselines.rule_based_controller import make_rule_based_policy
from src.common.config_loader import load_config
from src.data.lssd_encoder import LSSDEncoder
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


def main() -> None:
    config, ctx = build_run(phase="phase4_pilot", description="Phase 4 pilot: real timing/cost measurement.")
    inference_cfg = load_config("config/inference.yaml")
    sg = inference_cfg["sglang"]

    print("[pilot] connecting to SGLang servers (must already be running - see docstring)...")
    judge = SGLangJudge(server_url=sg["judge_server_url"], temperature=float(sg.get("temperature", 0.0)))

    dspy_compiler = RealDSPyCompiler(sglang_api_base=f"{sg['agent_server_url']}/v1")

    print("[pilot] building PyBullet env...")
    env_cfg = EnvConfig.from_config(use_gui=False, enable_reward_shaping=False)
    env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=make_rule_based_policy())
    env.reset()

    executor = MacroStrategyExecutor()
    backend = PyBulletSimulationBackend(
        env=env, executor=executor, judge=judge,
        cycles_per_node=config["mcts"]["decision_cycles"],
    )
    mcts = MCTS(
        backend, STRATEGIES,
        sim_budget=int(config["mcts"]["sim_budget"]),
        horizon=int(config["mcts"]["horizon"]),
        judge_prune_threshold=float(config["mcts"]["judge_prune_threshold"]),
    )

    ablation = AblationConfig.for_variant(config.get("ablation", {}).get("strategy", "none"))
    opponent_factory = _make_opponent_factory(config["opponent_pool"]["pilot_scope"])

    trainer = MatchLevelSBSOTrainer(
        ablation=ablation,
        mcts=mcts,
        opponent_pool=OpponentPool(
            warmup_episodes=config["opponent_pool"]["warmup_episodes"],
            total_episodes=config["episodes_total"],
        ),
        scheduler=RecompilationScheduler(
            k_episodes=int(config["dspy_recompilation"]["k_rollout_batches"]),
            window_w=int(config["dspy_recompilation"]["rolling_window_w"]),
            delta=float(config["dspy_recompilation"]["reward_drop_threshold_delta"]),
        ),
        checkpoint_mgr=SelfCheckpointManager(interval=int(config["self_checkpoint_interval_episodes"])),
        dspy_compiler=dspy_compiler,
        strategies=STRATEGIES,
        env=env,
        executor=executor,
        lssd_encoder=LSSDEncoder.from_config(),
        opponent_factory=opponent_factory,
        episodes=int(config["episodes_total"]),
        decision_cycles=int(config["mcts"]["decision_cycles"]),
        max_decisions_per_match=int(config["mcts"]["max_decisions_per_match"]),
    )

    print(f"[pilot] running {config['episodes_total']} episodes (real matches, real MCTS, "
          f"real Judge, real DSPy)... this will take a while.")
    summary = trainer.run()
    env.close()
    print(f"[pilot] summary: {summary}")

    cp = config["cost_projection"]
    projection = project_full_run(
        pilot_episodes=summary["episodes"],
        pilot_wall_clock_s=summary["total_wall_seconds"],
        full_episodes_per_variant=int(cp["full_episodes_per_variant"]),
        num_variants=int(cp["num_variants"]),
        gpu_rate_usd_per_hr=float(cp["gpu_rate_usd_per_hr"]),
    )
    print(f"[pilot] PROJECTED FULL RUN (5 variants x {cp['full_episodes_per_variant']} eps): {projection}")


if __name__ == "__main__":
    main()