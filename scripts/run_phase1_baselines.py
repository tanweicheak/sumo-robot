"""
scripts.run_phase1_baselines

Phase: Phase 1
Purpose: Train and freeze Baseline 2 (PPO) against the Baseline 1 rule-based
    opponent. Baseline 1 is deterministic and needs no training; this script's job is
    the PPO run plus a sparse-reward sanity evaluation. Supports a staged smoke run by
    setting a small total_timesteps in the config.

    Opponent-diversity fix: training used to run every episode, across the whole
    training run, against ONE frozen RuleBasedController with fixed default
    parameters. PPO would specialize to that single instance's specific blind spots
    rather than learning general pushing competence - the same overfitting risk
    self-play/self-checkpointing exists to avoid on the SBSO side. RandomizedOpponentWrapper
    below assigns a freshly RANDOMIZED opponent (RuleBasedParams.randomized()) on every
    episode reset instead. This also fixes a second, separate bug found in the same
    code path: opponent.reset() was never being called between episodes at all, so a
    rule-based opponent's internal state (edge-avoid timers, commit latch, search
    phase) was silently carrying over from one episode into the next.

    Set training.randomize_opponent: false in config to disable and reproduce the old
    fixed-single-instance behavior (e.g. for an A/B comparison).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from scripts._script_common import build_run
from src.baselines.ppo_controller import FlattenSumoObs
from src.baselines.rule_based_controller import make_randomized_opponent_factory, make_rule_based_policy
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


class RandomizedOpponentWrapper(gym.Wrapper):
    """Assigns a fresh, freshly-.reset() opponent to the underlying PyBulletSumoEnv on
    every reset() call, instead of the one frozen instance the env was constructed
    with. Subclasses gym.Wrapper (matching FlattenSumoObs's own pattern) since this
    gets wrapped again by FlattenSumoObs next - a plain duck-typed stand-in risks
    breaking gymnasium's own Wrapper assumptions downstream."""

    def __init__(self, env: gym.Env, opponent_factory) -> None:
        super().__init__(env)
        self.opponent_factory = opponent_factory
        # Set once here too, so the env has a valid opponent even if something reads
        # env.opponent_policy before the first reset() call.
        self.env.unwrapped.opponent_policy = opponent_factory()

    def reset(self, **kwargs):
        opponent = self.opponent_factory()
        if hasattr(opponent, "reset"):
            opponent.reset()
        self.env.unwrapped.opponent_policy = opponent
        return self.env.reset(**kwargs)


def _make_env_fn(env_cfg: EnvConfig, opponent_factory):
    """Return a thunk building one flattened sumo env whose opponent is re-randomized
    and re-reset on every episode (see RandomizedOpponentWrapper), instead of one
    fixed rule-based instance for the whole training run."""
    def _thunk():
        env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=opponent_factory())
        env = RandomizedOpponentWrapper(env, opponent_factory)
        return FlattenSumoObs(env)
    return _thunk


def main() -> None:
    config, ctx = build_run(
        phase="phase1",
        description="Phase 1: train/freeze Baseline 2 (PPO) vs rule-based Baseline 1.",
    )

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    train_cfg = config["training"]
    env_params = config["env"]
    ppo_cfg = config["ppo"]
    vn_cfg = config["vec_normalize"]
    eval_cfg = config["eval"]

    out_dir = Path(train_cfg["checkpoint_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training env: shaping ON (edge-pushing), physics from arena_config.yaml.
    train_env_cfg = EnvConfig.from_config(
        use_gui=False,
        enable_reward_shaping=bool(env_params["enable_reward_shaping"]),
    )

    n_envs = int(train_cfg["n_envs"])
    randomize_opponent = bool(train_cfg.get("randomize_opponent", True))
    if randomize_opponent:
        print("[phase1] opponent diversity: ON - training against randomized rule-based "
              "opponents (RuleBasedParams.randomized()), re-sampled every episode.")
        opponent_factory = make_randomized_opponent_factory(seed=int(train_cfg["seed"]))
    else:
        print("[phase1] opponent diversity: OFF (training.randomize_opponent=false) - "
              "fixed default RuleBasedParams every episode (the per-episode reset fix "
              "still applies either way - that part isn't the experimental variant).")
        opponent_factory = make_rule_based_policy
    venv = DummyVecEnv([_make_env_fn(train_env_cfg, opponent_factory) for _ in range(n_envs)])
    venv = VecNormalize(
        venv,
        norm_obs=bool(vn_cfg["norm_obs"]),
        norm_reward=bool(vn_cfg["norm_reward"]),
        clip_obs=float(vn_cfg["clip_obs"]),
    )

    model = PPO(
        ppo_cfg["policy"],
        venv,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        vf_coef=float(ppo_cfg["vf_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        seed=int(train_cfg["seed"]),
        device=str(train_cfg["device"]),
        tensorboard_log=str(out_dir / "tb"),
        verbose=1,
    )

    total_steps = int(train_cfg["total_timesteps"])
    print(f"[phase1] training PPO for {total_steps} steps on {n_envs} env(s)...")
    model.learn(total_timesteps=total_steps, progress_bar=True)

    model_path = out_dir / "ppo_baseline2.zip"
    vecnorm_path = out_dir / "vecnormalize.pkl"
    model.save(str(model_path))
    venv.save(str(vecnorm_path))
    print(f"[phase1] saved model -> {model_path}")
    print(f"[phase1] saved vecnormalize -> {vecnorm_path}")

    _sanity_eval(train_env_cfg, model_path, vecnorm_path, int(eval_cfg["n_eval_episodes"]))


def _sanity_eval(train_env_cfg, model_path, vecnorm_path, n_eval) -> None:
    """Sparse-reward win-rate check vs. the rule-based opponent, with an explicit
    assertion that vecnormalize stats loaded (a missing/mismatched stats file
    silently cripples the policy, so fail loudly instead)."""
    from src.baselines.ppo_controller import PPOController

    controller = PPOController.load(model_path, vecnorm_path, deterministic=True)
    assert controller.obs_rms is not None, (
        "VecNormalize stats failed to load. The policy was trained with normalized "
        "observations; without the stats it will underperform. Check that "
        f"{vecnorm_path} exists and matches the model."
    )

    # Sparse eval env: shaping OFF for faithful outcome measurement.
    eval_env_cfg = EnvConfig.from_config(use_gui=False, enable_reward_shaping=False)

    wins = losses = draws = 0
    for _ in range(n_eval):
        env = PyBulletSumoEnv(env_config=eval_env_cfg, opponent_policy=make_rule_based_policy())
        obs, _ = env.reset()
        outcome = "draw"
        for _ in range(env.max_steps):
            action = np.array(controller(obs), dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            outcome = info["outcome"]
            if terminated or truncated:
                break
        wins += outcome == "win"
        losses += outcome == "loss"
        draws += outcome == "draw"
        env.close()

    total = max(1, n_eval)
    print(f"[phase1] PPO vs rule-based (sparse, n={n_eval}): "
          f"win={wins/total:.1%} loss={losses/total:.1%} draw={draws/total:.1%}")


if __name__ == "__main__":
    main()