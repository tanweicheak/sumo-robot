"""
src.baselines.ppo_controller

Phase: Phase 1
Purpose: Baseline 2 - on-policy PPO controller (Stable-Baselines3), the standard
    robotic-control RL baseline. No reusable robot-sumo RL environment exists, so
    PPO is trained directly on PyBulletSumoEnv against a frozen rule-based (Baseline 1)
    opponent. This module provides:
      - FlattenSumoObs: dict observation -> flat Box vector for SB3 MlpPolicy
      - PPOController: wraps a trained policy into the env opponent_policy signature
    Training itself lives in scripts/run_phase1_baselines.py.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Raw sensor layout (raw sensors only, per Phase 1 decision):
#   tof: 7 rays, ir: 2 probes, encoder: 4 values  ->  13 dims
_TOF_DIM, _IR_DIM, _ENC_DIM = 7, 2, 4
FLAT_OBS_DIM = _TOF_DIM + _IR_DIM + _ENC_DIM


def flatten_obs(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the three raw sensor channels into one float32 vector."""
    return np.concatenate(
        [
            np.asarray(obs["tof"], dtype=np.float32).reshape(-1),
            np.asarray(obs["ir"], dtype=np.float32).reshape(-1),
            np.asarray(obs["encoder"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


class FlattenSumoObs(gym.ObservationWrapper):
    """Flattens PyBulletSumoEnv's Dict observation into a single Box vector so SB3's
    MlpPolicy can consume it. Encoder values are unbounded, so the flat space uses
    generous finite bounds; VecNormalize (applied in the training script) handles the
    actual state-vector normalization required by the report."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        low = np.full(FLAT_OBS_DIM, -1e3, dtype=np.float32)
        high = np.full(FLAT_OBS_DIM, 1e3, dtype=np.float32)
        # ToF and IR are physically bounded; tighten those for a cleaner space.
        low[: _TOF_DIM + _IR_DIM] = 0.0
        high[:_TOF_DIM] = 2.0          # >= tof_max_range_m
        high[_TOF_DIM : _TOF_DIM + _IR_DIM] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        return flatten_obs(observation)


class PPOController:
    """Wraps a trained PPO model into the env opponent_policy signature
    (obs_dict -> (left_pwm, right_pwm)). Deterministic by default for evaluation."""

    def __init__(
        self,
        model,
        deterministic: bool = True,
        obs_rms=None,
        clip_obs: float = 10.0,
        epsilon: float = 1e-8,
    ) -> None:
        self.model = model
        self.deterministic = deterministic
        self.obs_rms = obs_rms          # observation running mean/std from VecNormalize
        self.clip_obs = clip_obs
        self.epsilon = epsilon

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        vec_normalize_path: str | Path | None = None,
        deterministic: bool = True,
    ) -> "PPOController":
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path), device="cpu")

        obs_rms = None
        clip_obs = 10.0
        epsilon = 1e-8
        if vec_normalize_path is not None and Path(vec_normalize_path).exists():
            import pickle

            # Load the VecNormalize object directly from its pickle to pull out the
            # observation running-mean/std stats, without attaching a live venv
            # (VecNormalize.load(venv=None) is not supported for inference-only use).
            with open(vec_normalize_path, "rb") as f:
                vec_normalize = pickle.load(f)
            obs_rms = vec_normalize.obs_rms
            clip_obs = getattr(vec_normalize, "clip_obs", 10.0)
            epsilon = getattr(vec_normalize, "epsilon", 1e-8)

        return cls(
            model=model,
            deterministic=deterministic,
            obs_rms=obs_rms,
            clip_obs=clip_obs,
            epsilon=epsilon,
        )

    def __call__(self, obs: dict[str, np.ndarray]) -> tuple[float, float]:
        flat = flatten_obs(obs)
        if self.obs_rms is not None:
            # Apply the same normalization VecNormalize used during training.
            flat = np.clip(
                (flat - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            ).astype(np.float32)
        action, _ = self.model.predict(flat, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        return float(np.clip(action[0], -1.0, 1.0)), float(np.clip(action[1], -1.0, 1.0))