import numpy as np
import pytest

pytest.importorskip("pybullet")
from src.simulation.sumo_env import PyBulletSumoEnv, EnvConfig  # noqa: E402


def test_reset_returns_valid_obs():
    env = PyBulletSumoEnv(EnvConfig(use_gui=False))
    obs, info = env.reset()
    assert set(obs) == {"tof", "ir", "encoder"}
    assert obs["tof"].shape == (7,)
    assert np.all(obs["tof"] >= 0.0)
    env.close()


def test_forward_drive_terminates_with_outcome():
    env = PyBulletSumoEnv(EnvConfig(use_gui=False))
    env.reset()
    outcome = "ongoing"
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0]))
        outcome = info["outcome"]
        if terminated or truncated:
            break
    assert outcome in {"win", "loss", "draw"}
    env.close()