"""Integration smoke tests for baselines in the live sim (needs PyBullet)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pybullet")

from src.baselines.rule_based_controller import make_rule_based_policy  # noqa: E402
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv  # noqa: E402


def test_rule_based_completes_match():
    """Rule-based agent vs. idle opponent should terminate with a valid outcome."""
    agent = make_rule_based_policy()
    env = PyBulletSumoEnv(EnvConfig(use_gui=False))  # idle opponent by default
    obs, _ = env.reset()
    outcome = "ongoing"
    for _ in range(env.max_steps):
        action = np.array(agent(obs), dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        outcome = info["outcome"]
        if terminated or truncated:
            break
    assert outcome in {"win", "loss", "draw"}
    env.close()


def test_rule_based_vs_rule_based_runs():
    """Both robots rule-based: env must step without error and yield an outcome."""
    env = PyBulletSumoEnv(EnvConfig(use_gui=False), opponent_policy=make_rule_based_policy())
    agent = make_rule_based_policy()
    obs, _ = env.reset()
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, info = env.step(np.array(agent(obs), dtype=np.float32))
        if terminated or truncated:
            break
    assert info["outcome"] in {"win", "loss", "draw"}
    env.close()

def test_env_config_from_arena_config_builds():
    from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv

    env_cfg = EnvConfig.from_config(use_gui=False)
    assert env_cfg.substeps_ok if hasattr(env_cfg, "substeps_ok") else True
    env = PyBulletSumoEnv(env_config=env_cfg)
    obs, _ = env.reset()
    assert set(obs) == {"tof", "ir", "encoder"}
    env.close()