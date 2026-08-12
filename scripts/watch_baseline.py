"""
scripts.watch_baseline

Phase: Phase 1
Purpose: Visual sanity viewer for baselines. Opens the PyBullet GUI and runs matches
    so you can watch a controller behave before committing training time. Supports
    rule-based (Baseline 1) for both agent and opponent, and a trained PPO policy
    (Baseline 2) once its checkpoint exists.

Examples:
    # Rule-based agent vs. idle opponent
    python -m scripts.watch_baseline --agent rule_based --opponent idle

    # Rule-based vs. rule-based
    python -m scripts.watch_baseline --agent rule_based --opponent rule_based

    # Trained PPO agent vs. rule-based opponent
    python -m scripts.watch_baseline --agent ppo --opponent rule_based \
        --ppo-model checkpoints/baseline2_ppo/ppo_baseline2.zip \
        --ppo-vecnorm checkpoints/baseline2_ppo/vecnormalize.pkl
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from src.baselines.rule_based_controller import make_rule_based_policy
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


def _idle(_obs):
    return 0.0, 0.0


def _build_controller(kind: str, ppo_model: str | None, ppo_vecnorm: str | None):
    """Return a callable obs_dict -> (left_pwm, right_pwm)."""
    if kind == "rule_based":
        return make_rule_based_policy()
    if kind == "idle":
        return _idle
    if kind == "ppo":
        if not ppo_model:
            raise SystemExit("--ppo-model is required when using a ppo controller.")
        from src.baselines.ppo_controller import PPOController

        return PPOController.load(ppo_model, ppo_vecnorm, deterministic=True)
    raise SystemExit(f"Unknown controller kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a baseline controller in the PyBullet GUI.")
    parser.add_argument("--agent", default="rule_based", choices=["rule_based", "ppo", "idle"])
    parser.add_argument("--opponent", default="idle", choices=["rule_based", "ppo", "idle"])
    parser.add_argument("--ppo-model", default=None)
    parser.add_argument("--ppo-vecnorm", default=None)
    parser.add_argument("--opp-ppo-model", default=None, help="PPO model for the opponent, if any.")
    parser.add_argument("--opp-ppo-vecnorm", default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--realtime", action="store_true", default=True,
                        help="Sleep to approximate real-time playback (default on).")
    parser.add_argument("--fast", dest="realtime", action="store_false",
                        help="Disable real-time pacing; run as fast as possible.")
    args = parser.parse_args()

    agent = _build_controller(args.agent, args.ppo_model, args.ppo_vecnorm)
    opponent = _build_controller(args.opponent, args.opp_ppo_model, args.opp_ppo_vecnorm)

    # Evaluation-faithful: shaping OFF, GUI ON. Physics from arena_config.yaml.
    env_cfg = EnvConfig.from_config(use_gui=True, enable_reward_shaping=False)
    env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=opponent)

    tally = {"win": 0, "loss": 0, "draw": 0}
    for ep in range(args.episodes):
        # Rule-based controllers are stateful; reset per episode.
        if hasattr(agent, "reset"):
            agent.reset()
        if hasattr(opponent, "reset"):
            opponent.reset()

        obs, _ = env.reset()
        outcome = "draw"
        for step in range(env.max_steps):
            action = np.array(agent(obs), dtype=np.float32)

            if step < 40:
                st = getattr(agent, "state", "?")
                ax = info["agent_pos"][0] if step > 0 else -0.6
                print(f"step={step:3d} state={st!s:10s} "
                      f"min_tof={float(np.min(obs['tof'])):.2f} "
                      f"action=({action[0]:+.2f},{action[1]:+.2f})")

            obs, _, terminated, truncated, info = env.step(action)
            if step < 60:
                ax = info["agent_pos"][0]; ay = info["agent_pos"][1]
                ox = info["opponent_pos"][0]; oy = info["opponent_pos"][1]
                gap = ((ox-ax)**2 + (oy-ay)**2) ** 0.5
                print(f"step={step:3d} agent=({ax:+.2f},{ay:+.2f}) opp=({ox:+.2f},{oy:+.2f}) gap={gap:.3f}")
                
            outcome = info["outcome"]
            if args.realtime:
                time.sleep(env_cfg.control_dt_s)
            if terminated or truncated:
                break
        tally[outcome] += 1
        print(f"[watch] episode {ep + 1}/{args.episodes}: {outcome} "
              f"(steps={info['steps']})")

    print(f"[watch] agent record vs {args.opponent}: "
          f"win={tally['win']} loss={tally['loss']} draw={tally['draw']}")
    env.close()


if __name__ == "__main__":
    main()