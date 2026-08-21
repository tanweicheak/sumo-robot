"""
scripts.run_mirror_match

Phase: pre-Phase-5 baseline sanity gate
Purpose: The most basic structural check available - pit a controller against an
    identical copy of itself and confirm win rate lands near 50%. Perfect symmetry
    means neither side has any structural advantage; a skewed result means something
    asymmetric is leaking in (spawn order, a coordinate-frame bug, a non-symmetric
    outcome check) and every other win-rate number in the project is suspect until
    it's found. Distinct from scripts/run_stress_test.py, which tests generalization
    against a deliberately DIFFERENT opponent archetype - this tests structural
    balance against an IDENTICAL one.

Usage:
    python -m scripts.run_mirror_match --attacker rule_based --episodes 200
    python -m scripts.run_mirror_match --attacker rule_based --episodes 200 --randomize
        (samples a fresh RuleBasedParams.randomized() instance per side per episode -
        see run_stress_test.py's --randomize-rule-based for why this matters: with
        both sides at fixed default params AND fixed spawn geometry, every episode
        is a bit-for-bit identical replay, which tells you about ONE match, not
        about the controller's balance across the space of matches it could play.)

    PPO has no meaningful "mirror" in the same sense (both sides would run the
    identical deterministic policy - useful only as a pure environment-symmetry
    check, not a controller-competence check) - included for completeness via
    --attacker ppo, but the interesting result here is rule_based.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.baselines.rule_based_controller import RuleBasedParams, make_randomized_opponent_factory, make_rule_based_policy
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


def _load_pair(args: argparse.Namespace):
    if args.attacker == "rule_based":
        if args.randomize:
            factory = make_randomized_opponent_factory(seed=args.seed)
            return factory, factory  # each call produces an independent randomized instance
        return make_rule_based_policy, make_rule_based_policy

    if args.attacker == "ppo":
        if not args.ppo_model_path:
            raise SystemExit("--attacker ppo requires --ppo-model-path")
        from src.baselines.ppo_controller import PPOController

        def _factory():
            return PPOController.load(args.ppo_model_path, args.ppo_vecnorm_path, deterministic=True)

        return _factory, _factory

    raise SystemExit(f"--attacker {args.attacker} not supported for a mirror match")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mirror-match sanity check: a controller against an identical copy of itself.")
    p.add_argument("--attacker", choices=["rule_based", "ppo"], required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--randomize", action="store_true", help="rule_based only: fresh RuleBasedParams.randomized() per side per episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ppo-model-path", default=None)
    p.add_argument("--ppo-vecnorm-path", default=None)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--output-dir", default="results/mirror_match")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    agent_factory, opp_factory = _load_pair(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_file = results_path.open("a")
    print(f"[mirror_match] results -> {results_path}")

    env_cfg = EnvConfig.from_config(use_gui=args.gui, enable_reward_shaping=False)
    env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=opp_factory())

    outcomes = {"win": 0, "loss": 0, "draw": 0}
    for ep in range(args.episodes):
        env.opponent_policy = opp_factory()
        env.reset()
        agent_policy = agent_factory()
        if hasattr(agent_policy, "reset"):
            agent_policy.reset()

        outcome = "draw"
        for _ in range(env.max_steps):
            obs = env.agent_sensors.read()
            left, right = agent_policy(obs)
            _, _, terminated, truncated, info = env.step(np.array([left, right], dtype=np.float32))
            if terminated or truncated:
                outcome = info.get("outcome", "draw")
                break
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        results_file.write(json.dumps({
            "episode": ep, "attacker": args.attacker, "outcome": outcome, "randomized": args.randomize,
        }) + "\n")
        results_file.flush()

    env.close()
    results_file.close()

    total = sum(outcomes.values())
    win_rate = outcomes["win"] / total if total else 0.0
    print(f"Mirror match: {args.attacker} vs. itself" + (" (randomized)" if args.randomize else " (fixed default params)"))
    print(f"Episodes: {total}   Win: {outcomes['win']}   Loss: {outcomes['loss']}   Draw: {outcomes['draw']}")
    print(f"Win rate: {win_rate:.1%}  (expect close to 50% - anything far off suggests a structural asymmetry)")
    if not args.randomize and args.attacker == "rule_based":
        print(
            "\nNote: without --randomize, both sides use identical fixed default params "
            "and spawn geometry is fixed too - every episode is the same match replayed, "
            "so this result reflects ONE scenario's symmetry, not variance across many. "
            "Rerun with --randomize for a statistically meaningful check."
        )


if __name__ == "__main__":
    main()