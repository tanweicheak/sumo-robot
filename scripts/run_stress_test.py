"""
scripts.run_stress_test

Phase: Phase 4/5 diagnostic (not one of the five official evaluation conditions)
Purpose: Run a chosen attacker against src.baselines.adversarial_bait_controller's
    BaitController - a counter-puncher persona deliberately outside the
    rush-and-search family every other opponent in this project's training lineage
    descends from (see that module's docstring for the full rationale). A win-rate
    collapse here relative to the official Phase 5a/5b numbers is the clearest signal
    that a model's apparent dominance was tuned to chargers specifically, not general
    tactical competence.

Usage today (works right now, no dependency on Phase 4 completion):
    python -m scripts.run_stress_test --attacker rule_based --episodes 50
    python -m scripts.run_stress_test --attacker ppo --ppo-model-path checkpoints/baseline2_ppo/model.zip \
        --ppo-vecnorm-path checkpoints/baseline2_ppo/vecnormalize.pkl --episodes 50

Usage once Phase 4 training has produced a checkpoint (come back to this later):
    python -m scripts.run_stress_test --attacker benchmark2 --episodes 50 \
        --sglang-agent-url http://localhost:30000
    -> currently raises NotImplementedError with the exact extension point; wiring
       this in requires whatever Phase 5's match-runner ends up using to turn a
       trained Benchmark 1/2 checkpoint into a live opponent_policy-shaped callable
       (OAA+SA+TEA over SGLang/llama.cpp) - that plumbing doesn't exist yet
       (src/evaluation/match_runner.py is still an empty stub).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.baselines.adversarial_bait_controller import make_bait_controller_policy
from src.baselines.rule_based_controller import make_rule_based_policy
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


def _load_attacker_policy(args: argparse.Namespace):
    if args.attacker == "rule_based":
        return make_rule_based_policy()

    if args.attacker == "ppo":
        if not args.ppo_model_path:
            raise SystemExit("--attacker ppo requires --ppo-model-path")
        from src.baselines.ppo_controller import PPOController
        return PPOController.load(args.ppo_model_path, args.ppo_vecnorm_path, deterministic=True)

    if args.attacker in ("benchmark1", "benchmark2"):
        if not args.sglang_agent_url:
            raise SystemExit(f"--attacker {args.attacker} requires --sglang-agent-url")
        from src.evaluation.match_runner import load_benchmark_opponent
        if args.attacker == "benchmark2" and not args.prompt_history_path:
            raise SystemExit(
                "--attacker benchmark2 requires --prompt-history-path (e.g. "
                "checkpoints/benchmark2_full_sbso/prompt_history.jsonl) - without it, "
                "load_benchmark_opponent refuses to silently evaluate benchmark2 as "
                "zero-shot. See src/evaluation/match_runner.py's docstring."
            )
        return load_benchmark_opponent(
            args.attacker,
            sglang_agent_url=args.sglang_agent_url,
            prompt_history_path=args.prompt_history_path,
        )

    raise NotImplementedError(f"Unknown --attacker {args.attacker}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Out-of-family stress test vs. BaitController.")
    p.add_argument("--attacker", choices=["rule_based", "ppo", "benchmark1", "benchmark2"], required=True)
    p.add_argument("--ppo-model-path", default=None)
    p.add_argument("--ppo-vecnorm-path", default=None)
    p.add_argument("--sglang-agent-url", default=None,
                    help="Required for --attacker benchmark1/benchmark2, e.g. http://localhost:30000")
    p.add_argument("--prompt-history-path", default=None,
                    help="Required for --attacker benchmark2 only - path to the trained variant's "
                         "prompt_history.jsonl, e.g. checkpoints/benchmark2_full_sbso/prompt_history.jsonl")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=None, help="Override EnvConfig.max_episode_seconds-derived cap.")
    p.add_argument(
        "--episode-seconds", type=float, default=None,
        help="Override EnvConfig.max_episode_seconds for THIS stress test only (does not touch "
             "arena_config.yaml, so real Phase 4 training/eval timing is untouched). "
             "--max-steps alone cannot extend past the config's own truncation - env.step() "
             "checks max_episode_seconds internally regardless of the harness loop bound, so "
             "use this flag, not a larger --max-steps, to actually give a match more time.",
    )
    p.add_argument("--output-dir", default="results/stress_test_bait")
    p.add_argument("--gui", action="store_true", help="Render with PyBullet GUI (one episode at a time, slower).")
    p.add_argument(
        "--trace", action="store_true",
        help="Print agent/opponent distance and BaitController state every step - useful "
             "for diagnosing a draw-heavy result without watching the GUI live.",
    )
    p.add_argument(
        "--randomize-rule-based", action="store_true",
        help="rule_based attacker only: sample a fresh RuleBasedParams.randomized() instance "
             "per episode instead of fixed defaults. Without this, and with fixed spawn geometry, "
             "every episode is a bit-for-bit identical replay - --episodes 50 tells you about ONE "
             "match, not variance across 50 different ones.",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for --randomize-rule-based")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_file = results_path.open("a")
    print(f"[stress_test] results -> {results_path}")

    if args.randomize_rule_based and args.attacker != "rule_based":
        raise SystemExit("--randomize-rule-based only applies to --attacker rule_based")

    randomized_factory = None
    if args.randomize_rule_based:
        from src.baselines.rule_based_controller import make_randomized_opponent_factory
        randomized_factory = make_randomized_opponent_factory(seed=args.seed)
        print(f"[stress_test] rule_based attacker randomized per episode (seed={args.seed})")

    attacker_policy = _load_attacker_policy(args)

    env_cfg = EnvConfig.from_config(use_gui=args.gui, enable_reward_shaping=False)
    if args.episode_seconds is not None:
        env_cfg.max_episode_seconds = args.episode_seconds
        print(
            f"[stress_test] episode length overridden to {args.episode_seconds}s "
            f"({round(args.episode_seconds / env_cfg.control_dt_s)} steps) for this run only - "
            "arena_config.yaml and real training/eval timing are untouched."
        )
    bait = make_bait_controller_policy()
    env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=bait)

    outcomes = {"win": 0, "loss": 0, "draw": 0}
    for ep in range(args.episodes):
        env.reset()
        if randomized_factory is not None:
            # Fresh RuleBasedParams sample this episode - reset() alone only clears
            # timers/state, it does NOT re-sample params, so reusing one instance
            # across all episodes would still replay identical params every time.
            attacker_policy = randomized_factory()
        elif hasattr(attacker_policy, "reset"):
            attacker_policy.reset()
        env.opponent_policy.reset()

        outcome = "draw"
        reason = None
        min_dist = float("inf")
        for step_i in range(args.max_steps or env.max_steps):
            obs = env.agent_sensors.read()
            left, right = attacker_policy(obs)
            _, _, terminated, truncated, info = env.step(np.array([left, right], dtype=np.float32))

            agent_pos, opp_pos = info["agent_pos"], info["opponent_pos"]
            dist = float(np.hypot(agent_pos[0] - opp_pos[0], agent_pos[1] - opp_pos[1]))
            min_dist = min(min_dist, dist)
            if args.trace and step_i % 10 == 0:
                print(
                    f"  step {step_i:>3}  dist={dist:.3f}m  bait_state={bait.state.value:<11} "
                    f"agent=({agent_pos[0]:+.2f},{agent_pos[1]:+.2f})  opp=({opp_pos[0]:+.2f},{opp_pos[1]:+.2f})"
                )

            if terminated or truncated:
                outcome = info.get("outcome", "draw")
                reason = info.get("agent_out_reason") or info.get("opponent_out_reason")
                break
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        results_file.write(json.dumps({
            "episode": ep, "attacker": args.attacker, "outcome": outcome, "reason": reason,
            "min_dist": round(min_dist, 4), "randomized": args.randomize_rule_based,
            "episode_seconds": args.episode_seconds,
        }) + "\n")
        results_file.flush()
        reason_note = f"  reason={reason}" if outcome != "draw" and reason else ""
        print(f"[stress_test] episode {ep + 1}/{args.episodes}: {outcome}{reason_note}  (min_dist={min_dist:.3f}m)")
        if outcome == "draw" and min_dist > 0.5:
            print(
                "    NOTE: min_dist stayed large - the two never actually closed to contact "
                "range this episode (evasion/positioning issue, not a stuck-in-contact deadlock)."
            )
        elif outcome == "draw":
            print(
                "    NOTE: min_dist got close but still drew - likely oscillating near contact "
                "without either crossing the boundary (matches a 'stuck pushing at center' pattern)."
            )

    env.close()
    results_file.close()

    total = sum(outcomes.values())
    win_rate = outcomes["win"] / total if total else 0.0
    print()
    print(f"Attacker: {args.attacker}   Opponent: BaitController (out-of-family)")
    print(f"Episodes: {total}   Win: {outcomes['win']}   Loss: {outcomes['loss']}   Draw: {outcomes['draw']}")
    print(f"Win rate: {win_rate:.1%}")
    print()
    print("Compare this against the same attacker's win rate in Phase 5a/5b's official ")
    print("conditions. A large drop here relative to those numbers is evidence of a ")
    print("narrow-opponent-pool artifact, not evidence the attacker is actually weak.")


if __name__ == "__main__":
    main()