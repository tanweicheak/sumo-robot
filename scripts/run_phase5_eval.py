"""
scripts.run_phase5_eval

Purpose: the real Phase 5 evaluation harness - runs N matches for any
attacker/opponent pairing from the finalized 7-pairing plan, with genuine
per-episode randomization (not the deterministic-replay bug found in
run_stress_test.py's BaitController runs), structured JSONL + summary.json
output, and live terminal progress.

Reuses, does not reimplement:
- src.evaluation.match_runner.load_benchmark_opponent (benchmark1/benchmark2)
- src.baselines.rule_based_controller.make_randomized_opponent_factory (baseline1)
- src.baselines.ppo_controller.PPOController (baseline2)
- NEW: a small MonolithicReasoningAgent wrapper (baseline3) - the only genuinely
  new opponent-loading logic in this file, everything else is composition of
  already-verified pieces.

Usage:
    python -m scripts.run_phase5_eval \\
        --attacker benchmark2 --opponent baseline1 \\
        --sglang-agent-url http://localhost:30000 \\
        --agent-model-path /workspace/export/benchmark2_full_sbso/merged_fp16 \\
        --prompt-history-path checkpoints/benchmark2_full_sbso/progress.json \\
        --episodes 50 --episode-seconds 27 --seed 0

Output:
    results/phase5_eval/<attacker>_vs_<opponent>/results.jsonl  (one line per match)
    results/phase5_eval/<attacker>_vs_<opponent>/summary.json   (aggregated)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.agents.perception_agent import PerceptionAgent
from src.agents.actuator_bridge import ActuatorBridge
from src.agents.monolithic_reasoning_agent import MonolithicReasoningAgent
from src.baselines.ppo_controller import PPOController
from src.baselines.rule_based_controller import make_randomized_opponent_factory
from src.evaluation.match_runner import load_benchmark_opponent
from src.simulation.sumo_env import EnvConfig, PyBulletSumoEnv


class MonolithicPolicyWrapper:
    """Wraps MonolithicReasoningAgent (baseline3) into the OpponentPolicy-shaped
    callable (opp_obs) -> (left_pwm, right_pwm) - same contract SLMPolicyOpponent
    already satisfies (match_runner.py), so this can be used as env.opponent_policy
    exactly the same way. NOT MCTS/multi-agent - a single call per decision, per
    MonolithicReasoningAgent's own real, confirmed interface
    (decide(perception) -> TacticalCommand, no OAA/SA/TEA chain)."""

    def __init__(self, mra: MonolithicReasoningAgent, perception_agent: PerceptionAgent, bridge: ActuatorBridge):
        self.mra = mra
        self.perception_agent = perception_agent
        self.bridge = bridge

    def reset(self) -> None:
        self.perception_agent.reset()

    def __call__(self, opp_obs: dict) -> tuple[float, float]:
        perception = self.perception_agent.perceive(opp_obs["tof"], opp_obs["ir"], opp_obs["encoder"])
        command = self.mra.decide(perception)
        return self.bridge.to_pwm(command)


def _load_opponent_or_attacker(kind: str, args, client_cache: dict):
    """Loads any of the 5 real opponent/attacker kinds, reusing already-verified
    loading logic. Returns an OpponentPolicy-shaped callable in every case, so the
    SAME object works whether used as env.opponent_policy (opponent role) or driven
    directly step-by-step (attacker role) - both roles need identical
    (obs) -> (left_pwm, right_pwm) behavior; the harness's main loop decides which
    role calls it, not this function.
    """
    if kind == "baseline1":
        if "rb_factory" not in client_cache:
            client_cache["rb_factory"] = make_randomized_opponent_factory(seed=args.seed)
        return client_cache["rb_factory"]()

    if kind == "baseline2":
        if "ppo" not in client_cache:
            client_cache["ppo"] = PPOController.load(
                "checkpoints/baseline2_ppo/ppo_baseline2.zip",
                "checkpoints/baseline2_ppo/vecnormalize.pkl",
            )
        return client_cache["ppo"]

    if kind == "baseline3":
        if "mra_client" not in client_cache:
            from src.inference.sglang_server import SGLangSLMClient
            client_cache["mra_client"] = SGLangSLMClient(
                server_url=args.sglang_agent_url, model_path=args.agent_model_path,
            )
        mra = MonolithicReasoningAgent(client=client_cache["mra_client"])
        return MonolithicPolicyWrapper(mra, PerceptionAgent(), ActuatorBridge())

    if kind in ("benchmark1", "benchmark2"):
        prompt_history_path = args.prompt_history_path if kind == "benchmark2" else None
        return load_benchmark_opponent(
            kind, sglang_agent_url=args.sglang_agent_url,
            agent_model_path=args.agent_model_path, prompt_history_path=prompt_history_path,
        )

    raise ValueError(f"Unknown kind: {kind}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5 evaluation: run N matches for one attacker/opponent pairing.")
    p.add_argument("--attacker", required=True, choices=["baseline1", "baseline2", "baseline3", "benchmark1", "benchmark2"])
    p.add_argument("--opponent", required=True, choices=["baseline1", "baseline2", "baseline3", "benchmark1", "benchmark2"])
    p.add_argument("--sglang-agent-url", default=None, help="Required if attacker/opponent uses an SLM (all except baseline1/baseline2)")
    p.add_argument("--agent-model-path", default=None, help="Required alongside --sglang-agent-url")
    p.add_argument("--prompt-history-path", default=None, help="Required if attacker or opponent is benchmark2 - path to that checkpoint's progress.json")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--episode-seconds", type=float, default=27.0, help="max_episode_seconds override for THIS eval only")
    p.add_argument("--max-decisions", type=int, default=90)
    p.add_argument("--decision-cycles", type=int, default=6)
    p.add_argument("--seed", type=int, default=0, help="Seed for genuine per-episode randomization (opponent params + spawn)")
    p.add_argument("--output-dir", default=None, help="Default: results/phase5_eval/<attacker>_vs_<opponent>/")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    needs_sglang = args.attacker not in ("baseline1", "baseline2") or args.opponent not in ("baseline1", "baseline2")
    if needs_sglang and not (args.sglang_agent_url and args.agent_model_path):
        raise SystemExit("--sglang-agent-url and --agent-model-path are required when either "
                          "attacker or opponent is an SLM-based policy (baseline3/benchmark1/benchmark2).")

    out_dir = Path(args.output_dir or f"results/phase5_eval/{args.attacker}_vs_{args.opponent}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_file = results_path.open("a")

    print(f"[phase5_eval] {args.attacker} (attacker) vs {args.opponent} (opponent) - {args.episodes} episodes")
    print(f"[phase5_eval] results -> {results_path}")

    client_cache: dict = {}

    env_cfg = EnvConfig.from_config(use_gui=False, enable_reward_shaping=False)
    env_cfg.max_episode_seconds = args.episode_seconds   # EnvConfig.from_config has no
                                                            # override kwarg for this -
                                                            # confirmed from real source,
                                                            # override the dataclass field
                                                            # directly after construction.

    outcomes = []
    rng = random.Random(args.seed)

    for ep in range(args.episodes):
        ep_seed = rng.randint(0, 2**31 - 1)   # genuine, distinct seed per episode -
                                                # the real fix for the deterministic-
                                                # replay bug found in the BaitController
                                                # stress test. Both attacker (if
                                                # baseline1) and opponent (if
                                                # baseline1) get FRESH randomized
                                                # instances built with this seed, not
                                                # a shared cached factory instance.
        client_cache.pop("rb_factory", None)   # force a fresh factory -> fresh
                                                 # RuleBasedParams draw this episode
        episode_rng_args = argparse.Namespace(**vars(args))
        episode_rng_args.seed = ep_seed

        attacker_policy = _load_opponent_or_attacker(args.attacker, episode_rng_args, client_cache)
        opponent_policy = _load_opponent_or_attacker(args.opponent, episode_rng_args, client_cache)

        env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=opponent_policy)
        env.reset(seed=ep_seed)
        if hasattr(attacker_policy, "reset"):
            attacker_policy.reset()
        if hasattr(opponent_policy, "reset"):
            opponent_policy.reset()

        perception_agent = PerceptionAgent()
        bridge = ActuatorBridge()

        decisions = 0
        outcome = "draw"
        terminated_genuinely = False   # NEW: distinguishes real termination (push-out/
                                         # fall/capsize) from a truncation-default draw -
                                         # both currently collapse into outcome="draw"
                                         # without this, which was exactly the confound
                                         # behind this study's headline finding (Section 9).
        strategies_chosen = []          # NEW: per-decision macro-strategy, when the
                                         # attacker is an SLM-driven policy (benchmark1/
                                         # benchmark2) - baseline1/baseline2/baseline3
                                         # have no MacroStrategy concept, left empty for
                                         # those.
        t0 = time.perf_counter()
        decision_latencies = []

        while decisions < args.max_decisions:
            dec_t0 = time.perf_counter()

            obs = env.agent_sensors.read()
            left, right = attacker_policy(obs)

            # NEW: capture the attacker's last chosen macro-strategy, if it exposes
            # one. SLMPolicyOpponent (match_runner.py) tracks internal state per call
            # but doesn't currently expose the chosen strategy as a public attribute -
            # this reads it defensively via getattr, logging None rather than crashing
            # if that attribute doesn't exist on this particular policy object.
            last_strategy = getattr(attacker_policy, "_last_strategy", None)
            if last_strategy is not None:
                strategies_chosen.append(
                    last_strategy.value if hasattr(last_strategy, "value") else str(last_strategy)
                )

            decision_latencies.append(time.perf_counter() - dec_t0)

            for _ in range(args.decision_cycles):
                _, _, terminated, truncated, info = env.step(np.array([left, right], dtype=np.float32))
                if terminated or truncated:
                    outcome = info.get("outcome", "draw")
                    terminated_genuinely = bool(terminated)   # NEW: real termination
                                                                 # (agent_out or opp_out)
                                                                 # vs truncation-only
                    break
            decisions += 1
            if terminated or truncated:
                break

        match_seconds = time.perf_counter() - t0
        mean_latency = sum(decision_latencies) / len(decision_latencies) if decision_latencies else None

        # NEW: capture the opponent's actual sampled RuleBasedParams, if it's a
        # baseline1 instance - directly tests the "does the decision-count clustering
        # (e.g. 27/50 episodes landing at exactly 76) correlate with a specific
        # narrow band of sampled parameters" hypothesis, rather than leaving it
        # unexplained.
        opponent_params = None
        if hasattr(opponent_policy, "params"):
            from dataclasses import asdict
            try:
                opponent_params = asdict(opponent_policy.params)
            except Exception:
                opponent_params = None

        record = {
            "attacker": args.attacker, "opponent": args.opponent,
            "episode": ep, "outcome": outcome, "decisions": decisions,
            "match_seconds": round(match_seconds, 3),
            "mean_decision_latency_s": round(mean_latency, 4) if mean_latency else None,
            "seed": ep_seed,
            "terminated_genuinely": terminated_genuinely,
            "strategies_chosen": strategies_chosen,
            "opponent_params": opponent_params,
        }
        results_file.write(json.dumps(record) + "\n")
        results_file.flush()
        outcomes.append(outcome)

        env.close()
        print(f"[phase5_eval] episode {ep+1}/{args.episodes}: {outcome}   "
              f"decisions={decisions}   mean_latency={mean_latency:.3f}s" if mean_latency else
              f"[phase5_eval] episode {ep+1}/{args.episodes}: {outcome}   decisions={decisions}")

    results_file.close()

    from collections import Counter
    counts = Counter(outcomes)
    total = len(outcomes)
    wins = counts.get("win", 0)
    summary = {
        "attacker": args.attacker, "opponent": args.opponent, "episodes": total,
        "wins": wins, "losses": counts.get("loss", 0), "draws": counts.get("draw", 0),
        "win_rate": round(wins / total, 4) if total else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[phase5_eval] SUMMARY: {summary}")


if __name__ == "__main__":
    main()
