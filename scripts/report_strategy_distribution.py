"""
scripts.report_strategy_distribution

Phase: Phase 4/5 diagnostic
Purpose: Thin CLI wrapper around src.evaluation.strategy_distribution for a completed
    (or in-progress) run's training_pairs.jsonl.

Usage:
    python -m scripts.report_strategy_distribution --training-pairs checkpoints/benchmark2_full_sbso/training_pairs.jsonl
    python -m scripts.report_strategy_distribution --training-pairs <path> --buckets 10 --json-out results/strategy_dist.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.strategy_distribution import (
    compute_distribution, compute_distribution_by_prompt_version,
    format_report, format_prompt_version_report,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Report macro-strategy usage distribution for a completed Phase 4 run.")
    p.add_argument("--training-pairs", required=True, help="Path to training_pairs.jsonl")
    p.add_argument("--buckets", type=int, default=5, help="Number of timeline buckets to show progression across")
    p.add_argument("--json-out", default=None, help="Optional path to also write the full distribution as JSON")
    p.add_argument(
        "--prompt-history", default=None,
        help="Path to prompt_history.jsonl - if given, also prints strategy usage broken "
             "down by DSPy prompt version instead of only by episode-timeline bucket.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    dist = compute_distribution(args.training_pairs, n_buckets=args.buckets)
    print(format_report(dist))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dist, indent=2))
        print(f"\n[report_strategy_distribution] wrote full distribution -> {out_path}")

    if args.prompt_history:
        print()
        by_version = compute_distribution_by_prompt_version(args.training_pairs, args.prompt_history)
        print(format_prompt_version_report(by_version))


if __name__ == "__main__":
    main()
