"""
src.evaluation.strategy_distribution

Phase: Phase 4/5 diagnostic (not part of the five official evaluation conditions)
Purpose: Count how often each MacroStrategy (charge, flank, retreat, hold, evade_edge)
    actually appears in a completed training run's training_pairs.jsonl, both overall
    and bucketed across the run's timeline. Read this alongside win rate, never instead
    of it: a 100% win rate produced almost entirely by "charge" is evidence the training
    opponent pool never punished pure aggression, not evidence of tactical breadth,
    regardless of how good the win rate looks.

Input: checkpoints/<run_id>/training_pairs.jsonl, written incrementally by
    run_phase4_pilot.py / run_phase4_stage3_local.py's on_episode_end hook. One JSON
    object per line: {"episode": int, "lssd_text": str, "strategy": str}.

Usage (once a real run has produced a training_pairs.jsonl):
    from src.evaluation.strategy_distribution import compute_distribution, format_report
    dist = compute_distribution("checkpoints/benchmark2_full_sbso/training_pairs.jsonl", n_buckets=5)
    print(format_report(dist))

Or from the CLI wrapper: scripts/report_strategy_distribution.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.agents.schemas import MacroStrategy

ALL_STRATEGIES = [s.value for s in MacroStrategy]


def _read_pairs(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This file is written incrementally by "
            "run_phase4_pilot.py / run_phase4_stage3_local.py while training runs - "
            "there is nothing to analyze until a real run has produced it."
        )
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_no} is not valid JSON (crash-safe incremental writes "
                    "can leave a torn final line if a run was interrupted mid-write - "
                    "check whether this is the last line in the file)."
                ) from e
    if not rows:
        raise ValueError(f"{path} exists but contains zero decision records.")
    return rows


def _bucket_index(episode: int, max_episode: int, n_buckets: int) -> int:
    if max_episode <= 0 or n_buckets <= 1:
        return 0
    frac = episode / max_episode
    idx = int(frac * n_buckets)
    return min(idx, n_buckets - 1)


def compute_distribution(training_pairs_path: str | Path, n_buckets: int = 5) -> dict[str, Any]:
    """Returns overall counts/percentages plus an n_buckets-way breakdown across the
    run's episode range, so a strategy that only shows up early (or only after some
    DSPy recompile) doesn't get averaged away in an aggregate count."""
    rows = _read_pairs(training_pairs_path)

    episodes = [int(r["episode"]) for r in rows]
    min_ep, max_ep = min(episodes), max(episodes)

    overall = Counter(r["strategy"] for r in rows)
    total = sum(overall.values())

    buckets: list[Counter] = [Counter() for _ in range(n_buckets)]
    bucket_ranges: list[tuple[int, int]] = []
    span = max(1, max_ep - min_ep)
    for i in range(n_buckets):
        lo = min_ep + round(span * i / n_buckets)
        hi = min_ep + round(span * (i + 1) / n_buckets)
        bucket_ranges.append((lo, hi))

    for r in rows:
        idx = _bucket_index(int(r["episode"]) - min_ep, span, n_buckets)
        buckets[idx][r["strategy"]] += 1

    unused = [s for s in ALL_STRATEGIES if overall.get(s, 0) == 0]

    return {
        "source": str(training_pairs_path),
        "total_decisions": total,
        "episode_range": [min_ep, max_ep],
        "overall_counts": {s: overall.get(s, 0) for s in ALL_STRATEGIES},
        "overall_pct": {s: round(100.0 * overall.get(s, 0) / total, 1) for s in ALL_STRATEGIES} if total else {},
        "unused_strategies": unused,
        "buckets": [
            {
                "episode_range": list(bucket_ranges[i]),
                "counts": {s: buckets[i].get(s, 0) for s in ALL_STRATEGIES},
                "pct": (
                    {s: round(100.0 * buckets[i].get(s, 0) / sum(buckets[i].values()), 1) for s in ALL_STRATEGIES}
                    if sum(buckets[i].values()) else {}
                ),
            }
            for i in range(n_buckets)
        ],
    }


def format_report(dist: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Source: {dist['source']}")
    lines.append(f"Decisions: {dist['total_decisions']}   Episodes: {dist['episode_range'][0]}-{dist['episode_range'][1]}")
    lines.append("")
    lines.append("Overall usage:")
    for s in ALL_STRATEGIES:
        pct = dist["overall_pct"].get(s, 0.0)
        count = dist["overall_counts"].get(s, 0)
        bar = "#" * int(pct // 2)
        lines.append(f"  {s:<12} {pct:5.1f}%  ({count:>6})  {bar}")

    if dist["unused_strategies"]:
        lines.append("")
        lines.append(f"WARNING - never used even once: {', '.join(dist['unused_strategies'])}")

    lines.append("")
    lines.append("Usage over training timeline (does the distribution shift or collapse?):")
    header = "  bucket-episodes  " + "  ".join(f"{s[:8]:>8}" for s in ALL_STRATEGIES)
    lines.append(header)
    for b in dist["buckets"]:
        lo, hi = b["episode_range"]
        row = f"  {lo:>6}-{hi:<6}   " + "  ".join(f"{b['pct'].get(s, 0.0):7.1f}%" for s in ALL_STRATEGIES)
        lines.append(row)

    dominant = max(dist["overall_pct"], key=dist["overall_pct"].get) if dist["overall_pct"] else None
    if dominant and dist["overall_pct"][dominant] >= 70.0:
        lines.append("")
        lines.append(
            f"FLAG: '{dominant}' alone accounts for {dist['overall_pct'][dominant]:.1f}% of all decisions. "
            "A near-100% win rate built mostly on one strategy is a narrow-opponent-pool "
            "artifact until shown otherwise - see the out-of-family stress test."
        )
    return "\n".join(lines)


def _read_prompt_history(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_version_at(episode: int, history: list[dict]) -> int:
    """Prompt version active AT this episode - version 0 (base prompt) until the
    first recompile's episode is reached, then increments at each recompile's
    recorded episode."""
    version = 0
    for event in sorted(history, key=lambda e: e["episode"]):
        if episode >= event["episode"]:
            version = event["prompt_version"]
        else:
            break
    return version


def compute_distribution_by_prompt_version(
    training_pairs_path: str | Path, prompt_history_path: str | Path
) -> dict[str, Any]:
    """Breaks strategy usage down by which prompt version was active, instead of
    arbitrary episode-timeline buckets - directly answers 'did DSPy recompilation
    actually change what the agent does', which n_buckets time-slicing can only
    answer approximately (recompiles don't land on round episode-count fractions)."""
    rows = _read_pairs(training_pairs_path)
    history = _read_prompt_history(prompt_history_path)

    if not history:
        return {
            "source": str(training_pairs_path), "prompt_history_source": str(prompt_history_path),
            "note": "no prompt_history.jsonl entries found - either DSPy never recompiled, "
                    "or this run predates recompile_history tracking.",
            "by_version": [],
        }

    by_version: dict[int, Counter] = {}
    for r in rows:
        v = _prompt_version_at(int(r["episode"]), history)
        by_version.setdefault(v, Counter())[r["strategy"]] += 1

    version_meta = {0: {"episode": 0, "trigger_reason": "base_prompt"}}
    for event in history:
        version_meta[event["prompt_version"]] = event

    results = []
    for v in sorted(by_version):
        counts = by_version[v]
        total = sum(counts.values())
        results.append({
            "prompt_version": v,
            "activated_at_episode": version_meta.get(v, {}).get("episode"),
            "trigger_reason": version_meta.get(v, {}).get("trigger_reason"),
            "decisions": total,
            "pct": {s: round(100.0 * counts.get(s, 0) / total, 1) for s in ALL_STRATEGIES} if total else {},
        })

    return {
        "source": str(training_pairs_path), "prompt_history_source": str(prompt_history_path),
        "by_version": results,
    }


def format_prompt_version_report(dist: dict[str, Any]) -> str:
    if dist.get("note"):
        return dist["note"]
    lines = [f"Strategy usage by DSPy prompt version ({dist['prompt_history_source']}):", ""]
    header = "  v  activated@ep  trigger       decisions  " + "  ".join(f"{s[:8]:>8}" for s in ALL_STRATEGIES)
    lines.append(header)
    for row in dist["by_version"]:
        pct_str = "  ".join(f"{row['pct'].get(s, 0.0):7.1f}%" for s in ALL_STRATEGIES)
        lines.append(
            f"  {row['prompt_version']:<3}{str(row['activated_at_episode']):>10}  "
            f"{str(row['trigger_reason']):<12}  {row['decisions']:>9}  {pct_str}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.strategy_distribution <training_pairs.jsonl> [n_buckets]")
        raise SystemExit(1)
    n_buckets = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(format_report(compute_distribution(sys.argv[1], n_buckets=n_buckets)))
