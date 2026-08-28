"""
src.statistics.win_rate

Phase: Phase 5
Purpose: Win rate with a 95% confidence interval, computed from a pairing's
    results.jsonl (scripts/run_phase5_eval.py's output format). Replaces the
    Phase 0 stub - real implementation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def win_rate_from_file(results_path: str | Path) -> dict:
    """Reads a results.jsonl file (one JSON object per match, with an "outcome"
    field of "win"/"loss"/"draw") and returns win/loss/draw counts, win rate, and
    a Wilson-score 95% confidence interval on the win rate - more accurate than a
    naive normal-approximation interval at small n, which is what this study's
    reduced-scale evaluation (n=50-80 per pairing) actually needs."""
    outcomes = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outcomes.append(json.loads(line)["outcome"])

    n = len(outcomes)
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    draws = outcomes.count("draw")

    if n == 0:
        return {"n": 0, "wins": 0, "losses": 0, "draws": 0, "win_rate": None, "ci_95": None}

    p = wins / n
    z = 1.96   # 95% CI
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    ci_low = max(0.0, center - half_width)
    ci_high = min(1.0, center + half_width)

    return {
        "n": n, "wins": wins, "losses": losses, "draws": draws,
        "win_rate": round(p, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
    }


if __name__ == "__main__":
    import sys
    result = win_rate_from_file(sys.argv[1])
    print(json.dumps(result, indent=2))
