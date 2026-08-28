"""
src.statistics.bootstrap

Phase: Phase 5
Purpose: Bootstrap resampling for win-rate and decision-latency confidence
    intervals (Section 3.3.5.4's specified method for Phase 5a/5b). Replaces the
    Phase 0 stub - real implementation. Deliberately used alongside win_rate.py's
    Wilson-score CI, not instead of it - bootstrap makes no distributional
    assumption at all, useful as a cross-check given this study's reduced sample
    sizes (n=50-80 per pairing).
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def bootstrap_win_rate_ci(results_path: str | Path, n_resamples: int = 10000, seed: int = 0) -> dict:
    """Reads a results.jsonl file, resamples the match outcomes with replacement
    n_resamples times, computes the win rate each time, and returns the 2.5th/97.5th
    percentiles as a 95% bootstrap CI."""
    outcomes = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outcomes.append(json.loads(line)["outcome"])

    n = len(outcomes)
    if n == 0:
        return {"n": 0, "win_rate": None, "bootstrap_ci_95": None}

    wins_binary = [1.0 if o == "win" else 0.0 for o in outcomes]
    observed_rate = sum(wins_binary) / n

    rng = random.Random(seed)
    resampled_rates = []
    for _ in range(n_resamples):
        resample = [wins_binary[rng.randrange(n)] for _ in range(n)]
        resampled_rates.append(sum(resample) / n)

    resampled_rates.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = int(0.975 * n_resamples)

    return {
        "n": n, "win_rate": round(observed_rate, 4),
        "bootstrap_ci_95": [round(resampled_rates[lo_idx], 4), round(resampled_rates[hi_idx], 4)],
        "n_resamples": n_resamples,
    }


def bootstrap_mean_ci(values: list, n_resamples: int = 10000, seed: int = 0) -> dict:
    """Same bootstrap approach applied to a continuous metric (e.g. decision
    latency) rather than a binary win/loss outcome - shared implementation, per
    Section 3.3.5.4 listing Bootstrap as the method for BOTH win rate AND decision
    latency."""
    values = [v for v in values if v is not None]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "bootstrap_ci_95": None}

    observed_mean = sum(values) / n
    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(resample) / n)

    resampled_means.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = int(0.975 * n_resamples)

    return {
        "n": n, "mean": round(observed_mean, 4),
        "bootstrap_ci_95": [round(resampled_means[lo_idx], 4), round(resampled_means[hi_idx], 4)],
        "n_resamples": n_resamples,
    }


if __name__ == "__main__":
    import sys
    result = bootstrap_win_rate_ci(sys.argv[1])
    print(json.dumps(result, indent=2))
