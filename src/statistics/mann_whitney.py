"""
src.statistics.mann_whitney

Phase: Phase 5
Purpose: Mann-Whitney U test, per Section 3.3.5.4's specified method for win-rate
    comparisons (vs Tier 1, vs Tier 2, Benchmark 1 vs Benchmark 2 - the KPI table's
    explicit use cases). Replaces the Phase 0 stub - real implementation using
    scipy (already confirmed installed on this pod for the earlier power
    calculations).
"""

from __future__ import annotations

import json
from pathlib import Path


def _outcomes_to_numeric(results_path: str | Path) -> list[float]:
    """Maps outcome strings to numeric values (win=1.0, draw=0.5, loss=0.0) -
    same mapping already used throughout this project's training code
    (match_trainer.py's outcome_value), kept consistent here rather than
    reinventing a different scale."""
    values = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outcome = json.loads(line)["outcome"]
            values.append({"win": 1.0, "draw": 0.5, "loss": 0.0}.get(outcome, 0.5))
    return values


def compare_pairings(results_path_a: str | Path, results_path_b: str | Path) -> dict:
    """Mann-Whitney U test comparing the outcome distributions of two pairings
    (e.g. Benchmark1's results vs Benchmark2's results, for the KPI 4 head-to-head
    significance test). Non-parametric - makes no assumption of normality, correct
    per the ordinal win/draw/loss outcome scale, and appropriate at this study's
    reduced sample sizes (n=50-80)."""
    from scipy import stats

    a = _outcomes_to_numeric(results_path_a)
    b = _outcomes_to_numeric(results_path_b)

    if not a or not b:
        return {"n_a": len(a), "n_b": len(b), "u_statistic": None, "p_value": None}

    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")

    return {
        "n_a": len(a), "n_b": len(b),
        "mean_a": round(sum(a) / len(a), 4), "mean_b": round(sum(b) / len(b), 4),
        "u_statistic": round(float(u_stat), 4),
        "p_value": round(float(p_value), 6),
        "significant_at_0.05": bool(p_value < 0.05),
    }


if __name__ == "__main__":
    import sys
    result = compare_pairings(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
