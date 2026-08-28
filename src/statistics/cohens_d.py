"""
src.statistics.cohens_d

Phase: Phase 5
Purpose: Cohen's d for continuous metrics, plus Cohen's h (proportion variant) for
    win-rate secondary reporting - matches this file's own original stated purpose
    (see the Phase 0 stub docstring this replaces). Used for KPI 4 (Benchmark 1 vs
    Benchmark 2 head-to-head, target: +10pp or Cohen's d >= 0.5).
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _outcomes_to_numeric(results_path: str | Path) -> list[float]:
    values = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            outcome = json.loads(line)["outcome"]
            values.append({"win": 1.0, "draw": 0.5, "loss": 0.0}.get(outcome, 0.5))
    return values


def cohens_d(results_path_a: str | Path, results_path_b: str | Path) -> dict:
    """Standard Cohen's d (pooled standard deviation) between two pairings' outcome
    distributions on the 0.0/0.5/1.0 win/draw/loss scale."""
    a = _outcomes_to_numeric(results_path_a)
    b = _outcomes_to_numeric(results_path_b)
    n_a, n_b = len(a), len(b)

    if n_a < 2 or n_b < 2:
        return {"n_a": n_a, "n_b": n_b, "d": None, "note": "need n>=2 per group"}

    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_sd == 0:
        return {"n_a": n_a, "n_b": n_b, "d": None, "note": "zero pooled variance - both groups identical"}

    d = (mean_a - mean_b) / pooled_sd
    return {
        "n_a": n_a, "n_b": n_b, "mean_a": round(mean_a, 4), "mean_b": round(mean_b, 4),
        "d": round(d, 4),
        "magnitude": _magnitude_label(abs(d)),
    }


def cohens_h(p1: float, p2: float) -> dict:
    """Cohen's h - the proportion-comparison variant, for direct win-rate-percentage
    comparisons (e.g. 62% vs 50%) rather than the ordinal outcome scale cohens_d
    above uses. phi1/phi2 are arcsine-transformed proportions; h is their
    difference - the standard formula for comparing two independent proportions."""
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    h = phi1 - phi2
    return {"p1": p1, "p2": p2, "h": round(h, 4), "magnitude": _magnitude_label(abs(h))}


def _magnitude_label(abs_d: float) -> str:
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


if __name__ == "__main__":
    import sys
    result = cohens_d(sys.argv[1], sys.argv[2])
    print(json.dumps(result, indent=2))
