"""
scripts.report_mcts_judge_calibration

Phase: pilot-calibration / ongoing diagnostic, no extra simulation cost
Purpose: Answers two closely related open questions with the SAME underlying data:

    1. MCTS proxy-drift: MCTS searches against a scripted proxy opponent, never the
       real sampled opponent (see the Part 2 discussion for the full mechanism). Is
       the search-time "predicted value" for the chosen strategy actually predictive
       of what happens once that decision is played for real against the real
       opponent?
    2. Judge calibration: when a rollout doesn't terminate, its leaf value comes
       from judge.score_position() - is the Judge's own scoring predictive of real
       match outcomes at all?

    In this architecture these turn out to be nearly the same question - the
    "proxy-predicted value" logged per decision (MCTSResult.root_stats' mean_value
    for the chosen branch) is itself substantially built from Judge scores
    backpropagated up the tree. One correlation answers both: a low correlation
    means EITHER the proxy opponent is unrepresentative, OR the Judge's scoring is
    poorly calibrated, OR both - worth knowing before trusting either uncritically,
    but this diagnostic alone won't tell you which of the two is the cause.

    Reads mcts_calibration.jsonl, written incrementally by run_phase4_pilot.py /
    run_phase4_stage3_local.py's on_episode_end hook - no extra simulation or judge
    calls, purely free instrumentation on data already computed during search.

Usage:
    python -m scripts.report_mcts_judge_calibration --calibration-log checkpoints/<run>/mcts_calibration.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read_log(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Written incrementally during a real training run "
            "(run_phase4_pilot.py / run_phase4_stage3_local.py) - nothing to analyze "
            "until one has produced it."
        )
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("proxy_predicted_value") is not None:
                    rows.append(row)
    return rows


def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom > 0 else None


def compute_calibration(rows: list[dict], n_bins: int = 5) -> dict:
    xs = [r["proxy_predicted_value"] for r in rows]
    ys = [r["match_outcome_value"] for r in rows]
    r = _pearson_r(xs, ys)

    # Bin by predicted value, report actual mean outcome per bin - a well-calibrated
    # predictor should show actual outcome rate rising roughly monotonically with
    # predicted value.
    lo, hi = min(xs), max(xs)
    span = max(1e-9, hi - lo)
    bins = [[] for _ in range(n_bins)]
    for x, y in zip(xs, ys):
        idx = min(n_bins - 1, int((x - lo) / span * n_bins))
        bins[idx].append(y)

    bin_report = []
    for i, b in enumerate(bins):
        bin_lo = lo + span * i / n_bins
        bin_hi = lo + span * (i + 1) / n_bins
        bin_report.append({
            "predicted_range": [round(bin_lo, 3), round(bin_hi, 3)],
            "n": len(b),
            "actual_mean_outcome": round(sum(b) / len(b), 3) if b else None,
        })

    return {
        "n_decisions": len(rows),
        "pearson_r": round(r, 4) if r is not None else None,
        "bins": bin_report,
    }


def format_report(calib: dict) -> str:
    lines = [f"Decisions analyzed: {calib['n_decisions']}", f"Pearson r (predicted vs. actual): {calib['pearson_r']}", ""]
    lines.append("Predicted-value bin -> actual mean outcome (0=loss, 0.5=draw, 1=win):")
    for b in calib["bins"]:
        lo, hi = b["predicted_range"]
        actual = f"{b['actual_mean_outcome']:.3f}" if b["actual_mean_outcome"] is not None else "n/a"
        lines.append(f"  [{lo:.3f}, {hi:.3f})  n={b['n']:<5}  actual_mean_outcome={actual}")

    r = calib["pearson_r"]
    lines.append("")
    if r is None:
        lines.append("Not enough data to compute a correlation yet.")
    elif r >= 0.4:
        lines.append(
            f"r={r:.2f} - meaningful positive correlation. Search-time predictions and Judge "
            "scoring are reasonably informative about real outcomes. The proxy-opponent "
            "gap and Judge miscalibration are unlikely to be dominant problems."
        )
    elif r >= 0.15:
        lines.append(
            f"r={r:.2f} - weak positive correlation. Some signal, but noisy - worth watching "
            "as more data accumulates, not yet a clear red flag either way."
        )
    else:
        lines.append(
            f"r={r:.2f} - little to no correlation. Search-time predictions are not "
            "reliably informative about real outcomes. Given how intertwined they are in "
            "this architecture, this doesn't distinguish 'the proxy opponent is "
            "unrepresentative' from 'the Judge is poorly calibrated' - but it does say "
            "one or both are real problems, worth the more expensive real-rollout "
            "calibration (occasionally swapping in the real opponent during search) to "
            "isolate which."
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MCTS proxy-drift / Judge-calibration diagnostic.")
    p.add_argument("--calibration-log", required=True, help="Path to mcts_calibration.jsonl")
    p.add_argument("--bins", type=int, default=5)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = _read_log(args.calibration_log)
    calib = compute_calibration(rows, n_bins=args.bins)
    print(format_report(calib))


if __name__ == "__main__":
    main()