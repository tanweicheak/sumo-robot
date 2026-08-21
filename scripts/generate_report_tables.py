"""
scripts.generate_report_tables

Phase: Reporting
Purpose: Regenerate thesis sensitivity/robustness tables from results/ and
    checkpoints/ - the standard reporting artifact discussed in the "senior
    engineer" review: sensitivity checks (sim_budget, stress-test legitimacy,
    mirror-match balance, DSPy recompile behavior) built ad hoc one at a time
    during debugging should be regenerable as one consistent pass, not
    reconstructed by hand each time for the writeup.

    Each table function is independent and gracefully skips (prints a note, does
    not crash) if its source file doesn't exist yet - this script is meant to be
    run repeatedly as more real data accumulates over the project, not only once
    everything is finished. Every function is tested in this build against
    synthetic data matching the REAL format each producing script
    (run_mcts_sim_budget_sweep.py, run_stress_test.py, run_mirror_match.py,
    run_phase4_pilot.py's prompt_history.jsonl) actually writes today.

Usage:
    python -m scripts.generate_report_tables --config config/phase4_pilot.yaml \\
        --sim-budget-sweep results/mcts_sim_budget_sweep/sweep_summary.json \\
        --stress-test-results results/stress_test_bait/results.jsonl \\
        --mirror-match-results results/mirror_match/results.jsonl \\
        --prompt-history checkpoints/benchmark2_full_sbso/prompt_history.jsonl \\
        --output-dir results/report_tables
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts._script_common import build_run


def _read_jsonl(path: str | Path) -> list[dict] | None:
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sim_budget_sensitivity_table(sweep_summary_path: str | Path) -> dict | None:
    path = Path(sweep_summary_path)
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    rows = [
        {
            "sim_budget": r["sim_budget"], "win_rate": r.get("win_rate"),
            "avg_decision_seconds": r.get("avg_decision_seconds"),
            "judge_call_count": r.get("judge_call_count"),
        }
        for r in results
    ]
    return {"rows": rows}


def stress_test_outcome_table(results_path: str | Path) -> dict | None:
    """Reproduces the win/pushed_out vs win/capsized vs loss/pushed_out vs
    loss/capsized breakdown done by hand earlier this session - now a repeatable
    table instead of a one-off manual tally."""
    rows = _read_jsonl(results_path)
    if rows is None:
        return None

    by_attacker: dict[str, dict[str, int]] = {}
    for r in rows:
        attacker = r["attacker"]
        key = f"{r['outcome']}_{r['reason']}" if r["outcome"] != "draw" else "draw"
        by_attacker.setdefault(attacker, {}).setdefault(key, 0)
        by_attacker[attacker][key] += 1

    out = {}
    for attacker, counts in by_attacker.items():
        total = sum(counts.values())
        decided = total - counts.get("draw", 0)
        legitimate = counts.get("win_pushed_out", 0) + counts.get("loss_pushed_out", 0)
        out[attacker] = {
            "total_episodes": total,
            "counts": counts,
            "win_rate": round((counts.get("win_pushed_out", 0) + counts.get("win_capsized", 0)) / total, 4) if total else None,
            "legitimacy_rate": round(legitimate / decided, 4) if decided else None,
        }
    return out


def mirror_match_balance_table(results_path: str | Path) -> dict | None:
    rows = _read_jsonl(results_path)
    if rows is None:
        return None

    counts = {"win": 0, "loss": 0, "draw": 0}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    total = sum(counts.values())
    decided = counts["win"] + counts["loss"]
    win_rate_among_decided = counts["win"] / decided if decided else None

    # Standard error for a binomial proportion at n=decided, p=0.5 - flags whether
    # the observed win rate is within noise of a true 50/50 split, the same check
    # done by hand for the earlier 74W/69L/57D mirror-match result.
    se = math.sqrt(0.25 / decided) if decided else None
    within_noise = abs(win_rate_among_decided - 0.5) < 2 * se if (win_rate_among_decided is not None and se) else None

    return {
        "total_episodes": total, "counts": counts, "decided_matches": decided,
        "win_rate_among_decided": round(win_rate_among_decided, 4) if win_rate_among_decided is not None else None,
        "draw_rate": round(counts["draw"] / total, 4) if total else None,
        "standard_error": round(se, 4) if se else None,
        "within_2se_of_50_50": within_noise,
    }


def dspy_recompile_summary_table(prompt_history_path: str | Path) -> dict | None:
    rows = _read_jsonl(prompt_history_path)
    if rows is None:
        return None

    by_reason: dict[str, int] = {}
    accepted, rejected = 0, 0
    for r in rows:
        by_reason[r["trigger_reason"]] = by_reason.get(r["trigger_reason"], 0) + 1
        if r.get("accepted") is False:
            rejected += 1
        elif r.get("accepted") is True or "accepted" not in r:
            accepted += 1

    return {
        "total_recompile_attempts": len(rows), "by_trigger_reason": by_reason,
        "accepted": accepted, "rejected": rejected,
        "final_prompt_version": max((r["prompt_version"] for r in rows), default=0),
    }


def format_markdown_report(tables: dict) -> str:
    lines = ["# Sensitivity / Robustness Report", ""]

    lines.append("## MCTS sim_budget sensitivity")
    t = tables.get("sim_budget")
    if t is None:
        lines.append("_Not yet available - run scripts.run_mcts_sim_budget_sweep first._\n")
    else:
        lines.append("| sim_budget | win_rate | avg_decision_seconds | judge_call_count |")
        lines.append("|---|---|---|---|")
        for r in t["rows"]:
            lines.append(f"| {r['sim_budget']} | {r['win_rate']} | {r['avg_decision_seconds']} | {r['judge_call_count']} |")
        lines.append("")

    lines.append("## Stress-test outcome legitimacy (out-of-family opponent)")
    t = tables.get("stress_test")
    if t is None:
        lines.append("_Not yet available - run scripts.run_stress_test first._\n")
    else:
        for attacker, data in t.items():
            lines.append(f"**{attacker}** - {data['total_episodes']} episodes, "
                          f"win_rate={data['win_rate']}, legitimacy_rate={data['legitimacy_rate']}")
            lines.append(f"  counts: {data['counts']}")
        lines.append("")

    lines.append("## Mirror-match balance")
    t = tables.get("mirror_match")
    if t is None:
        lines.append("_Not yet available - run scripts.run_mirror_match first._\n")
    else:
        lines.append(f"- {t['total_episodes']} episodes ({t['decided_matches']} decided, "
                      f"draw_rate={t['draw_rate']})")
        lines.append(f"- win_rate_among_decided={t['win_rate_among_decided']} "
                      f"(within 2 SE of 50/50: {t['within_2se_of_50_50']})")
        lines.append("")

    lines.append("## DSPy recompilation summary")
    t = tables.get("dspy_recompile")
    if t is None:
        lines.append("_Not yet available - run a real training run with prompt_history.jsonl output first._\n")
    else:
        lines.append(f"- {t['total_recompile_attempts']} recompile attempts, "
                      f"{t['accepted']} accepted, {t['rejected']} rejected")
        lines.append(f"- by trigger reason: {t['by_trigger_reason']}")
        lines.append(f"- final prompt_version: {t['final_prompt_version']}")

    return "\n".join(lines)


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sim-budget-sweep", default="results/mcts_sim_budget_sweep/sweep_summary.json")
    parser.add_argument("--stress-test-results", default="results/stress_test_bait/results.jsonl")
    parser.add_argument("--mirror-match-results", default="results/mirror_match/results.jsonl")
    parser.add_argument("--prompt-history", default=None,
                         help="e.g. checkpoints/benchmark2_full_sbso/prompt_history.jsonl")
    parser.add_argument("--output-dir", default="results/report_tables")


def main() -> None:
    config, ctx, args = build_run(
        phase="reporting", description="Regenerate thesis sensitivity/robustness tables.",
        extra_args=_add_report_args,
    )

    tables = {
        "sim_budget": sim_budget_sensitivity_table(args.sim_budget_sweep),
        "stress_test": stress_test_outcome_table(args.stress_test_results),
        "mirror_match": mirror_match_balance_table(args.mirror_match_results),
        "dspy_recompile": dspy_recompile_summary_table(args.prompt_history) if args.prompt_history else None,
    }

    report = format_markdown_report(tables)
    print(report)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "REPORT.md"
    report_path.write_text(report)
    print(f"\n[reporting] wrote -> {report_path}")


if __name__ == "__main__":
    main()