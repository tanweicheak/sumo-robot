"""
src.statistics.elo

Phase: Phase 5
Purpose: ELO rating computation across all evaluated pairings, per Section
    3.3.5.4's KPI table (ELO of Benchmark 1/2 vs Tier 2/3, and Benchmark 1 vs
    Benchmark 2). Replaces the Phase 0 stub - real implementation.

Unlike win_rate/bootstrap/cohens_d/mann_whitney (all computed per-pairing, from
one results.jsonl file), ELO is inherently a POOLED calculation - each
participant's rating updates based on every match they played, across ALL
pairings, processed as one combined sequence. This function therefore takes a
directory of pairing folders (results/phase5_eval/), not a single file, and
reconstructs a single chronological-ish match sequence across all of them
(true chronological order isn't preserved across separately-run pairing batches,
so this processes each pairing's matches in file order, then pairings in
directory-listing order - a reasonable, disclosed approximation, not true
wall-clock chronology).
"""

from __future__ import annotations

import json
from pathlib import Path


def compute_elo_ratings(
    results_dir: str | Path, k_factor: float = 32.0, initial_rating: float = 1500.0,
) -> dict:
    """Reads every <attacker>_vs_<opponent>/results.jsonl under results_dir,
    processes all matches as a single sequential pool, and returns final ELO
    ratings for every participant that appeared as an attacker or opponent in any
    pairing. Draws are scored as 0.5/0.5, matching standard ELO convention."""
    results_dir = Path(results_dir)
    ratings: dict[str, float] = {}
    match_count: dict[str, int] = {}

    pairing_dirs = sorted(p for p in results_dir.iterdir() if p.is_dir())

    for pairing_dir in pairing_dirs:
        results_path = pairing_dir / "results.jsonl"
        if not results_path.exists():
            continue

        with results_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                attacker = record.get("attacker")
                opponent = record.get("opponent")
                outcome = record.get("outcome")
                if attacker is None or opponent is None:
                    continue   # older records without attacker/opponent fields - skip

                ratings.setdefault(attacker, initial_rating)
                ratings.setdefault(opponent, initial_rating)
                match_count.setdefault(attacker, 0)
                match_count.setdefault(opponent, 0)

                score_attacker = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(outcome, 0.5)
                score_opponent = 1.0 - score_attacker

                r_attacker, r_opponent = ratings[attacker], ratings[opponent]
                expected_attacker = 1.0 / (1.0 + 10 ** ((r_opponent - r_attacker) / 400))
                expected_opponent = 1.0 - expected_attacker

                ratings[attacker] = r_attacker + k_factor * (score_attacker - expected_attacker)
                ratings[opponent] = r_opponent + k_factor * (score_opponent - expected_opponent)

                match_count[attacker] += 1
                match_count[opponent] += 1

    return {
        "ratings": {name: round(r, 1) for name, r in ratings.items()},
        "matches_played": match_count,
        "k_factor": k_factor,
        "note": "Match order approximates directory/file order, not true wall-clock "
                "chronology across separately-run pairing batches - disclosed limitation.",
    }


def elo_gap(ratings: dict, name_a: str, name_b: str) -> dict:
    """Direct ELO gap between two specific participants, for KPI 5/6's '+-100 ELO'
    threshold check."""
    if name_a not in ratings or name_b not in ratings:
        return {"gap": None, "note": f"one or both of {name_a!r}/{name_b!r} not found in ratings"}
    gap = ratings[name_a] - ratings[name_b]
    return {
        "rating_a": round(ratings[name_a], 1), "rating_b": round(ratings[name_b], 1),
        "gap": round(gap, 1), "within_100": bool(abs(gap) <= 100),
    }


if __name__ == "__main__":
    import sys
    result = compute_elo_ratings(sys.argv[1])
    print(json.dumps(result, indent=2))
