"""
src.finetuning.cost_projection

Phase: Phase 4 (Stage 3 pilot)
Purpose: Extrapolate a pilot run's measured wall-clock time to the full 5-variant x
    5000-episode Phase 4 run, with a GPU rental cost estimate. Pure math, no deps.
"""

from __future__ import annotations


def project_full_run(
    pilot_episodes: int,
    pilot_wall_clock_s: float,
    full_episodes_per_variant: int = 5000,
    num_variants: int = 5,
    gpu_rate_usd_per_hr: float = 0.34,   # RunPod RTX 4090 community, fixed - update per current pricing
) -> dict:
    sec_per_episode = pilot_wall_clock_s / max(1, pilot_episodes)
    full_episodes_total = full_episodes_per_variant * num_variants
    projected_seconds = sec_per_episode * full_episodes_total
    projected_hours = projected_seconds / 3600.0
    projected_cost = projected_hours * gpu_rate_usd_per_hr
    return {
        "sec_per_episode": round(sec_per_episode, 3),
        "full_episodes_total": full_episodes_total,
        "projected_hours": round(projected_hours, 2),
        "projected_cost_usd": round(projected_cost, 2),
        "gpu_rate_usd_per_hr": gpu_rate_usd_per_hr,
    }