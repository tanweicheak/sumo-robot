"""
src.data.lssd_encoder

Phase: Phase 2
Purpose: Localized Semantic State Description encoder (report Section 3.3.2.3). A
    deterministic, non-SLM mapping from filtered sensor features to a token-efficient
    key-value string, e.g. "opp=near,dir=FC;edge=safe;mom=fwd". Thresholds come from
    config/arena_config.yaml (placeholder values, plan Assumption A5). Compact key-value
    format chosen over prose for token efficiency / lower inference latency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.common.config_loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARENA_CONFIG = _REPO_ROOT / "config" / "arena_config.yaml"


class LSSDEncoder:
    def __init__(self, thresholds: dict[str, Any], tof_num_rays: int = 7) -> None:
        self.t = thresholds
        self.tof_num_rays = tof_num_rays

    @classmethod
    def from_config(cls, config_path: str | Path = _DEFAULT_ARENA_CONFIG, tof_num_rays: int = 7) -> "LSSDEncoder":
        cfg = load_config(config_path)
        return cls(cfg["lssd_thresholds"], tof_num_rays=tof_num_rays)

    def _distance_label(self, min_dist_cm: float, detected: bool) -> str:
        if not detected:
            return "none"
        near_max = self.t["distance_cm"]["near_max"]
        mid_max = self.t["distance_cm"]["mid_max"]
        if min_dist_cm < near_max:
            return "near"
        if min_dist_cm < mid_max:
            return "mid"
        return "far"

    def _direction_label(self, min_idx: int, detected: bool) -> str:
        if not detected:
            return "none"
        third = self.tof_num_rays / 3.0
        if min_idx < third:
            return "FL"
        if min_idx < 2 * third:
            return "FC"
        return "FR"

    def _edge_label(self, approach_rate: float) -> str:
        stable_max = self.t["edge_approach_rate"]["stable_max"]
        approaching_max = self.t["edge_approach_rate"]["approaching_max"]
        if approach_rate < stable_max:
            return "safe"
        if approach_rate < approaching_max:
            return "warning"
        return "critical"

    def _momentum_label(self, ego: dict[str, float]) -> str:
        fwd, turn = ego["fwd"], ego["turn"]
        if abs(fwd) < 0.1 and abs(turn) < 0.1:
            return "still"
        if abs(turn) > abs(fwd):
            return "turn"
        return "fwd" if fwd >= 0 else "rev"

    def encode(
        self,
        tof_m,
        approach_rate: float,
        ego: dict[str, float],
        detect_range_m: float = 1.4,
    ) -> dict[str, Any]:
        """Return a dict with the LSSD text and its structured labels + raw features."""
        tof = np.asarray(tof_m, dtype=np.float32)
        min_idx = int(np.argmin(tof))
        min_dist_m = float(tof[min_idx])
        detected = min_dist_m < detect_range_m

        dist_label = self._distance_label(min_dist_m * 100.0, detected)
        dir_label = self._direction_label(min_idx, detected)
        edge_label = self._edge_label(approach_rate)
        mom_label = self._momentum_label(ego)

        text = f"opp={dist_label},dir={dir_label};edge={edge_label};mom={mom_label}"
        return {
            "lssd_text": text,
            "opp_distance": dist_label,
            "opp_direction": dir_label,
            "edge": edge_label,
            "momentum": mom_label,
            "opp_distance_m": min_dist_m,
            "edge_approach_rate": approach_rate,
        }