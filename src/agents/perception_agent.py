"""
src.agents.perception_agent

Phase: Phase 2
Purpose: Perception Agent (PA) - deterministic, non-SLM (report Section 3.3.2.1/3.3.2.3).
    Applies the preprocessing pipeline (Savitzky-Golay on ToF, rolling IR gradient,
    motor deadband + ego-motion fusion) then the LSSD encoder, producing a
    PerceptionState. Stateful across a match (the filters buffer history), so call
    reset() at the start of each episode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.schemas import (
    DirectionLabel,
    DistanceLabel,
    EdgeLabel,
    MomentumLabel,
    PerceptionState,
)
from src.data.lssd_encoder import LSSDEncoder
from src.preprocessing.ir_gradient import IRGradientFilter
from src.preprocessing.motor_deadband import MotorDeadbandFilter
from src.preprocessing.savitzky_golay_filter import SavitzkyGolayFilter


class PerceptionAgent:
    def __init__(
        self,
        config_path: str | Path | None = None,
        tof_num_rays: int = 7,
        n_ir_probes: int = 2,
        detect_range_m: float = 1.4,
        control_dt_s: float = 0.05,
    ) -> None:
        self.detect_range_m = detect_range_m
        self.tof_filter = SavitzkyGolayFilter(window_length=7, polyorder=2, n_channels=tof_num_rays)
        self.ir_filter = IRGradientFilter(window_length=5, n_probes=n_ir_probes, dt_s=control_dt_s)
        self.motor_filter = MotorDeadbandFilter(deadband=0.05)
        self.lssd = (
            LSSDEncoder.from_config(config_path, tof_num_rays=tof_num_rays)
            if config_path
            else LSSDEncoder.from_config(tof_num_rays=tof_num_rays)
        )

    def reset(self) -> None:
        self.tof_filter.reset()
        self.ir_filter.reset()

    def perceive(self, raw_tof, raw_ir, raw_encoder) -> PerceptionState:
        tof_smoothed = self.tof_filter.update(raw_tof)
        approach_rate = self.ir_filter.update(raw_ir)
        ego = self.motor_filter.apply(raw_encoder)

        enc = self.lssd.encode(
            tof_smoothed, approach_rate, ego, detect_range_m=self.detect_range_m
        )
        return PerceptionState(
            lssd_text=enc["lssd_text"],
            opp_distance=DistanceLabel(enc["opp_distance"]),
            opp_direction=DirectionLabel(enc["opp_direction"]),
            edge=EdgeLabel(enc["edge"]),
            momentum=MomentumLabel(enc["momentum"]),
            opp_distance_m=enc["opp_distance_m"],
            edge_approach_rate=enc["edge_approach_rate"],
        )