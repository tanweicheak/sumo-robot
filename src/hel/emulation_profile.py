"""
src.hel.emulation_profile

Phase: Phase 3
Purpose: Hardware Emulation Layer profile (report Section 3.3.3.4). Loads the Jetson
    Nano constraint profile. Per project scope, HEL is ACTIVE ONLY IN PHASE 5c - Phases
    5a/5b run full-power simulation and pass measured latency through unchanged. The
    profile object is built now (cheap); its throttling/latency-injection only takes
    effect when 5c enables it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.common.config_loader import load_config


@dataclass
class HELProfile:
    device_name: str
    cpu_quota_percent: float
    ram_cap_mb: float
    latency_injection_enabled: bool
    latency_additive_ms: float
    decision_window_ms: float = 50.0

    @classmethod
    def from_config(cls, profile_path: str | Path, decision_window_ms: float = 50.0) -> "HELProfile":
        cfg = load_config(profile_path)
        return cls(
            device_name=cfg.get("device_name", "unknown"),
            cpu_quota_percent=float(cfg["cpu"]["quota_percent"]),
            ram_cap_mb=float(cfg["memory"]["ram_cap_mb"]),
            latency_injection_enabled=bool(cfg["latency_injection"]["enabled"]),
            latency_additive_ms=float(cfg["latency_injection"].get("additive_ms", 0.0)),
            decision_window_ms=decision_window_ms,
        )

    def inject_latency_ms(self, measured_ms: float) -> float:
        """Apply emulated latency. In full-power sim (5a/5b) injection is disabled and
        the measured value passes through; in 5c it adds the profile's overhead."""
        if not self.latency_injection_enabled:
            return measured_ms
        return measured_ms + self.latency_additive_ms