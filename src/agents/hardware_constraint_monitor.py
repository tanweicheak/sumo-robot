"""
src.agents.hardware_constraint_monitor

Phase: Phase 2
Purpose: Hardware Constraint Monitor Agent (HCMA) - deterministic, non-SLM
    (report Section 3.3.2.1 / 3.3.3.4). Reads config/hcma_policy.yaml and provides two
    controls: a token-budget governor (scales each agent's max_new_tokens by remaining
    latency headroom) and an emergency SA-bypass circuit-breaker (skips SA when the
    decision-window budget is nearly exhausted, reusing the No-SA data path). In Phase 2
    the HEL latency feed is mocked; Phase 3 subscribes it to real HEL logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config_loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HCMA_CONFIG = _REPO_ROOT / "config" / "hcma_policy.yaml"


class HCMAPolicy:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.decision_window_ms = float(cfg.get("decision_window_ms", 50))
        self.gov = cfg.get("token_budget_governor", {})
        self.bypass = cfg.get("emergency_bypass", {})

    @classmethod
    def from_config(cls, config_path: str | Path = _DEFAULT_HCMA_CONFIG) -> "HCMAPolicy":
        return cls(load_config(config_path))

    def compute_token_budget(self, agent_name: str, headroom_ratio: float) -> int:
        """Return max_new_tokens for `agent_name` given remaining headroom (0..1).
        Returns -1 if the governor is disabled (meaning: no cap)."""
        if not self.gov.get("enabled", False):
            return -1
        base = int(self.gov["base_max_new_tokens"].get(agent_name, 128))
        floor = int(self.gov["min_max_new_tokens"].get(agent_name, 64))
        scaling = self.gov["headroom_scaling"]
        full_above = float(scaling["full_budget_above_ratio"])
        tighten_below = float(scaling["tighten_below_ratio"])
        if headroom_ratio >= full_above:
            return base
        if headroom_ratio <= tighten_below:
            return floor
        span = full_above - tighten_below
        frac = (headroom_ratio - tighten_below) / span if span > 0 else 1.0
        return int(floor + frac * (base - floor))

    def should_bypass_sa(self, consumed_ratio: float) -> bool:
        """True if consumed budget (0..1 of the decision window) exceeds the bypass
        trigger, meaning SA should be skipped and OAA routed straight to TEA."""
        if not self.bypass.get("enabled", False):
            return False
        return consumed_ratio >= float(self.bypass["trigger_consumed_ratio"])