"""
src.agents.actuator_bridge

Phase: Phase 2
Purpose: Actuator Bridge - deterministic, non-SLM (report Section 3.3.2.4). Maps the
    TEA's TacticalKeyword to normalized (left_pwm, right_pwm) in [-1, 1] for the env /
    hardware. PWM levels come from config so they track the calibrated Phase 1 physics.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.schemas import TacticalCommand, TacticalKeyword
from src.common.config_loader import load_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARENA_CONFIG = _REPO_ROOT / "config" / "arena_config.yaml"


class ActuatorBridge:
    def __init__(
        self,
        charge_speed: float = 1.0,
        turn_speed: float = 0.7,
        reverse_speed: float = 0.9,
        arc_inner_speed: float = 0.3,
    ) -> None:
        self.charge_speed = charge_speed
        self.turn_speed = turn_speed
        self.reverse_speed = reverse_speed
        self.arc_inner_speed = arc_inner_speed

    @classmethod
    def from_config(cls, config_path: str | Path = _DEFAULT_ARENA_CONFIG) -> "ActuatorBridge":
        # Reuse the rule-based-style motor levels if present; otherwise defaults.
        cfg = load_config(config_path)
        bridge = cfg.get("actuator_bridge", {})
        return cls(
            charge_speed=float(bridge.get("charge_speed", 1.0)),
            turn_speed=float(bridge.get("turn_speed", 0.7)),
            reverse_speed=float(bridge.get("reverse_speed", 0.9)),
            arc_inner_speed=float(bridge.get("arc_inner_speed", 0.3)),
        )

    def to_pwm(self, command: TacticalCommand) -> tuple[float, float]:
        k = command.keyword
        if k == TacticalKeyword.CHARGE_FORWARD:
            return (self.charge_speed, self.charge_speed)
        if k == TacticalKeyword.ARC_LEFT:
            return (self.arc_inner_speed, self.charge_speed)
        if k == TacticalKeyword.ARC_RIGHT:
            return (self.charge_speed, self.arc_inner_speed)
        if k == TacticalKeyword.PIVOT_LEFT:
            return (-self.turn_speed, self.turn_speed)
        if k == TacticalKeyword.PIVOT_RIGHT:
            return (self.turn_speed, -self.turn_speed)
        if k == TacticalKeyword.REVERSE:
            return (-self.reverse_speed, -self.reverse_speed)
        # STOP or any unmapped keyword.
        return (0.0, 0.0)