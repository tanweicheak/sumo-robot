"""
src.preprocessing.motor_deadband

Phase: Phase 2
Purpose: Deadband filter for motor/encoder feedback (report Section 3.3.2.2). Removes
    micro-fluctuations that produce no real wheel rotation, then fuses left/right wheel
    velocities into an ego-motion vector (forward + turn) so downstream reasoning uses
    true momentum rather than commanded voltage.
"""

from __future__ import annotations

import numpy as np


class MotorDeadbandFilter:
    def __init__(self, deadband: float = 0.05) -> None:
        self.deadband = deadband

    def apply(self, encoder_sample) -> dict[str, float]:
        """encoder_sample = [left_vel, right_vel, left_pos, right_pos]; returns ego-motion."""
        enc = np.asarray(encoder_sample, dtype=np.float32).reshape(-1)
        left_vel, right_vel = float(enc[0]), float(enc[1])
        if abs(left_vel) < self.deadband:
            left_vel = 0.0
        if abs(right_vel) < self.deadband:
            right_vel = 0.0
        fwd = 0.5 * (left_vel + right_vel)
        turn = 0.5 * (right_vel - left_vel)
        return {"fwd": fwd, "turn": turn, "left_vel": left_vel, "right_vel": right_vel}