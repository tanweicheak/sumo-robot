"""
src.preprocessing.ir_gradient

Phase: Phase 2
Purpose: Rolling-window gradient of the analog IR edge readings, yielding a continuous
    rate-of-approach signal (report Section 3.3.2.2) instead of a binary boundary
    trigger. A rising IR reading (probe approaching the void) produces a positive rate.

    dt_s normalization: update() previously returned a raw (buf[-1] - buf[0]) delta
    across the sample window with no time normalization - a unitless number that
    happened to be consistent throughout the project only because control_dt_s is a
    constant 0.05s everywhere it's called. It's compared against
    arena_config.yaml's edge_approach_rate.{stable_max, approaching_max} thresholds,
    which are named (and will eventually be calibrated) as if this were a genuine
    per-second rate. Dividing by actual elapsed time makes the value mean what its
    name says, and stops silently depending on every caller happening to use the
    same control_dt_s. NOTE: this changes the numeric scale of the output by a
    factor of (window_length-1)*dt_s - both edge_approach_rate thresholds in
    arena_config.yaml are still-uncalibrated placeholders, so nothing calibrated is
    being invalidated, but recalibrate them against these units, not the old ones.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class IRGradientFilter:
    def __init__(self, window_length: int = 5, n_probes: int = 2, dt_s: float = 0.05) -> None:
        self.window_length = window_length
        self.n_probes = n_probes
        self.dt_s = dt_s
        self._buffers = [deque(maxlen=window_length) for _ in range(n_probes)]

    def reset(self) -> None:
        for b in self._buffers:
            b.clear()

    def update(self, ir_sample) -> float:
        """Push one IR sample (per probe); return the max rate-of-approach (IR units
        per second) across probes."""
        ir_sample = np.asarray(ir_sample, dtype=np.float32).reshape(-1)
        rates = np.zeros(self.n_probes, dtype=np.float32)
        for i in range(self.n_probes):
            self._buffers[i].append(float(ir_sample[i]))
            buf = self._buffers[i]
            if len(buf) >= 2:
                elapsed_s = (len(buf) - 1) * self.dt_s
                rates[i] = (buf[-1] - buf[0]) / elapsed_s if elapsed_s > 0 else 0.0
        return float(np.max(rates))