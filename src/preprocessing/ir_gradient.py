"""
src.preprocessing.ir_gradient

Phase: Phase 2
Purpose: Rolling-window gradient of the analog IR edge readings, yielding a continuous
    rate-of-approach signal (report Section 3.3.2.2) instead of a binary boundary
    trigger. A rising IR reading (probe approaching the void) produces a positive rate.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class IRGradientFilter:
    def __init__(self, window_length: int = 5, n_probes: int = 2) -> None:
        self.window_length = window_length
        self.n_probes = n_probes
        self._buffers = [deque(maxlen=window_length) for _ in range(n_probes)]

    def reset(self) -> None:
        for b in self._buffers:
            b.clear()

    def update(self, ir_sample) -> float:
        """Push one IR sample (per probe); return the max rate-of-approach across probes."""
        ir_sample = np.asarray(ir_sample, dtype=np.float32).reshape(-1)
        rates = np.zeros(self.n_probes, dtype=np.float32)
        for i in range(self.n_probes):
            self._buffers[i].append(float(ir_sample[i]))
            buf = self._buffers[i]
            if len(buf) >= 2:
                rates[i] = buf[-1] - buf[0]   # change across the window
        return float(np.max(rates))