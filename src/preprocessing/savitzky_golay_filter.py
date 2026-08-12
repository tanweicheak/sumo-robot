"""
src.preprocessing.savitzky_golay_filter

Phase: Phase 2
Purpose: Savitzky-Golay smoothing for the ToF distance channel (report Section 3.3.2.2).
    Buffers recent samples per ray and returns the smoothed latest value. Until the
    buffer fills, it passes the raw latest sample through unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SavitzkyGolayFilter:
    def __init__(self, window_length: int = 7, polyorder: int = 2, n_channels: int = 1) -> None:
        if window_length % 2 == 0:
            window_length += 1
        if polyorder >= window_length:
            polyorder = window_length - 1
        self.window_length = window_length
        self.polyorder = polyorder
        self.n_channels = n_channels
        self._buffers = [deque(maxlen=window_length) for _ in range(n_channels)]

    def reset(self) -> None:
        for b in self._buffers:
            b.clear()

    def update(self, sample) -> np.ndarray:
        """Push one multi-channel sample; return the smoothed latest value per channel."""
        from scipy.signal import savgol_filter

        sample = np.asarray(sample, dtype=np.float32).reshape(-1)
        out = np.empty(self.n_channels, dtype=np.float32)
        for i in range(self.n_channels):
            self._buffers[i].append(float(sample[i]))
            buf = np.array(self._buffers[i], dtype=np.float32)
            if len(buf) < self.window_length:
                out[i] = buf[-1]
            else:
                out[i] = savgol_filter(buf, self.window_length, self.polyorder)[-1]
        return out