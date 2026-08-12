"""
src.sbso.recompilation_scheduler

Phase: Phase 4
Purpose: DSPy recompilation trigger (report Section 3.3.4). Recompile every K episodes,
    OR when the rolling win-rate drops by more than delta vs. the previous window -
    whichever first. Defaults: K=500 (aligned with self-checkpoint interval), W=100, delta=0.05.
"""

from __future__ import annotations

from typing import Optional


class RecompilationScheduler:
    def __init__(self, k_episodes: int = 500, window_w: int = 100, delta: float = 0.05) -> None:
        self.k_episodes = k_episodes
        self.window_w = window_w
        self.delta = delta
        self._last_recompile = 0
        self._prev_window_wr: Optional[float] = None

    def should_recompile(self, episode: int, rolling_winrate: Optional[float]) -> tuple[bool, Optional[str]]:
        # Trigger (a): every K episodes.
        if episode - self._last_recompile >= self.k_episodes:
            self._last_recompile = episode
            self._prev_window_wr = rolling_winrate
            return True, "k_episodes"
        # Trigger (b): rolling win-rate dropped by more than delta vs. prior window.
        if self._prev_window_wr is not None and rolling_winrate is not None:
            if rolling_winrate < self._prev_window_wr - self.delta:
                self._last_recompile = episode
                self._prev_window_wr = rolling_winrate
                return True, "reward_drop"
        if self._prev_window_wr is None:
            self._prev_window_wr = rolling_winrate
        return False, None