"""
src.sbso.macro_executor

Phase: Phase 4 (Stage 2)
Purpose: Deterministic MacroStrategy -> PWM proxy used ONLY inside MCTS tree search.
    Real agent inference (SA/TEA via SLM) is far too slow to call per rollout step
    (dozens of rollouts per decision x horizon steps). This scripted executor gives
    MCTS a physically plausible action for each macro-strategy so it can cheaply
    explore consequences in PyBullet. It is NOT what gets deployed - DSPy/LoRA later
    teach the real SLM pipeline to reproduce MCTS's discovered (state -> strategy)
    choices; TEA still does the actual nuanced execution at inference/deployment time.
    Edge-avoidance is a hard safety override regardless of the chosen strategy.
"""

from __future__ import annotations

from typing import Optional

from src.agents.schemas import MacroStrategy


class MacroStrategyExecutor:
    def __init__(
        self,
        charge_speed: float = 1.0,
        flank_speed: float = 0.9,
        flank_arc: float = 0.35,
        retreat_speed: float = 0.8,
        turn_speed: float = 0.7,
        edge_ir_threshold: float = 0.85,
    ) -> None:
        self.charge_speed = charge_speed
        self.flank_speed = flank_speed
        self.flank_arc = flank_arc
        self.retreat_speed = retreat_speed
        self.turn_speed = turn_speed
        self.edge_ir_threshold = edge_ir_threshold
        self._flank_dir = 1.0

    def reset(self) -> None:
        self._flank_dir = 1.0

    def to_pwm(self, strategy: MacroStrategy, tof, ir) -> tuple[float, float]:
        # Safety override: edge detected always evades, regardless of chosen strategy.
        if any(x >= self.edge_ir_threshold for x in ir) and strategy != MacroStrategy.EVADE_EDGE:
            return self._evade_edge(ir)

        if strategy == MacroStrategy.CHARGE:
            return (self.charge_speed, self.charge_speed)
        if strategy == MacroStrategy.FLANK:
            if self._flank_dir > 0:
                return (self.flank_speed * self.flank_arc, self.flank_speed)
            return (self.flank_speed, self.flank_speed * self.flank_arc)
        if strategy == MacroStrategy.RETREAT:
            return (-self.retreat_speed, -self.retreat_speed)
        if strategy == MacroStrategy.HOLD:
            return (0.0, 0.0)
        if strategy == MacroStrategy.EVADE_EDGE:
            return self._evade_edge(ir)
        return (0.0, 0.0)

    def _evade_edge(self, ir) -> tuple[float, float]:
        fl, fr = float(ir[0]), float(ir[1])
        pivot = 1.0 if fl >= fr else -1.0
        if max(fl, fr) < 0.95:
            return (-self.retreat_speed, -self.retreat_speed)
        return (self.turn_speed * pivot, -self.turn_speed * pivot)


class MacroStrategyExecutorOpponent:
    """Default fast opponent proxy for use ONLY inside MCTS rollouts (D2). Wraps a
    MacroStrategyExecutor driving a fixed macro-strategy, exposed as an OpponentPolicy
    callable (obs -> (left_pwm, right_pwm)) so it can be swapped directly onto
    PyBulletSumoEnv.opponent_policy. Used automatically by PyBulletSimulationBackend
    whenever no explicit opponent_policy is supplied, so the opponent side of every
    rollout stays as cheap and deterministic as the agent side - regardless of what the
    opponent "really" is (baseline controller or self-checkpoint) outside tree search."""

    def __init__(
        self,
        strategy: MacroStrategy = MacroStrategy.CHARGE,
        executor: Optional[MacroStrategyExecutor] = None,
    ) -> None:
        self.strategy = strategy
        self.executor = executor or MacroStrategyExecutor()

    def __call__(self, obs: dict) -> tuple[float, float]:
        return self.executor.to_pwm(self.strategy, obs["tof"], obs["ir"])