"""
src.agents.schemas

Phase: Phase 2
Purpose: Pydantic data contracts for the multi-agent pipeline. Defines the enums
    (OpponentBehavior, MacroStrategy, TacticalKeyword) and the state objects passed
    between PA -> {OAA, SA} -> TEA, plus the unified SumoRobotState carried through the
    LangGraph state graph. TEA emits a TacticalKeyword (option b); the Actuator Bridge
    maps that keyword to PWM.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


# --- Reasoning vocabulary -----------------------------------------------------

class OpponentBehavior(str, Enum):
    AGGRESSIVE = "aggressive"    # charging / pushing toward us
    EVASIVE = "evasive"          # dodging / circling
    DEFENSIVE = "defensive"      # holding position
    UNKNOWN = "unknown"          # insufficient information


class MacroStrategy(str, Enum):
    CHARGE = "charge"            # direct frontal attack
    FLANK = "flank"             # approach from the side
    RETREAT = "retreat"          # create distance
    HOLD = "hold"               # defensive wait
    EVADE_EDGE = "evade_edge"    # prioritize not falling off the ring


class TacticalKeyword(str, Enum):
    CHARGE_FORWARD = "charge_forward"
    ARC_LEFT = "arc_left"
    ARC_RIGHT = "arc_right"
    PIVOT_LEFT = "pivot_left"
    PIVOT_RIGHT = "pivot_right"
    REVERSE = "reverse"
    STOP = "stop"


# --- LSSD categorical labels (Perception Agent output vocabulary) --------------

class DistanceLabel(str, Enum):
    NEAR = "near"
    MID = "mid"
    FAR = "far"
    NONE = "none"      # no opponent detected within range


class DirectionLabel(str, Enum):
    FRONT_LEFT = "FL"
    FRONT_CENTER = "FC"
    FRONT_RIGHT = "FR"
    NONE = "none"


class EdgeLabel(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


class MomentumLabel(str, Enum):
    FORWARD = "fwd"
    REVERSE = "rev"
    TURNING = "turn"
    STILL = "still"


# --- Agent output objects -----------------------------------------------------

class PerceptionState(BaseModel):
    """Deterministic PA output: the LSSD text plus the structured labels it encodes."""
    lssd_text: str
    opp_distance: DistanceLabel
    opp_direction: DirectionLabel
    edge: EdgeLabel
    momentum: MomentumLabel
    opp_distance_m: float = Field(..., description="nearest ToF distance, meters")
    edge_approach_rate: float = Field(0.0, description="IR rate-of-approach signal")


class OpponentAnalysis(BaseModel):
    """OAA output. frame_stamp records which frame produced it, so SA can reason with
    a one-frame-stale classification under the pipelined design."""
    behavior: OpponentBehavior = OpponentBehavior.UNKNOWN
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    frame_stamp: int = -1


class MacroStrategyDecision(BaseModel):
    """SA output."""
    strategy: MacroStrategy = MacroStrategy.HOLD
    rationale: Optional[str] = None


class TacticalCommand(BaseModel):
    """TEA output. A single keyword; the Actuator Bridge maps it to (left_pwm, right_pwm).
    This is the schema Outlines constrains in Phase 3."""
    keyword: TacticalKeyword


# --- Unified LangGraph state --------------------------------------------------

class SumoRobotState(BaseModel):
    """State carried through the LangGraph graph for one decision cycle.

    Pipelined-staleness note: prev_opponent_analysis holds the OAA result from frame
    t-1. SA reads it (never waiting on the fresh OAA_t), while OAA_t runs to populate
    opponent_analysis for the *next* frame. This decouples SA from OAA latency.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_index: int = 0

    # Raw sensor snapshot for this frame.
    raw_tof: list[float] = Field(default_factory=list)
    raw_ir: list[float] = Field(default_factory=list)
    raw_encoder: list[float] = Field(default_factory=list)
    lssd_history: list[str] = Field(default_factory=list)

    # Agent outputs.
    perception: Optional[PerceptionState] = None
    opponent_analysis: Optional[OpponentAnalysis] = None          # OAA_t (for next frame)
    prev_opponent_analysis: Optional[OpponentAnalysis] = None      # OAA_{t-1} (SA reads this)
    macro_strategy: Optional[MacroStrategyDecision] = None
    tactical_command: Optional[TacticalCommand] = None

    # Final actuator output for this frame.
    left_pwm: float = 0.0
    right_pwm: float = 0.0

    # Per-node timing (ms), populated by the graph for latency measurement.
    timing_ms: dict[str, float] = Field(default_factory=dict)