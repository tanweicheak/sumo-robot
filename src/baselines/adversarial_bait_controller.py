"""
src.baselines.adversarial_bait_controller

Phase: Phase 4/5 diagnostic (not one of the five official evaluation conditions)
Purpose: An out-of-family stress-test opponent. Every opponent currently in the
    training/eval lineage (Baseline 1 rule-based, Baseline 2 PPO trained against it,
    self-checkpoints of the agent trained against both) descends from the same
    behavioral archetype: detect opponent -> charge straight at it. Randomizing
    RuleBasedParams jitters speed/threshold NUMBERS, not the underlying STRATEGY -
    nothing in the lineage ever baits, retreats deliberately, or lures an opponent
    toward the edge. A model that only ever had to counter chargers can look dominant
    (near-100% win rate) while having no real answer to a genuinely different
    archetype - this controller exists to make that gap visible instead of invisible.

    BaitController is a counter-puncher, not a rusher:
      1. LURE: on detecting an opponent, retreat/circle while keeping it roughly
         centered (drawing it forward and overextended), instead of charging.
      2. COUNTER: once the opponent has closed to point-blank range (it took the
         bait) or the lure has run long enough, snap into a full-speed charge
         while it's off-balance/overextended.
      3. When not engaged: patrol toward ring center (a real competitive-sumo habit -
         stay central, let the opponent overextend near the edge) instead of
         rule-based's spin-scan/creep search pattern.
      4. Edge avoidance still overrides everything else (safety, not strategy) -
         this controller isn't testing "can it survive," it's testing "does the
         agent's approach generalize past chargers."

    Matches the same opponent_policy signature as rule_based_controller.py:
    __call__(obs: dict[str, np.ndarray]) -> tuple[float, float], drop-in compatible
    with PyBulletSumoEnv.opponent_policy.

Status: usable immediately against Baseline 1 / Baseline 2 today (see
    scripts/run_stress_test.py --attacker rule_based|ppo). Benchmark 1/2 support is a
    stub pending Phase 5's match-runner / SLM-policy-loading path - see
    _load_attacker_policy()'s NotImplementedError for the exact extension point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class BaitState(str, Enum):
    PATROL = "patrol"
    LURE = "lure"
    COUNTER = "counter"
    EDGE_AVOID = "edge_avoid"


@dataclass
class BaitControllerParams:
    # --- detection (reuses the same sensor semantics as rule_based_controller) ---
    detect_range_m: float = 1.4
    center_band: float = 0.15

    # --- lure phase ---
    # Retreat while keeping the opponent centered, rather than closing distance.
    lure_retreat_speed: float = 0.5
    lure_turn_gain: float = 0.6          # how hard to steer while retreating-and-centering
    lure_max_cycles: int = 30            # give up luring and counter-charge anyway after this long
    # Distance at which the opponent is considered "committed"/overextended -> counter.
    bait_taken_range_m: float = 0.6
    # Soft edge-repulsion blended continuously into the lure steer, using live IR
    # readings - NOT a privileged-position center-seek (this controller only ever
    # sees the same sensor obs any opponent_policy gets). Spawning off-center means
    # a pure straight-line reverse can beeline toward this controller's OWN edge;
    # this bias curves the retreat away from whichever IR probe is heating up
    # fastest, well before the hard edge_avoid override would trigger.
    lure_edge_repulsion_start: float = 0.5   # IR reading above which repulsion starts blending in
    lure_edge_repulsion_gain: float = 1.2

    # --- counter phase ---
    counter_speed: float = 1.0
    counter_cycles: int = 8              # commit to the charge for this many cycles once triggered
    # Diagnosed from a real trace: once locked in contact (min_dist stays <=
    # bait_taken_range_m indefinitely), counter_timer expiring just re-arms counter
    # forever with no exit - a mutual push-of-war stalemate neither side breaks,
    # running out the episode clock as a draw every time. These two params detect
    # that pattern and force a genuine disengage instead of recommitting to a push
    # that's already proven not to resolve.
    #
    # UPDATE after 500-episode evidence: threshold was originally 3, too tight.
    # rule_based's commit-charge (commit_cycles=10) sustains pressure uninterrupted
    # once armed; at 3 x counter_cycles=8=24 total cycles before a forced 15-cycle
    # retreat, BaitController was breaking off contact before a genuinely winning
    # push could complete. Result: 0/14 losses in a 500-episode run were
    # "pushed_out" - all 14 were "capsized" instead, meaning the counter strategy
    # had never once demonstrably won by actually finishing a push. Raised to 10
    # to give real pushes more room before the stuck-detector intervenes - still
    # not a literal never-give-up (a true infinite stalemate still eventually
    # breaks), just less trigger-happy about it. If "pushed_out" wins still don't
    # appear after this change, the fix needs to be smarter than a fixed cycle
    # count - e.g. consuming obs["encoder"] to distinguish "wheels spinning
    # without net forward progress" (genuine stalemate) from "still advancing"
    # (working push), which BaitController doesn't currently look at at all.
    counter_max_consecutive_triggers: int = 10
    forced_disengage_cycles: int = 15           # cycles of forced retreat, ignoring bait_taken_range_m

    # --- patrol (center-seeking, not spin-search) ---
    patrol_speed: float = 0.4
    patrol_turn_bias: float = 0.2

    # --- edge avoidance (safety override, same semantics as rule-based) ---
    ir_edge_threshold: float = 0.85
    reverse_speed: float = 0.9
    turn_speed: float = 0.7
    reverse_cycles: int = 3
    pivot_cycles: int = 4


@dataclass
class BaitController:
    """Stateful controller. Same call signature/usage pattern as RuleBasedController -
    construct via make_bait_controller_policy(), call .reset() between episodes."""

    params: BaitControllerParams = field(default_factory=BaitControllerParams)
    state: BaitState = BaitState.PATROL
    _lure_timer: int = 0
    _counter_timer: int = 0
    _counter_retrigger_count: int = 0
    _forced_disengage_timer: int = 0
    _edge_phase: str = "idle"
    _edge_timer: int = 0
    _pivot_dir: float = 1.0
    _patrol_dir: float = 1.0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.state = BaitState.PATROL
        self._lure_timer = 0
        self._counter_timer = 0
        self._counter_retrigger_count = 0
        self._forced_disengage_timer = 0
        self._edge_phase = "idle"
        self._edge_timer = 0
        self._pivot_dir = 1.0
        self._patrol_dir = 1.0

    def __call__(self, obs: dict[str, np.ndarray]) -> tuple[float, float]:
        tof = np.asarray(obs["tof"], dtype=np.float32)
        ir = np.asarray(obs["ir"], dtype=np.float32)
        p = self.params

        # Safety override, same priority as rule-based - this controller is testing
        # strategic generalization, not whether it can be goaded off the edge.
        if self._edge_phase != "idle" or bool(np.any(ir >= p.ir_edge_threshold)):
            self._lure_timer = 0
            self._counter_timer = 0
            return self._handle_edge(ir)

        detected, offset, min_dist = self._detect(tof)

        # Already committed to the counter-charge: keep charging for counter_cycles
        # regardless of momentary detection dropout (same "commit" idea as rule-based,
        # applied to the counter-charge instead of the initial approach).
        if self._counter_timer > 0:
            self._counter_timer -= 1
            self.state = BaitState.COUNTER
            return (p.counter_speed, p.counter_speed)

        # Forced disengage window: a stalemate was just detected below. Retreat
        # for real, ignoring bait_taken_range_m, so the two don't immediately
        # re-lock at the same standoff distance next cycle.
        if self._forced_disengage_timer > 0:
            self._forced_disengage_timer -= 1
            self.state = BaitState.LURE
            return self._lure(offset if detected else 0.0, ir)

        if not detected:
            self.state = BaitState.PATROL
            self._lure_timer = 0
            self._counter_retrigger_count = 0
            return self._patrol()

        # Opponent took the bait (closed to point-blank) or we've lured long enough -
        # snap to the counter-charge while it's overextended.
        if min_dist <= p.bait_taken_range_m or self._lure_timer >= p.lure_max_cycles:
            self._counter_retrigger_count += 1
            if self._counter_retrigger_count > p.counter_max_consecutive_triggers:
                # counter_timer has now expired and immediately re-armed itself
                # this many times in a row without ever genuinely disengaging -
                # a straight-line push isn't resolving this contact. Break the
                # loop instead of recommitting to the same stalemate again.
                self._forced_disengage_timer = p.forced_disengage_cycles
                self._counter_retrigger_count = 0
                self.state = BaitState.LURE
                return self._lure(offset, ir)
            self._counter_timer = p.counter_cycles
            self.state = BaitState.COUNTER
            return (p.counter_speed, p.counter_speed)

        # Otherwise: lure. Retreat while keeping the opponent centered, so it keeps
        # closing distance on what looks like an open shot.
        self.state = BaitState.LURE
        self._lure_timer += 1
        self._counter_retrigger_count = 0
        return self._lure(offset, ir)

    # -- detection (same convention as rule_based_controller) -------------------

    def _detect(self, tof: np.ndarray) -> tuple[bool, float, float]:
        n = tof.shape[0]
        min_idx = int(np.argmin(tof))
        min_dist = float(tof[min_idx])
        if min_dist >= self.params.detect_range_m:
            return False, 0.0, min_dist
        center = (n - 1) / 2.0
        offset = (min_idx - center) / center if center > 0 else 0.0
        return True, float(offset), min_dist

    # -- lure ---------------------------------------------------------------------

    def _lure(self, offset: float, ir: np.ndarray) -> tuple[float, float]:
        p = self.params
        # Base retreat: reverse while steering to keep the opponent centered.
        center_turn = p.lure_turn_gain * offset if abs(offset) > p.center_band else 0.0

        # Soft repulsion: if either IR probe is heating up (approaching THIS
        # controller's own edge), bias the turn away from that side, scaled by how
        # far past the start threshold it is. This blends in gradually well before
        # the hard edge_avoid override would fire, so lure curves away from the
        # boundary instead of beelining into it and bouncing between LURE and
        # EDGE_AVOID.
        fl, fr = float(ir[0]), float(ir[1])
        repulsion = 0.0
        if fl > p.lure_edge_repulsion_start:
            repulsion -= p.lure_edge_repulsion_gain * (fl - p.lure_edge_repulsion_start)
        if fr > p.lure_edge_repulsion_start:
            repulsion += p.lure_edge_repulsion_gain * (fr - p.lure_edge_repulsion_start)

        turn = center_turn + repulsion
        left = -p.lure_retreat_speed - turn
        right = -p.lure_retreat_speed + turn
        return (float(np.clip(left, -1.0, 1.0)), float(np.clip(right, -1.0, 1.0)))

    # -- patrol (center-seeking, not spin-search) ----------------------------------

    def _patrol(self) -> tuple[float, float]:
        p = self.params
        # Gentle arcing forward drift rather than rule-based's alternating
        # spin/creep - deliberately a different search topology, not just different
        # numbers, so nothing about this controller is a parameter-jitter of
        # rule-based even incidentally.
        bias = p.patrol_turn_bias * self._patrol_dir
        return (
            float(np.clip(p.patrol_speed + bias, -1.0, 1.0)),
            float(np.clip(p.patrol_speed - bias, -1.0, 1.0)),
        )

    # -- edge avoidance (same mechanics as rule_based_controller) ------------------

    def _handle_edge(self, ir: np.ndarray) -> tuple[float, float]:
        p = self.params
        if self._edge_phase == "idle":
            self.state = BaitState.EDGE_AVOID
            fl, fr = float(ir[0]), float(ir[1])
            self._pivot_dir = 1.0 if fl >= fr else -1.0
            self._edge_phase = "reverse"
            self._edge_timer = p.reverse_cycles

        if self._edge_phase == "reverse":
            self._edge_timer -= 1
            if self._edge_timer <= 0:
                self._edge_phase = "pivot"
                self._edge_timer = p.pivot_cycles
            return (-p.reverse_speed, -p.reverse_speed)

        if self._edge_phase == "pivot":
            self._edge_timer -= 1
            if self._edge_timer <= 0:
                self._edge_phase = "idle"
                self._patrol_dir = self._pivot_dir
            return (p.turn_speed * self._pivot_dir, -p.turn_speed * self._pivot_dir)

        self._edge_phase = "idle"
        return (0.0, 0.0)


def make_bait_controller_policy(params: BaitControllerParams | None = None):
    """Factory matching make_rule_based_policy()'s convention - returns a fresh,
    correctly-reset stateful controller usable as an env opponent_policy."""
    controller = BaitController(params=params or BaitControllerParams())
    controller.reset()
    return controller