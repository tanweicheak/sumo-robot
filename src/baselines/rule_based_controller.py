"""
src.baselines.rule_based_controller

Phase: Phase 1
Purpose: Baseline 1 - hardcoded if-else rule-based controller, the conventional
    competition lower-bound reference. Implements the universal sumo priority stack:
        1. EDGE AVOIDANCE (highest priority) - reverse + pivot away from the ring edge
        2. ATTACK - drive at a detected opponent, steering to keep it centered (with a
           simple ray-drift lead term - see _attack)
        3. SEARCH (lowest priority) - alternates spin-scan with forward creep so the
           robot actually covers ground while searching, instead of spinning in place
           forever (see _search)
    Consumes only the raw three-channel sensor snapshot (ToF / IR / encoder) and
    emits normalized (left_pwm, right_pwm) in [-1, 1], matching the env's
    opponent_policy signature so it drops in as agent controller or opponent.

    Kept deliberately as a simple, legible deterministic state machine - this is meant
    to stay a fair, conventional lower-bound baseline, not accumulate adaptive/learned
    sophistication. The two fixes above are weaknesses in EXECUTING that simple
    strategy (standing still while "searching"; steering at stale opponent position),
    not additions of strategic complexity.

    RuleBasedParams.randomized() supports opponent-diversity training (e.g. for PPO,
    Baseline 2): training against ONE frozen instance with fixed default parameters
    teaches a narrow counter-strategy specific to that instance, not general pushing
    competence. Sampling a fresh randomized instance per training episode is the
    standard fix - the training script (scripts/run_phase1_baselines.py) is where this
    should actually get wired in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RuleState(str, Enum):
    SEARCH = "search"
    ATTACK = "attack"
    EDGE_AVOID = "edge_avoid"


@dataclass
class RuleBasedParams:
    # --- detection thresholds ---
    # ToF distance (m) below which the opponent is considered "detected" in a ray.
    detect_range_m: float = 1.4
    # A center ray this close counts as "dead ahead" -> full charge.
    attack_charge_range_m: float = 1.2

    # --- edge detection ---
    # IR analog reading in [0, 1]; >= this on a front probe means the platform has
    # dropped away beneath that corner (approaching / over the white edge).
    ir_edge_threshold: float = 0.85

    # --- motor levels (normalized PWM) ---
    charge_speed: float = 1.0
    turn_speed: float = 0.7
    search_spin_speed: float = 0.55
    reverse_speed: float = 0.9

    # --- edge-avoidance maneuver timing (in control cycles) ---
    reverse_cycles: int = 3
    pivot_cycles: int = 4

    # Steering deadzone: if the opponent's angular offset (in ray indices from
    # center, normalized) is within this band, treat it as centered -> charge.
    center_band: float = 0.15

    # Commitment latch: after acquiring the opponent at close range, keep charging
    # forward for this many cycles even if detection drops (point-blank ToF blackout).
    commit_cycles: int = 10
    # ToF below this counts as "closed to contact" -> arm the commitment latch.
    commit_arm_range_m: float = 0.35
    commit_charge_speed: float = 0.6  # slower than full charge during blind commit

    # --- search coverage (fix: was pure stationary spin) ---
    search_creep_speed: float = 0.45      # forward speed while creeping between scans
    search_spin_cycles: int = 8           # scan-in-place duration per phase
    search_creep_cycles: int = 6          # forward-creep duration per phase

    # --- attack lead prediction (fix: was pure reactive centering on stale position) ---
    # Multiplies the opponent's frame-to-frame ray-offset drift and adds it to the
    # current offset before steering, so the robot steers toward where the opponent is
    # heading rather than only where it was last seen. 0.0 reproduces the old
    # pure-reactive behavior exactly.
    lead_gain: float = 0.6

    @classmethod
    def randomized(
        cls, rng: random.Random, base: "RuleBasedParams | None" = None,
        spread_multiplier: float = 1.0,
    ) -> "RuleBasedParams":
        """Jittered variant for opponent-diversity training (e.g. PPO baseline curriculum) -
        NOT used by default; wire this into the training script's env-reset opponent
        factory to stop PPO from overfitting to one frozen instance. Jitters within
        +/-20% of the base values (default RuleBasedParams() if none given) so each
        sampled instance is still recognizably "the same simple strategy," just a
        different competence/aggressiveness level - not a different strategy class.

        spread_multiplier: scales every jitter spread uniformly. Default 1.0
        reproduces the exact original +/-20%/+/-10%/+/-50% ranges for every existing
        caller - added for SBSO training specifically (see run_phase4_pilot.py),
        which opts into a wider range for genuine difficulty diversity; PPO's
        training and any evaluation usage of this function are unaffected unless
        they explicitly pass a non-1.0 value themselves. At spread_multiplier=2.0,
        lead_gain's effective range covers 0.0 (pure reactive, the weakest variant
        this archetype supports per this file's own note) up to 2x base - a real,
        bounded difficulty range, not unbounded."""
        base = base or cls()

        def jitter(value: float, spread: float = 0.20) -> float:
            spread = spread * spread_multiplier
            return value * rng.uniform(1.0 - spread, 1.0 + spread)

        return cls(
            detect_range_m=jitter(base.detect_range_m),
            attack_charge_range_m=jitter(base.attack_charge_range_m),
            ir_edge_threshold=min(0.99, jitter(base.ir_edge_threshold, 0.10)),
            charge_speed=min(1.0, jitter(base.charge_speed)),
            turn_speed=min(1.0, jitter(base.turn_speed)),
            search_spin_speed=min(1.0, jitter(base.search_spin_speed)),
            reverse_speed=min(1.0, jitter(base.reverse_speed)),
            center_band=max(0.02, jitter(base.center_band)),
            commit_cycles=max(1, int(round(jitter(base.commit_cycles)))),
            commit_arm_range_m=jitter(base.commit_arm_range_m),
            commit_charge_speed=min(1.0, jitter(base.commit_charge_speed)),
            search_creep_speed=min(1.0, jitter(base.search_creep_speed)),
            search_spin_cycles=max(1, int(round(jitter(base.search_spin_cycles)))),
            search_creep_cycles=max(1, int(round(jitter(base.search_creep_cycles)))),
            lead_gain=max(0.0, jitter(base.lead_gain, 0.5)),
        )


@dataclass
class RuleBasedController:
    """Deterministic controller. Holds a small amount of state for the timed
    edge-avoidance maneuver, the search spin/creep cycle, and attack lead prediction."""

    params: RuleBasedParams = field(default_factory=RuleBasedParams)
    state: RuleState = RuleState.SEARCH
    _edge_phase: str = "idle"      # "idle" | "reverse" | "pivot"
    _edge_timer: int = 0
    _pivot_dir: float = 1.0         # +1 pivot right, -1 pivot left
    _search_dir: float = 1.0        # spin direction while searching
    _commit_timer: int = 0
    _search_phase: str = "spin"     # "spin" | "creep"
    _search_timer: int = 0
    _prev_offset: float | None = None   # last attack offset, for lead prediction

    def __post_init__(self) -> None:
        # The dataclass field defaults above (_search_timer=0 especially) are NOT a
        # valid standalone state on their own - reset() is what actually computes the
        # real starting values (e.g. _search_timer = params.search_spin_cycles, not 0).
        # Calling it here means ANY construction path (direct, or via
        # make_rule_based_policy()) ends up correctly initialized, instead of relying
        # on the caller to remember a separate .reset() call - a bare
        # RuleBasedController() used to start with _search_timer=0, which made the very
        # first _search() call immediately "expire" the spin phase and jump straight to
        # creep, instead of genuinely starting in spin as intended.
        self.reset()

    def reset(self) -> None:
        self.state = RuleState.SEARCH
        self._edge_phase = "idle"
        self._edge_timer = 0
        self._pivot_dir = 1.0
        self._search_dir = 1.0
        self._commit_timer = 0
        self._search_phase = "spin"
        self._search_timer = self.params.search_spin_cycles
        self._prev_offset = None
    # -- main entry point: env opponent_policy signature -----------------------

    def __call__(self, obs: dict[str, np.ndarray]) -> tuple[float, float]:
        tof = np.asarray(obs["tof"], dtype=np.float32)
        ir = np.asarray(obs["ir"], dtype=np.float32)

        # PRIORITY 1: edge avoidance (safety) always overrides, and cancels commit.
        if self._edge_phase != "idle" or self._edge_detected(ir):
            self._commit_timer = 0
            return self._handle_edge(ir)

        # PRIORITY 2: attack.
        detected, offset, min_dist = self._detect_opponent(tof)

        if detected:
            self.state = RuleState.ATTACK
            # Arm/refresh the commitment latch once we've closed to contact range,
            # so a point-blank detection dropout doesn't make us stop pushing.
            if min_dist <= self.params.commit_arm_range_m:
                self._commit_timer = self.params.commit_cycles
            action = self._attack(offset, min_dist)
            self._prev_offset = offset
            return action

        # Detection dropped. If committed (just been ramming at close range), keep
        # charging straight through the ToF blackout instead of reverting to search.
        if self._commit_timer > 0:
            self._commit_timer -= 1
            self.state = RuleState.ATTACK
            return (self.params.commit_charge_speed, self.params.commit_charge_speed)

        # PRIORITY 3: search.
        self.state = RuleState.SEARCH
        self._prev_offset = None   # lost contact - no stale drift signal to carry over
        return self._search()

    # -- edge avoidance --------------------------------------------------------

    def _edge_detected(self, ir: np.ndarray) -> bool:
        return bool(np.any(ir >= self.params.ir_edge_threshold))

    def _handle_edge(self, ir: np.ndarray) -> tuple[float, float]:
        p = self.params
        # Freshly triggered: decide pivot direction away from the endangered side
        # and enter the reverse phase.
        if self._edge_phase == "idle":
            self.state = RuleState.EDGE_AVOID
            fl, fr = float(ir[0]), float(ir[1])
            # If the left corner is over the edge, pivot right (away from left), and
            # vice versa. Tie -> pivot right by convention.
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
                # Flip next search direction so a re-trigger doesn't loop forever.
                self._search_dir = self._pivot_dir
            # Pivot in place: wheels counter-rotate.
            return (p.turn_speed * self._pivot_dir, -p.turn_speed * self._pivot_dir)

        # Fallback (should not reach).
        self._edge_phase = "idle"
        return (0.0, 0.0)

    # -- attack ----------------------------------------------------------------

    def _detect_opponent(self, tof: np.ndarray) -> tuple[bool, float, float]:
        """Return (detected, normalized_offset, min_distance).

        offset is in [-1, 1]: negative = opponent toward the left rays, positive =
        right rays, 0 = dead center. Derived from which ray sees the nearest hit.
        """
        n = tof.shape[0]
        min_idx = int(np.argmin(tof))
        min_dist = float(tof[min_idx])
        if min_dist >= self.params.detect_range_m:
            return False, 0.0, min_dist
        center = (n - 1) / 2.0
        offset = (min_idx - center) / center if center > 0 else 0.0
        return True, float(offset), min_dist

    def _attack(self, offset: float, min_dist: float) -> tuple[float, float]:
        p = self.params
        # Lead prediction fix: steer toward where the opponent is DRIFTING, not only
        # its current ray position. drift > 0 means it moved toward the right rays
        # since the last frame; project that drift forward by lead_gain so the robot
        # starts converging on an intercept course instead of chasing a stale bearing.
        if self._prev_offset is not None:
            drift = offset - self._prev_offset
            effective_offset = float(np.clip(offset + p.lead_gain * drift, -1.0, 1.0))
        else:
            effective_offset = offset

        # Opponent roughly centered (using the lead-adjusted offset): full straight charge.
        if abs(effective_offset) <= p.center_band:
            return (p.charge_speed, p.charge_speed)
        # Opponent off to one side: steer toward it. offset<0 -> turn left
        # (slow left wheel); offset>0 -> turn right (slow right wheel).
        if effective_offset < 0:
            return (p.charge_speed * (1.0 - abs(effective_offset)), p.charge_speed)
        return (p.charge_speed, p.charge_speed * (1.0 - abs(effective_offset)))

    # -- search ----------------------------------------------------------------

    def _search(self) -> tuple[float, float]:
        """Fix: alternates spin-scan with forward creep so the robot actually covers
        ground while searching, instead of spinning in one spot indefinitely (the old
        behavior could never find an opponent outside its ToF cone unless the opponent
        happened to wander into it)."""
        p = self.params
        self._search_timer -= 1
        if self._search_timer <= 0:
            if self._search_phase == "spin":
                self._search_phase = "creep"
                self._search_timer = p.search_creep_cycles
            else:
                self._search_phase = "spin"
                self._search_timer = p.search_spin_cycles
                # Alternate spin direction each cycle so creep drift doesn't bias the
                # robot into one arc of the ring forever.
                self._search_dir *= -1.0

        if self._search_phase == "creep":
            # Forward creep with a slight turn bias, so the path arcs rather than
            # running in a dead-straight line toward a single point on the boundary.
            s = p.search_creep_speed
            bias = 0.15 * self._search_dir
            return (float(np.clip(s + bias, -1.0, 1.0)), float(np.clip(s - bias, -1.0, 1.0)))

        s = p.search_spin_speed * self._search_dir
        # Spin in place: wheels counter-rotate.
        return (s, -s)


def make_rule_based_policy(params: RuleBasedParams | None = None):
    """Factory returning a fresh stateful controller usable as an env
    opponent_policy or agent controller."""
    controller = RuleBasedController(params=params or RuleBasedParams())
    controller.reset()   # ensure _search_timer etc. start from a clean, valid state
    return controller


def make_randomized_opponent_factory(
    seed: int = 0, base: RuleBasedParams | None = None, spread_multiplier: float = 1.0,
):
    """Returns a zero-arg callable that produces a FRESH randomized RuleBasedController
    each time it's called - drop this into the PPO training script's env-reset hook
    (e.g. `opponent_policy = opponent_factory()` inside a VecEnv reset callback) so PPO
    trains against a variety of rule-based competence levels instead of one frozen
    instance. Not wired into any training script here - scripts/run_phase1_baselines.py
    (not present in this project upload) is where that wiring belongs.

    spread_multiplier: forwarded to RuleBasedParams.randomized() - default 1.0
    preserves original behavior for existing callers; see that method's docstring."""
    rng = random.Random(seed)

    def _factory() -> RuleBasedController:
        controller = RuleBasedController(
            params=RuleBasedParams.randomized(rng, base, spread_multiplier)
        )
        controller.reset()
        return controller

    return _factory