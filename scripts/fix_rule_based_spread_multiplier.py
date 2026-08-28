"""One-shot fix for src/baselines/rule_based_controller.py.

Adds spread_multiplier (default 1.0, fully backward-compatible) to
RuleBasedParams.randomized() and make_randomized_opponent_factory(). Every existing
caller (run_phase1_baselines.py's PPO training, and the SBSO training fix already
applied to run_phase4_pilot.py) is unaffected unless it explicitly passes a non-1.0
value - the jitter spreads (0.20/0.10/0.5) are unchanged as defaults, just no longer
hardcoded as the ONLY option.

Why: randomization alone (the prior fix) removes overfitting to one frozen instance,
but does not guarantee the agent can beat this opponent archetype in general - the
jitter is deliberately narrow ("recognizably the same simple strategy... not a
different strategy class" per this file's own docstring). Widening the spread
SPECIFICALLY for SBSO training (not for PPO, not for evaluation baselines, both of
which should keep the standard, narrower range as their reference behavior) gives the
training loop a real chance to encounter and learn from genuinely weaker instances of
the same archetype, alongside the standard-strength ones - closer to a domain-
randomization curriculum than a single-difficulty target.

Run once on the pod, from the repo root:
    python fix_rule_based_spread_multiplier.py
"""

from pathlib import Path

target = Path("src/baselines/rule_based_controller.py")
content = target.read_text()

old_randomized = '''    @classmethod
    def randomized(cls, rng: random.Random, base: "RuleBasedParams | None" = None) -> "RuleBasedParams":
        """Jittered variant for opponent-diversity training (e.g. PPO baseline curriculum) -
        NOT used by default; wire this into the training script's env-reset opponent
        factory to stop PPO from overfitting to one frozen instance. Jitters within
        +/-20% of the base values (default RuleBasedParams() if none given) so each
        sampled instance is still recognizably "the same simple strategy," just a
        different competence/aggressiveness level - not a different strategy class."""
        base = base or cls()

        def jitter(value: float, spread: float = 0.20) -> float:
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
        )'''

new_randomized = '''    @classmethod
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
        )'''

old_factory_fn = '''def make_randomized_opponent_factory(seed: int = 0, base: RuleBasedParams | None = None):
    """Returns a zero-arg callable that produces a FRESH randomized RuleBasedController
    each time it's called - drop this into the PPO training script's env-reset hook
    (e.g. `opponent_policy = opponent_factory()` inside a VecEnv reset callback) so PPO
    trains against a variety of rule-based competence levels instead of one frozen
    instance. Not wired into any training script here - scripts/run_phase1_baselines.py
    (not present in this project upload) is where that wiring belongs."""
    rng = random.Random(seed)

    def _factory() -> RuleBasedController:
        controller = RuleBasedController(params=RuleBasedParams.randomized(rng, base))
        controller.reset()
        return controller

    return _factory'''

new_factory_fn = '''def make_randomized_opponent_factory(
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

    return _factory'''

for label, old, new in [("randomized() classmethod", old_randomized, new_randomized),
                         ("make_randomized_opponent_factory()", old_factory_fn, new_factory_fn)]:
    if old not in content:
        print(f"ERROR: {label} anchor not found - file may differ from what this script expects.")
        raise SystemExit(1)
    content = content.replace(old, new)

target.write_text(content)
print("Both replacements applied successfully.")

import ast
ast.parse(target.read_text())
print("Syntax check passed - file parses correctly.")
