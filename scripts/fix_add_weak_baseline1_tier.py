"""One-shot fix for scripts/run_phase4_pilot.py.

Adds a new, deliberately weakened opponent tier: baseline1_weak. Purpose: SBSO's
whole design premise is to discover and exploit opponent weaknesses via
MCTS/Judge/DSPy. Standard-strength baseline1 (a genuinely competent, hand-tuned
rule-based archetype - commitment latch + predictive lead-tracking) has so far
shown 0% win rate for the agent even after fixing the frozen-instance bug and the
draw-default env-truncation bug, with DSPy trending TOWARD more passive (hold-heavy)
behavior across recompiles rather than converging on an exploit. Before concluding
the optimization loop itself isn't working, this gives it a genuinely easier target
to test the mechanism against directly.

weak_base deliberately weakens exactly the mechanisms identified as making
baseline1 competent: slower charge_speed (was already at its 1.0 ceiling -
reduced directly, not multiplicatively), shorter detect_range_m (sees the agent
later, less time to react), reduced turn_speed.

This is ADDITIVE ONLY - baseline1/baseline2/self_checkpoint's existing logic is
completely untouched, so nothing already confirmed working is at risk. Use
opponent_type="baseline1_weak" only via a dedicated diagnostic config, not the
real training configs.

Run once on the pod, from the repo root:
    python fix_add_weak_baseline1_tier.py
"""

from pathlib import Path

target = Path("scripts/run_phase4_pilot.py")
content = target.read_text()

old_import = "from src.baselines.rule_based_controller import make_randomized_opponent_factory"
new_import = (
    "from src.baselines.rule_based_controller import (\n"
    "    make_randomized_opponent_factory, RuleBasedParams,\n"
    ")"
)

old_factory = '''    def factory(opp_type: str):
        if "rb_factory" not in _cache:
            _cache["rb_factory"] = make_randomized_opponent_factory(seed=0, spread_multiplier=2.0)

        if opp_type == "baseline1":
            return _cache["rb_factory"]()
        if opp_type == "baseline2":'''

new_factory = '''    def factory(opp_type: str):
        import os
        # DIAGNOSTIC OVERRIDE: OpponentPool.sample() hardcodes choices=["baseline1",
        # "baseline2"] internally (see opponent_pool.py) - no config key actually
        # controls this, so opponent_pool.pilot_scope in a YAML does NOT restrict
        # sampling. This env-var redirect is the safe way to force every episode to
        # the weak tier for a diagnostic run, WITHOUT editing OpponentPool.sample()
        # itself (a well-established, working file - not touching it deliberately).
        # Only active if SBSO_FORCE_WEAK_OPPONENT=1 is set; a normal run is
        # completely unaffected.
        if os.environ.get("SBSO_FORCE_WEAK_OPPONENT") == "1":
            opp_type = "baseline1_weak"

        if "rb_factory" not in _cache:
            _cache["rb_factory"] = make_randomized_opponent_factory(seed=0, spread_multiplier=2.0)

        if opp_type == "baseline1":
            return _cache["rb_factory"]()
        if opp_type == "baseline1_weak":
            # Diagnostic-only tier: deliberately weakened base instance, testing
            # whether SBSO's exploit-discovery mechanism shows real improvement
            # against an easier target. NOT used by any real training config -
            # standard baseline1 above is unaffected and remains the real
            # RQ1/RQ2 comparison baseline.
            if "rb_weak_factory" not in _cache:
                weak_base = RuleBasedParams(
                    detect_range_m=RuleBasedParams().detect_range_m * 0.6,
                    charge_speed=0.6,
                    turn_speed=RuleBasedParams().turn_speed * 0.7,
                )
                _cache["rb_weak_factory"] = make_randomized_opponent_factory(
                    seed=1, base=weak_base, spread_multiplier=1.0,
                )
            return _cache["rb_weak_factory"]()
        if opp_type == "baseline2":'''

checks = [
    ("import", old_import, new_import),
    ("factory body", old_factory, new_factory),
]

for label, old, new in checks:
    if old not in content:
        print(f"ERROR: {label} anchor not found - file may differ from what this script expects.")
        raise SystemExit(1)
    content = content.replace(old, new)

target.write_text(content)
print("Both replacements applied successfully.")

import ast
ast.parse(target.read_text())
print("Syntax check passed - file parses correctly.")
