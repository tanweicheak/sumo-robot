"""One-shot follow-up fix for scripts/run_phase4_pilot.py.

DEPENDS ON fix_baseline1_randomization.py already being applied (needs its
make_randomized_opponent_factory(seed=0) call as an anchor). Run that one first if
you haven't.

Opts SBSO training specifically into a wider randomization spread (2.0x the default
+/-20%/+/-10%/+/-50% ranges), per rule_based_controller.py's new spread_multiplier
parameter (see fix_rule_based_spread_multiplier.py - apply that one too, this won't
do anything without it). PPO's training (run_phase1_baselines.py) and any evaluation
usage of RuleBasedParams.randomized() are untouched - this only changes the
SBSO-training-specific call site.

2.0 is a reasoned starting point, not a validated optimum - it's bounded (e.g.
lead_gain's range becomes 0.0 to 2x base, still can't go negative or unbounded) but
its actual effect on win rate is not something that can be predicted in advance.
Treat as a first attempt, not a final tuned value.

Run once on the pod, from the repo root, AFTER fix_baseline1_randomization.py:
    python fix_widen_training_spread.py
"""

from pathlib import Path

target = Path("scripts/run_phase4_pilot.py")
content = target.read_text()

old = 'make_randomized_opponent_factory(seed=0)'
new = 'make_randomized_opponent_factory(seed=0, spread_multiplier=2.0)'

count = content.count(old)
if count == 0:
    print("ERROR: anchor 'make_randomized_opponent_factory(seed=0)' not found.")
    print("Has fix_baseline1_randomization.py been applied to this file yet? Run that first.")
    raise SystemExit(1)

content = content.replace(old, new)
target.write_text(content)
print(f"Applied: widened spread_multiplier=2.0 at {count} call site(s).")

import ast
ast.parse(target.read_text())
print("Syntax check passed - file parses correctly.")
