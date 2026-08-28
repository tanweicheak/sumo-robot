"""One-shot fix for src/evaluation/match_runner.py's SLMPolicyOpponent.

Adds self._last_strategy, set inside __call__ right after SA.decide() runs, so
run_phase5_eval.py's new strategies_chosen logging (getattr(attacker_policy,
"_last_strategy", None)) actually has something real to read, instead of silently
logging None for every decision. Purely additive - one new line inside __call__,
does not change any existing behavior or return value.

Run once on the pod, from the repo root:
    python fix_match_runner_expose_strategy.py
"""

from pathlib import Path

target = Path("src/evaluation/match_runner.py")
content = target.read_text()

old = "        macro = self.sa.decide(perception, state.prev_opponent_analysis)"
new = (
    "        macro = self.sa.decide(perception, state.prev_opponent_analysis)\n"
    "        self._last_strategy = macro.strategy   # exposed for external logging -\n"
    "                                                 # see scripts/run_phase5_eval.py"
)

if old not in content:
    print("ERROR: anchor not found - paste current SLMPolicyOpponent.__call__ content, do not guess.")
    raise SystemExit(1)

content = content.replace(old, new)
target.write_text(content)
print("Fix applied.")

import ast
ast.parse(target.read_text())
print("Syntax OK.")
