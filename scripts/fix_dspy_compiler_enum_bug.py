"""One-shot fix for src/sbso/dspy_compiler.py's validate_prompt_candidate function.

Bug: uses DirectionLabel.CENTER and MomentumLabel.STABLE, neither of which exist on
the real enums (confirmed against src/agents/schemas.py directly):
    DirectionLabel real members: FRONT_LEFT, FRONT_CENTER, FRONT_RIGHT, NONE
    MomentumLabel real members: FORWARD, REVERSE, TURNING, STILL
This is a pre-existing bug in validate_prompt_candidate that was never triggered
before tonight, because validate_recompiles defaulted to False - the function
literally never ran until this session's fix enabled it. Not something introduced
by tonight's changes; a latent bug in existing code, now exposed by exercising it
for the first time.

Targeted exact-match patch - does NOT touch or assume anything about the
SGLangNativeLM adapter already present in this file from earlier tonight.

Run once on the pod, from the repo root:
    python fix_dspy_compiler_enum_bug.py
"""

from pathlib import Path

target = Path("src/sbso/dspy_compiler.py")
content = target.read_text()

old = (
    'lssd_text=lssd_text, opp_distance=DistanceLabel.MID, opp_direction=DirectionLabel.CENTER,\n'
    '            edge=EdgeLabel.SAFE, momentum=MomentumLabel.STABLE, opp_distance_m=1.0,'
)
new = (
    'lssd_text=lssd_text, opp_distance=DistanceLabel.MID, opp_direction=DirectionLabel.FRONT_CENTER,\n'
    '            edge=EdgeLabel.SAFE, momentum=MomentumLabel.STILL, opp_distance_m=1.0,'
)

if old not in content:
    print("ERROR: exact broken text not found - file may differ from what this script expects.")
    print("Do not hand-edit; paste the current validate_prompt_candidate function back to Claude.")
    raise SystemExit(1)

content = content.replace(old, new)
target.write_text(content)
print("Fix applied: DirectionLabel.CENTER -> FRONT_CENTER, MomentumLabel.STABLE -> STILL")

import ast
ast.parse(target.read_text())
print("Syntax check passed - file parses correctly.")
