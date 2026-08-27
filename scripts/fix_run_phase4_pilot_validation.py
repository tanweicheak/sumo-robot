"""One-shot fix for scripts/run_phase4_pilot.py, applying two changes as exact
string matches (no manual retyping/indentation risk):

1. RE-APPLIES the opponent-pool self_checkpoint_manager fix that was lost when
   wandb code was reinstated from a separate session earlier tonight - confirmed
   via direct grep that line 125 still reads config["opponent_pool"]["pilot_scope"]
   and OpponentPool(...) still lacks self_checkpoint_manager/target_counts.

2. NEW: enables DSPy recompile validation (validate_prompt_candidate), which tonight's
   pilot run showed is currently OFF - prompt_history.jsonl showed 3/3 recompiles
   accepted unconditionally (validation: null) while rolling_winrate fell every time
   (0.667 -> 0.5 -> 0.444), with no check any recompile actually helped. Adds a
   standalone SGLangSLMClient (the plain client validate_prompt_candidate's
   StrategyAgent(client=...) needs - confirmed NOT the same as RealDSPyCompiler's
   internal SGLangNativeLM adapter, different interface) and passes
   slm_client=/validate_recompiles=True into MatchLevelSBSOTrainer.

Run once on the pod:
    python fix_run_phase4_pilot_validation.py
"""

from pathlib import Path

target = Path("scripts/run_phase4_pilot.py")
content = target.read_text()

# --- Fix 1: opponent-pool self-checkpoint fix, re-applied -----------------------

old_1a = '    opponent_factory = _make_opponent_factory(config["opponent_pool"]["pilot_scope"])'
new_1a = (
    '    # _make_opponent_factory only ever resolves baseline1/baseline2 into policies -\n'
    '    # a sampled self_checkpoint OpponentDescriptor already carries its own\n'
    '    # rollout_policy (see OpponentPool.sample()), so this scope is invariant\n'
    '    # between pilot and full run and does NOT come from config. Re-applied after\n'
    '    # being lost when wandb code was reinstated from a separate session.\n'
    '    opponent_factory = _make_opponent_factory(["baseline1", "baseline2"])'
)

old_1b = (
    '        opponent_pool=OpponentPool(\n'
    '            warmup_episodes=config["opponent_pool"]["warmup_episodes"],\n'
    '            total_episodes=episodes,\n'
    '        ),'
)
new_1b = (
    '        opponent_pool=OpponentPool(\n'
    '            warmup_episodes=config["opponent_pool"]["warmup_episodes"],\n'
    '            total_episodes=episodes,\n'
    '            self_checkpoint_manager=checkpoint_mgr,\n'
    '            target_counts=config["opponent_pool"].get("full_run_targets"),\n'
    '        ),'
)

# --- Fix 2: enable recompile validation ------------------------------------------

old_2a = '    opponent_factory = _make_opponent_factory(["baseline1", "baseline2"])'  # anchor AFTER fix 1a applied
new_2a = (
    '    opponent_factory = _make_opponent_factory(["baseline1", "baseline2"])\n\n'
    '    # Recompile validation: plain SGLangSLMClient (NOT dspy_compiler\'s internal\n'
    '    # SGLangNativeLM adapter - different interface, validate_prompt_candidate\'s\n'
    '    # StrategyAgent(client=slm_client) needs the generate_structured() contract).\n'
    '    # Same agent_server_url dspy_compiler already talks to, via the proven\n'
    '    # sglang_server.py client used everywhere else in this pipeline.\n'
    '    from src.inference.sglang_server import SGLangSLMClient\n'
    '    validation_slm_client = SGLangSLMClient(server_url=sg["agent_server_url"])'
)

old_2b = (
    '        checkpoint_mgr=checkpoint_mgr,\n'
    '        dspy_compiler=dspy_compiler,\n'
    '        strategies=STRATEGIES,'
)
new_2b = (
    '        checkpoint_mgr=checkpoint_mgr,\n'
    '        dspy_compiler=dspy_compiler,\n'
    '        slm_client=validation_slm_client,\n'
    '        validate_recompiles=True,\n'
    '        strategies=STRATEGIES,'
)

checks = [
    ("Fix 1a (opponent_factory line)", old_1a, new_1a),
    ("Fix 1b (OpponentPool construction)", old_1b, new_1b),
]

for label, old, new in checks:
    if old not in content:
        print(f"ERROR: {label} anchor not found - file may differ from what this script expects.")
        print("Do not hand-edit; paste the current file section back to Claude instead.")
        raise SystemExit(1)
    content = content.replace(old, new)

# Fix 2a's anchor is fix 1a's OUTPUT, so must run after the replace above
if old_2a not in content:
    print("ERROR: Fix 2a anchor not found after applying Fix 1 - unexpected state.")
    raise SystemExit(1)
content = content.replace(old_2a, new_2a)

if old_2b not in content:
    print("ERROR: Fix 2b anchor (checkpoint_mgr=.../dspy_compiler=... block) not found.")
    raise SystemExit(1)
content = content.replace(old_2b, new_2b)

target.write_text(content)
print("All 4 replacements applied successfully.")

import ast
ast.parse(target.read_text())
print("Syntax check passed - file parses correctly.")