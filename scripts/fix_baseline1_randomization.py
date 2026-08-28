"""One-shot fix for scripts/run_phase4_pilot.py's opponent factory.

Bug: make_rule_based_policy() was called with NO arguments at all three sites below,
meaning it defaults to RuleBasedParams() - a single fixed parameter set, IDENTICAL
every episode for the entire run. This is exactly the anti-pattern
rule_based_controller.py's own module docstring warns against ("training against ONE
frozen instance... teaches a narrow counter-strategy specific to that instance, not
general pushing competence") - a lesson the codebase already applied correctly for
PPO's training (run_phase1_baselines.py uses RuleBasedParams.randomized() per
episode) but never carried over to this file.

Confirmed as the real, direct cause of a 0% win rate against baseline1 (0/157) and
self_checkpoint (0/106, which shares this exact call via the fallback branch) in a
completed real training run, versus 100% (159/159) against baseline2 (PPO, which
already trains against randomized opponents). The fixed rule-based opponent has a
genuinely strong, deterministic mechanism (a commitment latch plus predictive
lead-tracking) that the training loop never had varied instances of to generalize
against.

Fix: use make_randomized_opponent_factory() (already implemented and correct in
rule_based_controller.py, just never called from this file) so each episode against
baseline1 or the self_checkpoint fallback gets a freshly-sampled, jittered instance -
same mechanism, varying competence level, matching PPO's already-established pattern.
seed=0 explicitly passed for reproducibility (matches the function's own default,
made explicit rather than implicit).

Run once on the pod, from the repo root:
    python fix_baseline1_randomization.py
"""

from pathlib import Path

target = Path("scripts/run_phase4_pilot.py")
content = target.read_text()

# --- Fix 1: import ------------------------------------------------------------------

old_import = "from src.baselines.rule_based_controller import make_rule_based_policy"
new_import = (
    "from src.baselines.rule_based_controller import (\n"
    "    make_rule_based_policy, make_randomized_opponent_factory,\n"
    ")"
)

# --- Fix 2: the opponent factory's baseline1 + self_checkpoint-fallback branches ----

old_factory = '''    def factory(opp_type: str):
        if opp_type == "baseline1":
            return make_rule_based_policy()
        if opp_type == "baseline2":
            if "ppo" not in _ppo_cache:
                _ppo_cache["ppo"] = PPOController.load(
                    "checkpoints/baseline2_ppo/ppo_baseline2.zip",
                    "checkpoints/baseline2_ppo/vecnormalize.pkl",
                )
            return _ppo_cache["ppo"]
        # self_checkpoint would go here in the full run; pilot excludes it.
        return make_rule_based_policy()'''

new_factory = '''    def factory(opp_type: str):
        # Reused across baseline1 and the self_checkpoint fallback below - both were
        # calling make_rule_based_policy() with no args (a single FIXED instance,
        # identical every episode) until this fix. _ppo_cache's name predates this
        # addition; kept as-is to minimize diff surface rather than renamed.
        if "rb_factory" not in _ppo_cache:
            _ppo_cache["rb_factory"] = make_randomized_opponent_factory(seed=0)

        if opp_type == "baseline1":
            return _ppo_cache["rb_factory"]()
        if opp_type == "baseline2":
            if "ppo" not in _ppo_cache:
                _ppo_cache["ppo"] = PPOController.load(
                    "checkpoints/baseline2_ppo/ppo_baseline2.zip",
                    "checkpoints/baseline2_ppo/vecnormalize.pkl",
                )
            return _ppo_cache["ppo"]
        # self_checkpoint lands here (real match continuation still falls back to
        # rule-based - see opponent_pool.py's documented limitation, NOT changed by
        # this fix). Now at least randomized rather than a second frozen instance
        # identical to baseline1's.
        return _ppo_cache["rb_factory"]()'''

# --- Fix 3: env construction-time default (provably overwritten every episode by the
# training loop before any real match happens - low stakes, fixed for consistency) ---

old_env_default = 'env = PyBulletSumoEnv(env_config=env_cfg, opponent_policy=make_rule_based_policy())'
new_env_default = (
    'env = PyBulletSumoEnv(\n'
    '        env_config=env_cfg,\n'
    '        opponent_policy=make_randomized_opponent_factory(seed=0)(),\n'
    '        # placeholder only - overwritten every episode by the training loop\n'
    '        # before any real match happens; randomized here too for consistency.\n'
    '    )'
)

checks = [
    ("import line", old_import, new_import),
    ("opponent factory (baseline1 + self_checkpoint fallback)", old_factory, new_factory),
    ("env construction-time default", old_env_default, new_env_default),
]

for label, old, new in checks:
    if old not in content:
        print(f"ERROR: {label} anchor not found - file may differ from what this script expects.")
        print("Do not hand-edit; paste the current file section back to Claude instead.")
        raise SystemExit(1)
    content = content.replace(old, new)

target.write_text(content)
print("All 3 replacements applied successfully.")

import ast
ast.parse(target.read_text())
print("Syntax check passed - file parses correctly.")
