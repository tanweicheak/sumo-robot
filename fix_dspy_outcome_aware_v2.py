"""One-shot fix for src/sbso/match_trainer.py and src/sbso/dspy_compiler.py.

Rebuilt fresh, anchored against real, current pod file content (confirmed via
direct grep before writing this - the original version of this fix was never
actually deployed).

Purpose: the weak-opponent diagnostic showed DSPy's exact-match _metric trains the
compiled prompt to imitate WHATEVER MCTS chose, regardless of win/loss/draw
outcome - a real, evidenced cause of this project's headline strategic lock-in
finding (100% single-strategy usage confirmed against both baseline1 and baseline2
in real Phase 5 evaluation). A targeted fix (filtering DSPy's few-shot examples to
WON episodes only) showed a measurable, positive behavioral shift in controlled
diagnostic testing. This applies that same fix to the real training loop, ahead of
a full Benchmark 2 retrain.

Backward compatible: if episode_outcomes is empty/not passed, or filtering would
leave zero examples (no wins yet), falls back to the original unfiltered
training_pairs - never starves DSPy of examples entirely.

Run once on the pod, from the repo root:
    python fix_dspy_outcome_aware_v2.py
"""

from pathlib import Path

# --- Fix 1: match_trainer.py --------------------------------------------------

target1 = Path("src/sbso/match_trainer.py")
content1 = target1.read_text()

old1_init = "        self.prompt_version = 0"
new1_init = (
    "        self.prompt_version = 0\n"
    "        self._episode_outcomes: dict = {}   # episode -> outcome_value (1.0/0.5/0.0),\n"
    "                                              # for DSPy's outcome-aware example filter"
)

old1_track = '            outcome_value = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(outcome, 0.5)'
new1_track = (
    '            outcome_value = {"win": 1.0, "draw": 0.5, "loss": 0.0}.get(outcome, 0.5)\n'
    "            self._episode_outcomes[ep] = outcome_value"
)

old1_call = "                    candidate_prompt = self.dspy_compiler.compile(self.training_pairs, self.prompt_program)"
new1_call = (
    "                    candidate_prompt = self.dspy_compiler.compile(\n"
    "                        self.training_pairs, self.prompt_program,\n"
    "                        episode_outcomes=self._episode_outcomes,\n"
    "                    )"
)

for label, old, new in [
    ("__init__ tracking dict", old1_init, new1_init),
    ("outcome tracking in run() loop", old1_track, new1_track),
    ("compile() call site", old1_call, new1_call),
]:
    if old not in content1:
        print(f"ERROR (match_trainer.py): {label} anchor not found. Paste current content, do not guess.")
        raise SystemExit(1)
    content1 = content1.replace(old, new)

target1.write_text(content1)
print("match_trainer.py: 3 replacements applied.")

# --- Fix 2: dspy_compiler.py (RealDSPyCompiler.compile, the real one at ~line 339) --

target2 = Path("src/sbso/dspy_compiler.py")
content2 = target2.read_text()

old2 = '''    def compile(self, training_pairs: list, current_prompt: str) -> str:
        import dspy

        self._ensure_lm()
        self.compile_count += 1

        class SAStrategySignature(dspy.Signature):
            """Select the macro strategy for a sumo robot given its LSSD state."""
            state_desc: str = dspy.InputField(desc="LSSD semantic state text")
            strategy: str = dspy.OutputField(desc="one of: charge, flank, retreat, hold, evade_edge")

        examples = []
        for state, strategy in self._select_examples(training_pairs):'''

new2 = '''    def compile(self, training_pairs: list, current_prompt: str, episode_outcomes: dict | None = None) -> str:
        import dspy

        self._ensure_lm()
        self.compile_count += 1

        # Outcome-aware filter: only use examples from WON episodes, not every
        # episode regardless of outcome. Real, evidenced fix for the strategic
        # lock-in finding - DSPy was previously trained to imitate whatever MCTS
        # chose regardless of win/loss/draw. Backward compatible: falls back to the
        # full, unfiltered training_pairs if episode_outcomes is empty (e.g. no
        # wins yet early in training) or not passed at all.
        if episode_outcomes:
            won_episodes = {ep for ep, val in episode_outcomes.items() if val == 1.0}
            filtered = [
                entry for entry in training_pairs
                if len(entry) == 3 and entry[0] in won_episodes
            ]
            if filtered:
                training_pairs = filtered
            # else: no won episodes yet - proceed with the full, unfiltered set

        class SAStrategySignature(dspy.Signature):
            """Select the macro strategy for a sumo robot given its LSSD state."""
            state_desc: str = dspy.InputField(desc="LSSD semantic state text")
            strategy: str = dspy.OutputField(desc="one of: charge, flank, retreat, hold, evade_edge")

        examples = []
        for state, strategy in self._select_examples(training_pairs):'''

if old2 not in content2:
    print("ERROR (dspy_compiler.py): compile() anchor not found. Paste current content, do not guess.")
    raise SystemExit(1)
content2 = content2.replace(old2, new2)

target2.write_text(content2)
print("dspy_compiler.py: 1 replacement applied.")

import ast
ast.parse(target1.read_text())
ast.parse(target2.read_text())
print("Both files parse correctly - syntax OK.")
