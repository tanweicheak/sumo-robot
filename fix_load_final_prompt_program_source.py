"""One-shot fix for src/evaluation/match_runner.py's _load_final_prompt_program.

Real bug found: prompt_history.jsonl's actual schema (confirmed from real file
content) is {episode, trigger_reason, prompt_version, rolling_winrate, accepted,
validation} - it NEVER contained a prompt_program key at all. That was a wrong
assumption baked into this function when it was built. The real compiled prompt
text lives in progress.json instead (written periodically at checkpoint
intervals, format: {"episode": N, "prompt_program": "SA_PROMPT_V8\\n...", ...}).

Fix: read progress.json's prompt_program field instead. Caveat, documented in the
new docstring: progress.json only snapshots periodically (at
self_checkpoint_interval_episodes boundaries, plus episode 0) - so this returns
whatever the LAST written snapshot was, which may not be the literal final
episode's exact prompt version if training continued past the last snapshot
before ending. This is the best available source without re-running training;
noted explicitly rather than silently presented as exact.

Run once on the pod, from the repo root:
    python fix_load_final_prompt_program_source.py
"""

from pathlib import Path

target = Path("src/evaluation/match_runner.py")
content = target.read_text()

old = '''def _load_final_prompt_program(prompt_history_path: str) -> str:
    """Reads prompt_history.jsonl (one JSON object per episode, each with a
    "prompt_program" field - confirmed against run_phase4_pilot.py's own
    _on_episode_end logging) and returns the LAST entry's compiled prompt - i.e.
    whatever DSPy had converged to by the end of training. Raises clearly rather
    than silently falling back to None if the file is empty/missing, since a silent
    None here is exactly the bug this function exists to prevent."""
    import json
    from pathlib import Path

    path = Path(prompt_history_path)
    if not path.exists():
        raise FileNotFoundError(f"prompt_history.jsonl not found at {path}")

    last_program: str | None = None
    with path.open() as f:'''

new = '''def _load_final_prompt_program(prompt_history_path: str) -> str:
    """CORRECTED: prompt_history.jsonl's real schema (confirmed against actual
    written data) is {episode, trigger_reason, prompt_version, rolling_winrate,
    accepted, validation} - it never contains a prompt_program field at all, despite
    this function's original docstring claiming otherwise. The real compiled prompt
    text is written to progress.json instead (format: {"episode": N,
    "prompt_program": "SA_PROMPT_VN\\n...", ...}), but only periodically - at
    self_checkpoint_interval_episodes boundaries and episode 0 - NOT every episode.
    This function now reads progress.json and returns its prompt_program value,
    which is the LAST periodic snapshot taken, not necessarily the exact final
    episode's prompt if training continued past the last snapshot before ending.
    Despite the parameter name (kept for call-site compatibility), pass the
    checkpoint's progress.json path here, not prompt_history.jsonl's path."""
    import json
    from pathlib import Path

    path = Path(prompt_history_path)
    if not path.exists():
        raise FileNotFoundError(f"progress.json not found at {path}")

    with path.open() as f:
        data = json.load(f)
    program = data.get("prompt_program")
    if not program:
        raise ValueError(
            f"{path} has no prompt_program value - training may not have reached "
            "its first checkpoint interval, or this is the wrong file."
        )
    return program


def _load_final_prompt_program_OLD_UNUSED(prompt_history_path: str) -> str:
    """Superseded - see _load_final_prompt_program above. Kept only so the
    remainder of this file's original loop-based parsing code doesn't need
    re-indenting; never called."""
    import json
    from pathlib import Path

    path = Path(prompt_history_path)
    if not path.exists():
        raise FileNotFoundError(f"prompt_history.jsonl not found at {path}")

    last_program: str | None = None
    with path.open() as f:'''

if old not in content:
    print("ERROR: anchor not found - paste current file content, do not guess.")
    raise SystemExit(1)

content = content.replace(old, new)
target.write_text(content)
print("Fix applied.")

import ast
ast.parse(target.read_text())
print("Syntax OK.")
