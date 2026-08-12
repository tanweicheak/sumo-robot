"""
src.finetuning.calibration_texts

Phase: Phase 4 (Stage 3)
Purpose: Generates the in-domain calibration text GPTQ needs (quantize_gptq.py's
    `calibration_texts` argument) from REAL data instead of hand-written or
    generic-corpus text. The source is SBSOTrainer.training_pairs - every decision
    made during actual training already carries a real PerceptionAgent-produced
    lssd_text string (e.g. "opponent close, edge safe, momentum forward"), which is
    exactly the kind of text the deployed model will see at inference. No separate
    generation run needed; this just extracts and writes what training already
    produced.

    training_pairs entries are (episode, state, strategy) triples (or, for older/
    Stage 1-2 callers, bare (state, strategy) pairs) - state is either a
    PyBulletMCTSState (state.lssd_text) or a plain dict ({"lssd": ...}), matching the
    same duck-typing RealDSPyCompiler._select_examples already uses.
"""

from __future__ import annotations

import random
from pathlib import Path


def _extract_lssd_text(state) -> str:
    if isinstance(state, dict):
        return str(state.get("lssd", "") or state.get("lssd_text", ""))
    return str(getattr(state, "lssd_text", ""))


def extract_calibration_texts(
    training_pairs: list,
    n_samples: int = 256,
    seed: int = 0,
    min_length: int = 1,
) -> list[str]:
    """Pulls a deduplicated, randomly-sampled set of real lssd_text strings out of a
    completed (or in-progress) training run's training_pairs. Returns a plain list of
    strings - use write_calibration_file() to persist it in the shape
    run_export_pipeline.py / quantize_gptq.py expect."""
    seen: dict[str, None] = {}   # dict, not set, to preserve first-seen order
    for entry in training_pairs:
        state = entry[1] if len(entry) == 3 else entry[0]
        text = _extract_lssd_text(state).strip()
        if len(text) >= min_length:
            seen[text] = None   # dedupe: many decisions produce identical/near-identical text

    unique_texts = list(seen.keys())
    if len(unique_texts) <= n_samples:
        return unique_texts

    rng = random.Random(seed)
    return rng.sample(unique_texts, n_samples)


def write_calibration_file(
    training_pairs: list,
    output_path: str | Path,
    n_samples: int = 256,
    seed: int = 0,
) -> Path:
    """Extracts and writes one calibration text per line - directly consumable by
    run_export_pipeline.py's --calibration-texts-file argument."""
    texts = extract_calibration_texts(training_pairs, n_samples=n_samples, seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(texts) + ("\n" if texts else ""))
    return output_path
