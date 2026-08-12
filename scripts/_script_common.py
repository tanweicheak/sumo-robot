"""
scripts._script_common

Phase: Phase 0
Purpose: Shared helpers for phase entry-point scripts: argument parsing for a config
    path, config loading, and RunContext construction. Keeps each run traceable to its
    resolved config + git commit from the very first scaffold (plan Section 8).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config_loader import load_config
from src.common.run_context import RunContext

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_run(phase: str, description: str) -> tuple[dict, RunContext]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the YAML config for this run.")
    parser.add_argument("--results-dir", default=None, help="Where to write run metadata.")
    args = parser.parse_args()

    config = load_config(args.config)
    variant = config.get("variant_name", Path(args.config).stem)

    ctx = RunContext(variant_name=variant, phase=phase, resolved_config=config)

    if args.results_dir:
        meta_path = ctx.save(args.results_dir)
        print(f"[{phase}] run_id={ctx.run_id}")
        print(f"[{phase}] wrote run metadata -> {meta_path}")
    else:
        print(f"[{phase}] run_id={ctx.run_id} (no --results-dir given; metadata not persisted)")

    return config, ctx
