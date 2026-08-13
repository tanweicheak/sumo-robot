"""
scripts._script_common

Phase: Phase 0
Purpose: Shared helpers for phase entry-point scripts: argument parsing for a config
    path, config loading, and RunContext construction. Keeps each run traceable to its
    resolved config + git commit from the very first scaffold (plan Section 8).

    build_run() takes an optional extra_args callback so scripts that need CLI flags
    beyond --config/--results-dir (e.g. --sim-budget, --judge-model-path) can still get
    RunContext tracking, instead of building their own separate ArgumentParser and
    losing run_id/git-commit/config-hash tracking entirely.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from src.common.config_loader import load_config
from src.common.run_context import RunContext

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_run(
    phase: str,
    description: str,
    extra_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> tuple[dict, RunContext, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to the YAML config for this run.")
    parser.add_argument("--results-dir", default=None, help="Where to write run metadata.")
    if extra_args is not None:
        # Caller registers its own flags on the SAME parser before parse_args() runs.
        # --config and --results-dir are reserved by build_run() itself - argparse will
        # raise a clear error at parse time if a caller's extra_args tries to redefine
        # either, rather than silently conflicting.
        extra_args(parser)
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

    return config, ctx, args

