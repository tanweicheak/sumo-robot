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
import logging
import shutil
from pathlib import Path
from typing import Callable

from src.common.config_loader import load_config, validate_config
from src.common.run_context import RunContext

REPO_ROOT = Path(__file__).resolve().parents[1]


def poll_gpu_stats() -> dict | None:
    """GPU utilization/memory snapshot via nvidia-smi - the one piece of live
    telemetry that genuinely needs real GPU hardware to verify; this sandbox and
    any non-GPU machine will always get None back, by design, never a crash.
    Written now so it's ready the moment a real RunPod run can confirm it behaves
    correctly; safe to call unconditionally from any script regardless of where
    it runs."""
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        util, mem_used, mem_total = (x.strip() for x in proc.stdout.strip().split(","))
        return {
            "gpu_utilization_pct": float(util),
            "gpu_memory_used_mb": float(mem_used),
            "gpu_memory_total_mb": float(mem_total),
        }
    except Exception:  # noqa: BLE001 - telemetry failing should never crash the training run
        return None


def setup_logging(out_dir: Path, name: str = "sbso") -> logging.Logger:
    """Structured logging replacing print() for long, unattended training runs.
    Writes to BOTH stdout (so tmux/live viewing works exactly as before) and a
    persistent checkpoints/<run_id>/run.log file that survives closing the
    terminal - the motivating gap: debugging a multi-hour RunPod run with only
    scrollback and no permanent, timestamped, severity-tagged record."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()   # avoid duplicate handlers if called more than once in a process

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(out_dir / "run.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


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
    config = validate_config(config, phase)
    variant = config.get("variant_name", Path(args.config).stem)

    ctx = RunContext(variant_name=variant, phase=phase, resolved_config=config)

    if args.results_dir:
        meta_path = ctx.save(args.results_dir)
        print(f"[{phase}] run_id={ctx.run_id}")
        print(f"[{phase}] wrote run metadata -> {meta_path}")
    else:
        print(f"[{phase}] run_id={ctx.run_id} (no --results-dir given; metadata not persisted)")

    return config, ctx, args