"""
src.common.run_context

Phase: Phase 0
Purpose: Reproducibility utility. Every Phase 4/5 run must be traceable back to the
    exact config and git commit that produced it (plan Section 8). This module builds
    a run identifier and a metadata record tying a run to its resolved config, git
    commit, and timestamp, and can persist that record alongside results.

Dependency-light: standard library only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a resolved config, for change detection."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass
class RunContext:
    """Metadata record for a single training or evaluation run."""

    variant_name: str
    phase: str
    resolved_config: dict[str, Any]
    run_id: str = ""
    git_commit: str = field(default_factory=_git_commit)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.config_hash:
            self.config_hash = _config_hash(self.resolved_config)
        if not self.run_id:
            # e.g. benchmark2_full_sbso__phase4__20260808T101500Z__a1b2c3d4e5f6
            stamp = (
                self.created_at.replace("-", "")
                .replace(":", "")
                .split(".")[0]
                .replace("+0000", "Z")
            )
            self.run_id = f"{self.variant_name}__{self.phase}__{stamp}__{self.config_hash}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, results_dir: str | Path) -> Path:
        """Write run metadata to `<results_dir>/<run_id>/run_context.json`."""
        run_dir = Path(results_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "run_context.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True, default=str)
        return out_path
