"""
scripts.preflight_static_check

Phase: pre-Phase-4-full-run gate
Purpose: Fast, local, no-GPU checks that catch the class of bug this project has
    repeatedly hit by hand (build_run() 2-vs-3-tuple drift, a malformed YAML value
    parsing as the wrong type, a config key silently going unread). Every check here
    is something that would otherwise only surface as a crash or a silent behavior
    change partway through a real RunPod run.

    This script does NOT need pybullet, torch, sglang, or a GPU - it only needs
    pyyaml (already in requirements.txt) and the repo's own source tree. Run it
    anywhere, anytime, as often as you like.

Usage:
    python -m scripts.preflight_static_check
    python -m scripts.preflight_static_check --repo-root /path/to/repo
    python -m scripts.preflight_static_check --strict   # exit non-zero on WARN too

Exit code: 0 if no FAILs (WARNs allowed unless --strict), 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    message: str


@dataclass
class CheckSuite:
    results: list[CheckResult] = field(default_factory=list)

    def run(self, name: str, fn: Callable[[], CheckResult]) -> None:
        try:
            result = fn()
        except Exception as e:  # noqa: BLE001 - a check crashing is itself a FAIL, not a script crash
            result = CheckResult(name, "FAIL", f"check raised {type(e).__name__}: {e}")
        self.results.append(result)
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[result.status]
        print(f"{icon} {result.name}: {result.message}")


# ---------------------------------------------------------------------------
# Check 1: DSPy recompilation trigger values are non-null in the resolved config
# ---------------------------------------------------------------------------

def check_dspy_triggers_nonnull(repo_root: Path) -> CheckResult:
    sys.path.insert(0, str(repo_root))
    from src.common.config_loader import load_config  # noqa: E402

    cfg = load_config(repo_root / "config" / "training" / "phase4_full_sbso.yaml")
    dspy_cfg = cfg.get("dspy_recompilation", {})
    required = ["k_rollout_batches", "reward_drop_threshold_delta", "rolling_window_w"]
    missing = [k for k in required if dspy_cfg.get(k) is None]

    if missing:
        return CheckResult(
            "dspy_triggers_nonnull", "FAIL",
            f"dspy_recompilation.{{{', '.join(missing)}}} is null in the resolved "
            "phase4_full_sbso.yaml config - DSPy recompilation would never/always fire.",
        )
    return CheckResult(
        "dspy_triggers_nonnull", "PASS",
        f"k_rollout_batches={dspy_cfg['k_rollout_batches']}, "
        f"reward_drop_threshold_delta={dspy_cfg['reward_drop_threshold_delta']}, "
        f"rolling_window_w={dspy_cfg['rolling_window_w']} (all non-null)",
    )


# ---------------------------------------------------------------------------
# Check 2: DSPy recompile frequency vs self-checkpoint interval alignment
# ---------------------------------------------------------------------------

def check_dspy_k_vs_checkpoint_alignment(repo_root: Path) -> CheckResult:
    sys.path.insert(0, str(repo_root))
    from src.common.config_loader import load_config  # noqa: E402

    cfg = load_config(repo_root / "config" / "training" / "phase4_full_sbso.yaml")
    k = cfg.get("dspy_recompilation", {}).get("k_rollout_batches")
    checkpoint_interval = cfg.get("self_checkpoint_interval_episodes")

    if k is None or checkpoint_interval is None:
        return CheckResult(
            "dspy_k_vs_checkpoint_alignment", "WARN",
            "could not compare - one or both of k_rollout_batches / "
            "self_checkpoint_interval_episodes is missing from the resolved config.",
        )

    ratio = checkpoint_interval / k if k else float("inf")
    # RecompilationScheduler's own docstring states the intended default is K=500,
    # "aligned with self-checkpoint interval" - i.e. ratio ~= 1. A large ratio means
    # DSPy would recompile far more often than the design intends.
    if ratio > 5:
        return CheckResult(
            "dspy_k_vs_checkpoint_alignment", "WARN",
            f"k_rollout_batches={k} but self_checkpoint_interval_episodes={checkpoint_interval} "
            f"(ratio {ratio:.0f}x). RecompilationScheduler's docstring states the intended "
            "default is K=500, aligned with the self-checkpoint interval (ratio ~1x). "
            f"At this ratio, DSPy would recompile ~{ratio:.0f}x more often than the design "
            "intends over the full run. Confirm this is deliberate before the real run.",
        )
    return CheckResult(
        "dspy_k_vs_checkpoint_alignment", "PASS",
        f"k_rollout_batches={k}, self_checkpoint_interval_episodes={checkpoint_interval} "
        f"(ratio {ratio:.1f}x, within tolerance of the documented ~1x design intent)",
    )


# ---------------------------------------------------------------------------
# Check 3: HCMA is not wired into Phase 4 (by design - deferred to 5c only)
# ---------------------------------------------------------------------------

def check_hcma_not_in_phase4(repo_root: Path) -> CheckResult:
    targets = ["src/sbso/match_trainer.py", "src/sbso/training_loop.py"]
    offenders = []
    for rel in targets:
        f = repo_root / rel
        if not f.exists():
            continue
        text = f.read_text()
        if re.search(r"hardware_constraint_monitor|\bHCMA\b", text):
            offenders.append(rel)

    if offenders:
        return CheckResult(
            "hcma_not_in_phase4", "FAIL",
            f"HCMA reference found in {', '.join(offenders)} - HEL/HCMA is meant to apply "
            "exclusively to Phase 5c, not Phase 4 training. Was this intentional?",
        )
    return CheckResult(
        "hcma_not_in_phase4", "PASS",
        "no HCMA import in match_trainer.py or training_loop.py (correct - HCMA is Phase 5c-only)",
    )


# ---------------------------------------------------------------------------
# Check 4: every scripts/*.py using build_run() unpacks the current 3-tuple
# ---------------------------------------------------------------------------

def check_build_run_tuple_consistency(repo_root: Path) -> CheckResult:
    scripts_dir = repo_root / "scripts"
    pattern = re.compile(r"^\s*([\w, ]+?)\s*=\s*build_run\(", re.MULTILINE)
    offenders = []
    checked = 0

    for f in sorted(scripts_dir.glob("*.py")):
        text = f.read_text()
        for match in pattern.finditer(text):
            checked += 1
            lhs_names = [n.strip() for n in match.group(1).split(",") if n.strip()]
            if len(lhs_names) != 3:
                line_no = text[: match.start()].count("\n") + 1
                offenders.append(f"{f.relative_to(repo_root)}:{line_no} unpacks {len(lhs_names)}-tuple")

    if offenders:
        return CheckResult(
            "build_run_tuple_consistency", "FAIL",
            f"{len(offenders)}/{checked} build_run() call(s) use the stale 2-tuple form: "
            + "; ".join(offenders),
        )
    return CheckResult(
        "build_run_tuple_consistency", "PASS",
        f"{checked}/{checked} build_run() call(s) use the current 3-tuple signature",
    )


# ---------------------------------------------------------------------------
# Check 5: no YAML value looks like an accidentally-pasted Python type annotation
# ---------------------------------------------------------------------------

def check_yaml_type_sanity(repo_root: Path) -> CheckResult:
    config_dir = repo_root / "config"
    # Catches patterns like "shaping_push_weight: float = 0.05" - a Python
    # dataclass-field definition pasted where a plain YAML value belonged.
    pattern = re.compile(r"^\s*[\w.]+\s*:\s*(float|int|str|bool)\s*=\s*\S", re.MULTILINE)
    offenders = []

    for f in sorted(config_dir.glob("*.yaml")):
        text = f.read_text()
        for match in pattern.finditer(text):
            line_no = text[: match.start()].count("\n") + 1
            line_text = text.splitlines()[line_no - 1].strip()
            offenders.append(f"{f.relative_to(repo_root)}:{line_no}  `{line_text}`")

    if offenders:
        return CheckResult(
            "yaml_type_sanity", "FAIL",
            f"{len(offenders)} value(s) look like a pasted Python type annotation, "
            "not a real YAML value (parses as a string, not the intended type): "
            + "; ".join(offenders),
        )
    return CheckResult(
        "yaml_type_sanity", "PASS",
        f"no malformed type-annotation-shaped values found across {len(list(config_dir.glob('*.yaml')))} config files",
    )


# ---------------------------------------------------------------------------
# Check 6: cost_projection's gpu_rate_usd_per_hr doesn't look like an untouched placeholder
# ---------------------------------------------------------------------------

def check_gpu_rate_not_placeholder(repo_root: Path) -> CheckResult:
    sys.path.insert(0, str(repo_root))
    from src.common.config_loader import load_config  # noqa: E402

    cfg = load_config(repo_root / "config" / "training" / "phase4_pilot.yaml")
    rate = cfg.get("cost_projection", {}).get("gpu_rate_usd_per_hr")

    # A bare round number (1, 1.0, 2, ...) is very likely an untouched placeholder -
    # real per-second cloud GPU rates are essentially never round numbers.
    if rate is not None and float(rate) == int(rate):
        return CheckResult(
            "gpu_rate_not_placeholder", "WARN",
            f"gpu_rate_usd_per_hr={rate} is a round number - looks like an untouched "
            "placeholder rather than a real measured rate. Update it to the actual rate "
            "of whichever GPU/tier you provision before trusting the cost projection.",
        )
    return CheckResult(
        "gpu_rate_not_placeholder", "PASS",
        f"gpu_rate_usd_per_hr={rate} (not a round-number placeholder)",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Static (no-GPU) preflight checks for Phase 4.")
    p.add_argument("--repo-root", default=".", help="Path to the repo root (default: cwd)")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on WARN as well as FAIL")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()

    print(f"[preflight_static_check] repo_root={repo_root}\n")

    suite = CheckSuite()
    suite.run("dspy_triggers_nonnull", lambda: check_dspy_triggers_nonnull(repo_root))
    suite.run("dspy_k_vs_checkpoint_alignment", lambda: check_dspy_k_vs_checkpoint_alignment(repo_root))
    suite.run("hcma_not_in_phase4", lambda: check_hcma_not_in_phase4(repo_root))
    suite.run("build_run_tuple_consistency", lambda: check_build_run_tuple_consistency(repo_root))
    suite.run("yaml_type_sanity", lambda: check_yaml_type_sanity(repo_root))
    suite.run("gpu_rate_not_placeholder", lambda: check_gpu_rate_not_placeholder(repo_root))

    n_pass = sum(1 for r in suite.results if r.status == "PASS")
    n_warn = sum(1 for r in suite.results if r.status == "WARN")
    n_fail = sum(1 for r in suite.results if r.status == "FAIL")
    print(f"\n{n_pass} passed, {n_warn} warned, {n_fail} failed")

    if n_fail > 0 or (args.strict and n_warn > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
