"""
scripts.preflight_runpod_check

Phase: pre-Phase-4-full-run gate (RunPod-only half)
Purpose: The checks that CANNOT run locally on your Mac because they need the
    actual provisioned GPU box: SGLang servers actually up and serving, real CUDA
    visible, the full pytest suite passing against the cloud image's exact dependency
    versions (not your local CPU/MPS versions). Run this once, right after
    scripts/launch_sglang_servers.sh, before kicking off the real pilot or full run.

    Companion to scripts/preflight_static_check.py, which covers everything that
    DOESN'T need a GPU and should be run locally first.

Usage (on the RunPod pod, after servers are launched):
    python -m scripts.preflight_runpod_check
    python -m scripts.preflight_runpod_check --skip-pytest   # if you've already run it separately
    python -m scripts.preflight_runpod_check --n-decode-samples 100

Exit code: 0 if no FAILs, 1 otherwise. WARNs never fail the run.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from collections import Counter
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
        except Exception as e:  # noqa: BLE001
            result = CheckResult(name, "FAIL", f"check raised {type(e).__name__}: {e}")
        self.results.append(result)
        icon = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[result.status]
        print(f"{icon} {result.name}: {result.message}")


# ---------------------------------------------------------------------------
# Check 1: a real GPU is actually visible
# ---------------------------------------------------------------------------

def check_gpu_visible() -> CheckResult:
    if shutil.which("nvidia-smi") is None:
        return CheckResult("gpu_visible", "FAIL", "nvidia-smi not found on PATH - is this actually a GPU pod?")

    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return CheckResult("gpu_visible", "FAIL", f"nvidia-smi failed: {proc.stderr.strip()[:200]}")

    return CheckResult("gpu_visible", "PASS", proc.stdout.strip().replace("\n", " | "))


# ---------------------------------------------------------------------------
# Check 2: both SGLang servers (agent + judge) are actually reachable
# ---------------------------------------------------------------------------

def check_sglang_servers_reachable(repo_root: Path) -> CheckResult:
    sys.path.insert(0, str(repo_root))
    from src.common.config_loader import load_config  # noqa: E402
    import requests  # noqa: E402

    cfg = load_config(repo_root / "config" / "inference.yaml")
    sglang_cfg = cfg.get("sglang", {})
    agent_url = sglang_cfg.get("agent_server_url")
    judge_url = sglang_cfg.get("judge_server_url")

    unreachable = []
    for label, url in [("agent", agent_url), ("judge", judge_url)]:
        try:
            resp = requests.get(f"{url}/health", timeout=10)
            if resp.status_code != 200:
                unreachable.append(f"{label} ({url}) returned HTTP {resp.status_code}")
        except Exception as e:  # noqa: BLE001
            unreachable.append(f"{label} ({url}) unreachable: {type(e).__name__}")

    if unreachable:
        return CheckResult(
            "sglang_servers_reachable", "FAIL",
            "; ".join(unreachable) + " - did launch_sglang_servers.sh finish starting up?",
        )
    return CheckResult("sglang_servers_reachable", "PASS", f"agent ({agent_url}) and judge ({judge_url}) both healthy")


# ---------------------------------------------------------------------------
# Check 3: constrained decoding is actually schema-valid under real load, and
# the silent values[0] fallback isn't quietly firing on every call
# ---------------------------------------------------------------------------

def check_constrained_decoding_under_load(repo_root: Path, n_samples: int) -> CheckResult:
    sys.path.insert(0, str(repo_root))
    from src.common.config_loader import load_config  # noqa: E402
    from src.agents.schemas import MacroStrategy  # noqa: E402
    from src.inference.grammar import enum_regex_pattern  # noqa: E402
    import requests  # noqa: E402

    cfg = load_config(repo_root / "config" / "inference.yaml")
    sglang_cfg = cfg.get("sglang", {})
    agent_url = sglang_cfg.get("agent_server_url")

    values = [s.value for s in MacroStrategy]
    regex = enum_regex_pattern(values)

    # Vary the prompt slightly per sample so a fixed fallback (always values[0])
    # is distinguishable from a genuinely uniform model preference.
    sample_prompts = [
        f"State: opponent at bearing {i * 37 % 360} degrees, distance {0.3 + (i % 5) * 0.2:.1f}m.\nStrategy:"
        for i in range(n_samples)
    ]

    outputs: list[str] = []
    invalid = 0
    t0 = time.time()
    for prompt in sample_prompts:
        try:
            resp = requests.post(
                f"{agent_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "regex": regex,
                        "temperature": sglang_cfg.get("temperature", 0.0),
                        "max_new_tokens": sglang_cfg.get("max_tokens", 8),
                    },
                },
                timeout=sglang_cfg.get("timeout_s", 30),
            )
            text = resp.json().get("text", "").strip()
        except Exception:  # noqa: BLE001
            text = ""

        outputs.append(text)
        if text not in values:
            invalid += 1

    elapsed = time.time() - t0
    valid_rate = 1.0 - (invalid / max(1, n_samples))

    dist = Counter(outputs)
    dominant_value, dominant_count = dist.most_common(1)[0] if dist else (None, 0)
    dominance_rate = dominant_count / max(1, n_samples)

    if invalid > 0:
        return CheckResult(
            "constrained_decoding_under_load", "FAIL",
            f"{invalid}/{n_samples} responses were NOT a valid MacroStrategy value "
            f"({valid_rate:.1%} valid, {elapsed:.1f}s total). test_grammar.py never caught "
            "this because it only tests grammar-string construction, not real decode-time "
            "enforcement under load.",
        )
    if dominance_rate > 0.9 and len(values) > 1:
        return CheckResult(
            "constrained_decoding_under_load", "WARN",
            f"all {n_samples} responses were schema-valid, but {dominance_rate:.0%} were the "
            f"same value ('{dominant_value}'). Both sglang_client.py and llama_cpp_server.py "
            "have a silent `values[0]` fallback with no logging when the constraint fails - "
            "this dominance pattern is consistent with (but not proof of) that fallback firing "
            "on most/all calls. Worth instrumenting that fallback path with a counter before "
            "trusting this result either way.",
        )
    return CheckResult(
        "constrained_decoding_under_load", "PASS",
        f"{n_samples}/{n_samples} valid ({elapsed:.1f}s total), output distribution: {dict(dist)}",
    )


# ---------------------------------------------------------------------------
# Check 4: full pytest suite passes on THIS image's exact dependency versions
# ---------------------------------------------------------------------------

def check_pytest_suite(repo_root: Path) -> CheckResult:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/unit/", "-q"],
        capture_output=True, text=True, cwd=repo_root, timeout=900,
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
    if proc.returncode != 0:
        return CheckResult(
            "pytest_suite", "FAIL",
            f"pytest exited {proc.returncode} - see tail below.\n{tail}",
        )
    return CheckResult("pytest_suite", "PASS", tail.splitlines()[-1] if tail else "all tests passed")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RunPod-only runtime preflight checks for Phase 4.")
    p.add_argument("--repo-root", default=".", help="Path to the repo root (default: cwd)")
    p.add_argument("--n-decode-samples", type=int, default=50, help="Samples for the constrained-decoding check")
    p.add_argument("--skip-pytest", action="store_true", help="Skip the full pytest run (e.g. already run separately)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()

    print(f"[preflight_runpod_check] repo_root={repo_root}\n")

    suite = CheckSuite()
    suite.run("gpu_visible", check_gpu_visible)
    suite.run("sglang_servers_reachable", lambda: check_sglang_servers_reachable(repo_root))
    suite.run(
        "constrained_decoding_under_load",
        lambda: check_constrained_decoding_under_load(repo_root, args.n_decode_samples),
    )
    if not args.skip_pytest:
        suite.run("pytest_suite", lambda: check_pytest_suite(repo_root))
    else:
        print("[SKIP] pytest_suite: --skip-pytest passed")

    n_pass = sum(1 for r in suite.results if r.status == "PASS")
    n_warn = sum(1 for r in suite.results if r.status == "WARN")
    n_fail = sum(1 for r in suite.results if r.status == "FAIL")
    print(f"\n{n_pass} passed, {n_warn} warned, {n_fail} failed")
    print(
        "\nNote: this does NOT rerun verify_a1_a2_real_pybullet.py automatically (it needs "
        "arguments specific to your last local baseline) - run it manually on this box too "
        "before trusting physics parity with your Mac."
    )

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
