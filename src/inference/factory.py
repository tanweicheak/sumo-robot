"""
src.inference.factory

Phase: Phase 3
Purpose: Construct an SLMClient from config. Selects mock (fast dev/CI) or the real
    llama.cpp backend, so the whole pipeline swaps between them without code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.config_loader import load_config
from src.inference.slm_client import MockSLMClient, SLMClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INFERENCE_CONFIG = _REPO_ROOT / "config" / "inference.yaml"


def build_slm_client(config_path: str | Path = _DEFAULT_INFERENCE_CONFIG) -> SLMClient:
    cfg: dict[str, Any] = load_config(config_path)
    backend = cfg.get("backend", "mock")

    if backend == "mock":
        m = cfg.get("mock", {})
        return MockSLMClient(seed=int(m.get("seed", 0)), latency_ms=float(m.get("latency_ms", 0.0)))

    if backend == "llama_cpp":
        from src.inference.llama_cpp_server import LlamaCppSLMClient

        lc = cfg["llama_cpp"]
        return LlamaCppSLMClient(
            model_path=lc["model_path"],
            n_ctx=int(lc.get("n_ctx", 2048)),
            n_gpu_layers=int(lc.get("n_gpu_layers", -1)),
            temperature=float(lc.get("temperature", 0.0)),
            max_tokens=int(lc.get("max_tokens", 8)),
            seed=int(lc.get("seed", 0)),
            verbose=bool(lc.get("verbose", False)),
        )

    if backend == "sglang":
        from src.inference.sglang_server import SGLangSLMClient

        sg = cfg["sglang"]
        return SGLangSLMClient(
            server_url=sg["agent_server_url"],
            temperature=float(sg.get("temperature", 0.0)),
            max_tokens=int(sg.get("max_tokens", 8)),
            timeout_s=float(sg.get("timeout_s", 30.0)),
            max_concurrency=int(sg.get("max_concurrency", 16)),
        )

    raise ValueError(f"Unknown inference backend: {backend}")