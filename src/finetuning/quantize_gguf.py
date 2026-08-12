"""
src.finetuning.quantize_gguf

Phase: Phase 4 (Stage 2/3)
Purpose: Q4_K_M quantization of a GGUF-f16 model via llama.cpp's own `llama-quantize`
    binary (report Section 3.3.2.7, step 4b) - the edge/llama.cpp deployment target.
    Mirrors quantize_gptq.py's one-file-one-job pattern; these two are INDEPENDENT,
    PARALLEL branches from the same merged fp16 model (see D4 in STATE_TRANSFER_SUMMARY),
    not a sequential chain. This one has no CUDA requirement and can run locally on the
    Mac once llama.cpp is built (`make llama-quantize` in a llama.cpp checkout, or via
    the llama-cpp-python package's bundled binary - see _find_llama_quantize_binary).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _find_llama_quantize_binary(explicit_path: str | Path | None) -> str:
    if explicit_path:
        return str(explicit_path)
    found = shutil.which("llama-quantize")
    if found:
        return found
    raise FileNotFoundError(
        "llama-quantize binary not found on PATH and no explicit path given. Build it "
        "from a llama.cpp checkout (`cmake --build build --target llama-quantize`, or "
        "the older `make llama-quantize`), or pass llama_quantize_binary= explicitly."
    )


def quantize_gguf(
    gguf_f16_path: str | Path,
    output_path: str | Path,
    llama_quantize_binary: str | Path | None = None,
    quant_type: str = "Q4_K_M",
) -> Path:
    """GGUF f16 (export_gguf.py's output) -> quantized GGUF (default Q4_K_M). This is
    llama.cpp's OWN quantization tool - not GPTQ, not chained from GPTQ's output."""
    gguf_f16_path = Path(gguf_f16_path)
    if not gguf_f16_path.exists():
        raise FileNotFoundError(f"gguf_f16_path does not exist: {gguf_f16_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    binary = _find_llama_quantize_binary(llama_quantize_binary)

    cmd = [binary, str(gguf_f16_path), str(output_path), quant_type]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF {quant_type} quantization failed:\n{result.stderr}")
    return output_path
