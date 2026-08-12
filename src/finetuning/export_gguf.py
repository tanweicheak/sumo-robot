"""
src.finetuning.export_gguf

Phase: Phase 4 (Stage 2/3 - fp16->GGUF-f16 conversion; llama-quantize is CLOUD or
    local-with-llama.cpp-built, see quantize_gguf.py)
Purpose: GGUF export for llama.cpp-based edge inference (report Section 3.3.2.7, step 4).
    Shells out to llama.cpp's convert_hf_to_gguf.py against a local llama.cpp checkout.

    D4 fix: this used to take `quantized_model_path` (GPTQ INT4 output) and feed it
    straight into convert_hf_to_gguf.py. GPTQ's packed-weight format and llama.cpp's own
    K-quant scheme are two different, non-chainable quantization ecosystems -
    convert_hf_to_gguf.py expects full-precision (fp16/bf16) HF weights, not GPTQ-packed
    ones. Fixed: this now takes `merged_model_path` (straight from merge_adapters.py,
    bypassing GPTQ entirely) and only does the fp16 -> GGUF-f16 conversion. The Q4_K_M
    quantization step lives in quantize_gguf.py (llama-quantize), a separate, parallel
    branch from GPTQ - see scripts/run_export_pipeline.py for how the two branches
    fan out from one merged model.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def export_to_gguf(
    merged_model_path: str | Path,
    output_path: str | Path,
    llama_cpp_convert_script: str | Path,
    outtype: str = "f16",
) -> Path:
    """fp16 merged HF model -> GGUF f16. NOT quantized yet - see quantize_gguf.py for
    the Q4_K_M step. outtype defaults to "f16" (not "q4_k_m" - convert_hf_to_gguf.py's
    own quantization support is much more limited than llama-quantize's; do the
    conversion at full precision, then quantize with the dedicated tool)."""
    merged_model_path = Path(merged_model_path)
    if not merged_model_path.exists():
        raise FileNotFoundError(
            f"merged_model_path does not exist: {merged_model_path}. This function "
            "expects merge_adapters.merge_lora_adapters()'s OUTPUT (fp16 HF weights) - "
            "NOT quantize_gptq's output, which is a separate, incompatible artifact."
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python3", str(llama_cpp_convert_script),
        str(merged_model_path),
        "--outfile", str(output_path),
        "--outtype", outtype,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GGUF f16 export failed:\n{result.stderr}")
    return output_path
