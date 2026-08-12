"""
scripts/run_export_pipeline.py

Phase: Phase 4 (Stage 2/3)
Purpose: One command, two deployment artifacts. Runs merge_lora_adapters() ONCE, then
    fans out into two INDEPENDENT, PARALLEL branches from that same merged fp16 model:
      - edge/llama.cpp:  merged fp16 -> GGUF f16 (export_gguf) -> Q4_K_M (quantize_gguf)
      - cloud/SGLang:    merged fp16 -> GPTQ INT4 (quantize_gptq)
    Neither branch depends on the other's output - this was the exact bug in D4 (the
    old export_gguf.py incorrectly chained off GPTQ's output). The GPTQ branch requires
    CUDA and is automatically skipped (with a clear message, not a crash) when running
    locally on the Mac - useful for local token-runs that only need the edge artifact.

Usage:
    python scripts/run_export_pipeline.py \\
        --base-model-path /workspace/models/phi-4-mini \\
        --adapter-path checkpoints/benchmark2_full_sbso/lora_adapter \\
        --output-dir /workspace/export \\
        --llama-cpp-convert-script /path/to/llama.cpp/convert_hf_to_gguf.py \\
        --calibration-texts-file config/gptq_calibration_texts.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.finetuning.export_gguf import export_to_gguf
from src.finetuning.merge_adapters import merge_lora_adapters
from src.finetuning.quantize_gguf import quantize_gguf
from src.finetuning.quantize_gptq import quantize_gptq


def _load_calibration_texts(path: str | None) -> list[str]:
    if not path:
        return []
    lines = Path(path).read_text().splitlines()
    return [line for line in (l.strip() for l in lines) if line]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model-path", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--output-dir", required=True, help="Root dir - subfolders created for each artifact")
    ap.add_argument("--llama-cpp-convert-script", required=True, help="Path to llama.cpp's convert_hf_to_gguf.py")
    ap.add_argument("--llama-quantize-binary", default=None, help="Defaults to searching PATH")
    ap.add_argument("--gguf-quant-type", default="Q4_K_M")
    ap.add_argument("--calibration-texts-file", default=None, help="One calibration text per line, for GPTQ")
    ap.add_argument("--skip-gptq", action="store_true", help="Skip the cloud/GPTQ branch even if CUDA is available")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "merged_fp16"
    gguf_f16_path = output_dir / "model-f16.gguf"
    gguf_quant_path = output_dir / f"model-{args.gguf_quant_type.lower()}.gguf"
    gptq_dir = output_dir / "gptq_int4"

    print(f"[1/3] Merging LoRA adapter into base model -> {merged_dir}")
    merge_lora_adapters(args.base_model_path, args.adapter_path, merged_dir)

    print(f"[2/3] Edge branch: GGUF f16 export -> {gguf_f16_path}")
    export_to_gguf(merged_dir, gguf_f16_path, args.llama_cpp_convert_script)
    print(f"[2/3] Edge branch: {args.gguf_quant_type} quantization -> {gguf_quant_path}")
    quantize_gguf(gguf_f16_path, gguf_quant_path, args.llama_quantize_binary, args.gguf_quant_type)
    print(f"[2/3] Edge branch DONE: {gguf_quant_path}")

    if args.skip_gptq:
        print("[3/3] Cloud/GPTQ branch: skipped (--skip-gptq)")
    else:
        calibration_texts = _load_calibration_texts(args.calibration_texts_file)
        if not calibration_texts:
            print(
                "[3/3] Cloud/GPTQ branch: skipped - no --calibration-texts-file given "
                "(or file was empty). GPTQ needs in-domain LSSD calibration texts; "
                "pass a real one to run this branch.",
                file=sys.stderr,
            )
        else:
            try:
                print(f"[3/3] Cloud branch: GPTQ INT4 quantization -> {gptq_dir}")
                quantize_gptq(merged_dir, gptq_dir, calibration_texts)
                print(f"[3/3] Cloud branch DONE: {gptq_dir}")
            except RuntimeError as e:
                # quantize_gptq.py itself raises a clear "no CUDA" RuntimeError -
                # surface it as a skip, not a pipeline failure, when running locally.
                print(f"[3/3] Cloud/GPTQ branch: skipped - {e}", file=sys.stderr)

    print("\nDone.")
    print(f"  Edge artifact (llama.cpp):  {gguf_quant_path}")
    print(f"  Cloud artifact (SGLang):    {gptq_dir}  (if not skipped above)")


if __name__ == "__main__":
    main()
