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

    Stable parameters (base_model_path, llama_cpp_convert_script,
    llama_quantize_binary, gguf_quant_type) live in the --config YAML - see
    config/export_pipeline.yaml, since these are the same across every export this
    project will ever run. Per-export parameters (--adapter-path, --output-dir,
    --calibration-texts-file, --skip-gptq) stay as CLI flags - they identify WHICH
    trained variant you're exporting right now.

Usage:
    python scripts/run_export_pipeline.py \\
        --config config/export_pipeline.yaml \\
        --adapter-path checkpoints/benchmark2_full_sbso/lora_adapter \\
        --output-dir /workspace/export/benchmark2_full_sbso \\
        --calibration-texts-file checkpoints/benchmark2_full_sbso/gptq_calibration_texts.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._script_common import build_run
from src.finetuning.export_gguf import export_to_gguf
from src.finetuning.merge_adapters import merge_lora_adapters
from src.finetuning.quantize_gguf import quantize_gguf
from src.finetuning.quantize_gptq import quantize_gptq


def _load_calibration_texts(path: str | None) -> list[str]:
    if not path:
        return []
    lines = Path(path).read_text().splitlines()
    return [line for line in (l.strip() for l in lines) if line]


def _add_export_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True, help="Root dir - subfolders created for each artifact")
    parser.add_argument("--calibration-texts-file", default=None, help="One calibration text per line, for GPTQ")
    parser.add_argument("--skip-gptq", action="store_true", help="Skip the cloud/GPTQ branch even if CUDA is available")


def main() -> None:
    config, ctx, args = build_run(
        phase="export_pipeline", description=__doc__, extra_args=_add_export_pipeline_args,
    )
    print(f"[export] run_id={ctx.run_id}")

    base_model_path = config["base_model_path"]
    llama_cpp_convert_script = config["llama_cpp_convert_script"]
    llama_quantize_binary = config.get("llama_quantize_binary")
    gguf_quant_type = config.get("gguf_quant_type", "Q4_K_M")

    output_dir = Path(args.output_dir)
    merged_dir = output_dir / "merged_fp16"
    gguf_f16_path = output_dir / "model-f16.gguf"
    gguf_quant_path = output_dir / f"model-{gguf_quant_type.lower()}.gguf"
    gptq_dir = output_dir / "gptq_int4"

    print(f"[1/3] Merging LoRA adapter into base model -> {merged_dir}")
    merge_lora_adapters(base_model_path, args.adapter_path, merged_dir)

    print(f"[2/3] Edge branch: GGUF f16 export -> {gguf_f16_path}")
    export_to_gguf(merged_dir, gguf_f16_path, llama_cpp_convert_script)
    print(f"[2/3] Edge branch: {gguf_quant_type} quantization -> {gguf_quant_path}")
    quantize_gguf(gguf_f16_path, gguf_quant_path, llama_quantize_binary, gguf_quant_type)
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