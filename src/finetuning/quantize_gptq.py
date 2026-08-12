"""
src.finetuning.quantize_gptq

Phase: Phase 4 (Stage 3 - CLOUD ONLY)
Purpose: INT4 quantization of the merged model via GPTQ (report Section 3.3.2.7, step 3).
    GPTQ has no MPS/CPU kernel support - this step cannot run on the M4 Air. Locally,
    the pipeline stops after merge_adapters() and validates the fp16 merged model;
    quantization runs only on the RunPod RTX 4090.
"""

from __future__ import annotations

import platform
from pathlib import Path


def quantize_gptq(merged_model_path: str | Path, output_dir: str | Path, calibration_texts: list[str], bits: int = 4) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPTQ quantization requires CUDA and will not run on this machine "
            f"(platform={platform.system()}, cuda_available=False). This step is "
            "CLOUD-ONLY (RunPod RTX 4090). Locally, stop after merge_adapters() and "
            "validate the fp16 merged model instead."
        )

    from transformers import AutoTokenizer
    from gptqmodel import GPTQModel, QuantizeConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(merged_model_path))
    quant_config = QuantizeConfig(bits=bits, group_size=128)
    model = GPTQModel.load(str(merged_model_path), quant_config)
    model.quantize(calibration_texts, tokenizer=tokenizer)   # in-domain LSSD states (Q16)
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir