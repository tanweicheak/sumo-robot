"""
src.finetuning.merge_adapters

Phase: Phase 4 (Stage 2/3)
Purpose: Merge LoRA adapters into the base model at full fp16 precision (report Section
    3.3.2.7, step 2). MUST run before quantization - reversing this order destroys the
    fine-tuned detail (report §2.2.5, K-quant super-block/sub-block scale calculation
    assumes full-precision merged weights).
"""

from __future__ import annotations

from pathlib import Path


def merge_lora_adapters(base_model_path: str | Path, adapter_path: str | Path, output_dir: str | Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(str(base_model_path), torch_dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, str(adapter_path))
    merged = merged.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir