"""
src.finetuning.lora_finetune

Phase: Phase 4 (Stage 2/3)
Purpose: LoRA fine-tuning of Phi-4-mini on SBSO-accumulated (state -> strategy) pairs
    (report Section 3.3.2.7, step 1). Runs on CPU/MPS for a local token-run proving the
    code path; the full-scale run happens on cloud GPU (device="cuda").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_sft_dataset(training_pairs: list, prompt_template: str) -> list[dict]:
    """Convert (episode, state, strategy) triples from SBSOTrainer into prompt/completion
    records. training_pairs is episode-tagged (see match_trainer.py / training_loop.py);
    episode is not needed for SFT record construction and is discarded here."""
    records = []
    for _episode, state, strategy in training_pairs:
        state_desc = state.get("lssd", "") if isinstance(state, dict) else getattr(state, "lssd_text", "")
        strat_value = strategy.value if hasattr(strategy, "value") else str(strategy)
        records.append({"prompt": prompt_template.format(state=state_desc), "completion": strat_value})
    return records


class LoRAFineTuner:
    def __init__(
        self,
        base_model_path: str | Path,
        output_dir: str | Path,
        rank: int = 16,
        alpha: int = 32,
        target_modules: list[str] | None = None,
        learning_rate: float = 2e-4,
        epochs: int = 3,
        device: str = "cpu",   # "mps" for local Mac token-run, "cuda" for cloud
    ) -> None:
        self.base_model_path = str(base_model_path)
        self.output_dir = Path(output_dir)
        self.rank = rank
        self.alpha = alpha
        # Phi-3/Phi-4-mini's HF implementation fuses Q/K/V into one qkv_proj and gate/up
        # into one gate_up_proj (no separate q_proj/k_proj/v_proj exist on this
        # architecture). PEFT has no built-in auto-target mapping for phi3/phi4, and
        # since o_proj alone does exist, the old Llama-style default silently attached
        # LoRA to o_proj only and skipped Q/K/V adaptation entirely, with no error.
        self.target_modules = target_modules or ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device

    def run(self, sft_records: list[dict]) -> Path:
        from peft import LoraConfig, get_peft_model
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
        from datasets import Dataset

        self.output_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        if tokenizer.pad_token is None:
            # Many base LMs (Phi-4-mini included) ship without a pad token; reuse EOS.
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.base_model_path, torch_dtype=dtype)
        model.to(self.device)

        lora_cfg = LoraConfig(
            r=self.rank, lora_alpha=self.alpha, target_modules=self.target_modules,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

        def _tokenize(ex):
            # No manual padding/labels here - the data collator below handles both:
            # it pads each batch dynamically and derives `labels` from `input_ids`,
            # masking padded positions with -100 so they don't count toward loss.
            return tokenizer(ex["prompt"] + ex["completion"], truncation=True, max_length=512)

        ds = Dataset.from_list(sft_records).map(_tokenize, remove_columns=["prompt", "completion"])

        # Causal-LM collator: builds `labels` = input_ids (shifted internally by the model),
        # dynamically pads each batch, masks pad tokens out of the loss.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir=str(self.output_dir), num_train_epochs=self.epochs,
            per_device_train_batch_size=4, learning_rate=self.learning_rate,
            logging_steps=10, save_strategy="epoch", report_to=[],
        )
        trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=data_collator)
        trainer.train()

        adapter_path = self.output_dir / "lora_adapters"
        model.save_pretrained(str(adapter_path))
        return adapter_path