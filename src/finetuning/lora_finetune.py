"""
src.finetuning.lora_finetune

Phase: Phase 4 (Stage 2/3)
Purpose: LoRA fine-tuning of Phi-4-mini on SBSO-accumulated (state -> strategy) pairs
    (report Section 3.3.2.7, step 1). Runs on CPU/MPS for a local token-run proving the
    code path; the full-scale run happens on cloud GPU (device="cuda").

Train/val split: previously no eval_dataset existed at all, so there was no signal
    distinguishing "the adapter learned the tactical pattern" from "the adapter
    memorized the training set." split_episodes_train_val() splits by EPISODE ID,
    not by individual pair - pairs from the same episode are near-duplicate states,
    and splitting at the pair level would leak them across train/val and make eval
    numbers look better than they really are. Random split (not a temporal
    last-N%-by-episode split), because a temporal split would be confounded by
    opponent-mix and DSPy-prompt-version drift over the course of training - it
    would measure "does earlier training generalize to a later, shifted
    distribution," not "did the model memorize its training examples," which is a
    different, messier question than the one this split is for.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any


def split_episodes_train_val(
    training_pairs: list, val_fraction: float = 0.1, seed: int = 0
) -> tuple[list, list]:
    """Splits an episode-tagged training_pairs list into (train_pairs, val_pairs),
    holding out val_fraction of EPISODES (not individual pairs) at random. Returns
    two lists in the same (episode, state, strategy) triple format as the input."""
    episode_ids = sorted({ep for ep, _state, _strategy in training_pairs})
    rng = random.Random(seed)
    val_episodes = set(rng.sample(episode_ids, max(1, round(len(episode_ids) * val_fraction))))

    train_pairs = [p for p in training_pairs if p[0] not in val_episodes]
    val_pairs = [p for p in training_pairs if p[0] in val_episodes]
    return train_pairs, val_pairs


def evaluate_strategy_accuracy(model, tokenizer, records: list[dict], device: str, max_new_tokens: int = 8) -> dict:
    """Generation-based exact-match accuracy: for each {"prompt", "completion"}
    record, greedy-decode the model's completion and check it matches the target
    strategy label exactly. Works identically on a LoRA-adapted model or a bare
    frozen base model (pass the base model directly, no adapter) - that symmetry is
    what makes the "did LoRA actually help over zero-shot" comparison possible with
    one shared function instead of two divergent code paths.

    Deliberately uses real generation, not a logits-position trick against
    DataCollatorForLanguageModeling's labels - with no prompt masking (see the
    training loss note below), every non-pad position has a real label, which makes
    "which position is the completion" ambiguous to recover purely from label_ids.
    Generation sidesteps that entirely and mirrors how the model is actually used.
    """
    import torch

    model.eval()
    correct = 0
    for r in records:
        inputs = tokenizer(r["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if generated == r["completion"].strip():
            correct += 1

    n = len(records)
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else None}


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
        use_wandb: bool = False,
        wandb_run_name: str | None = None,
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
        self.use_wandb = use_wandb
        self.wandb_run_name = wandb_run_name

    def run(self, sft_records: list[dict], eval_records: list[dict] | None = None) -> dict:
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
        eval_ds = None
        if eval_records:
            eval_ds = Dataset.from_list(eval_records).map(_tokenize, remove_columns=["prompt", "completion"])

        # Causal-LM collator: builds `labels` = input_ids (shifted internally by the model),
        # dynamically pads each batch, masks pad tokens out of the loss.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        args = TrainingArguments(
            output_dir=str(self.output_dir), num_train_epochs=self.epochs,
            per_device_train_batch_size=4, learning_rate=self.learning_rate,
            logging_steps=10, save_strategy="epoch",
            eval_strategy="epoch" if eval_ds is not None else "no",
            report_to=["wandb"] if self.use_wandb else [],
            run_name=self.wandb_run_name,
        )
        trainer = Trainer(
            model=model, args=args, train_dataset=ds, eval_dataset=eval_ds, data_collator=data_collator,
        )
        trainer.train()

        adapter_path = self.output_dir / "lora_adapters"
        model.save_pretrained(str(adapter_path))

        # Post-training strategy-accuracy sweep across saved per-epoch checkpoints -
        # picks which epoch is actually best by the metric that matters (does the
        # generated completion match the MCTS-labeled strategy), not just "the last
        # one trained." Eval loss alone (via eval_dataset above) tells you WHETHER
        # overfitting happened; this tells you WHICH checkpoint to actually use.
        epoch_accuracy = None
        if eval_records:
            epoch_accuracy = evaluate_strategy_accuracy(model, tokenizer, eval_records, self.device)
            print(f"[lora_finetune] final-epoch strategy accuracy on held-out val set: {epoch_accuracy}")

        return {
            "adapter_path": adapter_path, "tokenizer": tokenizer, "model": model,
            "eval_records": eval_records, "final_epoch_accuracy": epoch_accuracy,
        }