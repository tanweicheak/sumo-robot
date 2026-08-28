"""
scripts.run_lora_finetune

Purpose: The missing piece between a completed src/sbso data-collection run
    (run_phase4_pilot.py - produces training_pairs.jsonl) and a usable LoRA adapter
    (what merge_adapters.py / run_export_pipeline.py expect to find). No prior script
    in this project drives LoRAFineTuner against a REAL, full-scale completed run's
    data - run_phase4_stage2_token_run.py is an explicit tiny local proof-of-concept
    (--episodes 2 default, --lora-device mps default) whose own module docstring
    describes it as "Local token-run proving every Stage 2 code path executes for
    real, at tiny scale, before renting a GPU" - not the real driver. This script
    reuses the exact same LoRAFineTuner/build_sft_dataset/split_episodes_train_val
    interface that script already proved correct, pointed at a real completed run's
    full training_pairs.jsonl instead.

Usage (on the RunPod pod, after a real run_phase4_pilot.py run has completed):
    python -m scripts.run_lora_finetune \\
        --training-pairs checkpoints/benchmark2_full_sbso/training_pairs.jsonl \\
        --hf-model-path /workspace/models/phi-4-mini-hf \\
        --output-dir checkpoints/benchmark2_full_sbso \\
        --use-wandb

Writes the adapter to <output-dir>/lora_run/lora_adapters - matching exactly what
merge_adapters.py / run_export_pipeline.py already expect at that path.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.common.config_loader import load_config

logger = logging.getLogger(__name__)


def _load_training_pairs(path: str) -> list:
    """training_pairs.jsonl is one JSON object per line: {"episode": int,
    "lssd_text": str, "strategy": str, "opponent_type": str|None} - confirmed
    against run_phase4_pilot.py's own _on_episode_end write format. LoRAFineTuner's
    pipeline (split_episodes_train_val -> build_sft_dataset) expects
    (episode, state, strategy) tuples, matching MatchLevelSBSOTrainer.training_pairs'
    real in-memory shape - reconstructed here from the JSONL file's flat dict form."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            state = {"lssd": d.get("lssd_text", "")}
            pairs.append((d["episode"], state, d["strategy"]))
    return pairs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real, full-scale LoRA fine-tune from a completed run's training_pairs.jsonl.")
    p.add_argument("--training-pairs", required=True, help="Path to the completed run's training_pairs.jsonl")
    p.add_argument("--hf-model-path", required=True, help="Base HF model directory (the one used for SGLang serving)")
    p.add_argument("--output-dir", required=True, help="Same directory the run's other artifacts live in - adapter written to <this>/lora_run/lora_adapters")
    p.add_argument("--finetuning-config", default="config/finetuning.yaml",
                    help="LoRA hyperparameters (rank/alpha/target_modules/learning_rate) - see finetuning.yaml")
    p.add_argument("--device", default="cuda", help="'cuda' on RunPod (default), 'mps' only for local Mac runs")
    p.add_argument("--epochs", type=int, default=None, help="Override finetuning.yaml's epoch count if set")
    p.add_argument("--seed", type=int, default=0, help="Seed for the train/val episode split")
    p.add_argument("--use-wandb", action="store_true", help="Report LoRA training metrics to Weights & Biases")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = build_arg_parser().parse_args()

    from src.finetuning.lora_finetune import (
        LoRAFineTuner, build_sft_dataset, evaluate_strategy_accuracy, split_episodes_train_val,
    )

    logger.info(f"loading training pairs from {args.training_pairs} ...")
    training_pairs = _load_training_pairs(args.training_pairs)
    logger.info(f"{len(training_pairs)} training pair(s) loaded, spanning "
                f"{len({ep for ep, _, _ in training_pairs})} episode(s)")

    prompt_template = "State: {state}\nStrategy:"
    train_pairs, val_pairs = split_episodes_train_val(training_pairs, val_fraction=0.1, seed=args.seed)
    sft_records = build_sft_dataset(train_pairs, prompt_template=prompt_template)
    eval_records = build_sft_dataset(val_pairs, prompt_template=prompt_template) if val_pairs else None
    logger.info(f"{len(sft_records)} train SFT record(s), "
                f"{len(eval_records) if eval_records else 0} held-out val record(s)")

    out_dir = Path(args.output_dir)
    ft_config = load_config(args.finetuning_config)
    lora_cfg = ft_config.get("lora", {})
    tuner = LoRAFineTuner(
        base_model_path=args.hf_model_path, output_dir=out_dir / "lora_run",
        rank=int(lora_cfg.get("rank", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        target_modules=lora_cfg.get("target_modules"),   # None -> LoRAFineTuner's own
                                                            # fused-QKV-correct default
        learning_rate=float(lora_cfg.get("learning_rate", 2e-4)),
        epochs=args.epochs if args.epochs is not None else int(lora_cfg.get("epochs", 3)),
        device=args.device,
        use_wandb=args.use_wandb, wandb_run_name=f"lora-{out_dir.name}",
    )

    logger.info(f"starting LoRA fine-tune (device={args.device}, rank={tuner.rank}, "
                f"target_modules={tuner.target_modules})...")
    result = tuner.run(sft_records, eval_records=eval_records)
    adapter_path = result["adapter_path"]
    logger.info(f"LoRA adapters saved -> {adapter_path}")

    if eval_records:
        logger.info("evaluating strategy accuracy on held-out episodes...")
        # NOTE: matches run_phase4_stage2_token_run.py's own pattern - evaluate_
        # strategy_accuracy's exact signature (model/tokenizer objects, not paths)
        # means this needs the just-trained model still in memory. If tuner.run()
        # doesn't already return them, this call may need adjustment once actually
        # run - flagged here rather than guessed silently.
        try:
            acc = evaluate_strategy_accuracy(
                result.get("model"), result.get("tokenizer"), eval_records, device=args.device,
            )
            logger.info(f"held-out strategy accuracy: {acc}")
        except Exception as e:
            logger.warning(f"post-training accuracy eval skipped ({type(e).__name__}: {e}) - "
                            "adapter is already saved regardless, this is diagnostic-only")

    logger.info(f"DONE. Next step: merge_lora_adapters(base_model_path={args.hf_model_path!r}, "
                f"adapter_path={str(adapter_path)!r}, output_dir={str(out_dir / 'merged_fp16')!r})")


if __name__ == "__main__":
    main()
