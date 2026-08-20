"""
scripts.validate_quantization_quality

Phase: end-of-Phase-4, before trusting any Phase 5 result on the quantized artifact
Purpose: Benchmark 2's checkpoint is quantized exactly once, at the end of training
    (LoRA -> merge fp16 -> GGUF f16 -> llama-quantize Q4_K_M), and Phase 5 correctly
    evaluates the FINAL quantized artifact (the thing you'd actually ship), not the
    pre-quantization fp16 model. That's the right design - but it also means if Phase
    5's numbers come in weaker than expected, there is currently no way to tell
    whether that's "SBSO didn't learn a good policy" or "quantization degraded a
    policy that was actually fine." This script isolates the quantization step alone:
    both models loaded via the same llama.cpp backend, same grammar-constrained
    decoding, same validation prompts - so any difference in output is attributable
    to Q4_K_M quantization specifically, not to the HF-to-GGUF conversion or to any
    change in prompt/grammar between the two.

Usage:
    python -m scripts.validate_quantization_quality \\
        --fp16-gguf-path checkpoints/benchmark2_full_sbso/model-f16.gguf \\
        --quantized-gguf-path checkpoints/benchmark2_full_sbso/model-Q4_K_M.gguf \\
        --validation-prompts checkpoints/benchmark2_full_sbso/training_pairs.jsonl \\
        --n-samples 100

    --validation-prompts accepts a training_pairs.jsonl (this script pulls
    lssd_text/prompt fields from a random, seeded sample - deliberately held out
    from LoRA training would be even better if you kept a split; a random sample
    from the same file is still informative about agreement, just not a true
    held-out generalization check).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from src.agents.schemas import MacroStrategy
from src.inference.grammar import enum_regex_pattern

STRATEGY_VALUES = [s.value for s in MacroStrategy]
PROMPT_TEMPLATE = "State: {state}\nStrategy:"


def _load_validation_prompts(path: str | Path, n_samples: int, seed: int) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} has no rows to sample from")

    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_samples, len(rows)))
    return [PROMPT_TEMPLATE.format(state=r["lssd_text"]) for r in sample]


class _GGUFModel:
    """Thin wrapper matching LlamaCppJudge's own loading convention (llama_cpp Python
    bindings, grammar-constrained single-token-class output) - reused here rather than
    reinventing a second loading path."""

    def __init__(self, model_path: str, n_ctx: int = 1024, n_gpu_layers: int = -1, seed: int = 0):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self._llm = None
        self._grammar = None

    def _ensure_loaded(self):
        if self._llm is None:
            from llama_cpp import Llama, LlamaGrammar

            self._llm = Llama(
                model_path=self.model_path, n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers, seed=self.seed, verbose=False,
            )
            gbnf = 'root ::= ' + " | ".join(f'"{v}"' for v in STRATEGY_VALUES) + "\n"
            self._grammar = LlamaGrammar.from_string(gbnf)
        return self._llm

    def generate(self, prompt: str) -> str:
        llm = self._ensure_loaded()
        out = llm.create_completion(
            prompt=prompt, grammar=self._grammar, max_tokens=8, temperature=0.0,
        )
        return out["choices"][0]["text"].strip()


def run_comparison(fp16_path: str, quantized_path: str, prompts: list[str]) -> dict:
    fp16_model = _GGUFModel(fp16_path)
    quantized_model = _GGUFModel(quantized_path)

    fp16_outputs, quantized_outputs = [], []
    agreements = 0
    fp16_invalid, quantized_invalid = 0, 0

    for i, prompt in enumerate(prompts):
        fp16_out = fp16_model.generate(prompt)
        quantized_out = quantized_model.generate(prompt)
        fp16_outputs.append(fp16_out)
        quantized_outputs.append(quantized_out)

        if fp16_out not in STRATEGY_VALUES:
            fp16_invalid += 1
        if quantized_out not in STRATEGY_VALUES:
            quantized_invalid += 1
        if fp16_out == quantized_out:
            agreements += 1
        print(f"[{i + 1}/{len(prompts)}] fp16={fp16_out:<12} quantized={quantized_out:<12}"
              f"{'  <- DISAGREE' if fp16_out != quantized_out else ''}")

    n = len(prompts)
    return {
        "n_samples": n,
        "agreement_rate": round(agreements / n, 4) if n else None,
        "fp16_schema_valid_rate": round(1 - fp16_invalid / n, 4) if n else None,
        "quantized_schema_valid_rate": round(1 - quantized_invalid / n, 4) if n else None,
        "fp16_distribution": dict(Counter(fp16_outputs)),
        "quantized_distribution": dict(Counter(quantized_outputs)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare fp16 vs quantized GGUF decision agreement.")
    p.add_argument("--fp16-gguf-path", required=True)
    p.add_argument("--quantized-gguf-path", required=True)
    p.add_argument("--validation-prompts", required=True, help="Path to a training_pairs.jsonl to sample from")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json-out", default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    prompts = _load_validation_prompts(args.validation_prompts, args.n_samples, args.seed)
    print(f"[validate_quantization_quality] {len(prompts)} validation prompts sampled from {args.validation_prompts}\n")

    result = run_comparison(args.fp16_gguf_path, args.quantized_gguf_path, prompts)

    print()
    print(f"Agreement rate:              {result['agreement_rate']:.1%}")
    print(f"fp16 schema-valid rate:      {result['fp16_schema_valid_rate']:.1%}")
    print(f"quantized schema-valid rate: {result['quantized_schema_valid_rate']:.1%}")
    print(f"fp16 distribution:           {result['fp16_distribution']}")
    print(f"quantized distribution:      {result['quantized_distribution']}")

    if result["agreement_rate"] is not None and result["agreement_rate"] < 0.85:
        print(
            "\nFLAG: agreement rate below 85% - a meaningful fraction of decisions change "
            "under quantization. If Phase 5 numbers look weak, quantization degradation "
            "is now a real candidate explanation, not just training quality - worth "
            "comparing which specific strategies disagree most (see per-strategy "
            "distributions above) before concluding SBSO training itself underperformed."
        )
    else:
        print(
            "\nAgreement rate looks healthy - quantization is unlikely to be the "
            "dominant explanation if Phase 5 numbers come in weaker than expected."
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"\nwrote full result -> {out_path}")


if __name__ == "__main__":
    main()
