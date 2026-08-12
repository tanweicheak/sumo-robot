"""
src.inference.llama_cpp_client

Phase: Phase 3
Purpose: Real SLM backend via llama-cpp-python (report's specified llama.cpp + GGUF
    stack). Implements the SLMClient contract, so agents and the controller are
    unchanged from Phase 2. Latency mitigations: single-token GBNF-constrained output
    per call, tight max_tokens, grammar caching, and greedy (temperature 0) decoding.
    llama.cpp's built-in prompt-prefix KV caching is exploited by keeping each agent's
    system prompt constant across calls (Phase 2 prompts already do this).
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

from pydantic import BaseModel

from src.inference.grammar import enum_choice_grammar, primary_enum_field
from src.inference.slm_client import SLMClient


class LlamaCppSLMClient(SLMClient):
    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,   # -1 = offload all to Metal/GPU where available
        temperature: float = 0.0,  # greedy for deterministic, fast tactical decisions
        max_tokens: int = 8,       # enum values are short; tight cap for speed
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.verbose = verbose
        self.call_count = 0
        self._llm = None
        self._grammar_cache: dict[str, object] = {}

    def _ensure_loaded(self):
        if self._llm is None:
            from llama_cpp import Llama

            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                verbose=self.verbose,
            )
        return self._llm

    def _grammar_for(self, cache_key: str, gbnf_text: str):
        if cache_key not in self._grammar_cache:
            from llama_cpp import LlamaGrammar

            self._grammar_cache[cache_key] = LlamaGrammar.from_string(gbnf_text)
        return self._grammar_cache[cache_key]

    def generate_structured(
        self,
        prompt: str,
        schema: Type[BaseModel],
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> BaseModel:
        self.call_count += 1
        field_name, enum_cls = primary_enum_field(schema)
        if enum_cls is None:
            return schema()   # no enum field to constrain; return defaults

        values = [e.value for e in enum_cls]
        grammar = self._grammar_for(enum_cls.__name__, enum_choice_grammar(values))
        llm = self._ensure_loaded()

        out = llm.create_completion(
            prompt=prompt,
            grammar=grammar,
            max_tokens=max_new_tokens or self.max_tokens,
            temperature=self.temperature,
        )
        text = out["choices"][0]["text"].strip()
        try:
            value = enum_cls(text)
        except ValueError:
            value = enum_cls(values[0])   # grammar should prevent this, but be safe
        return schema(**{field_name: value})