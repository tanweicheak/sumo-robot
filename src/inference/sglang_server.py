"""
src.inference.sglang_server

Phase: Phase 4 (Stage 3)
Purpose: Real SLM backend via SGLang's native HTTP server. Implements the same
    SLMClient contract as MockSLMClient/LlamaCppSLMClient, so agents/MCTS are unchanged.
    IMPORTANT: SGLang serves HuggingFace-format checkpoints, NOT llama.cpp GGUF - point
    this at the same HF directory used for LoRA fine-tuning (Stage 2), not the Phase 3
    GGUF. generate_structured_batch fires requests CONCURRENTLY via a thread pool so
    SGLang's continuous batching engages server-side - this is the entire reason to use
    SGLang over llama.cpp for MCTS rollout sampling; the single-call path alone gets no
    benefit and should only be used for one-off calls (e.g. the final chosen action).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Type

from pydantic import BaseModel

from src.inference.grammar import enum_regex_pattern, primary_enum_field
from src.inference.slm_client import SLMClient

logger = logging.getLogger(__name__)


class SGLangSLMClient(SLMClient):
    def __init__(
        self,
        server_url: str,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout_s: float = 30.0,
        max_concurrency: int = 16,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self._model_path = model_path
        self._tokenizer = None   # lazy-loaded in _chat_format, not per-call
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_concurrency = max_concurrency
        self.call_count = 0
        self.fallback_count = 0   # incremented in _parse() when the enum-constrained
                                  # decode fails to parse and falls back to values[0] -
                                  # should stay at 0 if the grammar/regex constraint is
                                  # actually working; a nonzero, growing count means
                                  # results are silently defaulting, not real model output

    def generate_structured(
        self, prompt: str, schema: Type[BaseModel], max_new_tokens: int | None = None, **kwargs
    ) -> BaseModel:
        field_name, enum_cls = primary_enum_field(schema)
        if enum_cls is None:
            return schema()
        values = [e.value for e in enum_cls]
        payload = self._build_payload(prompt, values, max_new_tokens)
        data = self._post(payload)
        return self._parse(data, schema, field_name, enum_cls, values)

    def generate_structured_batch(
        self, requests_: list[tuple[str, Type[BaseModel]]]
    ) -> list[BaseModel]:
        """requests_: list of (prompt, schema) pairs. Dispatched concurrently so
        SGLang's continuous batcher can combine them into fewer forward passes.
        Returns results in the same order as the input list."""
        results: list[BaseModel | None] = [None] * len(requests_)

        def _one(i: int, prompt: str, schema: Type[BaseModel]) -> None:
            results[i] = self.generate_structured(prompt, schema)

        workers = min(self.max_concurrency, max(1, len(requests_)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_one, i, p, s) for i, (p, s) in enumerate(requests_)]
            for f in futures:
                f.result()   # propagate any exception
        return results  # type: ignore[return-value]

    def _chat_format(self, prompt: str) -> str:
        """Wraps `prompt` in the model's REAL chat template instead of sending it as
        raw completion text to SGLang's native /generate endpoint (a low-level
        text-completion endpoint - it does NOT apply chat formatting automatically).
        Real fix for the "model doesn't know where to stop generating" failure mode
        confirmed in held-out generation testing (e.g. 'evade_edge\\nmom=still\\n'
        instead of stopping after 'evade_edge') - no turn-boundary signal existed in
        plain concatenated text. Tokenizer loaded once, lazily, and cached - not
        per-call, given how many calls this client makes."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )

    def _build_payload(self, prompt: str, values: list[str], max_new_tokens: int | None) -> dict:
        return {
            "text": self._chat_format(prompt),
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": max_new_tokens or self.max_tokens,
                "regex": enum_regex_pattern(values),
            },
        }

    def _post(self, payload: dict) -> dict:
        import requests

        self.call_count += 1
        resp = requests.post(f"{self.server_url}/generate", json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def _parse(self, data: dict, schema, field_name: str, enum_cls, values: list[str]) -> BaseModel:
        text = (data.get("text") or "").strip()
        try:
            value = enum_cls(text)
        except ValueError:
            self.fallback_count += 1
            logger.warning(
                "SGLangSLMClient: constraint fallback fired (call %d, fallback %d so far) - "
                "raw text=%r did not match any %s value, defaulting to %r. If this fires "
                "often, the grammar/regex constraint is not actually working.",
                self.call_count, self.fallback_count, text, enum_cls.__name__, values[0],
            )
            value = enum_cls(values[0])   # regex constraint should prevent this
        return schema(**{field_name: value})