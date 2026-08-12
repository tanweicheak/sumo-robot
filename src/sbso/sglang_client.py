"""
src.inference.sglang_client

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

from concurrent.futures import ThreadPoolExecutor
from typing import Type

from pydantic import BaseModel

from src.inference.grammar import enum_regex_pattern, primary_enum_field
from src.inference.slm_client import SLMClient


class SGLangSLMClient(SLMClient):
    def __init__(
        self,
        server_url: str,
        temperature: float = 0.0,
        max_tokens: int = 8,
        timeout_s: float = 30.0,
        max_concurrency: int = 16,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_concurrency = max_concurrency
        self.call_count = 0

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

    def _build_payload(self, prompt: str, values: list[str], max_new_tokens: int | None) -> dict:
        return {
            "text": prompt,
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
            value = enum_cls(values[0])   # regex constraint should prevent this
        return schema(**{field_name: value})