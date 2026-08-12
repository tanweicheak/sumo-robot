"""
src.inference.slm_client

Phase: Phase 2
Purpose: SLM inference abstraction. Agents (OAA, SA, TEA) depend only on the SLMClient
    interface, so Phase 3 can swap the MockSLMClient for a real llama.cpp / SGLang +
    Outlines backend without touching agent code. The mock returns schema-valid outputs
    so the whole LangGraph pipeline is testable now, with an optional synthetic latency
    to exercise the timing/HCMA path.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Type

from pydantic import BaseModel


class SLMClient(ABC):
    """Contract for all SLM backends. generate_structured must return an instance of
    `schema` (constrained decoding guarantees validity in the real Phase 3 backend;
    the mock synthesizes a valid instance directly)."""

    @abstractmethod
    def generate_structured(self, prompt: str, schema: Type[BaseModel], **kwargs) -> BaseModel:
        ...


class MockSLMClient(SLMClient):
    """Deterministic-by-seed mock. Returns a schema-valid instance with fields filled
    by type: enum fields get a random choice, floats/ints/strs get placeholders.
    Optional latency_ms simulates inference cost for latency-path testing."""

    def __init__(self, seed: int = 0, latency_ms: float = 0.0) -> None:
        self._rng = random.Random(seed)
        self.latency_ms = latency_ms
        self.call_count = 0

    def generate_structured(self, prompt: str, schema: Type[BaseModel], **kwargs) -> BaseModel:
        self.call_count += 1
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        return self._synthesize(schema)

    def _synthesize(self, schema: Type[BaseModel]) -> BaseModel:
        fields = {}
        for name, field in schema.model_fields.items():
            ann = field.annotation
            if isinstance(ann, type) and issubclass(ann, Enum):
                fields[name] = self._rng.choice(list(ann))
            elif ann is float:
                fields[name] = round(self._rng.uniform(0.0, 1.0), 3)
            elif ann is int:
                fields[name] = self._rng.randint(0, 10)
            elif ann is str:
                fields[name] = "mock"
        try:
            return schema(**fields)
        except Exception:
            return schema()   # fall back to schema defaults if construction fails


class ScriptedSLMClient(SLMClient):
    """A mock that returns a fixed, caller-specified instance per schema type. Useful
    for deterministic end-to-end tests where you want a known OAA/SA/TEA output rather
    than a random one."""

    def __init__(self, responses: dict[Type[BaseModel], BaseModel]) -> None:
        self.responses = responses
        self.call_count = 0

    def generate_structured(self, prompt: str, schema: Type[BaseModel], **kwargs) -> BaseModel:
        self.call_count += 1
        if schema in self.responses:
            return self.responses[schema].model_copy(deep=True)
        return schema()