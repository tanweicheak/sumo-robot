"""
src.inference.grammar

Phase: Phase 3
Purpose: GBNF grammar construction for constrained decoding. Each SLM agent emits a
    single categorical token (its enum value), not a JSON object - this is both far
    faster (one constrained token vs. a full object) and guarantees a valid output, so
    it directly serves the "as fast as possible" latency goal while giving the same
    schema guarantee Outlines would. Pure string generation; no llama.cpp needed to test.
"""

from __future__ import annotations

from enum import Enum
from typing import Type

from pydantic import BaseModel

import re

def enum_choice_grammar(values: list[str]) -> str:
    """Build a GBNF grammar constraining output to exactly one of `values`."""
    if not values:
        raise ValueError("enum_choice_grammar requires at least one value")
    alternation = " | ".join(f'"{v}"' for v in values)
    return f"root ::= {alternation}\n"

def enum_regex_pattern(values: list[str]) -> str:
    """Regex-alternation constraint for SGLang (which does NOT consume GBNF). Same
    purpose as enum_choice_grammar: force output to exactly one enum value.
    Longest-first ordering prevents a short value (e.g. "charge") from short-matching
    a prefix of a longer one (e.g. "charge_forward") during alternation."""
    if not values:
        raise ValueError("enum_regex_pattern requires at least one value")
    ordered = sorted(values, key=len, reverse=True)
    escaped = [re.escape(v) for v in ordered]
    return "(" + "|".join(escaped) + ")"


def primary_enum_field(schema: Type[BaseModel]) -> tuple[str | None, Type[Enum] | None]:
    """Return (field_name, enum_class) of the first Enum-typed field in `schema`, or
    (None, None). This is the single field the SLM generates; the rest are filled by
    code (e.g. OAA's frame_stamp is stamped by the agent, not the model)."""
    for name, field in schema.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, Enum):
            return name, ann
    return None, None