"""Deterministic answer normalization for public MCQ evaluation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from ..harness.contracts import AnswerSchema
from ..harness.verification import parse_answer


def canonical_option_set(value: object, *, allowed: Iterable[str] | None = None) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        candidates = value.keys()
    elif isinstance(value, (list, tuple, set)):
        candidates = value
    elif isinstance(value, str):
        text = value.strip().upper()
        text = re.sub(r"^(?:RIGHT\s+OPTION|ANSWER|OPTION)\s*[:=-]?\s*", "", text)
        text = text.strip(" \t\r\n[](){}'\"")
        # Public snapshots encode correct labels as A, AC, A,C, or "A. option text".
        compact = re.fullmatch(r"([A-Z](?:\s*[,/&+]\s*[A-Z]|\s*[A-Z])*)", text)
        labeled = re.fullmatch(r"([A-Z])(?:[.)]\s*.+)", text)
        if compact:
            candidates = re.findall(r"[A-Z]", compact.group(1))
        elif labeled:
            candidates = (labeled.group(1),)
        else:
            raise ValueError("choice answer must contain only option labels")
    else:
        raise TypeError("choice answer must be a string or sequence")
    result = tuple(sorted({str(item).strip().upper() for item in candidates if str(item).strip()}))
    if not result or any(not re.fullmatch(r"[A-Z]", item) for item in result):
        raise ValueError("choice answer contains no canonical option labels")
    allowed_set = {str(item).strip().upper() for item in allowed} if allowed is not None else None
    if allowed_set is not None and not set(result).issubset(allowed_set):
        raise ValueError("choice answer references an unknown option")
    return result


def parse_choice(text: str, schema: AnswerSchema) -> str | tuple[str, ...] | None:
    value = parse_answer(text, schema)
    if schema is AnswerSchema.SINGLE_CHOICE:
        return value if isinstance(value, str) and len(value) == 1 else None
    if schema is AnswerSchema.MULTI_SELECT:
        return value if isinstance(value, tuple) else None
    return None
