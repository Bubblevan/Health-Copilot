"""Answer-schema parsing and response-side identity checks."""

from __future__ import annotations

import re

from .contracts import AnswerSchema

_ABSTENTION_MARKERS = (
    "insufficient evidence", "cannot determine", "can't determine", "unable to determine",
    "cannot reliably answer", "not enough information", "无法判断", "证据不足", "无法可靠",
)


def parse_answer(text: str, schema: AnswerSchema) -> str | tuple[str, ...] | None:
    answer = text.strip()
    if not answer:
        return None
    if schema in {AnswerSchema.FREE_TEXT, AnswerSchema.ABSTAINABLE}:
        if schema is AnswerSchema.ABSTAINABLE and any(
            marker in answer.casefold() for marker in _ABSTENTION_MARKERS
        ):
            return None
        return answer
    if schema is AnswerSchema.EXACT_TOKEN:
        match = re.search(r"[A-Za-z0-9_.:-]+", answer)
        return match.group(0) if match else None
    if schema is AnswerSchema.SINGLE_CHOICE:
        match = re.match(r"\s*(?:answer\s*[:=-]?\s*)?[\[(]?([A-Z])(?:[\]).:),\s]|$)",
                         answer, flags=re.IGNORECASE)
        return match.group(1).upper() if match else None
    if schema is AnswerSchema.MULTI_SELECT:
        candidate = answer.splitlines()[0].strip()
        match = re.match(
            r"(?:answer\s*[:=-]?\s*)?[\[(]?"
            r"([A-Z](?:[A-Z]|\s*[,/&+]\s*[A-Z])*)"
            r"[\])]?(?=$|\s|[.;:])",
            candidate,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        return tuple(sorted(set(re.findall(r"[A-Z]", match.group(1).upper()))))
    raise ValueError(f"unsupported answer schema: {schema}")
