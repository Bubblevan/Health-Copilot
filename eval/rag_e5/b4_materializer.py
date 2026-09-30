"""Deterministic state and provenance materialization for RAG-E5-B4."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

_TREND_VALUES = frozenset({"rising", "falling", "stable"})
_ALIAS_PATTERN = re.compile(r"\[(E\d+)\]")


@dataclass(frozen=True, slots=True)
class StateClaim:
    field: str
    value: str
    source: str = "longitudinal_state"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def classify_runtime_task(case: Mapping[str, Any]) -> str:
    """Classify only from runtime-visible question/state references, never teacher labels."""
    question = case.get("question")
    state_ref = case.get("state_packet_ref")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("runtime question must be non-empty text")
    if state_ref is None:
        if question.startswith("What general "):
            return "GUIDANCE_ONLY"
        raise ValueError("state-free runtime question does not match a frozen guidance task")
    if question.startswith("Describe only the recorded "):
        return "STATE_ONLY"
    if question.startswith("First, describe only the recorded "):
        return "STATE_AND_GUIDANCE"
    raise ValueError("state-bearing runtime question does not match a frozen task template")


def materialize_state_claims(
    question: str, state_packet: Mapping[str, Any] | None
) -> tuple[StateClaim, ...]:
    """Project just the metric explicitly requested in the question from the packet."""
    if state_packet is None:
        return ()
    normalized = question.casefold()
    if "body mass index" in normalized or re.search(r"\bbmi\b", normalized):
        requested_metric = "body_mass_index"
    elif "body weight" in normalized:
        requested_metric = "body_weight"
    else:
        raise ValueError("state-bearing question does not name an allowlisted metric")
    trends = state_packet.get("recent_measurement_trends")
    if not isinstance(trends, Sequence) or isinstance(trends, (str, bytes)):
        raise TypeError("state packet recent_measurement_trends must be a sequence")
    matches = [
        row
        for row in trends
        if isinstance(row, Mapping) and row.get("metric") == requested_metric
    ]
    if len(matches) != 1:
        raise ValueError("required decision-time state metric is missing or duplicated")
    trend = matches[0].get("trend")
    if not isinstance(trend, str) or trend not in _TREND_VALUES:
        raise ValueError("required state trend is outside the frozen enum")
    return (StateClaim(field=requested_metric, value=trend),)


def state_claim_text(claim: StateClaim) -> str:
    label = claim.field.replace("_", " ")
    return f"Recorded {label} trend: {claim.value}."


def extract_citations(
    text: str, alias_map: Sequence[Mapping[str, Any]]
) -> tuple[list[str], list[str]]:
    """Resolve only issued aliases; unknown E-number aliases are ignored and counted."""
    known = {
        row["alias"]: row["chunk_id"]
        for row in alias_map
        if isinstance(row.get("alias"), str) and isinstance(row.get("chunk_id"), str)
    }
    observed = _ALIAS_PATTERN.findall(text)
    valid = list(dict.fromkeys(known[alias] for alias in observed if alias in known))
    invented = sorted({alias for alias in observed if alias not in known})
    return valid, invented


def strip_aliases(text: str) -> str:
    return _ALIAS_PATTERN.sub("", text).strip()


def materialize_final_response(
    *,
    task_kind: str,
    state_claims: Sequence[StateClaim],
    guidance_text: str,
    alias_map: Sequence[Mapping[str, Any]],
) -> str:
    """Render final user text and replace aliases with deterministic chunk provenance."""
    claim_lines = [state_claim_text(claim) for claim in state_claims]
    if task_kind == "STATE_ONLY":
        if len(claim_lines) != 1:
            raise ValueError("state-only tasks require exactly one deterministic state claim")
        return claim_lines[0]
    if task_kind not in {"GUIDANCE_ONLY", "STATE_AND_GUIDANCE"}:
        raise ValueError(f"unknown runtime task kind: {task_kind}")

    known = {
        row["alias"]: row
        for row in alias_map
        if isinstance(row.get("alias"), str) and isinstance(row.get("chunk_id"), str)
    }

    def replace(match: re.Match[str]) -> str:
        row = known.get(match.group(1))
        if row is None:
            return ""
        section = row.get("section_path", [])
        section_text = " > ".join(str(item) for item in section) if section else "reference"
        return f"[Source: {row['source_id']}; {section_text}; chunk {row['chunk_id']}]"

    resolved_guidance = _ALIAS_PATTERN.sub(replace, guidance_text).strip()
    if task_kind == "GUIDANCE_ONLY":
        return resolved_guidance
    if len(claim_lines) != 1:
        raise ValueError("state-and-guidance tasks require exactly one deterministic state claim")
    return f"{claim_lines[0]}\n\nGeneral guidance:\n{resolved_guidance}".strip()


__all__ = [
    "StateClaim",
    "classify_runtime_task",
    "extract_citations",
    "materialize_final_response",
    "materialize_state_claims",
    "state_claim_text",
    "strip_aliases",
]
