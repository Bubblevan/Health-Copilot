"""Claim-first E5-B3 reader contract and projection-only state view."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

STATE_FIELDS = ("body_weight", "body_mass_index")
TREND_VALUES = ("rising", "falling", "stable")
MAX_STATE_FACTS = 4
MAX_GUIDANCE_FACTS = 4
MAX_CITATIONS_PER_FACT = 4
MAX_STATEMENT_LENGTH = 320

READER_STATE_PROJECTION_SPEC = {
    "projection_id": "READER_STATE_VIEW_V1",
    "source_packet_version": "e5-longitudinal-state-v4",
    "fields": [
        "age_bucket",
        "condition_categories",
        "recent_measurement_trends",
        "available_exam_categories",
        "history_span_days",
        "recent_event_count",
    ],
    "projection_only": True,
}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


READER_STATE_PROJECTION_SHA256 = _canonical_sha256(READER_STATE_PROJECTION_SPEC)

READER_V2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["state_facts", "guidance_facts"],
    "properties": {
        "state_facts": {
            "type": "array",
            "maxItems": MAX_STATE_FACTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "value"],
                "properties": {
                    "field": {"type": "string", "enum": list(STATE_FIELDS)},
                    "value": {"type": "string", "enum": list(TREND_VALUES)},
                },
            },
        },
        "guidance_facts": {
            "type": "array",
            "maxItems": MAX_GUIDANCE_FACTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "citations"],
                "properties": {
                    "statement": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_STATEMENT_LENGTH,
                    },
                    "citations": {
                        "type": "array",
                        "maxItems": MAX_CITATIONS_PER_FACT,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
}

READER_V2_PROMPT = """You are a claim-first reader for a health information system.
Return ONLY one JSON object matching the supplied schema. Do not include an answer,
summary outside the fields, or top-level citations.

State contract:
- Report only facts explicitly present in the supplied ReaderStateView.
- Copy each state metric name exactly from recent_measurement_trends[].metric.
- Copy its trend exactly from recent_measurement_trends[].trend.
- Never rename a metric or infer a trend. If no state is supplied, return no state_facts.

Guidance contract:
- Write at most four short factual statements, one sentence each.
- Each statement must be supported by supplied passages, not general memory.
- Put citations on the specific guidance fact they support.
- Cite only supplied chunk IDs. If no passage supports a statement, omit it; if no
  passage is supplied, return an empty guidance_facts array.
- Never invent a citation. Do not give diagnosis or individualized treatment advice.

Question:
{QUESTION}

ReaderStateView:
{STATE_VIEW}

Approved reference passages:
{PASSAGES}
"""


@dataclass(frozen=True, slots=True)
class ReaderStateView:
    """Runtime-safe projection of packet v4; it adds no facts or teacher metadata."""

    age_bucket: str
    condition_categories: tuple[str, ...]
    recent_measurement_trends: tuple[tuple[str, str], ...]
    available_exam_categories: tuple[str, ...]
    history_span_days: int
    recent_event_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["condition_categories"] = list(self.condition_categories)
        value["recent_measurement_trends"] = [
            {"metric": metric, "trend": trend}
            for metric, trend in self.recent_measurement_trends
        ]
        value["available_exam_categories"] = list(self.available_exam_categories)
        return value

    @classmethod
    def from_packet(cls, packet: Mapping[str, Any]) -> ReaderStateView:
        if packet.get("summary_builder_version") != READER_STATE_PROJECTION_SPEC[
            "source_packet_version"
        ]:
            raise ValueError("ReaderStateView requires LongitudinalStatePacket v4")
        age_bucket = packet.get("age_bucket")
        history_span_days = packet.get("history_span_days")
        recent_event_count = packet.get("recent_event_count")
        categories = packet.get("condition_categories")
        exams = packet.get("available_exam_categories")
        trends = packet.get("recent_measurement_trends")
        if not isinstance(age_bucket, str):
            raise TypeError("packet age_bucket must be a string")
        if not isinstance(history_span_days, int) or isinstance(history_span_days, bool):
            raise TypeError("packet history_span_days must be an integer")
        if not isinstance(recent_event_count, int) or isinstance(recent_event_count, bool):
            raise TypeError("packet recent_event_count must be an integer")
        if not _is_string_sequence(categories) or not _is_string_sequence(exams):
            raise TypeError("packet category fields must contain strings")
        if not isinstance(trends, Sequence) or isinstance(trends, (str, bytes)):
            raise TypeError("packet recent_measurement_trends must be a sequence")
        projected_trends: list[tuple[str, str]] = []
        for row in trends:
            if not isinstance(row, Mapping):
                raise TypeError("packet trend rows must be objects")
            metric, trend = row.get("metric"), row.get("trend")
            if not isinstance(metric, str) or not isinstance(trend, str):
                raise TypeError("packet trend rows must contain string metric and trend")
            projected_trends.append((metric, trend))
        return cls(
            age_bucket=age_bucket,
            condition_categories=tuple(categories),
            recent_measurement_trends=tuple(projected_trends),
            available_exam_categories=tuple(exams),
            history_span_days=history_span_days,
            recent_event_count=recent_event_count,
        )


def _is_string_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and all(
        isinstance(item, str) for item in value
    )


def render_reader_v2_prompt(
    *,
    question: str,
    state_view: Mapping[str, Any] | ReaderStateView | None,
    passages: Sequence[Mapping[str, Any]],
) -> str:
    view = state_view.to_dict() if isinstance(state_view, ReaderStateView) else state_view
    state_text = (
        json.dumps(view, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if view is not None
        else "None supplied."
    )
    blocks: list[str] = []
    for passage in passages:
        chunk_id, text = passage.get("chunk_id"), passage.get("text")
        if not isinstance(chunk_id, str) or not isinstance(text, str):
            raise TypeError("each reference passage must have string chunk_id and text")
        blocks.append(f"[{chunk_id}]\n{text}")
    passage_text = "\n\n".join(blocks) if blocks else "None supplied."
    return READER_V2_PROMPT.format(
        QUESTION=question,
        STATE_VIEW=state_text,
        PASSAGES=passage_text,
    )


def parse_reader_v2_output(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse and validate once; never repairs aliases, missing fields, or malformed JSON."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(value, dict) or set(value) != {"state_facts", "guidance_facts"}:
        return None, False
    state_facts, guidance_facts = value["state_facts"], value["guidance_facts"]
    if not isinstance(state_facts, list) or len(state_facts) > MAX_STATE_FACTS:
        return None, False
    if not isinstance(guidance_facts, list) or len(guidance_facts) > MAX_GUIDANCE_FACTS:
        return None, False
    seen_state_fields: set[str] = set()
    for fact in state_facts:
        if not isinstance(fact, dict) or set(fact) != {"field", "value"}:
            return None, False
        field, trend = fact["field"], fact["value"]
        if field not in STATE_FIELDS or trend not in TREND_VALUES or field in seen_state_fields:
            return None, False
        seen_state_fields.add(field)
    for fact in guidance_facts:
        if not isinstance(fact, dict) or set(fact) != {"statement", "citations"}:
            return None, False
        statement, citations = fact["statement"], fact["citations"]
        if (
            not isinstance(statement, str)
            or not statement.strip()
            or len(statement) > MAX_STATEMENT_LENGTH
            or not isinstance(citations, list)
            or len(citations) > MAX_CITATIONS_PER_FACT
            or any(not isinstance(item, str) or not item for item in citations)
        ):
            return None, False
    return value, True


__all__ = [
    "MAX_CITATIONS_PER_FACT",
    "MAX_GUIDANCE_FACTS",
    "MAX_STATEMENT_LENGTH",
    "MAX_STATE_FACTS",
    "READER_STATE_PROJECTION_SHA256",
    "READER_STATE_PROJECTION_SPEC",
    "READER_V2_PROMPT",
    "READER_V2_SCHEMA",
    "ReaderStateView",
    "parse_reader_v2_output",
    "render_reader_v2_prompt",
]
