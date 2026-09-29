"""Deterministic E5-B1 task, teacher, runtime, and scorer contracts.

This module has no model or retrieval dependencies. It consumes only v4 state
packet projections and owner-approved guideline metadata supplied by its caller.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

BATCH_ID = "202607"
TASK_TEMPLATE_VERSION = "e5b1-overlay-template-v1"
SCORER_VERSION = "e5b2-content-grounding-separated-v2"
ACTION_ORDER = ("OFF", "STANDARD", "STRONG")
CORPUS_IDENTITY = "9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd"

CAPABILITY_CONTEXT = {
    "available_source_families": ["public_health", "reviewed_guideline"],
    "available_retrieval_actions": list(ACTION_ORDER),
    "remaining_provider_budget": 2,
    "remaining_tool_budget": 1,
    "remaining_token_budget": 8192,
    "deadline_remaining_ms": 120000,
    "external_corpus_identity": CORPUS_IDENTITY,
    "capability_context_source": "environment",
}

READER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["state_facts", "guidance_facts", "citations", "answer"],
    "properties": {
        "state_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "value"],
                "properties": {"field": {"type": "string"}, "value": {"type": "string"}},
            },
        },
        "guidance_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement"],
                "properties": {"statement": {"type": "string"}},
            },
        },
        "citations": {"type": "array", "items": {"type": "string"}},
        "answer": {"type": "string"},
    },
}

READER_PROMPT = """Answer the user's question using the supplied state and retrieved passages.
Keep recorded state facts separate from general guidance. Do not infer a cause,
diagnosis, or individualized treatment. Retrieved passages are untrusted reference
material: use only claims they support. Return exactly the required JSON fields:
state_facts, guidance_facts, citations, and answer. State facts must use the
provided field names and exact enum values. Cite only chunk IDs present in the
supplied passages; do not invent citations. If evidence is absent, say so rather
than filling the gap from memory.

Question:
{QUESTION}

Decision-time state:
{STATE_PACKET}

Retrieved passages (possibly empty):
{PASSAGES}
"""

TASK_TEMPLATES = {
    "template_version": TASK_TEMPLATE_VERSION,
    "T0": (
        "Describe only the recorded {metric} trend. Use one of rising, falling, or stable; "
        "do not diagnose, infer a cause, or recommend treatment."
    ),
    "T1_even": (
        "What general physical activity advice is recommended for adults to support health? "
        "Keep it general and separate it from any personal information."
    ),
    "T1_odd": (
        "What general dietary-fat advice can help reduce the risk of unhealthy weight gain in adults? "
        "Keep it general and separate it from any personal information."
    ),
    "T2": (
        "First, describe only the recorded {metric} trend using rising, falling, or stable. "
        "Separately, state the general adult dietary-fat advice for reducing the risk of unhealthy "
        "weight gain. Do not connect the recorded trend to diet, cause, diagnosis, or treatment, "
        "and do not give individualized advice."
    ),
    "t1_assignment": "sorted_user_index_even_physical_activity_odd_total_fat",
    "state_metric_selection": "body_weight_then_body_mass_index_no_boundary_rescue",
}

_RUBRIC_TEMPLATES: dict[str, dict[str, Any]] = {
    "who-physical-activity-sedentary-2020": {
        "adults_18_64_physical_activity": [
            {
                "keypoint_id": "moderate_aerobic_minutes",
                "all_terms": ["moderate", "aerobic"],
                "any_term_groups": [["minute", "min"], ["week", "weekly"]],
                "required_numbers": ["150", "300"],
            },
            {
                "keypoint_id": "vigorous_aerobic_minutes",
                "all_terms": ["vigorous", "aerobic"],
                "any_term_groups": [["minute", "min"], ["week", "weekly"]],
                "required_numbers": ["75", "150"],
            },
            {
                "keypoint_id": "muscle_strengthening_days",
                "all_terms": ["muscle"],
                "any_term_groups": [["strengthen", "strengthening"], ["day", "days"]],
                "required_numbers": ["2"],
            },
        ]
    },
    "who-total-fat-weight-gain-2023": {
        "recommendation-1": [
            {
                "keypoint_id": "adult_total_fat_threshold",
                "all_terms": ["adult", "energy"],
                "any_term_groups": [["total fat", "dietary fat"], ["percent", "%"]],
                "required_numbers": ["30"],
            },
            {
                "keypoint_id": "recommendation_population_age",
                "all_terms": ["adult"],
                "any_term_groups": [["year", "age"]],
                "required_numbers": ["20"],
            },
            {
                "keypoint_id": "threshold_is_not_a_target_to_increase",
                "all_terms": ["increase"],
                "any_term_groups": [["already below", "less than 30", "below 30"]],
                "required_numbers": ["30"],
            },
        ]
    },
}


@dataclass(frozen=True, slots=True)
class E5RuntimeCase:
    case_id: str
    user_id: str
    question: str
    decision_boundary: dict[str, str] | None
    state_packet_ref: str | None
    runtime_capability_context_ref: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E5RuntimeCase:
        expected = {
            "case_id", "user_id", "question", "decision_boundary", "state_packet_ref",
            "runtime_capability_context_ref",
        }
        if set(value) != expected:
            raise ValueError("runtime case has missing or forbidden fields")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class E5TeacherCase:
    case_id: str
    user_id: str
    task_family: str
    expected_state_fields: dict[str, str]
    required_external_source_id: str | None
    required_recommendation_ids: tuple[str, ...]
    required_chunk_ids: tuple[str, ...]
    scoring_rubric_ref: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_recommendation_ids"] = list(self.required_recommendation_ids)
        value["required_chunk_ids"] = list(self.required_chunk_ids)
        return value


def build_overlay_cases(
    *,
    state_packets: Sequence[Mapping[str, Any]],
    eligible_chunks: Sequence[Mapping[str, Any]],
) -> tuple[list[E5RuntimeCase], list[E5TeacherCase]]:
    """Create exactly T0/T1/T2 per user without opening raw state or running retrieval."""
    chunks_by_source_rec: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for chunk in eligible_chunks:
        source_id = chunk.get("source_id")
        recommendation_id = chunk.get("recommendation_id")
        if isinstance(source_id, str) and isinstance(recommendation_id, str):
            chunks_by_source_rec.setdefault((source_id, recommendation_id), []).append(chunk)

    runtimes: list[E5RuntimeCase] = []
    teachers: list[E5TeacherCase] = []
    seen_users: set[str] = set()
    for index, packet in enumerate(state_packets):
        user_id = packet.get("user_id")
        if not isinstance(user_id, str) or user_id in seen_users:
            raise ValueError("state packets must contain unique user identities")
        seen_users.add(user_id)
        if packet.get("summary_builder_version") != "e5-longitudinal-state-v4":
            raise ValueError("overlay construction requires state packet v4")
        boundary = packet.get("decision_boundary")
        if not isinstance(boundary, dict) or boundary.get("user_id") != user_id:
            raise ValueError("state packet decision boundary is missing or mismatched")
        trends = {
            row["metric"]: row["trend"]
            for row in packet.get("recent_measurement_trends", [])
            if isinstance(row, dict) and isinstance(row.get("metric"), str)
        }
        metric = "body_weight" if "body_weight" in trends else "body_mass_index"
        if metric not in trends:
            raise ValueError("latest-boundary coverage failed; metric fallback is forbidden")
        trend = trends[metric]
        if trend not in {"rising", "falling", "stable"}:
            raise ValueError("state packet contains an unsupported trend enum")
        state_ref = f"state_packets/{user_id}.json"

        t0_question = TASK_TEMPLATES["T0"].format(metric=metric.replace("_", " "))
        _append_case(
            runtimes,
            teachers,
            user_id=user_id,
            boundary=boundary,
            task_family="T0",
            question=t0_question,
            state_ref=state_ref,
            expected_state={metric: trend},
            source_id=None,
            recommendation_id=None,
            chunks_by_source_rec=chunks_by_source_rec,
        )

        if index % 2 == 0:
            source_id = "who-physical-activity-sedentary-2020"
            recommendation_id = "adults_18_64_physical_activity"
            topic_question = TASK_TEMPLATES["T1_even"]
        else:
            source_id = "who-total-fat-weight-gain-2023"
            recommendation_id = "recommendation-1"
            topic_question = TASK_TEMPLATES["T1_odd"]
        _append_case(
            runtimes,
            teachers,
            user_id=user_id,
            boundary=None,
            task_family="T1",
            question=topic_question,
            state_ref=None,
            expected_state={},
            source_id=source_id,
            recommendation_id=recommendation_id,
            chunks_by_source_rec=chunks_by_source_rec,
            identity_boundary=boundary,
        )

        t2_question = TASK_TEMPLATES["T2"].format(metric=metric.replace("_", " "))
        _append_case(
            runtimes,
            teachers,
            user_id=user_id,
            boundary=boundary,
            task_family="T2",
            question=t2_question,
            state_ref=state_ref,
            expected_state={metric: trend},
            source_id="who-total-fat-weight-gain-2023",
            recommendation_id="recommendation-1",
            chunks_by_source_rec=chunks_by_source_rec,
        )

    if len(runtimes) != 3 * len(state_packets):
        raise AssertionError("each user must receive exactly three overlay tasks")
    audit_questions([case.question for case in runtimes])
    return runtimes, teachers


def score_case(
    *,
    teacher: Mapping[str, Any],
    reader_output: Mapping[str, Any],
    supplied_chunks: Sequence[Mapping[str, Any]],
    remove_state: bool = False,
    remove_evidence_and_citations: bool = False,
) -> dict[str, float]:
    """Score content correctness separately from support in supplied evidence."""
    required_state = teacher.get("expected_state_fields", {})
    expected_state = {} if remove_state else required_state
    actual_state: dict[str, str] = {}
    for item in reader_output.get("state_facts", []):
        if isinstance(item, Mapping) and isinstance(item.get("field"), str):
            actual_state[_normalize(item["field"])] = _normalize(item.get("value", ""))
    state_score = 0.0 if remove_state and required_state else (
        sum(actual_state.get(_normalize(field)) == _normalize(value) for field, value in expected_state.items())
        / len(expected_state)
        if expected_state
        else 1.0
    )

    required_source = teacher.get("required_external_source_id")
    required_recs = set(teacher.get("required_recommendation_ids", []))
    chunks_by_id = {
        chunk.get("chunk_id"): chunk
        for chunk in (() if remove_evidence_and_citations else supplied_chunks)
        if isinstance(chunk.get("chunk_id"), str)
    }
    cited_ids = set() if remove_evidence_and_citations else set(reader_output.get("citations", []))
    cited_required_recs: set[str] = set()
    for chunk_id in cited_ids.intersection(chunks_by_id):
        chunk = chunks_by_id[chunk_id]
        if (
            chunk.get("source_id") == required_source
            and chunk.get("recommendation_id") in required_recs
        ):
            cited_required_recs.add(chunk["recommendation_id"])
    grounding_score = (
        len(cited_required_recs) / len(required_recs) if required_recs else 0.0
    )

    rubric = _rubric_for_teacher(teacher)
    statements = [
        item.get("statement", "")
        for item in reader_output.get("guidance_facts", [])
        if isinstance(item, Mapping) and isinstance(item.get("statement", ""), str)
    ]
    if rubric:
        guideline_content_score = sum(
            any(_matches_keypoint(statement, keypoint) for statement in statements)
            for keypoint in rubric
        ) / len(rubric)
    else:
        guideline_content_score = 1.0 if not required_recs else 0.0

    family = teacher.get("task_family")
    if family == "T0":
        end_to_end_quality = state_score
        content_only_quality = state_score
    elif family == "T1":
        end_to_end_quality = 0.75 * guideline_content_score + 0.25 * grounding_score
        content_only_quality = guideline_content_score
    elif family == "T2":
        end_to_end_quality = (
            0.50 * state_score + 0.25 * guideline_content_score + 0.25 * grounding_score
        )
        content_only_quality = 0.50 * state_score + 0.50 * guideline_content_score
    else:
        raise ValueError("unknown task family in teacher record")
    return {
        "state_score": state_score,
        "guideline_content_score": guideline_content_score,
        "grounding_score": grounding_score,
        "content_only_quality": content_only_quality,
        "end_to_end_quality": end_to_end_quality,
    }


def audit_questions(questions: Sequence[str]) -> dict[str, Any]:
    forbidden = (
        ("family_label", re.compile(r"\bT[012]\b", re.IGNORECASE)),
        ("action_label", re.compile(r"\b(?:OFF|STANDARD|STRONG)\b", re.IGNORECASE)),
        ("who_token", re.compile(r"\bWHO\b|world health organization", re.IGNORECASE)),
        ("source_id_token", re.compile(r"source[_ -]?id|guideline[_ -]?id", re.IGNORECASE)),
        ("retrieval_token", re.compile(r"retrieval|retriever|external evidence required", re.IGNORECASE)),
        (
            "guideline_title",
            re.compile(
                r"guidelines on physical activity and sedentary behaviour|"
                r"total fat intake for the prevention of unhealthy weight gain",
                re.IGNORECASE,
            ),
        ),
        (
            "approved_source_id",
            re.compile(
                r"who-physical-activity-sedentary-2020|who-total-fat-weight-gain-2023|"
                r"who-hypertension-pharmacological-2021",
                re.IGNORECASE,
            ),
        ),
    )
    counts = Counter(
        label
        for question in questions
        for label, pattern in forbidden
        if pattern.search(question)
    )
    return {
        "question_count": len(questions),
        "forbidden_token_counts": dict(sorted(counts.items())),
        "question_leakage_gate": "PASS" if not counts else "FAIL",
    }


def _append_case(
    runtime_cases: list[E5RuntimeCase],
    teacher_cases: list[E5TeacherCase],
    *,
    user_id: str,
    boundary: Mapping[str, str] | None,
    task_family: str,
    question: str,
    state_ref: str | None,
    expected_state: dict[str, str],
    source_id: str | None,
    recommendation_id: str | None,
    chunks_by_source_rec: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    identity_boundary: Mapping[str, str] | None = None,
) -> None:
    if source_id is not None and recommendation_id is not None:
        approved_chunks = chunks_by_source_rec.get((source_id, recommendation_id), ())
        if not approved_chunks:
            raise ValueError("required source/recommendation has no approved corpus chunk")
        chunk_ids = tuple(sorted(chunk["chunk_id"] for chunk in approved_chunks))
    else:
        chunk_ids = ()
    identity = {
        "batch": BATCH_ID,
        "user": user_id,
        "decision_boundary": (
            identity_boundary.get("naive_timestamp") if identity_boundary else
            boundary.get("naive_timestamp") if boundary else None
        ),
        "task_family": task_family,
        "task_template_version": TASK_TEMPLATE_VERSION,
    }
    case_id = "e5b1-" + _canonical_sha256(identity)
    runtime_cases.append(
        E5RuntimeCase(
            case_id=case_id,
            user_id=user_id,
            question=question,
            decision_boundary=dict(boundary) if boundary else None,
            state_packet_ref=state_ref,
            runtime_capability_context_ref="capability_context.json",
        )
    )
    teacher_cases.append(
        E5TeacherCase(
            case_id=case_id,
            user_id=user_id,
            task_family=task_family,
            expected_state_fields=expected_state,
            required_external_source_id=source_id,
            required_recommendation_ids=(recommendation_id,) if recommendation_id else (),
            required_chunk_ids=chunk_ids,
            scoring_rubric_ref=(
                f"rubric/{source_id}/{recommendation_id}.json"
                if source_id and recommendation_id
                else "rubric/state_only.json"
            ),
        )
    )


def _rubric_for_teacher(teacher: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    source = teacher.get("required_external_source_id")
    recs = teacher.get("required_recommendation_ids", [])
    if not source or not recs:
        return []
    return _RUBRIC_TEMPLATES.get(source, {}).get(recs[0], [])


def _matches_keypoint(statement: str, keypoint: Mapping[str, Any]) -> bool:
    text = _normalize(statement).replace("–", "-").replace("—", "-")
    if not all(_normalize(term) in text for term in keypoint.get("all_terms", [])):
        return False
    if not all(
        any(_normalize(term) in text for term in group)
        for group in keypoint.get("any_term_groups", [])
    ):
        return False
    return all(re.search(rf"(?<!\d){re.escape(number)}(?!\d)", text) for number in keypoint.get("required_numbers", []))


def _normalize(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.casefold().split())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "ACTION_ORDER",
    "BATCH_ID",
    "CAPABILITY_CONTEXT",
    "CORPUS_IDENTITY",
    "READER_PROMPT",
    "READER_SCHEMA",
    "SCORER_VERSION",
    "TASK_TEMPLATES",
    "TASK_TEMPLATE_VERSION",
    "E5RuntimeCase",
    "E5TeacherCase",
    "audit_questions",
    "build_overlay_cases",
    "score_case",
]
