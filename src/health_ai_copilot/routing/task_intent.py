"""Typed evidence-task intent questions and response validation for Jev."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .jev import JevResult


class PrimaryTaskIntent(StrEnum):
    DIRECT_LOOKUP = "direct_lookup"
    GENERAL_EXPLANATION = "general_explanation"
    MULTI_TOPIC_SYNTHESIS = "multi_topic_synthesis"
    CROSS_AUTHORITY_COMPARISON = "cross_authority_comparison"
    CURRENT_GUIDELINE_LOOKUP = "current_guideline_lookup"
    CONFLICTING_GUIDANCE_REVIEW = "conflicting_guidance_review"
    SERIAL_FOLLOW_UP = "serial_follow_up"
    OTHER = "other"


TASK_INTENT_FACETS = (
    "needs_multi_topic_coverage",
    "needs_independent_sources",
    "requires_cross_source_comparison",
    "needs_current_guidance",
    "requires_conflict_review",
    "requires_serial_dependency",
)


@dataclass(frozen=True)
class TaskIntentAssessment:
    primary_intent: PrimaryTaskIntent
    primary_probabilities: Mapping[str, float]
    primary_confidence: float
    facet_probabilities: Mapping[str, float]

    def metadata(self) -> dict[str, Any]:
        return {
            "primary_task_intent": self.primary_intent.value,
            "primary_task_intent_probabilities": dict(self.primary_probabilities),
            "primary_task_intent_confidence": self.primary_confidence,
            "intent_facet_probabilities": dict(self.facet_probabilities),
        }


def task_intent_questions() -> dict[str, dict[str, Any]]:
    labels = {
        PrimaryTaskIntent.DIRECT_LOOKUP: "A direct lookup of one fact or recommendation.",
        PrimaryTaskIntent.GENERAL_EXPLANATION: "An explanation of one concept or topic.",
        PrimaryTaskIntent.MULTI_TOPIC_SYNTHESIS: (
            "A synthesis across multiple topics or subquestions."
        ),
        PrimaryTaskIntent.CROSS_AUTHORITY_COMPARISON: (
            "A comparison across institutions or jurisdictions."
        ),
        PrimaryTaskIntent.CURRENT_GUIDELINE_LOOKUP: (
            "A request for current or recently updated guidance."
        ),
        PrimaryTaskIntent.CONFLICTING_GUIDANCE_REVIEW: (
            "A request to explain or review differences between sources."
        ),
        PrimaryTaskIntent.SERIAL_FOLLOW_UP: (
            "A sequence where later subtasks depend on earlier results."
        ),
        PrimaryTaskIntent.OTHER: (
            "None of the listed task types or the task cannot be determined."
        ),
    }
    questions: dict[str, dict[str, Any]] = {
        "primary_task_intent": {
            "type": "choice",
            "instructions": (
                "Classify the evidence task requested by the question. This is workflow "
                "classification only; do not answer the medical question or judge medical "
                "correctness."
            ),
            "criteria": {intent.value: description for intent, description in labels.items()},
        }
    }
    facets = {
        "needs_multi_topic_coverage": (
            "Does the question require coverage of multiple distinct topics or subquestions?",
            "Several separate topics or subquestions need evidence coverage.",
            "One topic can answer the request.",
        ),
        "needs_independent_sources": (
            "Does the question require evidence from multiple independent sources?",
            "Independent source evidence is needed to answer the requested scope.",
            "One source or one source family is likely enough.",
        ),
        "requires_cross_source_comparison": (
            "Does the question ask to compare guidance across sources, institutions, "
            "or jurisdictions?",
            "The requested answer must compare multiple sources, institutions, or jurisdictions.",
            "The question does not request a cross-source comparison.",
        ),
        "needs_current_guidance": (
            "Does the user explicitly need current, latest, or recently updated guidance?",
            "Freshness or recent updates are part of the question's requirement.",
            "The question can be answered without checking recent updates.",
        ),
        "requires_conflict_review": (
            "Does the user ask to explain or review apparent differences between sources?",
            (
                "The workflow should compare or explain possible source differences; this "
                "does not assert that a real conflict exists."
            ),
            "The user does not ask to review differences between sources.",
        ),
        "requires_serial_dependency": (
            "Do later parts of the requested work depend on results from earlier parts?",
            "At least one later subtask depends on an earlier result.",
            "The requested subtasks can be handled independently or there are no subtasks.",
        ),
    }
    for name, (instruction, true_criteria, false_criteria) in facets.items():
        questions[name] = {
            "type": "noul",
            "instructions": instruction,
            "criteria": {"true": true_criteria, "false": false_criteria},
        }
    return questions


def parse_task_intent(result: JevResult) -> TaskIntentAssessment:
    choice = result.answers.get("primary_task_intent")
    if not isinstance(choice, Mapping) or choice.get("type") != "choice":
        raise ValueError("Jev returned an invalid primary_task_intent answer")
    try:
        primary = PrimaryTaskIntent(str(choice.get("choice", "")))
    except ValueError as exc:
        raise ValueError("Jev returned an unknown primary task intent") from exc
    raw_probabilities = choice.get("probabilities")
    if not isinstance(raw_probabilities, Mapping):
        raise ValueError("Jev did not return primary task intent probabilities")
    primary_probabilities = {
        intent.value: _probability(raw_probabilities.get(intent.value))
        for intent in PrimaryTaskIntent
    }
    if sum(primary_probabilities.values()) <= 0:
        raise ValueError("Jev returned empty primary task intent probabilities")
    confidence = _probability(choice.get("confidence"))
    facets: dict[str, float] = {}
    for name in TASK_INTENT_FACETS:
        answer = result.answers.get(name)
        if not isinstance(answer, Mapping) or answer.get("type") != "noul":
            raise ValueError(f"Jev returned an invalid task intent facet: {name}")
        facets[name] = _probability(answer.get("noul"))
    return TaskIntentAssessment(
        primary_intent=primary,
        primary_probabilities=primary_probabilities,
        primary_confidence=confidence,
        facet_probabilities=facets,
    )


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Jev probabilities must be numbers between 0 and 1")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("Jev probabilities must be between 0 and 1")
    return float(value)


__all__ = [
    "PrimaryTaskIntent",
    "TASK_INTENT_FACETS",
    "TaskIntentAssessment",
    "parse_task_intent",
    "task_intent_questions",
]
