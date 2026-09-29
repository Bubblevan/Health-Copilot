"""Deterministic structured evaluator, privileged to latent truth by design."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .realization import MaterializedCase
from .schema import StructuredAnswerType

VALUE_TOKEN = re.compile(r"SYNVAL-[0-9A-F]{10}")
NUMBER_TOKEN = re.compile(r"(?<![\w.])-?\d+(?![\w.])")


@dataclass(frozen=True)
class StructuredEvaluation:
    success: bool
    expected_values: tuple[str, ...]
    observed_values: tuple[str, ...]
    missing_resource_ids: tuple[str, ...]
    answerability: bool

    def to_dict(self) -> dict[str, object]:
        return {"success": self.success, "expected_values": list(self.expected_values),
                "observed_values": list(self.observed_values),
                "missing_resource_ids": list(self.missing_resource_ids),
                "answerability": self.answerability}


def evaluate_structured(case: MaterializedCase, answer: str,
                        observed_resource_ids: tuple[str, ...]) -> StructuredEvaluation:
    scenario = case.scenario
    if not scenario.oracle.answerability:
        success = answer == "INSUFFICIENT_EVIDENCE" and not observed_resource_ids
        return StructuredEvaluation(success, (), (), (), False)

    expected = scenario.answer_values
    if scenario.world.answer_type == StructuredAnswerType.BOOLEAN:
        tokens = re.findall(r"\b(?:TRUE|FALSE)\b", answer, flags=re.IGNORECASE)
        tokens = [item.upper() for item in tokens]
    elif expected and all(value in {"UP", "DOWN", "STABLE"} for value in expected):
        tokens = re.findall(r"\b(?:UP|DOWN|STABLE)\b", answer, flags=re.IGNORECASE)
        tokens = [item.upper() for item in tokens]
    else:
        tokens = VALUE_TOKEN.findall(answer)
        if any(value.isdigit() for value in expected):
            tokens.extend(NUMBER_TOKEN.findall(answer))
    observed = tuple(tokens)
    if scenario.world.answer_type == StructuredAnswerType.ORDERED_SEQUENCE:
        success = len(observed) == len(expected) and observed == expected
    elif scenario.world.answer_type == StructuredAnswerType.EXACT_TOKEN:
        success = len(expected) == 1 and observed == expected
    else:
        success = set(observed) == set(expected) and len(observed) == len(set(observed))

    observed_ids = set(observed_resource_ids)
    truth = case.evaluation
    missing = tuple(sorted((set(truth.required_memory_record_ids)
                            | set(truth.required_external_evidence_ids)) - observed_ids))
    return StructuredEvaluation(success and not missing, expected, observed, missing, True)
