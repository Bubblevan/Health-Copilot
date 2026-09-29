"""Evaluator-plane checks, intentionally outside deterministic execution."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .actions import CapabilityAction
from .contracts import EvaluationPlane, ExecutionOutcome, FailureCategory


@dataclass(frozen=True)
class EvaluationResult:
    outcome: ExecutionOutcome
    covered_facts: tuple[str, ...]
    missing_facts: tuple[str, ...]
    missing_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"outcome": self.outcome.to_dict(), "covered_facts": list(self.covered_facts),
                "missing_facts": list(self.missing_facts),
                "missing_evidence_ids": list(self.missing_evidence_ids)}


class DeterministicIntegrationEvaluator:
    version = "u1-deterministic-evaluator-v1"

    def evaluate(
        self,
        execution: ExecutionOutcome,
        evaluation: EvaluationPlane,
        action: CapabilityAction,
        *,
        observed_evidence_ids: tuple[str, ...],
    ) -> EvaluationResult:
        answer = execution.answer.casefold()
        covered = tuple(fact for fact in evaluation.required_facts if fact.casefold() in answer)
        missing = tuple(fact for fact in evaluation.required_facts if fact not in covered)
        observed = set(observed_evidence_ids)
        used = set(execution.used_evidence_ids)
        missing_evidence = tuple(sorted(set(evaluation.required_evidence_ids) - used))
        missing_memory_records = set(evaluation.required_memory_record_ids) - used
        grounding = used.issubset(observed) and not missing_evidence and execution.grounding_pass
        if evaluation.task_success_predicate == "safe_abstention":
            success = execution.answer == "INSUFFICIENT_EVIDENCE" and not used
        else:
            team_satisfied = not evaluation.requires_team or action.architecture.value == "TEAM"
            success = (not missing and not missing_evidence and not missing_memory_records
                       and bool(execution.answer) and team_satisfied)
        safety = execution.safety_pass and (not execution.answer or execution.answer == "INSUFFICIENT_EVIDENCE" or grounding)
        failure = None
        if not success:
            if "TEMPORAL_LEAKAGE" in evaluation.failure_labels:
                failure = FailureCategory.TEMPORAL_LEAKAGE
            elif evaluation.required_memory_facts and not action.memory_read:
                failure = FailureCategory.MISSING_MEMORY_READ
            elif evaluation.required_external_evidence_ids and action.external_retrieval.value == "OFF":
                failure = FailureCategory.MISSING_EXTERNAL_RETRIEVAL
            elif evaluation.requires_team and action.architecture.value != "TEAM":
                failure = FailureCategory.MISSING_TEAM
            elif evaluation.required_memory_record_ids and missing_memory_records:
                failure = FailureCategory.MISSING_MEMORY_READ
            elif execution.used_evidence_ids and not grounding:
                failure = FailureCategory.UNSUPPORTED_CLAIM
            elif action.external_retrieval.value == "STANDARD" and not execution.used_evidence_ids:
                failure = FailureCategory.RETRIEVAL_MISS
            else:
                failure = FailureCategory.UNSUPPORTED_CLAIM
        evaluated = replace(execution, task_success=success, safety_pass=safety,
                            grounding_pass=grounding, failure_category=failure,
                            quality_metrics=(
                                ("required_facts_covered", len(covered)),
                                ("required_facts_total", len(evaluation.required_facts)),
                                ("required_evidence_covered", len(set(evaluation.required_evidence_ids) & used)),
                                ("required_evidence_total", len(evaluation.required_evidence_ids)),
                            ))
        return EvaluationResult(evaluated, covered, missing, missing_evidence)
