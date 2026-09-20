"""Evidence-based M7 failure taxonomy mapping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from .schema import (
    CaseRunRecord,
    CaseRunStatus,
    EvalCase,
    FailureRecord,
    FailureStage,
    GraderResult,
    GraderStatus,
)


class FailureMapper:
    """Map observed deterministic grader evidence to one or more failure records."""

    _MAP: ClassVar = {
        "unexpected_route": (FailureStage.SAFETY, "unexpected_route"),
        "safety_short_circuit_failed": (FailureStage.SAFETY, "unexpected_abstain"),
        "retrieval_miss": (FailureStage.RETRIEVAL, "retrieval_miss"),
        "unnecessary_recovery": (FailureStage.ACTION_SELECTION, "unnecessary_recovery"),
        "unexpected_tool_execution": (FailureStage.TOOL_EXECUTION, "unexpected_tool_execution"),
        "policy_false_veto": (FailureStage.POLICY, "policy_false_veto"),
        "policy_false_allow": (FailureStage.POLICY, "policy_false_allow"),
        "policy_error": (FailureStage.POLICY, "policy_error"),
        "capability_topic_mismatch": (FailureStage.POLICY, "capability_topic_mismatch"),
        "citation_integrity_failed": (FailureStage.GROUNDING, "citation_integrity_failed"),
        "claim_verdict_mismatch": (FailureStage.GROUNDING, "claim_support_failed"),
        "max_model_turns": (FailureStage.BUDGET, "max_model_turns"),
        "max_tool_calls": (FailureStage.BUDGET, "max_tool_calls"),
        "replay_mismatch": (FailureStage.REPLAY, "replay_mismatch"),
        "model_error": (FailureStage.PROVIDER, "model_error"),
        "provider_error": (FailureStage.PROVIDER, "provider_error"),
        "tool_error": (FailureStage.TOOL_EXECUTION, "tool_error"),
        "evidence_group_missing": (FailureStage.ORCHESTRATION, "evidence_group_missing"),
        "team_not_observed": (FailureStage.ORCHESTRATION, "team_not_observed"),
        "ood_route_mismatch": (FailureStage.ORCHESTRATION, "ood_route_mismatch"),
        "team_budget_exhausted": (FailureStage.ORCHESTRATION, "team_budget_exhausted"),
        "invalid_delegation": (FailureStage.ORCHESTRATION, "invalid_delegation"),
        "lead_second_delegation": (FailureStage.ORCHESTRATION, "lead_second_delegation"),
    }

    def from_grader(
        self, case: EvalCase, record: CaseRunRecord, result: GraderResult
    ) -> list[FailureRecord]:
        if result.status != GraderStatus.FAIL:
            return []
        failures: list[FailureRecord] = []
        for reason in result.reason_codes:
            stage, code = self._MAP.get(reason, (FailureStage.EVAL_INFRA, reason))
            failures.append(
                FailureRecord(
                    case_id=case.case_id,
                    suite_id=record.suite_id,
                    trial=record.trial,
                    stage=stage,
                    failure_code=code,
                    source=f"grader:{result.grader_id}",
                    expected=result.expected_summary,
                    observed=result.observed_summary,
                    reason_codes=result.reason_codes,
                )
            )
        return failures

    def from_record(self, record: CaseRunRecord) -> list[FailureRecord]:
        if record.status != CaseRunStatus.ERROR:
            return []
        return [
            FailureRecord(
                suite_id=record.suite_id,
                case_id=record.case_id,
                trial=record.trial,
                stage=FailureStage.EVAL_INFRA,
                failure_code=record.error_code or "execution_error",
                source="runner",
                observed={"message": record.error_message},
            )
        ]

    def collect(
        self,
        cases: Sequence[EvalCase],
        records: Sequence[CaseRunRecord],
        grader_results: Sequence[GraderResult],
    ) -> list[FailureRecord]:
        case_by_id = {case.case_id: case for case in cases}
        records_by_key = {(row.case_id, row.trial): row for row in records}
        failures: list[FailureRecord] = []
        for row in records:
            failures.extend(self.from_record(row))
        for result in grader_results:
            record = records_by_key.get((result.case_id, result.trial))
            case = case_by_id.get(result.case_id)
            if record is not None and case is not None:
                failures.extend(self.from_grader(case, record, result))
        return failures
