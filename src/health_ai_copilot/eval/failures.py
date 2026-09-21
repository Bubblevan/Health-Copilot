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
        "mcp_protocol_version": (FailureStage.MCP, "protocol_version_mismatch"),
        "mcp_discovery": (FailureStage.MCP, "discovery_failed"),
        "mcp_catalog": (FailureStage.MCP, "catalog_error"),
        "mcp_invalid_arguments": (FailureStage.MCP, "schema_validation_failed"),
        "mcp_transport": (FailureStage.MCP, "transport_error"),
        "mcp_call_error": (FailureStage.MCP, "tool_call_error"),
        "permission_denied": (FailureStage.PERMISSION, "permission_denied"),
        "approval_denied": (FailureStage.PERMISSION, "approval_denied"),
        "unknown_capability": (FailureStage.PERMISSION, "unknown_capability"),
        "sandbox_unavailable": (FailureStage.SANDBOX, "sandbox_unavailable"),
        "filesystem_denied": (FailureStage.SANDBOX, "filesystem_denied"),
        "network_denied": (FailureStage.SANDBOX, "network_denied"),
        "authorization_failed": (FailureStage.AUTHORIZATION, "authorization_failed"),
        "mcp_sandbox": (FailureStage.SANDBOX, "sandbox_unavailable"),
        "protocol_version_mismatch": (FailureStage.MCP, "protocol_version_mismatch"),
        "catalog_error": (FailureStage.MCP, "catalog_error"),
        "permission_denied_unexpected": (FailureStage.PERMISSION, "permission_denied_unexpected"),
        "approval_binding_failed": (FailureStage.PERMISSION, "approval_binding_failed"),
        "filesystem_containment_failed": (FailureStage.SANDBOX, "filesystem_containment_failed"),
        "network_containment_failed": (FailureStage.SANDBOX, "network_containment_failed"),
        "security_control_failed": (FailureStage.EVAL_INFRA, "security_control_failed"),
    }
    _TEAM_LEAD_MAP: ClassVar = {
        "provider": "lead_provider_failure",
        "empty_response": "lead_empty_response",
        "json_decode": "lead_json_decode",
        "contract_validation": "lead_contract_validation",
        "internal": "lead_internal",
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

    def from_orchestration(self, record: CaseRunRecord) -> list[FailureRecord]:
        """Map fail-closed team failures even when the case is COMPLETE."""

        if record.status != CaseRunStatus.COMPLETE:
            return []
        observed = record.observed
        summary = {
            key: observed.get(key)
            for key in (
                "team_stop_reason",
                "lead_failure_kind",
                "provider_failure_kind",
                "lead_contract_version",
                "response_content_sha256",
                "response_length",
            )
            if observed.get(key) is not None
        }
        failures: list[FailureRecord] = []

        lead_kind = observed.get("lead_failure_kind")
        if lead_kind in self._TEAM_LEAD_MAP:
            failures.append(
                FailureRecord(
                    suite_id=record.suite_id,
                    case_id=record.case_id,
                    trial=record.trial,
                    stage=FailureStage.ORCHESTRATION,
                    failure_code=self._TEAM_LEAD_MAP[lead_kind],
                    source="orchestrator:team_lead",
                    observed=summary,
                )
            )

        stop_reason = observed.get("team_stop_reason")
        stop_map = {
            "invalid_delegation": "invalid_delegation",
            "lead_second_delegation": "lead_second_delegation",
            "budget_exhausted": "team_budget_exhausted",
            "worker_error": "worker_failure",
            "final_verification_failed": "final_verification_failed",
        }
        if stop_reason in stop_map:
            failures.append(
                FailureRecord(
                    suite_id=record.suite_id,
                    case_id=record.case_id,
                    trial=record.trial,
                    stage=FailureStage.ORCHESTRATION,
                    failure_code=stop_map[stop_reason],
                    source="orchestrator:team_stop",
                    observed=summary,
                )
            )

        seen_worker_codes: set[str] = set()
        for role_record in observed.get("team_role_records", ()):
            error_code = role_record.get("error_code")
            if not error_code or error_code in seen_worker_codes:
                continue
            seen_worker_codes.add(error_code)
            if error_code == "team_budget_exhausted":
                failure_code = "team_budget_exhausted"
            elif error_code == "worker_citation_provenance_violation":
                failure_code = error_code
            else:
                failure_code = "worker_failure"
            failures.append(
                FailureRecord(
                    suite_id=record.suite_id,
                    case_id=record.case_id,
                    trial=record.trial,
                    stage=FailureStage.ORCHESTRATION,
                    failure_code=failure_code,
                    source="orchestrator:worker",
                    observed={**summary, "worker_error_code": error_code},
                )
            )
        return failures

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
            failures.extend(self.from_orchestration(row))
        for result in grader_results:
            record = records_by_key.get((result.case_id, result.trial))
            case = case_by_id.get(result.case_id)
            if record is not None and case is not None:
                failures.extend(self.from_grader(case, record, result))
        return failures
