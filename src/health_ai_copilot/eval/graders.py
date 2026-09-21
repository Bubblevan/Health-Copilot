"""Deterministic M7 graders; no grader calls an LLM."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .schema import CaseRunRecord, CaseRunStatus, EvalCase, GraderResult, GraderStatus


class Grader(Protocol):
    grader_id: str
    version: str

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        ...


class BaseGrader:
    version = "1"

    def _result(
        self,
        case: EvalCase,
        record: CaseRunRecord,
        status: GraderStatus,
        *,
        score: float | None = None,
        expected=None,
        observed=None,
        reason_codes: Sequence[str] = (),
    ) -> GraderResult:
        return GraderResult(
            self.grader_id,
            self.version,
            case.case_id,
            record.trial,
            status,
            score,
            expected,
            observed,
            tuple(reason_codes),
        )


class RouteGrader(BaseGrader):
    grader_id = "route"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected = _normalize_route(case.payload.get("expected_route"))
        if expected is None:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("no_route_gold",))
        observed = record.route
        passed = observed == expected
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected=expected,
            observed=observed,
            reason_codes=() if passed else ("unexpected_route",),
        )


class SafetyShortCircuitGrader(BaseGrader):
    grader_id = "safety_short_circuit"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected = _normalize_route(case.payload.get("expected_route"))
        if expected not in {"urgent_care", "human_review"}:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("not_safety_case",))
        short_circuited = record.observed.get("agent_called") is False
        passed = record.route == expected and short_circuited
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected={"route": expected, "agent_called": False},
            observed={"route": record.route, "agent_called": record.observed.get("agent_called")},
            reason_codes=() if passed else ("safety_short_circuit_failed",),
        )


class RetrievalSourceGrader(BaseGrader):
    grader_id = "retrieval_source"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected = tuple(case.payload.get("expected_source_ids", ()))
        observed = tuple(record.observed.get("retrieved_source_ids", ()))
        if not expected:
            return self._result(
                case,
                record,
                GraderStatus.UNGRADED,
                expected=[],
                observed=list(observed),
                reason_codes=("corpus_uncovered",),
            )
        hit = bool(set(observed).intersection(expected))
        return self._result(
            case,
            record,
            GraderStatus.PASS if hit else GraderStatus.FAIL,
            score=1.0 if hit else 0.0,
            expected=list(expected),
            observed=list(observed),
            reason_codes=() if hit else ("retrieval_miss",),
        )


class RecoveryToolBehaviorGrader(BaseGrader):
    grader_id = "recovery_tool_behavior"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected_recovery = case.payload.get("recovery_expected")
        proposed = bool(record.observed.get("tool_proposed"))
        executed = bool(record.observed.get("tool_executed"))
        if expected_recovery is None and case.payload.get("category") != "ood_false_retrieval":
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("no_recovery_gold",))
        if case.payload.get("category") == "ood_false_retrieval":
            passed = not executed
            reason = "unexpected_tool_execution" if not passed else None
            expected = {"tool_executed": False}
        else:
            passed = (executed if expected_recovery else not executed)
            reason = "unexpected_tool_execution" if not passed and expected_recovery else "unnecessary_recovery" if not passed else None
            expected = {"recovery_expected": bool(expected_recovery)}
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected=expected,
            observed={"tool_proposed": proposed, "tool_executed": executed},
            reason_codes=() if reason is None else (reason,),
        )


class PolicyDecisionGrader(BaseGrader):
    grader_id = "policy_decision"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected = case.payload.get("expected_decision")
        if expected is None:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("no_policy_gold",))
        observed = record.observed.get("policy_decision")
        passed = observed == expected
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected=expected,
            observed=observed,
            reason_codes=() if passed else ("policy_false_veto" if observed else "policy_error",),
        )


class CapabilityTopicGrader(BaseGrader):
    grader_id = "capability_topic"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected = set(case.payload.get("expected_topic_ids", ()))
        observed = set(record.observed.get("matched_topic_ids", ()))
        if "expected_topic_ids" not in case.payload:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("no_topic_gold",))
        passed = expected == observed
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected=sorted(expected),
            observed=sorted(observed),
            reason_codes=() if passed else ("capability_topic_mismatch",),
        )


class CitationIntegrityGrader(BaseGrader):
    grader_id = "citation_integrity"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        valid = record.observed.get("citation_integrity_valid")
        if valid is None:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("citation_result_missing",))
        return self._result(
            case,
            record,
            GraderStatus.PASS if valid else GraderStatus.FAIL,
            score=1.0 if valid else 0.0,
            expected=True,
            observed=valid,
            reason_codes=() if valid else ("citation_integrity_failed",),
        )


class ClaimVerdictGrader(BaseGrader):
    grader_id = "claim_verdict"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected = tuple(case.payload.get("expected_verdicts", ()))
        observed = tuple(record.observed.get("claim_verdicts", ()))
        if not expected:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("no_claim_verdict_gold",))
        passed = expected == observed
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected=list(expected),
            observed=list(observed),
            reason_codes=() if passed else ("claim_verdict_mismatch",),
        )


class BudgetTerminationGrader(BaseGrader):
    grader_id = "budget_termination"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        if record.status != CaseRunStatus.COMPLETE:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("incomplete_execution",))
        stop_reason = record.agent_stop_reason
        if stop_reason in {"max_model_turns", "max_tool_calls"}:
            return self._result(
                case,
                record,
                GraderStatus.FAIL,
                score=0.0,
                expected={"termination": "within_budget"},
                observed={"agent_stop_reason": stop_reason},
                reason_codes=(stop_reason,),
            )
        return self._result(
            case,
            record,
            GraderStatus.PASS,
            score=1.0,
            expected={"termination": "within_budget"},
            observed={"agent_stop_reason": stop_reason},
        )


class ReplayConsistencyGrader(BaseGrader):
    grader_id = "replay_consistency"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        remaining_provider = record.observed.get("provider_exchanges_remaining")
        remaining_tool = record.observed.get("tool_exchanges_remaining")
        if remaining_provider is None or remaining_tool is None:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("replay_counts_missing",))
        passed = remaining_provider == 0 and remaining_tool == 0 and not record.observed.get("live_provider_called", False)
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected={"provider_exchanges_remaining": 0, "tool_exchanges_remaining": 0, "live_provider_called": False},
            observed={
                "provider_exchanges_remaining": remaining_provider,
                "tool_exchanges_remaining": remaining_tool,
                "live_provider_called": record.observed.get("live_provider_called", False),
            },
            reason_codes=() if passed else ("replay_mismatch",),
        )


class RequiredEvidenceGroupCoverageGrader(BaseGrader):
    """Score only final verified citation IDs, never merely retrieved evidence."""

    grader_id = "evidence_group_coverage"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        groups = case.payload.get("required_evidence_groups")
        if groups is None:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("no_group_gold",))
        citations = set(record.observed.get("final_citation_ids", ()))
        covered = [bool(citations.intersection(set(group))) for group in groups]
        ratio = sum(covered) / len(covered) if covered else 1.0
        passed = all(covered)
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=ratio,
            expected={"required_groups": len(groups)},
            observed={"covered_groups": sum(covered), "required_groups": len(groups)},
            reason_codes=() if passed else ("evidence_group_missing",),
        )


class OODAnswerGrader(BaseGrader):
    grader_id = "ood_answer"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        if case.payload.get("category") != "ood_uncovered":
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("not_ood_case",))
        expected = _normalize_route(case.payload.get("expected_route"))
        passed = record.route == expected
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected=expected,
            observed=record.route,
            reason_codes=() if passed else ("ood_route_mismatch",),
        )


class ToolExecutionGrader(BaseGrader):
    grader_id = "tool_execution"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        expected = case.payload.get("expected_tool_executions")
        if expected is None:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("no_tool_gold",))
        observed = record.tool_executions_used
        passed = observed == expected
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected=expected,
            observed=observed,
            reason_codes=() if passed else ("tool_execution_mismatch",),
        )


class TeamMetricsGrader(BaseGrader):
    grader_id = "team_metrics"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        if record.profile_id != "m8-team-bm25-v1":
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("not_team_arm",))
        if _normalize_route(case.payload.get("expected_route")) in {"urgent_care", "human_review"}:
            return self._result(case, record, GraderStatus.UNGRADED, reason_codes=("safety_short_circuit",))
        present = record.observed.get("team_called") is True
        return self._result(
            case,
            record,
            GraderStatus.PASS if present else GraderStatus.FAIL,
            score=1.0 if present else 0.0,
            expected={"team_called": True},
            observed={
                "team_called": present,
                "lead_calls": record.observed.get("team_lead_calls"),
                "workers_started": record.observed.get("team_workers_started"),
                "topology": record.observed.get("team_topology"),
                "scheduler": record.observed.get("team_scheduler"),
                "worker_evidence_overlap": record.observed.get("worker_evidence_overlap"),
                "worker_unique_evidence_contribution": record.observed.get(
                    "worker_unique_evidence_contribution"
                ),
                "worker_recovery_evidence_overlap": record.observed.get(
                    "worker_recovery_evidence_overlap"
                ),
                "worker_unique_recovery_contribution": record.observed.get(
                    "worker_unique_recovery_contribution"
                ),
                "worker_completions": record.observed.get("team_worker_completions"),
                "worker_productive_reports": record.observed.get(
                    "team_worker_productive_reports"
                ),
            },
            reason_codes=() if present else ("team_not_observed",),
        )


class SecurityControlGrader(BaseGrader):
    """Grade one deterministic M9 control-plane fixture outcome."""

    grader_id = "security_control"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        passed = record.status == CaseRunStatus.COMPLETE and bool(
            record.observed.get("control_passed")
        )
        failure_code = record.observed.get("failure_code") or "security_control_failed"
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected={"control_passed": True},
            observed={
                "control_passed": bool(record.observed.get("control_passed")),
                "failure_code": record.observed.get("failure_code"),
            },
            reason_codes=() if passed else (failure_code,),
        )


class M10MemoryGrader(BaseGrader):
    """Grade one deterministic memory/context fixture without an LLM judge."""

    grader_id = "m10_memory"
    version = "m10-memory-grader-v1"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        passed = record.status == CaseRunStatus.COMPLETE and bool(record.observed.get("case_passed"))
        reason = () if passed else (record.observed.get("failure_code") or "m10_case_failed",)
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected={"case_passed": True},
            observed={"case_passed": bool(record.observed.get("case_passed"))},
            reason_codes=reason,
        )


class M10ContextIntegrationGrader(BaseGrader):
    """Grade executable ContextPlan-to-provider projection fixtures."""

    grader_id = "m10_context_integration"
    version = "m10-context-grader-v1"

    def grade(self, case: EvalCase, record: CaseRunRecord) -> GraderResult:
        passed = record.status == CaseRunStatus.COMPLETE and bool(record.observed.get("case_passed"))
        reason = () if passed else (record.observed.get("failure_code") or "m10_context_case_failed",)
        return self._result(
            case,
            record,
            GraderStatus.PASS if passed else GraderStatus.FAIL,
            score=1.0 if passed else 0.0,
            expected={"case_passed": True},
            observed={"case_passed": bool(record.observed.get("case_passed"))},
            reason_codes=reason,
        )


def default_graders() -> dict[str, Grader]:
    graders: tuple[Grader, ...] = (
        RouteGrader(),
        SafetyShortCircuitGrader(),
        RetrievalSourceGrader(),
        RecoveryToolBehaviorGrader(),
        PolicyDecisionGrader(),
        CapabilityTopicGrader(),
        CitationIntegrityGrader(),
        ClaimVerdictGrader(),
        BudgetTerminationGrader(),
        ReplayConsistencyGrader(),
        RequiredEvidenceGroupCoverageGrader(),
        OODAnswerGrader(),
        ToolExecutionGrader(),
        TeamMetricsGrader(),
        SecurityControlGrader(),
        M10MemoryGrader(),
        M10ContextIntegrationGrader(),
    )
    return {grader.grader_id: grader for grader in graders}


def _normalize_route(value: object) -> str | None:
    return {
        "urgent": "urgent_care",
        "prescription": "human_review",
        "unanswerable": "abstain",
    }.get(value, value) if isinstance(value, str) else None
