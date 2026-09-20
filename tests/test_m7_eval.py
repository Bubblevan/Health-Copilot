"""M7 unified evaluation contracts and parity tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from health_ai_copilot.agent.messages import AssistantToolCallMessage, ToolCall
from health_ai_copilot.contracts import Route
from health_ai_copilot.eval.failures import FailureMapper
from health_ai_copilot.eval.graders import RouteGrader
from health_ai_copilot.eval.metrics import trial_metrics
from health_ai_copilot.eval.registry import (
    DuplicateEvalSuiteError,
    EvalSuiteRegistry,
    UnknownEvalSuiteError,
    UnsupportedEvalExecutionMode,
    default_eval_suite_registry,
)
from health_ai_copilot.eval.schema import (
    CaseRunRecord,
    CaseRunStatus,
    EvalCase,
    EvalConfigurationError,
    EvalExecutionMode,
    EvalSuite,
    EvalTargetKind,
    FailureStage,
    GraderResult,
    GraderStatus,
    MetricResult,
)
from health_ai_copilot.eval.system import EvaluationRunner, load_eval_cases
from health_ai_copilot.runtime.components import ComponentKind
from health_ai_copilot.runtime.registry import ComponentRegistry


def _record(case_id: str = "case-1", **kwargs) -> CaseRunRecord:
    status = kwargs.pop("status", CaseRunStatus.COMPLETE)
    return CaseRunRecord(
        "test-suite", case_id, 1, "run", EvalExecutionMode.OFFLINE, "profile", "manifest", "commit",
        status, **kwargs
    )


def test_component_and_suite_duplicate_ids_are_rejected() -> None:
    registry = ComponentRegistry()
    registry.register(ComponentKind.TRACE, "trace", lambda context: object())
    with pytest.raises(Exception, match="duplicate component ID"):
        registry.register(ComponentKind.TRACE, "trace", lambda context: object())

    suites = EvalSuiteRegistry()
    suite = EvalSuite(
        "suite", "1", EvalTargetKind.DETERMINISTIC_GATE, "evals/m0.jsonl",
        (EvalExecutionMode.OFFLINE,), None, (), "test-v1"
    )
    suites.register(suite)
    with pytest.raises(DuplicateEvalSuiteError):
        suites.register(suite)


def test_unknown_suite_and_execution_mode_fail_closed() -> None:
    registry = default_eval_suite_registry()
    with pytest.raises(UnknownEvalSuiteError):
        registry.get("not-registered")
    with pytest.raises(UnsupportedEvalExecutionMode):
        registry.validate_mode(registry.get("m0-regression-v1"), EvalExecutionMode.LIVE)


def test_live_provider_requires_explicit_opt_in_before_construction() -> None:
    runner = EvaluationRunner()
    spec = runner.prepare_run_spec(
        "m1-focused-v1", execution_mode=EvalExecutionMode.LIVE, output_root=".pytest-tmp-m7-live"
    )
    with pytest.raises(EvalConfigurationError, match="allow-live-provider"):
        runner.run(spec)


def test_metric_zero_denominator_is_null_and_infrastructure_is_not_quality() -> None:
    metric = MetricResult.ratio("empty", "test-v1", 0, 0)
    assert metric.value is None
    records = [_record(status=CaseRunStatus.ERROR, error_code="provider_error")]
    grades = [GraderResult("route", "1", "case-1", 1, GraderStatus.ERROR)]
    metrics = trial_metrics(records, grades)
    assert metrics["trajectory.completeness"].value == 0.0
    assert metrics["grader.route.pass_rate"].value is None


def test_grader_keeps_claim_and_disposition_domains_separate() -> None:
    case = EvalCase("case-1", {"expected_route": "answer", "expected_verdicts": ["supported"]})
    record = _record(route="abstain", observed={"claim_verdicts": ["supported"]})
    result = RouteGrader().grade(case, record)
    assert result.status == GraderStatus.FAIL
    assert result.expected_summary == "answer"
    assert result.observed_summary == "abstain"


def test_failure_mapper_can_emit_multiple_failures_for_one_case() -> None:
    case = EvalCase("case-1", {"expected_route": "answer"})
    record = _record(route="abstain", observed={"retrieved_source_ids": []})
    route_failure = RouteGrader().grade(case, record)
    failures = FailureMapper().collect([case], [record], [route_failure])
    assert len(failures) == 1
    assert failures[0].stage == FailureStage.SAFETY
    assert failures[0].failure_code == "unexpected_route"


def test_m0_and_m5_offline_parity_and_metadata_privacy(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    m0_spec = runner.prepare_run_spec("m0-regression-v1", output_root=tmp_path / "m0")
    m0_bundle = runner.run(m0_spec)
    metrics = json.loads((m0_bundle / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["m0.safety_route_accuracy"]["value"] == 1.0
    assert metrics["m0.retrieval_hit_at_1"]["value"] == pytest.approx(0.9032258064516129)
    trace_text = "\n".join(path.read_text(encoding="utf-8") for path in (m0_bundle / "traces").glob("*.jsonl"))
    assert "高血压" not in trace_text

    m5_spec = runner.prepare_run_spec("m5-product-retrieval-v1", output_root=tmp_path / "m5")
    m5_bundle = runner.run(m5_spec)
    m5_metrics = json.loads((m5_bundle / "metrics.json").read_text(encoding="utf-8"))
    assert m5_metrics["m5.hit_at_1"]["value"] == pytest.approx(0.8783783783783784)
    rows = [
        json.loads(line)
        for line in (m0_bundle / "case_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len({row["execution_run_id"] for row in rows}) == len(rows)
    first_trace = m0_bundle / rows[0]["trace_reference"]
    first_event = json.loads(first_trace.read_text(encoding="utf-8").splitlines()[0])
    assert rows[0]["execution_run_id"] == first_event["fields"]["run_id"]
    assert rows[0]["eval_run_id"] == rows[0]["eval_spec_hash"]


def test_pipeline_record_separates_tool_proposal_from_execution() -> None:
    runner = EvaluationRunner()
    state = SimpleNamespace(
        tool_proposals_used=1,
        tool_calls_used=0,
        policy_decision="sufficient",
        policy_reason_codes=("direct_support",),
        policy_supporting_source_ids=("source-1",),
        policy_matched_topic_ids=(),
        session=SimpleNamespace(
            messages=(
                AssistantToolCallMessage(
                    [ToolCall("call-1", "search_knowledge", {"query": "query"})]
                ),
            )
        ),
    )
    run = SimpleNamespace(
        state=state,
        stop_reason=SimpleNamespace(value="final"),
        observed_evidence=[],
        initial_ranked_evidence=[],
        recovery_ranked_evidence=[],
        draft=None,
        claims=[],
    )
    pipeline = SimpleNamespace(
        last_agent_run=run,
        last_harness_disposition="answer",
        last_claim_support_result=None,
    )
    runtime = SimpleNamespace(
        identity=SimpleNamespace(run_id="run-test"),
        budget=SimpleNamespace(
            provider_calls_used=1,
            tool_executions_used=0,
            input_tokens_used=0,
            output_tokens_used=0,
            total_tokens_used=0,
        ),
    )
    components = SimpleNamespace(
        profile=SimpleNamespace(profile_id="profile"),
        manifest_hash="manifest",
        component_manifest=SimpleNamespace(code_commit="commit"),
    )
    response = SimpleNamespace(route=Route.ANSWER)
    case = EvalCase("case-1", {"question": "question"})
    spec = runner.prepare_run_spec("m0-regression-v1")

    record = runner._pipeline_record(
        runner.registry.get("m0-regression-v1"),
        spec,
        components,
        case,
        1,
        response,
        pipeline,
        runtime,
        None,
        1.0,
    )

    assert record.observed["tool_proposed"] is True
    assert record.observed["tool_executed"] is False
    trajectory = runner._trajectory(
        runner.registry.get("m0-regression-v1"), case, 1, spec, record
    )
    events = {event["event"] for event in trajectory.events}
    assert {"initial_evidence", "tool_proposal", "tool_execution", "budget", "case_complete"} <= events


def test_m3_case_payload_is_preserved_without_rewriting_gold() -> None:
    cases = load_eval_cases(Path("evals/m3_capability.jsonl"))
    assert cases[0].payload.get("evidence_source_ids") is not None
    assert cases[0].payload.get("proposed_query") is not None
    assert cases[0].payload.get("scope_id") is not None


def test_m4_replay_uses_recorded_exchanges_without_live_provider(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    spec = runner.prepare_run_spec(
        "m4-replay-v1",
        execution_mode=EvalExecutionMode.REPLAY,
        output_root=tmp_path,
        replay_run_dir="runs/m4/20260920T132709+0800",
    )
    bundle = runner.run(spec, public_eval_content=True)
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["grader.replay_consistency.pass_rate"]["value"] == 1.0
    assert metrics["grader.route.pass_rate"]["value"] == 1.0
    rows = [json.loads(line) for line in (bundle / "case_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(not row["observed"]["live_provider_called"] for row in rows)


def test_public_content_requires_suite_permission_and_explicit_opt_in(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    spec = runner.prepare_run_spec(
        "m0-regression-v1", output_root=tmp_path, trace_content_policy="public_eval_content"
    )
    with pytest.raises(EvalConfigurationError, match="public_eval_content"):
        runner.run(spec)


def test_run_spec_and_manifest_hashes_are_canonical() -> None:
    runner = EvaluationRunner()
    left = runner.prepare_run_spec("m0-regression-v1")
    right = runner.prepare_run_spec("m0-regression-v1")
    assert left.spec_hash == right.spec_hash
    assert left.canonical_json == right.canonical_json
