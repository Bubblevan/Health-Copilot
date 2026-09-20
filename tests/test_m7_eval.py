"""M7 unified evaluation contracts and parity tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from health_ai_copilot.agent.messages import AssistantToolCallMessage, ToolCall
from health_ai_copilot.contracts import KnowledgeCard, Route
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
from health_ai_copilot.policy.evidence import EvidenceAssessment, EvidenceDecision
from health_ai_copilot.policy.model import OpenAICompatibleEvidencePolicy
from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderFailure,
    ProviderFailureKind,
    ProviderResponse,
    RunContext,
)
from health_ai_copilot.runtime.budget import RunBudgetConfig
from health_ai_copilot.runtime.components import ComponentKind
from health_ai_copilot.runtime.registry import ComponentRegistry
from health_ai_copilot.runtime.trace import RunTrace, TraceContentPolicy
from health_ai_copilot.verification.grounding import (
    ClaimResult,
    ClaimVerdict,
    GroundingResult,
)


def _record(case_id: str = "case-1", **kwargs) -> CaseRunRecord:
    status = kwargs.pop("status", CaseRunStatus.COMPLETE)
    return CaseRunRecord(
        "test-suite", case_id, 1, "run", EvalExecutionMode.OFFLINE, "profile", "manifest", "commit",
        status, **kwargs
    )


def _card(source_id: str = "source-1") -> KnowledgeCard:
    return KnowledgeCard(
        source_id,
        "Stored title",
        "Stored excerpt",
        "https://example.org/source",
        "reviewed-publisher",
        "2026-01-01",
        "2026-01-02",
        "2026-01-03",
        "reviewer",
        "v1",
        None,
    )


class _NoExecutionComponent:
    def __init__(self, *, policy=None, grounding=None, claim_support=None, profile_id="m2-bm25-default", budget=None):
        self.profile = SimpleNamespace(profile_id=profile_id)
        self.manifest_hash = "manifest"
        self.component_manifest = SimpleNamespace(code_commit="commit")
        self.knowledge_scope = None
        self.retriever_calls = 0
        self.pipeline_calls = 0
        self.evidence_policy = policy
        self.grounding_verifier = grounding
        self.claim_support_verifier = claim_support
        self._budget = budget or RunBudgetConfig()
        self.retriever = SimpleNamespace(search=self._unexpected_retriever_call)

    def _unexpected_retriever_call(self, *args, **kwargs):
        self.retriever_calls += 1
        raise AssertionError("direct target must not invoke retrieval")

    def pipeline(self, **kwargs):
        self.pipeline_calls += 1
        raise AssertionError("direct target must not construct an agent pipeline")

    def create_run_context(self, *, trace_path, content_policy=TraceContentPolicy.METADATA_ONLY):
        return RunContext.create(
            "m2",
            budget=self._budget,
            trace=RunTrace(trace_path, content_policy),
        )


class _FixturePolicy:
    def __init__(self, assessment):
        self.assessment = assessment
        self.calls = []

    def assess(self, question, evidence, proposed_query, *, runtime=None):
        self.calls.append((question, tuple(evidence), proposed_query))
        return self.assessment


class _FixtureGroundingVerifier:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def verify(self, answer, claims, evidence, *, runtime=None):
        self.calls.append((answer, tuple(claims), tuple(evidence)))
        return self.result


class _FixtureClaimSupportVerifier:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def verify(self, claims, cited_evidence, *, runtime=None):
        self.calls.append((tuple(claims), tuple(tuple(item) for item in cited_evidence)))
        return self.result


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


def test_m8_profile_allowlist_rejects_confounded_profiles_before_run() -> None:
    runner = EvaluationRunner()
    with pytest.raises(EvalConfigurationError, match="allowed"):
        runner.prepare_run_spec(
            "m8-agent-team-focused-v1",
            execution_mode=EvalExecutionMode.LIVE,
            profile_id="m3-hybrid-rerank-local",
            output_root=".pytest-tmp-m8-invalid-profile",
        )

    valid = runner.prepare_run_spec(
        "m8-agent-team-focused-v1",
        execution_mode=EvalExecutionMode.LIVE,
        profile_id="m8-team-bm25-v1",
        output_root=".pytest-tmp-m8-valid-profile",
    )
    invalid = replace(valid, profile_id="m3-hybrid-rerank-local")
    with pytest.raises(EvalConfigurationError, match="allowed"):
        runner.run(invalid)


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


def test_eval_budget_overrides_are_validated_and_executable(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    spec = runner.prepare_run_spec(
        "m2-policy-v1",
        execution_mode=EvalExecutionMode.LIVE,
        budget_overrides={"max_provider_calls": 0},
        output_root=tmp_path,
    )
    components = runner._build_offline_components(spec, "m0-bm25-default", [_card()], None)
    assert components.run_budget.max_provider_calls == 0
    with pytest.raises(EvalConfigurationError, match="unknown budget override"):
        runner.prepare_run_spec(
            "m2-policy-v1",
            execution_mode=EvalExecutionMode.LIVE,
            budget_overrides={"not_a_budget": 1},
        )
    with pytest.raises(EvalConfigurationError, match="invalid budget override"):
        runner.prepare_run_spec(
            "m2-policy-v1",
            execution_mode=EvalExecutionMode.LIVE,
            budget_overrides={"max_provider_calls": -1},
        )


def test_policy_target_uses_frozen_evidence_without_retrieval_or_agent(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    policy = _FixturePolicy(
        EvidenceAssessment(
            EvidenceDecision.SUFFICIENT,
            supporting_source_ids=("source-1",),
            reason_codes=("direct_support",),
        )
    )
    components = _NoExecutionComponent(policy=policy)
    suite = runner.registry.get("m2-policy-v1")
    spec = runner.prepare_run_spec(
        suite.suite_id, execution_mode=EvalExecutionMode.LIVE, output_root=tmp_path
    )
    case = EvalCase(
        "policy-case",
        {
            "question": "fixture question",
            "evidence_source_ids": ["source-1"],
            "proposed_query": "fixture query",
        },
    )
    bundle = tmp_path / "policy-bundle"
    (bundle / "traces").mkdir(parents=True)
    records, _ = runner._run_policy(bundle, suite, spec, [case], components, [_card()], False)

    assert records[0].status == CaseRunStatus.COMPLETE
    assert policy.calls[0][1][0].source_id == "source-1"
    assert components.retriever_calls == 0
    assert components.pipeline_calls == 0


def test_budget_zero_prevents_direct_policy_provider_call(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    provider = FakeProviderExecutor(
        [ProviderResponse("provider-1", ProviderCallKind.POLICY, "fixture", "{}")]
    )
    policy = OpenAICompatibleEvidencePolicy(provider_executor=provider, model="fixture")
    components = _NoExecutionComponent(
        policy=policy,
        budget=RunBudgetConfig(max_provider_calls=0),
    )
    suite = runner.registry.get("m2-policy-v1")
    spec = runner.prepare_run_spec(
        suite.suite_id,
        execution_mode=EvalExecutionMode.LIVE,
        budget_overrides={"max_provider_calls": 0},
        output_root=tmp_path,
    )
    case = EvalCase(
        "policy-budget-case",
        {
            "question": "fixture question",
            "evidence_source_ids": ["source-1"],
            "proposed_query": "fixture query",
        },
    )
    bundle = tmp_path / "policy-budget-bundle"
    (bundle / "traces").mkdir(parents=True)
    records, _ = runner._run_policy(bundle, suite, spec, [case], components, [_card()], False)

    assert records[0].status == CaseRunStatus.ERROR
    assert provider.requests == []
    assert records[0].provider_calls_used == 0


def test_provider_failure_preserves_runtime_budget_provenance(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    provider = FakeProviderExecutor(
        failure=ProviderFailure(ProviderFailureKind.CONNECTION)
    )
    policy = OpenAICompatibleEvidencePolicy(provider_executor=provider, model="fixture")
    components = _NoExecutionComponent(policy=policy)
    suite = runner.registry.get("m2-policy-v1")
    spec = runner.prepare_run_spec(
        suite.suite_id, execution_mode=EvalExecutionMode.LIVE, output_root=tmp_path
    )
    case = EvalCase(
        "policy-provider-error",
        {
            "question": "fixture question",
            "evidence_source_ids": ["source-1"],
            "proposed_query": "fixture query",
        },
    )
    bundle = tmp_path / "policy-error-bundle"
    (bundle / "traces").mkdir(parents=True)
    records, _ = runner._run_policy(bundle, suite, spec, [case], components, [_card()], False)
    record = records[0]

    assert record.status == CaseRunStatus.ERROR
    assert len(provider.requests) == 1
    assert record.provider_calls_used == 1
    assert record.tool_executions_used == 0
    assert record.input_tokens_used == 0
    assert record.output_tokens_used == 0
    assert record.total_tokens_used == 0


@pytest.mark.parametrize(
    ("coverage_ok", "verdict", "expected_route", "expected_disposition"),
    [
        (True, ClaimVerdict.SUPPORTED, Route.ANSWER.value, "answer"),
        (True, ClaimVerdict.UNSUPPORTED, Route.ABSTAIN.value, "grounding_failed"),
        (True, ClaimVerdict.CONTRADICTED, Route.ABSTAIN.value, "grounding_failed"),
        (False, ClaimVerdict.SUPPORTED, Route.ABSTAIN.value, "grounding_failed"),
    ],
)
def test_m2_grounding_target_matches_pipeline_disposition(
    tmp_path: Path,
    coverage_ok: bool,
    verdict: ClaimVerdict,
    expected_route: str,
    expected_disposition: str,
) -> None:
    runner = EvaluationRunner()
    result = GroundingResult(
        coverage_ok,
        (ClaimResult(0, verdict, ("source-1",) if verdict == ClaimVerdict.SUPPORTED else ()),),
    )
    verifier = _FixtureGroundingVerifier(result)
    components = _NoExecutionComponent(grounding=verifier)
    suite = runner.registry.get("m2-grounding-v1")
    spec = runner.prepare_run_spec(
        suite.suite_id, execution_mode=EvalExecutionMode.LIVE, output_root=tmp_path
    )
    case = EvalCase(
        "grounding-case",
        {
            "answer": "fixture answer",
            "claims": [{"text": "fixture claim", "citation_ids": ["source-1"]}],
            "evidence_source_ids": ["source-1"],
        },
    )
    bundle = tmp_path / "grounding-bundle"
    (bundle / "traces").mkdir(parents=True)
    records, _ = runner._run_verifier(bundle, suite, spec, [case], components, [_card()], False)

    assert records[0].route == expected_route
    assert records[0].harness_disposition == expected_disposition
    assert components.retriever_calls == 0
    assert components.pipeline_calls == 0
    assert len(verifier.calls) == 1
    assert verifier.calls[0][2][0].source_id == "source-1"


def test_m3_fabricated_citation_is_rejected_before_provider(tmp_path: Path) -> None:
    runner = EvaluationRunner()
    verifier = _FixtureClaimSupportVerifier(
        SimpleNamespace(claim_results=())
    )
    components = _NoExecutionComponent(
        claim_support=verifier,
        profile_id="m3-bm25-default",
    )
    suite = runner.registry.get("m3-claim-support-v1")
    spec = runner.prepare_run_spec(
        suite.suite_id, execution_mode=EvalExecutionMode.LIVE, output_root=tmp_path
    )
    case = EvalCase(
        "m3-fabricated",
        {
            "claims": [{"text": "fixture claim", "citation_ids": ["not-observed"]}],
            "evidence_source_ids": ["source-1"],
        },
    )
    bundle = tmp_path / "m3-fabricated-bundle"
    (bundle / "traces").mkdir(parents=True)
    records, _ = runner._run_verifier(bundle, suite, spec, [case], components, [_card()], False)

    assert records[0].route == Route.ABSTAIN.value
    assert records[0].harness_disposition == "invalid_citation"
    assert verifier.calls == []


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
