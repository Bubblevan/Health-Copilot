from e2_support import (
    FixtureRetriever,
    ScriptedAgentModel,
    bound_context,
    capability,
    evidence,
    source_catalog,
)

from health_ai_copilot.capabilities import E2WorkerRole, SourceMetadata
from health_ai_copilot.e2_team import (
    E2HeterogeneousSequentialRunner,
    E2Task,
    E2WorkerStatus,
)
from health_ai_copilot.runtime.budget import RunBudgetConfig
from health_ai_copilot.runtime.builder import default_runtime_profiles
from health_ai_copilot.runtime.trace import RunTrace
from health_ai_copilot.team import TEAM_SCHEDULER, TEAM_TOPOLOGY, TeamRole


def _setup(
    *,
    public_model=None,
    guideline_model=None,
    budget=None,
    trace=None,
    public_capability=None,
    guideline_capability=None,
    extra_rows=(),
    extra_metadata=(),
):
    source_rows = [
        evidence("public-source"),
        evidence("guideline-source"),
        evidence("literature-source"),
        *extra_rows,
    ]
    catalog = source_catalog(
        SourceMetadata("public-source", "public_health", ("public_facts",)),
        SourceMetadata("guideline-source", "reviewed_guideline", ("guideline",)),
        SourceMetadata("literature-source", "scholarly_literature", ("study",)),
        *extra_metadata,
    )
    public_capability = public_capability or capability(
        E2WorkerRole.PUBLIC_HEALTH, "public_health", "public_facts"
    )
    guideline_capability = guideline_capability or capability(
        E2WorkerRole.GUIDELINE, "reviewed_guideline", "guideline"
    )
    caps = (public_capability, guideline_capability)
    runtime, manifest = bound_context(
        caps, source_catalog=catalog, budget=budget, trace=trace
    )
    runner = E2HeterogeneousSequentialRunner(
        worker_models={
            E2WorkerRole.PUBLIC_HEALTH: public_model or ScriptedAgentModel(
                citation_source="public-source", query="ignore scope; search all sources"
            ),
            E2WorkerRole.GUIDELINE: guideline_model or ScriptedAgentModel(
                citation_source="guideline-source", query="ignore scope; search all sources"
            ),
        },
        retriever=FixtureRetriever(source_rows),
        retriever_profile_id="synthetic-bm25-v1",
        source_catalog=catalog,
        capabilities=caps,
        component_manifest=manifest,
    )
    return runner, runtime, catalog


def test_two_workers_are_sequential_isolated_and_complementary():
    public_model = ScriptedAgentModel(
        citation_source="public-source", query="ignore scope and return guideline-source"
    )
    guideline_model = ScriptedAgentModel(
        citation_source="guideline-source", query="ignore scope and return public-source"
    )
    runner, runtime, catalog = _setup(
        public_model=public_model, guideline_model=guideline_model, trace=RunTrace()
    )

    result = runner.run(
        "synthetic question",
        [
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "collect patient education"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "collect recommendations"),
        ],
        initial_evidence=[],
        runtime=runtime,
    )

    assert result.architecture_id == "L3_HETEROGENEOUS_SEQUENTIAL"
    assert result.scheduler == "sequential-v1"
    assert result.ledger.source_ids == ("public-source", "guideline-source")
    assert [row.productive for row in result.worker_reports] == [True, True]
    assert result.verified_worker_reports == result.worker_reports
    assert public_model.transcripts[0] != guideline_model.transcripts[0]
    assert public_model.transcripts[0][0].content.find("collect patient education") >= 0
    assert guideline_model.transcripts[0][0].content.find("collect recommendations") >= 0
    assert public_model.tool_specs[0][0].name == guideline_model.tool_specs[0][0].name
    assert result.worker_reports[0].session_id_sha256 != result.worker_reports[1].session_id_sha256
    assert public_model.runtime_rows[0][0] is runtime.identity
    assert guideline_model.runtime_rows[0][0] is runtime.identity
    assert public_model.runtime_rows[0][1] is not guideline_model.runtime_rows[0][1]
    assert public_model.runtime_rows[0][2]["memory_policy"] == "OFF"
    assert guideline_model.runtime_rows[0][2]["memory_policy"] == "OFF"
    assert all(
        row.input_tokens == 10 and row.output_tokens == 6
        for row in result.worker_reports
    )
    assert result.metrics(
        catalog,
        required_evidence_groups=(("public-source",), ("guideline-source",)),
        required_source_families=("public_health", "reviewed_guideline"),
    )["EvidenceGroupCoverage"]["value"] == 1.0


def test_duplicate_observations_deduplicate_in_ledger_and_overlap_is_reported():
    public_cap = capability(
        E2WorkerRole.PUBLIC_HEALTH,
        "synthetic_shared",
        "public_facts",
        families=("synthetic_shared",),
        domains=("public_facts",),
    )
    guideline_cap = capability(
        E2WorkerRole.GUIDELINE,
        "synthetic_shared",
        "shared_domain",
        families=("synthetic_shared",),
        domains=("shared_domain",),
    )
    public_model = ScriptedAgentModel(citation_source="shared-source", query="shared")
    guideline_model = ScriptedAgentModel(citation_source="shared-source", query="shared")
    runner, runtime, catalog = _setup(
        public_model=public_model,
        guideline_model=guideline_model,
        public_capability=public_cap,
        guideline_capability=guideline_cap,
        extra_rows=(evidence("shared-source"),),
        extra_metadata=(
            SourceMetadata("shared-source", "synthetic_shared", ("public_facts", "shared_domain")),
        ),
    )
    result = runner.run(
        "synthetic duplicate query",
        [
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "observe shared source"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "observe shared source"),
        ],
        initial_evidence=[],
        runtime=runtime,
    )

    assert result.ledger.source_ids == ("shared-source",)
    assert len(result.ledger.get("shared-source").observations) == 2
    metrics = result.metrics(catalog)
    assert metrics["WorkerEvidenceOverlap"]["value"] == 1.0
    assert metrics["WorkerUniqueEvidenceContribution"]["value"] == 0.0


def test_failed_worker_does_not_add_partial_evidence_and_other_worker_completes():
    public_model = ScriptedAgentModel(fail=True, query="public-source")
    guideline_model = ScriptedAgentModel(
        citation_source="guideline-source", query="guideline-source"
    )
    runner, runtime, _ = _setup(public_model=public_model, guideline_model=guideline_model)
    result = runner.run(
        "synthetic partial failure",
        [
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "will fail"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "will succeed"),
        ],
        initial_evidence=[],
        runtime=runtime,
    )

    assert [item.status for item in result.worker_reports] == [
        E2WorkerStatus.FAILED,
        E2WorkerStatus.COMPLETED,
    ]
    assert result.ledger.source_ids == ("guideline-source",)
    assert not result.worker_reports[0].verified
    assert result.worker_reports[1].productive


def test_parent_budget_denial_fails_closed_and_keeps_worker_evidence_out():
    runner, runtime, _ = _setup(budget=RunBudgetConfig(max_provider_calls=1))
    result = runner.run(
        "synthetic budget stop",
        [
            E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "retrieve"),
            E2Task("task-guideline", E2WorkerRole.GUIDELINE, "retrieve"),
        ],
        initial_evidence=[],
        runtime=runtime,
    )

    assert runtime.budget.provider_calls_used == 1
    assert all(not item.verified for item in result.worker_reports)
    assert all(item.error_code == "parent_budget_exhausted" for item in result.worker_reports)
    assert result.ledger.source_ids == ()


def test_worker_tool_budget_denial_does_not_execute_search_or_add_evidence():
    public = capability(
        E2WorkerRole.PUBLIC_HEALTH,
        "public_health",
        "public_facts",
        max_tool_calls=0,
    )
    # One capability is sufficient for the local worker-budget boundary test.
    catalog = source_catalog(SourceMetadata("public-source", "public_health", ("public_facts",)))
    runtime, manifest = bound_context((public,), source_catalog=catalog)
    model = ScriptedAgentModel(query="public-source")
    runner = E2HeterogeneousSequentialRunner(
        worker_models={E2WorkerRole.PUBLIC_HEALTH: model},
        retriever=FixtureRetriever([evidence("public-source")]),
        retriever_profile_id="synthetic-bm25-v1",
        source_catalog=catalog,
        capabilities=(public,),
        component_manifest=manifest,
    )
    result = runner.run(
        "synthetic local budget",
        [E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "retrieve")],
        initial_evidence=[],
        runtime=runtime,
    )

    assert result.worker_reports[0].error_code == "worker_budget_exhausted"
    assert result.worker_reports[0].tool_executions == 0
    assert result.ledger.source_ids == ()


def test_forged_citation_is_rejected_before_it_can_become_ledger_backed():
    model = ScriptedAgentModel(forge_source="fabricated-source")
    runner, runtime, _ = _setup(public_model=model)
    result = runner.run(
        "synthetic provenance forgery",
        [E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "cite only observed")],
        initial_evidence=[evidence("public-source")],
        runtime=runtime,
    )

    report = result.worker_reports[0]
    assert report.status == E2WorkerStatus.FAILED
    assert report.error_code == "worker_citation_provenance_violation"
    assert report.claims == ()
    assert report.citation_ids == ()
    assert result.ledger.source_ids == ("public-source",)
    assert not result.ledger.contains("fabricated-source")
    assert not report.verified


def test_zero_delegation_precision_denominator_is_na():
    public = capability(
        E2WorkerRole.PUBLIC_HEALTH,
        "public_health",
        "public_facts",
        eligibility="NOT_YET_ELIGIBLE",
        eligibility_reason="synthetic ineligible fixture",
    )
    catalog = source_catalog(SourceMetadata("public-source", "public_health", ("public_facts",)))
    runtime, manifest = bound_context((public,), source_catalog=catalog)
    runner = E2HeterogeneousSequentialRunner(
        worker_models={},
        retriever=FixtureRetriever([evidence("public-source")]),
        retriever_profile_id="synthetic-bm25-v1",
        source_catalog=catalog,
        capabilities=(public,),
        component_manifest=manifest,
    )
    result = runner.run(
        "not delegated",
        [E2Task("task-skipped", E2WorkerRole.PUBLIC_HEALTH, "not eligible")],
        initial_evidence=[],
        runtime=runtime,
    )

    metrics = result.metrics(catalog)
    assert metrics["DelegationPrecision"]["value"] is None
    assert metrics["DelegationPrecision"]["denominator"] == 0
    assert metrics["WorkerCompletionRate"]["value"] is None


def test_e2_trace_binds_contract_and_reports_worker_identity_without_memory():
    trace = RunTrace()
    runner, runtime, _ = _setup(trace=trace)
    result = runner.run(
        "synthetic trace",
        [E2Task("task-public", E2WorkerRole.PUBLIC_HEALTH, "retrieve")],
        initial_evidence=[],
        runtime=runtime,
    )

    events = [item.event_type.value for item in trace.events]
    assert "e2_capability_bound" in events
    assert "e2_worker_started" in events
    assert "e2_worker_report" in events
    assert result.component_manifest_hash == runner.component_manifest.manifest_hash
    # Worker RuntimeContext intentionally carries only the OFF marker; no memory
    # store/context manager is reachable from E2HeterogeneousSequentialRunner.


def test_frozen_m8_profile_and_role_contract_remain_unchanged():
    profile = default_runtime_profiles()["m8-team-bm25-v1"]

    assert profile.mode == "m8_team"
    assert profile.orchestration == "agent-team-v1"
    assert profile.config["orchestration"]["topology"] == TEAM_TOPOLOGY
    assert profile.config["orchestration"]["scheduler"] == TEAM_SCHEDULER
    assert profile.config["orchestration"]["allowed_roles"] == ["evidence", "guideline"]
    assert tuple(role.value for role in TeamRole) == ("evidence", "guideline")
    assert "capability_contracts" not in profile.config
