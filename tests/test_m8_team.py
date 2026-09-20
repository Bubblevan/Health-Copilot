from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from health_ai_copilot.agent.messages import FinalTurn, ToolCall, ToolCallTurn
from health_ai_copilot.agent.tools import ToolRegistry
from health_ai_copilot.contracts import Evidence
from health_ai_copilot.eval.schema import CaseRunStatus
from health_ai_copilot.eval.system import EvaluationRunner
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import CapabilityTopic, KnowledgeScope
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.policy.evidence import EvidenceAssessment, EvidenceDecision
from health_ai_copilot.runtime import (
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderResponse,
    RunContext,
    RuntimeBuilder,
    default_runtime_profiles,
)
from health_ai_copilot.runtime.budget import RunBudgetConfig
from health_ai_copilot.team import (
    AgentTeamOrchestrator,
    LeadDecision,
    LeadDecisionKind,
    LeadTaskProposal,
    Mailbox,
    TaskStore,
    TeamBudgetConfig,
    TeamRole,
    TeamStopReason,
    WorkerReport,
    worker_evidence_diversity,
    worker_recovery_evidence_diversity,
)
from health_ai_copilot.tools.search_knowledge import SearchKnowledgeTool
from health_ai_copilot.verification.grounding import GroundedClaim


def _evidence(source_id: str) -> Evidence:
    return Evidence(source_id, f"title-{source_id}", f"excerpt-{source_id}", "https://example.org", 1.0)


class ScriptedLead:
    def __init__(self, decisions: Sequence[LeadDecision]):
        self.decisions = list(decisions)
        self.calls = []

    def decide(self, question, initial_evidence, worker_reports, team_state_summary, *, runtime=None):
        self.calls.append((question, tuple(initial_evidence), tuple(worker_reports), dict(team_state_summary)))
        return self.decisions.pop(0)


class ScriptedWorker:
    def __init__(self, source_id: str, *, propose_tool: bool = False):
        self.source_id = source_id
        self.propose_tool = propose_tool
        self.sessions = []

    def respond(self, messages, tools, *, runtime=None):
        self.sessions.append(tuple(messages))
        if self.propose_tool and len(self.sessions) == 1:
            return ToolCallTurn([ToolCall("search-1", "search_knowledge", {"query": self.source_id})])
        return FinalTurn(
            "",
            [self.source_id],
            False,
            (GroundedClaim(f"claim-{self.source_id}", (self.source_id,)),),
        )


def _orchestrator(lead, worker, *, retriever=None, budget=None, policy=None, allowed_roles=None):
    retriever = retriever or (lambda query, top_k=3: [_evidence("worker-source")])
    registry = ToolRegistry([SearchKnowledgeTool(retriever)])
    return AgentTeamOrchestrator(
        lead,
        worker,
        registry,
        config=budget,
        evidence_policy=policy,
        allowed_roles=allowed_roles,
    )


def test_task_store_enforces_one_way_transitions_and_mailbox_graph():
    store = TaskStore()
    task = store.create_task(TeamRole.EVIDENCE, "collect facts")
    store.assign_task(task.task_id, "worker-evidence")
    store.start_task(task.task_id)
    store.complete_task(task.task_id)
    with pytest.raises(RuntimeError):
        store.start_task(task.task_id)

    mailbox = Mailbox()
    mailbox.send("lead", "worker-evidence", "task_assignment", task.task_id, {"x": 1})
    mailbox.send("worker-evidence", "lead", "worker_report", task.task_id, {"x": 1})
    with pytest.raises(ValueError):
        mailbox.send("worker-evidence", "worker-guideline", "worker_report", task.task_id, {})
    with pytest.raises(ValueError):
        mailbox.send("lead", "broadcast", "runtime_notice", None, {})


def test_team_lead_final_is_claim_first_and_has_no_worker_activity():
    source = _evidence("initial")
    lead = ScriptedLead([
        LeadDecision(LeadDecisionKind.FINAL, claims=(GroundedClaim("grounded", ("initial",)),))
    ])
    worker = ScriptedWorker("worker-source")
    orchestrator = _orchestrator(lead, worker)
    runtime = RunContext.create("m8", trace=None)
    result = orchestrator.run("question", [source], runtime=runtime)
    assert result.claims[0].citation_ids == ("initial",)
    assert result.stop_reason == TeamStopReason.FINAL
    assert result.state.tasks_created == 0
    assert result.state.workers_started == 0
    assert len(lead.calls) == 1


def test_workers_have_isolated_sessions_and_reports_are_the_only_cross_agent_path():
    source = _evidence("initial")
    lead = ScriptedLead([
        LeadDecision(
            LeadDecisionKind.DELEGATE,
            tasks=(
                LeadTaskProposal(TeamRole.EVIDENCE, "objective A"),
                LeadTaskProposal(TeamRole.GUIDELINE, "objective B"),
            ),
        ),
        LeadDecision(LeadDecisionKind.ABSTAIN),
    ])
    evidence_worker = ScriptedWorker("worker-source", propose_tool=True)
    guideline_worker = ScriptedWorker("initial")
    retriever = lambda query, top_k=3: [_evidence("worker-source")]
    registry = ToolRegistry([SearchKnowledgeTool(retriever)])
    orchestrator = AgentTeamOrchestrator(
        lead,
        {TeamRole.EVIDENCE: evidence_worker, TeamRole.GUIDELINE: guideline_worker},
        registry,
        config=TeamBudgetConfig(),
    )
    result = orchestrator.run("question", [source], runtime=RunContext.create("m8"))
    assert len(result.reports) == 2
    assert "objective A" in str(evidence_worker.sessions[0])
    assert "objective B" not in str(evidence_worker.sessions[0])
    assert "objective B" in str(guideline_worker.sessions[0])
    assert "objective A" not in str(guideline_worker.sessions[0])
    assert any(message.kind.value == "worker_report" for message in result.state.mailbox.list_messages())
    assert result.state.mailbox.list_messages()[0].kind.value == "task_assignment"
    assert result.state.topology == "star-supervisor-v1"
    assert result.state.scheduler == "sequential-v1"
    assert {item["role"] for item in result.state.to_metadata()["worker_roles"]} == {
        "evidence",
        "guideline",
    }


def test_builder_uses_role_specific_worker_system_contracts():
    cards = load_knowledge_cards("tests/fixtures/knowledge_cards")
    scope = KnowledgeScope(
        "fixture-scope",
        "1",
        "fixture-pack",
        "2026-09-20",
        "test",
        "fixture",
        ("synthetic",),
        (CapabilityTopic("synthetic", "fixture", tuple(card.id for card in cards)),),
    )
    components = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor()}
    ).build(
        default_runtime_profiles()["m8-team-bm25-v1"],
        cards=cards,
        knowledge_scope=scope,
    )
    orchestration_config = components.profile.config["orchestration"]
    assert orchestration_config["topology"] == "star-supervisor-v1"
    assert orchestration_config["scheduler"] == "sequential-v1"
    assert orchestration_config["allowed_roles"] == ["evidence", "guideline"]
    workers = components.orchestrator.worker_model
    evidence_prompt = workers[TeamRole.EVIDENCE]._system_prompt
    guideline_prompt = workers[TeamRole.GUIDELINE]._system_prompt
    assert evidence_prompt != guideline_prompt
    assert "factual evidence acquisition" in evidence_prompt.lower()
    assert "source coverage" in evidence_prompt.lower()
    assert "avoid recommendation synthesis" in evidence_prompt.lower()
    assert "guideline and recommendation context" in guideline_prompt.lower()
    assert "publisher and jurisdiction distinctions" in guideline_prompt.lower()
    assert "avoid unsupported factual expansion" in guideline_prompt.lower()


def test_allowed_roles_are_enforced_at_runtime():
    lead = ScriptedLead([
        LeadDecision(
            LeadDecisionKind.DELEGATE,
            tasks=(LeadTaskProposal(TeamRole.GUIDELINE, "guidance"),),
        )
    ])
    orchestrator = _orchestrator(
        lead,
        ScriptedWorker("initial"),
        allowed_roles=(TeamRole.EVIDENCE,),
    )
    result = orchestrator.run(
        "question", [_evidence("initial")], runtime=RunContext.create("m8")
    )
    assert result.stop_reason == TeamStopReason.INVALID_DELEGATION
    assert result.state.tasks_created == 0
    assert result.state.allowed_roles == ("evidence",)


def test_worker_evidence_overlap_and_unique_contribution_are_deterministic():
    reports = (
        WorkerReport(
            "task-e",
            "worker-evidence",
            TeamRole.EVIDENCE,
            "succeeded",
            citation_ids=("evidence-only",),
            observed_source_ids=("shared", "evidence-only"),
            recovery_source_ids=("evidence-only",),
            provider_calls=1,
        ),
        WorkerReport(
            "task-g",
            "worker-guideline",
            TeamRole.GUIDELINE,
            "succeeded",
            citation_ids=("guideline-only",),
            observed_source_ids=("shared", "guideline-only"),
            recovery_source_ids=("guideline-only",),
            provider_calls=1,
        ),
    )
    overlap, unique = worker_evidence_diversity(reports)
    assert overlap == pytest.approx(1 / 3)
    assert unique == pytest.approx(2 / 3)
    recovery_overlap, recovery_unique = worker_recovery_evidence_diversity(reports)
    assert recovery_overlap == pytest.approx(0.0)
    assert recovery_unique == pytest.approx(1.0)
    assert reports[0].to_metadata()["provider_calls"] == 1
    assert reports[0].to_metadata()["final_cited_source_ids"] == ["evidence-only"]
    assert reports[0].to_metadata()["recovery_source_ids"] == ["evidence-only"]


def test_m8_eval_aggregates_worker_diversity_metrics():
    record = SimpleNamespace(
        status=CaseRunStatus.COMPLETE,
        case_id="case-1",
        route="answer",
        harness_disposition="answer",
        provider_calls_used=4,
        tool_executions_used=0,
        input_tokens_used=0,
        output_tokens_used=0,
        total_tokens_used=0,
        elapsed_ms=1.0,
        observed={
            "team_delegated": True,
            "team_workers_started": 2,
            "team_lead_calls": 2,
            "worker_evidence_overlap": 1 / 3,
            "worker_unique_evidence_contribution": 2 / 3,
            "worker_recovery_evidence_overlap": 0.0,
            "worker_unique_recovery_contribution": 1.0,
            "team_worker_report_count": 2,
            "team_worker_completions": 2,
            "team_worker_productive_reports": 2,
        },
    )
    case = SimpleNamespace(
        case_id="case-1",
        payload={"category": "decomposable", "expected_route": "answer"},
    )
    metrics = EvaluationRunner()._m8_metrics([record], [case], "m8-test-v1")
    assert metrics["m8.worker_evidence_overlap"].value == pytest.approx(1 / 3)
    assert metrics["m8.worker_unique_evidence_contribution"].value == pytest.approx(2 / 3)
    assert metrics["m8.worker_recovery_evidence_overlap"].value == pytest.approx(0.0)
    assert metrics["m8.worker_unique_recovery_contribution"].value == pytest.approx(1.0)
    assert metrics["m8.worker_completion_rate"].value == pytest.approx(1.0)
    assert metrics["m8.worker_productive_report_rate"].value == pytest.approx(1.0)


def test_worker_completion_and_productivity_are_distinct():
    abstaining = WorkerReport(
        "task-e",
        "worker-evidence",
        TeamRole.EVIDENCE,
        "succeeded",
    )
    productive = WorkerReport(
        "task-g",
        "worker-guideline",
        TeamRole.GUIDELINE,
        "succeeded",
        recovery_source_ids=("guideline-source",),
    )
    assert abstaining.completed is True
    assert abstaining.productive is False
    assert productive.completed is True
    assert productive.productive is True


def test_worker_fabricated_citation_fails_task_and_is_not_added_to_ledger():
    lead = ScriptedLead([
        LeadDecision(LeadDecisionKind.DELEGATE, tasks=(LeadTaskProposal("evidence", "find facts"),)),
        LeadDecision(LeadDecisionKind.ABSTAIN),
    ])
    worker = ScriptedWorker("fabricated")
    result = _orchestrator(lead, worker).run("question", [_evidence("initial")], runtime=RunContext.create("m8"))
    assert result.reports[0].error_code == "worker_citation_provenance_violation"
    assert not result.state.evidence_ledger.contains("fabricated")
    assert result.state.task_store.list_tasks()[0].status.value == "failed"


def test_second_delegation_fails_closed():
    lead = ScriptedLead([
        LeadDecision(LeadDecisionKind.DELEGATE, tasks=(LeadTaskProposal("evidence", "one"),)),
        LeadDecision(LeadDecisionKind.DELEGATE, tasks=(LeadTaskProposal("guideline", "two"),)),
    ])
    result = _orchestrator(lead, ScriptedWorker("initial")).run(
        "question", [_evidence("initial")], runtime=RunContext.create("m8")
    )
    assert result.stop_reason == TeamStopReason.LEAD_SECOND_DELEGATION
    assert result.state.tasks_created == 1


def test_shared_parent_provider_budget_denies_before_extra_side_effect():
    cards = load_knowledge_cards("tests/fixtures/knowledge_cards")
    scope = KnowledgeScope(
        "fixture-scope",
        "1",
        "fixture-pack",
        "2026-09-20",
        "test",
        "fixture",
        ("synthetic",),
        (CapabilityTopic("synthetic", "fixture", tuple(card.id for card in cards)),),
    )
    executor = FakeProviderExecutor()
    profile = default_runtime_profiles()["m8-team-bm25-v1"]
    # The build smoke is intentionally separate from the lead-abstain mechanics;
    # this verifies the profile can be constructed with the existing provider seam.
    components = RuntimeBuilder(environment={"provider_executor": executor}).build(
        profile, cards=cards,
        knowledge_scope=scope,
    )
    assert components.orchestrator is not None


def test_built_m8_profile_runs_sequential_team_and_m3_final_verifier():
    cards = load_knowledge_cards("tests/fixtures/knowledge_cards")
    scope = KnowledgeScope(
        "fixture-scope",
        "1",
        "fixture-pack",
        "2026-09-20",
        "test",
        "fixture",
        ("synthetic",),
        (CapabilityTopic("synthetic", "fixture", tuple(card.id for card in cards)),),
    )
    responses = [
        ProviderResponse(
            "", ProviderCallKind.TEAM_LEAD, "ignored",
            '{"action":"delegate","tasks":[{"role":"evidence","objective":"facts"},{"role":"guideline","objective":"guidance"}]}',
        ),
        ProviderResponse(
            "", ProviderCallKind.TEAM_WORKER, "ignored",
            '{"claims":[{"text":"事实证据。","citation_ids":["fixture-hypertension"]}],"abstain":false}',
        ),
        ProviderResponse(
            "", ProviderCallKind.TEAM_WORKER, "ignored",
            '{"claims":[{"text":"指导证据。","citation_ids":["fixture-hypertension"]}],"abstain":false}',
        ),
        ProviderResponse(
            "", ProviderCallKind.TEAM_LEAD, "ignored",
            '{"action":"final","claims":[{"text":"最终事实。","citation_ids":["fixture-hypertension"]}]}',
        ),
        ProviderResponse(
            "", ProviderCallKind.CLAIM_SUPPORT_VERIFIER, "ignored",
            '{"claim_results":[{"claim_index":0,"verdict":"supported","supporting_source_ids":["fixture-hypertension"]}]}',
        ),
    ]
    components = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor(responses)}
    ).build(
        default_runtime_profiles()["m8-team-bm25-v1"],
        cards=cards,
        knowledge_scope=scope,
    )
    response = components.answer("高血压患者低盐饮食")
    assert response.route.value == "answer"
    assert components.orchestrator is not None


def test_worker_proposal_is_recorded_even_when_policy_vetoes_execution():
    class VetoPolicy:
        def assess(self, question, evidence, proposed_query, *, runtime=None):
            return EvidenceAssessment(
                EvidenceDecision.SUFFICIENT,
                supporting_source_ids=("initial",),
                reason_codes=("direct_support",),
            )

    lead = ScriptedLead([
        LeadDecision(LeadDecisionKind.DELEGATE, tasks=(LeadTaskProposal("evidence", "check"),)),
        LeadDecision(LeadDecisionKind.ABSTAIN),
    ])
    worker = ScriptedWorker("initial", propose_tool=True)
    result = _orchestrator(lead, worker, policy=VetoPolicy()).run(
        "question", [_evidence("initial")], runtime=RunContext.create("m8")
    )
    report = result.reports[0]
    assert report.tool_proposals == 1
    assert report.tool_executions == 0


def test_team_provider_budget_is_shared_and_denies_before_worker_b_side_effect():
    executor = FakeProviderExecutor(
        [
            ProviderResponse(
                "", ProviderCallKind.TEAM_LEAD, "ignored",
                '{"action":"delegate","tasks":[{"role":"evidence","objective":"A"},{"role":"guideline","objective":"B"}]}',
            ),
            ProviderResponse(
                "", ProviderCallKind.TEAM_WORKER, "ignored", None,
                tool_calls=(
                    {"id": "search-a", "function": {"name": "search_knowledge", "arguments": '{"query":"x"}'}},
                ),
            ),
            ProviderResponse(
                "", ProviderCallKind.TEAM_WORKER, "ignored",
                '{"claims":[{"text":"observed","citation_ids":["worker-source"]}],"abstain":false}',
            ),
        ]
    )
    from health_ai_copilot.agent.model import AgentOutputMode, OpenAICompatibleAgentModel
    from health_ai_copilot.team import OpenAICompatibleTeamLeadModel

    workers = OpenAICompatibleAgentModel(
        provider_executor=executor,
        model="worker",
        output_mode=AgentOutputMode.M3_CLAIM_FIRST,
        provider_call_kind=ProviderCallKind.TEAM_WORKER,
    )
    orchestrator = AgentTeamOrchestrator(
        OpenAICompatibleTeamLeadModel(executor, "lead"),
        workers,
        ToolRegistry([SearchKnowledgeTool(lambda query, top_k=3: [_evidence("worker-source")])]),
        config=TeamBudgetConfig(),
    )
    result = orchestrator.run(
        "question",
        [_evidence("initial")],
        runtime=RunContext.create("m8", budget=RunBudgetConfig(max_provider_calls=3)),
    )
    assert result.stop_reason == TeamStopReason.BUDGET_EXHAUSTED
    assert len(executor.requests) == 3
    assert result.reports[-1].error_code == "team_budget_exhausted"


def test_safety_route_creates_zero_team_activity():
    class ExplodingRetriever:
        def search(self, question, top_k=5):
            raise AssertionError("safety route must happen before retrieval")

    class ExplodingTeam:
        def run(self, question, initial_evidence, *, runtime):
            raise AssertionError("team must not be constructed for safety routes")

    pipeline = HealthCopilotPipeline(
        ExplodingRetriever(),
        orchestrator=ExplodingTeam(),
    )
    response = pipeline.answer("我现在胸痛、呼吸困难，需要马上怎么办？")
    assert response.route.value == "urgent_care"
    assert pipeline.last_team_run is None


def test_optional_orchestration_field_preserves_frozen_profile_hash():
    old = default_runtime_profiles()["m3-bm25-default"]
    equivalent = type(old)(
        old.profile_id,
        old.provider,
        old.retriever,
        old.policy,
        old.verifier,
        old.tool_set,
        old.trace,
        old.mode,
        old.config,
    )
    assert old.to_dict() == equivalent.to_dict()
    assert old.config_hash == equivalent.config_hash
