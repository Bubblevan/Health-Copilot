from health_ai_copilot.agent import FinalTurn, ToolCall, ToolCallTurn
from health_ai_copilot.agent.messages import ToolResultMessage
from health_ai_copilot.agent.state import StopReason
from health_ai_copilot.contracts import Evidence, Route
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.policy.evidence import EvidenceAssessment, EvidenceDecision
from health_ai_copilot.verification.grounding import (
    ClaimResult,
    ClaimVerdict,
    GroundedClaim,
    GroundingResult,
)


def _evidence(source_id: str = "source-a") -> Evidence:
    return Evidence(source_id, "Stored title", "Stored excerpt", "https://example.org/a", 1.0)


class _Retriever:
    def __init__(self, initial: list[Evidence], recovery: list[Evidence] | None = None):
        self.initial = initial
        self.recovery = recovery if recovery is not None else initial
        self.calls: list[str] = []

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.calls.append(query)
        return (self.initial if len(self.calls) == 1 else self.recovery)[:top_k]


class _Model:
    def __init__(self, *turns: object):
        self.turns = list(turns)
        self.calls = 0

    def respond(self, messages, tools):
        self.calls += 1
        return self.turns.pop(0)


class _Policy:
    def __init__(self, assessment: EvidenceAssessment):
        self.assessment = assessment
        self.calls = 0

    def assess(self, question, evidence, proposed_query):
        self.calls += 1
        return self.assessment


class _Verifier:
    def __init__(self, result: GroundingResult):
        self.result = result
        self.calls = 0

    def verify(self, answer, claims, evidence):
        self.calls += 1
        return self.result


def _claim(source_id: str = "source-a") -> GroundedClaim:
    return GroundedClaim("该陈述由资料支持。", (source_id,))


def _supported() -> GroundingResult:
    return GroundingResult(True, (ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-a",)),))


def _pipeline(model, retriever, policy, verifier):
    return HealthCopilotPipeline(
        retriever,
        agent_model=model,
        evidence_policy=policy,
        grounding_verifier=verifier,
    )


def test_recoverable_executes_search_and_keeps_proposals_distinct() -> None:
    model = _Model(
        ToolCallTurn([ToolCall("recover", "search_knowledge", {"query": "rewrite"})]),
        FinalTurn("answer", claims=(_claim(),)),
    )
    retriever = _Retriever([_evidence()], [_evidence()])
    policy = _Policy(EvidenceAssessment(EvidenceDecision.RECOVERABLE))
    verifier = _Verifier(_supported())

    result = _pipeline(model, retriever, policy, verifier).answer("question")

    assert result.route == Route.ANSWER
    assert retriever.calls == ["question", "rewrite"]
    assert policy.calls == 1
    assert verifier.calls == 1


def test_sufficient_veto_appends_observation_then_allows_final_turn() -> None:
    model = _Model(
        ToolCallTurn([ToolCall("veto", "search_knowledge", {"query": "unneeded"})]),
        FinalTurn("answer", claims=(_claim(),)),
    )
    retriever = _Retriever([_evidence()])
    policy = _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("source-a",), ("direct_support",)))
    verifier = _Verifier(_supported())
    pipeline = _pipeline(model, retriever, policy, verifier)

    result = pipeline.answer("question")

    assert result.route == Route.ANSWER
    assert retriever.calls == ["question"]
    assert pipeline.last_agent_run is not None
    assert pipeline.last_agent_run.state.tool_proposals_used == 1
    assert pipeline.last_agent_run.state.tool_calls_used == 0
    observations = [item for item in pipeline.last_agent_run.state.session.messages if isinstance(item, ToolResultMessage)]
    assert observations[0].result.error is not None
    assert observations[0].result.error.code == "policy_denied"


def test_insufficient_and_conflicting_policy_fail_closed_without_tool_or_verifier() -> None:
    for decision, reason in ((EvidenceDecision.INSUFFICIENT, StopReason.EVIDENCE_INSUFFICIENT), (EvidenceDecision.CONFLICTING, StopReason.EVIDENCE_CONFLICTING)):
        retriever = _Retriever([_evidence()])
        verifier = _Verifier(_supported())
        pipeline = _pipeline(
            _Model(ToolCallTurn([ToolCall("stop", "search_knowledge", {"query": "x"})])),
            retriever,
            _Policy(EvidenceAssessment(decision)),
            verifier,
        )
        result = pipeline.answer("question")
        assert result.route == Route.ABSTAIN
        assert retriever.calls == ["question"]
        assert verifier.calls == 0
        assert pipeline.last_agent_run is not None
        assert pipeline.last_agent_run.stop_reason == reason


def test_policy_failure_and_unobserved_policy_source_fail_closed() -> None:
    class _BrokenPolicy:
        def assess(self, question, evidence, proposed_query):
            raise RuntimeError("provider failure")

    for policy in (_BrokenPolicy(), _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("fabricated",)))):
        retriever = _Retriever([_evidence()])
        pipeline = _pipeline(
            _Model(ToolCallTurn([ToolCall("stop", "search_knowledge", {"query": "x"})])),
            retriever,
            policy,
            _Verifier(_supported()),
        )
        assert pipeline.answer("question").route == Route.ABSTAIN
        assert retriever.calls == ["question"]
        assert pipeline.last_agent_run is not None
        assert pipeline.last_agent_run.stop_reason == StopReason.POLICY_ERROR


def test_grounding_rejects_unsupported_contradicted_and_missing_coverage() -> None:
    for result in (
        GroundingResult(True, (ClaimResult(0, ClaimVerdict.UNSUPPORTED),)),
        GroundingResult(True, (ClaimResult(0, ClaimVerdict.CONTRADICTED),)),
        GroundingResult(False, (ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-a",)),)),
    ):
        verifier = _Verifier(result)
        pipeline = _pipeline(
            _Model(FinalTurn("answer", claims=(_claim(),))),
            _Retriever([_evidence()]),
            _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT)),
            verifier,
        )
        response = pipeline.answer("question")
        assert response.route == Route.ABSTAIN
        assert response.safety_reasons == ["grounding_failed"]
        assert verifier.calls == 1


def test_fabricated_claim_citation_is_rejected_before_verifier() -> None:
    verifier = _Verifier(_supported())
    pipeline = _pipeline(
        _Model(FinalTurn("answer", claims=(_claim("fabricated"),))),
        _Retriever([_evidence()]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT)),
        verifier,
    )

    response = pipeline.answer("question")

    assert response.route == Route.ABSTAIN
    assert response.safety_reasons == ["invalid_citation"]
    assert verifier.calls == 0


def test_verifier_cannot_support_claim_from_a_different_observed_source() -> None:
    claim = _claim("source-a")
    verifier = _Verifier(
        GroundingResult(True, (ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-b",)),))
    )
    pipeline = _pipeline(
        _Model(FinalTurn("answer", claims=(claim,))),
        _Retriever([_evidence("source-a"), _evidence("source-b")]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT)),
        verifier,
    )

    response = pipeline.answer("question")

    assert response.route == Route.ABSTAIN
    assert response.safety_reasons == ["verifier_error"]
    assert pipeline.last_agent_run is not None
    assert pipeline.last_agent_run.stop_reason == StopReason.FINAL
    assert pipeline.last_harness_disposition == "verifier_error"


def test_supported_verdict_requires_a_supporting_source() -> None:
    verifier = _Verifier(GroundingResult(True, (ClaimResult(0, ClaimVerdict.SUPPORTED),)))
    pipeline = _pipeline(
        _Model(FinalTurn("answer", claims=(_claim(),))),
        _Retriever([_evidence()]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT)),
        verifier,
    )

    assert pipeline.answer("question").route == Route.ABSTAIN
    assert pipeline.last_harness_disposition == "verifier_error"


def test_policy_rejects_obviously_inconsistent_decision_reason_pairs() -> None:
    for assessment in (
        EvidenceAssessment(EvidenceDecision.SUFFICIENT, reason_codes=("out_of_scope",)),
        EvidenceAssessment(EvidenceDecision.CONFLICTING, reason_codes=("direct_support",)),
    ):
        pipeline = _pipeline(
            _Model(ToolCallTurn([ToolCall("policy", "search_knowledge", {"query": "q"})])),
            _Retriever([_evidence()]),
            _Policy(assessment),
            _Verifier(_supported()),
        )
        assert pipeline.answer("question").route == Route.ABSTAIN
        assert pipeline.last_agent_run is not None
        assert pipeline.last_agent_run.stop_reason == StopReason.POLICY_ERROR


def test_policy_is_not_called_for_direct_final_safety_or_empty_evidence() -> None:
    direct_policy = _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT))
    direct = _pipeline(_Model(FinalTurn("answer", claims=(_claim(),))), _Retriever([_evidence()]), direct_policy, _Verifier(_supported()))
    assert direct.answer("question").route == Route.ANSWER
    assert direct_policy.calls == 0

    safety_policy = _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT))
    safety = _pipeline(_Model(FinalTurn("unused", claims=(_claim(),))), _Retriever([_evidence()]), safety_policy, _Verifier(_supported()))
    assert safety.answer("持续胸痛怎么办").route == Route.URGENT_CARE
    assert safety_policy.calls == 0

    empty_policy = _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT))
    empty = _pipeline(_Model(FinalTurn("unused", claims=(_claim(),))), _Retriever([]), empty_policy, _Verifier(_supported()))
    assert empty.answer("question").route == Route.ABSTAIN
    assert empty_policy.calls == 0
