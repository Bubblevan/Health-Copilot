
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from health_ai_copilot.agent import FinalTurn, ToolCall, ToolCallTurn
from health_ai_copilot.contracts import Evidence, Route
from health_ai_copilot.eval.runner import load_cases
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import (
    KnowledgeScopeLoadError,
    knowledge_scope_from_dict,
    load_knowledge_scope,
)
from health_ai_copilot.pipeline import HealthCopilotPipeline
from health_ai_copilot.policy.evidence import EvidenceAssessment, EvidenceDecision
from health_ai_copilot.tools.search_knowledge import SearchKnowledgeTool
from health_ai_copilot.verification.grounding import (
    ClaimResult,
    ClaimSupportResult,
    ClaimVerdict,
    GroundedClaim,
    GroundingResult,
    OpenAICompatibleClaimSupportVerifier,
    materialize_cited_evidence,
    validate_grounding_result,
)
from health_ai_copilot.verification.materialize import materialize_verified_claims


def _evidence(source_id: str = "source-a") -> Evidence:
    return Evidence(source_id, source_id, f"excerpt {source_id}", f"https://example.org/{source_id}", 1.0)


def _scope():
    card = load_knowledge_cards("tests/fixtures/knowledge_cards")[0]
    return knowledge_scope_from_dict(
        {
            "scope_id": "fixture-scope",
            "version": "1",
            "knowledge_pack_version": "fixture",
            "reviewed_at": "2026-09-20",
            "reviewer": "fixture-reviewer",
            "domain": "fixture hypertension education",
            "audiences": ["adult_patient_education"],
            "topics": [
                {
                    "id": "symptoms",
                    "description": "Reviewed symptom education.",
                    "source_ids": [card.id],
                }
            ],
        },
        [card],
    )


class _Retriever:
    def __init__(self, initial, recovery=None):
        self.initial = initial
        self.recovery = recovery if recovery is not None else initial
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append(query)
        return (self.initial if len(self.calls) == 1 else self.recovery)[:top_k]


class _Model:
    def __init__(self, *turns):
        self.turns = list(turns)

    def respond(self, messages, tools):
        return self.turns.pop(0)


class _Policy:
    def __init__(self, assessment):
        self.assessment = assessment
        self.inputs = []

    def assess(self, question, evidence, proposed_query):
        self.inputs.append((question, tuple(evidence), proposed_query))
        return self.assessment


class _SupportVerifier:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def verify(self, claims, evidence):
        self.calls += 1
        return self.result


def _claim(text="已验证事实。", source_id="source-a"):
    return GroundedClaim(text, (source_id,))


def _supported(*claims):
    return ClaimSupportResult(
        tuple(ClaimResult(index, ClaimVerdict.SUPPORTED, (claim.citation_ids[0],)) for index, claim in enumerate(claims))
    )


def _m3_pipeline(model, retriever, policy, verifier):
    return HealthCopilotPipeline(
        retriever,
        agent_model=model,
        evidence_policy=policy,
        knowledge_scope=_scope(),
        claim_support_verifier=verifier,
    )


def test_reviewed_product_scope_loads_and_covers_every_card() -> None:
    cards = load_knowledge_cards("data/knowledge_cards")
    scope = load_knowledge_scope("data/knowledge_scope.json", cards)

    assert scope.scope_id == "hypertension-patient-education-v1"
    assert len(scope.topics) == 10
    assert scope.reviewed_at == "2026-09-20"
    assert scope.reviewer == "manual-capability-mapping-review-2026-09-20"
    assert {source_id for topic in scope.topics for source_id in topic.source_ids} == {
        card.id for card in cards
    }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda data: {**data, "scope_id": ""},
        lambda data: {**data, "version": ""},
        lambda data: {**data, "reviewed_at": ""},
        lambda data: {**data, "reviewer": ""},
        lambda data: {**data, "topics": []},
        lambda data: {**data, "topics": [data["topics"][0], data["topics"][0]]},
        lambda data: {**data, "topics": [replace_topic(data["topics"][0], source_ids=["unknown"]) ]},
        lambda data: {**data, "topics": [replace_topic(data["topics"][0], source_ids=[])]},
    ],
)
def test_scope_validation_rejects_invalid_manifest(mutator) -> None:
    card = load_knowledge_cards("tests/fixtures/knowledge_cards")[0]
    data = {
        "scope_id": "scope",
        "version": "1",
        "knowledge_pack_version": "fixture",
        "reviewed_at": "2026-09-20",
        "reviewer": "fixture-reviewer",
        "domain": "fixture",
        "audiences": ["adult_patient_education"],
        "topics": [{"id": "topic", "description": "reviewed", "source_ids": [card.id]}],
    }

    with pytest.raises(KnowledgeScopeLoadError):
        knowledge_scope_from_dict(mutator(data), [card])


def replace_topic(topic, **changes):
    return {**topic, **changes}


def test_search_tool_exposes_scope_only_in_m3() -> None:
    retriever = _Retriever([_evidence()])
    m1 = SearchKnowledgeTool(retriever)
    m3 = SearchKnowledgeTool(retriever, knowledge_scope=_scope())

    assert m1.spec.description == "使用审核知识卡检索一个更适合当前问题的只读查询。"
    assert m1.spec.capability is None
    assert m3.spec.capability is not None
    assert m3.spec.capability.scope_id == "fixture-scope"
    assert "not web search" in m3.spec.description


def test_m3_eval_gold_freezes_scope_and_policy_inputs() -> None:
    cases = load_cases("evals/m3_capability.jsonl")
    scope = load_knowledge_scope(
        "data/knowledge_scope.json", load_knowledge_cards("data/knowledge_cards")
    )

    assert len(cases) == 16
    assert all(case["scope_id"] == scope.scope_id for case in cases)
    assert all(
        {
            "question",
            "evidence_source_ids",
            "proposed_query",
            "scope_id",
            "expected_decision",
            "expected_topic_ids",
        }
        <= case.keys()
        for case in cases
    )
    assert {case["category"] for case in cases} >= {
        "sufficient_in_scope",
        "recoverable_in_scope",
        "insufficient_out_of_scope",
        "insufficient_related_but_uncovered",
    }


def test_m3_recoverable_requires_valid_scope_topic_and_executes_once() -> None:
    claim = _claim()
    retriever = _Retriever([_evidence()], [_evidence()])
    policy = _Policy(EvidenceAssessment(EvidenceDecision.RECOVERABLE, matched_topic_ids=("symptoms",)))
    pipeline = _m3_pipeline(
        _Model(ToolCallTurn([ToolCall("search", "search_knowledge", {"query": "症状"})]), FinalTurn("ignored", claims=(claim,))),
        retriever,
        policy,
        _SupportVerifier(_supported(claim)),
    )

    response = pipeline.answer("高血压有症状吗？")

    assert response.route == Route.ANSWER
    assert retriever.calls == ["高血压有症状吗？", "症状"]
    assert pipeline.last_agent_run is not None
    assert pipeline.last_agent_run.state.policy_matched_topic_ids == ("symptoms",)


def test_m3_sufficient_policy_vetoes_unnecessary_tool_execution() -> None:
    claim = _claim()
    retriever = _Retriever([_evidence()])
    policy = _Policy(
        EvidenceAssessment(
            EvidenceDecision.SUFFICIENT,
            supporting_source_ids=("source-a",),
            matched_topic_ids=("symptoms",),
        )
    )
    pipeline = _m3_pipeline(
        _Model(
            ToolCallTurn([ToolCall("search", "search_knowledge", {"query": "不必要"})]),
            FinalTurn("ignored", claims=(claim,)),
        ),
        retriever,
        policy,
        _SupportVerifier(_supported(claim)),
    )

    response = pipeline.answer("当前资料足够的问题")

    assert response.route == Route.ANSWER
    assert retriever.calls == ["当前资料足够的问题"]


@pytest.mark.parametrize(
    "topics",
    [(), ("fabricated-topic",)],
)
def test_m3_invalid_recoverable_capability_fails_closed(topics) -> None:
    retriever = _Retriever([_evidence()])
    policy = _Policy(EvidenceAssessment(EvidenceDecision.RECOVERABLE, matched_topic_ids=topics))
    pipeline = _m3_pipeline(
        _Model(ToolCallTurn([ToolCall("search", "search_knowledge", {"query": "合理查询"})])),
        retriever,
        policy,
        _SupportVerifier(ClaimSupportResult(())),
    )

    response = pipeline.answer("当前证据不足的问题")

    assert response.route == Route.ABSTAIN
    assert response.safety_reasons == ["policy_error"]
    assert retriever.calls == ["当前证据不足的问题"]


def test_m3_plausible_but_uncovered_query_is_vetoed() -> None:
    retriever = _Retriever([_evidence()])
    policy = _Policy(EvidenceAssessment(EvidenceDecision.INSUFFICIENT, reason_codes=("out_of_scope",)))
    pipeline = _m3_pipeline(
        _Model(ToolCallTurn([ToolCall("search", "search_knowledge", {"query": "高血压 围术期 风险"})])),
        retriever,
        policy,
        _SupportVerifier(ClaimSupportResult(())),
    )

    response = pipeline.answer("高血压可以做全麻手术吗？")

    assert response.route == Route.ABSTAIN
    assert retriever.calls == ["高血压可以做全麻手术吗？"]


def test_m3_materializes_only_supported_claims_and_ignores_free_answer() -> None:
    claim = _claim("已验证事实。")
    pipeline = _m3_pipeline(
        _Model(FinalTurn("UNVERIFIED EXTRA FACT", claims=(claim,))),
        _Retriever([_evidence()]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("source-a",))),
        _SupportVerifier(_supported(claim)),
    )

    response = pipeline.answer("问题")

    assert response.route == Route.ANSWER
    assert response.message == "- 已验证事实。"
    assert "UNVERIFIED EXTRA FACT" not in response.message


@pytest.mark.parametrize(
    "result",
    [
        ClaimSupportResult((ClaimResult(0, ClaimVerdict.UNSUPPORTED),)),
        ClaimSupportResult((ClaimResult(0, ClaimVerdict.CONTRADICTED),)),
    ],
)
def test_m3_unsupported_and_contradicted_claims_abstain(result) -> None:
    claim = _claim()
    pipeline = _m3_pipeline(
        _Model(FinalTurn("ignored", claims=(claim,))),
        _Retriever([_evidence()]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("source-a",))),
        _SupportVerifier(result),
    )

    response = pipeline.answer("问题")

    assert response.route == Route.ABSTAIN
    assert response.safety_reasons == ["claim_support_failed"]
    assert pipeline.last_harness_disposition == "claim_support_failed"


def test_m3_rejects_fabricated_or_wrong_support_binding_before_materialization() -> None:
    fabricated = _claim(source_id="fabricated")
    fabricated_pipeline = _m3_pipeline(
        _Model(FinalTurn("ignored", claims=(fabricated,))),
        _Retriever([_evidence()]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("source-a",))),
        _SupportVerifier(_supported(fabricated)),
    )
    assert fabricated_pipeline.answer("问题").safety_reasons == ["invalid_citation"]

    claim = _claim()
    wrong_binding = ClaimSupportResult((ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-b",)),))
    binding_pipeline = _m3_pipeline(
        _Model(FinalTurn("ignored", claims=(claim,))),
        _Retriever([_evidence(), _evidence("source-b")]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("source-a",))),
        _SupportVerifier(wrong_binding),
    )
    assert binding_pipeline.answer("问题").safety_reasons == ["claim_support_verifier_error"]


def test_m3_rejects_empty_claim_set_or_text_and_deduplicates_exact_claims() -> None:
    for claims in ((), (_claim("   "),)):
        pipeline = _m3_pipeline(
            _Model(FinalTurn("ignored", claims=claims)),
            _Retriever([_evidence()]),
            _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("source-a",))),
            _SupportVerifier(ClaimSupportResult(())),
        )
        assert pipeline.answer("问题").safety_reasons == ["claim_materialization_error"]

    duplicate = (_claim("  事实 A  "), _claim("事实 A"), _claim("事实 B"))
    assert materialize_verified_claims(duplicate) == "- 事实 A\n- 事实 B"


@pytest.mark.parametrize(
    ("claims", "result"),
    [
        ((_claim(),), ClaimSupportResult(())),
        (
            (_claim(), _claim("另一条事实。")),
            ClaimSupportResult(
                (
                    ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-a",)),
                    ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-a",)),
                )
            ),
        ),
    ],
)
def test_m3_rejects_missing_or_duplicate_claim_result_index(claims, result) -> None:
    pipeline = _m3_pipeline(
        _Model(FinalTurn("ignored", claims=claims)),
        _Retriever([_evidence()]),
        _Policy(EvidenceAssessment(EvidenceDecision.SUFFICIENT, ("source-a",))),
        _SupportVerifier(result),
    )

    assert pipeline.answer("问题").safety_reasons == ["claim_support_verifier_error"]


def test_m2_rejects_duplicate_claim_result_index_without_changing_normal_replay() -> None:
    claims = (_claim(), _claim("另一条事实。"))
    evidence = (_evidence(),)
    duplicate = GroundingResult(
        True,
        (
            ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-a",)),
            ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-a",)),
        ),
    )

    with pytest.raises(ValueError, match="duplicate claim indexes"):
        validate_grounding_result(duplicate, claims, evidence)
    assert validate_grounding_result(
        GroundingResult(True, (ClaimResult(0, ClaimVerdict.SUPPORTED, ("source-a",)),)),
        (_claim(),),
        evidence,
    ).coverage_ok


def test_claim_support_provider_receives_only_each_claims_cited_evidence() -> None:
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            body = {
                "claim_results": [
                    {
                        "claim_index": 0,
                        "verdict": "supported",
                        "supporting_source_ids": ["source-a"],
                    }
                ]
            }
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))]
            )

    verifier = OpenAICompatibleClaimSupportVerifier.__new__(OpenAICompatibleClaimSupportVerifier)
    verifier._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    verifier.model_name = "fixture-model"
    verifier.temperature = 0
    claims = (_claim(source_id="source-a"),)
    cited = materialize_cited_evidence(claims, (_evidence("source-a"), _evidence("source-b")))

    result = verifier.verify(claims, cited)
    payload = json.loads(captured["messages"][1]["content"])

    assert result.claim_results[0].verdict == ClaimVerdict.SUPPORTED
    assert payload["claims"] == [
        {
            "claim_index": 0,
            "text": "已验证事实。",
            "cited_evidence": [{"source_id": "source-a", "excerpt": "excerpt source-a"}],
        }
    ]
    assert "source-b" not in captured["messages"][1]["content"]


def test_wrong_citation_binding_fixture_isolated_to_wrong_source() -> None:
    rows = [
        json.loads(line)
        for line in Path("evals/m3_claim_support.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    fixture = next(row for row in rows if row["id"] == "m3s-007")

    assert fixture["expected_verdicts"] == ["unsupported"]
    assert fixture["claims"][0]["citation_ids"] == ["who-hypertension-03-silent"]
    assert fixture["evidence_source_ids"] == [
        "who-hypertension-03-silent",
        "cdc-high-blood-pressure-measuring-02-repeat",
    ]
