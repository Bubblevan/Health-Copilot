import json

import pytest

from health_ai_copilot import cli
from health_ai_copilot.agent.messages import FinalTurn
from health_ai_copilot.contracts import Evidence
from health_ai_copilot.knowledge.loader import load_knowledge_cards
from health_ai_copilot.knowledge.scope import CapabilityTopic, KnowledgeScope
from health_ai_copilot.runtime import (
    ComponentIdentity,
    ComponentKind,
    ComponentManifest,
    FakeProviderExecutor,
    ProviderCallKind,
    ProviderFailure,
    ProviderFailureKind,
    ProviderRequest,
    ProviderResponse,
    ReplayMetadata,
    ReplayProviderExecutor,
    RunContext,
    RuntimeBuilder,
    RuntimeProfile,
    default_runtime_profiles,
)
from health_ai_copilot.runtime.budget import RunBudgetConfig
from health_ai_copilot.runtime.builder import RuntimeTraceFactory
from health_ai_copilot.runtime.components import LearnedArtifactIdentity, config_hash
from health_ai_copilot.runtime.registry import (
    ComponentBuildContext,
    ComponentConstructionError,
    ComponentRegistry,
    DuplicateComponentError,
    UnknownComponentError,
)
from health_ai_copilot.verification.grounding import GroundedClaim

FIXTURE_DIR = "tests/fixtures/knowledge_cards"


def _profile(*, retriever: str = "retriever", mode: str = "m0", config=None) -> RuntimeProfile:
    return RuntimeProfile(
        profile_id=f"test-{retriever}-{mode}",
        provider="provider",
        retriever=retriever,
        tool_set=("search-knowledge-v1",) if mode != "m0" else (),
        mode=mode,
        config=config or {},
    )


def _context(profile: RuntimeProfile) -> ComponentBuildContext:
    return ComponentBuildContext(profile, (), None, {}, {})


def test_duplicate_component_id_rejected() -> None:
    registry = ComponentRegistry()
    registry.register(ComponentKind.RETRIEVER, "same", lambda context: object())
    with pytest.raises(DuplicateComponentError):
        registry.register(ComponentKind.RETRIEVER, "same", lambda context: object())


def test_unknown_profile_component_rejected() -> None:
    registry = ComponentRegistry()
    with pytest.raises(UnknownComponentError):
        registry.build(ComponentKind.RETRIEVER, "missing", _context(_profile(retriever="missing")))


def test_optional_dependency_failure_is_explicit() -> None:
    registry = ComponentRegistry()
    registry.register(
        ComponentKind.RETRIEVER,
        "learned",
        lambda context: (_ for _ in ()).throw(ImportError("package is missing")),
        optional_dependency="sentence-transformers",
    )
    with pytest.raises(ComponentConstructionError, match="sentence-transformers"):
        registry.build(ComponentKind.RETRIEVER, "learned", _context(_profile(retriever="learned")))


def test_no_silent_fallback_when_component_factory_fails() -> None:
    registry = ComponentRegistry()
    registry.register(
        ComponentKind.RETRIEVER,
        "learned",
        lambda context: (_ for _ in ()).throw(RuntimeError("model artifact is missing")),
    )
    with pytest.raises(ComponentConstructionError, match="model artifact is missing"):
        registry.build(ComponentKind.RETRIEVER, "learned", _context(_profile(retriever="learned")))


def test_profile_rejects_python_objects_in_declarative_config() -> None:
    with pytest.raises(ValueError, match="JSON-compatible"):
        RuntimeProfile("bad", "provider", "retriever", config={"object": object()})


def test_same_pipeline_class_accepts_bm25_and_hybrid_profiles() -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    response = ProviderResponse(
        "",
        ProviderCallKind.GENERATOR,
        "injected-provider-model",
        '{"answer":"ok","citation_ids":["fixture-hypertension"],"abstain":false}',
    )
    bm25 = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor([response])}
    ).build(default_runtime_profiles()["m0-bm25-default"], cards=cards)
    hybrid_profile = RuntimeProfile(
        "m0-hybrid-demo",
        "openai-compatible-v1",
        "hybrid-hashing-demo-v1",
        mode="m0",
    )
    hybrid = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor([response])}
    ).build(hybrid_profile, cards=cards)

    assert type(bm25.pipeline()) is type(hybrid.pipeline())
    assert bm25.pipeline().answer("高血压患者低盐饮食").route.value == "answer"
    assert hybrid.pipeline().answer("高血压患者低盐饮食").route.value == "answer"


@pytest.mark.parametrize(
    ("role_config", "expected_agent", "expected_generator"),
    [
        ({"agent": {"model": "agent-model"}}, "agent-model", "base-model"),
        ({"generator": {"model": "generator-model"}}, "base-model", "generator-model"),
    ],
)
def test_agent_and_generator_role_overrides_never_cross_inherit(
    role_config, expected_agent, expected_generator
) -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    evidence = Evidence(
        "fixture-hypertension",
        "title",
        "excerpt",
        "https://example.org",
        1.0,
    )

    generator_executor = FakeProviderExecutor(
        [
            ProviderResponse(
                "",
                ProviderCallKind.GENERATOR,
                "ignored",
                '{"answer":"ok","citation_ids":["fixture-hypertension"],"abstain":false}',
            )
        ]
    )
    generator_profile = RuntimeProfile(
        "m0-role-resolution",
        "openai-compatible-v1",
        "bm25-v1",
        mode="m0",
        config={"provider": {"model": "base-model"}, **role_config},
    )
    generator_components = RuntimeBuilder(
        environment={"provider_executor": generator_executor}
    ).build(generator_profile, cards=cards)
    generator_components.generator.generate(
        "问题", (evidence,), runtime=generator_components.create_run_context()
    )

    agent_executor = FakeProviderExecutor(
        [
            ProviderResponse(
                "",
                ProviderCallKind.AGENT,
                "ignored",
                '{"answer":"","citation_ids":[],"abstain":true}',
            )
        ]
    )
    agent_profile = RuntimeProfile(
        "m1-role-resolution",
        "openai-compatible-v1",
        "bm25-v1",
        tool_set=("search-knowledge-v1",),
        mode="m1",
        config={"provider": {"model": "base-model"}, **role_config},
    )
    agent_components = RuntimeBuilder(
        environment={"provider_executor": agent_executor}
    ).build(agent_profile, cards=cards)
    agent_components.agent_model.respond(
        (), (), runtime=agent_components.create_run_context()
    )

    assert generator_executor.requests[0].model == expected_generator
    assert agent_executor.requests[0].model == expected_agent


def test_same_pipeline_handles_bm25_and_m5_learned_profile_replay_component() -> None:
    """Exercise replacement mechanics without downloading learned weights in CI."""

    cards = load_knowledge_cards(FIXTURE_DIR)
    scope = _fixture_scope(cards)
    evidence = Evidence(
        "fixture-hypertension",
        "Synthetic hypertension lifestyle card",
        "合成测试证据：高血压患者教育可以介绍低盐饮食和规律活动。",
        "https://example.org/fixtures/hypertension",
        1.0,
    )

    def responses():
        return [
            ProviderResponse(
                "",
                ProviderCallKind.AGENT,
                "ignored",
                None,
                tool_calls=(
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_knowledge",
                            "arguments": '{"query":"低盐饮食"}',
                        },
                    },
                ),
            ),
            ProviderResponse(
                "",
                ProviderCallKind.POLICY,
                "ignored",
                '{"decision":"recoverable","supporting_source_ids":[],"reason_codes":["related_but_incomplete"],"matched_topic_ids":["fixture-lifestyle"]}',
            ),
            ProviderResponse(
                "",
                ProviderCallKind.AGENT,
                "ignored",
                '{"claims":[{"text":"低盐饮食属于生活方式教育内容。","citation_ids":["fixture-hypertension"]}],"abstain":false}',
            ),
            ProviderResponse(
                "",
                ProviderCallKind.CLAIM_SUPPORT_VERIFIER,
                "ignored",
                '{"claim_results":[{"claim_index":0,"verdict":"supported","supporting_source_ids":["fixture-hypertension"]}]}',
            ),
        ]

    bm25 = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor(responses())}
    ).build(
        default_runtime_profiles()["m3-bm25-default"], cards=cards, knowledge_scope=scope
    )
    learned = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor(responses())}
    ).build(
        default_runtime_profiles()["m3-hybrid-rerank-local"],
        cards=cards,
        knowledge_scope=scope,
        replay_initial_evidence={"高血压患者低盐饮食": (evidence,)},
    )

    assert type(bm25.pipeline()) is type(learned.pipeline())
    assert bm25.answer("高血压患者低盐饮食").route.value == "answer"
    assert learned.answer("高血压患者低盐饮食").route.value == "answer"
    assert type(learned.retriever).__name__ == "_RecordedEvidenceRetriever"


def test_expensive_component_built_once_and_run_state_is_fresh() -> None:
    count = {"retriever": 0}
    registry = ComponentRegistry()
    registry.register(ComponentKind.PROVIDER, "provider", lambda context: FakeProviderExecutor())

    def build_retriever(context):
        count["retriever"] += 1
        return object()

    registry.register(ComponentKind.RETRIEVER, "retriever", build_retriever)
    registry.register(ComponentKind.TRACE, "trace", lambda context: RuntimeTraceFactory())
    profile = RuntimeProfile(
        "test-lifecycle",
        "provider",
        "retriever",
        trace="trace",
    )
    components = RuntimeBuilder(registry, environment={"provider_executor": FakeProviderExecutor()}).build(
        profile, cards=[]
    )
    first = components.create_run_context()
    second = components.create_run_context()

    assert count["retriever"] == 1
    assert first.identity.run_id != second.identity.run_id
    assert first.budget is not second.budget
    assert first.trace is not second.trace


def test_component_manifest_hash_is_canonical_and_config_sensitive() -> None:
    identity = ComponentIdentity(
        ComponentKind.RETRIEVER,
        "bm25-v1",
        "impl.BM25",
        "1",
        config_hash=config_hash({"b": 0.75, "k1": 1.5}),
    )
    left = ComponentManifest("profile", (identity,), "pack", "scope")
    right = ComponentManifest("profile", (identity,), "pack", "scope")
    changed = ComponentManifest(
        "profile",
        (
            ComponentIdentity(
                ComponentKind.RETRIEVER,
                "bm25-v1",
                "impl.BM25",
                "1",
                config_hash=config_hash({"b": 0.75, "k1": 2.0}),
            ),
        ),
        "pack",
        "scope",
    )
    assert left.canonical_json == right.canonical_json
    assert left.manifest_hash == right.manifest_hash
    assert left.manifest_hash != changed.manifest_hash


def test_changed_learned_artifact_revision_changes_manifest_hash() -> None:
    def manifest(revision: str) -> ComponentManifest:
        artifact = LearnedArtifactIdentity(
            "sentence-transformers", "embedding", "model", revision, True, 384
        )
        identity = ComponentIdentity(
            ComponentKind.RETRIEVER,
            "dense-st-v1",
            "impl.Dense",
            "1",
            artifact_revision=revision,
            config_hash=config_hash({"model": "model", "revision": revision}),
            learned_artifacts=(artifact,),
        )
        return ComponentManifest("profile", (identity,))

    assert manifest("rev-a").manifest_hash != manifest("rev-b").manifest_hash


def test_manifest_persisted_and_trace_contains_identity_without_medical_content(tmp_path) -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    components = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor()}
    ).build(default_runtime_profiles()["m0-bm25-default"], cards=cards)
    manifest_path = tmp_path / "component_manifest.json"
    trace_path = tmp_path / "trace.jsonl"
    components.write_manifest(manifest_path)
    runtime = components.create_run_context(trace_path=trace_path)
    runtime.trace.close(status="complete")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert payload["manifest_hash"] == components.manifest_hash
    assert events[0]["fields"]["profile_id"] == "m0-bm25-default"
    assert events[0]["fields"]["component_manifest_hash"] == components.manifest_hash
    assert "高血压" not in trace_path.read_text(encoding="utf-8")


def test_replay_rejects_incompatible_profile_manifest() -> None:
    request = ProviderRequest.create(
        kind=ProviderCallKind.GENERATOR,
        model="fixture",
        messages=({"role": "user", "content": "fixture"},),
    )
    recorded = ProviderResponse("", ProviderCallKind.GENERATOR, "fixture", "ok")
    from health_ai_copilot.runtime import RecordedProviderExchange, provider_request_fingerprint

    replay = ReplayProviderExecutor(
        [RecordedProviderExchange(request, recorded, provider_request_fingerprint(request))],
        recorded_metadata=ReplayMetadata("profile-a", "manifest-a"),
    )
    runtime = RunContext.create(
        "replay", profile_id="profile-b", component_manifest_hash="manifest-a"
    )
    with pytest.raises(ProviderFailure) as error:
        replay.execute(request, runtime)
    assert error.value.kind == ProviderFailureKind.REPLAY_MISMATCH


def test_replay_initial_evidence_skips_live_learned_retriever_construction() -> None:
    profile = RuntimeProfile(
        "m1-learned-replay",
        "openai-compatible-v1",
        "hybrid-rerank-st-mmarco-v1",
        tool_set=("search-knowledge-v1",),
        mode="m1",
    )
    evidence = Evidence("source", "title", "excerpt", "https://example.org", 1.0)
    components = RuntimeBuilder(
        environment={"provider_executor": FakeProviderExecutor()}
    ).build(profile, cards=[], replay_initial_evidence={"question": (evidence,)})

    assert type(components.retriever).__name__ == "_RecordedEvidenceRetriever"
    assert components.tool_registry.lookup("search_knowledge") is not None


def test_learned_identity_can_mark_unresolved_revision_without_claiming_frozen() -> None:
    artifact = LearnedArtifactIdentity("sentence-transformers", "embedding", "model", None)
    identity = ComponentIdentity(
        ComponentKind.RETRIEVER,
        "dense-st-v1",
        "impl.Dense",
        "1",
        learned_artifacts=(artifact,),
    )
    assert identity.to_dict()["learned_artifacts"][0]["revision"] is None


def _fixture_scope(cards) -> KnowledgeScope:
    return KnowledgeScope(
        scope_id="fixture-scope",
        version="1",
        knowledge_pack_version="fixture-pack",
        reviewed_at="2026-09-20",
        reviewer="test",
        domain="synthetic fixture corpus",
        audiences=("synthetic_test",),
        topics=(
            CapabilityTopic(
                "fixture-lifestyle",
                "Synthetic lifestyle education",
                tuple(card.id for card in cards),
            ),
        ),
    )


def test_role_aware_model_routing_uses_one_provider_with_explicit_roles() -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    scope = _fixture_scope(cards)
    responses = [
        ProviderResponse(
            "",
            ProviderCallKind.AGENT,
            "ignored",
            None,
            tool_calls=(
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search_knowledge",
                        "arguments": '{"query":"低盐饮食"}',
                    },
                },
            ),
        ),
        ProviderResponse(
            "",
            ProviderCallKind.POLICY,
            "ignored",
            '{"decision":"recoverable","supporting_source_ids":[],"reason_codes":["related_but_incomplete"],"matched_topic_ids":["fixture-lifestyle"]}',
        ),
        ProviderResponse(
            "",
            ProviderCallKind.AGENT,
            "ignored",
            '{"claims":[{"text":"低盐饮食属于生活方式教育内容。","citation_ids":["fixture-hypertension"]}],"abstain":false}',
        ),
        ProviderResponse(
            "",
            ProviderCallKind.CLAIM_SUPPORT_VERIFIER,
            "ignored",
            '{"claim_results":[{"claim_index":0,"verdict":"supported","supporting_source_ids":["fixture-hypertension"]}]}',
        ),
    ]
    executor = FakeProviderExecutor(responses)
    profile = RuntimeProfile(
        "m3-role-routing",
        "openai-compatible-v1",
        "bm25-v1",
        policy="evidence-policy-m3-v1",
        verifier="claim-support-v1",
        tool_set=("search-knowledge-v1",),
        mode="m3",
        config={
            "provider": {"model": "provider-default"},
            "agent": {"model": "agent-model"},
            "policy": {"model": "policy-model"},
            "verifier": {"model": "verifier-model"},
        },
    )
    components = RuntimeBuilder(
        environment={"provider_executor": executor, "build_commit": "fixture-commit"}
    ).build(profile, cards=cards, knowledge_scope=scope)

    result = components.answer("高血压患者低盐饮食")

    assert result.route.value == "answer"
    assert [(request.kind, request.model) for request in executor.requests] == [
        (ProviderCallKind.AGENT, "agent-model"),
        (ProviderCallKind.POLICY, "policy-model"),
        (ProviderCallKind.AGENT, "agent-model"),
        (ProviderCallKind.CLAIM_SUPPORT_VERIFIER, "verifier-model"),
    ]
    assert all(request.model != "injected-provider-model" for request in executor.requests)


def test_role_routing_falls_back_to_explicit_provider_default_not_injected_model() -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    scope = _fixture_scope(cards)
    executor = FakeProviderExecutor(
        [
            ProviderResponse("", ProviderCallKind.AGENT, "ignored", '{"claims":[],"abstain":true}'),
            ProviderResponse(
                "",
                ProviderCallKind.POLICY,
                "ignored",
                '{"decision":"insufficient","supporting_source_ids":[],"reason_codes":["out_of_scope"],"matched_topic_ids":[]}',
            ),
        ]
    )
    profile = RuntimeProfile(
        "m3-provider-default-routing",
        "openai-compatible-v1",
        "bm25-v1",
        policy="evidence-policy-m3-v1",
        verifier="claim-support-v1",
        tool_set=("search-knowledge-v1",),
        mode="m3",
        config={"provider": {"model": "provider-default"}},
    )
    components = RuntimeBuilder(environment={"provider_executor": executor}).build(
        profile, cards=cards, knowledge_scope=scope
    )

    components.agent_model.respond((), (), runtime=components.create_run_context())
    components.evidence_policy.assess(
        "问题", (), "查询", runtime=components.create_run_context()
    )

    assert [request.model for request in executor.requests] == [
        "provider-default",
        "provider-default",
    ]


def test_shared_components_isolate_interleaved_run_contexts() -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    scope = _fixture_scope(cards)
    evidence = Evidence("fixture-hypertension", "title", "excerpt", "https://example.org", 1.0)
    executor = FakeProviderExecutor(
        [
            ProviderResponse("", ProviderCallKind.AGENT, "ignored", '{"claims":[],"abstain":true}'),
            ProviderResponse(
                "",
                ProviderCallKind.POLICY,
                "ignored",
                '{"decision":"insufficient","supporting_source_ids":[],"reason_codes":["out_of_scope"],"matched_topic_ids":[]}',
            ),
            ProviderResponse(
                "",
                ProviderCallKind.CLAIM_SUPPORT_VERIFIER,
                "ignored",
                '{"claim_results":[{"claim_index":0,"verdict":"supported","supporting_source_ids":["fixture-hypertension"]}]}',
            ),
        ]
    )
    profile = RuntimeProfile(
        "m3-interleaving",
        "openai-compatible-v1",
        "bm25-v1",
        policy="evidence-policy-m3-v1",
        verifier="claim-support-v1",
        tool_set=("search-knowledge-v1",),
        mode="m3",
        config={"provider": {"model": "shared-model"}},
    )
    components = RuntimeBuilder(environment={"provider_executor": executor}).build(
        profile, cards=cards, knowledge_scope=scope
    )
    first = components.create_run_context(budget=RunBudgetConfig(max_provider_calls=2))
    second = components.create_run_context(budget=RunBudgetConfig(max_provider_calls=2))

    first_turn = components.agent_model.respond((), (), runtime=first)
    second_policy = components.evidence_policy.assess(
        "问题", (), "查询", runtime=second
    )
    verifier_result = components.claim_support_verifier.verify(
        (GroundedClaim("事实", ("fixture-hypertension",)),),
        ((evidence,),),
        runtime=first,
    )

    assert isinstance(first_turn, FinalTurn)
    assert first_turn.abstain is True
    assert second_policy.decision.value == "insufficient"
    assert verifier_result.claim_results[0].verdict.value == "supported"
    assert first.budget.provider_calls_used == 2
    assert second.budget.provider_calls_used == 1
    assert all(not hasattr(component, "_runtime") for component in (
        components.agent_model,
        components.evidence_policy,
        components.claim_support_verifier,
    ))
    first_kinds = {
        event.fields["kind"]
        for event in first.trace.events
        if event.event_type.value == "provider_start"
    }
    second_kinds = {
        event.fields["kind"]
        for event in second.trace.events
        if event.event_type.value == "provider_start"
    }
    assert first_kinds == {"agent", "claim_support_verifier"}
    assert second_kinds == {"policy"}


def test_runtime_components_answer_creates_fresh_run_contexts(tmp_path) -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    executor = FakeProviderExecutor(
        [
            ProviderResponse("", ProviderCallKind.GENERATOR, "ignored", '{"answer":"ok","citation_ids":["fixture-hypertension"],"abstain":false}'),
            ProviderResponse("", ProviderCallKind.GENERATOR, "ignored", '{"answer":"ok","citation_ids":["fixture-hypertension"],"abstain":false}'),
        ]
    )
    components = RuntimeBuilder(environment={"provider_executor": executor}).build(
        default_runtime_profiles()["m0-bm25-default"], cards=cards
    )
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"

    assert components.answer("高血压患者低盐饮食", trace_path=first_path).route.value == "answer"
    assert components.answer("高血压患者低盐饮食", trace_path=second_path).route.value == "answer"
    first_start = json.loads(first_path.read_text(encoding="utf-8").splitlines()[0])["fields"]
    second_start = json.loads(second_path.read_text(encoding="utf-8").splitlines()[0])["fields"]
    assert first_start["run_id"] != second_start["run_id"]
    assert first_start["component_manifest_hash"] == components.manifest_hash
    assert first_start["code_commit"] == components.component_manifest.code_commit


def test_code_commit_and_replay_evidence_content_bind_manifest_identity() -> None:
    cards = load_knowledge_cards(FIXTURE_DIR)
    profile = default_runtime_profiles()["m1-bm25-default"]
    evidence_a = Evidence("source", "title", "excerpt-a", "https://example.org", 1.0)
    evidence_b = Evidence("source", "title", "excerpt-b", "https://example.org", 1.0)

    def build(commit, evidence):
        return RuntimeBuilder(
            environment={
                "provider_executor": FakeProviderExecutor(),
                "build_commit": commit,
            }
        ).build(profile, cards=cards, replay_initial_evidence={"question": (evidence,)})

    first = build("commit-a", evidence_a)
    same_commit_changed_evidence = build("commit-a", evidence_b)
    changed_commit = build("commit-b", evidence_a)

    assert first.manifest_hash != same_commit_changed_evidence.manifest_hash
    assert first.manifest_hash != changed_commit.manifest_hash
    assert first.create_run_context().identity.code_commit == "commit-a"


def test_cli_uses_profile_aware_runtime_components_answer(monkeypatch, capsys) -> None:
    calls = []

    class FakeComponents:
        manifest_hash = "manifest"

        def answer(self, question):
            calls.append(("answer", question))
            from health_ai_copilot.contracts import AssistantResponse, Route

            return AssistantResponse(Route.ABSTAIN, "fixture")

        def pipeline(self):
            calls.append(("pipeline",))
            raise AssertionError("CLI must not bypass RuntimeComponents.answer")

    class FakeBuilder:
        def build(self, profile, *, cards, knowledge_scope):
            return FakeComponents()

    monkeypatch.setattr(cli, "RuntimeBuilder", FakeBuilder)
    exit_code = cli.main(
        [
            "--profile",
            "m0-bm25-default",
            "--knowledge-dir",
            FIXTURE_DIR,
            "--question",
            "fixture question",
        ]
    )

    assert exit_code == 0
    assert calls == [("answer", "fixture question")]
    assert "profile: m0-bm25-default" in capsys.readouterr().out
