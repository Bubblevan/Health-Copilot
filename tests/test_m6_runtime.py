import json

import pytest

from health_ai_copilot.contracts import Evidence
from health_ai_copilot.knowledge.loader import load_knowledge_cards
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
from health_ai_copilot.runtime.builder import RuntimeTraceFactory
from health_ai_copilot.runtime.components import LearnedArtifactIdentity, config_hash
from health_ai_copilot.runtime.registry import (
    ComponentBuildContext,
    ComponentConstructionError,
    ComponentRegistry,
    DuplicateComponentError,
    UnknownComponentError,
)

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
