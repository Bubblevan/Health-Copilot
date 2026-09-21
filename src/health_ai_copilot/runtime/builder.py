"""Declarative profile construction for the trusted in-process runtime."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..agent.loop import AgentLoopConfig
from ..agent.model import AgentOutputMode, OpenAICompatibleAgentModel
from ..agent.tools import ToolRegistry
from ..config import load_openai_config
from ..generation.openai_compatible import OpenAICompatibleGenerator
from ..knowledge.scope import KnowledgeScope
from ..mcp.client import MCP_PROTOCOL_VERSION, McpClient, McpServerConfig, McpToolAdapter
from ..mcp.permissions import (
    CapabilityClass,
    CapabilityPolicy,
    PermissionGuard,
    PermissionPolicy,
)
from ..mcp.sandbox import NoSandboxDevBackend
from ..mcp.server import build_search_knowledge_server
from ..pipeline import HealthCopilotPipeline
from ..policy.model import OpenAICompatibleEvidencePolicy
from ..retrieval.bm25 import BM25Retriever
from ..retrieval.dense import (
    DenseIndex,
    DenseRetriever,
    HashingEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from ..retrieval.documents import document_from_knowledge_card
from ..retrieval.hybrid import (
    HybridRetriever,
    RerankedRetriever,
    SentenceTransformerReranker,
    TokenOverlapReranker,
)
from ..team import (
    TEAM_ROLE_CONTRACT_IDS,
    TEAM_ROLE_SYSTEM_CONTRACTS,
    TEAM_SCHEDULER,
    TEAM_TOPOLOGY,
    AgentTeamOrchestrator,
    OpenAICompatibleTeamLeadModel,
    TeamBudgetConfig,
    TeamRole,
)
from ..tools.search_knowledge import SearchKnowledgeTool
from ..verification.grounding import (
    OpenAICompatibleClaimSupportVerifier,
    OpenAICompatibleGroundingVerifier,
)
from .budget import RunBudgetConfig
from .components import (
    BuiltComponent,
    ComponentIdentity,
    ComponentKind,
    ComponentManifest,
    LearnedArtifactIdentity,
    config_hash,
    implementation_name,
)
from .context import RunContext
from .profile import RuntimeProfile
from .provider import OpenAICompatibleProviderExecutor, ProviderCallKind, ProviderExecutor
from .registry import ComponentBuildContext, ComponentRegistry
from .trace import RunTrace, TraceContentPolicy, canonical_json_sha256


class RuntimeBuildError(RuntimeError):
    """Raised when a profile cannot be constructed without a fallback."""


@dataclass(frozen=True)
class RuntimeBuildConfig:
    """Per-builder environment/config boundary; profiles remain declarative."""

    provider_executor: ProviderExecutor | None = None
    provider_model: str | None = None
    provider_base_url: str | None = None
    artifact_revisions: Mapping[str, str] = field(default_factory=dict)
    knowledge_pack_version: str = "m0.2-2026-09-15"
    build_commit: str | None = None
    local_files_only: bool = True
    run_budget: RunBudgetConfig = field(default_factory=RunBudgetConfig)


class RuntimeTraceFactory:
    """A cheap per-run trace factory owned by a built runtime profile."""

    def __init__(self, default_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY):
        self.default_policy = default_policy

    def create(
        self,
        path: Path | None = None,
        *,
        content_policy: TraceContentPolicy | None = None,
    ) -> RunTrace:
        return RunTrace(path, content_policy or self.default_policy)


@dataclass
class RuntimeComponents:
    """Long-lived component graph plus factories for execution-local state."""

    profile: RuntimeProfile
    provider_executor: ProviderExecutor
    retriever: Any | None
    generator: Any | None
    agent_model: Any | None
    evidence_policy: Any | None
    grounding_verifier: Any | None
    claim_support_verifier: Any | None
    tool_registry: ToolRegistry
    trace_factory: RuntimeTraceFactory
    component_manifest: ComponentManifest
    model_name: str
    knowledge_scope: KnowledgeScope | None = None
    run_budget: RunBudgetConfig = field(default_factory=RunBudgetConfig)
    orchestrator: AgentTeamOrchestrator | None = None
    agent_config: AgentLoopConfig | None = None

    @property
    def manifest_hash(self) -> str:
        return self.component_manifest.manifest_hash

    @property
    def component_manifest_hash(self) -> str:
        return self.component_manifest.manifest_hash

    def write_manifest(self, path: Path) -> None:
        self.component_manifest.write(path)

    def create_run_context(
        self,
        *,
        runtime_mode: str | None = None,
        budget: RunBudgetConfig | None = None,
        trace_path: Path | None = None,
        content_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY,
    ) -> RunContext:
        trace = self.trace_factory.create(trace_path, content_policy=content_policy)
        return RunContext.create(
            runtime_mode or self.profile.mode,
            budget=budget or self.run_budget,
            trace=trace,
            profile_id=self.profile.profile_id,
            component_manifest_hash=self.manifest_hash,
            config_hash=self.manifest_hash,
            code_commit=self.component_manifest.code_commit,
        )

    def answer(
        self,
        question: str,
        *,
        budget: RunBudgetConfig | None = None,
        trace_path: Path | None = None,
        content_policy: TraceContentPolicy = TraceContentPolicy.METADATA_ONLY,
    ):
        """Primary profile-aware execution entry point with a fresh RunContext."""

        runtime = self.create_run_context(
            budget=budget,
            trace_path=trace_path,
            content_policy=content_policy,
        )
        return self.pipeline(runtime=runtime).answer(question)

    def pipeline(self, *, runtime: RunContext | None = None, tool_runner=None) -> HealthCopilotPipeline:
        """Create an execution facade over the already-built long-lived graph."""

        return HealthCopilotPipeline(
            self.retriever,
            generator=self.generator,
            agent_model=self.agent_model,
            tool_registry=self.tool_registry,
            evidence_policy=self.evidence_policy,
            grounding_verifier=self.grounding_verifier,
            claim_support_verifier=self.claim_support_verifier,
            orchestrator=self.orchestrator,
            agent_config=self.agent_config,
            knowledge_scope=self.knowledge_scope,
            runtime=runtime,
            tool_runner=tool_runner,
        )


def default_runtime_profiles() -> dict[str, RuntimeProfile]:
    """Return the source-defined M6 profiles and their explicit retrieval names."""

    provider = "openai-compatible-v1"
    search = ("search-knowledge-v1",)
    profiles = {
        "m0-bm25-default": RuntimeProfile(
            "m0-bm25-default", provider, "bm25-v1", tool_set=(), mode="m0"
        ),
        "m1-bm25-default": RuntimeProfile(
            "m1-bm25-default", provider, "bm25-v1", tool_set=search, mode="m1"
        ),
        "m2-bm25-default": RuntimeProfile(
            "m2-bm25-default",
            provider,
            "bm25-v1",
            policy="evidence-policy-m2-v1",
            verifier="grounding-v1",
            tool_set=search,
            mode="m2",
        ),
        "m3-bm25-default": RuntimeProfile(
            "m3-bm25-default",
            provider,
            "bm25-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m3",
        ),
        "m8-workflow-bm25-v1": RuntimeProfile(
            "m8-workflow-bm25-v1",
            provider,
            "bm25-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=(),
            mode="m8_workflow",
            config={"generator": {"model": None}},
        ),
        "m8-team-bm25-v1": RuntimeProfile(
            "m8-team-bm25-v1",
            provider,
            "bm25-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m8_team",
            orchestration="agent-team-v1",
            config={
                "team_lead": {"model": None},
                "team_worker": {"model": None},
                "orchestration": {
                    "topology": "star-supervisor-v1",
                    "scheduler": "sequential-v1",
                    "max_lead_calls": 2,
                    "max_workers_started": 2,
                    "max_tasks_created": 2,
                    "max_delegation_rounds": 1,
                    "max_worker_model_turns": 2,
                    "max_worker_tool_calls": 1,
                    "lead_contract": "team-lead-v2",
                    "worker_contract": "m3-claim-first-v1",
                    "allowed_roles": ["evidence", "guideline"],
                },
            },
        ),
        "m3-dense-hashing-demo": RuntimeProfile(
            "m3-dense-hashing-demo",
            provider,
            "dense-hashing-demo-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m3",
        ),
        "m3-hybrid-hashing-demo": RuntimeProfile(
            "m3-hybrid-hashing-demo",
            provider,
            "hybrid-hashing-demo-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m3",
        ),
        "m3-hybrid-token-rerank-demo": RuntimeProfile(
            "m3-hybrid-token-rerank-demo",
            provider,
            "hybrid-token-rerank-demo-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m3",
        ),
        "m3-dense-st-multilingual-minilm": RuntimeProfile(
            "m3-dense-st-multilingual-minilm",
            provider,
            "dense-st-multilingual-minilm-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m3",
            config={"retriever": {"embedding_revision": None}},
        ),
        "m3-hybrid-rrf-st": RuntimeProfile(
            "m3-hybrid-rrf-st",
            provider,
            "hybrid-rrf-st-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m3",
            config={"retriever": {"embedding_revision": None}},
        ),
        "m3-hybrid-rerank-local": RuntimeProfile(
            "m3-hybrid-rerank-local",
            provider,
            "hybrid-rerank-st-mmarco-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m3",
            config={
                "retriever": {
                    "embedding_revision": None,
                    "reranker_revision": None,
                    "candidate_top_k": 10,
                }
            },
        ),
        "m9-mcp-search-bm25-v1": RuntimeProfile(
            "m9-mcp-search-bm25-v1",
            provider,
            "bm25-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=("mcp-search-knowledge-v1",),
            mode="m9_mcp",
            config={
                "mcp": {
                    "server_id": "m9-search-knowledge-server",
                    "transport": "inproc_test",
                    "auth_mode": "unauthenticated_local",
                    "sandbox_profile_id": "no-sandbox-dev-v1",
                    "tool_namespace": "mcp::m9-search-knowledge-server",
                },
                "permission": {"policy_id": "permission-policy-v1"},
                "sandbox": {"profile_id": "no-sandbox-dev-v1", "contained": False},
            },
        ),
    }
    return profiles


def default_component_registry() -> ComponentRegistry:
    """Build the explicit source registration table used by RuntimeBuilder."""

    registry = ComponentRegistry()
    registry.register(
        ComponentKind.PROVIDER,
        "openai-compatible-v1",
        _build_provider,
        implementation="health_ai_copilot.runtime.provider.OpenAICompatibleProviderExecutor",
    )
    registry.register(ComponentKind.RETRIEVER, "bm25-v1", _build_bm25, implementation="health_ai_copilot.retrieval.bm25.BM25Retriever")
    registry.register(ComponentKind.RETRIEVER, "dense-hashing-demo-v1", _build_dense_hashing, implementation="health_ai_copilot.retrieval.dense.DenseRetriever[HashingEmbeddingBackend]")
    registry.register(ComponentKind.RETRIEVER, "hybrid-hashing-demo-v1", _build_hybrid_hashing, implementation="health_ai_copilot.retrieval.hybrid.HybridRetriever[hashing]")
    registry.register(ComponentKind.RETRIEVER, "hybrid-token-rerank-demo-v1", _build_hybrid_token, implementation="health_ai_copilot.retrieval.hybrid.HybridRetriever+TokenOverlapReranker")
    registry.register(ComponentKind.RETRIEVER, "dense-st-multilingual-minilm-v1", _build_dense_st, implementation="health_ai_copilot.retrieval.dense.DenseRetriever[SentenceTransformer]", optional_dependency="sentence-transformers")
    registry.register(ComponentKind.RETRIEVER, "hybrid-rrf-st-v1", _build_hybrid_rrf_st, implementation="health_ai_copilot.retrieval.hybrid.HybridRetriever[SentenceTransformer]", optional_dependency="sentence-transformers")
    registry.register(ComponentKind.RETRIEVER, "hybrid-rerank-st-mmarco-v1", _build_hybrid_rerank_st, implementation="health_ai_copilot.retrieval.hybrid.HybridRetriever+SentenceTransformerReranker", optional_dependency="sentence-transformers")
    registry.register(ComponentKind.POLICY, "evidence-policy-m2-v1", _build_policy_m2, implementation="health_ai_copilot.policy.model.OpenAICompatibleEvidencePolicy")
    registry.register(ComponentKind.POLICY, "evidence-policy-m3-v1", _build_policy_m3, implementation="health_ai_copilot.policy.model.OpenAICompatibleEvidencePolicy")
    registry.register(ComponentKind.VERIFIER, "grounding-v1", _build_grounding, implementation="health_ai_copilot.verification.grounding.OpenAICompatibleGroundingVerifier")
    registry.register(ComponentKind.VERIFIER, "claim-support-v1", _build_claim_support, implementation="health_ai_copilot.verification.grounding.OpenAICompatibleClaimSupportVerifier")
    registry.register(ComponentKind.TOOL, "search-knowledge-v1", _build_search_tool, implementation="health_ai_copilot.tools.search_knowledge.SearchKnowledgeTool")
    registry.register(ComponentKind.TOOL, "mcp-search-knowledge-v1", _build_mcp_search_tool, implementation="health_ai_copilot.mcp.client.McpToolAdapter")
    registry.register(ComponentKind.TRACE, "metadata-jsonl-v1", _build_metadata_trace, implementation="health_ai_copilot.runtime.trace.RunTrace")
    registry.register(ComponentKind.TRACE, "public-eval-jsonl-v1", _build_public_trace, implementation="health_ai_copilot.runtime.trace.RunTrace")
    registry.register(ComponentKind.ORCHESTRATION, "agent-team-v1", _build_orchestrator, implementation="health_ai_copilot.team.AgentTeamOrchestrator")
    registry.register(ComponentKind.MCP_CLIENT, "mcp-client-2026-07-28-v1", _build_mcp_client, implementation="health_ai_copilot.mcp.client.McpClient")
    registry.register(ComponentKind.PERMISSION, "permission-policy-v1", _build_permission, implementation="health_ai_copilot.mcp.permissions.PermissionGuard")
    registry.register(ComponentKind.SANDBOX, "no-sandbox-dev-v1", _build_dev_sandbox, implementation="health_ai_copilot.mcp.sandbox.NoSandboxDevBackend")
    return registry


class RuntimeBuilder:
    """Construct a profile graph once; request state is created separately."""

    def __init__(
        self,
        registry: ComponentRegistry | None = None,
        *,
        environment: RuntimeBuildConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.registry = registry or default_component_registry()
        if environment is None:
            self.environment = RuntimeBuildConfig()
        elif isinstance(environment, RuntimeBuildConfig):
            self.environment = environment
        else:
            self.environment = RuntimeBuildConfig(**dict(environment))
        if self.environment.build_commit is None:
            self.environment = replace(
                self.environment,
                build_commit=_discover_build_commit(),
            )

    def build(
        self,
        profile: RuntimeProfile,
        *,
        cards: Sequence[Any],
        knowledge_scope: KnowledgeScope | None = None,
        run_context_config: RunBudgetConfig | None = None,
        replay_initial_evidence: Mapping[str, Sequence[Any]] | None = None,
    ) -> RuntimeComponents:
        self._validate_profile(profile, knowledge_scope)
        instances: dict[tuple[ComponentKind, str], Any] = {}
        identities: list[ComponentIdentity] = []

        provider_default_model = self._provider_default_model(profile)
        agent_model_name = self._role_model(profile, "agent", provider_default_model)
        generator_model_name = self._role_model(profile, "generator", provider_default_model)
        team_lead_model_name = self._role_model(profile, "team_lead", provider_default_model)
        team_worker_model_name = self._role_model(profile, "team_worker", provider_default_model)
        build_environment = {
            **self._environment_dict(),
            "resolved_provider_model": provider_default_model,
            "resolved_team_lead_model": team_lead_model_name,
            "resolved_team_worker_model": team_worker_model_name,
        }

        def construct(kind: ComponentKind, component_id: str) -> Any:
            context = ComponentBuildContext(
                profile=profile,
                cards=tuple(cards),
                knowledge_scope=knowledge_scope,
                environment=build_environment,
                instances=instances,
            )
            binding = self.registry.build(kind, component_id, context)
            instances[(kind, component_id)] = binding.instance
            identities.append(binding.identity)
            return binding.instance

        provider = construct(ComponentKind.PROVIDER, profile.provider)
        if replay_initial_evidence is not None:
            retriever = _RecordedEvidenceRetriever(replay_initial_evidence)
            instances[(ComponentKind.RETRIEVER, profile.retriever)] = retriever
            retriever_identity = ComponentIdentity(
                ComponentKind.RETRIEVER,
                profile.retriever,
                "health_ai_copilot.runtime.builder.RecordedEvidenceRetriever",
                "1",
                config_hash=config_hash(
                    {
                        "replay_initial_evidence_sha256": _replay_evidence_hash(
                            replay_initial_evidence
                        )
                    }
                ),
            )
            identities.append(retriever_identity)
        else:
            retriever = construct(ComponentKind.RETRIEVER, profile.retriever)

        policy = construct(ComponentKind.POLICY, profile.policy) if profile.policy else None
        verifier = construct(ComponentKind.VERIFIER, profile.verifier) if profile.verifier else None

        if profile.mode == "m9_mcp":
            construct(ComponentKind.PERMISSION, "permission-policy-v1")
            construct(ComponentKind.SANDBOX, "no-sandbox-dev-v1")
            construct(ComponentKind.MCP_CLIENT, "mcp-client-2026-07-28-v1")
        tools = [construct(ComponentKind.TOOL, item) for item in profile.tool_set]
        tool_registry = ToolRegistry(tools)
        trace_factory = construct(ComponentKind.TRACE, profile.trace)

        generator = None
        agent_model = None
        if profile.mode == "m0":
            generator = OpenAICompatibleGenerator(
                provider_executor=provider,
                model=generator_model_name,
            )
        elif profile.mode in {"m1", "m2", "m3", "m9_mcp"}:
            output_mode = {
                "m1": AgentOutputMode.M1,
                "m2": AgentOutputMode.M2_GROUNDED,
                "m3": AgentOutputMode.M3_CLAIM_FIRST,
                "m9_mcp": AgentOutputMode.M3_CLAIM_FIRST,
            }[profile.mode]
            agent_model = OpenAICompatibleAgentModel(
                provider_executor=provider,
                model=agent_model_name,
                output_mode=output_mode,
            )
        elif profile.mode == "m8_workflow":
            agent_model = OpenAICompatibleAgentModel(
                provider_executor=provider,
                model=generator_model_name,
                output_mode=AgentOutputMode.M3_CLAIM_FIRST,
            )
        else:
            if profile.mode != "m8_team":
                raise RuntimeBuildError(f"unsupported runtime mode: {profile.mode}")

        orchestrator = None
        if profile.orchestration:
            orchestrator = construct(ComponentKind.ORCHESTRATION, profile.orchestration)
        agent_config = (
            AgentLoopConfig(max_model_turns=1, max_tool_calls=0)
            if profile.mode == "m8_workflow"
            else None
        )

        manifest = ComponentManifest(
            profile_id=profile.profile_id,
            components=tuple(identities),
            knowledge_pack_version=self.environment.knowledge_pack_version,
            knowledge_scope_version=knowledge_scope.version if knowledge_scope else None,
            profile_config_hash=profile.config_hash,
            code_commit=self.environment.build_commit,
        )
        return RuntimeComponents(
            profile,
            provider,
            retriever,
            generator,
            agent_model,
            policy,
            verifier if profile.mode == "m2" else None,
            verifier if profile.mode in {"m3", "m8_workflow", "m8_team", "m9_mcp"} else None,
            tool_registry,
            trace_factory,
            manifest,
            team_lead_model_name if profile.mode == "m8_team" else agent_model_name,
            knowledge_scope,
            run_context_config or self.environment.run_budget,
            orchestrator,
            agent_config,
        )

    def _environment_dict(self) -> dict[str, Any]:
        return {
            "provider_executor": self.environment.provider_executor,
            "provider_model": self.environment.provider_model,
            "provider_base_url": self.environment.provider_base_url,
            "artifact_revisions": dict(self.environment.artifact_revisions),
            "knowledge_pack_version": self.environment.knowledge_pack_version,
            "build_commit": self.environment.build_commit,
            "local_files_only": self.environment.local_files_only,
        }

    def _provider_default_model(self, profile: RuntimeProfile) -> str:
        config = _component_config(profile, ComponentKind.PROVIDER)
        explicit = self.environment.provider_model or config.get("model")
        if explicit:
            return str(explicit)
        if self.environment.provider_executor is not None:
            return "fixture-provider-model"
        return load_openai_config().model

    def _role_model(self, profile: RuntimeProfile, role: str, provider_default: str) -> str:
        role_config = profile.config.get(role, {})
        if not isinstance(role_config, Mapping):
            raise TypeError(f"profile {role} config must be an object")
        return str(role_config.get("model") or provider_default)

    def _validate_profile(self, profile: RuntimeProfile, scope: KnowledgeScope | None) -> None:
        if profile.mode not in {"m0", "m1", "m2", "m3", "m8_workflow", "m8_team", "m9_mcp"}:
            raise RuntimeBuildError(f"unsupported runtime mode: {profile.mode}")
        if profile.mode in {"m3", "m8_workflow", "m8_team", "m9_mcp"} and scope is None:
            raise RuntimeBuildError("claim-first profile requires a KnowledgeScope")
        if profile.mode == "m2" and (profile.policy is None or profile.verifier is None):
            raise RuntimeBuildError("m2 profile requires policy and verifier components")
        if profile.mode in {"m3", "m8_workflow", "m8_team", "m9_mcp"} and (profile.policy is None or profile.verifier is None):
            raise RuntimeBuildError("claim-first profile requires policy and verifier components")
        if profile.mode in {"m1", "m2", "m3"} and not profile.tool_set:
            raise RuntimeBuildError(f"{profile.mode} profile requires an explicit tool set")
        if profile.mode == "m8_team" and (profile.orchestration != "agent-team-v1" or not profile.tool_set):
            raise RuntimeBuildError("m8_team profile requires agent-team-v1 and an explicit tool set")
        if profile.mode == "m9_mcp" and profile.tool_set != ("mcp-search-knowledge-v1",):
            raise RuntimeBuildError("m9_mcp profile requires the MCP search tool")


class _RecordedEvidenceRetriever:
    """Replay-only initial evidence source; it never constructs a live retriever."""

    def __init__(self, rows: Mapping[str, Sequence[Any]]) -> None:
        self._rows = {key: tuple(value) for key, value in rows.items()}

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        return list(self._rows.get(query, ()))[:top_k]


def _component_config(profile: RuntimeProfile, kind: ComponentKind) -> Mapping[str, Any]:
    raw = profile.config.get(kind.value, {})
    if not isinstance(raw, Mapping):
        raise TypeError(f"profile {kind.value} config must be an object")
    return raw


def _identity(
    kind: ComponentKind,
    component_id: str,
    implementation: str,
    config: object,
    *,
    artifact_revision: str | None = None,
    learned_artifacts: tuple[LearnedArtifactIdentity, ...] = (),
) -> ComponentIdentity:
    return ComponentIdentity(
        kind,
        component_id,
        implementation,
        "1",
        artifact_revision=artifact_revision,
        config_hash=config_hash(config),
        learned_artifacts=learned_artifacts,
    )


def _build_provider(context: ComponentBuildContext) -> BuiltComponent:
    cfg = _component_config(context.profile, ComponentKind.PROVIDER)
    executor = context.environment.get("provider_executor")
    model = context.environment.get("resolved_provider_model") or context.environment.get(
        "provider_model"
    ) or cfg.get("model")
    base_url = context.environment.get("provider_base_url") or cfg.get("base_url")
    if executor is None:
        config = load_openai_config(model_override=str(model) if model else None)
        executor = OpenAICompatibleProviderExecutor(config)
        model, base_url = config.model, config.base_url
    if not hasattr(executor, "execute"):
        raise TypeError("provider executor must implement execute")
    roles = {
        "agent": _role_model(context, "agent"),
        "generator": _role_model(context, "generator"),
        "policy": _role_model(context, "policy"),
        "verifier": _role_model(context, "verifier"),
    }
    if context.profile.orchestration is not None:
        roles.update(
            {
                "team_lead": _role_model(context, "team_lead"),
                "team_worker": _role_model(context, "team_worker"),
            }
        )
    return BuiltComponent(
        executor,
        _identity(
            ComponentKind.PROVIDER,
            context.profile.provider,
            "health_ai_copilot.runtime.provider.OpenAICompatibleProviderExecutor",
            {
                "model": model,
                "base_url": base_url,
                "roles": roles,
            },
        ),
    )


def _build_bm25(context: ComponentBuildContext) -> BuiltComponent:
    cfg = _component_config(context.profile, ComponentKind.RETRIEVER)
    k1, b = float(cfg.get("k1", 1.5)), float(cfg.get("b", 0.75))
    retriever = BM25Retriever(context.cards, k1=k1, b=b)
    return BuiltComponent(retriever, _identity(ComponentKind.RETRIEVER, context.profile.retriever, implementation_name(retriever), {"k1": k1, "b": b}))


def _hashing_dense(context: ComponentBuildContext):
    cfg = _component_config(context.profile, ComponentKind.RETRIEVER)
    dimension = int(cfg.get("dimension", 256))
    backend = HashingEmbeddingBackend(dimension)
    retriever = DenseRetriever.from_knowledge_cards(
        context.cards,
        backend,
        knowledge_pack_version=context.environment["knowledge_pack_version"],
        build_commit=context.environment["build_commit"] or "unknown",
    )
    return retriever, {"dimension": dimension, "backend": backend.identity}


def _build_dense_hashing(context: ComponentBuildContext) -> BuiltComponent:
    retriever, cfg = _hashing_dense(context)
    return BuiltComponent(retriever, _identity(ComponentKind.RETRIEVER, context.profile.retriever, "health_ai_copilot.retrieval.dense.DenseRetriever[HashingEmbeddingBackend]", cfg))


def _build_hybrid_hashing(context: ComponentBuildContext) -> BuiltComponent:
    bm25 = BM25Retriever(context.cards)
    dense, cfg = _hashing_dense(context)
    retriever = HybridRetriever(bm25, dense)
    return BuiltComponent(retriever, _identity(ComponentKind.RETRIEVER, context.profile.retriever, "health_ai_copilot.retrieval.hybrid.HybridRetriever[hashing]", {**cfg, "rrf_k": retriever.rrf_k}))


def _build_hybrid_token(context: ComponentBuildContext) -> BuiltComponent:
    bm25 = BM25Retriever(context.cards)
    dense, dense_cfg = _hashing_dense(context)
    hybrid = HybridRetriever(bm25, dense)
    reranker = TokenOverlapReranker()
    retriever = RerankedRetriever(hybrid, reranker, candidate_top_k=int(_component_config(context.profile, ComponentKind.RETRIEVER).get("candidate_top_k", 10)))
    return BuiltComponent(retriever, _identity(ComponentKind.RETRIEVER, context.profile.retriever, "health_ai_copilot.retrieval.hybrid.HybridRetriever+TokenOverlapReranker", {**dense_cfg, "candidate_top_k": retriever.candidate_top_k}))


def _artifact_revision(context: ComponentBuildContext, key: str, cfg: Mapping[str, Any]) -> str | None:
    explicit = cfg.get(key)
    revisions = context.environment.get("artifact_revisions", {})
    if explicit is None:
        explicit = revisions.get(key) or revisions.get(context.profile.retriever)
    if explicit is None:
        return None
    if not isinstance(explicit, str) or not explicit.strip():
        raise ValueError(f"{key} must be a non-empty revision string or null")
    return explicit.strip()


def _learned_backend(context: ComponentBuildContext):
    cfg = _component_config(context.profile, ComponentKind.RETRIEVER)
    model_id = str(cfg.get("embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    revision = _artifact_revision(context, "embedding_revision", cfg)
    local_files_only = bool(cfg.get("local_files_only", context.environment["local_files_only"]))
    backend = SentenceTransformerEmbeddingBackend(
        model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    artifact = LearnedArtifactIdentity(
        "sentence-transformers",
        "embedding",
        model_id,
        revision,
        local_files_only,
        backend.dimension,
    )
    return backend, artifact


def _learned_dense(context: ComponentBuildContext):
    backend, artifact = _learned_backend(context)
    cfg = _component_config(context.profile, ComponentKind.RETRIEVER)
    index = DenseIndex.build(
        [document_from_knowledge_card(card) for card in context.cards],
        backend,
        knowledge_pack_version=context.environment["knowledge_pack_version"],
        build_commit=context.environment["build_commit"] or "unknown",
    )
    index_dir = cfg.get("index_dir")
    if index_dir is not None:
        # Loading is strict: a missing or stale manifest is a construction error,
        # and the freshly built candidate is never used as a fallback.
        index = DenseIndex.load(Path(str(index_dir)), expected=index.manifest)
    retriever = DenseRetriever(index, backend)
    return retriever, artifact


def _build_dense_st(context: ComponentBuildContext) -> BuiltComponent:
    retriever, artifact = _learned_dense(context)
    return BuiltComponent(retriever, _identity(ComponentKind.RETRIEVER, context.profile.retriever, "health_ai_copilot.retrieval.dense.DenseRetriever[SentenceTransformer]", {"embedding": artifact.to_dict()}, artifact_revision=artifact.revision, learned_artifacts=(artifact,)))


def _build_hybrid_rrf_st(context: ComponentBuildContext) -> BuiltComponent:
    bm25 = BM25Retriever(context.cards)
    dense, artifact = _learned_dense(context)
    retriever = HybridRetriever(bm25, dense)
    return BuiltComponent(retriever, _identity(ComponentKind.RETRIEVER, context.profile.retriever, "health_ai_copilot.retrieval.hybrid.HybridRetriever[SentenceTransformer]", {"embedding": artifact.to_dict(), "rrf_k": retriever.rrf_k}, artifact_revision=artifact.revision, learned_artifacts=(artifact,)))


def _build_hybrid_rerank_st(context: ComponentBuildContext) -> BuiltComponent:
    cfg = _component_config(context.profile, ComponentKind.RETRIEVER)
    bm25 = BM25Retriever(context.cards)
    dense, embedding_artifact = _learned_dense(context)
    hybrid = HybridRetriever(bm25, dense)
    model_id = str(cfg.get("reranker_model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    revision = _artifact_revision(context, "reranker_revision", cfg)
    local_files_only = bool(cfg.get("local_files_only", context.environment["local_files_only"]))
    reranker = SentenceTransformerReranker(model_id, revision=revision, local_files_only=local_files_only)
    reranker_artifact = LearnedArtifactIdentity("sentence-transformers", "cross-encoder", model_id, revision, local_files_only)
    retriever = RerankedRetriever(hybrid, reranker, candidate_top_k=int(cfg.get("candidate_top_k", 10)))
    revisions = [item.revision for item in (embedding_artifact, reranker_artifact) if item.revision]
    artifact_revision = ";".join(revisions) if len(revisions) == 2 else None
    return BuiltComponent(retriever, _identity(ComponentKind.RETRIEVER, context.profile.retriever, "health_ai_copilot.retrieval.hybrid.HybridRetriever+SentenceTransformerReranker", {"embedding": embedding_artifact.to_dict(), "reranker": reranker_artifact.to_dict(), "candidate_top_k": retriever.candidate_top_k}, artifact_revision=artifact_revision, learned_artifacts=(embedding_artifact, reranker_artifact)))


def _provider_from(context: ComponentBuildContext):
    return context.instances[(ComponentKind.PROVIDER, context.profile.provider)]


def _role_model(context: ComponentBuildContext, role: str) -> str:
    role_config = context.profile.config.get(role, {})
    if not isinstance(role_config, Mapping):
        raise TypeError(f"profile {role} config must be an object")
    return str(
        role_config.get("model")
        or context.environment.get("resolved_provider_model")
        or "fixture-provider-model"
    )


def _build_policy_m2(context: ComponentBuildContext) -> BuiltComponent:
    policy = OpenAICompatibleEvidencePolicy(
        provider_executor=_provider_from(context),
        model=_role_model(context, "policy"),
        knowledge_scope=None,
    )
    return BuiltComponent(
        policy,
        _identity(
            ComponentKind.POLICY,
            context.profile.policy or "evidence-policy-m2-v1",
            implementation_name(policy),
            {
                "model": policy.model_name,
                "temperature": policy.temperature,
                "contract": "evidence-policy-m2-v1",
                "scope_semantics": "none",
            },
        ),
    )


def _build_policy_m3(context: ComponentBuildContext) -> BuiltComponent:
    if context.knowledge_scope is None:
        raise ValueError("M3 evidence policy requires a knowledge scope")
    policy = OpenAICompatibleEvidencePolicy(
        provider_executor=_provider_from(context),
        model=_role_model(context, "policy"),
        knowledge_scope=context.knowledge_scope,
    )
    return BuiltComponent(
        policy,
        _identity(
            ComponentKind.POLICY,
            context.profile.policy or "evidence-policy-m3-v1",
            implementation_name(policy),
            {
                "model": policy.model_name,
                "temperature": policy.temperature,
                "contract": "evidence-policy-m3-v1",
                "scope_semantics": {
                    "scope_id": context.knowledge_scope.scope_id,
                    "scope_version": context.knowledge_scope.version,
                },
            },
        ),
    )


def _build_grounding(context: ComponentBuildContext) -> BuiltComponent:
    verifier = OpenAICompatibleGroundingVerifier(
        provider_executor=_provider_from(context),
        model=_role_model(context, "verifier"),
    )
    return BuiltComponent(
        verifier,
        _identity(
            ComponentKind.VERIFIER,
            context.profile.verifier or "grounding-v1",
            implementation_name(verifier),
            {
                "model": verifier.model_name,
                "temperature": verifier.temperature,
                "verifier_type": "grounding",
                "contract": "grounding-v1",
            },
        ),
    )


def _build_claim_support(context: ComponentBuildContext) -> BuiltComponent:
    verifier = OpenAICompatibleClaimSupportVerifier(
        provider_executor=_provider_from(context),
        model=_role_model(context, "verifier"),
    )
    return BuiltComponent(
        verifier,
        _identity(
            ComponentKind.VERIFIER,
            context.profile.verifier or "claim-support-v1",
            implementation_name(verifier),
            {
                "model": verifier.model_name,
                "temperature": verifier.temperature,
                "verifier_type": "claim_support",
                "contract": "claim-support-v1",
            },
        ),
    )


def _build_search_tool(context: ComponentBuildContext) -> BuiltComponent:
    retriever = context.instances[(ComponentKind.RETRIEVER, context.profile.retriever)]
    tool = SearchKnowledgeTool(retriever, knowledge_scope=context.knowledge_scope)
    return BuiltComponent(tool, _identity(ComponentKind.TOOL, "search-knowledge-v1", implementation_name(tool), {"top_k": 3, "scope_version": context.knowledge_scope.version if context.knowledge_scope else None}))


def _build_permission(context: ComponentBuildContext) -> BuiltComponent:
    cfg = _component_config(context.profile, ComponentKind.PERMISSION)
    mcp_cfg = context.profile.config.get("mcp", {})
    if not isinstance(mcp_cfg, Mapping):
        raise TypeError("profile mcp config must be an object")
    server_id = str(mcp_cfg.get("server_id", "m9-search-knowledge-server"))
    policy_id = str(cfg.get("policy_id", "permission-policy-v1"))
    policy = PermissionPolicy(
        {(server_id, "search_knowledge"): CapabilityPolicy(CapabilityClass.READ, decision="allow")},
        policy_id=policy_id,
    )
    guard = PermissionGuard(policy)
    return BuiltComponent(
        guard,
        _identity(
            ComponentKind.PERMISSION,
            "permission-policy-v1",
            "health_ai_copilot.mcp.permissions.PermissionGuard",
            {"policy_id": policy_id, "policy_hash": policy.config_hash},
        ),
    )


def _build_dev_sandbox(context: ComponentBuildContext) -> BuiltComponent:
    cfg = _component_config(context.profile, ComponentKind.SANDBOX)
    backend = NoSandboxDevBackend()
    return BuiltComponent(
        backend,
        _identity(
            ComponentKind.SANDBOX,
            "no-sandbox-dev-v1",
            "health_ai_copilot.mcp.sandbox.NoSandboxDevBackend",
            {
                "profile_id": cfg.get("profile_id", "no-sandbox-dev-v1"),
                "contained": False,
                "requires_real_enforcement": False,
            },
        ),
    )


def _build_mcp_client(context: ComponentBuildContext) -> BuiltComponent:
    cfg = dict(context.profile.config.get("mcp", {}))
    server_id = str(cfg.get("server_id", "m9-search-knowledge-server"))
    retriever = context.instances[(ComponentKind.RETRIEVER, context.profile.retriever)]
    server = build_search_knowledge_server(
        retriever,
        knowledge_scope=context.knowledge_scope,
        server_id=server_id,
    )
    config = McpServerConfig(
        server_id=server_id,
        transport=str(cfg.get("transport", "inproc_test")),
        protocol_version=MCP_PROTOCOL_VERSION,
        endpoint_identity=f"inproc:{server_id}",
        auth_mode=str(cfg.get("auth_mode", "unauthenticated_local")),
        sandbox_profile_id=str(cfg.get("sandbox_profile_id", "no-sandbox-dev-v1")),
        tool_namespace=str(cfg.get("tool_namespace", f"mcp::{server_id}")),
    )
    client = McpClient(config, server)
    catalog = client.catalog()
    return BuiltComponent(
        client,
        _identity(
            ComponentKind.MCP_CLIENT,
            "mcp-client-2026-07-28-v1",
            "health_ai_copilot.mcp.client.McpClient",
            {
                **config.to_safe_dict(),
                "sdk_version": "2.2.0",
                "catalog_hash": catalog.catalog_hash,
                "catalog_ttl_ms": catalog.ttl_ms,
                "catalog_cache_scope": catalog.cache_scope,
                "sandbox_profile_hash": config_hash(context.profile.config.get("sandbox", {})),
            },
        ),
    )


def _build_mcp_search_tool(context: ComponentBuildContext) -> BuiltComponent:
    client = context.instances[(ComponentKind.MCP_CLIENT, "mcp-client-2026-07-28-v1")]
    guard = context.instances[(ComponentKind.PERMISSION, "permission-policy-v1")]
    catalog = client.catalog()
    try:
        remote_spec = next(item for item in catalog.tools if item.name == "search_knowledge")
    except StopIteration as exc:
        raise RuntimeBuildError("MCP server did not expose search_knowledge") from exc
    adapter = McpToolAdapter(client, remote_spec, permission_guard=guard, exposed_name="search_knowledge")
    return BuiltComponent(
        adapter,
        _identity(
            ComponentKind.TOOL,
            "mcp-search-knowledge-v1",
            "health_ai_copilot.mcp.client.McpToolAdapter",
            {
                "remote_name": remote_spec.name,
                "namespace": client.config.tool_namespace,
                "catalog_hash": catalog.catalog_hash,
                "protocol_version": MCP_PROTOCOL_VERSION,
            },
        ),
    )


def _build_metadata_trace(context: ComponentBuildContext) -> BuiltComponent:
    return BuiltComponent(RuntimeTraceFactory(), _identity(ComponentKind.TRACE, context.profile.trace, "health_ai_copilot.runtime.trace.RunTrace", {"content_policy": "metadata_only"}))


def _build_public_trace(context: ComponentBuildContext) -> BuiltComponent:
    return BuiltComponent(RuntimeTraceFactory(TraceContentPolicy.PUBLIC_EVAL_CONTENT), _identity(ComponentKind.TRACE, context.profile.trace, "health_ai_copilot.runtime.trace.RunTrace", {"content_policy": "public_eval_content"}))


def _build_orchestrator(context: ComponentBuildContext) -> BuiltComponent:
    """Construct the fixed M8 lead/worker graph from the trusted registry."""

    provider = _provider_from(context)
    lead_model_name = str(
        context.environment.get("resolved_team_lead_model")
        or _role_model(context, "team_lead")
    )
    worker_model_name = str(
        context.environment.get("resolved_team_worker_model")
        or _role_model(context, "team_worker")
    )
    worker_models = {
        role: OpenAICompatibleAgentModel(
            provider_executor=provider,
            model=worker_model_name,
            output_mode=AgentOutputMode.M3_CLAIM_FIRST,
            provider_call_kind=ProviderCallKind.TEAM_WORKER,
            system_prompt=(
                f"{TEAM_ROLE_SYSTEM_CONTRACTS[role]} "
                "Work only on the assigned objective. Use only observed evidence or the "
                "single policy-approved search result. Never delegate or answer outside "
                "the claim-first JSON contract. Return claims with citation_ids, or abstain."
            ),
        )
        for role in TeamRole
    }
    cfg = _component_config(context.profile, ComponentKind.ORCHESTRATION)
    lead_contract = str(cfg.get("lead_contract", "team-lead-v2"))
    worker_contract = str(cfg.get("worker_contract", "m3-claim-first-v1"))
    lead_model = OpenAICompatibleTeamLeadModel(
        provider,
        lead_model_name,
        contract_version=lead_contract,
    )
    allowed_roles = tuple(
        TeamRole(role) for role in cfg.get("allowed_roles", [role.value for role in TeamRole])
    )
    topology = str(cfg.get("topology", TEAM_TOPOLOGY))
    scheduler = str(cfg.get("scheduler", TEAM_SCHEDULER))
    team_config = TeamBudgetConfig(
        max_lead_calls=int(cfg.get("max_lead_calls", 2)),
        max_workers_started=int(cfg.get("max_workers_started", 2)),
        max_tasks_created=int(cfg.get("max_tasks_created", 2)),
        max_delegation_rounds=int(cfg.get("max_delegation_rounds", 1)),
        max_worker_model_turns=int(cfg.get("max_worker_model_turns", 2)),
        max_worker_tool_calls=int(cfg.get("max_worker_tool_calls", 1)),
    )
    tool_instances = [
        context.instances[(ComponentKind.TOOL, tool_id)] for tool_id in context.profile.tool_set
    ]
    orchestrator = AgentTeamOrchestrator(
        lead_model,
        worker_models,
        ToolRegistry(tool_instances),
        config=team_config,
        evidence_policy=context.instances.get(
            (ComponentKind.POLICY, context.profile.policy)
        ),
        knowledge_scope=context.knowledge_scope,
        allowed_roles=allowed_roles,
        topology=topology,
        scheduler=scheduler,
    )
    identity_config = {
        "lead_model": lead_model_name,
        "worker_model": worker_model_name,
        "lead_contract": lead_contract,
        "worker_contract": worker_contract,
        "allowed_roles": list(allowed_roles),
        "role_contracts": {
            role.value: TEAM_ROLE_CONTRACT_IDS[role] for role in TeamRole
        },
        "topology": topology,
        "scheduler": scheduler,
        **team_config.to_dict(),
    }
    return BuiltComponent(
        orchestrator,
        _identity(
            ComponentKind.ORCHESTRATION,
            context.profile.orchestration or "agent-team-v1",
            "health_ai_copilot.team.AgentTeamOrchestrator",
            identity_config,
        ),
    )


def _discover_build_commit() -> str | None:
    """Discover the local source revision without network access or shell code."""

    explicit = os.getenv("HEALTH_COPILOT_BUILD_COMMIT", "").strip()
    if explicit:
        return explicit
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def _replay_evidence_hash(rows: Mapping[str, Sequence[Any]]) -> str:
    """Hash replay evidence content, not only its question keys."""

    payload = []
    for question, evidence_items in sorted(rows.items(), key=lambda item: item[0]):
        payload.append(
            {
                "question": question,
                "evidence": [
                    {
                        "source_id": item.source_id,
                        "title": item.title,
                        "excerpt": item.excerpt,
                        "source_url": item.source_url,
                        "score": item.score,
                    }
                    for item in evidence_items
                ],
            }
        )
    return canonical_json_sha256(payload)
