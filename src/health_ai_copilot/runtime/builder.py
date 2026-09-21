"""Declarative profile construction for the trusted in-process runtime."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..agent.loop import AgentLoopConfig
from ..agent.messages import MemoryContextMessage
from ..agent.model import AgentOutputMode, OpenAICompatibleAgentModel
from ..agent.session import AgentSession
from ..agent.tools import ToolRegistry
from ..config import load_openai_config
from ..contracts import AssistantResponse
from ..generation.openai_compatible import OpenAICompatibleGenerator
from ..knowledge.scope import KnowledgeScope
from ..mcp.client import MCP_PROTOCOL_VERSION, McpClient, McpServerConfig, McpToolAdapter
from ..mcp.permissions import (
    CapabilityClass,
    CapabilityPolicy,
    PermissionGuard,
    PermissionPolicy,
)
from ..mcp.sandbox import (
    BubblewrapSandboxBackend,
    NoSandboxDevBackend,
    SandboxFilesystemPolicy,
    SandboxNetworkPolicy,
    SandboxPolicy,
    SandboxProfile,
    WslBubblewrapSandboxBackend,
)
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
from .context_manager import (
    ContextBudget,
    ContextManager,
    ContextPlan,
    DeterministicTokenEstimator,
    StructuredCompactorV1,
)
from .memory import (
    InMemoryMemoryStore,
    MemoryPolicy,
    MemoryQuery,
    MemorySnapshotIdentity,
    SQLiteMemoryStore,
)
from .profile import RuntimeProfile
from .provider import OpenAICompatibleProviderExecutor, ProviderCallKind, ProviderExecutor
from .registry import ComponentBuildContext, ComponentRegistry
from .session import InMemorySessionStore, SessionEventType, SessionRecord, SQLiteSessionStore
from .trace import RunTrace, TraceContentPolicy, TraceEventType, canonical_json_sha256


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
    state_dir: Path | None = None
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
    session_store: Any | None = None
    context_manager: ContextManager | None = None
    memory_store: Any | None = None
    memory_policy: MemoryPolicy | None = None

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

    def answer_in_session(
        self,
        session_id: str,
        question: str,
        *,
        intent: Any | None = None,
        current_values: Mapping[str, Any] | None = None,
        retrieval_query: str | None = None,
        budget: RunBudgetConfig | None = None,
        trace_path: Path | None = None,
    ) -> SessionAnswer:
        """Run one experimental M10 turn with explicit persistence boundaries."""

        if self.session_store is None or self.context_manager is None:
            raise RuntimeBuildError("profile does not select M10 session/context components")
        session = self.session_store.resume_session(session_id)
        records = []
        if self.memory_store is not None:
            records = self.memory_store.query(
                MemoryQuery(
                    scope_id=session_id,
                    text=question,
                    intent=intent,
                    top_k=8,
                    current_values=current_values or {},
                )
            )
        effective_query = retrieval_query or _memory_expanded_query(question, records)
        plan = self.context_manager.build_plan(
            session_id=session.session_id,
            session_revision=session.current_revision,
            current_user=question,
            current_evidence=(),
            memory_records=records,
            history=self.session_store.list_events(session.session_id),
            retrieval_query=effective_query,
        )
        snapshot = (
            self.memory_store.snapshot(session.session_id, session_revision=session.current_revision)
            if self.memory_store is not None
            else None
        )
        runtime = self.create_run_context(budget=budget, trace_path=trace_path)
        if runtime.trace is not None:
            runtime.trace.emit(
                TraceEventType.SESSION_LOADED,
                session_id=session.session_id,
                session_revision=session.current_revision,
            )
            runtime.trace.emit(
                TraceEventType.MEMORY_QUERY,
                scope_id_sha256=canonical_json_sha256(session.session_id),
                query_sha256=canonical_json_sha256(question),
                candidate_count=len(records),
            )
            if snapshot is not None:
                runtime.trace.emit(
                    TraceEventType.MEMORY_SELECTED,
                    scope_id_sha256=snapshot.scope_id_sha256,
                    snapshot_sha256=snapshot.snapshot_sha256,
                    selected_count=len(records),
                )
            runtime.trace.emit(
                TraceEventType.CONTEXT_PLAN,
                session_revision=plan.session_revision,
                context_plan_hash=plan.plan_hash,
                selected_memory_count=len(plan.selected_memory_ids),
                selected_history_count=len(plan.selected_event_ids),
                compaction_count=plan.compaction_count,
                estimated_tokens=plan.estimated_tokens,
            )
            if plan.compaction_count:
                runtime.trace.emit(
                    TraceEventType.CONTEXT_COMPACTED,
                    compaction_count=plan.compaction_count,
                    compacted_event_count=len(plan.compacted_event_ids),
                )
        ephemeral = AgentSession()
        if records:
            ephemeral.append(MemoryContextMessage(tuple(records)))
        response = self.pipeline(runtime=runtime).answer(
            question,
            agent_session=ephemeral,
            retrieval_query=effective_query,
        )
        self.session_store.append(
            session.session_id,
            SessionEventType.USER_INPUT,
            {"question": question},
            expected_revision=session.current_revision,
            run_id=runtime.identity.run_id,
        )
        latest = self.session_store.resume_session(session.session_id)
        self.session_store.append(
            session.session_id,
            SessionEventType.ASSISTANT_OUTPUT,
            {"route": response.route.value, "message": response.message},
            expected_revision=latest.current_revision,
            run_id=runtime.identity.run_id,
        )
        latest = self.session_store.resume_session(session.session_id)
        if runtime.trace is not None:
            runtime.trace.emit(
                TraceEventType.SESSION_COMMITTED,
                session_id=latest.session_id,
                session_revision=latest.current_revision,
            )
        return SessionAnswer(response, latest, plan, snapshot)

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


@dataclass(frozen=True)
class SessionAnswer:
    """Response plus non-sensitive session/context provenance."""

    response: AssistantResponse
    session: SessionRecord
    context_plan: ContextPlan
    memory_snapshot: MemorySnapshotIdentity | None

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def session_revision(self) -> int:
        return self.session.current_revision

    def to_metadata(self) -> dict[str, Any]:
        return {
            "session_id": self.session.session_id,
            "session_revision": self.session.current_revision,
            "context_plan_hash": self.context_plan.plan_hash,
            "memory_snapshot_hash": self.memory_snapshot.snapshot_sha256 if self.memory_snapshot else None,
        }


def _memory_expanded_query(question: str, records: Sequence[Any]) -> str:
    parts = [question]
    for record in records:
        if isinstance(record.value, str) and record.value.strip():
            parts.append(record.value)
        if record.key not in question:
            parts.append(record.key)
    return " ".join(parts)


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
            mcp_client="mcp-client-2026-07-28-v1",
            permission="permission-policy-v1",
            sandbox="no-sandbox-dev-v1",
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
        "m10-context-bm25-v1": RuntimeProfile(
            "m10-context-bm25-v1",
            provider,
            "bm25-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m10_context",
            session_store="sqlite-session-v1",
            context_manager="context-manager-v1",
            config={
                "session_store": {"backend": "sqlite", "schema_version": "v1"},
                "context_manager": {
                    "budget": {
                        "max_estimated_input_tokens": 4096,
                        "reserved_system_tokens": 256,
                        "reserved_current_turn_tokens": 512,
                        "max_memory_tokens": 0,
                        "max_history_tokens": 2048,
                        "max_tool_observation_tokens": 1024,
                    },
                    "estimator": "chars4-cjk1-v1",
                    "compactor": "structured-compactor-v1",
                    "history_window": 24,
                },
            },
        ),
        "m10-memory-bm25-v1": RuntimeProfile(
            "m10-memory-bm25-v1",
            provider,
            "bm25-v1",
            policy="evidence-policy-m3-v1",
            verifier="claim-support-v1",
            tool_set=search,
            mode="m10_memory",
            session_store="sqlite-session-v1",
            context_manager="context-manager-v1",
            memory_store="sqlite-memory-v1",
            memory_policy="memory-policy-v1",
            config={
                "session_store": {"backend": "sqlite", "schema_version": "v1"},
                "context_manager": {
                    "budget": {
                        "max_estimated_input_tokens": 4096,
                        "reserved_system_tokens": 256,
                        "reserved_current_turn_tokens": 512,
                        "max_memory_tokens": 1024,
                        "max_history_tokens": 2048,
                        "max_tool_observation_tokens": 1024,
                    },
                    "estimator": "chars4-cjk1-v1",
                    "compactor": "structured-compactor-v1",
                    "history_window": 24,
                },
                "memory_store": {"backend": "sqlite", "schema_version": "v1"},
                "memory_policy": {
                    "version": "m10-policy-v1",
                    "sensitive_health_default": "deny",
                    "allowed_sources": ["user_explicit", "trusted_application", "session_derived"],
                },
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
    registry.register(ComponentKind.SANDBOX, "bubblewrap-sandbox-v1", _build_bubblewrap_sandbox, implementation="health_ai_copilot.mcp.sandbox.BubblewrapSandboxBackend")
    registry.register(ComponentKind.SANDBOX, "bubblewrap-sandbox-wsl-v1", _build_wsl_bubblewrap_sandbox, implementation="health_ai_copilot.mcp.sandbox.WslBubblewrapSandboxBackend")
    registry.register(ComponentKind.SESSION_STORE, "in-memory-session-v1", _build_in_memory_session, implementation="health_ai_copilot.runtime.session.InMemorySessionStore")
    registry.register(ComponentKind.SESSION_STORE, "sqlite-session-v1", _build_sqlite_session, implementation="health_ai_copilot.runtime.session.SQLiteSessionStore")
    registry.register(ComponentKind.CONTEXT_MANAGER, "context-manager-v1", _build_context_manager, implementation="health_ai_copilot.runtime.context_manager.ContextManager")
    registry.register(ComponentKind.MEMORY_STORE, "in-memory-memory-v1", _build_in_memory_memory, implementation="health_ai_copilot.runtime.memory.InMemoryMemoryStore")
    registry.register(ComponentKind.MEMORY_STORE, "sqlite-memory-v1", _build_sqlite_memory, implementation="health_ai_copilot.runtime.memory.SQLiteMemoryStore")
    registry.register(ComponentKind.MEMORY_POLICY, "memory-policy-v1", _build_memory_policy, implementation="health_ai_copilot.runtime.memory.MemoryPolicy")
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

        for kind, component_id in (
            (ComponentKind.PERMISSION, profile.permission),
            (ComponentKind.SANDBOX, profile.sandbox),
            (ComponentKind.MCP_CLIENT, profile.mcp_client),
            (ComponentKind.SESSION_STORE, profile.session_store),
            (ComponentKind.MEMORY_POLICY, profile.memory_policy),
            (ComponentKind.MEMORY_STORE, profile.memory_store),
            (ComponentKind.CONTEXT_MANAGER, profile.context_manager),
        ):
            if component_id is not None:
                construct(kind, component_id)
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
        elif profile.mode in {"m1", "m2", "m3", "m9_mcp", "m10_context", "m10_memory"}:
            output_mode = {
                "m1": AgentOutputMode.M1,
                "m2": AgentOutputMode.M2_GROUNDED,
                "m3": AgentOutputMode.M3_CLAIM_FIRST,
                "m9_mcp": AgentOutputMode.M3_CLAIM_FIRST,
                "m10_context": AgentOutputMode.M3_CLAIM_FIRST,
                "m10_memory": AgentOutputMode.M3_CLAIM_FIRST,
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
            verifier if profile.mode in {"m3", "m8_workflow", "m8_team", "m9_mcp", "m10_context", "m10_memory"} else None,
            tool_registry,
            trace_factory,
            manifest,
            team_lead_model_name if profile.mode == "m8_team" else agent_model_name,
            knowledge_scope,
            run_context_config or self.environment.run_budget,
            orchestrator,
            agent_config,
            instances.get((ComponentKind.SESSION_STORE, profile.session_store))
            if profile.session_store
            else None,
            instances.get((ComponentKind.CONTEXT_MANAGER, profile.context_manager))
            if profile.context_manager
            else None,
            instances.get((ComponentKind.MEMORY_STORE, profile.memory_store))
            if profile.memory_store
            else None,
            instances.get((ComponentKind.MEMORY_POLICY, profile.memory_policy))
            if profile.memory_policy
            else None,
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
            "state_dir": Path(self.environment.state_dir) if self.environment.state_dir else None,
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
        if profile.mode not in {"m0", "m1", "m2", "m3", "m8_workflow", "m8_team", "m9_mcp", "m10_context", "m10_memory"}:
            raise RuntimeBuildError(f"unsupported runtime mode: {profile.mode}")
        if profile.mode in {"m3", "m8_workflow", "m8_team", "m9_mcp", "m10_context", "m10_memory"} and scope is None:
            raise RuntimeBuildError("claim-first profile requires a KnowledgeScope")
        if profile.mode == "m2" and (profile.policy is None or profile.verifier is None):
            raise RuntimeBuildError("m2 profile requires policy and verifier components")
        if profile.mode in {"m3", "m8_workflow", "m8_team", "m9_mcp", "m10_context", "m10_memory"} and (profile.policy is None or profile.verifier is None):
            raise RuntimeBuildError("claim-first profile requires policy and verifier components")
        if profile.mode in {"m1", "m2", "m3"} and not profile.tool_set:
            raise RuntimeBuildError(f"{profile.mode} profile requires an explicit tool set")
        if profile.mode == "m8_team" and (profile.orchestration != "agent-team-v1" or not profile.tool_set):
            raise RuntimeBuildError("m8_team profile requires agent-team-v1 and an explicit tool set")
        if profile.mode == "m9_mcp":
            if profile.tool_set != ("mcp-search-knowledge-v1",):
                raise RuntimeBuildError("m9_mcp profile requires the MCP search tool")
            missing = [
                name
                for name in ("mcp_client", "permission", "sandbox")
                if getattr(profile, name) is None
            ]
            if missing:
                raise RuntimeBuildError(
                    "m9_mcp profile requires declarative component selections: "
                    + ", ".join(missing)
                )
        if profile.mode in {"m10_context", "m10_memory"}:
            required = ["session_store", "context_manager"]
            if profile.mode == "m10_memory":
                required.extend(["memory_store", "memory_policy"])
            missing = [name for name in required if getattr(profile, name) is None]
            if missing:
                raise RuntimeBuildError(
                    f"{profile.mode} requires declarative component selections: " + ", ".join(missing)
                )


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


def _state_path(context: ComponentBuildContext, filename: str) -> Path:
    configured = context.environment.get("state_dir")
    state_dir = Path(configured) if configured else Path(".health-copilot-state")
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / filename


def _safe_state_identity(path: Path) -> dict[str, str]:
    """Bind storage identity without leaking a developer's absolute path."""

    return {
        "backend": "sqlite",
        "schema_version": "v1",
        "state_path_sha256": config_hash(str(path.resolve())),
    }


def _build_in_memory_session(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.session_store
    if component_id is None:
        raise RuntimeBuildError("session store component is not selected")
    instance = InMemorySessionStore()
    return BuiltComponent(instance, _identity(ComponentKind.SESSION_STORE, component_id, implementation_name(instance), {"backend": "in_memory", "schema_version": "v1"}))


def _build_sqlite_session(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.session_store
    if component_id is None:
        raise RuntimeBuildError("session store component is not selected")
    path = _state_path(context, "session.sqlite")
    instance = SQLiteSessionStore(path)
    return BuiltComponent(instance, _identity(ComponentKind.SESSION_STORE, component_id, implementation_name(instance), _safe_state_identity(path)))


def _build_memory_policy(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.memory_policy
    if component_id is None:
        raise RuntimeBuildError("memory policy component is not selected")
    cfg = _component_config(context.profile, ComponentKind.MEMORY_POLICY)
    instance = MemoryPolicy()
    identity_config = {
        "version": str(cfg.get("version", instance.version)),
        "allowed_kinds": sorted(item.value for item in instance.allowed_kinds),
        "allowed_sources": sorted(item.value for item in instance.allowed_sources),
        "sensitivity_policy": str(cfg.get("sensitive_health_default", "deny")),
    }
    return BuiltComponent(instance, _identity(ComponentKind.MEMORY_POLICY, component_id, implementation_name(instance), identity_config))


def _build_in_memory_memory(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.memory_store
    if component_id is None:
        raise RuntimeBuildError("memory store component is not selected")
    policy = context.instances.get((ComponentKind.MEMORY_POLICY, context.profile.memory_policy))
    instance = InMemoryMemoryStore(policy=policy)
    return BuiltComponent(instance, _identity(ComponentKind.MEMORY_STORE, component_id, implementation_name(instance), {"backend": "in_memory", "schema_version": "v1"}))


def _build_sqlite_memory(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.memory_store
    if component_id is None:
        raise RuntimeBuildError("memory store component is not selected")
    policy = context.instances.get((ComponentKind.MEMORY_POLICY, context.profile.memory_policy))
    path = _state_path(context, "memory.sqlite")
    instance = SQLiteMemoryStore(path, policy=policy)
    return BuiltComponent(instance, _identity(ComponentKind.MEMORY_STORE, component_id, implementation_name(instance), _safe_state_identity(path)))


def _build_context_manager(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.context_manager
    if component_id is None:
        raise RuntimeBuildError("context manager component is not selected")
    cfg = _component_config(context.profile, ComponentKind.CONTEXT_MANAGER)
    budget_cfg = cfg.get("budget", {})
    if not isinstance(budget_cfg, Mapping):
        raise TypeError("context manager budget config must be an object")
    budget = ContextBudget(**dict(budget_cfg))
    instance = ContextManager(
        budget=budget,
        estimator=DeterministicTokenEstimator(),
        compactor=StructuredCompactorV1(),
        history_window=int(cfg.get("history_window", 24)),
    )
    return BuiltComponent(
        instance,
        _identity(
            ComponentKind.CONTEXT_MANAGER,
            component_id,
            implementation_name(instance),
            {
                "budget": budget.to_dict(),
                "estimator": instance.estimator.version,
                "compactor": instance.compactor.version,
                "history_window": instance.history_window,
            },
        ),
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
    component_id = context.profile.permission
    if component_id is None:
        raise RuntimeBuildError("permission component is not selected by the profile")
    cfg = _component_config(context.profile, ComponentKind.PERMISSION)
    mcp_cfg = context.profile.config.get("mcp", {})
    if not isinstance(mcp_cfg, Mapping):
        raise TypeError("profile mcp config must be an object")
    server_id = str(mcp_cfg.get("server_id", "m9-search-knowledge-server"))
    policy_id = str(cfg.get("policy_id", component_id))
    policy = PermissionPolicy(
        {(server_id, "search_knowledge"): CapabilityPolicy(CapabilityClass.READ, decision="allow")},
        policy_id=policy_id,
    )
    guard = PermissionGuard(policy)
    return BuiltComponent(
        guard,
        _identity(
            ComponentKind.PERMISSION,
            component_id,
            "health_ai_copilot.mcp.permissions.PermissionGuard",
            {"policy_id": policy_id, "policy_hash": policy.config_hash},
        ),
    )


def _build_dev_sandbox(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.sandbox
    if component_id is None:
        raise RuntimeBuildError("sandbox component is not selected by the profile")
    cfg = _component_config(context.profile, ComponentKind.SANDBOX)
    backend = NoSandboxDevBackend()
    return BuiltComponent(
        backend,
        _identity(
            ComponentKind.SANDBOX,
            component_id,
            "health_ai_copilot.mcp.sandbox.NoSandboxDevBackend",
            {
                "profile_id": cfg.get("profile_id", component_id),
                "contained": False,
                "requires_real_enforcement": False,
            },
        ),
    )


def _build_bubblewrap_sandbox(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.sandbox
    if component_id is None:
        raise RuntimeBuildError("sandbox component is not selected by the profile")
    return BuiltComponent(
        BubblewrapSandboxBackend(),
        _identity(
            ComponentKind.SANDBOX,
            component_id,
            "health_ai_copilot.mcp.sandbox.BubblewrapSandboxBackend",
            _component_config(context.profile, ComponentKind.SANDBOX),
        ),
    )


def _build_wsl_bubblewrap_sandbox(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.sandbox
    if component_id is None:
        raise RuntimeBuildError("sandbox component is not selected by the profile")
    cfg = _component_config(context.profile, ComponentKind.SANDBOX)
    return BuiltComponent(
        WslBubblewrapSandboxBackend(
            distribution=str(cfg.get("distribution", "Ubuntu-24.04"))
        ),
        _identity(
            ComponentKind.SANDBOX,
            component_id,
            "health_ai_copilot.mcp.sandbox.WslBubblewrapSandboxBackend",
            cfg,
        ),
    )


def _sandbox_profile(context: ComponentBuildContext) -> SandboxProfile | None:
    component_id = context.profile.sandbox
    if component_id is None:
        return None
    cfg = _component_config(context.profile, ComponentKind.SANDBOX)
    filesystem = cfg.get("filesystem", {})
    if not isinstance(filesystem, Mapping):
        raise TypeError("profile sandbox filesystem config must be an object")
    read_roots = tuple(Path(item) for item in filesystem.get("read_roots", ()))
    write_roots = tuple(Path(item) for item in filesystem.get("write_roots", ()))
    network = SandboxNetworkPolicy(cfg.get("network", SandboxNetworkPolicy.DENY_ALL))
    network_origins = tuple(str(item) for item in cfg.get("network_origins", ()))
    return SandboxProfile(
        str(cfg.get("profile_id", component_id)),
        SandboxPolicy(
            SandboxFilesystemPolicy(read_roots=read_roots, write_roots=write_roots),
            network=network,
            network_origins=network_origins,
        ),
        backend_id=component_id,
        requires_real_enforcement=bool(cfg.get("requires_real_enforcement", False)),
    )


def _build_mcp_client(context: ComponentBuildContext) -> BuiltComponent:
    component_id = context.profile.mcp_client
    if component_id is None:
        raise RuntimeBuildError("MCP client component is not selected by the profile")
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
        sandbox_profile_id=(
            str(cfg["sandbox_profile_id"])
            if "sandbox_profile_id" in cfg and cfg["sandbox_profile_id"] is not None
            else None
        ),
        tool_namespace=str(cfg.get("tool_namespace", f"mcp::{server_id}")),
    )
    sandbox_backend = None
    sandbox_profile = None
    if config.transport == "stdio" and config.sandbox_profile_id is not None:
        sandbox_id = context.profile.sandbox
        if sandbox_id is None:
            raise RuntimeBuildError("stdio MCP requires a selected sandbox component")
        try:
            sandbox_backend = context.instances[(ComponentKind.SANDBOX, sandbox_id)]
        except KeyError as exc:
            raise RuntimeBuildError(
                f"stdio MCP requires sandbox dependency {sandbox_id}"
            ) from exc
        sandbox_profile = _sandbox_profile(context)
        if sandbox_profile is None:
            raise RuntimeBuildError("stdio MCP requires a sandbox profile")
    client = McpClient(
        config,
        server,
        sandbox_backend=sandbox_backend,
        sandbox_profile=sandbox_profile,
    )
    catalog = client.catalog()
    return BuiltComponent(
        client,
        _identity(
            ComponentKind.MCP_CLIENT,
            component_id,
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
    client_id = context.profile.mcp_client
    permission_id = context.profile.permission
    if client_id is None or permission_id is None:
        raise RuntimeBuildError("MCP tool requires MCP client and permission selections")
    try:
        client = context.instances[(ComponentKind.MCP_CLIENT, client_id)]
        guard = context.instances[(ComponentKind.PERMISSION, permission_id)]
    except KeyError as exc:
        raise RuntimeBuildError(
            "MCP tool dependencies were not constructed"
        ) from exc
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
