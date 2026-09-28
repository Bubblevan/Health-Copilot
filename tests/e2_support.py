from __future__ import annotations

from collections.abc import Sequence

from health_ai_copilot.agent.messages import (
    AgentMessage,
    FinalTurn,
    ToolCall,
    ToolCallTurn,
    ToolResultMessage,
    UserMessage,
)
from health_ai_copilot.agent.tools import ToolSpec
from health_ai_copilot.capabilities import (
    CapabilityEligibility,
    CapabilitySourceCatalog,
    E2WorkerRole,
    SourceMetadata,
    WorkerCapabilitySpec,
)
from health_ai_copilot.contracts import Evidence
from health_ai_copilot.runtime.budget import RunBudgetConfig
from health_ai_copilot.runtime.components import ComponentManifest
from health_ai_copilot.runtime.context import RunContext
from health_ai_copilot.runtime.provider import ProviderUsage
from health_ai_copilot.runtime.trace import RunTrace
from health_ai_copilot.verification.grounding import GroundedClaim


def evidence(source_id: str, *, title: str | None = None) -> Evidence:
    return Evidence(
        source_id=source_id,
        title=title or f"title:{source_id}",
        excerpt=f"excerpt:{source_id}",
        source_url=f"https://example.test/{source_id}",
        score=1.0,
    )


class FixtureRetriever:
    def __init__(self, rows: Sequence[Evidence]) -> None:
        self.rows = tuple(rows)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        self.calls.append((query, top_k))
        return list(self.rows[:top_k])


def capability(
    role: E2WorkerRole,
    family: str,
    domain: str,
    *,
    capability_id: str | None = None,
    families: tuple[str, ...] | None = None,
    domains: tuple[str, ...] | None = None,
    max_model_turns: int = 2,
    max_tool_calls: int = 1,
    eligibility: CapabilityEligibility = CapabilityEligibility.ELIGIBLE,
    eligibility_reason: str | None = None,
) -> WorkerCapabilitySpec:
    return WorkerCapabilitySpec(
        capability_id=capability_id or f"synthetic-{role.value}-v1",
        role=role,
        allowed_source_families=families or (family,),
        allowed_capability_domains=domains or (domain,),
        tool_ids=("search_knowledge",),
        retriever_profile_id="synthetic-bm25-v1",
        authority_scope=f"Synthetic-only authority for {family}/{domain}.",
        objective_contract_id=f"synthetic-{role.value}-objective-v1",
        max_model_turns=max_model_turns,
        max_tool_calls=max_tool_calls,
        eligibility=eligibility,
        eligibility_reason=eligibility_reason,
    )


def source_catalog(*rows: SourceMetadata) -> CapabilitySourceCatalog:
    return CapabilitySourceCatalog(tuple(rows))


def bound_context(
    capabilities: Sequence[WorkerCapabilitySpec],
    source_catalog: CapabilitySourceCatalog,
    *,
    budget: RunBudgetConfig | None = None,
    trace: RunTrace | None = None,
) -> tuple[RunContext, ComponentManifest]:
    from health_ai_copilot.capabilities import e2_component_manifest

    manifest = e2_component_manifest(
        ComponentManifest("e2-synthetic-v1", ()),
        capabilities,
        source_catalog=source_catalog,
    )
    runtime = RunContext.create(
        "e2_synthetic",
        budget=budget,
        trace=trace,
        profile_id=manifest.profile_id,
        component_manifest_hash=manifest.manifest_hash,
        config_hash=manifest.manifest_hash,
        code_commit="synthetic-fixture",
    )
    return runtime, manifest


class ScriptedAgentModel:
    """Offline model that accounts provider effects and can return one retrieval."""

    def __init__(
        self,
        *,
        citation_source: str | None = None,
        query: str | None = None,
        fail: bool = False,
        forge_source: str | None = None,
    ) -> None:
        self.citation_source = citation_source
        self.query = query
        self.fail = fail
        self.forge_source = forge_source
        self.calls = 0
        self.transcripts: list[tuple[AgentMessage, ...]] = []
        self.tool_specs: list[tuple[ToolSpec, ...]] = []
        self.runtime_rows = []

    def respond(self, messages, tools, *, runtime=None):
        self.calls += 1
        self.transcripts.append(tuple(messages))
        self.tool_specs.append(tuple(tools))
        if runtime is not None:
            self.runtime_rows.append((runtime.identity, runtime.budget, dict(runtime.metadata)))
            runtime.budget.guard_provider()
            runtime.budget.record_usage(ProviderUsage(5, 3, 8))
        if self.fail:
            raise RuntimeError("synthetic provider failure")
        if self.query is not None and self.calls == 1:
            return ToolCallTurn(
                [ToolCall("synthetic-search", "search_knowledge", {"query": self.query})]
            )
        source_id = self.forge_source or self.citation_source or self._observed_source(messages)
        claims = (
            (GroundedClaim("synthetic supported claim", (source_id,)),)
            if source_id
            else ()
        )
        return FinalTurn(
            "synthetic answer" if claims else "",
            [source_id] if source_id else [],
            abstain=not bool(claims),
            claims=claims,
        )

    @staticmethod
    def _observed_source(messages: Sequence[AgentMessage]) -> str | None:
        for message in reversed(messages):
            if isinstance(message, ToolResultMessage) and message.result.observed_evidence:
                return message.result.observed_evidence[0].source_id
            if isinstance(message, UserMessage) and message.evidence:
                return message.evidence[0].source_id
        return None
