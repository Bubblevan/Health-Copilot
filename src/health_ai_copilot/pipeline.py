"""M0 safety-gated pipeline plus the bounded M1 recovery path."""

from collections.abc import Callable, Sequence
from typing import Protocol

from .agent.events import AgentEvent
from .agent.loop import AgentLoop, AgentLoopConfig, AgentRunResult
from .agent.model import AgentModel
from .agent.tools import ToolRegistry
from .contracts import AssistantResponse, Citation, Evidence, GenerationDraft, Route
from .generation.base import Generator
from .knowledge.scope import KnowledgeScope
from .policy.evidence import EvidencePolicy
from .runtime.context import RunContext
from .runtime.tools import ToolRunner
from .runtime.trace import TraceEventType
from .safety import route_question
from .tools.search_knowledge import SearchKnowledgeTool
from .verification.citations import verify_citations
from .verification.grounding import (
    ClaimSupportResult,
    ClaimSupportVerifier,
    ClaimVerdict,
    GroundingResult,
    GroundingVerifier,
    materialize_cited_evidence,
    validate_claim_support_result,
    validate_grounding_result,
)
from .verification.materialize import materialize_verified_claims, normalize_claims


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        ...


ABSTAIN_MESSAGE = "当前审核资料不足以支持可靠回答，因此本原型不对该问题作推断。"
ABSTAIN_MESSAGES = {
    "insufficient_evidence": (
        f"{ABSTAIN_MESSAGE}如涉及个人健康决策，请咨询有资质的医疗专业人员。"
    ),
    "generator_abstained": (
        f"{ABSTAIN_MESSAGE}如涉及个人健康决策，请咨询有资质的医疗专业人员。"
    ),
    "retrieval_error": (
        "当前检索服务无法正常工作，因此本原型暂时无法生成可靠回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "generation_error": (
        "当前回答生成服务无法正常工作，因此本原型暂时无法生成可靠回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "invalid_citation": (
        "当前回答未通过来源校验，因此本原型不会返回该回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "missing_citation": (
        "当前回答未通过来源校验，因此本原型不会返回该回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "abstain": (
        f"{ABSTAIN_MESSAGE}如涉及个人健康决策，请咨询有资质的医疗专业人员。"
    ),
    "max_model_turns": (
        "回答生成未在安全预算内完成，因此本原型不会强行补全回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "max_tool_calls": (
        "检索恢复次数已达到安全上限，因此本原型不会强行补全回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "model_error": (
        "当前回答生成服务无法正常工作，因此本原型暂时无法生成可靠回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "tool_error": (
        "当前检索恢复服务无法正常工作，因此本原型暂时无法生成可靠回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "agent_error": (
        "当前受限回答运行时无法正常工作，因此本原型暂时无法生成可靠回答。"
        "请稍后重试或咨询有资质的医疗专业人员。"
    ),
    "evidence_insufficient": (
        f"{ABSTAIN_MESSAGE}如涉及个人健康决策，请咨询有资质的医疗专业人员。"
    ),
    "evidence_conflicting": (
        f"{ABSTAIN_MESSAGE}如涉及个人健康决策，请咨询有资质的医疗专业人员。"
    ),
    "policy_error": "当前证据审核服务无法正常工作，因此本原型不会生成可靠回答。",
    "grounding_failed": "当前回答未通过逐项证据支持校验，因此本原型不会返回该回答。",
    "verifier_error": "当前回答校验服务无法正常工作，因此本原型不会返回该回答。",
    "claim_support_failed": "当前回答未通过逐项证据支持校验，因此本原型不会返回该回答。",
    "claim_support_verifier_error": "当前回答校验服务无法正常工作，因此本原型不会返回该回答。",
    "claim_materialization_error": "当前回答未通过结构化输出校验，因此本原型不会返回该回答。",
    "invalid_input": "当前问题输入无效，暂时无法生成可靠回答。",
}


def abstain_response(reason: str) -> AssistantResponse:
    return AssistantResponse(
        route=Route.ABSTAIN,
        message=ABSTAIN_MESSAGES.get(reason, ABSTAIN_MESSAGE),
        safety_reasons=[reason],
    )


class HealthCopilotPipeline:
    """Orchestrate M0 components, with an optional bounded M1 agent path."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator | None = None,
        top_k: int = 5,
        *,
        recovery_top_k: int = 3,
        agent_model: AgentModel | None = None,
        agent_config: AgentLoopConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        event_sink: Callable[[AgentEvent], None] | None = None,
        evidence_policy: EvidencePolicy | None = None,
        grounding_verifier: GroundingVerifier | None = None,
        knowledge_scope: KnowledgeScope | None = None,
        claim_support_verifier: ClaimSupportVerifier | None = None,
        runtime: RunContext | None = None,
        tool_runner: ToolRunner | None = None,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if recovery_top_k <= 0:
            raise ValueError("recovery_top_k must be greater than zero")
        if generator is not None and agent_model is not None:
            raise ValueError("generator and agent_model are mutually exclusive")
        if generator is None and agent_model is None:
            raise ValueError("either generator or agent_model must be provided")
        if knowledge_scope is None and (evidence_policy is None) != (grounding_verifier is None):
            raise ValueError("M2 requires both evidence_policy and grounding_verifier")
        if (
            evidence_policy is not None or grounding_verifier is not None
        ) and agent_model is None:
            raise ValueError("M2 requires an agent_model")
        m3_values = (knowledge_scope, claim_support_verifier)
        if any(value is not None for value in m3_values) and not all(
            value is not None for value in m3_values
        ):
            raise ValueError("M3 requires knowledge_scope and claim_support_verifier")
        if knowledge_scope is not None and evidence_policy is None:
            raise ValueError("M3 requires an evidence_policy")
        if knowledge_scope is not None and grounding_verifier is not None:
            raise ValueError("M3 uses claim support instead of the M2 grounding verifier")
        if knowledge_scope is not None and agent_model is None:
            raise ValueError("M3 requires an agent_model")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.agent_model = agent_model
        self.evidence_policy = evidence_policy
        self.grounding_verifier = grounding_verifier
        self.knowledge_scope = knowledge_scope
        self.claim_support_verifier = claim_support_verifier
        self.runtime = runtime
        self.last_grounding_result: GroundingResult | None = None
        self.last_claim_support_result: ClaimSupportResult | None = None
        self.last_harness_disposition: str | None = None
        self.last_agent_run: AgentRunResult | None = None
        self.agent_loop: AgentLoop | None = None
        if agent_model is not None:
            registry = tool_registry or ToolRegistry(
                [
                    SearchKnowledgeTool(
                        retriever,
                        top_k=recovery_top_k,
                        knowledge_scope=knowledge_scope,
                    )
                ]
            )
            self.agent_loop = AgentLoop(
                agent_model,
                registry,
                config=agent_config or AgentLoopConfig(),
                event_sink=event_sink,
                evidence_policy=evidence_policy,
                knowledge_scope=knowledge_scope,
                runtime=runtime,
                tool_runner=tool_runner,
            )

    def answer(self, question: str) -> AssistantResponse:
        active_runtime = self.runtime or RunContext.create("pipeline")
        self._bind_runtime(active_runtime)
        self.last_agent_run = None
        self.last_grounding_result = None
        self.last_claim_support_result = None
        self.last_harness_disposition = None
        if not isinstance(question, str) or not question.strip():
            return self._finalize(abstain_response("invalid_input"), active_runtime)

        terminal_response = route_question(question)
        if terminal_response is not None:
            return self._finalize(terminal_response, active_runtime)

        try:
            evidence = self.retriever.search(question, top_k=self.top_k)
        except Exception:  # noqa: BLE001 - pipeline must fail closed at component boundaries
            return self._finalize(abstain_response("retrieval_error"), active_runtime)
        if not evidence:
            return self._finalize(abstain_response("insufficient_evidence"), active_runtime)

        if self.agent_loop is not None:
            return self._finalize(self._answer_with_agent(question, evidence, active_runtime), active_runtime)

        try:
            # The M0 path remains intentionally replayable when no AgentModel is supplied.
            draft = self.generator.generate(question, evidence)  # type: ignore[union-attr]
            if not isinstance(draft, GenerationDraft):
                raise TypeError("generator returned an invalid draft")
        except Exception:  # noqa: BLE001 - pipeline must fail closed at component boundaries
            return self._finalize(abstain_response("generation_error"), active_runtime)
        return self._finalize(self._response_from_draft(draft, evidence), active_runtime)

    def _answer_with_agent(
        self,
        question: str,
        initial_evidence: Sequence[Evidence],
        runtime: RunContext,
    ) -> AssistantResponse:
        try:
            run = self.agent_loop.run(question, initial_evidence, runtime=runtime)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001 - runtime boundary must fail closed
            return abstain_response("agent_error")
        self.last_agent_run = run

        if run.draft is None:
            reason = run.stop_reason.value if run.stop_reason else "agent_error"
            return abstain_response(reason)
        if run.draft.abstain:
            return abstain_response("abstain")
        if self.claim_support_verifier is not None:
            return self._response_from_claim_first_final(run)
        if self.grounding_verifier is not None:
            return self._response_from_grounded_final(run)
        return self._response_from_draft(run.draft, run.observed_evidence)

    def _bind_runtime(self, runtime: RunContext) -> None:
        """Inject one control-plane context without changing frozen adapter APIs."""

        for component in (
            self.generator,
            self.agent_model,
            self.evidence_policy,
            self.grounding_verifier,
            self.claim_support_verifier,
        ):
            if component is not None and hasattr(component, "_runtime"):
                component._runtime = runtime  # type: ignore[attr-defined]

    def _finalize(self, response: AssistantResponse, runtime: RunContext) -> AssistantResponse:
        if runtime.trace is not None:
            disposition = self.last_harness_disposition or response.route.value
            runtime.trace.emit(
                TraceEventType.HARNESS_DISPOSITION,
                route=response.route.value,
                disposition=disposition,
                safety_reason_count=len(response.safety_reasons),
            )
            runtime.trace.close(status="complete")
        return response

    def _response_from_grounded_final(self, run: AgentRunResult) -> AssistantResponse:
        """M2 order: claim citation integrity, coverage/support, trusted citations."""
        draft = run.draft
        if draft is None or not run.claims:
            self.last_harness_disposition = "grounding_failed"
            return abstain_response("grounding_failed")
        claims = run.claims
        claim_ids = [source_id for claim in claims for source_id in claim.citation_ids]
        if not claim_ids or any(not claim.text.strip() or not claim.citation_ids for claim in claims):
            self.last_harness_disposition = "grounding_failed"
            return abstain_response("grounding_failed")
        integrity = verify_citations(
            GenerationDraft(draft.answer, claim_ids, abstain=False), run.observed_evidence
        )
        if not integrity.valid:
            self.last_harness_disposition = "invalid_citation"
            return abstain_response(integrity.reasons[0])
        run.state.verifier_calls_used += 1
        try:
            result = validate_grounding_result(
                self.grounding_verifier.verify(draft.answer, claims, run.observed_evidence),
                claims,
                run.observed_evidence,
            )
        except Exception:  # noqa: BLE001 - semantic verification fails closed
            self.last_harness_disposition = "verifier_error"
            return abstain_response("verifier_error")
        self.last_grounding_result = result
        if not result.coverage_ok or any(
            item.verdict != ClaimVerdict.SUPPORTED for item in result.claim_results
        ):
            self.last_harness_disposition = "grounding_failed"
            return abstain_response("grounding_failed")
        self.last_harness_disposition = "answer"
        evidence_by_id = {item.source_id: item for item in run.observed_evidence}
        return AssistantResponse(
            route=Route.ANSWER,
            message=draft.answer,
            citations=[
                Citation(source_id, evidence_by_id[source_id].title, evidence_by_id[source_id].excerpt, evidence_by_id[source_id].source_url)
                for source_id in integrity.citation_ids
            ],
        )

    def _response_from_claim_first_final(self, run: AgentRunResult) -> AssistantResponse:
        """M3 path: claims -> citation integrity -> support -> materialized response."""
        if not run.claims:
            self.last_harness_disposition = "claim_materialization_error"
            return abstain_response("claim_materialization_error")
        try:
            claims = normalize_claims(run.claims)
        except ValueError:
            self.last_harness_disposition = "claim_materialization_error"
            return abstain_response("claim_materialization_error")
        claim_ids = [source_id for claim in claims for source_id in claim.citation_ids]
        integrity = verify_citations(
            GenerationDraft(answer="", citation_ids=claim_ids, abstain=False),
            run.observed_evidence,
        )
        if not integrity.valid:
            self.last_harness_disposition = "invalid_citation"
            return abstain_response(integrity.reasons[0])
        run.state.verifier_calls_used += 1
        try:
            cited_evidence = materialize_cited_evidence(claims, run.observed_evidence)
            result = validate_claim_support_result(
                self.claim_support_verifier.verify(claims, cited_evidence),  # type: ignore[union-attr]
                claims,
                run.observed_evidence,
            )
        except Exception:  # noqa: BLE001 - semantic verification fails closed
            self.last_harness_disposition = "claim_support_verifier_error"
            return abstain_response("claim_support_verifier_error")
        self.last_claim_support_result = result
        if any(item.verdict != ClaimVerdict.SUPPORTED for item in result.claim_results):
            self.last_harness_disposition = "claim_support_failed"
            return abstain_response("claim_support_failed")
        try:
            message = materialize_verified_claims(claims)
        except ValueError:
            self.last_harness_disposition = "claim_materialization_error"
            return abstain_response("claim_materialization_error")
        evidence_by_id = {item.source_id: item for item in run.observed_evidence}
        self.last_harness_disposition = "answer"
        return AssistantResponse(
            route=Route.ANSWER,
            message=message,
            citations=[
                Citation(
                    source_id,
                    evidence_by_id[source_id].title,
                    evidence_by_id[source_id].excerpt,
                    evidence_by_id[source_id].source_url,
                )
                for source_id in integrity.citation_ids
            ],
        )

    @staticmethod
    def _response_from_draft(
        draft: GenerationDraft, evidence: Sequence[Evidence]
    ) -> AssistantResponse:
        if draft.abstain:
            return abstain_response("generator_abstained")

        verification = verify_citations(draft, evidence)
        if not verification.valid:
            return abstain_response(verification.reasons[0])

        evidence_by_id = {item.source_id: item for item in evidence}
        citations = [
            Citation(
                source_id=source_id,
                title=evidence_by_id[source_id].title,
                excerpt=evidence_by_id[source_id].excerpt,
                source_url=evidence_by_id[source_id].source_url,
            )
            for source_id in verification.citation_ids
        ]
        return AssistantResponse(
            route=Route.ANSWER,
            message=draft.answer,
            citations=citations,
        )
