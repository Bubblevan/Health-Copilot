"""M0 safety-gated, evidence-grounded answer pipeline."""

from typing import Protocol

from .contracts import AssistantResponse, Citation, Evidence, GenerationDraft, Route
from .generation.base import Generator
from .safety import route_question
from .verification.citations import verify_citations


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
    "invalid_input": "当前问题输入无效，暂时无法生成可靠回答。",
}


def abstain_response(reason: str) -> AssistantResponse:
    return AssistantResponse(
        route=Route.ABSTAIN,
        message=ABSTAIN_MESSAGES.get(reason, ABSTAIN_MESSAGE),
        safety_reasons=[reason],
    )


class HealthCopilotPipeline:
    """Orchestrate safety, retrieval, generation and citation integrity."""

    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> AssistantResponse:
        if not isinstance(question, str) or not question.strip():
            return abstain_response("invalid_input")

        terminal_response = route_question(question)
        if terminal_response is not None:
            return terminal_response

        try:
            evidence = self.retriever.search(question, top_k=self.top_k)
        except Exception:  # noqa: BLE001 - pipeline must fail closed at component boundaries
            return abstain_response("retrieval_error")
        if not evidence:
            return abstain_response("insufficient_evidence")

        try:
            draft = self.generator.generate(question, evidence)
            if not isinstance(draft, GenerationDraft):
                raise TypeError("generator returned an invalid draft")
        except Exception:  # noqa: BLE001 - pipeline must fail closed at component boundaries
            return abstain_response("generation_error")
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
