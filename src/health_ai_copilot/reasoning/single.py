"""SingleReasoner, using only the provider and context supplied by Harness V1."""

from __future__ import annotations

import json
import re

from ..providers.model import ModelProvider, ModelRequest
from .base import ReasoningContext, ReasoningResult

_SINGLE_PROMPT = (
    "你是 Health-Copilot 的 Strong Single Agent。你可使用全部患者状态、时间线、"
    "医学检索、外部证据、风险与可回答性能力。只能依据当前问题和 Harness 实际观察；"
    "不得虚构病史、证据、引文或来源 ID。若合成研究问题中要求精确 token，逐字保留。"
    "不要给出诊断或药物剂量调整。只返回面向用户的直接答案，不展示推理过程。"
)


def _strip_nonanswer(text: str) -> str:
    answer = text.strip()
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.IGNORECASE).strip()
    return answer


class SingleReasoner:
    def __init__(self, provider: ModelProvider, *, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        observations: list[dict[str, str]] = []
        if context.patient_state:
            observations.append({
                "skill": "HarnessPatientState",
                "status": "observed",
                "output": "\n".join(context.patient_state),
            })
        if context.external_evidence:
            evidence = "\n\n".join(
                f"[{item.evidence_id}] {item.source}: {item.excerpt}"
                for item in context.external_evidence
            )
            observations.append({
                "skill": "HarnessExternalEvidence",
                "status": "observed",
                "output": evidence,
            })
        user_payload = json.dumps({
            "query": context.query,
            "conversation_context": list(context.conversation_context),
            "answer_schema": context.answer_schema.value,
            "harness_observations": observations,
            "hospital_knowledge_search": "disabled; top-level retrieval is owned by Harness",
        }, ensure_ascii=False, separators=(",", ":"))
        reply = await self.provider.complete(ModelRequest(
            model=self.model,
            messages=(
                {"role": "system", "content": _SINGLE_PROMPT},
                {"role": "user", "content": user_payload},
            ),
            max_output_tokens=512,
        ))
        answer = _strip_nonanswer(reply.content)
        cited = tuple(dict.fromkeys(match.group(1) for match in re.finditer(
            r"\[([A-Za-z0-9_.:-]+)\]", answer
        )))
        return ReasoningResult(
            answer_text=answer,
            citation_ids=cited,
            provider_calls=1,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            reasoning_events=({"event": "reasoning_single_complete"},),
        )
