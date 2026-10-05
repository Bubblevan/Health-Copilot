"""SingleReasoner, using only the provider and context supplied by Harness V1."""

from __future__ import annotations

import json
import re

from ..providers.model import ModelProvider, ModelRequest
from .base import ReasoningContext, ReasoningResult

_SINGLE_PROMPT = (
    "You are the single clinical reasoning strategy inside Health-Copilot Harness. "
    "Answer the current query using only the query and observations actually supplied by Harness. "
    "Do not invent patient history, evidence, citations, or source IDs. Follow answer_schema exactly. "
    "For single_choice return only one option label; for multi_select return only the selected "
    "option labels in alphabetical order. Do not reveal private reasoning."
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
