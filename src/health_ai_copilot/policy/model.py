"""OpenAI-compatible implementation of the narrow EvidencePolicy protocol."""

import json
import os
from collections.abc import Sequence

from ..config import ConfigurationError, load_openai_config
from ..contracts import Evidence
from ..knowledge.scope import KnowledgeScope
from ..runtime import (
    OpenAICompatibleProviderExecutor,
    ProviderCallKind,
    ProviderExecutor,
    ProviderFailure,
    ProviderRequest,
    RunContext,
)
from .evidence import EvidenceAssessment, EvidenceDecision, validate_assessment

M2_SYSTEM_PROMPT = """You are the M2 EvidencePolicy. Assess the full tuple
(question, current evidence, proposed_query), not the provided evidence alone.

Choose exactly one decision:
- sufficient: current observed evidence is enough to answer the original
  question. The proposed search is unnecessary; do not authorize it.
- recoverable: current evidence is insufficient, but proposed_query is
  semantically aligned with the question and is a reasonable in-domain
  retrieval recovery. This is the only decision that authorizes one
  search_knowledge call.
- insufficient: current evidence is insufficient and proposed_query cannot
  reasonably recover the missing evidence within this knowledge pack, or the
  question/query is out of domain. Deny the tool and abstain.
- conflicting: current evidence materially conflicts on a fact needed for the
  question. Deny the tool and abstain.

Return JSON with decision (sufficient, recoverable, insufficient, conflicting),
supporting_source_ids, and reason_codes. supporting_source_ids must be source
IDs from current evidence only. Every reason_codes item must be exactly one of:
direct_support, related_but_incomplete, out_of_scope, missing_required_evidence,
conflicting_sources, policy_error. Do not add facts or other fields."""

M3_SYSTEM_PROMPT = """You are the M3 capability-aware EvidencePolicy. Assess
the full tuple (question, current evidence, proposed_query, knowledge scope).
The query may be linguistically reasonable but is RECOVERABLE only when the
reviewed closed corpus capability can actually retrieve the missing evidence.

Choose exactly one decision:
- sufficient: current observed evidence answers the original question; deny search.
- recoverable: evidence is insufficient, proposed_query is aligned, and at least
  one reviewed scope topic covers the recovery. Include that topic ID.
- insufficient: current evidence is insufficient and no reviewed scope topic
  covers what the query needs, including medically related but uncovered topics.
  Deny search and abstain.
- conflicting: current evidence materially conflicts on a needed fact; deny search.

Return JSON with decision, supporting_source_ids, reason_codes, and
matched_topic_ids. supporting_source_ids must be from current evidence.
matched_topic_ids must be IDs from the supplied scope; recoverable requires at
least one. reason_codes must use only: direct_support, related_but_incomplete,
out_of_scope, missing_required_evidence, conflicting_sources, policy_error.
Do not add facts or other fields."""


class OpenAICompatibleEvidencePolicy:
    def __init__(
        self,
        *,
        knowledge_scope: KnowledgeScope | None = None,
        provider_executor: ProviderExecutor | None = None,
        model: str | None = None,
        runtime: RunContext | None = None,
    ) -> None:
        if provider_executor is None:
            try:
                config = load_openai_config(model_override=os.getenv("HEALTH_COPILOT_POLICY_MODEL"))
                provider_executor = OpenAICompatibleProviderExecutor(config)
            except (ConfigurationError, ProviderFailure) as exc:
                raise RuntimeError("M2 evidence policy is not configured") from exc
            model = config.model
            self.base_url = config.base_url
        else:
            self.base_url = None
        self._provider_executor = provider_executor
        self.model_name = model or "injected-provider-model"
        self._runtime = runtime
        self.temperature = 0
        self.timeout_seconds = 30.0
        self.max_retries = 0
        self.knowledge_scope = knowledge_scope

    def assess(self, question: str, evidence: Sequence[Evidence], proposed_query: str) -> EvidenceAssessment:
        payload: dict[str, object] = {
            "question": question,
            "proposed_query": proposed_query,
            "evidence": [
                {"source_id": item.source_id, "excerpt": item.excerpt}
                for item in evidence
            ],
        }
        if self.knowledge_scope is not None:
            payload["knowledge_scope"] = {
                "scope_id": self.knowledge_scope.scope_id,
                "version": self.knowledge_scope.version,
                "domain": self.knowledge_scope.domain,
                "topics": [
                    {"id": topic.id, "description": topic.description}
                    for topic in self.knowledge_scope.topics
                ],
            }
        try:
            response = self._provider_executor.execute(
                ProviderRequest.create(
                    kind=ProviderCallKind.POLICY,
                    model=self.model_name,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout_seconds=self.timeout_seconds,
                    messages=(
                    {
                        "role": "system",
                        "content": (
                            M3_SYSTEM_PROMPT
                            if self.knowledge_scope is not None
                            else M2_SYSTEM_PROMPT
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ),
                ),
                self._runtime or RunContext.create("policy"),
            )
            parsed = json.loads(response.content)
            assessment = EvidenceAssessment(
                EvidenceDecision(parsed["decision"]),
                tuple(parsed.get("supporting_source_ids", [])),
                tuple(parsed.get("reason_codes", [])),
                tuple(parsed.get("matched_topic_ids", [])),
            )
        except Exception as exc:
            raise RuntimeError("evidence policy failed") from exc
        return validate_assessment(assessment, evidence, self.knowledge_scope)
