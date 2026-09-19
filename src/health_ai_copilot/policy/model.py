"""OpenAI-compatible implementation of the narrow EvidencePolicy protocol."""

import json
import os
from collections.abc import Sequence
from typing import Any

from ..config import ConfigurationError, load_openai_config
from ..contracts import Evidence
from .evidence import EvidenceAssessment, EvidenceDecision, validate_assessment

SYSTEM_PROMPT = """You are the M2 EvidencePolicy. Assess the full tuple
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


class OpenAICompatibleEvidencePolicy:
    def __init__(self) -> None:
        try:
            config = load_openai_config(model_override=os.getenv("HEALTH_COPILOT_POLICY_MODEL"))
            from openai import OpenAI
        except (ConfigurationError, ImportError) as exc:
            raise RuntimeError("M2 evidence policy is not configured") from exc
        kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs, timeout=30.0, max_retries=0)
        self.model_name = config.model
        self.base_url = config.base_url
        self.temperature = 0
        self.timeout_seconds = 30.0
        self.max_retries = 0

    def assess(self, question: str, evidence: Sequence[Evidence], proposed_query: str) -> EvidenceAssessment:
        payload = {"question": question, "proposed_query": proposed_query, "evidence": [{"source_id": item.source_id, "excerpt": item.excerpt} for item in evidence]}
        try:
            response = self._client.chat.completions.create(model=self.model_name, temperature=self.temperature, response_format={"type": "json_object"}, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            parsed = json.loads(response.choices[0].message.content)
            assessment = EvidenceAssessment(EvidenceDecision(parsed["decision"]), tuple(parsed.get("supporting_source_ids", [])), tuple(parsed.get("reason_codes", [])))
        except Exception as exc:
            raise RuntimeError("evidence policy failed") from exc
        return validate_assessment(assessment, evidence)
