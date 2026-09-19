"""OpenAI-compatible implementation of the narrow EvidencePolicy protocol."""

import json
import os
from collections.abc import Sequence
from typing import Any

from ..config import ConfigurationError, load_openai_config
from ..contracts import Evidence
from .evidence import EvidenceAssessment, EvidenceDecision, validate_assessment


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
        self._client = OpenAI(**kwargs)
        self.model_name = config.model

    def assess(self, question: str, evidence: Sequence[Evidence], proposed_query: str) -> EvidenceAssessment:
        payload = {"question": question, "proposed_query": proposed_query, "evidence": [{"source_id": item.source_id, "excerpt": item.excerpt} for item in evidence]}
        try:
            response = self._client.chat.completions.create(model=self.model_name, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "system", "content": "Classify provided evidence only. Return JSON decision (sufficient, recoverable, insufficient, conflicting), supporting_source_ids, reason_codes. Do not add facts."}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            parsed = json.loads(response.choices[0].message.content)
            assessment = EvidenceAssessment(EvidenceDecision(parsed["decision"]), tuple(parsed.get("supporting_source_ids", [])), tuple(parsed.get("reason_codes", [])))
        except Exception as exc:
            raise RuntimeError("evidence policy failed") from exc
        return validate_assessment(assessment, evidence)
