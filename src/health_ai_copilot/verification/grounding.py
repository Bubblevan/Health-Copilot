"""Claim-level coverage and evidence-relation contracts for M2."""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from ..config import ConfigurationError, load_openai_config
from ..contracts import Evidence


@dataclass(frozen=True)
class GroundedClaim:
    text: str
    citation_ids: tuple[str, ...]


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True)
class ClaimResult:
    claim_index: int
    verdict: ClaimVerdict
    supporting_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundingResult:
    coverage_ok: bool
    claim_results: tuple[ClaimResult, ...]


class GroundingVerifier(Protocol):
    def verify(self, answer: str, claims: Sequence[GroundedClaim], evidence: Sequence[Evidence]) -> GroundingResult:
        ...


def validate_grounding_result(result: GroundingResult, claims: Sequence[GroundedClaim], evidence: Sequence[Evidence]) -> GroundingResult:
    if not isinstance(result, GroundingResult):
        raise TypeError("verifier returned an invalid result")
    if {item.claim_index for item in result.claim_results} != set(range(len(claims))):
        raise ValueError("verifier did not return one result per claim")
    observed_ids = {item.source_id for item in evidence}
    if any(not set(item.supporting_source_ids).issubset(observed_ids) for item in result.claim_results):
        raise ValueError("verifier referenced an unobserved source")
    return result


class OpenAICompatibleGroundingVerifier:
    def __init__(self) -> None:
        try:
            config = load_openai_config(model_override=os.getenv("HEALTH_COPILOT_VERIFIER_MODEL"))
            from openai import OpenAI
        except (ConfigurationError, ImportError) as exc:
            raise RuntimeError("M2 grounding verifier is not configured") from exc
        kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs)
        self.model_name = config.model

    def verify(self, answer: str, claims: Sequence[GroundedClaim], evidence: Sequence[Evidence]) -> GroundingResult:
        payload = {"answer": answer, "claims": [{"text": item.text, "citation_ids": list(item.citation_ids)} for item in claims], "evidence": [{"source_id": item.source_id, "excerpt": item.excerpt} for item in evidence]}
        try:
            response = self._client.chat.completions.create(model=self.model_name, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "system", "content": "Verify claim coverage and evidence relation only. Return JSON coverage_ok and claim_results (claim_index, verdict, supporting_source_ids). Do not give advice."}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
            parsed = json.loads(response.choices[0].message.content)
            result = GroundingResult(bool(parsed["coverage_ok"]), tuple(ClaimResult(item["claim_index"], ClaimVerdict(item["verdict"]), tuple(item.get("supporting_source_ids", []))) for item in parsed["claim_results"]))
        except Exception as exc:
            raise RuntimeError("grounding verifier failed") from exc
        return validate_grounding_result(result, claims, evidence)
