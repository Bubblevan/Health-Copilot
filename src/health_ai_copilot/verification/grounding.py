"""Claim-level coverage and evidence-relation contracts for M2."""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ..config import ConfigurationError, load_openai_config
from ..contracts import Evidence
from ..runtime import (
    OpenAICompatibleProviderExecutor,
    ProviderCallKind,
    ProviderExecutor,
    ProviderFailure,
    ProviderRequest,
    RunContext,
)


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


@dataclass(frozen=True)
class ClaimSupportResult:
    """M3 claim-only semantic verification result; intentionally no coverage field."""

    claim_results: tuple[ClaimResult, ...]


class GroundingVerifier(Protocol):
    def verify(
        self,
        answer: str,
        claims: Sequence[GroundedClaim],
        evidence: Sequence[Evidence],
        *,
        runtime: RunContext | None = None,
    ) -> GroundingResult:
        ...


class ClaimSupportVerifier(Protocol):
    """Verify only each cited claim against observed evidence in the M3 path."""

    def verify(
        self,
        claims: Sequence[GroundedClaim],
        cited_evidence: Sequence[Sequence[Evidence]],
        *,
        runtime: RunContext | None = None,
    ) -> ClaimSupportResult:
        ...


def validate_grounding_result(result: GroundingResult, claims: Sequence[GroundedClaim], evidence: Sequence[Evidence]) -> GroundingResult:
    if not isinstance(result, GroundingResult):
        raise TypeError("verifier returned an invalid result")
    _validate_claim_results(result.claim_results, claims, evidence)
    return result


def materialize_cited_evidence(
    claims: Sequence[GroundedClaim], evidence: Sequence[Evidence]
) -> tuple[tuple[Evidence, ...], ...]:
    """Bind each M3 claim to only its cited observed evidence, in citation order."""
    evidence_by_id = {item.source_id: item for item in evidence}
    bound: list[tuple[Evidence, ...]] = []
    for claim in claims:
        try:
            bound.append(tuple(evidence_by_id[source_id] for source_id in claim.citation_ids))
        except KeyError as exc:
            raise ValueError("claim cites an unobserved source") from exc
    return tuple(bound)


def _validate_claim_results(
    claim_results: Sequence[ClaimResult],
    claims: Sequence[GroundedClaim],
    evidence: Sequence[Evidence],
) -> None:
    if len(claim_results) != len(claims):
        raise ValueError("verifier did not return one result per claim")
    indices = [item.claim_index for item in claim_results]
    if len(set(indices)) != len(indices):
        raise ValueError("verifier returned duplicate claim indexes")
    if set(indices) != set(range(len(claims))):
        raise ValueError("verifier did not return the expected claim indexes")
    observed_ids = {item.source_id for item in evidence}
    for item in claim_results:
        claim_citations = set(claims[item.claim_index].citation_ids)
        if not set(item.supporting_source_ids).issubset(observed_ids):
            raise ValueError("verifier referenced an unobserved source")
        if not set(item.supporting_source_ids).issubset(claim_citations):
            raise ValueError("verifier used a source not cited by the claim")
        if item.verdict == ClaimVerdict.SUPPORTED and not item.supporting_source_ids:
            raise ValueError("supported claim needs a supporting source")


def validate_claim_support_result(
    result: ClaimSupportResult,
    claims: Sequence[GroundedClaim],
    evidence: Sequence[Evidence],
) -> ClaimSupportResult:
    if not isinstance(result, ClaimSupportResult):
        raise TypeError("claim support verifier returned an invalid result")
    _validate_claim_results(result.claim_results, claims, evidence)
    return result


class OpenAICompatibleGroundingVerifier:
    def __init__(self, *, provider_executor: ProviderExecutor | None = None, model: str | None = None, runtime: RunContext | None = None) -> None:
        if provider_executor is None:
            try:
                config = load_openai_config(model_override=os.getenv("HEALTH_COPILOT_VERIFIER_MODEL"))
                provider_executor = OpenAICompatibleProviderExecutor(config)
            except (ConfigurationError, ProviderFailure) as exc:
                raise RuntimeError("M2 grounding verifier is not configured") from exc
            model, self.base_url = config.model, config.base_url
        else:
            self.base_url = None
        self._provider_executor, self.model_name = (
            provider_executor,
            model or "injected-provider-model",
        )
        self.temperature = 0
        self.timeout_seconds = 30.0
        self.max_retries = 0

    def verify(
        self,
        answer: str,
        claims: Sequence[GroundedClaim],
        evidence: Sequence[Evidence],
        *,
        runtime: RunContext | None = None,
    ) -> GroundingResult:
        payload = {"answer": answer, "claims": [{"text": item.text, "citation_ids": list(item.citation_ids)} for item in claims], "evidence": [{"source_id": item.source_id, "excerpt": item.excerpt} for item in evidence]}
        try:
            response = self._provider_executor.execute(
                ProviderRequest.create(
                    kind=ProviderCallKind.GROUNDING_VERIFIER,
                    model=self.model_name,
                    temperature=0,
                    response_format={"type": "json_object"},
                    timeout_seconds=self.timeout_seconds,
                    messages=(
                        {
                            "role": "system",
                            "content": "Verify claim coverage and evidence relation only. For each claim, judge support only against the sources listed in that claim's citation_ids; supporting_source_ids must be a non-empty subset of that claim's citation_ids for a supported verdict. Return JSON coverage_ok and claim_results (claim_index, verdict, supporting_source_ids). Do not give advice.",
                        },
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ),
                ),
                runtime or RunContext.create("grounding_verifier"),
            )
            parsed = json.loads(response.content)
            result = GroundingResult(bool(parsed["coverage_ok"]), tuple(ClaimResult(item["claim_index"], ClaimVerdict(item["verdict"]), tuple(item.get("supporting_source_ids", []))) for item in parsed["claim_results"]))
        except Exception as exc:
            raise RuntimeError("grounding verifier failed") from exc
        return validate_grounding_result(result, claims, evidence)


class OpenAICompatibleClaimSupportVerifier:
    """OpenAI-compatible M3 verifier for claim support only, without coverage judging."""

    def __init__(self, *, provider_executor: ProviderExecutor | None = None, model: str | None = None, runtime: RunContext | None = None) -> None:
        if provider_executor is None:
            try:
                config = load_openai_config(model_override=os.getenv("HEALTH_COPILOT_VERIFIER_MODEL"))
                provider_executor = OpenAICompatibleProviderExecutor(config)
            except (ConfigurationError, ProviderFailure) as exc:
                raise RuntimeError("M3 claim support verifier is not configured") from exc
            model, self.base_url = config.model, config.base_url
        else:
            self.base_url = None
        self._provider_executor, self.model_name = (
            provider_executor,
            model or "injected-provider-model",
        )
        self.temperature = 0
        self.timeout_seconds = 30.0
        self.max_retries = 0

    def verify(
        self,
        claims: Sequence[GroundedClaim],
        cited_evidence: Sequence[Sequence[Evidence]],
        *,
        runtime: RunContext | None = None,
    ) -> ClaimSupportResult:
        if len(cited_evidence) != len(claims):
            raise ValueError("cited evidence must contain one entry per claim")
        payload = {
            "claims": [
                {
                    "claim_index": index,
                    "text": claim.text,
                    "cited_evidence": [
                        {"source_id": item.source_id, "excerpt": item.excerpt}
                        for item in evidence_for_claim
                    ],
                }
                for index, (claim, evidence_for_claim) in enumerate(
                    zip(claims, cited_evidence, strict=True)
                )
            ],
        }
        prompt = (
            "Verify only claim support. Each claim includes only its cited evidence; "
            "do not infer support from any other source. Return JSON claim_results "
            "(claim_index, verdict supported|unsupported|contradicted, "
            "supporting_source_ids). A supported verdict needs a non-empty subset of "
            "the source_ids in that claim's cited_evidence. Do not judge answer coverage "
            "or give advice."
        )
        try:
            response = self._provider_executor.execute(
                ProviderRequest.create(
                    kind=ProviderCallKind.CLAIM_SUPPORT_VERIFIER,
                    model=self.model_name,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    timeout_seconds=self.timeout_seconds,
                    messages=(
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ),
                ),
                runtime or RunContext.create("claim_support_verifier"),
            )
            parsed = json.loads(response.content)
            result = ClaimSupportResult(
                tuple(
                    ClaimResult(
                        item["claim_index"],
                        ClaimVerdict(item["verdict"]),
                        tuple(item.get("supporting_source_ids", [])),
                    )
                    for item in parsed["claim_results"]
                )
            )
        except Exception as exc:
            raise RuntimeError("claim support verifier failed") from exc
        flattened_evidence = tuple(
            item for evidence_for_claim in cited_evidence for item in evidence_for_claim
        )
        return validate_claim_support_result(result, claims, flattened_evidence)
