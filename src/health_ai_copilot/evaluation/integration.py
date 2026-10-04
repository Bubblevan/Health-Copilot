"""Boundary adapters for owned longitudinal integration episodes."""

from __future__ import annotations

from dataclasses import dataclass

from ..harness.contracts import AnswerSchema, HarnessRequest, HarnessResponse
from ..research.integration.contracts import IntegrationEpisode


@dataclass(frozen=True)
class IntegrationRuntimeBinding:
    """Scoped environment identities providers must enforce outside the prompt."""

    episode_id: str
    patient_snapshot_id: str | None
    patient_record_types: tuple[str, ...]
    external_world_id: str | None
    external_world_version: str | None
    external_source_families: tuple[str, ...]


@dataclass(frozen=True)
class IntegrationHarnessResult:
    answer: str
    used_evidence_ids: tuple[str, ...]
    safety_flags: tuple[str, ...]
    trace_id: str
    provider_calls: int
    tool_calls: int
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


def integration_episode_to_request(episode: IntegrationEpisode) -> HarnessRequest:
    """Project runtime-visible episode fields only; evaluator planes are not accepted."""
    return HarnessRequest(
        request_id=f"integration-{episode.episode_id}",
        query=episode.query,
        answer_schema=AnswerSchema.FREE_TEXT,
        subject_id=episode.subject_id,
        as_of_time=episode.decision_time,
        benchmark_case_id=episode.episode_id,
    )


def integration_episode_binding(episode: IntegrationEpisode) -> IntegrationRuntimeBinding:
    patient_ref = episode.patient_state_ref
    evidence_ref = episode.external_world_ref
    return IntegrationRuntimeBinding(
        episode_id=episode.episode_id,
        patient_snapshot_id=patient_ref.snapshot_id if patient_ref else None,
        patient_record_types=patient_ref.record_types if patient_ref else (),
        external_world_id=evidence_ref.world_id if evidence_ref else None,
        external_world_version=evidence_ref.version if evidence_ref else None,
        external_source_families=evidence_ref.source_families if evidence_ref else (),
    )


def harness_response_to_integration_result(
    response: HarnessResponse,
) -> IntegrationHarnessResult:
    return IntegrationHarnessResult(
        answer=response.answer_text,
        used_evidence_ids=tuple(item.evidence_id for item in response.citations),
        safety_flags=response.safety_flags,
        trace_id=response.trace_id,
        provider_calls=response.provider_calls,
        tool_calls=response.tool_calls,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        latency_ms=response.latency_ms,
    )
