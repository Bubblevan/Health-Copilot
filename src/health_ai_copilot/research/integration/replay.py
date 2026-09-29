"""Stable replay identities and semantic result comparison."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import CapabilityAction
from .contracts import IntegrationEpisode, stable_hash
from .evidence_world import ExternalEvidenceWorld
from .executor import DeterministicIntegrationExecutor, ExecutionResources
from .state import PatientStateStore


@dataclass(frozen=True)
class ReplayIdentity:
    episode_hash: str
    observable_state_hash: str
    patient_snapshot_hash: str
    external_world_hash: str
    capability_manifest_hash: str
    action_hash: str
    executor_version: str
    evaluator_version: str

    def to_dict(self) -> dict[str, str]:
        return {"episode_hash": self.episode_hash,
                "observable_state_hash": self.observable_state_hash,
                "patient_snapshot_hash": self.patient_snapshot_hash,
                "external_world_hash": self.external_world_hash,
                "capability_manifest_hash": self.capability_manifest_hash,
                "action_hash": self.action_hash,
                "executor_version": self.executor_version,
                "evaluator_version": self.evaluator_version}


def replay_identity(
    episode: IntegrationEpisode,
    action: CapabilityAction,
    resources: ExecutionResources,
    executor: DeterministicIntegrationExecutor,
    evaluator_version: str,
) -> ReplayIdentity:
    store: PatientStateStore | None = resources.patient_state_store
    world: ExternalEvidenceWorld | None = resources.external_evidence_world
    ref = episode.patient_state_ref
    patient_hash = stable_hash([])
    if store is not None and ref is not None:
        rows = store.snapshot(ref.subject_id, episode.decision_time)
        rows = tuple(row for row in rows if row.record_type.value in set(ref.record_types))
        patient_hash = stable_hash([row.to_dict() for row in rows])
    world_ref = episode.external_world_ref
    external_hash = stable_hash(None)
    if world is not None and world_ref is not None:
        external_hash = world.snapshot_hash(episode.decision_time, world_ref.source_families)
    return ReplayIdentity(
        episode_hash=episode.episode_hash,
        observable_state_hash=stable_hash(episode.observable_state.to_dict()),
        patient_snapshot_hash=patient_hash,
        external_world_hash=external_hash,
        capability_manifest_hash=stable_hash(episode.tool_surface_ref.to_dict()
                                             if episode.tool_surface_ref else None),
        action_hash=stable_hash(action.to_dict()),
        executor_version=executor.version,
        evaluator_version=evaluator_version,
    )


def semantic_execution_hash(result: object) -> str:
    """Hash all deterministic execution values, excluding wall time by design."""
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    else:
        payload = result
    return stable_hash(payload)
