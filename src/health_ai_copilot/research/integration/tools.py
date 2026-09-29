"""Shared deterministic tool registry for Single and Team executions."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from .contracts import IntegrationEpisode, stable_hash
from .evidence_world import ExternalEvidenceWorld
from .state import PatientStateRecord, PatientStateStore


@dataclass(frozen=True)
class ToolInvocation:
    """Architecture-neutral inputs to one deterministic tool call."""

    tool_id: str
    episode: IntegrationEpisode
    patient_state_store: PatientStateStore | None = None
    external_evidence_world: ExternalEvidenceWorld | None = None

    @property
    def resource_versions(self) -> tuple[tuple[str, str], ...]:
        episode = self.episode
        versions = [("environment", episode.environment_version)]
        if episode.tool_surface_ref is not None:
            versions.append(("tool_surface", episode.tool_surface_ref.version))
        if episode.patient_state_ref is not None:
            versions.append(("patient_snapshot", episode.patient_state_ref.snapshot_id))
        if episode.external_world_ref is not None:
            versions.append(("external_world", f"{episode.external_world_ref.world_id}@{episode.external_world_ref.version}"))
        return tuple(sorted(versions))

    @property
    def input_hash(self) -> str:
        return stable_hash({
            "tool_id": self.tool_id,
            "query": self.episode.query,
            "decision_time": self.episode.decision_time.isoformat(),
            "resource_versions": self.resource_versions,
            "patient_scope": self.episode.patient_state_ref.to_dict()
            if self.episode.patient_state_ref else None,
            "external_scope": self.episode.external_world_ref.to_dict()
            if self.episode.external_world_ref else None,
        })


@dataclass(frozen=True)
class ToolObservation:
    tool_id: str
    output: str
    resource_ids: tuple[str, ...]
    resource_versions: tuple[tuple[str, str], ...]
    input_hash: str
    implementation_hash: str
    output_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "output": self.output,
            "resource_ids": list(self.resource_ids),
            "resource_versions": dict(self.resource_versions),
            "input_hash": self.input_hash,
            "implementation_hash": self.implementation_hash,
            "output_hash": self.output_hash,
        }


ToolImplementation = Callable[[ToolInvocation], tuple[str, tuple[str, ...]]]


@dataclass(frozen=True)
class DeterministicTool:
    tool_id: str
    version: str
    implementation: ToolImplementation
    implementation_hash: str

    def invoke(self, request: ToolInvocation) -> ToolObservation:
        if request.tool_id != self.tool_id:
            raise ValueError("tool invocation id differs from registered implementation")
        output, resource_ids = self.implementation(request)
        normalized_ids = tuple(resource_ids)
        output_hash = stable_hash({
            "tool_id": self.tool_id,
            "version": self.version,
            "output": output,
            "resource_ids": normalized_ids,
            "resource_versions": request.resource_versions,
            "input_hash": request.input_hash,
            "implementation_hash": self.implementation_hash,
        })
        return ToolObservation(
            self.tool_id,
            output,
            normalized_ids,
            request.resource_versions,
            request.input_hash,
            self.implementation_hash,
            output_hash,
        )


class DeterministicToolRegistry:
    """One immutable-per-run implementation per tool id."""

    def __init__(self, tools: Iterable[DeterministicTool] = ()) -> None:
        rows = tuple(tools)
        by_id = {item.tool_id: item for item in rows}
        if len(by_id) != len(rows):
            raise ValueError("a deterministic tool id may be registered exactly once")
        self._tools = by_id

    @classmethod
    def from_implementations(
        cls,
        implementations: Iterable[tuple[str, str, ToolImplementation]],
    ) -> DeterministicToolRegistry:
        tools = []
        for tool_id, version, implementation in implementations:
            try:
                source = inspect.getsource(implementation)
            except (OSError, TypeError) as exc:
                raise ValueError(f"tool implementation source unavailable for {tool_id}") from exc
            fingerprint = sha256(
                f"{tool_id}\n{version}\n{source}".encode()
            ).hexdigest()
            tools.append(DeterministicTool(tool_id, version, implementation, fingerprint))
        return cls(tools)

    @property
    def tool_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def registered_tools(self) -> tuple[DeterministicTool, ...]:
        return tuple(self._tools[key] for key in sorted(self._tools))

    @property
    def implementation_hashes(self) -> tuple[tuple[str, str], ...]:
        return tuple((key, self._tools[key].implementation_hash) for key in sorted(self._tools))

    def contains(self, tool_id: str) -> bool:
        return tool_id in self._tools

    def implementation_hash(self, tool_id: str) -> str | None:
        tool = self._tools.get(tool_id)
        return tool.implementation_hash if tool else None

    def missing(self, tool_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(set(tool_ids) - set(self._tools)))

    def invoke(self, request: ToolInvocation) -> ToolObservation:
        try:
            tool = self._tools[request.tool_id]
        except KeyError as exc:
            raise ValueError(f"NO_DETERMINISTIC_IMPLEMENTATION:{request.tool_id}") from exc
        return tool.invoke(request)


def _query_matches(query: str, terms: tuple[str, ...]) -> bool:
    if not terms:
        return True
    text = query.casefold()
    return any(term in text for term in terms)


def _query_partition(request: ToolInvocation) -> tuple[str, tuple[str, ...]]:
    match = re.fullmatch(r"query_part_([a-z])", request.tool_id)
    if not match:
        return "", ()
    label = match.group(1)
    found = re.search(rf"part_{label}\s*=\s*([^;]+)", request.episode.query, flags=re.IGNORECASE)
    return (found.group(1).strip(), ()) if found else ("", ())


def _memory_read(request: ToolInvocation) -> tuple[str, tuple[str, ...]]:
    episode = request.episode
    ref = episode.patient_state_ref
    if request.patient_state_store is None or ref is None:
        raise ValueError("PATIENT_STATE_RESOURCE_MISSING")
    rows: tuple[PatientStateRecord, ...] = request.patient_state_store.snapshot(
        ref.subject_id, episode.decision_time
    )
    allowed_types = set(ref.record_types)
    rows = tuple(row for row in rows if row.record_type.value in allowed_types)
    rows = tuple(row for row in rows if _query_matches(episode.query, row.retrieval_terms))
    return " ".join(row.content for row in rows), tuple(row.record_id for row in rows)


def _external_retrieval(request: ToolInvocation) -> tuple[str, tuple[str, ...]]:
    episode = request.episode
    ref = episode.external_world_ref
    if request.external_evidence_world is None or ref is None:
        raise ValueError("EXTERNAL_EVIDENCE_RESOURCE_MISSING")
    if (request.external_evidence_world.world_id, request.external_evidence_world.version) != (
        ref.world_id, ref.version
    ):
        raise ValueError("EXTERNAL_WORLD_VERSION_MISMATCH")
    rows = request.external_evidence_world.retrieve(
        episode.query, ref.source_families, as_of_time=episode.decision_time
    )
    return " ".join(row.content for row in rows), tuple(row.source_id for row in rows)


def default_tool_registry() -> DeterministicToolRegistry:
    """Build the pinned synthetic implementations used by U1.1 fixtures."""
    return DeterministicToolRegistry.from_implementations((
        ("external_retrieval", "u1.1-external-retrieval-v1", _external_retrieval),
        ("memory_read", "u1.1-memory-read-v1", _memory_read),
        ("query_part_a", "u1.1-query-partition-v1", _query_partition),
        ("query_part_b", "u1.1-query-partition-v1", _query_partition),
    ))
