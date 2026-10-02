"""Gold-blind TRAIN episode/corpus projection for RAG-E6A."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.rag_e6.split import (
    PARTITION_SIZES,
    U2F_MANIFEST_SHA256,
    U2F_ROOT_SHA256,
    U2F_TRAIN_EPISODES_SHA256,
    canonical_json_bytes,
    sha256_file,
    write_immutable_json,
)
from eval.u3r_rag_transfer import (
    LATENT_ID_PATTERN,
    OwnedExternalCorpusAdapter,
    U3RDocument,
    load_runtime_corpora,
    select_runtime_world_fields,
)

RUNTIME_EPISODE_FIELDS = frozenset({
    "episode_id", "environment_version", "source_provenance", "decision_time",
    "subject_id", "query", "observable_state", "patient_state_ref",
    "external_world_ref", "tool_surface_ref", "budget", "evaluator_ref",
})
_ISO_TIME = re.compile(r"^\d{4}-\d\d-\d\dT")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path.name}")
    return value


def _selected_top_level(line: str, allowed: frozenset[str]) -> dict[str, Any]:
    """Decode only ID fields to skip rows outside the selected subject partition."""
    from eval.u3r_rag_transfer import _selected_json_object

    selected, position = _selected_json_object(line, 0, allowed_fields=allowed)
    if line[position:].strip():
        raise ValueError("unexpected trailing content in runtime JSONL row")
    return selected


@dataclass(frozen=True)
class E6Episode:
    episode_id: str
    subject_id: str
    partition: str
    split: str
    query: str
    query_sha256: str
    decision_time: datetime
    available_source_families: tuple[str, ...]

    @classmethod
    def from_runtime_row(cls, row: Mapping[str, Any], *, partition: str) -> E6Episode:
        if set(row) != RUNTIME_EPISODE_FIELDS:
            raise ValueError("runtime episode does not match the exact U2-F runtime contract")
        episode_id = row.get("episode_id")
        subject_id = row.get("subject_id")
        query = row.get("query")
        if not all(isinstance(item, str) and item.strip() for item in (episode_id, subject_id, query)):
            raise ValueError("episode ID, subject ID, and query must be non-empty strings")
        if partition not in PARTITION_SIZES:
            raise ValueError("unknown E6A subject partition")
        timestamp = row.get("decision_time")
        if not isinstance(timestamp, str) or not _ISO_TIME.match(timestamp):
            raise ValueError("decision_time must be an ISO-8601 timestamp")
        decision_time = datetime.fromisoformat(timestamp)
        if decision_time.tzinfo is None or decision_time.utcoffset() is None:
            raise ValueError("decision_time must be timezone-aware")
        observable = row.get("observable_state")
        world_ref = row.get("external_world_ref")
        if not isinstance(observable, Mapping) or not isinstance(world_ref, Mapping):
            raise TypeError("runtime observable state and evidence-world reference are required")
        families = observable.get("available_external_source_families")
        world_families = world_ref.get("source_families")
        if (
            not isinstance(families, list)
            or any(not isinstance(item, str) or not item for item in families)
            or not isinstance(world_families, list)
            or not set(families).issubset(world_families)
        ):
            raise ValueError("runtime evidence-family scope is malformed")
        if not isinstance(world_ref.get("world_id"), str) or not world_ref["world_id"].strip():
            raise ValueError("runtime external evidence-world ID must be non-empty")
        return cls(
            episode_id=episode_id,
            subject_id=subject_id,
            partition=partition,
            split="TRAIN",
            query=query,
            query_sha256=hashlib.sha256(query.encode()).hexdigest(),
            decision_time=decision_time,
            available_source_families=tuple(sorted(set(families))),
        )


def load_partition_episodes(
    *, u2f_root: Path, split_manifest_path: Path, partition: str
) -> tuple[E6Episode, ...]:
    """Load only episodes whose subject is in BUILD or FROZEN_DEV."""
    if partition not in {"BUILD", "FROZEN_DEV"}:
        raise ValueError("E6A runtime may load BUILD or FROZEN_DEV only")
    if sha256_file(u2f_root / "manifest.json") != U2F_MANIFEST_SHA256:
        raise ValueError("frozen U2-F manifest SHA-256 mismatch")
    if sha256_file(u2f_root / "train/episodes.jsonl") != U2F_TRAIN_EPISODES_SHA256:
        raise ValueError("frozen U2-F TRAIN runtime-episodes SHA-256 mismatch")
    manifest = _read_json(split_manifest_path)
    if (
        manifest.get("schema_version") != "rag-e6a-subject-split-v1"
        or manifest.get("dataset_root_sha256") != U2F_ROOT_SHA256
        or manifest.get("source_u2f_manifest_sha256") != U2F_MANIFEST_SHA256
        or manifest.get("source_train_runtime_episodes_sha256") != U2F_TRAIN_EPISODES_SHA256
        or manifest.get("evaluator_truth_opened") is not False
    ):
        raise ValueError("E6A split manifest failed identity or gold-blind validation")
    selected_rows = manifest.get("subjects_by_partition", {}).get(partition)
    if not isinstance(selected_rows, list):
        raise TypeError("split manifest subject assignment must be a list")
    selected_subjects = {item.get("subject_id") for item in selected_rows if isinstance(item, dict)}
    if len(selected_subjects) != PARTITION_SIZES[partition]:
        raise ValueError("split manifest subject assignment has the wrong size")

    episodes_path = u2f_root / "train/episodes.jsonl"
    episodes: list[E6Episode] = []
    seen_ids: set[str] = set()
    with episodes_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            selected = _selected_top_level(line, frozenset({"episode_id", "subject_id"}))
            if selected.get("subject_id") not in selected_subjects:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"runtime episode row {line_number} is not an object")
            episode = E6Episode.from_runtime_row(row, partition=partition)
            if episode.subject_id != selected.get("subject_id"):
                raise ValueError("selective runtime decoder returned an inconsistent subject ID")
            if episode.episode_id in seen_ids:
                raise ValueError("duplicate E6A runtime episode ID")
            seen_ids.add(episode.episode_id)
            episodes.append(episode)
    expected_count = manifest.get("partition_episode_counts", {}).get(partition)
    if len(episodes) != expected_count or not episodes:
        raise ValueError("partition runtime episode count differs from the frozen split manifest")
    episodes.sort(key=lambda item: item.episode_id)
    return tuple(episodes)


def materialize_partition_corpus(
    *, u2f_root: Path, split_manifest_path: Path, partition: str, output_root: Path
) -> dict[str, Any]:
    """Project selected episode-visible documents; never reads truth or future outcomes."""
    episodes = load_partition_episodes(
        u2f_root=u2f_root, split_manifest_path=split_manifest_path, partition=partition
    )
    episode_by_id = {item.episode_id: item for item in episodes}
    corpora: dict[str, tuple[U3RDocument, ...]] = {}
    worlds_path = u2f_root / "latent_worlds.jsonl"
    for line in worlds_path.open(encoding="utf-8"):
        match = LATENT_ID_PATTERN.search(line)
        if match is None or match.group(1) not in episode_by_id:
            continue
        if match.group(1) in corpora:
            raise ValueError("duplicate visible-world row for an E6A runtime episode")
        episode = episode_by_id[match.group(1)]
        projection = select_runtime_world_fields(line)
        corpora[episode.episode_id] = OwnedExternalCorpusAdapter.documents_for_episode(
            episode, projection
        )
    if set(corpora) != set(episode_by_id):
        raise ValueError("selected runtime episode is missing its visible evidence world")

    rows = [{
        "episode_id": episode.episode_id,
        "partition": partition,
        "documents": [item.to_dict() for item in corpora[episode.episode_id]],
    } for episode in episodes]
    content = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    corpus_path = output_root / f"{partition.lower()}_runtime_corpus.jsonl"
    if corpus_path.exists():
        if corpus_path.read_bytes() != content:
            raise FileExistsError("existing E6A runtime corpus differs; refusing overwrite")
    else:
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with corpus_path.open("xb") as handle:
            handle.write(content)
            handle.flush()
    manifest = {
        "schema_version": "rag-e6a-runtime-corpus-v1",
        "dataset_id": "health-copilot-owned-longitudinal-v1",
        "dataset_root_sha256": U2F_ROOT_SHA256,
        "source_u2f_manifest_sha256": U2F_MANIFEST_SHA256,
        "source_train_runtime_episodes_sha256": U2F_TRAIN_EPISODES_SHA256,
        "partition": partition,
        "subject_count": PARTITION_SIZES[partition],
        "episode_count": len(episodes),
        "visible_document_count": sum(map(len, corpora.values())),
        "runtime_corpus_sha256": sha256_bytes(content),
        "evaluator_truth_opened": False,
        "future_train_outcomes_opened": False,
        "reserved_test_ood_materialized": False,
        "reserved_test_ood_opened": False,
    }
    write_immutable_json(output_root / f"{partition.lower()}_corpus_manifest.json", manifest)
    return manifest


def load_partition_corpus(path: Path, expected_episode_ids: set[str]) -> dict[str, tuple[U3RDocument, ...]]:
    result = load_runtime_corpora(path)
    if set(result) != expected_episode_ids:
        raise ValueError("runtime corpus IDs differ from the frozen partition episodes")
    return result
