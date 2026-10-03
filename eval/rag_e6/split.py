"""Deterministic subject-level split for RAG-E6A, built from runtime IDs only."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPLIT_VERSION = "rag-e6a-subject-split-v1"
U2F_MANIFEST_SHA256 = "34e6d1a8123ee63eea220c1a1b9b0b9b5f2e3349aeeec05a4504a508d4ca814e"
U2F_TRAIN_EPISODES_SHA256 = "11f19dba164c6832a6cfd807e1784247d7af4ecd7ea423b0d3562677233f0f42"
U2F_ROOT_SHA256 = "e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134"
PARTITION_SIZES = {"BUILD": 64, "FROZEN_DEV": 128, "FUTURE_TRAIN": 128}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def assign_subjects(subject_episode_counts: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Hash-sort subjects, then allocate fixed BUILD/DEV/FUTURE blocks."""
    if len(subject_episode_counts) != sum(PARTITION_SIZES.values()):
        raise ValueError("RAG-E6A split requires exactly 320 unique TRAIN subjects")
    if any(not subject.strip() for subject in subject_episode_counts):
        raise ValueError("subject IDs must be non-empty strings")
    if any(count < 1 for count in subject_episode_counts.values()):
        raise ValueError("every subject must have at least one runtime episode")

    ranked = sorted(
        (
            hashlib.sha256(f"{SPLIT_VERSION}\0{subject_id}".encode()).hexdigest(),
            subject_id,
        )
        for subject_id in subject_episode_counts
    )
    assignments: dict[str, list[dict[str, Any]]] = {key: [] for key in PARTITION_SIZES}
    offset = 0
    for partition, size in PARTITION_SIZES.items():
        for subject_hash, subject_id in ranked[offset : offset + size]:
            assignments[partition].append({
                "subject_id": subject_id,
                "subject_hash_sha256": subject_hash,
                "episode_count": int(subject_episode_counts[subject_id]),
            })
        offset += size
    return assignments


def build_split_manifest(
    *,
    episodes_path: Path,
    source_manifest_path: Path,
    verify_pins: bool = True,
) -> dict[str, Any]:
    """Read TRAIN runtime rows, consulting only episode_id and subject_id fields."""
    manifest_sha = sha256_file(source_manifest_path)
    episodes_sha = sha256_file(episodes_path)
    if verify_pins and manifest_sha != U2F_MANIFEST_SHA256:
        raise ValueError("frozen U2-F manifest SHA-256 mismatch")
    if verify_pins and episodes_sha != U2F_TRAIN_EPISODES_SHA256:
        raise ValueError("frozen U2-F TRAIN runtime-episodes SHA-256 mismatch")

    counts: Counter[str] = Counter()
    episode_ids: set[str] = set()
    with episodes_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected a runtime episode object at line {line_number}")
            episode_id = row.get("episode_id")
            subject_id = row.get("subject_id")
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise ValueError(f"invalid runtime episode ID at line {line_number}")
            if not isinstance(subject_id, str) or not subject_id.strip():
                raise ValueError(f"invalid runtime subject ID at line {line_number}")
            if episode_id in episode_ids:
                raise ValueError(f"duplicate runtime episode ID at line {line_number}")
            episode_ids.add(episode_id)
            counts[subject_id] += 1

    if len(episode_ids) != 4096:
        raise ValueError("frozen U2-F TRAIN must contain exactly 4096 runtime episodes")
    if any(not 12 <= count <= 14 for count in counts.values()):
        raise ValueError("U2-F TRAIN subject episode counts must remain within 12–14")
    assignments = assign_subjects(dict(counts))
    subject_partition = {
        row["subject_id"]: partition
        for partition, rows in assignments.items()
        for row in rows
    }
    if len(subject_partition) != 320:
        raise ValueError("subject assignment is incomplete")
    partition_episode_counts = {
        partition: sum(row["episode_count"] for row in rows)
        for partition, rows in assignments.items()
    }
    if sum(partition_episode_counts.values()) != len(episode_ids):
        raise ValueError("subject split lost or duplicated runtime episodes")

    return {
        "schema_version": "rag-e6a-subject-split-v1",
        "algorithm": "sort by SHA256(UTF8(version + NUL + subject_id)); allocate contiguous fixed-size blocks",
        "seed_version": SPLIT_VERSION,
        "dataset_id": "health-copilot-owned-longitudinal-v1",
        "dataset_root_sha256": U2F_ROOT_SHA256,
        "source_u2f_manifest_sha256": manifest_sha,
        "source_train_runtime_episodes_sha256": episodes_sha,
        "source_episode_count": len(episode_ids),
        "source_subject_count": len(counts),
        "partition_subject_counts": dict(PARTITION_SIZES),
        "partition_episode_counts": partition_episode_counts,
        "subject_disjoint": True,
        "counterfactual_sibling_disjoint": True,
        "sibling_isolation_basis": "U2-F sibling groups are subject-local; E6A assigns each subject to exactly one partition",
        "subjects_by_partition": assignments,
        "evaluator_truth_opened": False,
        "future_train_outcomes_opened": False,
        "reserved_test_ood_materialized": False,
        "reserved_test_ood_opened": False,
    }


def write_immutable_json(path: Path, value: Any) -> None:
    content = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
