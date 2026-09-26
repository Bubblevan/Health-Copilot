"""Deterministic identities and append-only artifacts for MEM-1 runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_hash(identity: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(identity))


def make_cache_identity(
    *,
    system: str,
    question_id: str,
    dataset_sha256: str,
    system_config_hash: str,
    prompt_hashes: dict[str, str],
    reader_artifact_sha256: str,
    embedding_model: str | None,
    code_patch_hash: str,
) -> dict[str, Any]:
    identity = {
        "system": system,
        "question_id": question_id,
        "dataset_sha256": dataset_sha256,
        "system_config_hash": system_config_hash,
        "prompt_hashes": dict(sorted(prompt_hashes.items())),
        "reader_artifact_sha256": reader_artifact_sha256,
        "embedding_model": embedding_model,
        "code_patch_hash": code_patch_hash,
    }
    return {**identity, "identity_sha256": identity_hash(identity)}


def write_run_manifest(
    path: str | Path,
    *,
    run_id: str,
    track: str,
    dataset: dict[str, Any],
    split: str,
    question_ids: list[str],
    roles: dict[str, dict[str, Any]],
    system_config_hash: str,
    code_patch_sha256: str,
    runner_code_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    required_roles = {
        "reader_answer_model",
        "memory_system",
        "embedding_model",
        "judge_model",
    }
    if set(roles) != required_roles:
        raise ValueError(f"Run manifest roles must be exactly {sorted(required_roles)}")
    if split != "DEV":
        raise ValueError("MEM-1 run manifests may reference DEV only")
    if not question_ids or len(question_ids) != len(set(question_ids)):
        raise ValueError("Run manifest requires unique, non-empty question IDs")
    if dataset.get("test_access") is not False:
        raise ValueError("MEM-1 run manifest must explicitly record test_access=false")
    forbidden = ("api_key", "secret", "access_token", "authorization")

    def reject_secrets(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if any(fragment in key.lower() for fragment in forbidden):
                    raise ValueError("Run manifests must never contain credentials")
                reject_secrets(child)
        elif isinstance(value, list):
            for child in value:
                reject_secrets(child)

    reject_secrets(roles)
    manifest = {
        "manifest_version": "mem1-run-v1",
        "run_id": run_id,
        "track": track,
        "created_at": created_at,
        "dataset": dataset,
        "split": split,
        "question_ids": question_ids,
        "roles": roles,
        "system_config_hash": system_config_hash,
        "code_patch_sha256": code_patch_sha256,
        "runner_code_sha256": runner_code_sha256,
        "test_access": False,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "created_at"}
        requested = {key: value for key, value in manifest.items() if key != "created_at"}
        if comparable != requested:
            raise FileExistsError(f"Refusing to rewrite frozen run manifest {destination}")
        return existing
    destination.write_text(encoded, encoding="utf-8", newline="\n")
    return manifest


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL row {line_number} in {source}") from error
    return rows


def find_cached_prediction(
    predictions_path: str | Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    wanted = verified_identity_hash(expected_identity)
    for row in reversed(read_jsonl(predictions_path)):
        if (
            row.get("cache_identity", {}).get("identity_sha256") == wanted
            and verified_identity_hash(row.get("cache_identity", {})) == wanted
            and row.get("quality_status") == "OK"
            and isinstance(row.get("predicted"), str)
        ):
            return row
    return None


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def freeze_jsonl(path: str | Path) -> str:
    return sha256_file(path)


def verified_identity_hash(identity: dict[str, Any]) -> str:
    claimed = identity.get("identity_sha256")
    material = {key: value for key, value in identity.items() if key != "identity_sha256"}
    actual = identity_hash(material)
    if claimed != actual:
        raise ValueError("Cache identity hash does not match its canonical fields")
    return actual


def write_hash_sidecar(path: str | Path, digest_path: str | Path) -> str:
    digest = freeze_jsonl(path)
    destination = Path(digest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(f"{digest}  {Path(path).name}\n", encoding="ascii")
    return digest


def verify_hash_sidecar(path: str | Path, digest_path: str | Path) -> bool:
    sidecar = Path(digest_path).read_text(encoding="ascii").strip().split()
    return len(sidecar) == 2 and sidecar[1] == Path(path).name and sidecar[0] == sha256_file(path)


def write_failures(path: str | Path, failures: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as output:
        for failure in failures:
            output.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
