"""Encode the frozen MIRAGE questions with the already-pinned local BGE model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.e1_3_router_dataset import (
    canonical_sha256,
    case_order_sha256,
    load_cases,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, path)


def question_set_sha256(cases: list[Any]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.case_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(case.question.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_cache(
    embedding_path: Path,
    manifest_path: Path,
    expected: dict[str, Any],
) -> np.ndarray:
    manifest = read_json(manifest_path)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Existing BGE cache metadata mismatch at {key}")
    if manifest.get("embedding_file_sha256") != sha256_file(embedding_path):
        raise ValueError("Existing BGE embedding cache SHA-256 mismatch")
    matrix = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if list(matrix.shape) != expected["embedding_shape"]:
        raise ValueError("Existing BGE embedding cache shape mismatch")
    if str(matrix.dtype) != expected["dtype"] or not np.isfinite(matrix).all():
        raise ValueError("Existing BGE embedding cache dtype/value validation failed")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--cache-root", type=Path, default=Path(r"E:\Health-Copilot-E1.3\router")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    cache_root = args.cache_root.resolve()
    protocol_path = repo_root / "runs" / "e1_3" / "router_protocol.json"
    protocol = read_json(protocol_path)
    protocol_hash = protocol.get("protocol_sha256")
    if canonical_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"}) != protocol_hash:
        raise ValueError("Frozen protocol SHA-256 validation failed")

    input_path = Path(protocol["benchmark"]["input_artifact"])
    if sha256_file(input_path) != protocol["benchmark"]["input_sha256"]:
        raise ValueError("Frozen MIRAGE question-set input hash changed")
    cases = load_cases(input_path)
    ids = [case.case_id for case in cases]
    order_hash = case_order_sha256(ids)
    if order_hash != protocol["benchmark"]["case_order_sha256"]:
        raise ValueError("MIRAGE case ID order differs from the frozen protocol")
    content_hash = question_set_sha256(cases)

    bge_config = protocol["models"]["bge"]
    model_path = Path(bge_config["model_path"])
    model_file = model_path / bge_config["weights_file"]
    if not model_file.is_file() or model_file.stat().st_size != bge_config["weights_bytes"]:
        raise ValueError("Pinned local BGE model file is missing or has the wrong size")
    if sha256_file(model_file) != bge_config["weights_sha256"]:
        raise ValueError("Pinned local BGE model file SHA-256 changed")

    cache_root.mkdir(parents=True, exist_ok=True)
    embedding_path = cache_root / "bge_embeddings.npy"
    manifest_path = cache_root / "bge_embedding_manifest.json"
    expected_shape = [len(cases), 768]
    expected = {
        "schema_version": "e1-3-bge-question-embeddings-v1",
        "model_repo_id": bge_config["repo_id"],
        "model_revision": bge_config["revision"],
        "model_weights_sha256": bge_config["weights_sha256"],
        "benchmark_input_sha256": protocol["benchmark"]["input_sha256"],
        "question_set_identity_sha256": content_hash,
        "case_id_order_sha256": order_hash,
        "case_count": len(cases),
        "embedding_shape": expected_shape,
        "dtype": "float32",
        "normalized": bool(bge_config["encoding"]["normalize_embeddings"]),
        "max_sequence_length": int(bge_config["encoding"]["max_sequence_length"]),
        "batch_size": int(bge_config["encoding"]["batch_size"]),
        "protocol_sha256": protocol_hash,
    }
    if embedding_path.exists() and manifest_path.exists():
        matrix = validate_cache(embedding_path, manifest_path, expected)
        verified_manifest = read_json(manifest_path)
        atomic_json(repo_root / "runs" / "e1_3" / "router_embedding_manifest.json", verified_manifest)
        print(json.dumps({"status": "REUSED_VALID_CACHE", "shape": list(matrix.shape), "model_sha256": bge_config["weights_sha256"]}))
        return 0
    if embedding_path.exists() or manifest_path.exists():
        raise FileExistsError("Partial BGE cache exists; refusing to overwrite ambiguous user data")

    encoding = bge_config["encoding"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(cache_root / "hf-cache")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_root / "sentence-transformers-cache")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(protocol["outer_cv"]["seed"]))
    model = SentenceTransformer(
        str(model_path),
        device=device,
        local_files_only=True,
        revision=bge_config["revision"],
        cache_folder=str(cache_root / "sentence-transformers-cache"),
    )
    model.max_seq_length = int(encoding["max_sequence_length"])
    model.eval()
    matrix = model.encode(
        [case.question for case in cases],
        batch_size=int(encoding["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(encoding["normalize_embeddings"]),
        precision=encoding["precision"],
    )
    matrix = np.asarray(matrix, dtype=np.float32)
    if list(matrix.shape) != expected_shape or not np.isfinite(matrix).all():
        raise ValueError(f"Unexpected BGE embedding shape or non-finite values: {matrix.shape}")

    with tempfile.NamedTemporaryFile(mode="wb", dir=cache_root, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        np.save(temporary, matrix, allow_pickle=False)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, embedding_path)
    manifest = {**expected, "embedding_file_sha256": sha256_file(embedding_path)}
    atomic_json(manifest_path, manifest)
    atomic_json(repo_root / "runs" / "e1_3" / "router_embedding_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "BGE_QUESTION_CACHE_READY",
                "case_count": len(cases),
                "embedding_shape": list(matrix.shape),
                "device": device,
                "embedding_file_sha256": manifest["embedding_file_sha256"],
                "question_set_identity_sha256": content_hash,
                "model_weights_sha256": bge_config["weights_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
