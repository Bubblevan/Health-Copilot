"""Run the DEV-only upstream BM25 and BGE-large original-query baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_crb import BGE_QUERY_PREFIX
from eval.r2med_crb_data import SOURCE_MANIFEST_PATH, load_partition_inputs, load_source_manifest
from eval.r2med_crb_evaluator import evaluate_rankings
from eval.r2med_multiview import LuceneBM25Index, RankedDocument, dense_search_many, encode_bge
from tools.verify_r2med_models import E_ROOT, verify_models

DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
RANKING_ROOT = E_ROOT / "r2med/rankings/dev"
EMBEDDING_ROOT = E_ROOT / "r2med/bge-large"
MANIFEST_PATH = ROOT / "runs/rag_r2med_crb/dev/base_retrieval_manifest.json"


def _write_rankings(path: Path, rows: list[dict[str, Any]]) -> str:
    expected_digest = hashlib.sha256()
    for row in rows:
        expected_digest.update(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        )
    if path.is_file():
        existing = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        identity = lambda item: (
            item.get("subset"),
            item.get("query_id"),
            item.get("method"),
            [doc["doc_id"] for doc in item.get("ranking", [])],
        )
        if [identity(item) for item in existing] != [identity(item) for item in rows]:
            raise ValueError(f"existing ranking artifact differs from deterministic replay: {path}")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(path.suffix + ".partial").exists():
        raise FileExistsError(f"refusing to overwrite baseline ranking artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return expected_digest.hexdigest()


def _ranking_rows(subset: str, method: str, query_ids: list[str], results: list[list[RankedDocument]]) -> list[dict[str, Any]]:
    return [
        {
            "subset": subset,
            "query_id": query_id,
            "method": method,
            "ranking": [{"doc_id": item.doc_id, "score": item.score} for item in ranking],
        }
        for query_id, ranking in zip(query_ids, results, strict=True)
    ]


def _load_or_encode_corpus(
    model, subset, model_sha: str, partition: str = "DEV"
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    entry = next(item for item in manifest["datasets"][partition] if item["name"] == subset.name)
    cache_root = EMBEDDING_ROOT if partition == "DEV" else EMBEDDING_ROOT / partition.lower()
    vectors_path = cache_root / f"{subset.name}.npy"
    identity_path = cache_root / f"{subset.name}.json"
    expected_identity = {
        "schema_version": "r2med-bge-large-corpus-embeddings-v1",
        "subset": subset.name,
        "corpus_sha256": entry["files"]["corpus.jsonl"]["sha256"],
        "model_revision": manifest["models"]["dense_retriever"]["revision"],
        "model_weights_sha256": model_sha,
        "document_count": len(subset.dense_documents),
        "source_rows_including_duplicate_ids": len(subset.documents),
        "dimensions": 1024,
        "max_sequence_length": 512,
        "normalized": False,
    }
    if vectors_path.exists() or identity_path.exists():
        if not vectors_path.is_file() or not identity_path.is_file():
            raise FileExistsError(f"incomplete BGE embedding cache for {subset.name}; inspect before continuing")
        if json.loads(identity_path.read_text(encoding="utf-8")) != expected_identity:
            raise ValueError(f"BGE embedding identity mismatch for {subset.name}")
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.shape != (len(subset.dense_documents), 1024) or vectors.dtype != np.float32:
            raise ValueError(f"BGE embedding shape/dtype mismatch for {subset.name}")
        return vectors, expected_identity

    dense_documents = subset.dense_documents
    texts = [document.text for document in dense_documents]
    vectors = encode_bge(model, texts, batch_size=32, show_progress_bar=True)
    if vectors.shape != (len(dense_documents), 1024):
        raise ValueError(f"unexpected BGE embedding shape for {subset.name}: {vectors.shape}")
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = vectors_path.with_suffix(".npy.partial")
    with temporary.open("xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(vectors_path)
    identity_path.write_text(json.dumps(expected_identity, indent=2) + "\n", encoding="utf-8")
    return vectors, expected_identity


def _rankings_from_file(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["query_id"]] = [item["doc_id"] for item in row["ranking"]]
    return result


def run_partition_baselines(
    partition: str,
    source_root: Path,
    bge_model_path: Path,
) -> dict[str, Any]:
    if partition not in {"DEV", "TEST"}:
        raise ValueError("partition must be DEV or TEST")
    load_source_manifest(SOURCE_MANIFEST_PATH)
    verified = verify_models(bge_root=bge_model_path)
    model_sha = verified["bge_large"]["weights_sha256"]

    # Set all model/cache locations to E before importing HF model classes.
    os.environ["HF_HOME"] = str(E_ROOT / "cache/huggingface")
    os.environ["HF_HUB_CACHE"] = str(E_ROOT / "cache/huggingface/hub")
    os.environ["TRANSFORMERS_CACHE"] = str(E_ROOT / "cache/huggingface/transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(bge_model_path),
        device="cuda",
        local_files_only=True,
    )
    model.max_seq_length = 512
    model.eval()
    subsets = load_partition_inputs(partition, source_root=source_root, source_manifest_path=SOURCE_MANIFEST_PATH)
    ranking_root = RANKING_ROOT if partition == "DEV" else E_ROOT / "r2med/rankings/test"
    source_ids = {
        subset.name: {query.query_id: query.text for query in subset.queries}
        for subset in subsets
    }
    bm25_rankings: dict[str, dict[str, list[str]]] = {}
    bge_rankings: dict[str, dict[str, list[str]]] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}

    for subset in subsets:
        documents = [(document.doc_id, document.text) for document in subset.documents]
        query_ids = [query.query_id for query in subset.queries]
        bm25 = LuceneBM25Index(documents)
        bm25_rows = [bm25.search(query.text, top_k=100) for query in subset.queries]
        bm25_rankings[subset.name] = {qid: [row.doc_id for row in rows] for qid, rows in zip(query_ids, bm25_rows, strict=True)}
        bm25_path = ranking_root / subset.name / "bm25_original.jsonl"
        bm25_hash = _write_rankings(
            bm25_path,
            _ranking_rows(subset.name, "bm25_original", query_ids, bm25_rows),
        )

        corpus_vectors, _ = _load_or_encode_corpus(model, subset, model_sha, partition)
        query_texts = [BGE_QUERY_PREFIX + query.text for query in subset.queries]
        query_vectors = encode_bge(model, query_texts, batch_size=32, show_progress_bar=False)
        dense_doc_ids = [document.doc_id for document in subset.dense_documents]
        bge_rows = dense_search_many(
            query_vectors,
            corpus_vectors,
            dense_doc_ids,
            top_k=100,
            device="cuda",
        )
        bge_rankings[subset.name] = {qid: [row.doc_id for row in rows] for qid, rows in zip(query_ids, bge_rows, strict=True)}
        bge_path = ranking_root / subset.name / "bge_large_original.jsonl"
        bge_hash = _write_rankings(
            bge_path,
            _ranking_rows(subset.name, "bge_large_original", query_ids, bge_rows),
        )
        artifact_hashes[subset.name] = {
            "bm25_original_sha256": bm25_hash,
            "bge_large_original_sha256": bge_hash,
        }
        print(f"{subset.name}: BM25 and BGE-large original-query rankings saved ({len(query_ids)} queries)")

    del model
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows, summary = evaluate_rankings(partition, bm25_rankings, source_root=source_root)
    bge_rows, bge_summary = evaluate_rankings(partition, bge_rankings, source_root=source_root)
    source_manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    duplicate_id_counts = {
        subset.name: {
            "source_rows": len(subset.documents),
            "unique_doc_ids": len(subset.dense_documents),
            "repeated_id_rows": len(subset.documents) - len(subset.dense_documents),
        }
        for subset in subsets
    }
    result = {
        "schema_version": "r2med-crb-base-retrieval-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "partition": partition,
        "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest(),
        "test_status": source_manifest["test_status"],
        "bge_identity": verified["bge_large"],
        "query_texts_by_subset": {name: len(queries) for name, queries in source_ids.items()},
        "corpus_id_behavior": duplicate_id_counts,
        "rank_artifacts": artifact_hashes,
        "B0_BM25": {"metrics": summary, "query_metrics": rows},
        "B1_BGE_large": {"metrics": bge_summary, "query_metrics": bge_rows},
        "gold_boundary": "Only this evaluator call read qrels; indexing, encoding, and ranking consumed query/corpus inputs only.",
    }
    report_path = (
        MANIFEST_PATH
        if partition == "DEV"
        else E_ROOT / "r2med/test/reports/test_base_retrieval_manifest.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {partition} baseline report: {report_path}")
    partial_path = report_path.with_suffix(report_path.suffix + ".partial")
    with partial_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    partial_path.replace(report_path)
    return result


def run_dev_baselines(source_root: Path, bge_model_path: Path) -> dict[str, Any]:
    return run_partition_baselines("DEV", source_root, bge_model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("DEV",), default="DEV")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--bge-model", type=Path, default=E_ROOT / "models/bge-large-en-v1.5")
    args = parser.parse_args()
    result = run_dev_baselines(args.source_root, args.bge_model)
    print(json.dumps({"B0": result["B0_BM25"]["metrics"], "B1": result["B1_BGE_large"]["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
