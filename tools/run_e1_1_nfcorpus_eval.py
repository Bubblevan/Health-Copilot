"""Run the frozen four-arm E1.1 NFCorpus retrieval ablation offline.

The input is the E0 normalized NFCorpus artifact, not a newly downloaded BEIR
directory.  Learned components are required to load from the local Hugging
Face cache with explicit revisions; a missing dependency or model is a hard
failure, never a fallback to the hashing or token-overlap demos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from health_ai_copilot.eval.nfcorpus import NFCorpusDataset, RetrievalEvalCase, evaluate_nfcorpus
from health_ai_copilot.retrieval import (
    BM25Retriever,
    DenseIndex,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
    RetrievalDocument,
    SentenceTransformerEmbeddingBackend,
    SentenceTransformerReranker,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_RERANKER_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default=".health-bench-data/normalized/nfcorpus-v1"
    )
    parser.add_argument("--manifest", default="benchmarks/nfcorpus/manifest.json")
    parser.add_argument("--output-dir", default="runs/e1_1/nfcorpus_external_v1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision", default=DEFAULT_EMBEDDING_REVISION)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--reranker-revision", default=DEFAULT_RERANKER_REVISION)
    args = parser.parse_args(argv)

    if args.top_k <= 0 or args.candidate_top_k <= 0:
        parser.error("top-k and candidate-top-k must be positive")
    if args.candidate_top_k < args.top_k:
        parser.error("candidate-top-k must be at least top-k")
    if args.rrf_k < 0:
        parser.error("rrf-k must be non-negative")

    data_root = Path(args.data_dir)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing E1.1 artifact directory: {output_dir}"
        )

    dataset, identity, split_manifest = _load_normalized_dataset(data_root)
    provenance = _validate_provenance(data_root, manifest_path, identity, split_manifest)

    # These constructors deliberately use explicit revisions and local-only
    # loading.  Any unavailable learned artifact aborts the run.
    embedding = SentenceTransformerEmbeddingBackend(
        args.embedding_model,
        revision=args.embedding_revision,
        local_files_only=True,
    )
    reranker = SentenceTransformerReranker(
        args.reranker_model,
        revision=args.reranker_revision,
        local_files_only=True,
    )

    build_commit = _git_sha()
    bm25 = BM25Retriever(dataset.documents)
    dense = DenseRetriever(
        DenseIndex.build(
            dataset.documents,
            embedding,
            knowledge_pack_version=None,
            build_commit=build_commit,
        ),
        embedding,
    )
    hybrid = HybridRetriever(bm25, dense, rrf_k=args.rrf_k)
    hybrid_rerank = RerankedRetriever(
        hybrid,
        reranker,
        candidate_top_k=args.candidate_top_k,
    )
    arms = {
        "bm25": bm25,
        "learned_dense": dense,
        "hybrid_rrf": hybrid,
        "hybrid_crossencoder": hybrid_rerank,
    }

    metrics: dict[str, Any] = {}
    case_rows: list[dict[str, Any]] = []
    for name, retriever in arms.items():
        captured = _CaptureRetriever(retriever)
        metrics[name] = evaluate_nfcorpus(dataset, top_k=args.top_k, retriever=captured)
        case_rows.extend(_case_rows(name, dataset, captured.results, args.top_k))
    summary = _summary(metrics, case_rows, args.top_k)
    result = {
        "experiment": "E1.1",
        "benchmark": "nfcorpus-v1",
        "dataset": provenance,
        "execution": {
            "git_sha": build_commit,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "split": dataset.split,
            "top_k": args.top_k,
            "candidate_top_k": args.candidate_top_k,
            "rrf_k": args.rrf_k,
            "arms": list(arms),
        },
        "learned_artifacts": {
            "embedding": {
                "model": args.embedding_model,
                "revision": args.embedding_revision,
                "dimension": embedding.dimension,
                "local_files_only": True,
            },
            "reranker": {
                "model": args.reranker_model,
                "revision": args.reranker_revision,
                "local_files_only": True,
            },
        },
        "metrics": metrics,
        "failure_summary": summary,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "result.json", result)
    _write_json(output_dir / "metrics.json", metrics)
    _write_jsonl(output_dir / "case_results.jsonl", case_rows)
    _write_json(
        output_dir / "run_config.json",
        {
            "data_dir": str(data_root),
            "manifest": str(manifest_path),
            "embedding_model": args.embedding_model,
            "embedding_revision": args.embedding_revision,
            "reranker_model": args.reranker_model,
            "reranker_revision": args.reranker_revision,
            "local_files_only": True,
            "top_k": args.top_k,
            "candidate_top_k": args.candidate_top_k,
            "rrf_k": args.rrf_k,
            "git_sha": build_commit,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _load_normalized_dataset(
    root: Path,
) -> tuple[NFCorpusDataset, dict[str, Any], dict[str, Any]]:
    identity = _read_json(root / "identity.json")
    split_manifest = _read_json(root / "split_manifest.json")
    documents = tuple(
        RetrievalDocument(
            id=_required_string(row, "document_id"),
            title=row.get("title") or "",
            text=_required_string(row, "text"),
            metadata={"dataset": "beir-nfcorpus"},
        )
        for row in _read_jsonl(root / "corpus.jsonl")
    )
    cases: list[RetrievalEvalCase] = []
    for row in _read_jsonl(root / "cases.jsonl"):
        payload = row.get("payload", {})
        gold = row.get("gold", {})
        split = _required_string(payload, "split")
        qrels = gold.get("qrels")
        if not isinstance(qrels, dict) or not qrels:
            raise ValueError(f"case {row.get('case_id')} has no graded qrels")
        cases.append(
            RetrievalEvalCase(
                id=_required_string(payload, "query_id"),
                question=_required_string(payload, "query"),
                relevance={str(key): int(value) for key, value in qrels.items()},
            )
        )
    split = str(split_manifest.get("split", ""))
    if not split or any(_required_string(row.get("payload", {}), "split") != split for row in _read_jsonl(root / "cases.jsonl")):
        raise ValueError("normalized NFCorpus cases do not share one split")
    return NFCorpusDataset(documents=documents, cases=tuple(cases), split=split), identity, split_manifest


def _validate_provenance(
    data_root: Path,
    manifest_path: Path,
    identity: dict[str, Any],
    split_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    raw_sha = [item.get("sha256") for item in manifest.get("raw_artifacts", [])]
    identity_raw_sha = identity.get("raw_artifact_sha256", [])
    if raw_sha != identity_raw_sha:
        raise ValueError("normalized identity raw SHA does not match the pinned manifest")
    cases_path = data_root / "cases.jsonl"
    split_path = data_root / "split_manifest.json"
    normalized_sha = _sha256(cases_path)
    split_sha = _sha256(split_path)
    if normalized_sha != identity.get("normalized_sha256"):
        raise ValueError("normalized cases SHA does not match identity.json")
    if split_sha != identity.get("split_manifest_sha256"):
        raise ValueError("split manifest SHA does not match identity.json")
    if identity.get("benchmark_id") != manifest.get("benchmark_id"):
        raise ValueError("normalized benchmark ID does not match manifest")
    if identity.get("adapter_version") != manifest.get("adapter_version"):
        raise ValueError("normalized adapter version does not match manifest")
    if identity.get("schema_version") != manifest.get("normalization_schema_version"):
        raise ValueError("normalized schema version does not match manifest")
    if identity.get("case_count") != len(split_manifest.get("case_ids", [])):
        raise ValueError("normalized case count does not match split manifest")
    return {
        "benchmark_id": manifest["benchmark_id"],
        "version": manifest["version"],
        "upstream_revision": manifest["upstream_revision"],
        "manifest_sha256": _sha256(manifest_path),
        "raw_artifact_sha256": raw_sha,
        "normalized_sha256": identity["normalized_sha256"],
        "split_manifest_sha256": identity["split_manifest_sha256"],
        "adapter_version": identity["adapter_version"],
        "schema_version": identity["schema_version"],
        "case_count": identity["case_count"],
        "split": split_manifest["split"],
        "data_files": {
            "corpus_sha256": _sha256(data_root / "corpus.jsonl"),
            "cases_sha256": normalized_sha,
            "split_manifest_sha256": split_sha,
        },
    }


def _case_rows(
    name: str, dataset: NFCorpusDataset, results: list[list[Any]], top_k: int
) -> list[dict[str, Any]]:
    if len(results) != len(dataset.cases):
        raise ValueError(f"captured result count does not match cases for {name}")
    rows: list[dict[str, Any]] = []
    for case, retrieved in zip(dataset.cases, results, strict=True):
        relevant = {doc_id for doc_id, score in case.relevance.items() if score > 0}
        retrieved_ids = [item.source_id for item in retrieved]
        rank = next((index for index, doc_id in enumerate(retrieved_ids, 1) if doc_id in relevant), None)
        rows.append(
            {
                "arm": name,
                "query_id": case.id,
                "retrieved_ids": retrieved_ids,
                "relevant_count": len(relevant),
                "hit_at_1": bool(retrieved_ids and retrieved_ids[0] in relevant),
                "reciprocal_rank": 1 / rank if rank else 0.0,
                "recall_at_k": len(set(retrieved_ids).intersection(relevant)) / len(relevant),
                "ndcg_at_k": _ndcg_at_k(retrieved_ids, case.relevance, top_k),
            }
        )
    return rows


class _CaptureRetriever:
    """Record one arm's results while reusing the frozen metric evaluator."""

    def __init__(self, retriever) -> None:
        self.retriever = retriever
        self.results: list[list[Any]] = []

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        result = self.retriever.search(query, top_k=top_k)
        self.results.append(result)
        return result


def _summary(metrics: dict[str, Any], rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for name, metric in metrics.items():
        arm_rows = [row for row in rows if row["arm"] == name]
        by_arm[name] = {
            "evaluated_cases": metric["evaluated_cases"],
            "hit_at_1": sum(row["hit_at_1"] for row in arm_rows) / len(arm_rows),
            "misses_at_k": sum(row["recall_at_k"] < 1.0 for row in arm_rows),
            "metric_keys": [f"recall_at_{top_k}", "mrr", f"ndcg_at_{top_k}"],
        }
    return {"arms": by_arm, "failure_taxonomy": "graded_qrels_retrieval_v1"}


def _ndcg_at_k(retrieved_ids: list[str], relevance: dict[str, int], k: int) -> float:
    def gain(score: int) -> float:
        return float(2**score - 1)

    dcg = sum(
        gain(relevance.get(document_id, 0)) / math.log2(rank + 2)
        for rank, document_id in enumerate(retrieved_ids[:k])
    )
    ideal_scores = sorted((score for score in relevance.values() if score > 0), reverse=True)[:k]
    ideal_dcg = sum(gain(score) / math.log2(rank + 2) for rank, score in enumerate(ideal_scores))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _required_string(value: Any, key: str) -> str:
    if not isinstance(value, dict) or not isinstance(value.get(key), str) or not value[key]:
        raise ValueError(f"expected non-empty string field: {key}")
    return value[key]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
