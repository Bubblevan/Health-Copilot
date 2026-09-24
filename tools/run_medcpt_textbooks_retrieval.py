"""Retrieve MIRAGE evidence with MedCPT dense retrieval and cross-encoder reranking.

The embedding cache is resumable and lives outside the model directories. This
is retrieval-only: the script never calls an answer-generation API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_medrag_textbooks_retrieval import corpus_manifest, read_mirage_cases
else:
    from .run_medrag_textbooks_retrieval import corpus_manifest, read_mirage_cases

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT.parent / "models"
DEFAULT_CORPUS_DIR = ROOT / "data/raw/medrag_textbooks"
DEFAULT_INDEX = ROOT / "data/processed/medrag_textbooks/fts5_bm25.sqlite3"
DEFAULT_CACHE_DIR = ROOT / "data/processed/medcpt_textbooks"
DEFAULT_BENCHMARK = ROOT.parent / "data/mirage/benchmark.json"
DEFAULT_SUBDATASETS = ("pubmedqa", "bioasq")
EMBEDDING_DIMENSION = 768


def write_json(path: Path, value: Any, *, atomic: bool = False) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if not atomic:
        path.write_text(content, encoding="utf-8")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def weight_path(model_dir: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no local model weights found in {model_dir}")


def load_models(model_root: Path, torch: Any) -> dict[str, Any]:
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BertConfig,
    )

    query_dir = model_root / "MedCPT-Query-Encoder"
    article_dir = model_root / "MedCPT-Article-Encoder"
    cross_dir = model_root / "MedCPT-Cross-Encoder"
    for model_dir in (query_dir, article_dir, cross_dir):
        if not model_dir.is_dir():
            raise FileNotFoundError(f"MedCPT model directory is missing: {model_dir}")

    # The local article and cross-encoder downloads lack config.json. These
    # settings are identical to NCBI's published model configs. Passing configs
    # in memory avoids altering or silently repairing the user's model folder.
    bert_config = BertConfig.from_pretrained(query_dir, local_files_only=True)
    query_tokenizer = AutoTokenizer.from_pretrained(query_dir, local_files_only=True)
    query_model = AutoModel.from_pretrained(
        query_dir, config=bert_config, local_files_only=True, use_safetensors=True
    )

    article_tokenizer = AutoTokenizer.from_pretrained(article_dir, local_files_only=True)
    article_model = AutoModel.from_pretrained(
        article_dir,
        config=BertConfig.from_dict(bert_config.to_dict()),
        local_files_only=True,
        use_safetensors=True,
    )

    cross_tokenizer = AutoTokenizer.from_pretrained(cross_dir, local_files_only=True)
    cross_config = BertConfig.from_dict(bert_config.to_dict())
    cross_config.num_labels = 1
    cross_config.id2label = {0: "LABEL_0"}
    cross_config.label2id = {"LABEL_0": 0}
    cross_config.architectures = ["BertForSequenceClassification"]
    cross_model = AutoModelForSequenceClassification.from_pretrained(
        cross_dir,
        config=cross_config,
        local_files_only=True,
        use_safetensors=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for model in (query_model, article_model, cross_model):
        model.eval()
        model.to(device)

    model_files = {
        "query_encoder": weight_path(query_dir),
        "article_encoder": weight_path(article_dir),
        "cross_encoder": weight_path(cross_dir),
    }
    model_manifest = {
        name: {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in model_files.items()
    }
    model_manifest["config_provenance"] = {
        "query_encoder": "local config.json",
        "article_encoder": (
            "local config.json was absent; loaded with the matching MedCPT query-encoder BERT config"
        ),
        "cross_encoder": (
            "local config.json was absent; reconstructed as BertForSequenceClassification with one logit from NCBI's published config"
        ),
    }
    return {
        "query_tokenizer": query_tokenizer,
        "query_model": query_model,
        "article_tokenizer": article_tokenizer,
        "article_model": article_model,
        "cross_tokenizer": cross_tokenizer,
        "cross_model": cross_model,
        "model_manifest": model_manifest,
        "device": device,
    }


def cache_spec(
    corpus_dir: Path,
    expected_chunks: int,
    article_weight_hash: str,
    device: str,
) -> dict[str, Any]:
    return {
        "corpus": corpus_manifest(corpus_dir, expected_chunks),
        "expected_chunks": expected_chunks,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "article_weight_sha256": article_weight_hash,
        "document_representation": "[title, content] token pair",
        "pooling": "last_hidden_state[:, 0, :] (CLS)",
        "max_length": 512,
        "dtype": "float32",
        "inference_device": device,
    }


def embed_corpus(
    connection: sqlite3.Connection,
    article_model: Any,
    article_tokenizer: Any,
    cache_dir: Path,
    spec: dict[str, Any],
    *,
    expected_chunks: int,
    batch_size: int,
    checkpoint_rows: int,
    device: str,
    torch: Any,
) -> tuple[np.ndarray, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "article_embeddings.npy"
    manifest_path = cache_dir / "article_embeddings.manifest.json"
    partial_path = cache_dir / "article_embeddings.partial.npy"
    progress_path = cache_dir / "article_embeddings.progress.json"

    if cache_path.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"embedding cache has no manifest: {manifest_path}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("spec") != spec:
            raise RuntimeError("existing MedCPT embedding cache does not match this corpus/model")
        embeddings = np.load(cache_path, mmap_mode="r")
        if embeddings.shape != (expected_chunks, EMBEDDING_DIMENSION):
            raise RuntimeError(f"unexpected embedding cache shape: {embeddings.shape}")
        return embeddings, True

    if partial_path.exists() != progress_path.exists():
        raise RuntimeError(
            "incomplete MedCPT cache checkpoint; inspect the partial embedding and progress files"
        )
    if partial_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("spec") != spec:
            raise RuntimeError("partial MedCPT cache belongs to a different corpus/model")
        offset = int(progress["completed_rows"])
        if not 0 <= offset <= expected_chunks:
            raise RuntimeError(f"invalid MedCPT cache checkpoint offset: {offset}")
        embeddings = np.lib.format.open_memmap(partial_path, mode="r+")
        if embeddings.shape != (expected_chunks, EMBEDDING_DIMENSION):
            raise RuntimeError(f"unexpected partial embedding shape: {embeddings.shape}")
        print(f"Resuming article embeddings at row {offset:,}/{expected_chunks:,}", flush=True)
    else:
        offset = 0
        embeddings = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(expected_chunks, EMBEDDING_DIMENSION),
        )

    cursor = connection.execute("SELECT title, content FROM chunks ORDER BY rowid")
    skipped = 0
    while skipped < offset:
        batch = cursor.fetchmany(min(batch_size, offset - skipped))
        if not batch:
            raise RuntimeError("corpus ended before the cached embedding checkpoint")
        skipped += len(batch)

    started = time.perf_counter()
    last_checkpoint = offset
    while offset < expected_chunks:
        rows = cursor.fetchmany(min(batch_size, expected_chunks - offset))
        if not rows:
            raise RuntimeError(f"corpus ended after {offset:,} of {expected_chunks:,} rows")
        pairs = [[str(title), str(content)] for title, content in rows]
        tokens = article_tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            vectors = article_model(**tokens).last_hidden_state[:, 0, :]
        vectors = vectors.detach().to(dtype=torch.float32, device="cpu").numpy()
        if vectors.shape != (len(rows), EMBEDDING_DIMENSION):
            raise RuntimeError(f"unexpected article encoder output shape: {vectors.shape}")
        embeddings[offset : offset + len(rows)] = vectors
        offset += len(rows)

        if offset - last_checkpoint >= checkpoint_rows or offset == expected_chunks:
            embeddings.flush()
            write_json(
                progress_path,
                {"spec": spec, "completed_rows": offset},
                atomic=True,
            )
            elapsed = max(time.perf_counter() - started, 0.001)
            print(
                f"Encoded {offset:,}/{expected_chunks:,} textbook chunks "
                f"({offset / elapsed:,.1f} chunks/s)",
                flush=True,
            )
            last_checkpoint = offset

    embeddings.flush()
    del embeddings
    partial_path.replace(cache_path)
    write_json(
        manifest_path,
        {
            "spec": spec,
            "shape": [expected_chunks, EMBEDDING_DIMENSION],
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    progress_path.unlink()
    return np.load(cache_path, mmap_mode="r"), False


def encode_queries(
    cases: list[dict[str, Any]], models: dict[str, Any], torch: Any, batch_size: int
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    tokenizer = models["query_tokenizer"]
    model = models["query_model"]
    for start in range(0, len(cases), batch_size):
        questions = [case["question"] for case in cases[start : start + batch_size]]
        tokens = tokenizer(
            questions,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=64,
        ).to(models["device"])
        with torch.inference_mode():
            encoded = model(**tokens).last_hidden_state[:, 0, :]
        vectors.append(encoded.detach().to(dtype=torch.float32, device="cpu").numpy())
    if not vectors:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    return np.concatenate(vectors, axis=0)


def dense_candidates(
    query_vectors: np.ndarray,
    embeddings: np.ndarray,
    candidate_count: int,
    *,
    batch_size: int = 16,
) -> list[list[tuple[int, float]]]:
    doc_count = embeddings.shape[0]
    k = min(candidate_count, doc_count)
    results: list[list[tuple[int, float]]] = []
    for start in range(0, len(query_vectors), batch_size):
        score_batch = np.asarray(
            query_vectors[start : start + batch_size] @ embeddings.T,
            dtype=np.float32,
        )
        for scores in score_batch:
            indices = np.argpartition(scores, doc_count - k)[doc_count - k :]
            ordered = sorted(indices.tolist(), key=lambda index: float(scores[index]), reverse=True)
            results.append([(index + 1, float(scores[index])) for index in ordered])
    return results


def load_candidate_rows(
    connection: sqlite3.Connection,
    candidates: list[tuple[int, float]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    rowids = [rowid for rowid, _ in candidates]
    placeholders = ",".join("?" for _ in rowids)
    rows = connection.execute(
        f"SELECT rowid, id, title, content, contents FROM chunks WHERE rowid IN ({placeholders})",
        rowids,
    )
    by_rowid = {
        int(row[0]): {"id": row[1], "title": row[2], "content": row[3], "contents": row[4]}
        for row in rows
    }
    result = []
    for dense_rank, (rowid, score) in enumerate(candidates, 1):
        if rowid not in by_rowid:
            raise RuntimeError(f"embedding cache refers to missing corpus rowid {rowid}")
        result.append(
            {**by_rowid[rowid], "rowid": rowid, "dense_rank": dense_rank, "dense_score": score}
        )
    return result


def rerank(
    question: str,
    documents: list[dict[str, Any]],
    cross_tokenizer: Any,
    cross_model: Any,
    *,
    batch_size: int,
    torch: Any,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        pairs = [[question, doc["contents"]] for doc in batch]
        tokens = cross_tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        ).to(next(cross_model.parameters()).device)
        with torch.inference_mode():
            logits = cross_model(**tokens).logits.reshape(-1)
        scores = logits.detach().to(dtype=torch.float32, device="cpu").tolist()
        scored.extend({**doc, "rerank_score": float(score)} for doc, score in zip(batch, scores))
    scored.sort(key=lambda doc: doc["rerank_score"], reverse=True)
    for rank, doc in enumerate(scored, 1):
        doc["rerank_rank"] = rank
    return scored


def summarize(rows: list[dict[str, Any]], latencies_ms: list[float]) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "queries_with_results": sum(bool(row["retrieved_evidence"]) for row in rows),
        "retrieved_evidence_coverage": (
            sum(bool(row["retrieved_evidence"]) for row in rows) / len(rows) if rows else None
        ),
        "mean_latency_ms": sum(latencies_ms) / len(latencies_ms) if latencies_ms else None,
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else None,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else None,
    }


def write_results(
    output_dir: Path,
    subsets: dict[str, list[dict[str, Any]]],
    candidate_rows: list[list[dict[str, Any]]],
    *,
    top_k: int,
    dense_depth: int,
    cache_reused: bool,
    device: str,
    model_manifest: dict[str, Any],
    elapsed_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_rows: list[dict[str, Any]] = []
    subset_metrics: dict[str, Any] = {}
    offset = 0
    for subset, cases in subsets.items():
        output_rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        for case in cases:
            result = candidate_rows[offset]
            offset += 1
            evidence = result[:top_k]
            row = {
                **case,
                "retrieved_evidence": evidence,
                "retrieval_metadata": {
                    "retriever": "MedCPT Query/Article Encoder dot-product + Cross-Encoder",
                    "dense_candidate_depth": dense_depth,
                    "top_k": top_k,
                    "compute_device": device,
                    "embedding_cache_reused": cache_reused,
                },
            }
            output_rows.append(row)
            latencies.append(float(result[0].get("query_latency_ms", 0.0)) if result else 0.0)
        with (output_dir / f"{subset}.jsonl").open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        subset_metrics[subset] = summarize(output_rows, latencies)
        overall_rows.extend(output_rows)

    latencies = [
        float(row["retrieved_evidence"][0].get("query_latency_ms", 0.0))
        if row["retrieved_evidence"]
        else 0.0
        for row in overall_rows
    ]
    metrics = summarize(overall_rows, latencies)
    write_json(
        output_dir / "metrics.json",
        {
            "status": "completed",
            "benchmark": "MIRAGE benchmark.json",
            "retriever": "MedCPT-Query-Encoder + MedCPT-Article-Encoder, inner product; MedCPT-Cross-Encoder rerank",
            "corpus": "local MedRAG Textbooks, 125847 chunks",
            "top_k": top_k,
            "dense_candidate_depth": dense_depth,
            "compute_device": device,
            "case_count": len(overall_rows),
            "retrieved_evidence_coverage": metrics["retrieved_evidence_coverage"],
            "mean_latency_ms": metrics["mean_latency_ms"],
            "p50_latency_ms": metrics["p50_latency_ms"],
            "p95_latency_ms": metrics["p95_latency_ms"],
            "per_subdataset": subset_metrics,
            "model_weights": model_manifest,
            "embedding_cache_reused": cache_reused,
            "answer_generation": False,
            "llm_calls": 0,
            "retrieval_recall": "not measured: this MIRAGE artifact does not provide qrels for the textbook corpus",
            "elapsed_seconds": round(elapsed_seconds, 3),
            "built_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    report = [
        "# MIRAGE × MedRAG Textbooks — MedCPT retrieval-only run",
        "",
        f"- Cases retrieved: **{len(overall_rows)}**",
        "- Corpus chunks: **125,847**",
        f"- Dense candidate depth: **{dense_depth}**; cross-encoder output: **top {top_k}**",
        f"- Compute device: **{device}**",
        f"- Evidence coverage: **{metrics['retrieved_evidence_coverage']:.1%}**"
        if overall_rows
        else "- Evidence coverage: **n/a**",
        f"- Mean per-query retrieval/rerank latency: **{metrics['mean_latency_ms']:.1f} ms**"
        if overall_rows
        else "- Mean per-query retrieval/rerank latency: **n/a**",
        f"- Article embedding cache reused: **{cache_reused}**",
        "- LLM/API calls: **0**",
        "- Answer accuracy and retrieval recall: **not measured** (retrieval-only; no textbook qrels).",
        "- Embeddings: MedCPT [title, content] pairs, CLS vectors, inner-product ranking; official second-stage cross-encoder scores are higher-is-better.",
        "",
        "Per-subdataset details and local model checksums are in `metrics.json`; JSONL rows are directly usable as evidence input for the answer runner.",
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subdatasets", nargs="+", default=list(DEFAULT_SUBDATASETS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-candidate-depth", type=int, default=100)
    parser.add_argument("--limit-per-subset", type=int)
    parser.add_argument("--expected-corpus-chunks", type=int, default=125847)
    parser.add_argument("--article-batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-rows", type=int, default=1024)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    if args.top_k < 1 or args.dense_candidate_depth < args.top_k:
        parser.error("--top-k must be positive and no larger than --dense-candidate-depth")
    if any(
        value < 1
        for value in (
            args.article_batch_size,
            args.query_batch_size,
            args.rerank_batch_size,
            args.checkpoint_rows,
            args.torch_threads,
        )
    ):
        parser.error("batch sizes, checkpoint rows, and torch threads must be positive")
    if args.limit_per_subset is not None and args.limit_per_subset < 1:
        parser.error("--limit-per-subset must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is non-empty; choose a new path: {args.output_dir}"
        )
    if not args.index_path.is_file():
        raise FileNotFoundError(f"BM25 corpus index is missing: {args.index_path}")

    import torch

    torch.set_num_threads(args.torch_threads)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MedCPT device: {device}; torch threads: {args.torch_threads}", flush=True)
    started = time.perf_counter()
    models = load_models(args.model_root, torch)
    connection = sqlite3.connect(args.index_path)
    try:
        corpus_count = int(connection.execute("SELECT count(*) FROM chunks").fetchone()[0])
        if corpus_count != args.expected_corpus_chunks:
            raise RuntimeError(
                f"index has {corpus_count} chunks; expected {args.expected_corpus_chunks}"
            )
        rowid_range = connection.execute("SELECT min(rowid), max(rowid) FROM chunks").fetchone()
        if rowid_range != (1, corpus_count):
            raise RuntimeError(
                "corpus rowids must be contiguous from 1 so each embedding maps to its source chunk"
            )
        subsets = read_mirage_cases(args.benchmark_json, args.subdatasets, args.limit_per_subset)
        cases = [case for rows in subsets.values() for case in rows]
        if any(
            not isinstance(case.get("question"), str) or not case["question"].strip()
            for case in cases
        ):
            raise ValueError("every MIRAGE case must contain a non-empty question")

        spec = cache_spec(
            args.corpus_dir,
            args.expected_corpus_chunks,
            models["model_manifest"]["article_encoder"]["sha256"],
            device,
        )
        embeddings, cache_reused = embed_corpus(
            connection,
            models["article_model"],
            models["article_tokenizer"],
            args.cache_dir,
            spec,
            expected_chunks=args.expected_corpus_chunks,
            batch_size=args.article_batch_size,
            checkpoint_rows=args.checkpoint_rows,
            device=device,
            torch=torch,
        )
        query_started = time.perf_counter()
        query_vectors = encode_queries(cases, models, torch, args.query_batch_size)
        mean_query_encode_ms = (
            (time.perf_counter() - query_started) * 1000.0 / len(cases) if cases else 0.0
        )
        dense_started = time.perf_counter()
        candidate_lists = dense_candidates(
            query_vectors,
            embeddings,
            args.dense_candidate_depth,
            batch_size=args.query_batch_size,
        )
        mean_dense_search_ms = (
            (time.perf_counter() - dense_started) * 1000.0 / len(cases) if cases else 0.0
        )
        all_results: list[list[dict[str, Any]]] = []
        for index, (case, candidates) in enumerate(zip(cases, candidate_lists), 1):
            rerank_started = time.perf_counter()
            docs = load_candidate_rows(connection, candidates)
            ranked = rerank(
                case["question"],
                docs,
                models["cross_tokenizer"],
                models["cross_model"],
                batch_size=args.rerank_batch_size,
                torch=torch,
            )
            elapsed_ms = (time.perf_counter() - rerank_started) * 1000.0
            if ranked:
                ranked[0]["query_latency_ms"] = (
                    elapsed_ms + mean_query_encode_ms + mean_dense_search_ms
                )
            all_results.append(ranked)
            if index % 25 == 0 or index == len(cases):
                print(f"Reranked {index:,}/{len(cases):,} MIRAGE queries", flush=True)

        write_results(
            args.output_dir,
            subsets,
            all_results,
            top_k=args.top_k,
            dense_depth=args.dense_candidate_depth,
            cache_reused=cache_reused,
            device=device,
            model_manifest=models["model_manifest"],
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        connection.close()

    print(
        json.dumps(
            {
                "status": "completed",
                "cases": sum(len(rows) for rows in subsets.values()),
                "cache_reused": cache_reused,
                "output_dir": str(args.output_dir),
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
