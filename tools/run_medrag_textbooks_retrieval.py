"""Build a local SQLite FTS5 BM25 index and retrieve MIRAGE evidence.

This is a retrieval-only experiment. It does not call an LLM, claim retrieval
recall (MIRAGE does not ship relevance qrels here), or overwrite run artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SUBDATASETS = ("pubmedqa", "bioasq")
SUBDATASET_ALIASES = {"bioasq_yes_no": "bioasq", "pubmedqa_yes_no": "pubmedqa"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def file_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return {"name": path.name, "bytes": size, "sha256": digest.hexdigest()}


def corpus_manifest(corpus_dir: Path, expected_chunks: int) -> dict[str, Any]:
    files = sorted(corpus_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL corpus files under {corpus_dir}")
    return {
        "files": [file_manifest(path) for path in files],
        "expected_chunks": expected_chunks,
    }


def read_mirage_cases(benchmark_path: Path, subsets: list[str], limit_per_subset: int | None) -> dict[str, list[dict[str, Any]]]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(benchmark, dict):
        raise TypeError("MIRAGE benchmark.json must be an object keyed by subdataset")
    result: dict[str, list[dict[str, Any]]] = {}
    for subset in subsets:
        source_subset = SUBDATASET_ALIASES.get(subset, subset)
        rows = benchmark.get(source_subset)
        if not isinstance(rows, dict):
            raise KeyError(f"MIRAGE subdataset {source_subset!r} is absent or malformed")
        selected: list[dict[str, Any]] = []
        for case_id, raw in rows.items():
            if not isinstance(raw, dict) or not isinstance(raw.get("question"), str):
                raise TypeError(f"invalid question record {source_subset}:{case_id}")
            case = dict(raw)
            case["case_id"] = str(case_id)
            case["subdataset"] = subset
            selected.append(case)
            if limit_per_subset and len(selected) >= limit_per_subset:
                break
        result[subset] = selected
    return result


def build_or_validate_index(index_path: Path, corpus_dir: Path, expected_chunks: int) -> dict[str, Any]:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest = corpus_manifest(corpus_dir, expected_chunks)

    if index_path.exists():
        connection = sqlite3.connect(index_path)
        try:
            row = connection.execute(
                "SELECT value FROM index_meta WHERE key='corpus_manifest'"
            ).fetchone()
            if row is None or json.loads(row[0]) != source_manifest:
                raise RuntimeError(f"existing index does not match current corpus: {index_path}")
            count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            if count != expected_chunks:
                raise RuntimeError(f"existing index has {count} chunks; expected {expected_chunks}")
        finally:
            connection.close()
        return {"index_path": str(index_path), "chunks": expected_chunks, "reused": True}

    building_path = index_path.with_name(index_path.name + ".building")
    if building_path.exists():
        raise FileExistsError(f"incomplete build artifact exists; inspect before retrying: {building_path}")

    connection = sqlite3.connect(building_path)
    seen_ids: set[str] = set()
    inserted = 0
    try:
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE chunks USING fts5(id UNINDEXED, title UNINDEXED, "
                "content UNINDEXED, contents, tokenize='porter unicode61 remove_diacritics 2')"
            )
        except sqlite3.OperationalError as exc:
            raise RuntimeError("this SQLite build does not provide FTS5") from exc
        connection.execute("CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("BEGIN")
        batch: list[tuple[str, str, str, str]] = []
        for source_path in sorted(corpus_dir.glob("*.jsonl")):
            with source_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise TypeError(f"{source_path}:{line_number} is not a JSON object")
                    fields = (row.get("id"), row.get("title"), row.get("content"), row.get("contents"))
                    if not all(isinstance(field, str) and field.strip() for field in fields):
                        raise ValueError(f"{source_path}:{line_number} has missing or empty required fields")
                    doc_id, title, content, contents = fields
                    if contents != f"{title}. {content}":
                        raise ValueError(f"{source_path}:{line_number} has inconsistent contents field")
                    if doc_id in seen_ids:
                        raise ValueError(f"duplicate corpus ID {doc_id!r}")
                    seen_ids.add(doc_id)
                    batch.append((doc_id, title, content, contents))
                    if len(batch) >= 1000:
                        connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", batch)
                        inserted += len(batch)
                        batch.clear()
        if batch:
            connection.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)", batch)
            inserted += len(batch)
        if inserted != expected_chunks:
            raise ValueError(f"corpus has {inserted} chunks; expected {expected_chunks}")
        connection.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            ("corpus_manifest", json.dumps(source_manifest, sort_keys=True)),
        )
        connection.execute(
            "INSERT INTO index_meta (key, value) VALUES (?, ?)",
            ("retriever", "sqlite_fts5_bm25_question_only_v1"),
        )
        connection.commit()
        connection.close()
        building_path.replace(index_path)
    except Exception:
        connection.close()
        for temp_path in (building_path, Path(str(building_path) + "-wal"), Path(str(building_path) + "-shm")):
            if temp_path.exists():
                temp_path.unlink()
        raise
    return {"index_path": str(index_path), "chunks": inserted, "reused": False}


def build_fts_query(question: str) -> str:
    terms = list(dict.fromkeys(term.lower() for term in TOKEN_RE.findall(question)))
    return " OR ".join(f'"{term}"' for term in terms)


def retrieve(connection: sqlite3.Connection, question: str, top_k: int) -> list[dict[str, Any]]:
    query = build_fts_query(question)
    if not query:
        return []
    rows = connection.execute(
        "SELECT id, title, content, contents, bm25(chunks) AS score "
        "FROM chunks WHERE contents MATCH ? ORDER BY score LIMIT ?",
        (query, top_k),
    )
    return [
        {"id": row[0], "title": row[1], "content": row[2], "contents": row[3], "bm25_score": row[4]}
        for row in rows
    ]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/raw/medrag_textbooks"))
    parser.add_argument("--index-path", type=Path, default=Path("data/processed/medrag_textbooks/fts5_bm25.sqlite3"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subdatasets", nargs="+", default=list(DEFAULT_SUBDATASETS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-per-subset", type=int)
    parser.add_argument("--expected-corpus-chunks", type=int, default=125847)
    args = parser.parse_args()
    if args.top_k < 1 or (args.limit_per_subset is not None and args.limit_per_subset < 1):
        parser.error("--top-k and --limit-per-subset must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is non-empty; choose a new directory: {args.output_dir}")

    started = time.perf_counter()
    index_info = build_or_validate_index(args.index_path, args.corpus_dir, args.expected_corpus_chunks)
    subsets = read_mirage_cases(args.benchmark_json, args.subdatasets, args.limit_per_subset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.index_path)
    all_latencies: list[float] = []
    subset_metrics: dict[str, Any] = {}
    case_count = 0
    try:
        for subset, cases in subsets.items():
            result_path = args.output_dir / f"{subset}.jsonl"
            results: list[dict[str, Any]] = []
            latencies: list[float] = []
            retrieved_total = 0
            for case in cases:
                query_started = time.perf_counter()
                evidence = retrieve(connection, case["question"], args.top_k)
                elapsed_ms = (time.perf_counter() - query_started) * 1000
                latencies.append(elapsed_ms)
                all_latencies.append(elapsed_ms)
                retrieved_total += len(evidence)
                result = {
                    **case,
                    "retrieved_evidence": evidence,
                    "retrieval_metadata": {
                        "retriever": "sqlite_fts5_bm25_question_only_v1",
                        "corpus_chunks": index_info["chunks"],
                        "top_k": args.top_k,
                        "query_protocol": "question_only",
                        "latency_ms": round(elapsed_ms, 3),
                    },
                }
                results.append(result)
            append_jsonl(result_path, results)
            count = len(cases)
            case_count += count
            subset_metrics[subset] = {
                "queries": count,
                "queries_with_results": sum(bool(row["retrieved_evidence"]) for row in results),
                "mean_retrieved_chunks": retrieved_total / count if count else None,
                "mean_latency_ms": sum(latencies) / count if count else None,
                "p50_latency_ms": percentile(latencies, 0.5),
                "p95_latency_ms": percentile(latencies, 0.95),
                "evidence_file": str(result_path),
            }
    finally:
        connection.close()

    config = {
        "benchmark": "MIRAGE Medical",
        "corpus": "MedRAG/Textbooks",
        "retriever": "SQLite FTS5 BM25 (porter unicode61), question-only query",
        "subdatasets": args.subdatasets,
        "top_k": args.top_k,
        "limit_per_subset": args.limit_per_subset,
        "index_path": str(args.index_path),
        "index_reused": index_info["reused"],
    }
    metrics = {
        "status": "RETRIEVAL_ONLY_COMPLETED",
        "case_count": case_count,
        "retrieved_evidence_coverage": sum(v["queries_with_results"] for v in subset_metrics.values()) / case_count if case_count else None,
        "mean_latency_ms": sum(all_latencies) / len(all_latencies) if all_latencies else None,
        "p50_latency_ms": percentile(all_latencies, 0.5),
        "p95_latency_ms": percentile(all_latencies, 0.95),
        "by_subdataset": subset_metrics,
        "answer_accuracy": None,
        "retrieval_recall": None,
        "limitation": "No answer generation or relevance qrels were run; these metrics do not establish answer quality or retrieval recall.",
    }
    write_json(args.output_dir / "run_config.json", config)
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "manifest.json", {
        "status": "RETRIEVAL_ONLY_COMPLETED",
        "benchmark": "MIRAGE Medical",
        "corpus": "MedRAG/Textbooks",
        "index": index_info,
        "case_count": case_count,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "llm_calls": 0,
        "answer_generation": False,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    })
    report = [
        "# MIRAGE × MedRAG Textbooks — BM25 retrieval-only run",
        "",
        f"- Cases retrieved: **{case_count}**",
        f"- Corpus chunks: **{index_info['chunks']}**",
        f"- Retriever: **{config['retriever']}**",
        f"- Top-k: **{args.top_k}**",
        f"- Evidence coverage: **{metrics['retrieved_evidence_coverage']:.1%}**" if case_count else "- Evidence coverage: **n/a**",
        f"- Mean retrieval latency: **{metrics['mean_latency_ms']:.2f} ms/query**" if case_count else "- Mean retrieval latency: **n/a**",
        "- LLM/API calls: **0**",
        "- Answer accuracy and retrieval recall: **not measured** (no generation and no relevance qrels).",
        "",
        "Per-subdataset metrics are in `metrics.json`. The JSONL files contain the original MIRAGE fields plus top-k evidence and can be used by a separately configured answer runner.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": metrics["status"], "cases": case_count, "chunks": index_info["chunks"], "index_reused": index_info["reused"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
