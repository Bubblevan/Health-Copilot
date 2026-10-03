"""Shared paths and report formatting for Huiyi command-line tools."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "huiyi"
RUN_ROOT = REPO_ROOT / "runs" / "huiyi" / "hy-data-0"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def copy_if_exists(source: Path, target: Path) -> None:
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def render_summary() -> str:
    source = _json_or_empty(RUN_ROOT / "source_report.json")
    normalization = _json_or_empty(RUN_ROOT / "normalization_report.json")
    dedup = _json_or_empty(RUN_ROOT / "dedup_report.json")
    chunk = _json_or_empty(RUN_ROOT / "chunk_report.json")
    embedding = _json_or_empty(RUN_ROOT / "embedding_report.json")
    milvus = _json_or_empty(RUN_ROOT / "milvus_report.json")
    build = _json_or_empty(RUN_ROOT / "build_report.json")
    retrieval = _json_or_empty(RUN_ROOT / "retrieval_smoke.json")
    retrieval_metrics = retrieval.get("metrics", {})
    table_rows = []
    for name, label in (("bm25", "BM25"), ("dense", "Qwen3 Dense"), ("hybrid", "Hybrid RRF")):
        metric = retrieval_metrics.get(name)
        if metric:
            table_rows.append(
                f"| {label} | {metric.get('hit_at_1', 0):.3f} | {metric.get('hit_at_3', 0):.3f} | "
                f"{metric.get('hit_at_5', 0):.3f} | {metric.get('mrr_at_5', 0):.3f} | "
                f"{metric.get('latency_p50_ms', 0):.1f} | {metric.get('latency_p95_ms', 0):.1f} |"
            )
        else:
            table_rows.append(f"| {label} | pending | pending | pending | pending | pending | pending |")
    document_types = normalization.get("document_types", {})
    freshness = normalization.get("freshness_classes", {})
    return "\n".join([
        "# HY-DATA-0 Summary",
        "",
        "**HY-DATA-0 SMOKE · NOT FROZEN BENCHMARK · NOT CLINICAL ACCURACY**",
        "",
        f"- Corpus identity: `{build.get('corpus_identity_sha256', 'pending')}`; index identity: `{build.get('index_identity_sha256', 'pending')}`.",
        "",
        "## Data quality",
        "",
        f"- URLs discovered / fetched / failed: {source.get('discovered_urls', 0)} / {source.get('fetched_successfully', 0)} / {source.get('failed', 0)}.",
        f"- Raw snapshots: {source.get('fetched_successfully', 0)}; raw-only dynamic pages: {source.get('raw_only_dynamic', 0)}.",
        f"- Canonical documents / rejected: {normalization.get('document_count', 0)} / {normalization.get('rejected_count', 0)}.",
        f"- Document types: {json.dumps(document_types, ensure_ascii=False, sort_keys=True)}.",
        f"- Freshness classes: {json.dumps(freshness, ensure_ascii=False, sort_keys=True)}.",
        f"- Exact duplicate documents / near-duplicate flags: {dedup.get('exact_duplicate_documents', 0)} / {dedup.get('near_duplicate_flag_count', 0)}.",
        f"- Chunks: {chunk.get('chunk_count', 0)}; median chars/tokens {chunk.get('median_chars', 0)} / {chunk.get('median_tokens', 0)}; p95 chars/tokens {chunk.get('p95_chars', 0)} / {chunk.get('p95_tokens', 0)}.",
        f"- Embeddings successful / failed: {embedding.get('embedding_successes', 0)} / {embedding.get('embedding_failures', 0)}; dimension {embedding.get('embedding_dimension', 'pending')}.",
        f"- Milvus collection / entities: {milvus.get('collection_name', 'pending')} / {milvus.get('entity_count', 'pending')}.",
        "",
        "## Retrieval smoke",
        "",
        "| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@5 | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        f"- Smoke queries: {retrieval.get('query_count', 0)}.",
        f"- Metadata filter smoke: {milvus.get('metadata_filter_smoke', 'pending')}.",
        "- The evaluation set is a retrieval sanity check, not a frozen benchmark, clinical accuracy measure, or production metric.",
        "",
    ])


def update_summary() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "summary.md").write_text(render_summary(), encoding="utf-8", newline="\n")


def _json_or_empty(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
