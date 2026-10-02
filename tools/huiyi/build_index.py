"""Embed the canonical Huiyi chunks locally and insert them into Milvus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from _common import DATA_ROOT, REPO_ROOT, RUN_ROOT, load_jsonl, update_summary, write_json

sys.path.insert(0, str(REPO_ROOT / "src"))

from health_ai_copilot.huiyi.bm25 import HuiyiBM25
from health_ai_copilot.huiyi.chunk import CHUNKER_VERSION
from health_ai_copilot.huiyi.embed import (
    DEFAULT_MODEL_ROOT,
    LocalModelUnavailable,
    Qwen3LocalEmbedder,
)
from health_ai_copilot.huiyi.milvus_store import COLLECTION_NAME, HuiyiMilvusStore
from health_ai_copilot.huiyi.schema import canonical_json_sha256, sha256_bytes
from health_ai_copilot.huiyi.validation import validate_corpus


def _save_vectors(path: Path, chunks: list[dict[str, Any]], vectors: list[list[float]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for chunk, vector in zip(chunks, vectors, strict=True):
            row = {"chunk_id": chunk["chunk_id"], "vector": vector}
            handle.write((json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--milvus-batch-size", type=int, default=256)
    args = parser.parse_args()

    corpus = validate_corpus(DATA_ROOT)
    chunks = load_jsonl(DATA_ROOT / "chunks" / "chunks.jsonl")
    if not chunks:
        raise SystemExit("No canonical chunks found; run tools/huiyi/build_corpus.py first.")
    try:
        embedder = Qwen3LocalEmbedder(args.model_root)
    except LocalModelUnavailable as exc:
        raise SystemExit(str(exc)) from exc
    inputs = [embedder.document_input(chunk) for chunk in chunks]
    vectors = embedder.encode_documents(inputs, batch_size=args.batch_size)
    if len(vectors) != len(chunks):
        raise SystemExit(f"Embedding count {len(vectors)} does not match chunk count {len(chunks)}")

    vector_path = REPO_ROOT / "artifacts" / "huiyi" / "hy-data-0" / "embeddings.jsonl"
    vector_sha = _save_vectors(vector_path, chunks, vectors)
    embedding_manifest = {
        **embedder.manifest(batch_size=args.batch_size),
        "chunk_count": len(chunks),
        "embedding_successes": len(vectors),
        "embedding_failures": 0,
        "chunks_jsonl_sha256": corpus["chunks_jsonl_sha256"],
        "vectors_path": vector_path.relative_to(REPO_ROOT).as_posix(),
        "vectors_sha256": vector_sha,
    }
    write_json(DATA_ROOT / "index" / "embedding_manifest.json", embedding_manifest)
    write_json(RUN_ROOT / "embedding_report.json", {
        "model_id": embedding_manifest["model_id"],
        "resolved_revision": embedding_manifest["resolved_revision"],
        "embedding_dimension": embedding_manifest["embedding_dimension"],
        "dtype": embedding_manifest["dtype"],
        "device": embedding_manifest["device"],
        "embedding_successes": len(vectors),
        "embedding_failures": 0,
        "vectors_sha256": vector_sha,
    })

    bm25 = HuiyiBM25(chunks)
    bm25_manifest = bm25.manifest(corpus_sha256=corpus["chunks_jsonl_sha256"])
    write_json(DATA_ROOT / "index" / "bm25_manifest.json", bm25_manifest)

    store = HuiyiMilvusStore(args.milvus_uri, collection_name=COLLECTION_NAME)
    store.create_collection()
    inserted = store.insert(chunks, vectors, batch_size=args.milvus_batch_size)
    entity_count = store.entity_count()
    if inserted != len(chunks) or entity_count != len(chunks):
        raise SystemExit(
            f"Milvus entity mismatch: inserted={inserted}, entity_count={entity_count}, expected={len(chunks)}"
        )
    milvus_manifest = {
        "collection_name": COLLECTION_NAME,
        "milvus_version": store.server_version,
        "pymilvus_version": store.pymilvus_version,
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "index_params": {"M": 16, "efConstruction": 128},
        "search_params": {"ef": 64},
        "collection_schema": HuiyiMilvusStore.schema_manifest(),
        "entity_count": entity_count,
        "chunk_count": len(chunks),
        "chunks_jsonl_sha256": corpus["chunks_jsonl_sha256"],
        "uri": args.milvus_uri,
    }
    write_json(DATA_ROOT / "index" / "milvus_manifest.json", milvus_manifest)
    write_json(RUN_ROOT / "milvus_report.json", {
        **milvus_manifest,
        "metadata_filter_smoke": "pending retrieval smoke",
    })

    embedding_manifest_sha = sha256_bytes((DATA_ROOT / "index" / "embedding_manifest.json").read_bytes())
    identity = {
        "schema_version": corpus.get("schema_version", "huiyi-knowledge-corpus-v0"),
        "source_catalog_sha256": corpus["source_catalog_sha256"],
        "raw_manifest_sha256": corpus["raw_manifest_sha256"],
        "documents_sha256": corpus["documents_jsonl_sha256"],
        "chunks_sha256": corpus["chunks_jsonl_sha256"],
        "embedding_manifest_sha256": embedding_manifest_sha,
        "embedding_model_identity": {
            "model_id": embedding_manifest["model_id"],
            "resolved_revision": embedding_manifest["resolved_revision"],
        },
        "chunker_version": CHUNKER_VERSION,
    }
    corpus_manifest = {**corpus, "embedding_manifest_sha256": embedding_manifest_sha, **identity}
    corpus_manifest["corpus_identity_sha256"] = canonical_json_sha256(identity)
    write_json(DATA_ROOT / "index" / "corpus_manifest.json", corpus_manifest)
    write_json(RUN_ROOT / "build_report.json", {
        "pipeline": "HY-DATA-0",
        "status": "indexed",
        "corpus": corpus,
        "embedding_count": len(vectors),
        "milvus_entity_count": entity_count,
        "corpus_identity_sha256": corpus_manifest["corpus_identity_sha256"],
        "index_status": "complete",
    })
    update_summary()
    print(json.dumps({"embedding_manifest": embedding_manifest, "milvus": milvus_manifest, "corpus_identity_sha256": corpus_manifest["corpus_identity_sha256"]}, ensure_ascii=False, indent=2, sort_keys=True))
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
