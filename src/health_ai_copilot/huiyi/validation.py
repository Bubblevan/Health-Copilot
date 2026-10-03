"""Corpus and provenance invariants for build and smoke commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import CORPUS_SCHEMA_VERSION
from .chunk import CHUNKER_VERSION
from .normalize import NORMALIZER_VERSION
from .schema import canonical_json_sha256, sha256_bytes


def validate_corpus(data_root: Path) -> dict[str, Any]:
    catalog_path = data_root / "source_catalog.json"
    raw_manifest_path = data_root / "raw" / "manifest.jsonl"
    documents_path = data_root / "normalized" / "documents.jsonl"
    chunks_path = data_root / "chunks" / "chunks.jsonl"
    for path in (catalog_path, raw_manifest_path, documents_path, chunks_path):
        if not path.is_file():
            raise FileNotFoundError(f"required Huiyi corpus artifact missing: {path}")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    source_ids = [row["source_id"] for row in catalog]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source catalog has duplicate source IDs")
    sources_by_id = {row["source_id"]: row for row in catalog}
    documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    doc_ids = [row["document_id"] for row in documents]
    chunk_ids = [row["chunk_id"] for row in chunks]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("canonical document IDs are not unique")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk IDs are not unique")
    docs_by_id = {row["document_id"]: row for row in documents}
    for document in documents:
        if document["source_id"] not in sources_by_id:
            raise ValueError(f"document refers to unknown source ID {document['source_id']}")
        if document["ingestion_policy"] != "canonical":
            raise ValueError(f"non-canonical document entered canonical artifacts: {document['document_id']}")
        if hashlib.sha256(document["content"].encode("utf-8")).hexdigest() != document["content_sha256"]:
            raise ValueError(f"document content hash mismatch: {document['document_id']}")
        topics = document.get("topics")
        if not isinstance(topics, list) or any(not isinstance(topic, str) or not topic for topic in topics):
            raise ValueError(f"document topics must be a list of non-empty strings: {document['document_id']}")
        if len(topics) > 32 or any(len(topic) > 128 for topic in topics):
            raise ValueError(f"document topics exceed the Milvus field limits: {document['document_id']}")
        if len(topics) != len(set(topics)):
            raise ValueError(f"document topics contain duplicates: {document['document_id']}")
        primary_topic = document.get("primary_topic")
        if primary_topic is not None and (not isinstance(primary_topic, str) or not primary_topic):
            raise ValueError(f"document primary topic must be a non-empty string or null: {document['document_id']}")
        if primary_topic is not None and primary_topic not in topics:
            raise ValueError(f"primary topic is missing from topics: {document['document_id']}")
        if document["document_type"] == "hospital_info" and (document.get("primary_topic") is not None or topics):
            raise ValueError(f"hospital-info documents must remain untagged: {document['document_id']}")
    for chunk in chunks:
        parent = docs_by_id.get(chunk["document_id"])
        if parent is None:
            raise ValueError(f"chunk has no canonical document: {chunk['chunk_id']}")
        if parent["source_id"] != chunk["source_id"] or parent["source_url"] != chunk["source_url"]:
            raise ValueError(f"chunk provenance does not match document: {chunk['chunk_id']}")
        if chunk.get("primary_topic") != parent.get("primary_topic") or chunk.get("topics") != parent.get("topics"):
            raise ValueError(f"chunk topic metadata does not match document: {chunk['chunk_id']}")
        if chunk["freshness_class"] == "DYNAMIC":
            raise ValueError(f"dynamic content entered static chunks: {chunk['chunk_id']}")
    raw_rows = [json.loads(line) for line in raw_manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for raw in raw_rows:
        raw_path = data_root / Path(raw["raw_path"])
        if not raw_path.is_file() or sha256_bytes(raw_path.read_bytes()) != raw["raw_sha256"]:
            raise ValueError(f"raw source snapshot missing or hash mismatch: {raw['url']}")
    source_catalog_sha256 = sha256_bytes(catalog_path.read_bytes())
    raw_manifest_sha256 = sha256_bytes(raw_manifest_path.read_bytes())
    documents_jsonl_sha256 = sha256_bytes(documents_path.read_bytes())
    chunks_jsonl_sha256 = sha256_bytes(chunks_path.read_bytes())
    identity = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "source_catalog_sha256": source_catalog_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "documents_sha256": documents_jsonl_sha256,
        "chunks_sha256": chunks_jsonl_sha256,
        "normalizer_version": NORMALIZER_VERSION,
        "chunker_version": CHUNKER_VERSION,
    }
    return {
        **identity,
        "corpus_identity_sha256": canonical_json_sha256(identity),
        "documents_jsonl_sha256": documents_jsonl_sha256,
        "chunks_jsonl_sha256": chunks_jsonl_sha256,
        "source_count": len(catalog),
        "raw_snapshot_count": len(raw_rows),
        "document_count": len(documents),
        "chunk_count": len(chunks),
    }
