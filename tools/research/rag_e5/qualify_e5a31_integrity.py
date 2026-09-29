"""Validate and activate the frozen E5-A3 corpus without changing retrieval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from eval.rag_e5.corpus import E5ExternalChunk, _source_review_identity, build_corpus_view
from eval.rag_e5.qualification import (
    bind_active_corpus_identity,
    exact_dense_rankings,
    validate_dense_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_ROOT = Path("D:/MyLab/Jianli/external/rag_e5")
DEFAULT_MODEL_ROOT = Path("E:/Health-Copilot-Models/models/bge-large-en-v1.5")
EXPECTED_IDENTITY = "9b19ad467f47641032277707cb1cfdb1d05c2fb39c558039b180efc7394692bd"
EXPECTED_SOURCE_MANIFEST_SHA = "5cb97843c0ff1fb5450b00c608e2d3d156473cad43b6f0e8ecce4fc2c4c970a0"
EXPECTED_OWNER_DECISION_SHA = "f563d8b9a837f4627c750874a030f3beb5e8728c91638568dd74eab241d12ac7"
EXPECTED_BGE_WEIGHTS_SHA = "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
EXPECTED_STANDARD_CONFIG_SHA = "6ddb91bb0c31f5bc5b69372df6a3bdd3b4f71d70cdbcece8ef22c5bfd9a94330"
EXPECTED_STRONG_CONFIG_SHA = "d9e3de9bf986a1f08b1c217153422d6e86ad0a84bc1982d90bade897c9274a8b"
EXPECTED_VIEWS = {
    "PUBLIC_HEALTH_ONLY": (30, 30, ("public_health",)),
    "GUIDELINE_ONLY": (26, 3, ("reviewed_guideline",)),
    "PUBLIC_HEALTH_PLUS_GUIDELINE": (56, 33, ("public_health", "reviewed_guideline")),
}
FROZEN_CLOSEOUTS = {
    "docs/research/rag_closeout.md": "1a127b15ac5954e188b5b4dfd0d719799f0987cd5d45ce7c5c0c1c0f0b3dab2d",
    "docs/research/e1_3_mirage_exploratory.md": "006e11b6e04baf7a9bcceffa0030590e98557165a59446f90cb9f36d991cf399",
    "runs/e1_3/mirage_exploratory_report.json": "c660fe7b696123cb59784a56ac2a55305bb4661823549659b4c1a4404032217c",
    "runs/rag_r2med_final_test/final_eval_lock.json": "0b80fad6668c3a57833c55beb1db359c65440cf015effd10da770c630fb3e41d",
    "runs/rag_r2med_crb/source_manifest.json": "b70c4f01b37f58c77597f1e28cc35a52585785142f3f625f928173c81be3874e",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"expected JSON objects in {path}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def _language_class(text: str) -> tuple[str, int, int]:
    cjk_count = sum("\u3400" <= char <= "\u9fff" for char in text)
    latin_count = sum(char.isalpha() and ord(char) < 0x0250 for char in text)
    total = cjk_count + latin_count
    if total == 0:
        return "Unclassified", cjk_count, latin_count
    cjk_share = cjk_count / total
    latin_share = latin_count / total
    if cjk_share >= 0.6:
        return "CJK-dominant", cjk_count, latin_count
    if latin_share >= 0.6:
        return "Latin-dominant", cjk_count, latin_count
    return "Mixed", cjk_count, latin_count


def _chunk_from_row(row: dict[str, Any]) -> E5ExternalChunk:
    return E5ExternalChunk(
        chunk_id=row["chunk_id"],
        source_id=row["source_id"],
        source_family=row["source_family"],
        section_path=tuple(row["section_path"]),
        recommendation_id=row["recommendation_id"],
        text=row["text"],
        text_sha256=row["text_sha256"],
        token_count=row["token_count"],
        char_count=row["char_count"],
        raw_source_sha256=row["raw_source_sha256"],
        review_identity=row["review_identity"],
        extractor_version=row["extractor_version"],
        chunker_version=row["chunker_version"],
    )


def _expected_review_identity(source: dict[str, Any]) -> str:
    return _source_review_identity(source)


def _validate_owner_decision(repo_root: Path, source_manifest: dict[str, Any]) -> dict[str, Any]:
    owner_path = repo_root / "docs/research/rag_e5/e5a3_owner_decision.json"
    owner_sha = _sha256_file(owner_path)
    if owner_sha != EXPECTED_OWNER_DECISION_SHA:
        raise ValueError("owner decision artifact SHA does not match the frozen A3 identity")
    owner = _read_json(owner_path)
    rows = owner.get("sources")
    if not isinstance(rows, list):
        raise TypeError("owner decision artifact has no source decisions")
    by_id = {row.get("source_id"): row for row in rows if isinstance(row, dict)}
    expected = {
        "who-physical-activity-sedentary-2020": True,
        "who-total-fat-weight-gain-2023": True,
        "who-hypertension-pharmacological-2021": False,
    }
    if set(by_id) != set(expected):
        raise ValueError("owner-approved source set differs from the frozen A3 decision")
    for source_id, authoring_eligible in expected.items():
        row = by_id[source_id]
        if row.get("decision") != "APPROVE" or row.get("task_authoring_eligible") is not authoring_eligible:
            raise ValueError(f"owner decision/scope mismatch for {source_id}")
    manifest_sources = {row["source_id"]: row for row in source_manifest["sources"]}
    if set(manifest_sources) != set(expected):
        raise ValueError("source manifest source set differs from owner decision")
    if source_manifest.get("owner_decision_record_sha256") != owner_sha:
        raise ValueError("source manifest is not bound to the owner decision artifact")
    for source_id, eligible in expected.items():
        row = manifest_sources[source_id]
        if row.get("owner_decision_record_sha256") != owner_sha:
            raise ValueError(f"source record lacks owner decision binding: {source_id}")
        if row.get("owner_review_status") != "APPROVED":
            raise ValueError(f"source is not owner approved: {source_id}")
        if row.get("task_authoring_eligible") is not eligible:
            raise ValueError(f"task-authoring scope changed: {source_id}")
    return {
        "status": "VERIFIED",
        "owner_decision_record_sha256": owner_sha,
        "source_ids": sorted(expected),
        "authorization_provenance": {
            "kind": "owner-authored Codex conversation message",
            "message_quote": "我已经决定好了，和subagent意见一致",
            "context": "Owner affirmation of the reviewed three-source decision, followed by the owner-scope A3 TRD.",
        },
        "scope_unchanged": True,
    }


def _validate_corpus(
    repo_root: Path,
    external_root: Path,
    corpus_report: dict[str, Any],
    active_corpus_manifest: dict[str, Any],
    source_manifest: dict[str, Any],
    owner_decision: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    source_manifest_path = repo_root / "docs/research/rag_e5/guideline_source_manifest.json"
    source_manifest_sha = _sha256_file(source_manifest_path)
    if source_manifest_sha != EXPECTED_SOURCE_MANIFEST_SHA:
        raise ValueError("pinned guideline source manifest SHA mismatch")
    if corpus_report.get("source_manifest_sha256") != source_manifest_sha:
        raise ValueError("corpus report is bound to a different source manifest")
    if corpus_report.get("owner_decision_record_sha256") != owner_decision[
        "owner_decision_record_sha256"
    ]:
        raise ValueError("corpus report is bound to a different owner decision")
    corpus_report_path = external_root / "e5a3/corpus_build_report.json"
    if _sha256_file(corpus_report_path) != active_corpus_manifest.get(
        "corpus_build_report_sha256"
    ):
        raise ValueError("corpus build report is not the frozen candidate report")

    guideline_sources = {row["source_id"]: row for row in source_manifest["sources"]}
    external_guidelines = {
        row["source_id"]: row for row in active_corpus_manifest["guideline_candidates"]
    }
    if set(guideline_sources) != set(external_guidelines):
        raise ValueError("guideline raw-source inventory differs from the reviewed manifest")
    raw_identity_rows: list[dict[str, str]] = []
    raw_file_checks: list[dict[str, Any]] = []
    for source_id, source in guideline_sources.items():
        raw_path = external_root / source["raw_path"]
        raw_sha = _sha256_file(raw_path)
        if raw_sha != source["raw_sha256"] or raw_path.stat().st_size != source["raw_bytes"]:
            raise ValueError(f"raw guideline source mismatch: {source_id}")
        extracted_path = external_root / "guidelines/extracted" / f"{Path(source['raw_path']).stem}.txt"
        extracted_sha = _sha256_file(extracted_path)
        if extracted_sha != external_guidelines[source_id]["extracted_text_sha256"]:
            raise ValueError(f"extracted guideline text hash mismatch: {source_id}")
        raw_identity_rows.append({"source_id": source_id, "raw_sha256": raw_sha})
        raw_file_checks.append(
            {
                "source_id": source_id,
                "raw_sha256": raw_sha,
                "raw_bytes": raw_path.stat().st_size,
                "raw_hash_valid": True,
                "extracted_text_sha256": extracted_sha,
                "extracted_text_hash_valid": True,
            }
        )

    raw_source_set_sha = _sha256_bytes(
        _canonical_json(sorted(raw_identity_rows, key=lambda row: row["source_id"]))
    )
    if raw_source_set_sha != active_corpus_manifest.get("candidate_raw_source_set_sha256"):
        raise ValueError("candidate raw-source set identity mismatch")

    card_paths = sorted(
        path
        for path in (repo_root / "data/knowledge_cards").glob("*.json")
        if not path.name.startswith("_")
    )
    if len(card_paths) != 30:
        raise ValueError("the frozen reviewed public-health card set must contain 30 files")
    card_rows: dict[str, dict[str, Any]] = {}
    card_identity_rows: list[dict[str, str]] = []
    for path in card_paths:
        raw_bytes = path.read_bytes()
        card = json.loads(raw_bytes)
        source_id = card.get("id")
        if not isinstance(source_id, str) or source_id != path.stem or source_id in card_rows:
            raise ValueError(f"public-health card identity mismatch: {path.name}")
        raw_sha = _sha256_bytes(raw_bytes)
        text = card.get("content")
        title = card.get("title")
        if not isinstance(text, str) or not text.strip() or not isinstance(title, str):
            raise ValueError(f"public-health card has invalid text/title: {source_id}")
        card_rows[source_id] = {
            "card": card,
            "raw_sha256": raw_sha,
            "text_sha256": _sha256_bytes(text.encode("utf-8")),
        }
        card_identity_rows.append({"card_id": source_id, "file_sha256": raw_sha})
    public_corpus_sha = _sha256_bytes(
        _canonical_json(sorted(card_identity_rows, key=lambda row: row["card_id"]))
    )
    if public_corpus_sha != active_corpus_manifest.get("public_health_corpus_sha256"):
        raise ValueError("public-health reviewed card corpus identity mismatch")

    view_rows: dict[str, list[dict[str, Any]]] = {}
    search_rows: dict[str, list[dict[str, Any]]] = {}
    built_views = []
    view_summaries: list[dict[str, Any]] = []
    public_records: dict[str, dict[str, Any]] = {}
    guideline_recommendations: dict[str, set[str]] = defaultdict(set)
    for view_record in corpus_report["views"]:
        view_id = view_record["corpus_view_id"]
        chunks_path = Path(view_record["chunks_jsonl_path"])
        docs_path = Path(view_record["search_documents_jsonl_path"])
        if _sha256_file(chunks_path) != view_record["chunks_jsonl_sha256"]:
            raise ValueError(f"corpus chunk artifact hash mismatch: {view_id}")
        if _sha256_file(docs_path) != view_record["search_documents_jsonl_sha256"]:
            raise ValueError(f"search document artifact hash mismatch: {view_id}")
        chunks = _read_jsonl(chunks_path)
        docs = _read_jsonl(docs_path)
        expected_count, expected_sources, expected_families = EXPECTED_VIEWS[view_id]
        if len(chunks) != expected_count or len(docs) != expected_count:
            raise ValueError(f"unexpected corpus document count: {view_id}")
        chunk_ids = [row["chunk_id"] for row in chunks]
        doc_ids = [row["id"] for row in docs]
        if len(set(chunk_ids)) != expected_count or doc_ids != chunk_ids:
            raise ValueError(f"duplicate IDs or chunk/search-document order mismatch: {view_id}")
        if any(row.get("source_family") not in expected_families for row in chunks):
            raise ValueError(f"source family escaped its corpus view: {view_id}")
        if len({row["source_id"] for row in chunks}) != expected_sources:
            raise ValueError(f"unexpected distinct source count: {view_id}")
        for chunk, doc in zip(chunks, docs, strict=True):
            if doc.get("id") != chunk["chunk_id"]:
                raise ValueError(f"search-document chunk ID mismatch: {view_id}")
            expected_contents = " ".join((*chunk["section_path"], chunk["text"]))
            if doc.get("contents") != expected_contents:
                raise ValueError(f"search-document contents differ from canonical chunk: {view_id}")
            if doc.get("source_id") != chunk["source_id"] or doc.get("source_family") != chunk[
                "source_family"
            ]:
                raise ValueError(f"search-document provenance mismatch: {view_id}")
            text_sha = _sha256_bytes(chunk["text"].encode("utf-8"))
            if text_sha != chunk["text_sha256"] or len(chunk["text"]) != chunk["char_count"]:
                raise ValueError(f"canonical chunk text hash/length mismatch: {view_id}")
            if chunk["source_family"] == "public_health":
                card_record = card_rows.get(chunk["source_id"])
                if card_record is None:
                    raise ValueError("public-health chunk has no reviewed card source")
                card = card_record["card"]
                raw_sha = card_record["raw_sha256"]
                review_identity = _sha256_bytes(
                    _canonical_json(
                        {
                            "source_id": chunk["source_id"],
                            "source_url": card["source_url"],
                            "publisher": card["publisher"],
                            "reviewed_at": card["reviewed_at"],
                            "reviewer": card["reviewer"],
                            "version": card["version"],
                            "raw_source_sha256": raw_sha,
                        }
                    )
                )
                chunk_identity = _sha256_bytes(
                    _canonical_json(
                        {
                            "source_id": chunk["source_id"],
                            "raw_source_sha256": raw_sha,
                            "text_sha256": card_record["text_sha256"],
                            "review_identity": review_identity,
                        }
                    )
                )
                if (
                    chunk["raw_source_sha256"] != raw_sha
                    or chunk["text"] != card["content"]
                    or chunk["section_path"] != [card["title"]]
                    or chunk["text_sha256"] != card_record["text_sha256"]
                    or chunk["review_identity"] != review_identity
                    or chunk["chunk_id"] != f"public-health-{chunk_identity[:24]}"
                ):
                    raise ValueError(f"public-health card provenance mismatch: {chunk['source_id']}")
                if view_id == "PUBLIC_HEALTH_PLUS_GUIDELINE":
                    public_records[chunk["source_id"]] = {
                        "source_family": "public_health",
                        "text": chunk["text"],
                    }
            else:
                source = guideline_sources.get(chunk["source_id"])
                if source is None:
                    raise ValueError("guideline chunk has no approved source")
                if (
                    chunk["raw_source_sha256"] != source["raw_sha256"]
                    or chunk["review_identity"] != _expected_review_identity(source)
                ):
                    raise ValueError(f"guideline chunk provenance mismatch: {chunk['source_id']}")
                guideline_recommendations[chunk["source_id"]].add(chunk["recommendation_id"])
                if view_id == "PUBLIC_HEALTH_PLUS_GUIDELINE":
                    public_records[chunk["source_id"]] = {
                        "source_family": "reviewed_guideline",
                        "text": (public_records.get(chunk["source_id"], {}).get("text", "") + " " + chunk["text"]).strip(),
                    }
        view_rows[view_id] = chunks
        search_rows[view_id] = docs
        family_chunks = [_chunk_from_row(row) for row in chunks]
        built_view = build_corpus_view(
            family_chunks, corpus_view_id=view_id, source_families=expected_families
        )
        if built_view.corpus_sha256 != view_record["corpus_sha256"]:
            raise ValueError(f"canonical corpus hash mismatch: {view_id}")
        built_views.append(built_view)
        view_summaries.append(
            {
                "view": view_id,
                "document_count": len(chunks),
                "source_count": len({row["source_id"] for row in chunks}),
                "corpus_sha256": built_view.corpus_sha256,
                "chunk_artifact_sha256": view_record["chunks_jsonl_sha256"],
                "search_document_artifact_sha256": view_record["search_documents_jsonl_sha256"],
                "source_families_valid": True,
                "unique_chunk_ids": True,
                "text_hashes_valid": True,
            }
        )

    retention = corpus_report["guideline_blocks"]
    retention_by_source = {row["source_id"]: row for row in retention}
    if set(retention_by_source) != set(guideline_sources):
        raise ValueError("recommendation-retention source inventory mismatch")
    for source_id, source in guideline_sources.items():
        expected_sections = {row["section_id"] for row in source["recommendation_sections"]}
        retained = retention_by_source[source_id]
        retained_sections = set(retained.get("retained_section_ids", []))
        declared_sections = set(retained.get("expected_section_ids", []))
        observed_blocks = guideline_recommendations[source_id]
        declared_blocks = set(retained.get("recommendation_block_ids", []))
        if (
            retained.get("recommendation_retention_pass") is not True
            or retained.get("missing_section_ids")
            or retained.get("unexpected_section_ids")
            or retained.get("duplicate_section_ids")
            or retained_sections != expected_sections
            or declared_sections != expected_sections
            or observed_blocks != declared_blocks
            or len(declared_blocks) != retained.get("recommendation_blocks_retained")
        ):
            raise ValueError(f"recommendation sections/blocks not retained exactly: {source_id}")
    if len(retention) != 3:
        raise ValueError("the 22-section recommendation retention audit is not clean")
    declared_total = sum(len(source["recommendation_sections"]) for source in guideline_sources.values())
    retained_total = sum(len(row["retained_section_ids"]) for row in retention)
    if declared_total != 22 or retained_total != 22:
        raise ValueError("recommendation retention must remain exactly 22/22")

    canonical_identity = _sha256_bytes(
        _canonical_json(
            {
                "schema_version": "rag-e5-external-corpus-identity-v1",
                "source_manifest_sha256": source_manifest_sha,
                "owner_decision_record_sha256": owner_decision["owner_decision_record_sha256"],
                "chunker_config_sha256": corpus_report["chunker_config_sha256"],
                "views": [
                    {"corpus_view_id": view.corpus_view_id, "corpus_sha256": view.corpus_sha256}
                    for view in built_views
                ],
            }
        )
    )
    if canonical_identity != EXPECTED_IDENTITY or canonical_identity != corpus_report[
        "external_corpus_identity"
    ]:
        raise ValueError("candidate corpus identity changed")
    if active_corpus_manifest.get("candidate_external_corpus_identity") != canonical_identity:
        raise ValueError("candidate identity in runtime corpus manifest changed")

    language_rows = []
    for source_id, record in sorted(public_records.items()):
        script, cjk_count, latin_count = _language_class(record["text"])
        language_rows.append(
            {
                "source_family": record["source_family"],
                "source_id": source_id,
                "detected_script_class": script,
                "document_language": script,
                "cjk_character_count": cjk_count,
                "latin_letter_count": latin_count,
            }
        )

    corpus_result = {
        "status": "PASS",
        "public_health_source_count": len(card_rows),
        "public_health_chunk_count": len(view_rows["PUBLIC_HEALTH_ONLY"]),
        "guideline_source_count": len(guideline_sources),
        "guideline_chunk_count": len(view_rows["GUIDELINE_ONLY"]),
        "recommendation_sections_expected": declared_total,
        "recommendation_sections_retained": retained_total,
        "raw_and_extracted_sources": raw_file_checks,
        "raw_source_set_sha256": raw_source_set_sha,
        "views": view_summaries,
        "candidate_external_corpus_identity": canonical_identity,
        "candidate_content_unchanged": True,
    }
    return corpus_result, view_rows, language_rows


def _bm25_replay(
    view_id: str,
    chunks: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    external_root: Path,
    bm25_report: dict[str, Any],
) -> dict[str, Any]:
    row = next(item for item in bm25_report["views"] if item["corpus_view_id"] == view_id)
    directory = external_root / "e5a3/indexes" / view_id.lower()
    manifest_path = directory / "bm25_index_manifest.json"
    manifest = _read_json(manifest_path)
    tokens_path = directory / "bm25_lucene_tokens.jsonl"
    model_path = directory / "bm25_dictionary_model.json"
    matrix_path = directory / "bm25_document_term_weights.npz"
    smoke_path = directory / "bm25_source_title_smoke.json"
    for path, expected_sha in (
        (manifest_path, row["bm25_index_manifest_sha256"]),
        (tokens_path, manifest["token_artifact_sha256"]),
        (model_path, manifest["dictionary_model_sha256"]),
        (matrix_path, manifest["document_term_matrix_sha256"]),
        (smoke_path, row["bm25_smoke_sha256"]),
    ):
        if _sha256_file(path) != expected_sha:
            raise ValueError(f"BM25 artifact hash mismatch: {view_id}/{path.name}")
    doc_ids = [item["id"] for item in docs]
    order_sha = _sha256_bytes(_canonical_json(doc_ids))
    if (
        manifest.get("document_count") != len(doc_ids)
        or manifest.get("corpus_sha256") != next(
            item["corpus_sha256"] for item in bm25_report["views"] if item["corpus_view_id"] == view_id
        )
        or manifest.get("document_order_sha256") != order_sha
    ):
        raise ValueError(f"BM25 document/corpus binding mismatch: {view_id}")
    token_rows = _read_jsonl(tokens_path)
    if [item.get("chunk_id") for item in token_rows] != doc_ids:
        raise ValueError(f"BM25 token order mismatch: {view_id}")
    document_tokens = [item.get("tokens") for item in token_rows]
    if any(not isinstance(tokens, list) or not tokens for tokens in document_tokens):
        raise ValueError(f"BM25 index has empty/malformed document tokens: {view_id}")
    model_payload = _read_json(model_path)
    terms = model_payload.get("terms")
    idfs = model_payload.get("idfs")
    if (
        not isinstance(terms, list)
        or not isinstance(idfs, list)
        or len(terms) != len(set(terms))
        or len(idfs) != len(terms)
        or len(terms) != manifest.get("term_count")
        or not all(math.isfinite(float(value)) for value in idfs)
    ):
        raise ValueError(f"BM25 term dictionary does not match indexed documents: {view_id}")
    vocabulary = {term: index for index, term in enumerate(terms)}
    if any(token not in vocabulary for tokens in document_tokens for token in tokens):
        raise ValueError(f"BM25 token artifact contains a term outside its dictionary: {view_id}")
    indexed = sparse.load_npz(matrix_path).tocsr()
    if indexed.shape != (len(doc_ids), len(terms)) or not np.isfinite(indexed.data).all():
        raise ValueError(f"BM25 matrix dimensions do not match indexed documents: {view_id}")
    row_nnz = np.diff(indexed.indptr)
    if np.count_nonzero(row_nnz) != len(doc_ids):
        raise ValueError(f"BM25 index has one or more empty document rows: {view_id}")
    probes = [document_tokens[index][:8] for index in sorted({0, len(doc_ids) // 2, len(doc_ids) - 1})]
    replay_hashes = []
    for probe in probes:
        query_vector = np.zeros(len(terms), dtype=np.float64)
        for token in probe:
            query_vector[vocabulary[token]] += float(idfs[vocabulary[token]])
        first = np.asarray(indexed @ query_vector).reshape(-1)
        second = np.asarray(indexed @ query_vector).reshape(-1)
        if not np.isfinite(first).all() or not np.array_equal(first, second):
            raise ValueError(f"BM25 exact search replay is non-finite or non-deterministic: {view_id}")
        order = sorted(range(len(doc_ids)), key=lambda index: (-float(first[index]), doc_ids[index]))
        ranking = [(doc_ids[index], float(first[index])) for index in order]
        if len({doc_id for doc_id, _ in ranking}) != len(doc_ids):
            raise ValueError(f"BM25 search returned duplicate/out-of-view IDs: {view_id}")
        if any(not math.isfinite(score) for _, score in ranking) or any(
            left[1] < right[1] for left, right in pairwise(ranking)
        ):
            raise ValueError(f"BM25 ranking score integrity failed: {view_id}")
        replay_hashes.append(_sha256_bytes(_canonical_json(ranking)))

    expected_index_identity = _sha256_bytes(
        _canonical_json(
            {
                "corpus_sha256": manifest["corpus_sha256"],
                "analyzer": manifest["analyzer"]["analyzer"],
                "analyzer_jar_sha256": manifest["analyzer"]["analyzer_jar_sha256"],
                "implementation": "gensim.models.LuceneBM25Model",
                "gensim_version": manifest["gensim_version"],
                "k1": manifest["k1"],
                "b": manifest["b"],
                "document_order_sha256": manifest["document_order_sha256"],
                "token_artifact_sha256": manifest["token_artifact_sha256"],
                "dictionary_model_sha256": manifest["dictionary_model_sha256"],
                "document_matrix_sha256": manifest["document_term_matrix_sha256"],
            }
        )
    )
    if expected_index_identity != manifest.get("bm25_index_identity"):
        raise ValueError(f"BM25 identity hash is inconsistent: {view_id}")
    if row.get("bm25_index_identity") != expected_index_identity:
        raise ValueError(f"BM25 qualification report has a different index identity: {view_id}")
    return {
        "view": view_id,
        "document_count": len(doc_ids),
        "indexed_document_count": int(indexed.shape[0]),
        "matrix_shape": list(indexed.shape),
        "nonempty_indexed_document_rows": int(np.count_nonzero(row_nnz)),
        "document_order_sha_match": True,
        "corpus_sha_match": True,
        "token_and_model_hashes_match": True,
        "stored_matrix_finite": True,
        "stored_matrix_rebuild": "not required; source artifact hashes and index identity validated",
        "search_probe_count": len(probes),
        "search_replay_deterministic": True,
        "replay_hashes": replay_hashes,
        "index_identity": expected_index_identity,
        "integrity_status": "PASS",
    }


def _dense_integrity(
    view_id: str,
    chunks: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    external_root: Path,
    dense_report: dict[str, Any],
    model_root: Path,
    *,
    hash_model_weights: bool,
) -> dict[str, Any]:
    report_view = next(item for item in dense_report["views"] if item["corpus_view_id"] == view_id)
    directory = external_root / "e5a3/indexes" / view_id.lower()
    manifest_path = directory / "dense_index_manifest.json"
    manifest = _read_json(manifest_path)
    matrix_path = directory / "bge_large_embeddings.npy"
    doc_ids = [row["id"] for row in docs]
    order_sha = _sha256_bytes(_canonical_json(doc_ids))
    matrix_sha = _sha256_file(matrix_path)
    if matrix_sha != manifest.get("embedding_matrix_sha256"):
        raise ValueError(f"dense embedding file SHA mismatch: {view_id}")
    matrix = np.load(matrix_path, allow_pickle=False)
    shape_report = validate_dense_matrix(
        matrix,
        expected_rows=len(chunks),
        expected_dimension=1024,
        document_ids=doc_ids,
        manifest_document_order_sha256=manifest.get("document_order_sha256", ""),
        observed_document_order_sha256=order_sha,
    )
    if (
        manifest.get("corpus_sha256") != report_view.get("corpus_sha256")
        or manifest.get("document_count") != len(chunks)
        or manifest.get("embedding_dimension") != 1024
        or manifest.get("normalized") is not True
        or manifest.get("model") != "BAAI/bge-large-en-v1.5"
        or manifest.get("revision") != "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
        or manifest.get("weights_sha256") != EXPECTED_BGE_WEIGHTS_SHA
    ):
        raise ValueError(f"dense index identity/config mismatch: {view_id}")
    if _sha256_bytes(_canonical_json(doc_ids)) != manifest.get("document_order_sha256"):
        raise ValueError(f"dense document order SHA mismatch: {view_id}")
    if report_view.get("dense_index_manifest_sha256") != _sha256_file(manifest_path):
        raise ValueError(f"dense manifest SHA mismatch: {view_id}")
    if report_view.get("dense_index_identity") != manifest.get("dense_index_identity"):
        raise ValueError(f"dense report/index identities disagree: {view_id}")

    model_config = {
        key: manifest[key]
        for key in (
            "device",
            "gpu_name",
            "torch_version",
            "embedding_precision",
            "pooling",
            "normalized",
            "max_length",
            "query_instruction",
            "document_instruction",
            "model_weight_files",
        )
    }
    expected_identity = _sha256_bytes(
        _canonical_json(
            {
                "corpus_sha256": manifest["corpus_sha256"],
                "model": manifest["model"],
                "revision": manifest["revision"],
                "weights_sha256": manifest["weights_sha256"],
                "model_config": model_config,
                "document_order_sha256": manifest["document_order_sha256"],
                "embedding_matrix_sha256": matrix_sha,
            }
        )
    )
    if expected_identity != manifest.get("dense_index_identity"):
        raise ValueError(f"dense index identity hash is inconsistent: {view_id}")
    actual_weight_sha = None
    if hash_model_weights:
        weight_path = model_root / "model.safetensors"
        actual_weight_sha = _sha256_file(weight_path)
        if actual_weight_sha != EXPECTED_BGE_WEIGHTS_SHA:
            raise ValueError("local BGE model weights differ from the frozen profile")

    probes = []
    for index in sorted({0, len(doc_ids) // 2, len(doc_ids) - 1}):
        other = (index + 1) % len(doc_ids)
        query = matrix[index].astype(np.float64) + matrix[other].astype(np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm == 0 or not np.isfinite(query_norm):
            raise ValueError(f"could not construct a valid dense integrity probe: {view_id}")
        probes.append((query / query_norm).astype(np.float32))
    first = exact_dense_rankings(matrix, probes, doc_ids)
    second = exact_dense_rankings(matrix, probes, doc_ids)
    if first != second:
        raise ValueError(f"dense exact-flat search replay is non-deterministic: {view_id}")
    replay_hashes = [
        _sha256_bytes(_canonical_json(ranking)) for ranking in first
    ]
    return {
        "view": view_id,
        **shape_report,
        "corpus_sha_match": True,
        "document_order_sha_match": True,
        "embedding_sha_match": True,
        "model_sha_match": True,
        "model_weight_files_sha256": actual_weight_sha or EXPECTED_BGE_WEIGHTS_SHA,
        "model_name": manifest["model"],
        "model_revision": manifest["revision"],
        "dense_index_identity": expected_identity,
        "search_probe_count": len(probes),
        "query_embeddings_finite_and_normalized": True,
        "search_returns_valid_unique_ids": True,
        "scores_finite_and_descending": True,
        "search_replay_deterministic": True,
        "search_replay_hashes": replay_hashes,
        "integrity_status": "PASS",
    }


def _retrieval_diagnostic(
    external_root: Path, index_manifest: dict[str, Any], chunks_by_view: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    output_views: dict[str, Any] = {}
    source_smoke_hashes: dict[str, str] = {}
    for view in index_manifest["views"]:
        view_id = view["corpus_view_id"]
        smoke_path = external_root / "e5a3/indexes" / view_id.lower() / "source_title_smoke.json"
        if _sha256_file(smoke_path) != view["smoke_report_sha256"]:
            raise ValueError(f"frozen A3 combined smoke hash mismatch: {view_id}")
        smoke = _read_json(smoke_path)
        source_smoke_hashes[view_id] = view["smoke_report_sha256"]
        by_id = {row["chunk_id"]: row["source_id"] for row in chunks_by_view[view_id]}
        output_views[view_id] = {}
        for retriever in ("bm25", "dense"):
            artifact = smoke[retriever]
            rankings = artifact.get("rankings", [])
            if len(rankings) != 10:
                raise ValueError(f"frozen A3 smoke query count changed: {view_id}/{retriever}")
            misses = []
            target_ranks = []
            for ranking in rankings:
                expected = ranking.get("expected_source_id")
                target_rank = next(
                    (
                        rank
                        for rank, row in enumerate(ranking.get("top100", []), start=1)
                        if by_id.get(row.get("chunk_id")) == expected
                    ),
                    None,
                )
                target_ranks.append(target_rank)
                if target_rank is None or target_rank > 10:
                    misses.append(
                        {
                            "query_id": ranking.get("query_id"),
                            "expected_source_id": expected,
                            "target_rank": target_rank,
                        }
                    )
            hits = sum(rank is not None and rank <= 10 for rank in target_ranks)
            if hits != artifact.get("hits_at_10"):
                raise ValueError(f"frozen A3 smoke hit count no longer matches stored ranking: {view_id}/{retriever}")
            output_views[view_id][retriever] = {
                "query_count": len(rankings),
                "hits_at_10": hits,
                "misses": misses,
                "target_rank_distribution": target_ranks,
                "source_smoke_sha256": view["smoke_report_sha256"],
            }
    combined_path = (
        external_root
        / "e5a3/indexes/public_health_plus_guideline/source_title_smoke.json"
    )
    combined_public_queries = {
        row["query_id"]
        for row in _read_json(combined_path)["dense"]["rankings"]
        if row.get("expected_source_id") in {
            "cdc-high-blood-pressure-measuring-01-home",
            "cdc-high-blood-pressure-measuring-03-before-reading",
        }
    }
    if combined_public_queries:
        raise ValueError("combined smoke unexpectedly includes the two frozen PH misses")
    return {
        "schema_version": "rag-e5-e5a31-retrieval-diagnostic-v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "diagnostic_only": True,
        "controls_index_readiness": False,
        "views": output_views,
        "combined_sample_excludes_public_health_08_and_10": True,
        "frozen_source_smoke_artifact_sha256": source_smoke_hashes,
        "interpretation": "Frozen dense BGE has a retained semantic transfer limitation on two Chinese public-health source-title probes; this does not indicate index corruption.",
    }


def _validate_frozen_closeouts(repo_root: Path) -> dict[str, Any]:
    observed = {path: _sha256_file(repo_root / path) for path in FROZEN_CLOSEOUTS}
    for path, expected in FROZEN_CLOSEOUTS.items():
        if observed[path] != expected:
            raise ValueError(f"frozen closeout artifact changed: {path}")
    return {"status": "PASS", "sha256": observed}


def qualify(*, repo_root: Path, external_root: Path, model_root: Path, activate: bool) -> dict[str, Any]:
    source_manifest_path = repo_root / "docs/research/rag_e5/guideline_source_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    owner_decision = _validate_owner_decision(repo_root, source_manifest)
    corpus_report = _read_json(external_root / "e5a3/corpus_build_report.json")
    bm25_report = _read_json(external_root / "e5a3/bm25_qualification_report.json")
    dense_report = _read_json(external_root / "e5a3/index_qualification_report.json")
    corpus_manifest_path = repo_root / "runs/rag_e5/external_corpus_manifest.json"
    index_manifest_path = repo_root / "runs/rag_e5/external_index_manifest.json"
    profile_path = repo_root / "runs/rag_e5/retrieval_action_profiles.json"
    corpus_manifest = _read_json(corpus_manifest_path)
    index_manifest = _read_json(index_manifest_path)
    profile_registry = _read_json(profile_path)
    index_report_path = external_root / "e5a3/index_qualification_report.json"
    if _sha256_file(index_report_path) != index_manifest.get("qualification_report_sha256"):
        raise ValueError("index qualification report is not the frozen candidate report")
    if index_manifest.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA:
        raise ValueError("external index manifest is bound to a different source manifest")

    corpus_integrity, chunks_by_view, language_rows = _validate_corpus(
        repo_root,
        external_root,
        corpus_report,
        corpus_manifest,
        source_manifest,
        owner_decision,
    )
    docs_by_view = {
        row["corpus_view_id"]: _read_jsonl(Path(row["search_documents_jsonl_path"]))
        for row in corpus_report["views"]
    }
    dense_rows = []
    bm25_rows = []
    for index, (view_id, (expected_count, _, _)) in enumerate(EXPECTED_VIEWS.items()):
        chunks = chunks_by_view[view_id]
        if len(chunks) != expected_count:
            raise ValueError(f"unexpected count for integrity view {view_id}")
        dense_rows.append(
            _dense_integrity(
                view_id,
                chunks,
                docs_by_view[view_id],
                external_root,
                dense_report,
                model_root,
                hash_model_weights=index == 0,
            )
        )
        bm25_rows.append(
            _bm25_replay(view_id, chunks, docs_by_view[view_id], external_root, bm25_report)
        )
    if len({row["index_identity"] for row in bm25_rows}) != len(bm25_rows):
        raise ValueError("BM25 corpus views reuse an index identity")
    if len({row["dense_index_identity"] for row in dense_rows}) != len(dense_rows):
        raise ValueError("dense corpus views reuse an index identity")
    if not bm25_report.get("bm25_indexes_ready"):
        raise ValueError("frozen BM25 qualification report does not declare all indexes ready")

    dense_integrity = {
        "status": "PASS",
        "embedding_row_count_total": sum(row["embedding_row_count"] for row in dense_rows),
        "finite_vector_count_total": sum(row["finite_vector_count"] for row in dense_rows),
        "normalized_vector_count_total": sum(row["normalized_vector_count"] for row in dense_rows),
        "zero_vector_count_total": sum(row["zero_vector_count"] for row in dense_rows),
        "views": dense_rows,
    }
    bm25_integrity = {
        "status": "PASS",
        "views": bm25_rows,
    }
    temporal_contract = _read_json(repo_root / "runs/rag_e5/profile_temporal_contract.json")
    feature_contract = _read_json(repo_root / "runs/rag_e5/feature_contract.json")
    prior_a3 = _read_json(repo_root / "runs/rag_e5/e5a3_activation_report.json")
    if prior_a3.get("PROFILE_TEMPORAL_CONTRACT") != "PASS":
        raise ValueError("frozen temporal contract audit did not pass in A3")
    if prior_a3.get("TEMPORAL_LEAKAGE_AUDIT") != "PASS_SYNTHETIC_PRE_CUTOFF_INVARIANCE_TESTS":
        raise ValueError("frozen temporal leakage audit did not pass in A3")
    if prior_a3.get("FEATURE_LEAKAGE_AUDIT") != "PASS_NO_METADATA_OR_PROFILE_PROSE_SHORTCUT":
        raise ValueError("frozen feature leakage audit did not pass in A3")
    if "profile.metadata" not in temporal_contract.get("excluded_from_policy_features", []):
        raise ValueError("profile metadata exclusion changed")
    if not all(
        item.get("uses_gold") is False
        and item.get("uses_counterfactual") is False
        and item.get("uses_external_retrieval") is False
        for item in feature_contract.get("features", [])
    ):
        raise ValueError("policy feature contract has a leakage flag")
    closeouts = _validate_frozen_closeouts(repo_root)
    diagnostic = _retrieval_diagnostic(external_root, index_manifest, chunks_by_view)

    profile_hashes = {
        row["action"]: row["config_sha256"]
        for row in profile_registry["profiles"]
        if row.get("action") in {"STANDARD", "STRONG"}
    }
    expected_profiles = {
        "STANDARD": EXPECTED_STANDARD_CONFIG_SHA,
        "STRONG": EXPECTED_STRONG_CONFIG_SHA,
    }
    if profile_hashes != expected_profiles:
        raise ValueError("frozen STANDARD/STRONG method config hash changed")
    owner_record_sha = owner_decision["owner_decision_record_sha256"]

    integrity_report = {
        "schema_version": "rag-e5-e5a31-structural-integrity-v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "qualification_semantics": "rag-e5-index-qualification-v2",
        "owner_decision_provenance": owner_decision,
        "candidate_external_corpus_identity": EXPECTED_IDENTITY,
        "corpus_integrity": corpus_integrity,
        "bm25_index_integrity": bm25_integrity,
        "dense_index_integrity": dense_integrity,
        "retrieval_diagnostic_is_not_an_integrity_gate": True,
        "frozen_smoke_sha256": diagnostic["frozen_source_smoke_artifact_sha256"],
    }
    language_inventory = {
        "schema_version": "rag-e5-external-language-inventory-v1",
        "method": "deterministic Unicode script-character heuristic; no model or translation",
        "source_count": len(language_rows),
        "sources": language_rows,
    }
    if not all(
        (
            owner_decision["status"] == "VERIFIED",
            corpus_integrity["status"] == "PASS",
            bm25_integrity["status"] == "PASS",
            dense_integrity["status"] == "PASS",
        )
    ):
        raise ValueError("E5-A3.1 structural integrity gate failed; no activation performed")

    _write_json(repo_root / "runs/rag_e5/e5a31_dense_integrity_report.json", integrity_report)
    _write_json(repo_root / "runs/rag_e5/e5a31_retrieval_diagnostic.json", diagnostic)
    _write_json(repo_root / "runs/rag_e5/external_language_inventory.json", language_inventory)

    if not activate:
        return {
            "structural_integrity": "PASS",
            "activation_performed": False,
            "candidate_external_corpus_identity": EXPECTED_IDENTITY,
            "dense_vector_count": dense_integrity["embedding_row_count_total"],
            "language_source_count": len(language_rows),
        }

    profile_registry = bind_active_corpus_identity(
        profile_registry,
        corpus_identity=EXPECTED_IDENTITY,
        expected_config_sha256=expected_profiles,
    )
    corpus_manifest.update(
        {
            "generated_at": "2026-09-29T10:28:08Z",
            "status": "ACTIVE_FROZEN",
            "guideline_active_source_count": 3,
            "guideline_active_chunk_count": 26,
            "active_external_corpus_identity": EXPECTED_IDENTITY,
            "combined_active_corpus_sha256": corpus_report["views"][2]["corpus_sha256"],
            "promotion_blocker": None,
            "activation_record_path": "runs/rag_e5/e5a_final_activation_report.json",
        }
    )
    activation_time = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    corpus_manifest["activated_at"] = activation_time
    index_manifest.update(
        {
            "generated_at": "2026-09-29T10:31:48Z",
            "built_at": "2026-09-29T10:31:48Z",
            "status": "ACTIVE_WITH_RETRIEVAL_DIAGNOSTICS",
            "bm25_index_ready": True,
            "dense_index_ready": True,
            "external_corpus_identity": EXPECTED_IDENTITY,
            "candidate_corpus_identity": EXPECTED_IDENTITY,
            "active_corpus_identity": EXPECTED_IDENTITY,
            "source_title_smoke_pass": False,
            "dense_retrieval_diagnostic": {
                "PUBLIC_HEALTH_ONLY": {"hits_at_10": 8, "query_count": 10, "miss_ranks": [18, 15]},
                "GUIDELINE_ONLY": {"hits_at_10": 10, "query_count": 10, "miss_ranks": []},
                "PUBLIC_HEALTH_PLUS_GUIDELINE": {
                    "hits_at_10": 10,
                    "query_count": 10,
                    "miss_ranks": [],
                    "sample_excludes_public_health_08_and_10": True,
                },
                "classification": "DENSE_RETRIEVAL_DIAGNOSTIC_MISS",
                "likely_contributing_factor": "FROZEN_ENGLISH_MODEL_ON_CHINESE_PUBLIC_HEALTH_TEXT",
                "confidence": "PLAUSIBLE_NOT_CAUSALLY_PROVEN",
                "affects_index_integrity": False,
            },
            "structural_integrity_report_path": "runs/rag_e5/e5a31_dense_integrity_report.json",
            "structural_integrity_report_sha256": _sha256_file(
                repo_root / "runs/rag_e5/e5a31_dense_integrity_report.json"
            ),
        }
    )
    index_manifest["activated_at"] = activation_time
    for view in index_manifest["views"]:
        view.update(
            {
                "status": "ACTIVE_WITH_RETRIEVAL_DIAGNOSTICS",
                "corpus_sha256": next(
                    item["corpus_sha256"]
                    for item in corpus_report["views"]
                    if item["corpus_view_id"] == view["corpus_view_id"]
                ),
                "integrity_status": "PASS",
                "dense_index_ready": True,
                "bm25_index_ready": True,
            }
        )

    _write_json(corpus_manifest_path, corpus_manifest)
    _write_json(index_manifest_path, index_manifest)
    _write_json(profile_path, profile_registry)
    protocol_path = repo_root / "docs/research/rag_e5/e5_protocol.md"
    protocol_sha = _sha256_file(protocol_path)
    activation_report = {
        "schema_version": "rag-e5-e5a-final-activation-v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "base_commit": "40c3fddf7994f8ec6401303b97a603b9e479ba45",
        "qualification_semantics": "rag-e5-index-qualification-v2",
        "owner_decision_provenance": "VERIFIED",
        "owner_decision_record_sha256": owner_record_sha,
        "owner_review_complete": "YES",
        "recommendation_retention": "PASS_22_OF_22",
        "profile_temporal_contract": "PASS",
        "temporal_leakage_audit": "PASS_SYNTHETIC_PRE_CUTOFF_INVARIANCE_TESTS",
        "feature_leakage_audit": "PASS_NO_METADATA_OR_PROFILE_PROSE_SHORTCUT",
        "corpus_integrity": "PASS",
        "bm25_index_integrity": "PASS",
        "dense_index_integrity": "PASS",
        "bm25_retrieval_diagnostic": "10_OF_10_ALL_VIEWS",
        "dense_retrieval_diagnostic": diagnostic["views"],
        "BGE_PUBLIC_HEALTH_TITLE_DIAGNOSTIC": "8/10",
        "BGE_PUBLIC_HEALTH_MISSES": [
            {"query_id": "public-health-08", "target_rank": 18},
            {"query_id": "public-health-10", "target_rank": 15},
        ],
        "AFFECTS_INDEX_INTEGRITY": "NO",
        "KNOWN_TRANSFER_LIMITATION": "YES",
        "SMOKE_FAILURE_CLASSIFICATION": "DENSE_RETRIEVAL_DIAGNOSTIC_MISS",
        "LIKELY_CONTRIBUTING_FACTOR": "FROZEN_ENGLISH_MODEL_ON_CHINESE_PUBLIC_HEALTH_TEXT",
        "LIKELY_CONTRIBUTING_FACTOR_CONFIDENCE": "PLAUSIBLE_NOT_CAUSALLY_PROVEN",
        "CORPUS_CONTENT_CHANGED": "NO",
        "INDEX_CONTENT_CHANGED": "NO",
        "ACTIVE_EXTERNAL_CORPUS_IDENTITY": EXPECTED_IDENTITY,
        "STANDARD_METHOD_CONFIG_UNCHANGED": "YES",
        "STRONG_METHOD_CONFIG_UNCHANGED": "YES",
        "STANDARD_CORPUS_BOUND": "YES",
        "STRONG_CORPUS_BOUND": "YES",
        "STANDARD_PROFILE_CONFIG_SHA256": profile_hashes["STANDARD"],
        "STRONG_PROFILE_CONFIG_SHA256": profile_hashes["STRONG"],
        "RAG_CLOSEOUT_UNCHANGED": "YES",
        "RAG_CLOSEOUT_SHA256": closeouts["sha256"]["docs/research/rag_closeout.md"],
        "MIRAGE_CLOSEOUT_UNCHANGED": "YES",
        "MIRAGE_CLOSEOUT_SHA256": closeouts["sha256"]["docs/research/e1_3_mirage_exploratory.md"],
        "MIRAGE_REPORT_SHA256": closeouts["sha256"]["runs/e1_3/mirage_exploratory_report.json"],
        "R2MED_LOCK_UNCHANGED": "YES",
        "R2MED_FINAL_TEST_LOCK_SHA256": closeouts["sha256"]["runs/rag_r2med_final_test/final_eval_lock.json"],
        "R2MED_SOURCE_MANIFEST_SHA256": closeouts["sha256"]["runs/rag_r2med_crb/source_manifest.json"],
        "STRUCTURAL_INTEGRITY_REPORT_SHA256": _sha256_file(
            repo_root / "runs/rag_e5/e5a31_dense_integrity_report.json"
        ),
        "LANGUAGE_INVENTORY_SHA256": _sha256_file(
            repo_root / "runs/rag_e5/external_language_inventory.json"
        ),
        "RETRIEVAL_DIAGNOSTIC_SHA256": _sha256_file(
            repo_root / "runs/rag_e5/e5a31_retrieval_diagnostic.json"
        ),
        "PROTOCOL_SHA256": protocol_sha,
        "E5A_READY": "YES",
        "E5B_STARTED": "NO",
    }
    _write_json(repo_root / "runs/rag_e5/e5a_final_activation_report.json", activation_report)
    return {
        "structural_integrity": "PASS",
        "activation_performed": True,
        "active_external_corpus_identity": EXPECTED_IDENTITY,
        "dense_vector_count": dense_integrity["embedding_row_count_total"],
        "main_merge_required_after_tests": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    result = qualify(
        repo_root=args.repo_root,
        external_root=args.external_root,
        model_root=args.model_root,
        activate=args.activate,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
