"""Build A3 corpus artifacts outside Git from owner-approved, hash-pinned sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from eval.rag_e5.corpus import (
    RecommendationBlock,
    approved_guideline_sources,
    build_corpus_view,
    build_guideline_chunks,
    build_public_health_chunks,
    task_authoring_guideline_sources,
    verify_raw_source,
)
from eval.rag_e5.who_extraction import (
    WHO_EXTRACTOR_VERSION,
    audit_recommendation_retention,
    canonical_retained_section_ids,
    extract_who_blocks,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_ROOT = Path("D:/MyLab/Jianli/external/rag_e5")
DEFAULT_MODEL_ROOT = Path("E:/Health-Copilot-Models/models/bge-large-en-v1.5")
DEFAULT_PDFTOTEXT = Path("E:/Software/MiKTeX/miktex/bin/x64/pdftotext.exe")
CHUNKER_VERSION = "recommendation-boundary-paragraph-pack-v1"
TARGET_TOKENS = 384
HARD_MAX_TOKENS = 480
_ACTIVITY_PAGES = (35, 39, 42, 48, 53, 56, 57, 61, 62, 68, 70, 71, 74)
_TOTAL_FAT_PAGES = (30, 31, 32)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _run_pdftotext(executable: Path, pdf_path: Path, *extra: str) -> str:
    result = subprocess.run(
        [str(executable), *extra, str(pdf_path), "-"],
        check=True,
        capture_output=True,
        text=False,
    )
    return result.stdout.decode("utf-8", errors="strict")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(_canonical_json(row) + b"\n" for row in rows)
    path.write_bytes(data)
    return _sha256(data)


def build(repo_root: Path, external_root: Path, model_root: Path, pdftotext: Path) -> dict[str, Any]:
    manifest_path = repo_root / "docs/research/rag_e5/guideline_source_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    owner_record_path = repo_root / manifest["owner_decision_record"]
    owner_record_bytes = owner_record_path.read_bytes()
    owner_record_sha = _sha256(owner_record_bytes)
    if owner_record_sha != manifest.get("owner_decision_record_sha256"):
        raise ValueError("owner decision record SHA-256 does not match the source manifest")

    approved = approved_guideline_sources(manifest)
    authoring_eligible = task_authoring_guideline_sources(manifest)
    if len(approved) != 3 or len(authoring_eligible) != 2:
        raise ValueError("A3 requires three approved sources, exactly two task-authoring eligible")
    for source in approved:
        if source.get("owner_decision_record_sha256") != owner_record_sha:
            raise ValueError(f"source does not bind the owner decision record: {source['source_id']}")

    version_output = subprocess.run(
        [str(pdftotext), "-v"], capture_output=True, text=True, check=False
    )
    version_text = (version_output.stdout + version_output.stderr).strip()
    if "pdftotext version 23.13.0" not in version_text:
        raise ValueError(f"unexpected Poppler pdftotext version: {version_text}")

    raw_paths: dict[str, Path] = {}
    blocks_by_source: dict[str, tuple[RecommendationBlock, ...]] = {}
    extraction_rows: list[dict[str, Any]] = []
    for source in approved:
        source_id = str(source["source_id"])
        raw_path = external_root / str(source["raw_path"])
        observed_raw_sha = verify_raw_source(source, raw_path)
        if raw_path.stat().st_size != source.get("raw_bytes"):
            raise ValueError(f"source byte-count mismatch: {source_id}")
        full_text = _run_pdftotext(pdftotext, raw_path, "-raw", "-enc", "UTF-8")
        page_map = None
        if source_id == "who-physical-activity-sedentary-2020":
            page_map = {
                page: _run_pdftotext(
                    pdftotext,
                    raw_path,
                    "-raw",
                    "-enc",
                    "UTF-8",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                )
                for page in _ACTIVITY_PAGES
            }
        elif source_id == "who-total-fat-weight-gain-2023":
            page_map = {
                page: _run_pdftotext(
                    pdftotext,
                    raw_path,
                    "-layout",
                    "-enc",
                    "UTF-8",
                    "-f",
                    str(page),
                    "-l",
                    str(page),
                )
                for page in _TOTAL_FAT_PAGES
            }
        blocks = extract_who_blocks(source_id=source_id, full_text=full_text, pages=page_map)
        raw_paths[source_id] = raw_path
        blocks_by_source[source_id] = blocks
        declared_sections = source.get("recommendation_sections", [])
        expected_section_ids = tuple(str(row["section_id"]) for row in declared_sections)
        retained_section_ids = canonical_retained_section_ids(
            source_id=source_id,
            declared_section_ids=expected_section_ids,
            blocks=blocks,
        )
        retention_audit = audit_recommendation_retention(
            expected_section_ids=expected_section_ids,
            retained_section_ids=retained_section_ids,
        )
        if any(retention_audit.values()):
            raise ValueError(f"recommendation retention failed for {source_id}: {retention_audit}")
        extraction_rows.append(
            {
                "source_id": source_id,
                "source_title": source["title"],
                "source_raw_sha256": observed_raw_sha,
                "full_raw_extraction_sha256": _sha256(full_text.encode("utf-8")),
                "recommendation_section_count_declared": len(declared_sections),
                "recommendation_sections_retained": len(retained_section_ids),
                "recommendation_blocks_retained": len(blocks),
                "expected_section_ids": list(expected_section_ids),
                "retained_section_ids": list(retained_section_ids),
                **retention_audit,
                "recommendation_retention_pass": True,
                "recommendation_block_ids": [block.recommendation_id for block in blocks],
                "extractor_version": WHO_EXTRACTOR_VERSION,
                "extraction_mode": (
                    "Poppler pdftotext 23.13.0 -layout -enc UTF-8; PDF pages 30-32 only"
                    if source_id == "who-total-fat-weight-gain-2023"
                    else "Poppler pdftotext 23.13.0 -raw -enc UTF-8; selected recommendation spans only"
                ),
                "third_party_material_included": False,
                "excluded_material": [
                    "rationale and supporting evidence",
                    "evidence tables and annexes",
                    "figures and attributed third-party assets",
                    "bibliography and source-study text",
                ],
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True)
    tokenizer_sha = _sha256((model_root / "tokenizer.json").read_bytes())

    def tokenize(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False, truncation=False)

    guideline_chunks = build_guideline_chunks(
        manifest=manifest,
        raw_paths=raw_paths,
        blocks_by_source=blocks_by_source,
        tokenize=tokenize,
        extractor_version=(
            f"{WHO_EXTRACTOR_VERSION}; {version_text.splitlines()[0]}; "
            "pinned source-specific page/section mapping v1"
        ),
        chunker_version=CHUNKER_VERSION,
        target_tokens=TARGET_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )
    card_paths = sorted(
        path
        for path in (repo_root / "data/knowledge_cards").glob("*.json")
        if not path.name.startswith("_")
    )
    public_chunks = build_public_health_chunks(card_paths=card_paths, tokenize=tokenize)
    if len(public_chunks) != 30:
        raise ValueError(f"expected 30 public-health cards, got {len(public_chunks)}")

    all_chunks = (*public_chunks, *guideline_chunks)
    views = (
        build_corpus_view(
            all_chunks, corpus_view_id="PUBLIC_HEALTH_ONLY", source_families=("public_health",)
        ),
        build_corpus_view(
            all_chunks, corpus_view_id="GUIDELINE_ONLY", source_families=("reviewed_guideline",)
        ),
        build_corpus_view(
            all_chunks,
            corpus_view_id="PUBLIC_HEALTH_PLUS_GUIDELINE",
            source_families=("public_health", "reviewed_guideline"),
        ),
    )
    chunker_config = {
        "chunker_version": CHUNKER_VERSION,
        "extractor_version": WHO_EXTRACTOR_VERSION,
        "target_tokens": TARGET_TOKENS,
        "hard_max_tokens": HARD_MAX_TOKENS,
        "tokenizer_model": "BAAI/bge-large-en-v1.5",
        "tokenizer_revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "tokenizer_json_sha256": tokenizer_sha,
        "public_health_chunk_policy": "one unchanged reviewed card per chunk",
        "guideline_chunk_policy": "paragraph packing; never cross a source recommendation boundary",
    }
    chunker_config_sha = _sha256(_canonical_json(chunker_config))
    view_summaries: list[dict[str, Any]] = []
    for view in views:
        directory = external_root / "e5a3" / "corpus" / view.corpus_view_id.lower()
        chunk_rows = [chunk.to_dict() for chunk in sorted(view.chunks, key=lambda item: item.chunk_id)]
        chunk_artifact_sha = _write_jsonl(directory / "chunks.jsonl", chunk_rows)
        search_rows = []
        for chunk in sorted(view.chunks, key=lambda item: item.chunk_id):
            contents = " ".join((*chunk.section_path, chunk.text))
            doc_tokens = tokenize(contents)
            if len(doc_tokens) > 512:
                raise ValueError(f"BGE document would truncate at 512 tokens: {chunk.chunk_id}")
            search_rows.append(
                {
                    "id": chunk.chunk_id,
                    "contents": contents,
                    "source_id": chunk.source_id,
                    "source_family": chunk.source_family,
                }
            )
        search_artifact_sha = _write_jsonl(directory / "search_documents.jsonl", search_rows)
        view_summaries.append(
            {
                "corpus_view_id": view.corpus_view_id,
                "source_families": list(view.source_families),
                "source_count": len({chunk.source_id for chunk in view.chunks}),
                "chunk_count": len(view.chunks),
                "corpus_sha256": view.corpus_sha256,
                "chunks_jsonl_path": str(directory / "chunks.jsonl"),
                "chunks_jsonl_sha256": chunk_artifact_sha,
                "search_documents_jsonl_path": str(directory / "search_documents.jsonl"),
                "search_documents_jsonl_sha256": search_artifact_sha,
            }
        )

    corpus_identity = _sha256(
        _canonical_json(
            {
                "schema_version": "rag-e5-external-corpus-identity-v1",
                "source_manifest_sha256": _sha256(manifest_bytes),
                "owner_decision_record_sha256": owner_record_sha,
                "chunker_config_sha256": chunker_config_sha,
                "views": [
                    {"corpus_view_id": view.corpus_view_id, "corpus_sha256": view.corpus_sha256}
                    for view in views
                ],
            }
        )
    )
    report = {
        "schema_version": "rag-e5-e5a3-corpus-build-v1",
        "stage": "E5-A3-corpus-build",
        "pdftotext_version": version_text.splitlines()[0],
        "source_manifest_sha256": _sha256(manifest_bytes),
        "owner_decision_record_sha256": owner_record_sha,
        "chunker_config": chunker_config,
        "chunker_config_sha256": chunker_config_sha,
        "guideline_sources_approved": len(approved),
        "guideline_sources_task_authoring_eligible": len(authoring_eligible),
        "guideline_blocks": extraction_rows,
        "guideline_chunk_count": len(guideline_chunks),
        "guideline_chunk_tokens": [chunk.token_count for chunk in guideline_chunks],
        "public_health_source_count": len(card_paths),
        "public_health_chunk_count": len(public_chunks),
        "views": view_summaries,
        "external_corpus_identity": corpus_identity,
        "storage_boundary": "All WHO text-derived chunks and indexes are external artifacts; none are written into the Git repository.",
        "attribution": "World Health Organization (WHO), source-specific titles/years/URLs and CC BY-NC-SA 3.0 IGO; WHO endorsement is not implied.",
    }
    report_path = external_root / "e5a3" / "corpus_build_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(_canonical_json(report) + b"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--pdftotext", type=Path, default=DEFAULT_PDFTOTEXT)
    args = parser.parse_args()
    result = build(args.repo_root, args.external_root, args.model_root, args.pdftotext)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
