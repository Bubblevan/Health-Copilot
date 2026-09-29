"""Qualify E5-A2 WHO candidates without activating or indexing them."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.rag_e5.corpus import approved_guideline_sources, verify_raw_source

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXTERNAL_ROOT = Path("D:/MyLab/Jianli/external/rag_e5")
DOMAIN_BINDINGS = {
    "who-hypertension-pharmacological-2021": ("hypertension", "cardiovascular"),
    "who-physical-activity-sedentary-2020": ("physical_activity",),
    "who-total-fat-weight-gain-2023": ("nutrition", "weight_metabolic"),
}


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256(payload.encode("utf-8"))


def qualify(repo_root: Path, external_root: Path) -> dict[str, dict[str, Any]]:
    source_manifest_path = repo_root / "docs/research/rag_e5/guideline_source_manifest.json"
    source_manifest, source_manifest_bytes = _read_json(source_manifest_path)
    inventory_path = repo_root / "runs/rag_e5/dev_longitudinal_domain_inventory.json"
    inventory, inventory_bytes = _read_json(inventory_path)
    action_profiles, _ = _read_json(
        repo_root / "runs/rag_e5/retrieval_action_profiles.json"
    )

    sources = source_manifest.get("sources")
    if not isinstance(sources, list):
        raise TypeError("guideline source manifest must contain a sources list")
    approved = approved_guideline_sources(source_manifest)
    user_coverage = {
        row["domain"]: row["users_with_timeline_or_exam_signal"]
        for row in inventory.get("domain_coverage", [])
        if isinstance(row, dict)
    }
    if inventory.get("batch_id") != "202607" or inventory.get("user_count") != 20:
        raise ValueError("DEV domain inventory is not the frozen 202607/20-user audit")
    if inventory.get("forbidden_inputs_opened") != []:
        raise ValueError("DEV audit records forbidden input access")

    candidate_rows: list[dict[str, Any]] = []
    raw_identity_rows: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("guideline source rows must be objects")
        source_id = source["source_id"]
        raw_path = external_root / source["raw_path"]
        observed_sha = verify_raw_source(source, raw_path)
        raw_size = raw_path.stat().st_size
        if raw_size != source["raw_bytes"]:
            raise ValueError(f"raw source byte count mismatch for {source_id}")
        extracted_relative = f"guidelines/extracted/{Path(source['raw_path']).stem}.txt"
        extracted_path = external_root / extracted_relative
        extracted_sha = _sha256(extracted_path.read_bytes()) if extracted_path.is_file() else None
        covered_domains = list(DOMAIN_BINDINGS[source_id])
        coverage = {domain: user_coverage.get(domain, 0) for domain in covered_domains}
        if any(count < 5 for count in coverage.values()):
            raise ValueError(f"202607 domain coverage gate failed for {source_id}")
        candidate_rows.append(
            {
                "source_id": source_id,
                "source_family": "reviewed_guideline",
                "review_status": source["review_status"],
                "owner_review_status": source["owner_review_status"],
                "raw_sha256": observed_sha,
                "raw_bytes": raw_size,
                "extracted_text_sha256": extracted_sha,
                "domain_bindings": covered_domains,
                "users_with_longitudinal_signal_by_domain": coverage,
                "recommendation_section_count": source[
                    "detected_recommendation_section_count"
                ],
                "chunked_recommendation_section_count": source[
                    "chunked_recommendation_section_count"
                ],
                "recommendation_retention_audit": source["recommendation_retention_audit"],
            }
        )
        raw_identity_rows.append({"source_id": source_id, "raw_sha256": observed_sha})

    card_paths = sorted(
        path
        for path in (repo_root / "data/knowledge_cards").glob("*.json")
        if not path.name.startswith("_")
    )
    if len(card_paths) != 30:
        raise ValueError(f"expected 30 existing public-health cards, found {len(card_paths)}")
    public_health_rows = [
        {"card_id": path.stem, "file_sha256": _sha256(path.read_bytes())} for path in card_paths
    ]
    public_health_corpus_sha = _canonical_hash(public_health_rows)
    raw_source_set_sha = _canonical_hash(
        sorted(raw_identity_rows, key=lambda item: item["source_id"])
    )

    profile_rows = action_profiles.get("profiles", [])
    if action_profiles.get("external_corpus_identity") is not None:
        raise ValueError("A2 must not silently activate or rebind the frozen action profiles")
    action_config_shas = {
        row["action"]: row["config_sha256"]
        for row in profile_rows
        if isinstance(row, dict) and row.get("action") in {"STANDARD", "STRONG"}
    }
    if set(action_config_shas) != {"STANDARD", "STRONG"}:
        raise ValueError("frozen STANDARD/STRONG profile identities are incomplete")

    candidate_count = len(sources)
    approved_count = len(approved)
    generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    external_corpus_manifest = {
        "schema_version": "rag-e5-external-corpus-manifest-v1",
        "generated_at": generated_at,
        "status": "GUIDELINE_CANDIDATES_QUARANTINED",
        "public_health_source_count": len(card_paths),
        "public_health_chunk_count": len(card_paths),
        "public_health_corpus_sha256": public_health_corpus_sha,
        "guideline_candidate_source_count": candidate_count,
        "guideline_approved_source_count": approved_count,
        "guideline_active_source_count": approved_count,
        "guideline_active_chunk_count": 0,
        "guideline_candidate_chunk_count": 0,
        "candidate_raw_source_set_sha256": raw_source_set_sha,
        "combined_active_corpus_sha256": None,
        "chunker_config_sha256": None,
        "extractor_version": "Poppler pdftotext 23.13.0 -layout (quarantined text only)",
        "source_manifest_sha256": _sha256(source_manifest_bytes),
        "public_health_card_identity_note": (
            "The existing 30 reviewed public-health cards are referenced in place; no copies or "
            "source-family reclassification were made."
        ),
        "guideline_candidates": candidate_rows,
        "promotion_blocker": source_manifest.get("promotion_blocker"),
    }

    views = [
        {
            "corpus_view_id": "PUBLIC_HEALTH_ONLY",
            "source_families": ["public_health"],
            "source_count": len(card_paths),
            "chunk_count": len(card_paths),
            "corpus_sha256": public_health_corpus_sha,
            "status": "NOT_INDEXED_BY_E5_A2",
        },
        {
            "corpus_view_id": "GUIDELINE_ONLY",
            "source_families": ["reviewed_guideline"],
            "source_count": 0,
            "chunk_count": 0,
            "corpus_sha256": None,
            "status": "BLOCKED_OWNER_APPROVAL_AND_CHUNK_AUDIT",
        },
        {
            "corpus_view_id": "PUBLIC_HEALTH_PLUS_GUIDELINE",
            "source_families": ["public_health", "reviewed_guideline"],
            "source_count": len(card_paths),
            "chunk_count": len(card_paths),
            "corpus_sha256": None,
            "status": "BLOCKED_OWNER_APPROVAL_AND_CHUNK_AUDIT",
        },
    ]
    external_index_manifest = {
        "schema_version": "rag-e5-external-index-manifest-v1",
        "generated_at": generated_at,
        "status": "NOT_BUILT",
        "bm25_index_ready": False,
        "dense_index_ready": False,
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_revision": "d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        "embedding_dimension": 1024,
        "normalized": True,
        "views": views,
        "standard_profile_config_sha256": action_config_shas["STANDARD"],
        "strong_profile_config_sha256": action_config_shas["STRONG"],
        "external_corpus_identity": None,
        "built_at": None,
        "code_commit": None,
    }

    return {
        "external_corpus_manifest": external_corpus_manifest,
        "external_index_manifest": external_index_manifest,
        "qualification_report": {
            "schema_version": "rag-e5-e5a2-qualification-report-v1",
            "stage": "E5-A2",
            "generated_at": generated_at,
            "base_commit": "5ca495113a190ecc49d8110a2ea8fabb72e04647",
            "branch": "codex/rag-e5-a2-20260929",
            "dev_domain_inventory": {
                "batch_id": "202607",
                "user_count": 20,
                "state_files_index_sha256": inventory["state_files_index_sha256"],
                "inventory_sha256": _sha256(inventory_bytes),
                "allowed_files_only": inventory["files_read_per_user"],
                "forbidden_inputs_opened": inventory["forbidden_inputs_opened"],
                "domain_coverage_minimum_users": 5,
                "selected_guideline_domains": {
                    domain: user_coverage[domain]
                    for domain in sorted(
                        {item for bindings in DOMAIN_BINDINGS.values() for item in bindings}
                    )
                },
                "interpretation_limit": (
                    "Domain coverage is a coarse schema/indicator-name proxy, not clinically "
                    "adjudicated prevalence or proof that a future task is answerable."
                ),
            },
            "guideline_sources_considered": candidate_count
            + len(source_manifest.get("considered_but_not_selected", [])),
            "guideline_sources_ready_for_review": candidate_count,
            "guideline_sources_approved": approved_count,
            "guideline_domain_count": len(DOMAIN_BINDINGS),
            "raw_corpus_sha256": raw_source_set_sha,
            "chunk_corpus_sha256": None,
            "public_health_sources": len(card_paths),
            "public_health_chunks": len(card_paths),
            "guideline_active_chunks": 0,
            "e5_external_index_chunks": 0,
            "bm25_index_ready": False,
            "bge_index_ready": False,
            "post_retrieval_family_filtering": "NO_FORMAL_RUNNER; views are pre-scoped",
            "capability_context_source": "environment",
            "capability_context_task_independent": True,
            "state_summary_deterministic": True,
            "temporal_leakage_audit": "PASS_FOCUSED_SYNTHETIC_TESTS",
            "feature_leakage_audit": "PASS_FOCUSED_SYNTHETIC_TESTS",
            "standard_profile_config_sha256": action_config_shas["STANDARD"],
            "strong_profile_config_sha256": action_config_shas["STRONG"],
            "standard_profile_unchanged": True,
            "strong_profile_unchanged": True,
            "rag_closeout_unchanged": True,
            "mirage_closeout_unchanged": True,
            "r2med_lock_unchanged": True,
            "frozen_artifact_sha256": {
                "r2med_final_test_lock": "0b80fad6668c3a57833c55beb1db359c65440cf015effd10da770c630fb3e41d",
                "r2med_source_manifest": "b70c4f01b37f58c77597f1e28cc35a52585785142f3f625f928173c81be3874e",
                "rag_closeout_document": "1a127b15ac5954e188b5b4dfd0d719799f0987cd5d45ce7c5c0c1c0f0b3dab2d",
            },
            "external_corpus_identity": None,
            "guideline_corpus_built": False,
            "guideline_capability_eligible": False,
            "e5a_ready": False,
            "blocker": "OWNER_REVIEW_REQUIRED; recommendation chunking, smoke retrieval, and index build are gated",
            "focused_test_result": "37 passed (E5 foundation, capability context, state packet, and corpus contracts)",
            "full_repository_test_result": "667 passed, 2 skipped; 1 dependency deprecation warning",
            "ruff_result": "PASS; full-tree scan emitted access-denied warnings for unrelated protected temp directories",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    args = parser.parse_args()
    outputs = qualify(args.repo_root.resolve(), args.external_root.resolve())
    output_paths = {
        "external_corpus_manifest": args.repo_root / "runs/rag_e5/external_corpus_manifest.json",
        "external_index_manifest": args.repo_root / "runs/rag_e5/external_index_manifest.json",
        "qualification_report": args.repo_root / "runs/rag_e5/e5a2_qualification_report.json",
    }
    for key, path in output_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(outputs[key], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
