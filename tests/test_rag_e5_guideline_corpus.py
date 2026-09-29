from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from eval.rag_e5.corpus import (
    E5ExternalChunk,
    RecommendationBlock,
    approved_guideline_sources,
    build_corpus_view,
    build_guideline_chunks,
    build_public_health_chunks,
    task_authoring_guideline_sources,
    validate_index_binding,
    validate_ranked_chunk_ids,
    verify_raw_source,
)


def _manifest(path: Path, *, status: str = "APPROVED") -> dict[str, object]:
    raw_hash = sha256(path.read_bytes()).hexdigest()
    return {
        "source_family": "reviewed_guideline",
        "owner_decision_record_sha256": "d" * 64,
        "sources": [
            {
                "source_id": "who-synthetic-1",
                "source_family": "reviewed_guideline",
                "review_status": status,
                "owner_review_status": status,
                "owner_reviewed_at": "2026-01-01T00:00:00Z" if status == "APPROVED" else None,
                "owner_decision_record_sha256": "d" * 64,
                "approval_scope": "test public-health scope",
                "task_authoring_eligible": True,
                "allowed_task_intents": ["general_guideline_information"],
                "prohibited_task_intents": ["diagnosis"],
                "raw_sha256": raw_hash,
            }
        ],
    }


def _tokenize(text: str) -> list[str]:
    return text.split()


def _chunk(path: Path):
    manifest = _manifest(path)
    chunks = build_guideline_chunks(
        manifest=manifest,
        raw_paths={"who-synthetic-1": path},
        blocks_by_source={
            "who-synthetic-1": [
                RecommendationBlock(
                    section_path=("Adults", "Recommendation 1"),
                    recommendation_id="rec-1",
                    paragraphs=("alpha beta", "gamma delta"),
                )
            ]
        },
        tokenize=_tokenize,
        extractor_version="test-extractor-v1",
        chunker_version="test-chunker-v1",
        target_tokens=3,
        hard_max_tokens=4,
    )
    return manifest, chunks


def test_only_fully_approved_sources_enter_guideline_corpus(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest = _manifest(raw_path, status="READY_FOR_OWNER_REVIEW")

    assert approved_guideline_sources(manifest) == ()
    with pytest.raises(ValueError, match="only owner-approved"):
        build_guideline_chunks(
            manifest=manifest,
            raw_paths={"who-synthetic-1": raw_path},
            blocks_by_source={"who-synthetic-1": []},
            tokenize=_tokenize,
            extractor_version="test-v1",
            chunker_version="test-v1",
        )


def test_guideline_source_family_is_not_public_health(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest = _manifest(raw_path)
    source = approved_guideline_sources(manifest)[0]

    assert source["source_family"] == "reviewed_guideline"
    assert source["source_family"] != "public_health"


def test_task_authoring_requires_explicit_source_eligibility(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest = _manifest(raw_path)
    source = manifest["sources"][0]  # type: ignore[index]
    source["task_authoring_eligible"] = False

    assert task_authoring_guideline_sources(manifest) == ()
    assert len(approved_guideline_sources(manifest)) == 1


def test_approved_source_requires_machine_readable_scope_limits(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest = _manifest(raw_path)
    source = manifest["sources"][0]  # type: ignore[index]
    del source["prohibited_task_intents"]

    with pytest.raises(ValueError, match="prohibited_task_intents"):
        approved_guideline_sources(manifest)


def test_approved_source_must_bind_the_manifest_owner_decision(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest = _manifest(raw_path)
    source = manifest["sources"][0]  # type: ignore[index]
    source["owner_decision_record_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="bind the owner decision record"):
        approved_guideline_sources(manifest)


def test_raw_source_hash_must_match_manifest(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    source = _manifest(raw_path)["sources"][0]  # type: ignore[index]

    assert verify_raw_source(source, raw_path) == sha256(raw_path.read_bytes()).hexdigest()
    raw_path.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_raw_source(source, raw_path)


def test_chunks_bind_source_hash_and_chunk_ids_are_deterministic(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest, first = _chunk(raw_path)
    _, second = _chunk(raw_path)

    assert first == second
    assert len(first) == 2
    assert all(chunk.source_raw_sha256 == manifest["sources"][0]["raw_sha256"] for chunk in first)  # type: ignore[index]
    assert len({chunk.chunk_id for chunk in first}) == 2
    assert all(chunk.chunk_id.startswith("who-guideline-") for chunk in first)


def test_no_empty_chunks_or_paragraphs(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest = _manifest(raw_path)

    with pytest.raises(ValueError, match="empty recommendation paragraphs"):
        build_guideline_chunks(
            manifest=manifest,
            raw_paths={"who-synthetic-1": raw_path},
            blocks_by_source={
                "who-synthetic-1": [RecommendationBlock(("S",), "R1", ("",))]
            },
            tokenize=_tokenize,
            extractor_version="test-v1",
            chunker_version="test-v1",
        )


def test_chunk_max_token_contract_is_enforced(tmp_path: Path) -> None:
    raw_path = tmp_path / "source.pdf"
    raw_path.write_bytes(b"synthetic guideline source")
    manifest = _manifest(raw_path)

    with pytest.raises(ValueError, match="exceeds hard_max_tokens"):
        build_guideline_chunks(
            manifest=manifest,
            raw_paths={"who-synthetic-1": raw_path},
            blocks_by_source={
                "who-synthetic-1": [
                    RecommendationBlock(("S",), "R1", ("one two three four five",))
                ]
            },
            tokenize=_tokenize,
            extractor_version="test-v1",
            chunker_version="test-v1",
            target_tokens=3,
            hard_max_tokens=4,
        )


def test_corpus_views_are_family_scoped_before_ranking() -> None:
    common = {
        "section_path": ("section",),
        "recommendation_id": "rec",
        "text": "text",
        "text_sha256": "a" * 64,
        "token_count": 1,
        "char_count": 4,
        "raw_source_sha256": "b" * 64,
        "review_identity": "c" * 64,
        "extractor_version": "v1",
        "chunker_version": "v1",
    }
    guideline = E5ExternalChunk(
        chunk_id="guideline-1", source_id="who-1", source_family="reviewed_guideline", **common
    )
    public_health = E5ExternalChunk(
        chunk_id="public-1", source_id="ph-1", source_family="public_health", **common
    )

    combined = build_corpus_view(
        (guideline, public_health),
        corpus_view_id="PUBLIC_HEALTH_PLUS_GUIDELINE",
        source_families=("public_health", "reviewed_guideline"),
    )
    guideline_only = build_corpus_view(
        (guideline, public_health),
        corpus_view_id="GUIDELINE_ONLY",
        source_families=("reviewed_guideline",),
    )

    assert {chunk.source_family for chunk in combined.chunks} == {
        "public_health",
        "reviewed_guideline",
    }
    assert all(chunk.source_family == "reviewed_guideline" for chunk in guideline_only.chunks)
    with pytest.raises(ValueError, match="outside its prebuilt corpus view"):
        validate_ranked_chunk_ids(guideline_only, ("public-1",))


def test_each_reviewed_knowledge_card_maps_to_one_unmodified_chunk(tmp_path: Path) -> None:
    card_path = tmp_path / "card-1.json"
    card = {
        "id": "card-1",
        "title": "A reviewed title",
        "content": "Keep this exact public-health body.",
        "source_url": "https://example.org/source",
        "publisher": "Example publisher",
        "reviewed_at": "2026-01-01",
        "reviewer": "reviewer-1",
        "version": "1",
    }
    raw = json.dumps(card).encode("utf-8")
    card_path.write_bytes(raw)

    chunks = build_public_health_chunks(card_paths=(card_path,), tokenize=_tokenize)

    assert len(chunks) == 1
    assert chunks[0].source_id == card["id"]
    assert chunks[0].source_family == "public_health"
    assert chunks[0].text == card["content"]
    assert chunks[0].raw_source_sha256 == sha256(raw).hexdigest()
    assert chunks[0].recommendation_id is None

    view = build_corpus_view(
        chunks, corpus_view_id="PUBLIC_HEALTH_ONLY", source_families=("public_health",)
    )
    expected_payload = [{"card_id": "card-1", "file_sha256": sha256(raw).hexdigest()}]
    expected_sha = sha256(
        json.dumps(
            expected_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert view.corpus_sha256 == expected_sha


def test_knowledge_card_identity_must_match_its_filename(tmp_path: Path) -> None:
    card_path = tmp_path / "filename-id.json"
    card_path.write_text(
        json.dumps(
            {
                "id": "different-id",
                "title": "A reviewed title",
                "content": "Reviewed body.",
                "source_url": "https://example.org/source",
                "publisher": "Example publisher",
                "reviewed_at": "2026-01-01",
                "reviewer": "reviewer-1",
                "version": "1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frozen filename identity"):
        build_public_health_chunks(card_paths=(card_path,), tokenize=_tokenize)


def test_bm25_and_dense_indexes_must_bind_exact_corpus_hash() -> None:
    view = build_corpus_view(
        (), corpus_view_id="GUIDELINE_ONLY", source_families=("reviewed_guideline",)
    )
    manifest = {
        "corpus_view_id": view.corpus_view_id,
        "corpus_sha256": view.corpus_sha256,
        "source_families": list(view.source_families),
        "bm25_index_identity": "lucene-test-index",
        "dense_index_identity": "bge-test-index",
    }
    validate_index_binding(manifest, view)
    manifest["corpus_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        validate_index_binding(manifest, view)
