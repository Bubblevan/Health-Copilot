"""Fail-closed admission and identity helpers for the E5 guideline corpus."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

_REVIEW_STATUSES = frozenset({"PENDING", "READY_FOR_OWNER_REVIEW", "APPROVED", "REJECTED"})
_GUIDELINE_FAMILY = "reviewed_guideline"


@dataclass(frozen=True, slots=True)
class RecommendationBlock:
    """A manually/structurally extracted block that stays within one recommendation."""

    section_path: tuple[str, ...]
    recommendation_id: str
    paragraphs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuidelineChunk:
    """A deterministic retrieval chunk with source and section provenance."""

    chunk_id: str
    source_id: str
    source_family: str
    section_path: tuple[str, ...]
    recommendation_id: str
    text: str
    text_sha256: str
    token_count: int
    char_count: int
    source_raw_sha256: str
    extractor_version: str
    chunker_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "section_path": list(self.section_path),
            "recommendation_id": self.recommendation_id,
            "text": self.text,
            "text_sha256": self.text_sha256,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "source_raw_sha256": self.source_raw_sha256,
            "extractor_version": self.extractor_version,
            "chunker_version": self.chunker_version,
        }


@dataclass(frozen=True, slots=True)
class CorpusView:
    """A pre-filtered family view whose identity can be bound to an index."""

    corpus_view_id: str
    source_families: tuple[str, ...]
    chunks: tuple[GuidelineChunk, ...]
    corpus_sha256: str


def approved_guideline_sources(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return only owner-approved guideline records; malformed states fail closed."""
    if manifest.get("source_family") != _GUIDELINE_FAMILY:
        raise ValueError("guideline manifest source_family must be reviewed_guideline")
    rows = manifest.get("sources")
    if not isinstance(rows, list):
        raise TypeError("guideline manifest sources must be a list")
    admitted: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for source in rows:
        if not isinstance(source, Mapping):
            raise TypeError("guideline source rows must be objects")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or source_id in seen_ids:
            raise ValueError("guideline source_id must be non-empty and unique")
        seen_ids.add(source_id)
        if source.get("source_family") != _GUIDELINE_FAMILY:
            raise ValueError("guideline source cannot be reclassified into another family")
        review_status = source.get("review_status")
        owner_status = source.get("owner_review_status")
        if review_status not in _REVIEW_STATUSES or owner_status not in _REVIEW_STATUSES:
            raise ValueError("guideline source has an unsupported review status")
        if (review_status == "APPROVED") != (owner_status == "APPROVED"):
            raise ValueError("automated and owner approval states must agree before admission")
        if review_status == "APPROVED":
            if not source.get("owner_reviewed_at"):
                raise ValueError("approved guideline source requires owner_reviewed_at")
            admitted.append(source)
    return tuple(admitted)


def verify_raw_source(source: Mapping[str, Any], raw_path: Path) -> str:
    """Verify retrieved bytes against the frozen SHA-256 in a source record."""
    expected = source.get("raw_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("source raw_sha256 must be a SHA-256 digest")
    observed = sha256(raw_path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"raw source hash mismatch for {source.get('source_id', '<unknown>')}")
    return observed


def build_guideline_chunks(
    *,
    manifest: Mapping[str, Any],
    raw_paths: Mapping[str, Path],
    blocks_by_source: Mapping[str, Sequence[RecommendationBlock]],
    tokenize: Callable[[str], Sequence[object]],
    extractor_version: str,
    chunker_version: str,
    target_tokens: int = 384,
    hard_max_tokens: int = 480,
) -> tuple[GuidelineChunk, ...]:
    """Pack paragraphs without crossing a recommendation or admitting unapproved sources."""
    if target_tokens <= 0 or hard_max_tokens < target_tokens:
        raise ValueError("chunk token limits must satisfy 0 < target <= hard maximum")
    approved = {str(source["source_id"]): source for source in approved_guideline_sources(manifest)}
    if set(blocks_by_source) - set(approved):
        raise ValueError("only owner-approved sources may enter the active guideline corpus")
    if set(raw_paths) != set(blocks_by_source):
        raise ValueError("raw_paths must exactly match sources with extracted blocks")

    chunks: list[GuidelineChunk] = []
    for source_id in sorted(blocks_by_source):
        source = approved[source_id]
        raw_hash = verify_raw_source(source, raw_paths[source_id])
        source_blocks = blocks_by_source[source_id]
        seen_recommendations: set[tuple[tuple[str, ...], str]] = set()
        part_by_section: dict[tuple[tuple[str, ...], str], int] = {}
        for block in source_blocks:
            if not block.section_path or any(not part.strip() for part in block.section_path):
                raise ValueError("section_path must contain non-empty heading components")
            if not block.recommendation_id.strip():
                raise ValueError("recommendation_id must be non-empty")
            section_key = (block.section_path, block.recommendation_id)
            if section_key in seen_recommendations:
                raise ValueError("recommendation blocks must be consolidated before chunking")
            seen_recommendations.add(section_key)
            normalized_paragraphs = tuple(" ".join(text.split()) for text in block.paragraphs)
            if not normalized_paragraphs or any(not paragraph for paragraph in normalized_paragraphs):
                raise ValueError("empty recommendation paragraphs are not valid chunks")

            packed: list[str] = []
            for paragraph in normalized_paragraphs:
                paragraph_tokens = len(tokenize(paragraph))
                if paragraph_tokens == 0:
                    raise ValueError("tokenizer produced an empty paragraph")
                if paragraph_tokens > hard_max_tokens:
                    raise ValueError("a paragraph exceeds hard_max_tokens; split it at a safe boundary")
                candidate = "\n\n".join((*packed, paragraph))
                candidate_tokens = len(tokenize(candidate))
                if packed and candidate_tokens > target_tokens:
                    chunks.append(
                        _make_chunk(
                            source=source,
                            source_id=source_id,
                            section_path=block.section_path,
                            recommendation_id=block.recommendation_id,
                            part=part_by_section.get(section_key, 0),
                            text="\n\n".join(packed),
                            token_count=len(tokenize("\n\n".join(packed))),
                            source_raw_sha256=raw_hash,
                            extractor_version=extractor_version,
                            chunker_version=chunker_version,
                        )
                    )
                    part_by_section[section_key] = part_by_section.get(section_key, 0) + 1
                    packed = [paragraph]
                else:
                    packed.append(paragraph)
            if packed:
                text = "\n\n".join(packed)
                token_count = len(tokenize(text))
                if token_count > hard_max_tokens:
                    raise ValueError("chunk exceeds hard_max_tokens")
                chunks.append(
                    _make_chunk(
                        source=source,
                        source_id=source_id,
                        section_path=block.section_path,
                        recommendation_id=block.recommendation_id,
                        part=part_by_section.get(section_key, 0),
                        text=text,
                        token_count=token_count,
                        source_raw_sha256=raw_hash,
                        extractor_version=extractor_version,
                        chunker_version=chunker_version,
                    )
                )
    if any(not chunk.text.strip() for chunk in chunks):
        raise ValueError("guideline corpus contains an empty chunk")
    return tuple(chunks)


def build_corpus_view(
    chunks: Sequence[GuidelineChunk], *, corpus_view_id: str, source_families: Sequence[str]
) -> CorpusView:
    """Build family-scoped input before ranking; never filter a global top-k afterward."""
    families = tuple(sorted(set(source_families)))
    if not corpus_view_id.strip() or not families:
        raise ValueError("corpus view id and source families must be non-empty")
    selected = tuple(chunk for chunk in chunks if chunk.source_family in families)
    if any(chunk.source_family not in families for chunk in selected):
        raise AssertionError("corpus view contains an undeclared source family")
    canonical = [
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "source_family": chunk.source_family,
            "text_sha256": chunk.text_sha256,
            "source_raw_sha256": chunk.source_raw_sha256,
        }
        for chunk in sorted(selected, key=lambda item: item.chunk_id)
    ]
    corpus_hash = sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return CorpusView(corpus_view_id, families, selected, corpus_hash)


def validate_index_binding(index_manifest: Mapping[str, Any], view: CorpusView) -> None:
    """Fail closed unless BM25 and dense indexes both bind this exact corpus view."""
    if index_manifest.get("corpus_view_id") != view.corpus_view_id:
        raise ValueError("index manifest corpus_view_id does not match the requested view")
    if index_manifest.get("corpus_sha256") != view.corpus_sha256:
        raise ValueError("index manifest corpus_sha256 does not match the requested view")
    if index_manifest.get("source_families") != list(view.source_families):
        raise ValueError("index manifest source_families do not match the requested view")
    for name in ("bm25_index_identity", "dense_index_identity"):
        value = index_manifest.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"index manifest is missing {name}")


def validate_ranked_chunk_ids(view: CorpusView, ranked_chunk_ids: Sequence[str]) -> None:
    """Reject out-of-view results instead of silently post-filtering a ranked list."""
    available = {chunk.chunk_id for chunk in view.chunks}
    unknown = set(ranked_chunk_ids) - available
    if unknown:
        raise ValueError("ranking contains chunks outside its prebuilt corpus view")


def _make_chunk(
    *,
    source: Mapping[str, Any],
    source_id: str,
    section_path: tuple[str, ...],
    recommendation_id: str,
    part: int,
    text: str,
    token_count: int,
    source_raw_sha256: str,
    extractor_version: str,
    chunker_version: str,
) -> GuidelineChunk:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("cannot create an empty guideline chunk")
    text_hash = sha256(normalized_text.encode("utf-8")).hexdigest()
    identity = {
        "source_id": source_id,
        "source_raw_sha256": source_raw_sha256,
        "section_path": list(section_path),
        "recommendation_id": recommendation_id,
        "part": part,
        "text_sha256": text_hash,
    }
    chunk_id = "who-guideline-" + sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return GuidelineChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_family=str(source["source_family"]),
        section_path=section_path,
        recommendation_id=recommendation_id,
        text=normalized_text,
        text_sha256=text_hash,
        token_count=token_count,
        char_count=len(normalized_text),
        source_raw_sha256=source_raw_sha256,
        extractor_version=extractor_version,
        chunker_version=chunker_version,
    )


__all__ = [
    "CorpusView",
    "GuidelineChunk",
    "RecommendationBlock",
    "approved_guideline_sources",
    "build_corpus_view",
    "build_guideline_chunks",
    "validate_index_binding",
    "validate_ranked_chunk_ids",
    "verify_raw_source",
]
