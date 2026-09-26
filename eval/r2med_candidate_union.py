"""Qrels-blind candidate union and deterministic dual-source rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    lamer_rank: int | None
    crb_rank: int | None
    score: float

    @property
    def both_sources(self) -> bool:
        return self.lamer_rank is not None and self.crb_rank is not None


def fuse_dual_source_rrf(
    lamer_doc_ids: Sequence[str],
    crb_doc_ids: Sequence[str],
    *,
    lamer_weight: float = 1.0,
    crb_weight: float = 1.0,
    k: int = 60,
) -> list[Candidate]:
    """Fuse two ranked lists without relevance labels, preserving source ranks."""
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if lamer_weight < 0 or crb_weight < 0 or (lamer_weight == 0 and crb_weight == 0):
        raise ValueError("RRF weights must be non-negative and not both zero")
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in (*lamer_doc_ids, *crb_doc_ids)):
        raise ValueError("candidate document IDs must be non-empty strings")
    if len(lamer_doc_ids) != len(set(lamer_doc_ids)) or len(crb_doc_ids) != len(set(crb_doc_ids)):
        raise ValueError("each input ranking must be deduplicated")

    lamer_ranks = {doc_id: rank for rank, doc_id in enumerate(lamer_doc_ids, start=1)}
    crb_ranks = {doc_id: rank for rank, doc_id in enumerate(crb_doc_ids, start=1)}
    candidates: list[Candidate] = []
    for doc_id in lamer_ranks.keys() | crb_ranks.keys():
        lamer_rank = lamer_ranks.get(doc_id)
        crb_rank = crb_ranks.get(doc_id)
        score = 0.0
        if lamer_rank is not None:
            score += lamer_weight / (k + lamer_rank)
        if crb_rank is not None:
            score += crb_weight / (k + crb_rank)
        candidates.append(Candidate(doc_id, lamer_rank, crb_rank, score))

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            candidate.lamer_rank if candidate.lamer_rank is not None else float("inf"),
            candidate.crb_rank if candidate.crb_rank is not None else float("inf"),
            candidate.doc_id,
        ),
    )


def source_agreement(candidate: Candidate) -> int:
    """Binary source-agreement feature used only by the gated AAR variant."""
    return int(candidate.both_sources)
