"""Generic retrieval documents, deliberately separate from reviewed product cards."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..contracts import KnowledgeCard


@dataclass(frozen=True)
class RetrievalDocument:
    """A corpus unit for retrieval; it carries no reviewed-product provenance claim."""

    id: str
    text: str
    title: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("retrieval document id must be non-empty")
        if not self.text.strip() and not (self.title or "").strip():
            raise ValueError("retrieval document needs text or title")


def document_from_knowledge_card(card: KnowledgeCard) -> RetrievalDocument:
    """One-way product adapter retaining trusted ID/content/source metadata."""

    return RetrievalDocument(
        id=card.id,
        title=card.title,
        text=card.content,
        metadata={"source_url": card.source_url, "tags": tuple(card.tags)},
    )
