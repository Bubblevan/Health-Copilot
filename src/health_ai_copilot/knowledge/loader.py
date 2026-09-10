"""Deterministic loader for reviewed JSON knowledge cards."""

import json
from collections.abc import Iterable
from pathlib import Path

from ..contracts import KnowledgeCard
from .schema import KnowledgeCardValidationError, knowledge_card_from_dict


class KnowledgeCardLoadError(ValueError):
    """Raised when a knowledge-card directory cannot be loaded safely."""


def _json_files(directory: Path) -> Iterable[Path]:
    return sorted(directory.glob("*.json"), key=lambda path: path.name)


def load_knowledge_cards(directory: str | Path) -> list[KnowledgeCard]:
    """Load every JSON card, failing loudly on the first invalid document."""
    path = Path(directory)
    if not path.exists():
        raise KnowledgeCardLoadError(f"knowledge-card directory does not exist: {path}")
    if not path.is_dir():
        raise KnowledgeCardLoadError(f"knowledge-card path is not a directory: {path}")

    cards: list[KnowledgeCard] = []
    seen_ids: set[str] = set()
    for card_path in _json_files(path):
        try:
            data = json.loads(card_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgeCardLoadError(
                f"invalid JSON in {card_path}: line {exc.lineno}, column {exc.colno}"
            ) from exc
        except OSError as exc:
            raise KnowledgeCardLoadError(f"could not read {card_path}: {exc}") from exc

        try:
            card = knowledge_card_from_dict(data)
        except KnowledgeCardValidationError as exc:
            raise KnowledgeCardLoadError(f"invalid card {card_path}: {exc}") from exc

        if card.id in seen_ids:
            raise KnowledgeCardLoadError(f"duplicate knowledge-card id '{card.id}'")
        seen_ids.add(card.id)
        cards.append(card)

    return sorted(cards, key=lambda card: card.id)
