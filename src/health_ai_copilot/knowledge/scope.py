"""Reviewed, versioned capability boundary for the closed KnowledgeCard corpus."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import KnowledgeCard


class KnowledgeScopeLoadError(ValueError):
    """Raised when the reviewed capability manifest is invalid."""


@dataclass(frozen=True)
class CapabilityTopic:
    """One reviewed capability topic and its explicit source membership."""

    id: str
    description: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeScope:
    """Versioned source of truth for what the closed retrieval tool can cover."""

    scope_id: str
    version: str
    knowledge_pack_version: str
    domain: str
    audiences: tuple[str, ...]
    topics: tuple[CapabilityTopic, ...]

    @property
    def topic_ids(self) -> frozenset[str]:
        return frozenset(topic.id for topic in self.topics)

    def compact_summary(self) -> str:
        """Compact reviewed capability text for the M3 tool/policy prompts."""
        descriptions = "; ".join(topic.description for topic in self.topics)
        return (
            f"Closed reviewed corpus: {self.domain}. Coverage: {descriptions}. "
            "It is not web search and cannot retrieve arbitrary medical, legal, "
            "insurance, travel, or other information outside these reviewed topics."
        )


def load_knowledge_scope(
    path: str | Path, cards: list[KnowledgeCard] | tuple[KnowledgeCard, ...]
) -> KnowledgeScope:
    """Load and validate a reviewed manifest against the actual product cards."""
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeScopeLoadError(
            f"invalid knowledge scope JSON: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise KnowledgeScopeLoadError(f"could not read knowledge scope: {exc}") from exc
    if not isinstance(data, dict):
        raise KnowledgeScopeLoadError("knowledge scope must be a JSON object")
    return knowledge_scope_from_dict(data, cards)


def knowledge_scope_from_dict(
    data: dict[str, Any], cards: list[KnowledgeCard] | tuple[KnowledgeCard, ...]
) -> KnowledgeScope:
    """Validate a manifest payload; exposed for deterministic unit tests."""
    required_text = ("scope_id", "version", "knowledge_pack_version", "domain")
    values: dict[str, str] = {}
    for field in required_text:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeScopeLoadError(f"knowledge scope field '{field}' must be non-empty")
        values[field] = value.strip()
    audiences = data.get("audiences")
    if not isinstance(audiences, list) or not audiences or not all(
        isinstance(value, str) and value.strip() for value in audiences
    ):
        raise KnowledgeScopeLoadError("knowledge scope audiences must be a non-empty string list")
    raw_topics = data.get("topics")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise KnowledgeScopeLoadError("knowledge scope topics must be a non-empty list")
    known_source_ids = {card.id for card in cards}
    topic_ids: set[str] = set()
    covered_source_ids: set[str] = set()
    topics: list[CapabilityTopic] = []
    for raw_topic in raw_topics:
        if not isinstance(raw_topic, dict):
            raise KnowledgeScopeLoadError("knowledge scope topic must be an object")
        topic_id = raw_topic.get("id")
        description = raw_topic.get("description")
        source_ids = raw_topic.get("source_ids")
        if not isinstance(topic_id, str) or not topic_id.strip():
            raise KnowledgeScopeLoadError("knowledge scope topic id must be non-empty")
        if topic_id in topic_ids:
            raise KnowledgeScopeLoadError(f"duplicate knowledge scope topic id '{topic_id}'")
        if not isinstance(description, str) or not description.strip():
            raise KnowledgeScopeLoadError(f"knowledge scope topic '{topic_id}' has empty description")
        if not isinstance(source_ids, list) or not source_ids:
            raise KnowledgeScopeLoadError(f"knowledge scope topic '{topic_id}' has no sources")
        if not all(isinstance(source_id, str) and source_id.strip() for source_id in source_ids):
            raise KnowledgeScopeLoadError(f"knowledge scope topic '{topic_id}' has invalid source IDs")
        if len(set(source_ids)) != len(source_ids):
            raise KnowledgeScopeLoadError(f"knowledge scope topic '{topic_id}' duplicates a source ID")
        unknown = set(source_ids) - known_source_ids
        if unknown:
            raise KnowledgeScopeLoadError(
                f"knowledge scope topic '{topic_id}' has unknown sources: {sorted(unknown)}"
            )
        topic_ids.add(topic_id)
        covered_source_ids.update(source_ids)
        topics.append(CapabilityTopic(topic_id, description.strip(), tuple(source_ids)))
    uncovered = known_source_ids - covered_source_ids
    if uncovered:
        raise KnowledgeScopeLoadError(
            f"knowledge scope leaves product cards uncovered: {sorted(uncovered)}"
        )
    return KnowledgeScope(
        scope_id=values["scope_id"],
        version=values["version"],
        knowledge_pack_version=values["knowledge_pack_version"],
        domain=values["domain"],
        audiences=tuple(audience.strip() for audience in audiences),
        topics=tuple(topics),
    )
