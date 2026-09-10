"""Validation for source-versioned public knowledge cards."""

from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from ..contracts import KnowledgeCard


class KnowledgeCardValidationError(ValueError):
    """Raised when a knowledge-card document violates the M0 schema."""


_REQUIRED_FIELDS = (
    "id",
    "title",
    "content",
    "source_url",
    "publisher",
    "collected_at",
    "reviewer",
    "version",
    "audience",
    "tags",
)
_OPTIONAL_DATE_FIELDS = ("published_at", "reviewed_at", "expires_at")


def _required_text(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeCardValidationError(
            f"field '{field_name}' must be a non-empty string"
        )
    return value.strip()


def _optional_date(data: dict[str, Any], field_name: str) -> str | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeCardValidationError(
            f"field '{field_name}' must be an ISO date string or null"
        )
    try:
        datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise KnowledgeCardValidationError(
            f"field '{field_name}' must be an ISO date or datetime"
        ) from exc
    return value.strip()


def _required_date(data: dict[str, Any], field_name: str) -> str:
    value = _required_text(data, field_name)
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise KnowledgeCardValidationError(
            f"field '{field_name}' must be an ISO date or datetime"
        ) from exc
    return value


def _string_list(data: dict[str, Any], field_name: str) -> list[str]:
    value = data.get(field_name)
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise KnowledgeCardValidationError(
            f"field '{field_name}' must be a list of non-empty strings"
        )
    return [item.strip() for item in value]


def knowledge_card_from_dict(data: Any) -> KnowledgeCard:
    """Validate one JSON object and convert it to a typed KnowledgeCard."""
    if not isinstance(data, dict):
        raise KnowledgeCardValidationError("knowledge card must be a JSON object")

    missing = [field for field in _REQUIRED_FIELDS if field not in data]
    if missing:
        raise KnowledgeCardValidationError(
            f"missing required fields: {', '.join(missing)}"
        )

    source_url = _required_text(data, "source_url")
    parsed = urlparse(source_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise KnowledgeCardValidationError(
            "field 'source_url' must be an absolute http(s) URL"
        )

    for field_name in ("id", "title", "content", "publisher", "reviewer", "version"):
        _required_text(data, field_name)
    collected_at = _required_date(data, "collected_at")
    if not data["content"].strip():
        raise KnowledgeCardValidationError("field 'content' must not be empty")

    optional_dates = {
        field_name: _optional_date(data, field_name)
        for field_name in _OPTIONAL_DATE_FIELDS
    }

    return KnowledgeCard(
        id=data["id"].strip(),
        title=data["title"].strip(),
        content=data["content"].strip(),
        source_url=source_url,
        publisher=data["publisher"].strip(),
        published_at=optional_dates["published_at"],
        collected_at=collected_at,
        reviewed_at=optional_dates["reviewed_at"],
        reviewer=data["reviewer"].strip(),
        version=data["version"].strip(),
        expires_at=optional_dates["expires_at"],
        audience=_string_list(data, "audience"),
        tags=_string_list(data, "tags"),
    )
