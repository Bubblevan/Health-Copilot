"""Independent synthetic external evidence namespace."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .contracts import _aware, _nonempty, stable_hash

ALLOWED_SYNTHETIC_SOURCE_FAMILIES = frozenset({"PUBLIC_HEALTH", "GUIDELINE", "LITERATURE"})


@dataclass(frozen=True)
class ExternalEvidenceRecord:
    source_id: str
    source_family: str
    publication_time: datetime
    effective_time: datetime | None
    authority_metadata: tuple[tuple[str, str], ...]
    content: str
    retrieval_terms: tuple[str, ...] = ()
    effective_until: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("source_id", "source_family", "content"):
            _nonempty(getattr(self, name), name)
        if self.source_family not in ALLOWED_SYNTHETIC_SOURCE_FAMILIES:
            raise ValueError("synthetic source family is outside the U1 namespace contract")
        _aware(self.publication_time, "publication_time")
        if self.effective_time:
            _aware(self.effective_time, "effective_time")
        if self.effective_until:
            _aware(self.effective_until, "effective_until")
            if self.effective_time is None or self.effective_until <= self.effective_time:
                raise ValueError("effective_until must follow effective_time")
        object.__setattr__(self, "authority_metadata", tuple(sorted(tuple(x) for x in self.authority_metadata)))
        object.__setattr__(self, "retrieval_terms", tuple(sorted({x.casefold() for x in self.retrieval_terms})))

    def to_dict(self) -> dict[str, object]:
        return {"source_id": self.source_id, "source_family": self.source_family,
                "publication_time": self.publication_time.isoformat(),
                "effective_time": self.effective_time.isoformat() if self.effective_time else None,
                "effective_until": (self.effective_until.isoformat()
                                    if self.effective_until else None),
                "authority_metadata": dict(self.authority_metadata), "content": self.content,
                "retrieval_terms": list(self.retrieval_terms)}


@dataclass(frozen=True)
class ExternalEvidenceWorld:
    world_id: str
    version: str
    records: tuple[ExternalEvidenceRecord, ...]

    def __post_init__(self) -> None:
        _nonempty(self.world_id, "world_id")
        _nonempty(self.version, "version")
        rows = tuple(sorted(self.records, key=lambda row: row.source_id))
        if len({row.source_id for row in rows}) != len(rows):
            raise ValueError("external source IDs must be unique")
        object.__setattr__(self, "records", rows)

    @classmethod
    def from_records(cls, world_id: str, version: str,
                     records: Iterable[ExternalEvidenceRecord]) -> ExternalEvidenceWorld:
        return cls(world_id, version, tuple(records))

    @property
    def world_hash(self) -> str:
        return stable_hash({"world_id": self.world_id, "version": self.version,
                            "records": [row.to_dict() for row in self.records]})

    def snapshot_hash(self, as_of_time: datetime, source_families: tuple[str, ...]) -> str:
        _aware(as_of_time, "as_of_time")
        families = set(source_families)
        rows = [row.to_dict() for row in self.records
                if row.source_family in families
                and row.publication_time <= as_of_time
                and (row.effective_time is None or row.effective_time <= as_of_time)
                and (row.effective_until is None or as_of_time < row.effective_until)]
        return stable_hash({"world_id": self.world_id, "version": self.version, "records": rows})

    def retrieve(
        self, query: str, source_families: tuple[str, ...], *, as_of_time: datetime
    ) -> tuple[ExternalEvidenceRecord, ...]:
        """Deterministic token/phrase match, constrained to the given families."""
        _aware(as_of_time, "as_of_time")
        text = query.casefold()
        families = set(source_families)
        approved = re.search(
            r"approved family\s*:\s*(public_health|guideline|literature)", text
        )
        if approved:
            families.intersection_update({approved.group(1).upper()})
        hits = [row for row in self.records
                if row.source_family in families
                and row.publication_time <= as_of_time
                and (row.effective_time is None or row.effective_time <= as_of_time)
                and (row.effective_until is None or as_of_time < row.effective_until)
                and (not row.retrieval_terms or any(term in text for term in row.retrieval_terms))]
        return tuple(hits)
