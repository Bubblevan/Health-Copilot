"""Subject-scoped synthetic patient state with fail-closed temporal snapshots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .contracts import PatientRecordType, _aware, _nonempty, stable_hash


@dataclass(frozen=True)
class PatientStateRecord:
    record_id: str
    subject_id: str
    timestamp: datetime
    record_type: PatientRecordType | str
    source_provenance: str
    content: str
    retrieval_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("record_id", "subject_id", "source_provenance", "content"):
            _nonempty(getattr(self, name), name)
        _aware(self.timestamp, "record timestamp")
        object.__setattr__(self, "record_type", PatientRecordType(self.record_type))
        terms = tuple(sorted({term.casefold() for term in self.retrieval_terms}))
        object.__setattr__(self, "retrieval_terms", terms)

    def to_dict(self) -> dict[str, object]:
        return {"record_id": self.record_id, "subject_id": self.subject_id,
                "timestamp": self.timestamp.isoformat(), "record_type": self.record_type.value,
                "source_provenance": self.source_provenance, "content": self.content,
                "retrieval_terms": list(self.retrieval_terms)}


class PatientStateStore:
    """Small interface prototype; no persistence or production Memory binding."""

    def __init__(self, records: Iterable[PatientStateRecord] = ()) -> None:
        rows = tuple(records)
        if len({item.record_id for item in rows}) != len(rows):
            raise ValueError("patient state record IDs must be unique")
        self._records = tuple(sorted(rows, key=lambda row: (row.timestamp, row.record_id)))

    @property
    def records(self) -> tuple[PatientStateRecord, ...]:
        return self._records

    def snapshot(self, subject_id: str, as_of_time: datetime) -> tuple[PatientStateRecord, ...]:
        _nonempty(subject_id, "subject_id")
        _aware(as_of_time, "as_of_time")
        selected = tuple(row for row in self._records
                         if row.subject_id == subject_id and row.timestamp <= as_of_time)
        if any(row.timestamp > as_of_time for row in selected):
            raise RuntimeError("temporal snapshot invariant violated")
        return selected

    def snapshot_hash(self, subject_id: str, as_of_time: datetime) -> str:
        return stable_hash([row.to_dict() for row in self.snapshot(subject_id, as_of_time)])
