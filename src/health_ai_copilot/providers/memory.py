"""Read-only longitudinal memory provider contract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class MemoryFact:
    fact_id: str
    text: str
    observed_at: datetime | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.fact_id.strip():
            raise ValueError("memory fact requires a stable ID")
        if not isinstance(self.text, str):
            raise TypeError("memory fact text must be a string")


@dataclass(frozen=True)
class MemoryResult:
    facts: tuple[MemoryFact, ...] = ()
    trace: tuple[Mapping[str, str], ...] = ()
    cost: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        facts = tuple(self.facts)
        if len({fact.fact_id for fact in facts}) != len(facts):
            raise ValueError("memory fact IDs must be unique")
        object.__setattr__(self, "facts", facts)


class MemoryProvider(Protocol):
    async def read(
        self,
        *,
        subject_id: str,
        query: str,
        as_of_time: datetime,
    ) -> MemoryResult:
        ...


class FinalMemoryProvider:
    """Read adapter over the repository's policy-governed M10 MemoryStore."""

    def __init__(self, store, *, top_k: int = 8) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.store = store
        self.top_k = top_k

    async def read(
        self,
        *,
        subject_id: str,
        query: str,
        as_of_time: datetime,
    ) -> MemoryResult:
        if as_of_time.tzinfo is None or as_of_time.utcoffset() is None:
            raise ValueError("as_of_time must be timezone-aware")
        from ..runtime.memory import MemoryQuery

        rows = await asyncio.to_thread(
            self.store.query,
            MemoryQuery(
                scope_id=subject_id,
                text=query,
                now=as_of_time,
                top_k=self.top_k,
            ),
        )
        facts = tuple(
            MemoryFact(
                fact_id=str(row.memory_id),
                text=(row.value if isinstance(row.value, str) else json.dumps(
                    row.value, ensure_ascii=False, sort_keys=True,
                )),
                observed_at=(
                    datetime.fromisoformat(row.valid_from or row.created_at)
                    if row.valid_from or row.created_at else None
                ),
                source_id=str(row.memory_id),
            )
            for row in rows
        )
        return MemoryResult(
            facts=facts,
            trace=({"event": "m10_memory_store_read", "snapshot_time": as_of_time.isoformat()},),
            cost={"memory_reads": 1, "facts": len(facts)},
        )
