"""Timezone-aware deterministic clocks used by generated synthetic worlds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

EPOCH = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


def subject_epoch(persona_seed: int) -> datetime:
    return EPOCH + timedelta(days=persona_seed % 240)


def event_time(persona_seed: int, day_offset: int) -> datetime:
    return subject_epoch(persona_seed) + timedelta(days=day_offset)


def boundary_triplet(persona_seed: int, day_offset: int = 10) -> tuple[datetime, datetime, datetime]:
    boundary = event_time(persona_seed, day_offset)
    return boundary - timedelta(seconds=1), boundary, boundary + timedelta(seconds=1)
