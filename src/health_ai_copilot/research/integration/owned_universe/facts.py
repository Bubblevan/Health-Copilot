"""Layer-A project-owned synthetic fact identifiers and validity."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256

from .schema import (
    CONTENT_ORIGIN,
    FactLocation,
    FactValidityInterval,
    LatentFact,
)


def stable_code(seed: int, namespace: str, width: int = 8) -> str:
    digest = sha256(f"u2e-v1|{namespace}|{seed}".encode()).hexdigest().upper()
    return digest[:width]


def key_token(seed: int, namespace: str = "slot") -> str:
    return f"SYNKEY-{stable_code(seed, namespace)}"


def value_token(seed: int, namespace: str = "answer") -> str:
    return f"SYNVAL-{stable_code(seed, namespace, 10)}"


def make_fact(
    *, fact_id: str, value: str, location: FactLocation,
    source_artifact_id: str, valid_from: datetime,
    valid_until: datetime | None = None, revision_of: str | None = None,
) -> LatentFact:
    return LatentFact(
        fact_id=fact_id, value=value, location=location,
        source_artifact_id=source_artifact_id,
        validity=FactValidityInterval(valid_from, valid_until),
        revision_of=revision_of, content_origin=CONTENT_ORIGIN,
    )
