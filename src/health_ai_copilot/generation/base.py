"""Minimal generator contract used by the M0 pipeline."""

from collections.abc import Sequence
from typing import Protocol

from ..contracts import Evidence, GenerationDraft
from ..runtime.context import RunContext


class GenerationError(RuntimeError):
    """Raised when a generator cannot produce a valid structured draft."""


class Generator(Protocol):
    """Generate only from the evidence supplied by the pipeline."""

    def generate(
        self,
        question: str,
        evidence: Sequence[Evidence],
        *,
        runtime: RunContext | None = None,
    ) -> GenerationDraft:
        ...
