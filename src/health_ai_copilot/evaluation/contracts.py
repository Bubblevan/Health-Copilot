"""Evaluator-plane contracts. Gold labels never cross into HarnessRequest."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..harness.contracts import AnswerSchema, HarnessResponse


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    answer_schema: AnswerSchema
    gold: str | tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.query.strip():
            raise ValueError("evaluation case needs a stable ID and query")
        if not isinstance(self.answer_schema, AnswerSchema):
            object.__setattr__(self, "answer_schema", AnswerSchema(self.answer_schema))


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    correct: bool
    parse_success: bool
    metric: str
    category: str | None = None


class DatasetAdapter(Protocol):
    dataset_id: str
    source_revision: str

    def cases(self) -> Iterable[EvalCase]:
        ...

    def score(self, case: EvalCase, response: HarnessResponse) -> CaseScore:
        ...
