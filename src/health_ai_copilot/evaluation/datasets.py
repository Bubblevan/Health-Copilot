"""Local-only adapters for DiagnosisArena and CMB public evaluation snapshots."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..harness.contracts import AnswerSchema, HarnessResponse
from .contracts import CaseScore, EvalCase
from .parser import canonical_option_set


class DatasetNotReady(RuntimeError):
    pass


class JsonlPublicAdapter:
    dataset_id = ""
    expected_count: int | None = None

    def __init__(self, path: Path, *, source_revision: str | None = None) -> None:
        self.path = Path(path)
        self.source_revision = source_revision or "UNPINNED"
        self.snapshot_sha256 = _file_sha256(self.path) if self.path.is_file() else None

    def _rows(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            raise DatasetNotReady(f"missing public dataset snapshot: {self.path}")
        current_hash = _file_sha256(self.path)
        if self.snapshot_sha256 is not None and current_hash != self.snapshot_sha256:
            raise ValueError("dataset snapshot changed after adapter construction")
        self.snapshot_sha256 = current_hash
        if self.path.suffix.casefold() == ".json":
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value = value.get("data", value.get("test"))
            if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
                raise ValueError("JSON dataset snapshot must be an array of row objects")
            rows = value
            if self.expected_count is not None and len(rows) != self.expected_count:
                raise ValueError(f"expected {self.expected_count} rows, found {len(rows)}")
            return rows
        rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"row {line_no} must be a JSON object")
                    rows.append(value)
        if self.expected_count is not None and len(rows) != self.expected_count:
            raise ValueError(f"expected {self.expected_count} rows, found {len(rows)}")
        return rows


class DiagnosisArenaAdapter(JsonlPublicAdapter):
    dataset_id = "diagnosisarena"
    expected_count = 915

    def cases(self) -> tuple[EvalCase, ...]:
        output = []
        for index, row in enumerate(self._rows()):
            case_id = _stable_case_id(row, index)
            details = [
                ("Case Information", _first(row, "case_information", "Case Information")),
                ("Physical Examination", _first(row, "physical_examination", "Physical Examination")),
                ("Diagnostic Tests", _first(row, "diagnostic_tests", "Diagnostic Tests")),
            ]
            options = _options(_first(row, "options", "Options"))
            query = "\n\n".join(f"{title}:\n{value}" for title, value in details if value)
            if options:
                query += "\n\nOptions:\n" + "\n".join(f"{key}. {value}" for key, value in options.items())
            gold = canonical_option_set(_first(row, "right_option", "Right Option", "answer"),
                                       allowed=options or None)
            if not query.strip():
                raise ValueError(f"DiagnosisArena case {case_id} has no visible question fields")
            if len(gold) != 1:
                raise ValueError(f"DiagnosisArena case {case_id} must have one correct option")
            output.append(EvalCase(
                case_id=case_id,
                query=query.strip(),
                answer_schema=AnswerSchema.SINGLE_CHOICE,
                gold=gold[0],
                metadata={"source_dataset": self.dataset_id, "public": True},
            ))
        if len({case.case_id for case in output}) != len(output):
            raise ValueError("DiagnosisArena source contains duplicate case IDs")
        return tuple(output)

    def score(self, case: EvalCase, response: HarnessResponse) -> CaseScore:
        parsed = response.parsed_answer
        ok = isinstance(parsed, str) and len(parsed) == 1
        return CaseScore(
            case_id=case.case_id,
            correct=bool(ok and parsed == case.gold),
            parse_success=ok,
            metric="exact_choice_accuracy",
        )


class CMBAdapter(JsonlPublicAdapter):
    dataset_id = "cmb-exam"
    expected_count = 11_200
    expected_category_count = 28

    def cases(self) -> tuple[EvalCase, ...]:
        output = []
        for index, row in enumerate(self._rows()):
            case_id = _stable_case_id(row, index)
            question = str(_first(row, "question", "Question", "problem") or "").strip()
            options = _options(_first(row, "options", "Options"))
            if options:
                question += "\n\nOptions:\n" + "\n".join(
                    f"{key}. {value}" for key, value in options.items()
                )
            gold = canonical_option_set(_first(row, "answer", "Answer", "label"),
                                       allowed=options or None)
            category = str(_first(row, "subcategory", "subject", "category", "exam_type") or "unknown")
            schema = AnswerSchema.SINGLE_CHOICE if len(gold) == 1 else AnswerSchema.MULTI_SELECT
            if not question:
                raise ValueError(f"CMB case {case_id} has no question")
            output.append(EvalCase(
                case_id=case_id,
                query=question,
                answer_schema=schema,
                gold=gold[0] if schema is AnswerSchema.SINGLE_CHOICE else gold,
                metadata={"subcategory": category, "public": True},
            ))
        categories = {str(case.metadata["subcategory"]) for case in output}
        if self.expected_category_count is not None and len(categories) != self.expected_category_count:
            raise ValueError(
                f"expected {self.expected_category_count} CMB subcategories, found {len(categories)}"
            )
        if len({case.case_id for case in output}) != len(output):
            raise ValueError("CMB source contains duplicate case IDs")
        return tuple(output)

    def score(self, case: EvalCase, response: HarnessResponse) -> CaseScore:
        parsed = response.parsed_answer
        if case.answer_schema is AnswerSchema.SINGLE_CHOICE:
            parse_success = isinstance(parsed, str) and len(parsed) == 1
            correct = bool(parse_success and parsed == case.gold)
        else:
            parse_success = isinstance(parsed, tuple) and bool(parsed)
            expected = tuple(case.gold) if isinstance(case.gold, tuple) else (case.gold,)
            correct = bool(parse_success and tuple(sorted(set(parsed))) == tuple(sorted(set(expected))))
        return CaseScore(
            case_id=case.case_id,
            correct=correct,
            parse_success=parse_success,
            metric="exact_choice_accuracy" if case.answer_schema is AnswerSchema.SINGLE_CHOICE
            else "exact_set_accuracy",
            category=str(case.metadata.get("subcategory", "unknown")),
        )


class CMBCommon1024Adapter(CMBAdapter):
    dataset_id = "cmb-common-1024"
    expected_count = None

    def __init__(
        self,
        path: Path,
        *,
        source_revision: str | None,
        case_ids: tuple[str, ...],
        selection_seed: int,
        expected_subset_sha256: str,
    ) -> None:
        super().__init__(path, source_revision=source_revision)
        if len(case_ids) != 1024 or len(set(case_ids)) != 1024:
            raise ValueError("CMB-COMMON-1024 requires exactly 1024 unique frozen IDs")
        self.case_ids = tuple(case_ids)
        self.selection_seed = int(selection_seed)
        self.expected_subset_sha256 = expected_subset_sha256
        self.subset_sha256 = expected_subset_sha256
        actual_hash = case_ids_sha256(self.case_ids)
        if expected_subset_sha256 != actual_hash:
            raise ValueError("frozen CMB subset hash does not match case IDs")

    def cases(self) -> tuple[EvalCase, ...]:
        all_cases = super().cases()
        by_id = {case.case_id: case for case in all_cases}
        if not set(self.case_ids).issubset(by_id):
            raise ValueError("frozen CMB subset references IDs missing from its source snapshot")
        expected_order = stratified_case_ids(
            all_cases, size=1024, seed=self.selection_seed,
        )
        if self.case_ids != expected_order:
            raise ValueError("frozen CMB IDs do not match the recorded deterministic selection protocol")
        return tuple(by_id[item] for item in self.case_ids)


def stratified_case_ids(
    cases: tuple[EvalCase, ...], *, size: int, seed: int,
) -> tuple[str, ...]:
    """Deterministic even-allocation subset selection; output remains evaluator-only."""
    if size <= 0 or size > len(cases):
        raise ValueError("subset size must be within the available case count")
    buckets: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        buckets[str(case.metadata.get("subcategory", "unknown"))].append(case)
    names = sorted(buckets)
    allocation = {name: size // len(names) for name in names}
    for name in names[:size % len(names)]:
        allocation[name] += 1
    selected: list[EvalCase] = []
    for index, name in enumerate(names):
        bucket = sorted(buckets[name], key=lambda case: case.case_id)
        count = allocation[name]
        if len(bucket) < count:
            raise ValueError(f"subcategory {name} has only {len(bucket)} rows; needs {count}")
        selected.extend(random.Random(seed + index).sample(bucket, count))
    if len(selected) != size:
        raise AssertionError("stratified subset selection returned the wrong size")
    return tuple(case.case_id for case in selected)


def case_ids_sha256(case_ids: tuple[str, ...]) -> str:
    payload = "\n".join(case_ids).encode("utf-8")
    return sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_case_id(row: dict[str, Any], index: int) -> str:
    raw = _first(row, "case_id", "id", "ID", "question_id", "Question ID")
    return str(raw).strip() if raw is not None else f"row-{index:05d}"


def _first(row: dict[str, Any], *keys: str) -> Any:
    return next((row[key] for key in keys if key in row and row[key] not in (None, "")), None)


def _options(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key).strip().upper(): str(item).strip() for key, item in value.items()}
    if isinstance(value, list):
        result: dict[str, str] = {}
        for index, item in enumerate(value):
            if isinstance(item, dict):
                key = str(item.get("key", item.get("label", chr(65 + index)))).upper()
                result[key] = str(item.get("text", item.get("value", "")))
            else:
                result[chr(65 + index)] = str(item)
        return result
    return {}
