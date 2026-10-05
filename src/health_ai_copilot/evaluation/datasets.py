"""Local-only adapters for DiagnosisArena and CMB public evaluation snapshots."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..harness.contracts import AnswerSchema, HarnessResponse
from .contracts import CaseScore, DatasetAdapter, EvalCase
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
                metadata={
                    "source_dataset": self.dataset_id,
                    "public": True,
                    "valid_options": tuple(options),
                },
            ))
        if len({case.case_id for case in output}) != len(output):
            raise ValueError("DiagnosisArena source contains duplicate case IDs")
        return tuple(output)

    def score(self, case: EvalCase, response: HarnessResponse) -> CaseScore:
        parsed = response.parsed_answer
        ok = isinstance(parsed, str) and len(parsed) == 1
        valid_options = set(case.metadata.get("valid_options", ()))
        if ok and valid_options and parsed not in valid_options:
            ok = False
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
                metadata={
                    "subcategory": category,
                    "public": True,
                    "valid_options": tuple(options),
                },
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
        valid_options = set(case.metadata.get("valid_options", ()))
        if case.answer_schema is AnswerSchema.SINGLE_CHOICE:
            parse_success = isinstance(parsed, str) and len(parsed) == 1
            if parse_success and valid_options and parsed not in valid_options:
                parse_success = False
            correct = bool(parse_success and parsed == case.gold)
        else:
            parse_success = isinstance(parsed, tuple) and bool(parsed)
            if parse_success and valid_options and not set(parsed).issubset(valid_options):
                parse_success = False
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
        # The frozen ID manifest is the authority. Re-running a locally chosen
        # sampler here could silently substitute a different 1,024-case set.
        return tuple(by_id[item] for item in self.case_ids)


class PreparedViewsAdapter(JsonlPublicAdapter):
    """Join model-visible candidates with evaluator-only labels by frozen ID."""

    def __init__(
        self,
        *,
        dataset_id: str,
        candidate_path: Path,
        scorer_path: Path,
        ids_manifest_path: Path,
        source_revision: str,
        expected_count: int,
        expected_candidate_sha256: str,
        expected_scorer_sha256: str,
        expected_ids_manifest_sha256: str,
        expected_ids_sequence_sha256: str | None = None,
        expected_subset_sha256: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.expected_count = expected_count
        self.candidate_path = Path(candidate_path)
        self.scorer_path = Path(scorer_path)
        self.ids_manifest_path = Path(ids_manifest_path)
        self.source_revision = source_revision
        self.candidate_sha256 = _file_sha256(self.candidate_path)
        self.scorer_sha256 = _file_sha256(self.scorer_path)
        self.ids_manifest_sha256 = _file_sha256(self.ids_manifest_path)
        if self.candidate_sha256 != expected_candidate_sha256:
            raise ValueError("candidate view hash does not match the frozen evaluation manifest")
        if self.scorer_sha256 != expected_scorer_sha256:
            raise ValueError("scorer view hash does not match the frozen evaluation manifest")
        if self.ids_manifest_sha256 != expected_ids_manifest_sha256:
            raise ValueError("ID manifest hash does not match the frozen evaluation manifest")

        id_document = json.loads(self.ids_manifest_path.read_text(encoding="utf-8"))
        if isinstance(id_document, dict):
            raw_ids = id_document.get("ids")
            declared_sequence_sha256 = id_document.get("ids_sequence_sha256")
        else:
            raw_ids = id_document
            declared_sequence_sha256 = None
        if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
            raise ValueError("frozen ID manifest must contain a string ID list")
        self.case_ids = tuple(raw_ids)
        if len(self.case_ids) != expected_count or len(set(self.case_ids)) != expected_count:
            raise ValueError(f"expected {expected_count} unique frozen evaluation IDs")
        sequence_hash = sha256(("\n".join(self.case_ids) + "\n").encode("utf-8")).hexdigest()
        if declared_sequence_sha256 and declared_sequence_sha256 != sequence_hash:
            raise ValueError("ID sequence hash inside the manifest is invalid")
        if expected_ids_sequence_sha256 and sequence_hash != expected_ids_sequence_sha256:
            raise ValueError("ID sequence hash does not match the frozen evaluation manifest")
        self.subset_sha256 = expected_subset_sha256 or sequence_hash
        if self.subset_sha256 != sequence_hash:
            raise ValueError("dataset selection hash does not match its frozen ID sequence")
        self.snapshot_sha256 = sha256(
            f"{self.candidate_sha256}\0{self.scorer_sha256}\0{self.ids_manifest_sha256}".encode()
        ).hexdigest()

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"{path.name}:{line_no} must be a JSON object")
                rows.append(row)
        return rows

    def cases(self) -> tuple[EvalCase, ...]:
        candidates = self._read_jsonl(self.candidate_path)
        scorers = self._read_jsonl(self.scorer_path)
        candidate_by_id = _unique_rows_by_id(candidates, "candidate")
        scorer_by_id = _unique_rows_by_id(scorers, "scorer")
        expected = set(self.case_ids)
        if set(candidate_by_id) != expected or set(scorer_by_id) != expected:
            raise ValueError("candidate/scorer IDs must exactly match the frozen evaluation IDs")
        forbidden = {"answer", "right_option", "gold", "label", "answer_key"}
        output: list[EvalCase] = []
        for case_id in self.case_ids:
            candidate = candidate_by_id[case_id]
            scorer = scorer_by_id[case_id]
            if forbidden.intersection(candidate):
                raise ValueError(f"candidate view contains evaluator-only label fields for {case_id}")
            query = candidate.get("prompt")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"candidate view has no prompt for {case_id}")
            allowed = candidate.get("valid_options")
            if allowed is not None and not isinstance(allowed, list):
                raise ValueError(f"valid_options must be a list for {case_id}")
            gold = canonical_option_set(scorer.get("answer"), allowed=allowed)
            if self.dataset_id == "diagnosisarena":
                if len(gold) != 1:
                    raise ValueError(f"DiagnosisArena case {case_id} must be single-choice")
                schema = AnswerSchema.SINGLE_CHOICE
                expected_gold: str | tuple[str, ...] = gold[0]
            else:
                question_type = str(candidate.get("question_type", ""))
                is_multi = "多项" in question_type or "多选" in question_type or "multi" in question_type.casefold()
                if is_multi:
                    schema = AnswerSchema.MULTI_SELECT
                    expected_gold = gold
                else:
                    if len(gold) != 1:
                        raise ValueError(f"CMB single-choice case {case_id} has {len(gold)} labels")
                    schema = AnswerSchema.SINGLE_CHOICE
                    expected_gold = gold[0]
            output.append(EvalCase(
                case_id=case_id,
                query=query.strip(),
                answer_schema=schema,
                gold=expected_gold,
                metadata={
                    "source_dataset": self.dataset_id,
                    "public": True,
                    "subcategory": scorer.get("subcategory"),
                    "question_type": candidate.get("question_type"),
                    "valid_options": tuple(str(item).upper() for item in (allowed or ())),
                },
            ))
        return tuple(output)

    def score(self, case: EvalCase, response: HarnessResponse) -> CaseScore:
        parsed = response.parsed_answer
        valid_options = set(case.metadata.get("valid_options", ()))
        if case.answer_schema is AnswerSchema.SINGLE_CHOICE:
            parse_success = isinstance(parsed, str) and len(parsed) == 1
            if parse_success and valid_options and parsed not in valid_options:
                parse_success = False
            correct = bool(parse_success and parsed == case.gold)
            metric = "exact_choice_accuracy"
        else:
            parse_success = isinstance(parsed, tuple) and bool(parsed)
            if parse_success and valid_options and not set(parsed).issubset(valid_options):
                parse_success = False
            expected = tuple(case.gold) if isinstance(case.gold, tuple) else (case.gold,)
            correct = bool(parse_success and tuple(sorted(set(parsed))) == tuple(sorted(set(expected))))
            metric = "exact_set_accuracy"
        return CaseScore(
            case_id=case.case_id,
            correct=correct,
            parse_success=parse_success,
            metric=metric,
            category=str(case.metadata.get("subcategory") or "unknown"),
        )


def _unique_rows_by_id(rows: list[dict[str, Any]], view_name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = _stable_case_id(row, len(output))
        if case_id in output:
            raise ValueError(f"duplicate {view_name} ID: {case_id}")
        output[case_id] = row
    return output


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
    payload = ("\n".join(case_ids) + "\n").encode("utf-8")
    return sha256(payload).hexdigest()


class SelectedCaseAdapter:
    """Restrict a frozen adapter to an explicit, hashed case list."""

    def __init__(self, adapter: DatasetAdapter, case_ids: tuple[str, ...]) -> None:
        if not case_ids or any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
            raise ValueError("selected case IDs must be a non-empty tuple of strings")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("selected case IDs must be unique")

        available = {case.case_id: case for case in adapter.cases()}
        missing = set(case_ids) - set(available)
        if missing:
            raise ValueError(f"selected case IDs are absent from frozen dataset: {sorted(missing)[:5]}")

        self._adapter = adapter
        self._cases = tuple(available[case_id] for case_id in case_ids)
        self.selected_case_ids = case_ids
        self.selected_case_ids_sha256 = case_ids_sha256(case_ids)
        self.frozen_subset_sha256 = getattr(adapter, "subset_sha256", None)
        self.subset_sha256 = self.selected_case_ids_sha256

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def cases(self) -> tuple[EvalCase, ...]:
        return self._cases

    def score(self, case: EvalCase, response: HarnessResponse) -> CaseScore:
        return self._adapter.score(case, response)


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
