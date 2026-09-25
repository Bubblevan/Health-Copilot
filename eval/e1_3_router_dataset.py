"""Question-only data and historical-arm loaders for the exposed MIRAGE router sprint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXED_ACTIONS = ("closed_book", "rag_bm25", "rag_medcpt")
HISTORICAL_ARMS = (*FIXED_ACTIONS, "cheap_router", "jev_router")


@dataclass(frozen=True)
class RouterCase:
    case_id: str
    subdataset: str
    question: str


@dataclass(frozen=True)
class ArmOutcome:
    case_id: str
    subdataset: str
    is_correct: bool
    answer_input_tokens: int | None
    context_characters: int | None
    component_latency_proxy_ms: float | None
    retrieval_calls: int
    status: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def case_order_sha256(case_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(case_ids) + "\n").encode("utf-8")).hexdigest()


def load_cases(path: Path) -> list[RouterCase]:
    """Load only the question field; answer options are deliberately never read."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("MIRAGE TEST artifact must map subdatasets to case mappings")

    cases: list[RouterCase] = []
    for subdataset, records in payload.items():
        if not isinstance(records, dict):
            raise TypeError(f"MIRAGE subdataset {subdataset!r} must be a case mapping")
        for source_id, record in records.items():
            question = record.get("question") if isinstance(record, dict) else None
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question for {subdataset}:{source_id}")
            cases.append(
                RouterCase(
                    case_id=f"{subdataset}:{source_id}",
                    subdataset=str(subdataset),
                    question=question,
                )
            )
    cases.sort(key=lambda case: case.case_id)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate MIRAGE case IDs")
    return cases


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object in {path} at line {line_number}")
            rows.append(row)
    return rows


def historical_row_is_correct(row: dict[str, Any]) -> bool:
    """Provider failures and any non-completed result are incorrect by definition."""
    return row.get("status") == "completed" and row.get("is_correct") is True


def load_historical_arms(
    scratch_root: Path, case_ids: list[str]
) -> dict[str, dict[str, ArmOutcome]]:
    expected = set(case_ids)
    arms: dict[str, dict[str, ArmOutcome]] = {}
    reference_config: str | None = None

    for arm in HISTORICAL_ARMS:
        arm_root = scratch_root / "runs" / "e1_2" / "test" / arm
        manifest = json.loads((arm_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETED" or manifest.get("partition") != "TEST":
            raise ValueError(f"Historical {arm} is not a completed TEST arm")
        if manifest.get("case_count") != len(case_ids):
            raise ValueError(f"Historical {arm} case count does not match the frozen set")
        config_hash = manifest.get("config_sha256")
        if reference_config is None:
            reference_config = config_hash
        elif config_hash != reference_config:
            raise ValueError("Historical E1.2 arms do not share one frozen config identity")

        results: dict[str, ArmOutcome] = {}
        for row in read_jsonl(arm_root / "case_results.jsonl"):
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or case_id in results:
                raise ValueError(f"Missing or duplicate case ID in historical {arm}")
            if row.get("partition") != "TEST":
                raise ValueError(f"Non-TEST result appeared in historical {arm}")
            status = str(row.get("status", "unknown"))
            # Provider failures count incorrect, even if a malformed row says otherwise.
            is_correct = historical_row_is_correct(row)
            results[case_id] = ArmOutcome(
                case_id=case_id,
                subdataset=str(row.get("subdataset", "")),
                is_correct=is_correct,
                answer_input_tokens=row.get("answer_input_tokens"),
                context_characters=row.get("context_characters"),
                component_latency_proxy_ms=row.get("component_latency_proxy_ms"),
                retrieval_calls=int(row.get("retrieval_calls") or 0),
                status=status,
            )
        if set(results) != expected:
            missing = len(expected - set(results))
            extra = len(set(results) - expected)
            raise ValueError(f"Historical {arm} case IDs mismatch: missing={missing}, extra={extra}")
        arms[arm] = results
    return arms


def cost_oracle_v2_action(fixed_correct: dict[str, bool]) -> tuple[str, str, bool]:
    """Pick the cheapest successful action; all-wrong cases stay closed-book."""
    closed = bool(fixed_correct["closed_book"])
    bm25 = bool(fixed_correct["rag_bm25"])
    medcpt = bool(fixed_correct["rag_medcpt"])
    if closed:
        return "closed_book", "CLOSED_SUFFICIENT", False
    if bm25:
        return "rag_bm25", "BM25_RESCUE", True
    if medcpt:
        return "rag_medcpt", "MEDCPT_RESCUE", True
    return "closed_book", "UNRESOLVED", False


def make_cost_oracle_rows(
    cases: list[RouterCase], arms: dict[str, dict[str, ArmOutcome]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        fixed_correct = {
            action: arms[action][case.case_id].is_correct for action in FIXED_ACTIONS
        }
        action, reason, benefit = cost_oracle_v2_action(fixed_correct)
        rows.append(
            {
                "case_id": case.case_id,
                "subdataset": case.subdataset,
                "fixed_correct": fixed_correct,
                "cost_oracle_action": action,
                "cost_oracle_class": reason,
                "retrieval_benefit": benefit,
                "stage2_target": (
                    "rag_bm25"
                    if benefit and fixed_correct["rag_bm25"]
                    else "rag_medcpt" if benefit else None
                ),
            }
        )
    return rows
