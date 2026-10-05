"""Resumable system-level evaluation through the production Harness contract."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Iterable
from hashlib import sha256
from pathlib import Path
from statistics import mean
from time import monotonic
from typing import Any

from ..harness.contracts import HarnessRequest
from ..harness.profiles import SystemProfile
from ..harness.runtime import HealthCopilotHarness
from ..harness.verification import ANSWER_PARSER_REVISION, is_explicit_abstention
from .contracts import CaseScore, DatasetAdapter


def model_config_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


async def run_dataset(
    *,
    harness: HealthCopilotHarness,
    adapter: DatasetAdapter,
    profile: SystemProfile,
    model_hash: str,
    output_dir: Path,
    resume: bool = False,
    concurrency: int = 1,
    trace_reader: Callable[[str], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Execute independent cases concurrently; fsync each completed checkpoint row."""
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if len(model_hash) != 64 or any(ch not in "0123456789abcdef" for ch in model_hash.lower()):
        raise ValueError("model_hash must be a SHA-256 hex digest")
    if not adapter.source_revision or adapter.source_revision == "UNPINNED":
        raise ValueError("dataset source revision must be pinned before evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "cases.jsonl"
    summary_path = output_dir / "summary.json"
    cases = tuple(adapter.cases())
    known_ids = {case.case_id for case in cases}
    if len(known_ids) != len(cases):
        raise ValueError("dataset adapter returned duplicate case IDs")

    completed: dict[str, dict[str, Any]] = {}
    if resume and checkpoint.is_file():
        with checkpoint.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("run_identity") != _run_identity(
                    adapter.dataset_id, profile.profile_id, model_hash
                ):
                    raise ValueError(f"checkpoint identity mismatch on line {line_no}")
                if record.get("source_revision") != adapter.source_revision:
                    raise ValueError(f"checkpoint dataset revision mismatch on line {line_no}")
                if record.get("dataset_snapshot_sha256") != getattr(adapter, "snapshot_sha256", None):
                    raise ValueError(f"checkpoint dataset hash mismatch on line {line_no}")
                if record.get("dataset_selection_sha256") != getattr(adapter, "subset_sha256", None):
                    raise ValueError(f"checkpoint subset identity mismatch on line {line_no}")
                case_id = str(record["case_id"])
                if case_id not in known_ids:
                    raise ValueError(f"checkpoint contains unknown case ID {case_id}")
                if case_id in completed:
                    raise ValueError(f"duplicate checkpoint record for case ID {case_id}")
                completed[case_id] = record
    elif checkpoint.exists():
        raise FileExistsError("case checkpoint exists; pass resume=True or choose a new output directory")

    identity = _run_identity(adapter.dataset_id, profile.profile_id, model_hash)
    pending = [case for case in cases if case.case_id not in completed]
    semaphore = asyncio.Semaphore(concurrency)
    checkpoint_lock = asyncio.Lock()
    started = monotonic()

    async def execute_case(case) -> None:
        async with semaphore:
            request = HarnessRequest(
                request_id=f"eval-{adapter.dataset_id}-{case.case_id}",
                query=case.query,
                answer_schema=case.answer_schema,
                benchmark_case_id=case.case_id,
            )
            response = await harness.execute(profile, request)
            score = adapter.score(case, response)
            trace = trace_reader(response.trace_id) if trace_reader else None
            record = {
                "run_identity": identity,
                "dataset_id": adapter.dataset_id,
                "source_revision": adapter.source_revision,
                "dataset_snapshot_sha256": getattr(adapter, "snapshot_sha256", None),
                "dataset_selection_sha256": getattr(adapter, "subset_sha256", None),
                "case_id": case.case_id,
                "profile_id": profile.profile_id,
                "model_hash": model_hash,
                "response": response.to_dict(),
                "score": _score_dict(score),
                "trace_summary": _trace_summary(trace),
                "case_concurrency": concurrency,
            }
            async with checkpoint_lock:
                with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed[case.case_id] = record

    outcomes = await asyncio.gather(
        *(execute_case(case) for case in pending), return_exceptions=True,
    )
    errors = [item for item in outcomes if isinstance(item, BaseException)]
    elapsed_seconds = max(monotonic() - started, 1e-9)
    if errors:
        raise RuntimeError(
            f"{len(errors)} case(s) failed; completed rows remain checkpointed: "
            f"{type(errors[0]).__name__}: {errors[0]}"
        ) from errors[0]

    ordered = [completed[case.case_id] for case in cases]
    executed = [completed[case.case_id] for case in pending]
    summary = summarize_records(ordered)
    summary.update({
        "dataset_id": adapter.dataset_id,
        "source_revision": adapter.source_revision,
        "dataset_snapshot_sha256": getattr(adapter, "snapshot_sha256", None),
        "dataset_selection_sha256": getattr(adapter, "subset_sha256", None),
        "profile_id": profile.profile_id,
        "model_hash": model_hash,
        "case_count": len(cases),
        "completed_cases_this_invocation": len(pending),
        "checkpoint": checkpoint.name,
        "case_concurrency": concurrency,
        "run_wall_seconds": round(elapsed_seconds, 3),
        "cases_per_second": round(len(pending) / elapsed_seconds, 5),
        "provider_requests_per_second": round(
            sum(int(item["response"].get("provider_calls", 0)) for item in executed) / elapsed_seconds,
            5,
        ),
        "tokens_per_second": None if any(
            item["response"].get("input_tokens") is None
            or item["response"].get("output_tokens") is None for item in executed
        ) else round(sum(
            int(item["response"]["input_tokens"]) + int(item["response"]["output_tokens"])
            for item in executed
        ) / elapsed_seconds, 3),
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        return {
            "correct": 0, "parse_success": 0, "accuracy": None,
            "parse_rate": None, "provider_calls_per_case": None,
            "input_tokens_per_case": None, "output_tokens_per_case": None,
            "total_tokens_per_case": None,
            "mean_latency_ms": None, "p50_latency_ms": None, "p95_latency_ms": None,
            "retrieval_calls": None, "retrieved_documents": None,
            "mdt_activation": None, "mdt_complexity_distribution": {},
            "macro_accuracy_by_category": {},
            "safety_route_abstentions": 0,
            "safety_route_abstention_rate": None,
            "reasoning_failures": 0,
            "model_abstentions": 0,
            "answer_format_failures": 0,
            "unparsed_non_safety": 0,
            "unparsed_non_safety_rate": None,
            "scoring_revision": ANSWER_PARSER_REVISION,
        }
    score_rows = [item["score"] for item in rows]
    correct = sum(bool(item["correct"]) for item in score_rows)
    parse_success = sum(bool(item["parse_success"]) for item in score_rows)
    response_rows = [item["response"] for item in rows]

    def is_safety_abstention(response: dict[str, Any]) -> bool:
        return any(
            str(flag).startswith(("urgent_marker:", "prescription_marker:"))
            for flag in response.get("safety_flags", ())
        )

    def is_reasoning_failure(response: dict[str, Any]) -> bool:
        return any(
            str(flag).startswith("reasoning_failure:")
            for flag in response.get("safety_flags", ())
        )

    safety_route_abstentions = sum(is_safety_abstention(item) for item in response_rows)
    reasoning_failures = sum(is_reasoning_failure(item) for item in response_rows)
    model_abstentions = sum(
        not is_safety_abstention(response)
        and not is_reasoning_failure(response)
        and is_explicit_abstention(str(response.get("answer_text", "")))
        for response in response_rows
    )
    answer_format_failures = sum(
        not bool(score["parse_success"])
        and not is_safety_abstention(response)
        and not is_reasoning_failure(response)
        and not is_explicit_abstention(str(response.get("answer_text", "")))
        for score, response in zip(score_rows, response_rows, strict=True)
    )
    latency = sorted(float(item["latency_ms"]) for item in response_rows)
    categories: dict[str, list[bool]] = {}
    trace_summaries = [item.get("trace_summary") for item in rows]
    complexity_counts: dict[str, int] = {}
    for trace in trace_summaries:
        complexity = trace.get("mdt_complexity") if trace else None
        if complexity:
            complexity_counts[str(complexity)] = complexity_counts.get(str(complexity), 0) + 1
    for score in score_rows:
        if score.get("category") is not None:
            categories.setdefault(str(score["category"]), []).append(bool(score["correct"]))
    input_values = [item.get("input_tokens") for item in response_rows]
    output_values = [item.get("output_tokens") for item in response_rows]

    def percentile(p: float) -> float:
        return round(latency[min(len(latency) - 1, int((len(latency) - 1) * p))], 3)

    def average_or_unknown(field: str) -> float | None:
        values = [item[field] for item in trace_summaries if item is not None and item.get(field) is not None]
        return round(mean(values), 4) if values else None

    return {
        "correct": correct,
        "parse_success": parse_success,
        "accuracy": correct / len(rows),
        "parse_rate": parse_success / len(rows),
        "safety_route_abstentions": safety_route_abstentions,
        "safety_route_abstention_rate": safety_route_abstentions / len(rows),
        "reasoning_failures": reasoning_failures,
        "model_abstentions": model_abstentions,
        "answer_format_failures": answer_format_failures,
        "unparsed_non_safety": answer_format_failures,
        "unparsed_non_safety_rate": answer_format_failures / len(rows),
        "scoring_revision": ANSWER_PARSER_REVISION,
        "provider_calls_per_case": round(mean(float(item["provider_calls"]) for item in response_rows), 4),
        "input_tokens_per_case": None if any(item is None for item in input_values)
        else round(mean(float(item) for item in input_values), 4),
        "output_tokens_per_case": None if any(item is None for item in output_values)
        else round(mean(float(item) for item in output_values), 4),
        "total_tokens_per_case": None if any(
            input_value is None or output_value is None
            for input_value, output_value in zip(input_values, output_values, strict=True)
        ) else round(mean(
            float(input_value) + float(output_value)
            for input_value, output_value in zip(input_values, output_values, strict=True)
        ), 4),
        "mean_latency_ms": round(mean(latency), 3),
        "p50_latency_ms": percentile(0.50),
        "p95_latency_ms": percentile(0.95),
        "retrieval_calls": average_or_unknown("retrieval_calls"),
        "retrieved_documents": average_or_unknown("retrieved_documents"),
        "mdt_activation": average_or_unknown("mdt_activated"),
        "mdt_complexity_distribution": dict(sorted(complexity_counts.items())),
        "macro_accuracy_by_category": {
            category: round(mean(values), 6) for category, values in sorted(categories.items())
        },
    }


def _run_identity(dataset_id: str, profile_id: str, model_hash: str) -> str:
    payload = json.dumps(
        [dataset_id, profile_id, model_hash], ensure_ascii=False, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _score_dict(score: CaseScore) -> dict[str, Any]:
    return {
        "case_id": score.case_id,
        "correct": score.correct,
        "parse_success": score.parse_success,
        "metric": score.metric,
        "category": score.category,
    }


def _trace_summary(trace: dict[str, Any] | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    events = trace.get("events", [])
    complexity = next((
        item.get("fields", {}).get("complexity")
        for item in events
        if item.get("kind") == "reasoning_detail"
        and item.get("fields", {}).get("event") == "reasoning_adaptive_mdt"
    ), None)
    return {
        "retrieval_calls": sum(item.get("kind") == "retrieval_completed" for item in events),
        "retrieved_documents": sum(
            int(item.get("fields", {}).get("evidence_count", 0))
            for item in events if item.get("kind") == "retrieval_completed"
        ),
        "mdt_activated": complexity in {"intermediate", "advanced"},
        "mdt_complexity": complexity,
    }
