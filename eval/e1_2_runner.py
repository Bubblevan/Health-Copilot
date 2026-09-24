"""Deterministic identities, sample selection, and fixed-denominator scoring for E1.2."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PILOT_SEED = "health-copilot-e1.2-generator-pilot-v1"
CLEAN_SUBDATASETS = ("medqa", "medmcqa", "mmlu")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def load_partition(
    cases_path: Path, split_manifest: Mapping[str, Any], partition: str
) -> list[dict[str, Any]]:
    split_name = partition.upper()
    if split_name not in {"DEV", "TEST"}:
        raise ValueError("partition must be DEV or TEST")
    assigned = {
        str(row["case_id"]): row
        for row in split_manifest.get("cases", [])
        if row.get("split") == split_name
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in read_jsonl(cases_path):
        case_id = str(case.get("case_id", ""))
        assignment = assigned.get(case_id)
        if assignment is None:
            continue
        if case_id in seen:
            raise ValueError(f"duplicate normalized case: {case_id}")
        seen.add(case_id)
        if assignment.get("subdataset") not in CLEAN_SUBDATASETS:
            raise ValueError(f"non-clean subdataset assigned to {split_name}: {case_id}")
        selected.append(case)
    expected = sum(int(split_manifest.get("counts", {}).get(name, {}).get(split_name, 0)) for name in CLEAN_SUBDATASETS)
    if len(selected) != expected or len(seen) != len(assigned):
        raise ValueError(f"normalized {split_name} cases do not match the frozen split manifest")
    selected.sort(key=lambda case: str(case["case_id"]))
    return selected


def select_generator_pilot(
    dev_cases: Iterable[Mapping[str, Any]], *, per_subdataset: int = 30, seed: str = PILOT_SEED
) -> list[Mapping[str, Any]]:
    if per_subdataset < 1:
        raise ValueError("per_subdataset must be positive")
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in dev_cases:
        subdataset = str(case.get("metadata", {}).get("subdataset", "")).lower()
        if subdataset in CLEAN_SUBDATASETS:
            buckets[subdataset].append(case)
    if set(buckets) != set(CLEAN_SUBDATASETS):
        raise ValueError("generator pilot requires all clean DEV subdatasets")
    selected: list[Mapping[str, Any]] = []
    for subdataset in CLEAN_SUBDATASETS:
        ordered = sorted(
            buckets[subdataset],
            key=lambda case: hashlib.sha256(
                f"{seed}\0{case.get('case_id', '')}".encode()
            ).hexdigest(),
        )
        if len(ordered) < per_subdataset:
            raise ValueError(f"not enough {subdataset} DEV cases for the frozen pilot")
        selected.extend(ordered[:per_subdataset])
    return sorted(selected, key=lambda case: str(case["case_id"]))


def case_for_answer(case: Mapping[str, Any]) -> dict[str, Any]:
    payload = case.get("payload")
    if not isinstance(payload, Mapping):
        raise TypeError("normalized MIRAGE case is missing payload")
    metadata = case.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("normalized MIRAGE case is missing metadata")
    question = payload.get("question")
    options = payload.get("options")
    if not isinstance(question, str) or not isinstance(options, Mapping):
        raise TypeError("normalized MIRAGE case requires payload.question and payload.options")
    return {
        "case_id": str(case["case_id"]),
        "subdataset": str(metadata["subdataset"]),
        "question": question,
        "options": dict(options),
    }


def gold_label(case: Mapping[str, Any]) -> str:
    gold = case.get("gold")
    if not isinstance(gold, Mapping) or not isinstance(gold.get("answer"), str):
        raise TypeError(f"case {case.get('case_id')} has no normalized gold answer")
    return str(gold["answer"]).strip().upper()


def is_correct(prediction: Any, gold: str) -> bool:
    return isinstance(prediction, str) and prediction.strip().upper() == gold.strip().upper()


def result_identity(*, frozen_config_sha256: str, partition: str, arm: str, model: str) -> str:
    return sha256_text(
        canonical_json(
            {
                "frozen_config_sha256": frozen_config_sha256,
                "partition": partition.upper(),
                "arm": arm,
                "model": model,
            }
        )
    )


def load_retrieval_results(directory: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        subset = path.stem
        for row in read_jsonl(path):
            raw_id = str(row.get("case_id", ""))
            case_id = raw_id if raw_id.startswith(f"{subset}:") else f"{subset}:{raw_id}"
            if case_id in result:
                raise ValueError(f"duplicate retrieval row for {case_id}")
            result[case_id] = row
    return result


def per_case_retrieval_latency(retrieved: Mapping[str, Any], retriever: str) -> float:
    metadata = retrieved.get("retrieval_metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("latency_ms")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
    evidence = retrieved.get("retrieved_evidence")
    if retriever == "medcpt" and isinstance(evidence, list) and evidence:
        value = evidence[0].get("query_latency_ms") if isinstance(evidence[0], Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, float(value))
    return 0.0


def summarize_results(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("subdataset", "unknown"))].append(row)

    def match_rate(group: list[Mapping[str, Any]], field: str) -> float | None:
        values = [row.get(field) for row in group if row.get(field) is not None]
        return sum(value is True for value in values) / len(values) if values else None

    def summarize(group: list[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(group)
        completed = [row for row in group if row.get("status") == "completed"]

        def complete_token_total(field: str) -> int | None:
            values = [row.get(field) for row in group]
            if count == 0 or any(value is None for value in values):
                return None
            return sum(int(value) for value in values)

        def token_measurement_coverage(field: str) -> float | None:
            values = [row.get(field) for row in group]
            return sum(value is not None for value in values) / count if count else None

        invalid_responses = [row for row in completed if row.get("invalid_answer") is not None]
        abstention_responses = [row for row in completed if row.get("abstained") is not None]
        latency = [
            float(row["component_latency_proxy_ms"])
            for row in group
            if row.get("component_latency_proxy_ms") is not None
        ]
        latency_coverage = len(latency) / count if count else None
        return {
            "cases": count,
            "completed": len(completed),
            "provider_failures": sum(
                row.get("status") != "completed"
                and row.get("failure_type") != "missing_retrieval_artifact"
                for row in group
            ),
            "retrieval_artifact_failures": sum(
                row.get("failure_type") == "missing_retrieval_artifact" for row in group
            ),
            "correct": sum(row.get("is_correct") is True for row in group),
            "accuracy_fixed_denominator": (
                sum(row.get("is_correct") is True for row in group) / count if count else None
            ),
            "answer_coverage": (
                sum(row.get("status") == "completed" for row in group) / count if count else None
            ),
            "invalid_answers": sum(row.get("invalid_answer") is True for row in invalid_responses),
            "invalid_answer_rate": (
                sum(row.get("invalid_answer") is True for row in invalid_responses) / len(invalid_responses)
                if invalid_responses
                else None
            ),
            "abstentions": sum(row.get("abstained") is True for row in abstention_responses),
            "abstain_rate": (
                sum(row.get("abstained") is True for row in abstention_responses) / len(abstention_responses)
                if abstention_responses
                else None
            ),
            "retrieval_calls": sum(int(row.get("retrieval_calls", 0)) for row in group),
            "router_calls": sum(int(row.get("router_calls", 0)) for row in group),
            "answer_input_tokens": complete_token_total("answer_input_tokens"),
            "answer_input_token_measurement_coverage": token_measurement_coverage("answer_input_tokens"),
            "answer_output_tokens": complete_token_total("answer_output_tokens"),
            "answer_output_token_measurement_coverage": token_measurement_coverage("answer_output_tokens"),
            "router_input_tokens": complete_token_total("router_input_tokens"),
            "router_input_token_measurement_coverage": token_measurement_coverage("router_input_tokens"),
            "context_characters": sum(int(row.get("context_characters", 0)) for row in group),
            "p50_component_latency_proxy_ms": _percentile(latency, 0.50),
            "p95_component_latency_proxy_ms": _percentile(latency, 0.95),
            "component_latency_proxy_measurement_coverage": latency_coverage,
            "random_context_doc_count_match_rate": match_rate(group, "random_context_doc_count_matched"),
            "random_context_char_budget_match_rate": match_rate(group, "random_context_char_budget_matched"),
        }

    total = summarize(rows)
    return {
        "cases": len(rows),
        "accuracy_fixed_denominator": (
            sum(row.get("is_correct") is True for row in rows) / len(rows) if rows else None
        ),
        "completed": total["completed"],
        "provider_failures": total["provider_failures"],
        "retrieval_artifact_failures": total["retrieval_artifact_failures"],
        "answer_coverage": total["answer_coverage"],
        "invalid_answers": total["invalid_answers"],
        "invalid_answer_rate": total["invalid_answer_rate"],
        "abstentions": total["abstentions"],
        "abstain_rate": total["abstain_rate"],
        "retrieval_calls": total["retrieval_calls"],
        "router_calls": total["router_calls"],
        "answer_input_tokens": total["answer_input_tokens"],
        "answer_input_token_measurement_coverage": total["answer_input_token_measurement_coverage"],
        "answer_output_tokens": total["answer_output_tokens"],
        "answer_output_token_measurement_coverage": total["answer_output_token_measurement_coverage"],
        "router_input_tokens": total["router_input_tokens"],
        "router_input_token_measurement_coverage": total["router_input_token_measurement_coverage"],
        "context_characters": total["context_characters"],
        "p50_component_latency_proxy_ms": total["p50_component_latency_proxy_ms"],
        "p95_component_latency_proxy_ms": total["p95_component_latency_proxy_ms"],
        "component_latency_proxy_measurement_coverage": total[
            "component_latency_proxy_measurement_coverage"
        ],
        "random_context_doc_count_match_rate": total["random_context_doc_count_match_rate"],
        "random_context_char_budget_match_rate": total["random_context_char_budget_match_rate"],
        "by_subdataset": {name: summarize(group) for name, group in sorted(grouped.items())},
        "metrics_semantics": (
            "Provider failures and invalid answers count incorrect in the fixed case denominator; coverage is reported separately. "
            "Invalid and abstain rates use completed responses; token totals are omitted unless measured for every case. "
            "Latency is a component-summed proxy from separately timed route, retrieval, and answer phases, not observed production end-to-end latency."
        ),
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


__all__ = [
    "CLEAN_SUBDATASETS",
    "PILOT_SEED",
    "canonical_json",
    "case_for_answer",
    "file_sha256",
    "gold_label",
    "is_correct",
    "load_partition",
    "load_retrieval_results",
    "per_case_retrieval_latency",
    "read_jsonl",
    "result_identity",
    "select_generator_pilot",
    "sha256_text",
    "summarize_results",
]
