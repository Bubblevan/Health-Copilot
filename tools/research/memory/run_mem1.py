"""Run the pinned LongMemEval-S baselines under the fully local MEM-1 stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mem1_artifacts import (
    append_jsonl,
    canonical_json,
    find_cached_prediction,
    make_cache_identity,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_hash_sidecar,
    write_hash_sidecar,
    write_run_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "docs" / "research" / "memory" / "split_manifest.json"
MODEL_PATH = ROOT / "docs" / "research" / "memory" / "model_protocol.json"
LOCAL_PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem_1_local_only_protocol.json"
DATASET_PATH = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
PINNED_MEMEVAL_SHA = "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4"
SYSTEMS = ("fullcontext", "openclaw", "mem0", "simplemem", "propmem")
CATEGORIES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def _exception_chain(error: BaseException) -> list[dict[str, Any]]:
    """Capture failure classes without copying prompts or response bodies."""
    chain = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(chain) < 8:
        seen.add(id(current))
        status_code = getattr(current, "status_code", None)
        chain.append({
            "type": type(current).__name__,
            "http_status": status_code if isinstance(status_code, int) else None,
        })
        current = current.__cause__ or current.__context__
    return chain


def _digest_ids(question_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{question_id}\n" for question_id in sorted(question_ids)).encode()
    ).hexdigest()


def resolve_question_ids(
    *, split_manifest: dict[str, Any], selected_ids: list[str] | None = None,
) -> list[str]:
    dev_ids = set(split_manifest["dev"]["question_ids"])
    ids = list(selected_ids) if selected_ids is not None else sorted(dev_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection must contain unique, non-empty question IDs")
    outside_dev = set(ids) - dev_ids
    if outside_dev:
        raise ValueError(f"Selection contains IDs outside frozen DEV: {sorted(outside_dev)}")
    if split_manifest.get("test_access", False):
        raise ValueError("Split manifest does not certify test_access=false")
    return ids


def select_records(raw_items: list[dict[str, Any]], question_ids: list[str], normalize):
    wanted = set(question_ids)
    records = [normalize(item) for item in raw_items if item.get("question_id") in wanted]
    observed = [record["qa"][0]["question_id"] for record in records]
    if set(observed) != wanted or len(observed) != len(wanted):
        raise ValueError("Dataset does not contain exactly the requested DEV questions")
    return records


def _source_prompt_hash(memeval_src: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted((memeval_src / "agents_memory").rglob("*.py")):
        digest.update(source.relative_to(memeval_src).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_pinned_patch(memeval_root: Path, patch_path: Path, expected_patch_sha: str) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(memeval_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PINNED_MEMEVAL_SHA:
        raise RuntimeError(f"MemEval checkout must be pinned at {PINNED_MEMEVAL_SHA}; found {head}")
    if sha256_file(patch_path) != expected_patch_sha:
        raise RuntimeError("MemEval patch SHA does not match the locked compatibility manifest")
    subprocess.run(
        ["git", "-C", str(memeval_root), "apply", "--reverse", "--check", str(patch_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_system_registry(memeval_root: Path, selected: list[str]):
    source_root = memeval_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    os.environ["HC_MEM1_SYSTEMS"] = ",".join(selected)
    importlib.invalidate_caches()
    registry = importlib.reload(importlib.import_module("agents_memory.systems"))
    systems = registry.SYSTEMS
    if set(systems) != set(selected):
        raise RuntimeError(f"Patched MemEval did not load requested systems: {sorted(systems)}")
    return systems


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate",), required=True)
    parser.add_argument("--memeval-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--system", choices=SYSTEMS, action="append", required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_PATH)
    parser.add_argument(
        "--patch", type=Path,
        default=ROOT / "tools/research/memory/patches/memeval_qwen_main_v1.patch",
    )
    return parser.parse_args()


def _load_locked_inputs(args: argparse.Namespace):
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    model_protocol = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    local_protocol = json.loads(LOCAL_PROTOCOL_PATH.read_text(encoding="utf-8"))
    if split.get("status") != "FROZEN_ACTIVE_PROTOCOL_LOCKED":
        raise RuntimeError("MEM-1 requires the frozen active MEM-0 split")
    if split.get("dataset_sha256") != sha256_file(args.dataset):
        raise RuntimeError("Local LongMemEval-S bytes do not match the frozen dataset SHA")
    if split.get("dataset_revision") != "98d7416c24c778c2fee6e6f3006e7a073259d48f":
        raise RuntimeError("LongMemEval-S revision differs from the frozen protocol")
    if local_protocol.get("effective_status") != "ACTIVE_LOCAL_ONLY":
        raise RuntimeError("MEM-1 local-only amendment is not active")
    if local_protocol.get("tracks", {}).get("upstream_parity_sanity", {}).get("status") != (
        "CANCELLED_BY_LOCAL_ONLY_AMENDMENT"
    ):
        raise RuntimeError("Historical GPT-4.1 parity track must remain explicitly cancelled")
    selected = args.question_id
    if args.selection_manifest:
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        if selection.get("dataset_sha256") != split.get("dataset_sha256"):
            raise ValueError("Selection manifest dataset SHA differs from frozen DEV")
        ids = selection.get("question_ids")
        if not isinstance(ids, list):
            raise ValueError("Selection manifest must contain a question_ids list")
        selected = ids
    ids = resolve_question_ids(split_manifest=split, selected_ids=selected)
    return split, model_protocol, local_protocol, ids


def _provider_manifest(
    config, selected_systems, system_info, reader_sha, embedding_artifact,
    slot_context, memory_internal_generation,
):
    return {
        "reader_answer_model": {
            "provider": "local_qwen",
            "model": config.reader_model,
            "artifact_sha256": reader_sha,
            "endpoint": config.reader_base_url,
            "endpoint_policy": "loopback only; trust_env=false; outbound request guard enabled",
            "client_policy": {
                "request_timeout_seconds": config.request_timeout_seconds,
                "max_retries": config.max_retries,
                "answer_max_new_tokens": config.reader_answer_max_new_tokens,
            },
            "slot_context_tokens": slot_context,
        },
        "memory_internal_llm": {
            "provider": "local_qwen",
            "model": config.reader_model,
            "artifact_sha256": reader_sha,
            "same_artifact_as_reader": True,
            "endpoint": config.reader_base_url,
            "client_policy": {
                "request_timeout_seconds": config.request_timeout_seconds,
                "max_retries": config.max_retries,
            },
            "generation": memory_internal_generation,
        },
        "memory_system": {
            "systems": selected_systems,
            "display_names": {
                name: system_info[name].get("architecture", name) for name in selected_systems
            },
        },
        "embedding_model": {
            "provider": "local_transformers",
            **embedding_artifact,
        },
        "judge_model": {"provider": "none", "model": None, "used": False},
    }


def _latest_predictions(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for row in rows:
        latest[(row["system"], row["question_id"])] = row
    return latest


def _is_abstention(text: str | None) -> bool:
    if text is None:
        return False
    normalized = " ".join(text.lower().strip(" .!?\t\n").split())
    return normalized in {"none", "unknown", "not mentioned", "not stated", "cannot be determined"}


def _summarize_predictions(rows: list[dict[str, Any]], systems: list[str], ids: list[str]) -> dict:
    latest = _latest_predictions(rows)
    output = {}
    for system in systems:
        selected = [latest[(system, qid)] for qid in ids if (system, qid) in latest]
        quality = [row for row in selected if row.get("quality_status") == "OK" and row.get("f1") is not None]
        category_metrics = {}
        for category in CATEGORIES:
            group = [row for row in quality if row.get("category") == category]
            category_metrics[category] = {
                "n": len(group),
                "token_f1": sum(row["f1"] for row in group) / len(group) if group else None,
            }
        abstention_rows = [
            row for row in quality
            if "abstention" in str(row.get("category", "")).lower()
        ]
        output[system] = {
            "requested": len(ids),
            "completed": len(selected),
            "quality_n": len(quality),
            "infra_failure_n": sum(row.get("quality_status") != "OK" for row in selected),
            "f1_mean": sum(row["f1"] for row in quality) / len(quality) if quality else None,
            "by_category": category_metrics,
            "abstention_accuracy": (
                sum(_is_abstention(row.get("predicted")) for row in abstention_rows)
                / len(abstention_rows) if abstention_rows else None
            ),
            "abstention_n": len(abstention_rows),
        }
    return output


def _summarize_memory_diagnostics(rows: list[dict[str, Any]], systems: list[str], ids: list[str], calls: list[dict[str, Any]]) -> dict:
    latest = _latest_predictions(rows)
    output = {}
    for system in systems:
        selected = [latest[(system, qid)] for qid in ids if (system, qid) in latest]
        system_calls = [row for row in calls if row.get("system") == system]
        reader_rows = [
            row for row in system_calls
            if row.get("role") == "reader_answer" and row.get("success") is True
        ]
        for row in selected:
            groups = row.get("ranked_session_groups")
            answer_sessions = row.get("answer_session_ids") or []
            if row.get("session_provenance_available") is False or not isinstance(groups, list) or not answer_sessions:
                row.setdefault("answer_session_recall_at_5", None)
                row.setdefault("answer_session_recall_at_10", None)
                row.setdefault("answer_session_mrr", None)
                continue
            expected_sessions = set(str(value) for value in answer_sessions)
            recalled = set()
            reciprocal_rank = 0.0
            for rank, group in enumerate(groups, 1):
                matched = expected_sessions.intersection(str(value) for value in group)
                recalled.update(matched)
                if matched and reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
            row["answer_session_recall_at_5"] = len(
                set().union(*(set(str(value) for value in group) for group in groups[:5]))
                & expected_sessions
            ) / len(expected_sessions)
            row["answer_session_recall_at_10"] = len(
                set().union(*(set(str(value) for value in group) for group in groups[:10]))
                & expected_sessions
            ) / len(expected_sessions)
            row["answer_session_mrr"] = reciprocal_rank

        by_category = {}
        for category in CATEGORIES:
            group = [row for row in selected if row.get("category") == category]
            by_category[category] = {
                "n": len(group),
                "answer_session_recall_at_5": _mean_metric(group, "answer_session_recall_at_5"),
                "answer_session_recall_at_10": _mean_metric(group, "answer_session_recall_at_10"),
                "mrr": _mean_metric(group, "answer_session_mrr"),
                "retrieved_context_tokens": _mean_metric(group, "retrieved_context_tokens"),
                "retrieval_latency_ms": _mean_metric(group, "retrieval_latency_ms"),
                "ingestion_latency_ms": _mean_metric(group, "ingestion_latency_ms"),
            }
        output[system] = {
            "answer_session_recall_at_5": _mean_metric(selected, "answer_session_recall_at_5"),
            "answer_session_recall_at_10": _mean_metric(selected, "answer_session_recall_at_10"),
            "mrr": _mean_metric(selected, "answer_session_mrr"),
            "retrieved_context_tokens": _mean_metric(selected, "retrieved_context_tokens"),
            "reader_prompt_tokens": _mean_metric(selected, "reader_prompt_tokens"),
            "retrieval_latency_ms": _mean_metric(selected, "retrieval_latency_ms"),
            "ingestion_latency_ms": _mean_metric(selected, "ingestion_latency_ms"),
            "by_category": by_category,
            "embedding_prompt_tokens": sum(
                row.get("prompt_tokens") or 0 for row in system_calls
                if row.get("role") == "embedding" and row.get("success") is True
            ),
            "reader_calls": len(reader_rows),
        }
    return output


def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summarize_calls(call_rows: list[dict[str, Any]], systems: list[str]) -> dict:
    output = {}
    roles = ("reader_answer", "memory_ingest", "memory_reasoning", "embedding")
    for system in systems:
        rows = [row for row in call_rows if row.get("system") == system]
        role_usage = {}
        for role in roles:
            role_rows = [row for row in rows if row.get("role") == role]
            role_usage[role] = {
                "provider": sorted({row["provider"] for row in role_rows if row.get("provider")}),
                "calls": len(role_rows),
                "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in role_rows),
                "completion_tokens": sum(row.get("completion_tokens") or 0 for row in role_rows),
                "failed_calls": sum(row.get("success") is False for row in role_rows),
                "truncated_calls": sum(row.get("truncated") is True for row in role_rows),
            }
        answer_latencies = [
            float(row["latency_ms"]) for row in rows
            if row.get("role") == "reader_answer" and row.get("latency_ms") is not None
        ]
        output[system] = {
            "by_role": role_usage,
            "answer_latency_p50_ms": _percentile(answer_latencies, 0.50),
            "answer_latency_p95_ms": _percentile(answer_latencies, 0.95),
            "local_reader_wall_ms": sum(
                float(row["latency_ms"]) for row in rows
                if row.get("provider") == "local_qwen" and row.get("latency_ms") is not None
            ),
            "local_embedding_wall_ms": sum(
                float(row["latency_ms"]) for row in rows
                if row.get("provider") == "local_transformers" and row.get("latency_ms") is not None
            ),
        }
    return output


def _write_report(run_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    systems = manifest["roles"]["memory_system"]["systems"]
    lines = [
        "# MEM-1 Local Main Track Run Report",
        "",
        f"- Run: `{manifest['run_id']}`",
        f"- Track: `{manifest['track']}`",
        f"- Split: `{manifest['split']}` ({len(manifest['question_ids'])} frozen DEV questions)",
        f"- Dataset SHA256: `{manifest['dataset']['sha256']}`",
        f"- Prediction SHA256: `{metrics.get('prediction_sha256') or 'NOT_FROZEN_INFRA_FAILURE'}`",
        "- TEST access: `false`",
        "- Hosted API: `NONE`; judge: `NONE`; required API key: `NONE`",
        "",
        "| System | Quality N | Infra failures | Deterministic token F1 | Abstention N | Abstention accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in systems:
        prediction = metrics["systems"][system]
        lines.append(
            f"| {system} | {prediction['quality_n']} | {prediction['infra_failure_n']} | "
            f"{_format_metric(prediction['f1_mean'])} | {prediction['abstention_n']} | "
            f"{_format_metric(prediction['abstention_accuracy'])} |"
        )
    lines.extend(["", "## Category F1", "", "| System | Category | N | Token F1 |", "|---|---|---:|---:|"])
    for system in systems:
        for category in CATEGORIES:
            summary = metrics["systems"][system]["by_category"][category]
            lines.append(f"| {system} | {category} | {summary['n']} | {_format_metric(summary['token_f1'])} |")
    lines.extend([
        "",
        "## Memory Diagnostics",
        "",
        "| System | Recall@5 | Recall@10 | MRR | Retrieved tokens | Reader prompt tokens | Retrieval ms | Ingestion ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for system in systems:
        diagnostic = metrics["memory_diagnostics"][system]
        lines.append(
            f"| {system} | {_format_metric(diagnostic['answer_session_recall_at_5'])} | "
            f"{_format_metric(diagnostic['answer_session_recall_at_10'])} | "
            f"{_format_metric(diagnostic['mrr'])} | {_format_metric(diagnostic['retrieved_context_tokens'])} | "
            f"{_format_metric(diagnostic['reader_prompt_tokens'])} | "
            f"{_format_metric(diagnostic['retrieval_latency_ms'])} | "
            f"{_format_metric(diagnostic['ingestion_latency_ms'])} |"
        )
    lines.extend([
        "",
        "## Retrieval Diagnostics by Category",
        "",
        "| System | Category | Recall@5 | Recall@10 | MRR | Retrieved tokens | Retrieval ms | Ingestion ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for system in systems:
        for category in CATEGORIES:
            diagnostic = metrics["memory_diagnostics"][system]["by_category"][category]
            lines.append(
                f"| {system} | {category} | {_format_metric(diagnostic['answer_session_recall_at_5'])} | "
                f"{_format_metric(diagnostic['answer_session_recall_at_10'])} | "
                f"{_format_metric(diagnostic['mrr'])} | {_format_metric(diagnostic['retrieved_context_tokens'])} | "
                f"{_format_metric(diagnostic['retrieval_latency_ms'])} | "
                f"{_format_metric(diagnostic['ingestion_latency_ms'])} |"
            )
    lines.extend([
        "",
        "Session-retrieval metrics are null when a baseline does not expose auditable source-session provenance; this is not scored as a retrieval miss.",
    ])
    lines.extend([
        "",
        "Reader / answer model, memory-internal LLM, memory system, embedding model and judge are separate manifest roles.",
        "Published upstream results are historical coordinates, not controlled-stack comparisons.",
        "",
    ])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _fullcontext_is_valid(rows: dict[tuple[str, str], dict[str, Any]], ids: list[str], calls: list[dict[str, Any]]) -> bool:
    for question_id in ids:
        prediction = rows.get(("fullcontext", question_id))
        if prediction is None or prediction.get("quality_status") != "OK":
            return False
        answer_calls = [
            row for row in calls
            if row.get("question_id") == question_id
            and row.get("system") == "fullcontext"
            and row.get("role") == "reader_answer"
            and row.get("provider") == "local_qwen"
            and row.get("success") is True
        ]
        if not answer_calls or any(row.get("truncated") is not False for row in answer_calls):
            return False
        if any(row.get("prompt_tokens") is None for row in answer_calls):
            return False
        if any(row.get("max_model_length") != 131072 for row in answer_calls):
            return False
    return True


def _finalize_generation(run_dir: Path, manifest: dict[str, Any], systems: list[str], question_ids: list[str], embedding_model: str | None) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    sidecar_path = run_dir / "predictions.sha256"
    if sidecar_path.exists() and not verify_hash_sidecar(predictions_path, sidecar_path):
        raise RuntimeError("Frozen prediction hash mismatch; refusing to score")
    rows = read_jsonl(predictions_path)
    latest = _latest_predictions(rows)
    expected = {(name, question_id) for name in systems for question_id in question_ids}
    if set(latest) != expected:
        missing = sorted(expected - set(latest))
        unexpected = sorted(set(latest) - expected)
        raise RuntimeError(f"Generation coverage mismatch; missing={missing}, unexpected={unexpected}")
    call_rows = read_jsonl(run_dir / "call_ledger.jsonl")
    every_answer_valid = all(latest[key].get("quality_status") == "OK" for key in expected)
    long_context_valid = "fullcontext" not in systems or _fullcontext_is_valid(latest, question_ids, call_rows)
    frozen = sidecar_path.exists()
    if not frozen and every_answer_valid and long_context_valid:
        digest = write_hash_sidecar(predictions_path, sidecar_path)
    elif frozen:
        digest = sidecar_path.read_text(encoding="ascii").split()[0]
    else:
        digest = None

    metrics = {
        "prediction_sha256": digest,
        "prediction_frozen": digest is not None,
        "fullcontext_validation": "PASS" if long_context_valid else "FAIL_OR_MISSING_TRUNCATION_TELEMETRY",
        "systems": _summarize_predictions(rows, systems, question_ids),
        "memory_diagnostics": _summarize_memory_diagnostics(rows, systems, question_ids, call_rows),
    }
    (run_dir / "deterministic_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "token_efficiency.json").write_text(
        json.dumps(_summarize_calls(call_rows, systems), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    embedding_rows = [row for row in call_rows if row.get("role") == "embedding"]
    (run_dir / "embeddings_usage.json").write_text(
        json.dumps({
            "model": embedding_model,
            "provider": "local_transformers",
            "api_cost": 0,
            "by_system": {
                system: {
                    "calls": sum(row.get("system") == system and row.get("success") is True for row in embedding_rows),
                    "input_tokens": sum(row.get("prompt_tokens") or 0 for row in embedding_rows if row.get("system") == system),
                    "failed_calls": sum(row.get("system") == system and row.get("success") is False for row in embedding_rows),
                    "local_wall_ms": sum(row.get("system") == system and (row.get("latency_ms") or 0) for row in embedding_rows),
                }
                for system in systems
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(run_dir, manifest, metrics)
    return metrics


def generate(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["HC_MEMORY_TRACK"] = "main_local_only"
    os.environ["HC_MEM1_RUN_DIR"] = str(args.run_dir.resolve())
    split, model_protocol, local_protocol, question_ids = _load_locked_inputs(args)
    compatibility = json.loads((ROOT / "docs/research/memory/baseline_compatibility_matrix.json").read_text(encoding="utf-8"))
    patch_sha = compatibility["patch_provenance"]["patch_sha256"]
    verify_pinned_patch(args.memeval_root, args.patch, patch_sha)

    system_names = list(dict.fromkeys(args.system))
    systems = _load_system_registry(args.memeval_root, system_names)
    from agents_memory.benchmarks.longmemeval import _normalize
    from agents_memory.healthcopilot_provider import (
        ProviderConfig,
        configure_call_ledger,
        initialize_local_embedding,
        reader_slot_context,
    )
    from agents_memory.locomo import CATEGORY_NAMES

    config = ProviderConfig.from_env()
    configured_answer_budget = local_protocol["roles"]["reader_answer_model"]["generation"]["answer_max_new_tokens"]
    if config.reader_answer_max_new_tokens != configured_answer_budget:
        raise RuntimeError(
            "Reader answer token budget differs from the frozen MEM-1 local protocol"
        )
    reader_sha = model_protocol["main_track"]["sha256"]
    embedding_runtime = initialize_local_embedding(config)
    embedding_artifact = vars(embedding_runtime.artifact)
    slot_context = reader_slot_context(config)
    if "fullcontext" in system_names and slot_context < local_protocol["roles"]["reader_answer_model"]["long_context"]["max_model_length"]:
        raise RuntimeError(f"FullContext requires a 131072-token loopback slot; server reports {slot_context}")

    raw_items = json.loads(args.dataset.read_text(encoding="utf-8"))
    conversations = select_records(raw_items, question_ids, _normalize)
    source_hash = _source_prompt_hash(args.memeval_root / "src")
    roles = _provider_manifest(
        config, system_names, systems, reader_sha, embedding_artifact, slot_context,
        local_protocol["roles"]["memory_internal_llm"]["generation"],
    )
    runner_code_sha = sha256_bytes(canonical_json({
        "runner": sha256_file(__file__),
        "artifact_helpers": sha256_file(Path(__file__).with_name("mem1_artifacts.py")),
        "local_protocol": sha256_file(LOCAL_PROTOCOL_PATH),
    }))
    cache_code_hash = sha256_bytes(canonical_json({
        "memeval_patch_sha256": patch_sha,
        "healthcopilot_runner_sha256": runner_code_sha,
    }))
    config_hashes = {
        name: sha256_bytes(canonical_json({
            "system": name,
            "architecture": systems[name].get("architecture"),
            "infrastructure": systems[name].get("infrastructure"),
            "reader_model": config.reader_model,
            "reader_generation": local_protocol["roles"]["reader_answer_model"]["generation"],
            "memory_internal_generation": local_protocol["roles"]["memory_internal_llm"]["generation"],
            "embedding_model": config.embedding_model if name != "fullcontext" else None,
            "embedding_artifact_sha256": embedding_artifact["model_sha256"] if name != "fullcontext" else None,
        }))
        for name in system_names
    }
    roles["memory_system"]["system_config_sha256"] = config_hashes
    roles["memory_system"]["prompt_source_sha256"] = source_hash
    roles["reader_answer_model"]["generation"] = local_protocol["roles"]["reader_answer_model"]["generation"]
    roles["embedding_model"]["runtime_settings"] = {
        key: embedding_artifact[key] for key in (
            "repo", "revision", "model_sha256", "weights_sha256", "dimensions",
            "normalized", "query_instruction", "document_instruction", "batch_size",
            "device", "dtype", "max_length", "truncation",
        )
    }
    run_config_hash = sha256_bytes(canonical_json(config_hashes))
    manifest = write_run_manifest(
        args.run_dir / "run_manifest.json",
        run_id=args.run_dir.name,
        track="main_local_only",
        dataset={
            "id": split["dataset_id"],
            "revision": split["dataset_revision"],
            "sha256": split["dataset_sha256"],
            "dev_ids_sha256": split["dev"]["question_ids_sha256_sorted_lf"],
            "test_access": False,
        },
        split="DEV",
        question_ids=question_ids,
        roles=roles,
        system_config_hash=run_config_hash,
        code_patch_sha256=patch_sha,
        runner_code_sha256=runner_code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    predictions_path = args.run_dir / "predictions.jsonl"
    sidecar_path = args.run_dir / "predictions.sha256"
    if sidecar_path.exists():
        _finalize_generation(args.run_dir, manifest, system_names, question_ids, config.embedding_model)
        return
    configure_call_ledger(args.run_dir / "call_ledger.jsonl")
    source_prompt_hash = {"memory_system_and_prompts": source_hash}
    for system_name in system_names:
        system = systems[system_name]
        for conversation in conversations:
            qa = conversation["qa"][0]
            question_id = qa["question_id"]
            embedding_model = config.embedding_model if system_name != "fullcontext" else None
            cache_identity = make_cache_identity(
                system=system_name,
                question_id=question_id,
                dataset_sha256=split["dataset_sha256"],
                system_config_hash=config_hashes[system_name],
                prompt_hashes=source_prompt_hash,
                reader_artifact_sha256=reader_sha,
                embedding_model=embedding_model,
                embedding_artifact_sha256=embedding_artifact["model_sha256"] if embedding_model else None,
                code_patch_hash=cache_code_hash,
            )
            if find_cached_prediction(predictions_path, cache_identity) is not None:
                continue
            from agents_memory.healthcopilot_provider import clear_question_metrics
            clear_question_metrics()
            call_count_before = len(read_jsonl(args.run_dir / "call_ledger.jsonl"))
            system_started = time.perf_counter()
            try:
                with_qa = {**conversation, "qa": [qa]}
                result_rows = system["fn"](
                    with_qa,
                    config.reader_model,
                    False,
                    category_names=CATEGORY_NAMES,
                    judge_fn="longmemeval",
                )
                if len(result_rows) != 1:
                    raise RuntimeError("MEM-1 adapter must return exactly one row per question")
                result = result_rows[0]
            except Exception as error:
                result = {
                    "question_id": question_id,
                    "sample_id": question_id,
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "predicted": None,
                    "category": qa["category"],
                    "category_name": qa["category"],
                    "answer_session_ids": qa.get("answer_session_ids", []),
                    "f1": None,
                    "quality_status": "INFRA_FAILURE",
                    "reader_error_type": type(error).__name__,
                    "reader_error_chain": _exception_chain(error),
                    "reader_messages": None,
                }
            call_rows = read_jsonl(args.run_dir / "call_ledger.jsonl")
            new_calls = call_rows[call_count_before:]
            if system_name == "fullcontext" and result.get("quality_status") == "OK":
                fullcontext_calls = [row for row in new_calls if row.get("role") == "reader_answer"]
                if not fullcontext_calls or any(
                    row.get("truncated") is not False or row.get("prompt_tokens") is None
                    for row in fullcontext_calls
                ):
                    result = {
                        **result,
                        "predicted": None,
                        "f1": None,
                        "quality_status": "INFRA_FAILURE",
                        "reader_error_type": "FullContextLengthNotVerified",
                    }
            answer_calls = [
                row for row in new_calls if row.get("role") == "reader_answer"
                and row.get("success") is True
            ]
            if answer_calls:
                result["reader_prompt_tokens"] = sum(row.get("prompt_tokens") or 0 for row in answer_calls)
            if system_name == "fullcontext" and result.get("reader_prompt_tokens") is not None:
                result["retrieved_context_tokens"] = result["reader_prompt_tokens"]
            result.setdefault("answer_session_ids", qa.get("answer_session_ids", []))
            row = {
                **result,
                "system": system_name,
                "question_id": question_id,
                "cache_identity": cache_identity,
                "system_wall_time_ms": round((time.perf_counter() - system_started) * 1000, 3),
            }
            append_jsonl(predictions_path, row)
            if row["quality_status"] != "OK":
                failures = read_jsonl(args.run_dir / "failures.jsonl")
                failure_type = (
                    "INFRA_EMBEDDING" if any(call.get("role") == "embedding" and call.get("success") is False for call in new_calls)
                    else row.get("reader_error_type", "INFRA_LIBRARY")
                )
                append_jsonl(args.run_dir / "failures.jsonl", {
                    "question_id": question_id,
                    "system": system_name,
                    "failure_type": failure_type,
                    "exception_chain": row.get("reader_error_chain", []),
                    "quality_status": "INFRA_FAILURE",
                    "prior_failure_count": len(failures),
                })

    metrics = _finalize_generation(
        args.run_dir,
        json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8")),
        system_names,
        question_ids,
        config.embedding_model,
    )
    print(f"Prediction artifact frozen: {metrics['prediction_frozen']}")


def main() -> None:
    args = _parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if len(args.system) != len(set(args.system)):
        raise SystemExit("Duplicate --system arguments are not allowed")
    generate(args)


if __name__ == "__main__":
    main()
