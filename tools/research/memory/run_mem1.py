"""Run the pinned LongMemEval-S baselines under the fully local MEM-1 stack."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from context_bundle import verify_context_bundle
from final_reader_contract import (
    build_reader_messages as build_final_reader_messages,
)
from final_reader_contract import (
    contract_binding,
    load_final_reader_contract,
    verify_contract_binding,
)
from mem1_artifacts import (
    append_jsonl,
    canonical_json,
    find_cached_prediction,
    make_cache_identity,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verified_identity_hash,
    verify_hash_sidecar,
    write_hash_sidecar,
    write_run_manifest,
)

ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "docs" / "research" / "memory" / "split_manifest.json"
MODEL_PATH = ROOT / "docs" / "research" / "memory" / "model_protocol.json"
LOCAL_PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem_1_local_only_protocol.json"
DATASET_PATH = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
D1_SELECTION_PATH = ROOT / "docs" / "research" / "memory" / "main_smoke_10_manifest.json"
D1_SELECTION_SHA256 = "5a38ff79d79be6a9db531227d63d6dc22dc619d11d2c701b2b4cc0295f04c911"
D1_DATASET_ID = "longmemeval_s"
D1_DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
D1_DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
D1_DEV_IDS_SHA256 = "c6e0b423f720bcb06e1c571a8f6b0d018e0c1707d21fe17108349962d30f739f"
D1_QUESTION_IDS = (
    "1cea1afa",
    "1c549ce4",
    "778164c6",
    "fca70973",
    "a82c026e",
    "gpt4_e061b84g",
    "gpt4_f420262c",
    "8550ddae",
    "06878be2",
    "c4ea545c",
)
D1_QUESTION_IDS_SHA256 = "a5ce841c2a60f9248bd1fc4fdc4f79a075c24a3a26921be2f96b042ec56e8b0d"
D1_READER_TEMPLATE_SHA256 = "0ff70b000bd4b43db85ae587731fea35b691febc815cfbb57c2acb1c8b295b2b"
PINNED_MEMEVAL_SHA = "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4"
PINNED_SIMPLEMEM_SHA = "7da777f56a15db81bb261d296c89cad5915e8d67"
PINNED_SIMPLEMEM_TREE_SHA = "e84e01b775296db1fcde799fac1a5ec2ffafe4a00de4a9f2000eb885aa5c1191"
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


def _source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts):
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(source).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_official_simplemem_source(root: Path) -> dict[str, str]:
    audit_path = ROOT / "docs/research/memory/simplemem_official_v010_fidelity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    planning_calls = audit.get("call_counts", {}).get("planning_true", {})
    if audit.get("decision") != "PASS" or not all(
        planning_calls.get(name, 0) > 0
        for name in ("semantic", "keyword", "structured", "merge")
    ):
        raise RuntimeError("Official SimpleMem synthetic fidelity gate is not recorded as PASS")
    remote = subprocess.check_output(
        ["git", "-C", str(root), "remote", "get-url", "origin"],
        text=True,
    ).strip().rstrip("/").removesuffix(".git")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    tree_sha = _source_tree_sha256(root)
    if remote != "https://github.com/aiming-lab/SimpleMem":
        raise RuntimeError("Official SimpleMem remote does not match the frozen upstream")
    if head != PINNED_SIMPLEMEM_SHA or audit.get("official_release", {}).get("commit") != head:
        raise RuntimeError("Official SimpleMem checkout differs from the pinned v0.1.0 commit")
    if status:
        raise RuntimeError("Official SimpleMem source checkout must remain clean")
    if tree_sha != PINNED_SIMPLEMEM_TREE_SHA or audit.get("official_release", {}).get("source_tree_sha256") != tree_sha:
        raise RuntimeError("Official SimpleMem source tree differs from its audited hash")
    return {
        "repository": "https://github.com/aiming-lab/SimpleMem.git",
        "tag": "v0.1.0",
        "commit": head,
        "source_tree_sha256": tree_sha,
        "fidelity_artifact_sha256": sha256_file(audit_path),
    }


def _baseline_warning_rows(
    system: str,
    question_id: str,
    captured_output: str,
    provider_rows: list[dict[str, Any]],
    *,
    adapter_completed: bool,
) -> list[dict[str, Any]]:
    """Keep warning classifications while deliberately discarding captured log text."""
    grouped: dict[tuple[str, str, str, bool | None, str, str], int] = {}

    def add(
        phase: str,
        warning_type: str,
        classification: str,
        recovered: bool | None,
        operation: str,
        evidence_source: str,
        count: int = 1,
    ) -> None:
        key = (phase, warning_type, classification, recovered, operation, evidence_source)
        grouped[key] = grouped.get(key, 0) + count

    for line in captured_output.splitlines():
        lowered = line.strip().lower()
        if not lowered:
            continue
        if system == "mem0" and (
            "invalid json response" in lowered
            or "invalid json" in lowered
            or "json decode" in lowered
        ):
            add("memory_ingest", "MEM0_INVALID_JSON_RESPONSE", "BASELINE_INTERNAL_WARNING", adapter_completed, "structured_output", "baseline_log")
        elif system == "mem0" and (
            "delete" in lowered and ("error" in lowered or "fail" in lowered)
            or re.search(r"error:\s*['\"]?\d+", lowered)
        ):
            add("memory_ingest", "MEM0_ACTION_HANDLER_WARNING", "BASELINE_INTERNAL_WARNING", adapter_completed, "DELETE" if "delete" in lowered or "error: '14'" in lowered else "unknown", "baseline_log")
        elif "fts index creation skipped" in lowered:
            add("memory_ingest", "SIMPLEMEM_FTS_INDEX_WARNING", "BASELINE_INTERNAL_WARNING", adapter_completed, "fts_index", "baseline_log")
        elif system == "simplemem" and ("failed to parse" in lowered or "parser recovery" in lowered):
            add("memory_ingest", "SIMPLEMEM_PARSE_RECOVERY", "BASELINE_INTERNAL_WARNING", adapter_completed, "structured_output", "baseline_log")
        elif "failed to parse" in lowered or "parser recovery" in lowered:
            add("baseline_context", "BASELINE_PARSER_RECOVERY", "BASELINE_INTERNAL_WARNING", adapter_completed, "parser", "baseline_log")
        elif system == "simplemem" and ("retrying" in lowered or "retry attempt" in lowered or "falling back to sequential" in lowered):
            add("baseline_context", "SIMPLEMEM_INTERNAL_RECOVERY", "BASELINE_INTERNAL_WARNING", adapter_completed, "upstream_recovery", "baseline_log")
        elif "retrying" in lowered or "retry attempt" in lowered:
            add("baseline_context", "MODEL_INTERNAL_RETRY", "BASELINE_INTERNAL_WARNING", adapter_completed, "model_call", "baseline_log")
        elif system == "simplemem" and ("invalid json" in lowered or "json decode" in lowered):
            add("baseline_context", "SIMPLEMEM_JSON_RECOVERY", "BASELINE_INTERNAL_WARNING", adapter_completed, "structured_output", "baseline_log")

    for row in provider_rows:
        phase = str(row.get("role") or "provider_call")
        operation = phase
        if row.get("success") is False:
            add(phase, "PROVIDER_FAILURE", "INFRA_FAILURE", False, operation, "provider_call_ledger")
        retry_count = int(row.get("retry_count") or 0)
        if retry_count > 0:
            add(phase, "PROVIDER_RETRY", "BASELINE_INTERNAL_WARNING", row.get("success") is True, operation, "provider_call_ledger", retry_count)
        if phase == "embedding" and row.get("truncated") is True:
            add(phase, "EMBEDDING_TRUNCATION", "BASELINE_INTERNAL_WARNING", False, operation, "provider_call_ledger")
        if phase in {"memory_ingest", "memory_reasoning", "reader_answer"} and row.get("success") is True and (
            row.get("prompt_tokens") is None or row.get("completion_tokens") is None
        ):
            add(phase, "MISSING_USAGE_TELEMETRY", "BASELINE_INTERNAL_WARNING", False, operation, "provider_call_ledger")

    if not adapter_completed and not any(row.get("success") is False for row in provider_rows):
        add("baseline_context", "ADAPTER_FAILURE", "INFRA_FAILURE", False, "context_adapter", "runner_status")

    return [
        {
            "system": system,
            "question_id": question_id,
            "phase": phase,
            "warning_type": warning_type,
            "classification": classification,
            "count": count,
            "recovered": recovered,
            "affected_operation": operation,
            "evidence_source": evidence_source,
        }
        for (phase, warning_type, classification, recovered, operation, evidence_source), count in sorted(
            grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    ]


def _summarize_baseline_warnings(
    rows: list[dict[str, Any]], systems: list[str]
) -> dict[str, Any]:
    return {
        system: {
            "warning_rows": sum(row.get("system") == system for row in rows),
            "baseline_internal_warning_count": sum(
                int(row.get("count") or 1)
                for row in rows
                if row.get("system") == system
                and row.get("classification") == "BASELINE_INTERNAL_WARNING"
            ),
            "infra_failure_count": sum(
                int(row.get("count") or 1)
                for row in rows
                if row.get("system") == system
                and row.get("classification") == "INFRA_FAILURE"
            ),
            "by_type": {
                warning_type: sum(
                    int(row.get("count") or 1)
                    for row in rows
                    if row.get("system") == system
                    and row.get("warning_type") == warning_type
                )
                for warning_type in sorted({
                    row.get("warning_type")
                    for row in rows
                    if row.get("system") == system
                })
            },
        }
        for system in systems
    }


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


def validate_mem1d1_selection(
    selection: dict[str, Any], split: dict[str, Any], selection_path: Path
) -> None:
    if selection_path.resolve() != D1_SELECTION_PATH.resolve():
        raise ValueError("MEM-1D1 requires the canonical frozen 10-case selection manifest")
    if sha256_file(selection_path) != D1_SELECTION_SHA256:
        raise ValueError("MEM-1D1 selection manifest bytes differ from the frozen artifact")
    expected = {
        "manifest_version": "memeval-main-smoke-10-v1",
        "status": "FROZEN_BEFORE_ANY_10_CASE_RESULTS",
        "dataset_id": D1_DATASET_ID,
        "dataset_revision": D1_DATASET_REVISION,
        "dataset_sha256": D1_DATASET_SHA256,
        "dev_manifest": "split_manifest.json",
        "dev_ids_sha256_sorted_lf": D1_DEV_IDS_SHA256,
        "question_count": 10,
        "question_ids": list(D1_QUESTION_IDS),
        "question_ids_sha256_sorted_lf": D1_QUESTION_IDS_SHA256,
        "test_access": False,
    }
    if any(selection.get(key) != value for key, value in expected.items()):
        raise ValueError("MEM-1D1 selection manifest does not match the frozen 10-case contract")
    if (
        split.get("dataset_id") != D1_DATASET_ID
        or split.get("dataset_revision") != D1_DATASET_REVISION
        or split.get("dataset_sha256") != D1_DATASET_SHA256
        or split.get("dev", {}).get("question_ids_sha256_sorted_lf") != D1_DEV_IDS_SHA256
    ):
        raise ValueError("MEM-1D1 frozen split manifest differs from its selection contract")
    dev_ids = set(split.get("dev", {}).get("question_ids", []))
    test_ids = set(split.get("test", {}).get("question_ids", []))
    if not set(D1_QUESTION_IDS).issubset(dev_ids) or set(D1_QUESTION_IDS) & test_ids:
        raise ValueError("MEM-1D1 selection contains non-DEV or TEST IDs")
    if _digest_ids(list(D1_QUESTION_IDS)) != D1_QUESTION_IDS_SHA256:
        raise RuntimeError("Internal MEM-1D1 question digest constant is invalid")


def _verify_frozen_evidence(run_dir: Path, *, require_ledgers: bool) -> bool:
    pairs = [
        ("predictions.jsonl", "predictions.sha256"),
        ("context_bundles.jsonl", "context_bundles.sha256"),
    ]
    if require_ledgers:
        pairs.extend([
            ("call_ledger.jsonl", "call_ledger.sha256"),
            ("baseline_warnings.jsonl", "baseline_warnings.sha256"),
        ])
    sidecar_presence = [(run_dir / sidecar).exists() for _, sidecar in pairs]
    if not any(sidecar_presence):
        return False
    if not all(sidecar_presence):
        raise RuntimeError("Frozen MEM-1 evidence is incomplete; refusing to resume or score")
    for artifact_name, sidecar_name in pairs:
        artifact = run_dir / artifact_name
        sidecar = run_dir / sidecar_name
        if not artifact.is_file() or not verify_hash_sidecar(artifact, sidecar):
            raise RuntimeError(f"Frozen evidence hash mismatch: {artifact_name}")
    return True


def select_records(raw_items: list[dict[str, Any]], question_ids: list[str], normalize):
    wanted = set(question_ids)
    records = []
    for item in raw_items:
        if item.get("question_id") not in wanted:
            continue
        question_date = item.get("question_date")
        if not isinstance(question_date, str) or not question_date:
            raise ValueError(
                f"LongMemEval record {item.get('question_id')} lacks an official question_date"
            )
        record = normalize(item)
        if not record.get("qa") or not isinstance(record["qa"][0], dict):
            raise ValueError("LongMemEval normalizer did not return a QA object")
        record["qa"][0]["question_date"] = question_date
        records.append(record)
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
        [
            "git", "-C", str(memeval_root.resolve()), "apply", "--reverse", "--check",
            str(patch_path.resolve()),
        ],
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
    parser.add_argument(
        "--answer-track",
        choices=("native", "context_controlled"),
        default="context_controlled",
        help="Use the shared reader head by default; select native only for an audit.",
    )
    parser.add_argument("--memeval-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--mem1d1-frozen-10", action="store_true")
    parser.add_argument("--execution-question-id", action="append")
    parser.add_argument("--execution-system", choices=SYSTEMS, action="append")
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
    if args.mem1d1_frozen_10:
        if args.answer_track != "context_controlled":
            raise ValueError("MEM-1D1 permits only the context_controlled answer track")
        if args.question_id is not None:
            raise ValueError("MEM-1D1 refuses manually supplied --question-id values")
        if args.selection_manifest is None:
            raise ValueError("MEM-1D1 requires main_smoke_10_manifest.json")
        if list(args.system) != list(SYSTEMS):
            raise ValueError("MEM-1D1 requires the exact frozen five-system matrix in order")
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        validate_mem1d1_selection(selection, split, args.selection_manifest)
        if selection.get("dataset_sha256") != split.get("dataset_sha256"):
            raise ValueError("MEM-1D1 selection manifest dataset SHA differs from frozen DEV")
        requested_questions = args.execution_question_id
        execution_questions = list(D1_QUESTION_IDS) if requested_questions is None else requested_questions
        if (
            not execution_questions
            or len(execution_questions) != len(set(execution_questions))
            or not set(execution_questions).issubset(D1_QUESTION_IDS)
        ):
            raise ValueError("MEM-1D1 execution question chunk must be a unique subset of frozen IDs")
        requested_systems = args.execution_system
        execution_systems = list(SYSTEMS) if requested_systems is None else requested_systems
        if (
            not execution_systems
            or len(execution_systems) != len(set(execution_systems))
            or not set(execution_systems).issubset(SYSTEMS)
        ):
            raise ValueError("MEM-1D1 execution system chunk must be a unique subset of the frozen matrix")
        args.d1_selection_sha256 = D1_SELECTION_SHA256
        args.execution_question_ids = execution_questions
        args.execution_systems = execution_systems
        return split, model_protocol, local_protocol, list(D1_QUESTION_IDS)

    if args.execution_question_id is not None or args.execution_system is not None:
        raise ValueError("Execution chunks are available only with --mem1d1-frozen-10")
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
    args.d1_selection_sha256 = None
    args.execution_question_ids = ids
    args.execution_systems = list(args.system)
    return split, model_protocol, local_protocol, ids


def _provider_manifest(
    config, selected_systems, system_info, reader_sha, embedding_artifact,
    slot_context, memory_internal_generation, reader_runtime,
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
            "runtime": reader_runtime,
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


_REFUSAL_PHRASES = {
    "none",
    "unknown",
    "not mentioned",
    "not stated",
    "cannot be determined",
    "cannot determine",
    "not specified",
    "not available",
    "no information",
    "no info",
    "no evidence",
    "not found",
    "not provided",
    "not addressed",
    "no relevant information",
    "no memory",
    "no record",
    "no data",
    "i dont know",
    "i do not know",
    "i dont remember",
    "i do not remember",
}


def _is_deterministic_refusal(text: str | None) -> bool:
    if text is None:
        return False
    import re

    normalized = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    return normalized in _REFUSAL_PHRASES


def _is_abstention(text: str | None) -> bool:
    return _is_deterministic_refusal(text)


def _is_abstention_question(question_id: str) -> bool:
    return question_id.endswith("_abs")


def _answer_metrics(predicted: str, expected: str) -> dict[str, float]:
    import re

    pred_tokens = set(re.findall(r"\w+", predicted.lower()))
    gold_tokens = set(re.findall(r"\w+", expected.lower()))
    if not gold_tokens:
        correct_refusal = _is_deterministic_refusal(predicted)
        score = 1.0 if correct_refusal else 0.0
        return {
            "token_precision": score,
            "token_recall": score,
            "f1": score,
            "normalized_exact_match": score,
        }
    if not pred_tokens:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0,
            "f1": 0.0,
            "normalized_exact_match": 0.0,
        }
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    normalized_prediction = " ".join(re.findall(r"\w+", predicted.lower()))
    normalized_expected = " ".join(re.findall(r"\w+", expected.lower()))
    return {
        "token_precision": precision,
        "token_recall": recall,
        "f1": f1,
        "normalized_exact_match": float(normalized_prediction == normalized_expected),
    }


def _sum_captured(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return int(sum(values)) if values else None


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
                "token_precision": _mean_metric(group, "token_precision"),
                "token_recall": _mean_metric(group, "token_recall"),
                "normalized_exact_match": _mean_metric(group, "normalized_exact_match"),
            }
        abstention_rows = [
            row for row in quality
            if _is_abstention_question(str(row.get("question_id", "")))
        ]
        output[system] = {
            "requested": len(ids),
            "completed": len(selected),
            "quality_n": len(quality),
            "infra_failure_n": sum(row.get("quality_status") != "OK" for row in selected),
            "f1_mean": sum(row["f1"] for row in quality) / len(quality) if quality else None,
            "token_precision_mean": _mean_metric(quality, "token_precision"),
            "token_recall_mean": _mean_metric(quality, "token_recall"),
            "normalized_exact_match_mean": _mean_metric(quality, "normalized_exact_match"),
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
            expected_sessions = {str(value) for value in answer_sessions}
            recalled = set()
            reciprocal_rank = 0.0
            for rank, group in enumerate(groups, 1):
                matched = expected_sessions.intersection(str(value) for value in group)
                recalled.update(matched)
                if matched and reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / rank
            row["answer_session_recall_at_5"] = len(
                set().union(*({str(value) for value in group} for group in groups[:5]))
                & expected_sessions
            ) / len(expected_sessions)
            row["answer_session_recall_at_10"] = len(
                set().union(*({str(value) for value in group} for group in groups[:10]))
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
                "context_reader_tokens": _mean_metric(group, "context_reader_tokens"),
                "context_embedding_tokens": _mean_metric(group, "context_embedding_tokens"),
                "retrieval_latency_ms": _mean_metric(group, "retrieval_latency_ms"),
                "ingestion_latency_ms": _mean_metric(group, "ingestion_latency_ms"),
            }
        output[system] = {
            "answer_session_recall_at_5": _mean_metric(selected, "answer_session_recall_at_5"),
            "answer_session_recall_at_10": _mean_metric(selected, "answer_session_recall_at_10"),
            "mrr": _mean_metric(selected, "answer_session_mrr"),
            "context_reader_tokens": _mean_metric(selected, "context_reader_tokens"),
            "context_embedding_tokens": _mean_metric(selected, "context_embedding_tokens"),
            "reader_prompt_tokens": _mean_metric(selected, "reader_prompt_tokens"),
            "retrieval_latency_ms": _mean_metric(selected, "retrieval_latency_ms"),
            "ingestion_latency_ms": _mean_metric(selected, "ingestion_latency_ms"),
            "by_category": by_category,
            "embedding_prompt_tokens": _sum_captured(
                [row for row in system_calls if row.get("role") == "embedding" and row.get("success") is True],
                "prompt_tokens",
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
    roles = ("reader_answer", "memory_ingest", "memory_reasoning", "embedding", "judge_local")
    for system in systems:
        rows = [row for row in call_rows if row.get("system") == system]
        role_usage = {}
        for role in roles:
            role_rows = [row for row in rows if row.get("role") == role]
            role_usage[role] = {
                "provider": sorted({row["provider"] for row in role_rows if row.get("provider")}),
                "calls": len(role_rows),
                "prompt_tokens": _sum_captured(role_rows, "prompt_tokens"),
                "completion_tokens": _sum_captured(role_rows, "completion_tokens"),
                "usage_capture_status": (
                    "CAPTURED" if any(row.get("prompt_tokens") is not None or row.get("completion_tokens") is not None for row in role_rows)
                    else "NOT_CAPTURED"
                ),
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
        f"- ContextBundle content valid: `{metrics.get('context_bundle_content_valid')}`",
        f"- ContextBundle hash frozen: `{metrics.get('context_bundle_hash_frozen')}`",
        "- TEST access: `false`",
        "- Hosted API: `NONE`; judge: `NONE`; required API key: `NONE`",
        "",
        "| System | Quality N | Infra failures | Token F1 | Precision | Recall | Norm. EM | Abstention N | Abstention accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in systems:
        prediction = metrics["systems"][system]
        lines.append(
            f"| {system} | {prediction['quality_n']} | {prediction['infra_failure_n']} | "
            f"{_format_metric(prediction['f1_mean'])} | "
            f"{_format_metric(prediction['token_precision_mean'])} | "
            f"{_format_metric(prediction['token_recall_mean'])} | "
            f"{_format_metric(prediction['normalized_exact_match_mean'])} | "
            f"{prediction['abstention_n']} | "
                f"{_format_metric(prediction['abstention_accuracy'])} |"
            )
    predictions = _latest_predictions(read_jsonl(run_dir / "predictions.jsonl"))
    simplemem_trace = next(
        (
            predictions[("simplemem", question_id)].get("memory_system_diagnostics")
            for question_id in manifest["question_ids"]
            if ("simplemem", question_id) in predictions
            and predictions[("simplemem", question_id)].get("memory_system_diagnostics")
        ),
        None,
    )
    if simplemem_trace:
        trace_calls = simplemem_trace.get("calls", {})
        lines.extend([
            "",
            "## SimpleMem Retrieval Trace",
            "",
            f"Official source: `{simplemem_trace.get('source_tag')}` / `{simplemem_trace.get('source_commit')}`.",
            "This is adapter telemetry, not a quality or ranking claim.",
            "",
            "| Semantic | Keyword | Structured | Merge/deduplicate | Reflection | Native answer head |",
            "|---:|---:|---:|---:|---:|---|",
            "| " + " | ".join(str(trace_calls.get(key, 0)) for key in (
                "semantic", "keyword", "structured", "merge_deduplicate", "reflection"
            )) + f" | {simplemem_trace.get('native_answer_head_invoked')} |",
        ])
    lines.extend([
        "", "## Category Metrics", "",
        "| System | Category | N | Token F1 | Precision | Recall | Norm. EM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for system in systems:
        for category in CATEGORIES:
            summary = metrics["systems"][system]["by_category"][category]
            lines.append(
                f"| {system} | {category} | {summary['n']} | "
                f"{_format_metric(summary['token_f1'])} | "
                f"{_format_metric(summary['token_precision'])} | "
                f"{_format_metric(summary['token_recall'])} | "
                f"{_format_metric(summary['normalized_exact_match'])} |"
            )
    lines.extend([
        "",
        "## Memory Diagnostics",
        "",
        "| System | Context reader tokens | Context embedding tokens | Recall@5 | Recall@10 | MRR | Reader prompt tokens | Retrieval ms | Ingestion ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for system in systems:
        diagnostic = metrics["memory_diagnostics"][system]
        lines.append(
            f"| {system} | {_format_metric(diagnostic['context_reader_tokens'])} | "
            f"{_format_metric(diagnostic['context_embedding_tokens'])} | "
            f"{_format_metric(diagnostic['answer_session_recall_at_5'])} | "
            f"{_format_metric(diagnostic['answer_session_recall_at_10'])} | "
            f"{_format_metric(diagnostic['mrr'])} | "
            f"{_format_metric(diagnostic['reader_prompt_tokens'])} | "
            f"{_format_metric(diagnostic['retrieval_latency_ms'])} | "
            f"{_format_metric(diagnostic['ingestion_latency_ms'])} |"
        )
    lines.extend([
        "",
        "## Retrieval Diagnostics by Category",
        "",
        "| System | Category | Context reader tokens | Recall@5 | Recall@10 | MRR | Retrieval ms | Ingestion ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for system in systems:
        for category in CATEGORIES:
            diagnostic = metrics["memory_diagnostics"][system]["by_category"][category]
            lines.append(
                f"| {system} | {category} | {_format_metric(diagnostic['context_reader_tokens'])} | "
                f"{_format_metric(diagnostic['answer_session_recall_at_5'])} | "
                f"{_format_metric(diagnostic['answer_session_recall_at_10'])} | "
                f"{_format_metric(diagnostic['mrr'])} | "
                f"{_format_metric(diagnostic['retrieval_latency_ms'])} | "
                f"{_format_metric(diagnostic['ingestion_latency_ms'])} |"
        )
    warning_summary = metrics.get("baseline_warnings", {})
    lines.extend([
        "",
        "## Baseline Warnings",
        "",
        "`BASELINE_INTERNAL_WARNING` is reported separately from `INFRA_FAILURE`; full captured log lines and prompts are not stored.",
        "",
        "| System | Internal warning count | Infra failure count | Warning types |",
        "|---|---:|---:|---|",
    ])
    for system in systems:
        summary = warning_summary.get(system, {})
        lines.append(
            f"| {system} | {summary.get('baseline_internal_warning_count', 0)} | "
            f"{summary.get('infra_failure_count', 0)} | "
            f"{', '.join(sorted(summary.get('by_type', {}))) or '-'} |"
        )
    lines.extend([
        "",
        "Session-retrieval metrics are null when a baseline does not expose auditable source-session provenance; this is not scored as a retrieval miss.",
        "Context reader tokens use the loaded frozen Qwen3-8B llama.cpp tokenizer; context embedding tokens use the frozen Qwen3-Embedding-0.6B tokenizer. Context-size comparisons use context reader tokens.",
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
        if any(row.get("prompt_tokens", 131073) + 256 > 131072 for row in answer_calls):
            return False
    return True


def _validate_mem1d1_completion(
    manifest: dict[str, Any],
    systems: list[str],
    question_ids: list[str],
    predictions: dict[tuple[str, str], dict[str, Any]],
    call_rows: list[dict[str, Any]],
    warning_rows: list[dict[str, Any]],
    *,
    fullcontext_valid: bool,
    context_valid: bool,
) -> None:
    if manifest.get("selection_manifest_sha256") != D1_SELECTION_SHA256:
        raise RuntimeError("MEM-1D1 run manifest is not bound to the frozen selection artifact")
    dataset = manifest.get("dataset", {})
    if (
        dataset.get("id") != D1_DATASET_ID
        or dataset.get("revision") != D1_DATASET_REVISION
        or dataset.get("sha256") != D1_DATASET_SHA256
        or dataset.get("test_access") is not False
        or manifest.get("test_access") is not False
        or manifest.get("track") != "main_local_only_context_controlled"
    ):
        raise RuntimeError("MEM-1D1 run manifest violates the frozen local DEV protocol")
    if systems != list(SYSTEMS) or question_ids != list(D1_QUESTION_IDS):
        raise RuntimeError("MEM-1D1 terminal matrix differs from the frozen 5-by-10 selection")
    if not all(predictions[key].get("quality_status") == "OK" for key in predictions):
        raise RuntimeError("MEM-1D1 contains unresolved infrastructure-failure predictions")
    template_hashes = {
        row.get("shared_reader_template_sha256") for row in predictions.values()
    }
    if template_hashes != {D1_READER_TEMPLATE_SHA256}:
        raise RuntimeError("MEM-1D1 did not use the frozen shared-reader template for every row")
    if not fullcontext_valid or not context_valid:
        raise RuntimeError("MEM-1D1 FullContext or ContextBundle integrity gate failed")

    roles = manifest.get("roles", {})
    reader = roles.get("reader_answer_model", {})
    memory_llm = roles.get("memory_internal_llm", {})
    embedding = roles.get("embedding_model", {})
    judge = roles.get("judge_model", {})
    reader_runtime = reader.get("runtime", {})
    reader_generation = reader.get("generation", {})
    if (
        reader.get("provider") != "local_qwen"
        or reader.get("artifact_sha256") != "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
        or urlsplit(reader.get("endpoint", "")).hostname not in {"127.0.0.1", "::1"}
        or reader_generation != {
            "temperature": 0,
            "seed": 42,
            "enable_thinking": False,
            "answer_max_new_tokens": 256,
        }
        or memory_llm.get("provider") != "local_qwen"
        or memory_llm.get("artifact_sha256") != reader.get("artifact_sha256")
    ):
        raise RuntimeError("MEM-1D1 reader or memory-internal LLM differs from the frozen local Qwen")
    if (
        reader_runtime.get("server") != "llama.cpp llama-server 10068 (571d0d540)"
        or reader_runtime.get("server_binary_sha256") != "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
        or reader_runtime.get("gpu_layers") != 99
        or reader_runtime.get("flash_attention") is not True
        or reader_runtime.get("kv_cache_type_k") != "q4_0"
        or reader_runtime.get("kv_cache_type_v") != "q4_0"
        or reader_runtime.get("context_length") != 131072
        or reader_runtime.get("rope_scaling") != "yarn"
        or reader_runtime.get("rope_scale") != 4
        or reader_runtime.get("rope_original_context") != 32768
    ):
        raise RuntimeError("MEM-1D1 llama.cpp runtime differs from the frozen long-context configuration")
    if (
        embedding.get("provider") != "local_transformers"
        or embedding.get("repo") != "Qwen/Qwen3-Embedding-0.6B"
        or embedding.get("revision") != "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
        or embedding.get("model_sha256") != "9d2d790d6448ef2c0911ffeb03f959d035c71ac3d2b14b7d586f2d2b39fb0efa"
        or embedding.get("weights_sha256") != "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
        or embedding.get("device") != "cuda:0"
        or embedding.get("dtype") != "float16"
    ):
        raise RuntimeError("MEM-1D1 embedding runtime differs from the frozen local CUDA model")
    if judge.get("provider") != "none" or judge.get("used") is not False:
        raise RuntimeError("MEM-1D1 must not configure or use a judge")
    if roles.get("memory_system", {}).get("systems") != list(SYSTEMS):
        raise RuntimeError("MEM-1D1 run manifest system identities differ from the frozen matrix")
    if any("judge" in str(row.get("role", "")).lower() for row in call_rows):
        raise RuntimeError("MEM-1D1 call ledger contains a judge invocation")
    allowed_providers = {"local_qwen", "local_transformers"}
    if any(row.get("provider") not in allowed_providers for row in call_rows):
        raise RuntimeError("MEM-1D1 call ledger contains a non-local or unknown provider")
    if any(row.get("classification") == "INFRA_FAILURE" for row in warning_rows):
        raise RuntimeError("MEM-1D1 warning ledger contains an unresolved infrastructure failure")

    for question_id in D1_QUESTION_IDS:
        row = predictions[("simplemem", question_id)]
        trace = row.get("memory_system_diagnostics") or {}
        if (
            trace.get("source_tag") != "v0.1.0"
            or trace.get("source_commit") != PINNED_SIMPLEMEM_SHA
            or trace.get("native_answer_head_invoked") is not False
            or trace.get("planning_enabled") is not True
            or not isinstance(trace.get("reflection_enabled"), bool)
            or not {
                "semantic", "keyword", "structured", "merge_deduplicate", "reflection"
            }.issubset(trace.get("calls", {}))
        ):
            raise RuntimeError(f"SimpleMem fidelity telemetry missing for {question_id}")

    runner_code_sha = manifest.get("runner_code_sha256")
    expected_cache_patch_hash = sha256_bytes(canonical_json({
        "memeval_patch_sha256": manifest.get("code_patch_sha256"),
        "healthcopilot_runner_sha256": runner_code_sha,
    }))
    expected_configs = roles.get("memory_system", {}).get("system_config_sha256", {})
    expected_reader_sha = roles.get("reader_answer_model", {}).get("artifact_sha256")
    expected_embedding_sha = roles.get("embedding_model", {}).get("model_sha256")
    expected_prompt_hash = roles.get("memory_system", {}).get("prompt_source_sha256")
    for (system, question_id), row in predictions.items():
        identity = row.get("cache_identity", {})
        verified_identity_hash(identity)
        if (
            identity.get("system") != system
            or identity.get("question_id") != question_id
            or identity.get("dataset_sha256") != D1_DATASET_SHA256
            or identity.get("system_config_hash") != expected_configs.get(system)
            or identity.get("reader_artifact_sha256") != expected_reader_sha
            or identity.get("code_patch_hash") != expected_cache_patch_hash
            or identity.get("prompt_hashes", {}).get("memory_system_and_prompts") != expected_prompt_hash
        ):
            raise RuntimeError(f"MEM-1D1 cache identity mismatch for {system}/{question_id}")
        if system == "fullcontext":
            if identity.get("embedding_model") is not None or identity.get("embedding_artifact_sha256") is not None:
                raise RuntimeError("FullContext cache identity unexpectedly includes embeddings")
        elif identity.get("embedding_artifact_sha256") != expected_embedding_sha:
            raise RuntimeError(f"Shared embedding identity mismatch for {system}/{question_id}")


def _context_bundle_rows_valid(
    run_dir: Path,
    systems: list[str],
    question_ids: list[str],
    predictions: dict[tuple[str, str], dict[str, Any]],
    *,
    freeze: bool,
) -> bool:
    bundle_path = run_dir / "context_bundles.jsonl"
    sidecar_path = run_dir / "context_bundles.sha256"
    if not bundle_path.exists():
        return False
    if sidecar_path.exists() and not verify_hash_sidecar(bundle_path, sidecar_path):
        raise RuntimeError("Frozen ContextBundle hash mismatch; refusing to score")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(bundle_path):
        latest[(row.get("system"), row.get("question_id"))] = row
    expected = {(system, qid) for system in systems for qid in question_ids}
    if set(latest) != expected:
        return False
    for key, row in latest.items():
        bundle = row.get("context_bundle")
        prediction = predictions.get(key)
        if (
            not isinstance(bundle, dict)
            or not verify_context_bundle(bundle)
            or prediction is None
            or prediction.get("context_bundle_sha256") != bundle.get("context_bundle_sha256")
        ):
            return False
    if sidecar_path.exists():
        return True
    if freeze:
        write_hash_sidecar(bundle_path, sidecar_path)
    return True


def _context_bundle_hash_frozen(run_dir: Path) -> bool:
    bundle_path = run_dir / "context_bundles.jsonl"
    sidecar_path = run_dir / "context_bundles.sha256"
    return sidecar_path.is_file() and verify_hash_sidecar(bundle_path, sidecar_path)


def _finalize_generation(run_dir: Path, manifest: dict[str, Any], systems: list[str], question_ids: list[str], embedding_model: str | None) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    sidecar_path = run_dir / "predictions.sha256"
    d1_run = manifest.get("selection_manifest_sha256") == D1_SELECTION_SHA256
    if sidecar_path.exists():
        if d1_run:
            _verify_frozen_evidence(run_dir, require_ledgers=True)
        elif not verify_hash_sidecar(predictions_path, sidecar_path):
            raise RuntimeError("Frozen prediction hash mismatch; refusing to score")
    rows = read_jsonl(predictions_path)
    latest = _latest_predictions(rows)
    expected = {(name, question_id) for name in systems for question_id in question_ids}
    if set(latest) != expected:
        missing = sorted(expected - set(latest))
        unexpected = sorted(set(latest) - expected)
        raise RuntimeError(f"Generation coverage mismatch; missing={missing}, unexpected={unexpected}")
    call_rows = read_jsonl(run_dir / "call_ledger.jsonl")
    warning_path = run_dir / "baseline_warnings.jsonl"
    warning_path.touch(exist_ok=True)
    warning_rows = read_jsonl(warning_path)
    every_answer_valid = all(latest[key].get("quality_status") == "OK" for key in expected)
    long_context_valid = "fullcontext" not in systems or _fullcontext_is_valid(latest, question_ids, call_rows)
    context_track = manifest.get("track") == "main_local_only_context_controlled"
    context_valid = (
        _context_bundle_rows_valid(
            run_dir, systems, question_ids, latest,
            freeze=every_answer_valid and long_context_valid and not d1_run,
        )
        if context_track else True
    )
    if d1_run:
        _validate_mem1d1_completion(
            manifest,
            systems,
            question_ids,
            latest,
            call_rows,
            warning_rows,
            fullcontext_valid=long_context_valid,
            context_valid=context_valid,
        )
        if every_answer_valid and long_context_valid and context_valid:
            _context_bundle_rows_valid(
                run_dir, systems, question_ids, latest, freeze=True
            )
    frozen = sidecar_path.exists()
    if not frozen and every_answer_valid and long_context_valid and context_valid:
        digest = write_hash_sidecar(predictions_path, sidecar_path)
        write_hash_sidecar(
            run_dir / "call_ledger.jsonl", run_dir / "call_ledger.sha256"
        )
        write_hash_sidecar(
            warning_path, run_dir / "baseline_warnings.sha256"
        )
        if d1_run:
            _verify_frozen_evidence(run_dir, require_ledgers=True)
    elif frozen:
        digest = sidecar_path.read_text(encoding="ascii").split()[0]
    else:
        digest = None

    metrics = {
        "prediction_sha256": digest,
        "prediction_frozen": digest is not None,
        "context_bundle_content_valid": context_valid if context_track else None,
        "context_bundle_hash_frozen": (
            _context_bundle_hash_frozen(run_dir) if context_track else None
        ),
        "fullcontext_validation": "PASS" if long_context_valid else "FAIL_OR_MISSING_TRUNCATION_TELEMETRY",
        "systems": _summarize_predictions(rows, systems, question_ids),
        "memory_diagnostics": _summarize_memory_diagnostics(rows, systems, question_ids, call_rows),
        "baseline_warnings": _summarize_baseline_warnings(warning_rows, systems),
        "baseline_warnings_sha256": sha256_file(warning_path),
        "call_ledger_sha256": sha256_file(run_dir / "call_ledger.jsonl"),
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


def _shared_reader_messages(
    question: str,
    serialized_context: str,
    question_date: str,
) -> list[dict[str, str]]:
    _, _, system_template, user_template = load_final_reader_contract()
    return build_final_reader_messages(
        question,
        question_date,
        serialized_context,
        system_template=system_template,
        user_template=user_template,
    )


def _reader_context_token_counter(config):
    parsed = urlsplit(config.reader_base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Reader tokenizer must use the explicit loopback Qwen endpoint")
    url = config.reader_base_url.removesuffix("/v1") + "/tokenize"

    def count(texts: list[str]) -> int:
        import httpx

        total = 0
        with httpx.Client(timeout=120, trust_env=False) as client:
            for text in texts:
                if not text:
                    continue
                response = client.post(
                    url,
                    json={"content": text, "add_special": False},
                )
                response.raise_for_status()
                tokens = response.json().get("tokens")
                if not isinstance(tokens, list):
                    raise TypeError("llama.cpp /tokenize did not return token IDs")
                total += len(tokens)
        return total

    return count


def _failure_attribution_hint(
    question_id: str, predicted: str | None, expected: str, context: str,
) -> list[str]:
    import re

    if predicted is None:
        return ["INFRA_FAILURE"]
    labels = []
    is_abstention = _is_abstention(predicted)
    if _is_abstention_question(question_id) and not is_abstention:
        labels.append("SHOULD_ABSTAIN")
    elif not _is_abstention_question(question_id) and is_abstention:
        labels.append("FALSE_ABSTENTION")
    expected_tokens = set(re.findall(r"\w+", expected.lower()))
    context_tokens = set(re.findall(r"\w+", context.lower()))
    predicted_tokens = set(re.findall(r"\w+", predicted.lower()))
    if expected_tokens and not expected_tokens.issubset(predicted_tokens):
        labels.append(
            "CONTEXT_HAS_ANSWER_READER_MISSED"
            if expected_tokens.issubset(context_tokens)
            else "CONTEXT_MISSING_ANSWER"
        )
    return labels


def _run_context_controlled(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    systems: dict[str, Any],
    conversations: list[dict[str, Any]],
    question_ids: list[str],
    split: dict[str, Any],
    source_hash: str,
    cache_code_hash: str,
    config_hashes: dict[str, str],
    config,
    reader_sha: str,
    embedding_artifact: dict[str, Any],
    local_protocol: dict[str, Any],
    category_names: dict[str, str],
) -> dict[str, Any]:
    from agents_memory.healthcopilot_provider import (
        answer_request_kwargs,
        call_context,
        clear_question_metrics,
        configure_call_ledger,
        measure_full_context_prompt,
        reader_client,
    )

    run_dir = args.run_dir
    contract, contract_sha, system_template, user_template = load_final_reader_contract()
    verify_contract_binding(manifest.get("final_reader_contract"), contract, contract_sha)
    predictions_path = run_dir / "predictions.jsonl"
    bundles_path = run_dir / "context_bundles.jsonl"
    sidecar_path = run_dir / "predictions.sha256"
    if sidecar_path.exists():
        frozen_rows = read_jsonl(predictions_path)
        if not frozen_rows or any(
            row.get("final_reader_contract_sha256") != contract_sha
            for row in frozen_rows
        ):
            raise RuntimeError("Frozen predictions do not match the final reader contract SHA")
        return _finalize_generation(
            run_dir,
            manifest,
            manifest["roles"]["memory_system"]["systems"],
            manifest["question_ids"],
            config.embedding_model,
        )

    configure_call_ledger(run_dir / "call_ledger.jsonl")
    warnings_path = run_dir / "baseline_warnings.jsonl"
    warnings_path.touch(exist_ok=True)
    bundle_rows = read_jsonl(bundles_path)
    latest_bundles = {
        (row.get("system"), row.get("question_id")): row
        for row in bundle_rows
    }
    source_prompt_hash = {
        "memory_system_and_prompts": source_hash,
        "final_reader_contract_sha256": contract_sha,
    }
    provider = importlib.import_module("agents_memory.healthcopilot_provider")
    from context_bundle import build_context_bundle

    # Initialize once so every bundle uses the same frozen tokenizer/settings.
    embedding_runtime = provider.initialize_local_embedding(config)
    reader_token_counter = _reader_context_token_counter(config)
    embedding_tokenizer_name = (
        f"{embedding_runtime.artifact.repo}@{embedding_runtime.artifact.revision}:tokenizer"
    )
    reader_tokenizer_name = "frozen Qwen3-8B llama.cpp /tokenize; add_special=false"
    answer_budget = local_protocol["roles"]["reader_answer_model"]["generation"][
        "answer_max_new_tokens"
    ]
    for system_name, system in systems.items():
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
                embedding_artifact_sha256=(
                    embedding_artifact["model_sha256"] if embedding_model else None
                ),
                code_patch_hash=cache_code_hash,
            )
            cached_prediction = find_cached_prediction(predictions_path, cache_identity)
            cached_bundle = latest_bundles.get((system_name, question_id))
            if cached_prediction is not None:
                if (
                    cached_bundle is None
                    or cached_prediction.get("context_bundle_sha256")
                    != cached_bundle.get("context_bundle", {}).get("context_bundle_sha256")
                    or not verify_context_bundle(cached_bundle.get("context_bundle", {}))
                ):
                    raise RuntimeError(
                        f"Cached prediction for {system_name}/{question_id} lacks its valid ContextBundle"
                    )
                continue

            clear_question_metrics()
            call_count_before = len(read_jsonl(run_dir / "call_ledger.jsonl"))
            started = time.perf_counter()
            bundle = None
            predicted = None
            context_row = None
            error: BaseException | None = None
            captured_output = io.StringIO()
            adapter_completed = False
            try:
                with_qa = {**conversation, "qa": [qa]}
                with contextlib.redirect_stdout(captured_output), contextlib.redirect_stderr(captured_output):
                    context_rows = system["fn"](
                        with_qa,
                        config.reader_model,
                        False,
                        category_names=category_names,
                        judge_fn="longmemeval",
                        context_only=True,
                    )
                if len(context_rows) != 1 or context_rows[0].get("quality_status") != "OK":
                    raise RuntimeError("Adapter did not return exactly one valid ContextBundle input")
                adapter_completed = True
                context_row = context_rows[0]
                bundle = build_context_bundle(
                    system=system_name,
                    question_id=question_id,
                    items=context_row["context_items"],
                    embedding_token_counter=embedding_runtime.count_tokens,
                    embedding_tokenizer_name=embedding_tokenizer_name,
                    reader_token_counter=reader_token_counter,
                    reader_tokenizer_name=reader_tokenizer_name,
                    provenance_available=context_row["provenance_available"],
                    retrieval_latency_ms=context_row["retrieval_latency_ms"],
                    ingestion_latency_ms=context_row["ingestion_latency_ms"],
                ).to_dict()
                if not verify_context_bundle(bundle):
                    raise RuntimeError("ContextBundle canonical hash verification failed")
                bundle_row = {
                    "system": system_name,
                    "question_id": question_id,
                    "cache_identity": cache_identity,
                    "context_bundle": bundle,
                }
                append_jsonl(bundles_path, bundle_row)
                latest_bundles[(system_name, question_id)] = bundle_row

                messages = build_final_reader_messages(
                    qa["question"],
                    qa["question_date"],
                    bundle["serialized_context"],
                    system_template=system_template,
                    user_template=user_template,
                )
                shared_template_sha = contract["template"][
                    "canonical_message_template_sha256"
                ]
                prompt_sha = sha256_bytes(canonical_json(messages))
                if system_name == "fullcontext":
                    preflight = measure_full_context_prompt(
                        messages,
                        config,
                        max_model_length=local_protocol["roles"]["reader_answer_model"][
                            "long_context"
                        ]["max_model_length"],
                        output_reserve=answer_budget,
                    )
                    if preflight["truncated"]:
                        raise RuntimeError("FullContext shared-reader prompt exceeds its 131072-token limit")

                client = reader_client(
                    "reader_answer", system_name, question_id, config
                )
                with call_context(system_name, question_id, "reader_answer"):
                    response = client.chat.completions.create(
                        **answer_request_kwargs(
                            config,
                            messages=messages,
                            max_tokens=answer_budget,
                        )
                    )
                content = response.choices[0].message.content
                if not isinstance(content, str):
                    raise TypeError("Shared reader returned no text answer")
                predicted = content.strip()
            except Exception as caught:  # noqa: BLE001 - all provider/adapter failures become explicit INFRA_FAILURE rows
                error = caught

            call_rows = read_jsonl(run_dir / "call_ledger.jsonl")
            new_calls = call_rows[call_count_before:]
            warning_rows = _baseline_warning_rows(
                system_name,
                question_id,
                captured_output.getvalue(),
                new_calls,
                adapter_completed=adapter_completed,
            )
            for warning in warning_rows:
                append_jsonl(warnings_path, warning)
            quality_status = "OK" if predicted is not None else "INFRA_FAILURE"
            answer_scores = (
                _answer_metrics(predicted, qa["answer"])
                if predicted is not None else {
                    "token_precision": None,
                    "token_recall": None,
                    "normalized_exact_match": None,
                }
            )
            answer_call_rows = [
                row for row in new_calls
                if row.get("role") == "reader_answer" and row.get("success") is True
            ]
            prompt_tokens = _sum_captured(answer_call_rows, "prompt_tokens")
            bundle_dict = bundle if bundle is not None else {}
            serialized_context = bundle_dict.get("serialized_context", "")
            row = {
                "system": system_name,
                "question_id": question_id,
                "sample_id": question_id,
                "question": qa["question"],
                "question_date": qa["question_date"],
                "ground_truth": qa["answer"],
                "predicted": predicted,
                "category": qa["category"],
                "category_name": qa["category"],
                "answer_session_ids": qa.get("answer_session_ids", []),
                "quality_status": quality_status,
                "f1": answer_scores.get("f1"),
                "token_precision": answer_scores["token_precision"],
                "token_recall": answer_scores["token_recall"],
                "normalized_exact_match": answer_scores["normalized_exact_match"],
                "reader_prompt_tokens": prompt_tokens,
                "baseline_warnings": warning_rows,
                "context_reader_tokens": bundle_dict.get("context_reader_tokens"),
                "context_embedding_tokens": bundle_dict.get("context_embedding_tokens"),
                "retrieval_latency_ms": bundle_dict.get("retrieval_latency_ms"),
                "ingestion_latency_ms": bundle_dict.get("ingestion_latency_ms"),
                "session_provenance_available": bundle_dict.get("provenance_available"),
                "ranked_session_groups": (
                    [item.get("source_session_ids", []) for item in bundle_dict.get("items", [])]
                    if bundle_dict.get("provenance_available") else None
                ),
                "context_bundle_sha256": bundle_dict.get("context_bundle_sha256"),
                "failure_attribution_hint": (
                    _failure_attribution_hint(
                        question_id, predicted, qa["answer"], serialized_context
                    ) if predicted is not None else ["INFRA_FAILURE"]
                ),
                "failure_attribution_heuristic": True,
                "memory_system_diagnostics": (
                    context_row.get("retrieval_trace") if context_row is not None else None
                ),
                "shared_reader_template_sha256": (
                    shared_template_sha if predicted is not None else None
                ),
                "final_reader_contract_sha256": contract_sha,
                "shared_reader_prompt_sha256": prompt_sha if predicted is not None else None,
                "cache_identity": cache_identity,
                "system_wall_time_ms": round((time.perf_counter() - started) * 1000, 3),
                "reader_error_type": type(error).__name__ if error is not None else None,
                "reader_error_chain": _exception_chain(error) if error is not None else [],
            }
            append_jsonl(predictions_path, row)
            if error is not None:
                failures = read_jsonl(run_dir / "failures.jsonl")
                embedding_failed = any(
                    call.get("role") == "embedding" and call.get("success") is False
                    for call in new_calls
                )
                append_jsonl(run_dir / "failures.jsonl", {
                    "question_id": question_id,
                    "system": system_name,
                    "failure_type": "INFRA_EMBEDDING" if embedding_failed else "INFRA_FAILURE",
                    "exception_chain": _exception_chain(error),
                    "quality_status": "INFRA_FAILURE",
                    "prior_failure_count": len(failures),
                })
                if getattr(args, "mem1d1_frozen_10", False):
                    raise RuntimeError(
                        f"MEM-1D1 infrastructure failure at {system_name}/{question_id}; stopped for review"
                    ) from error

    locked_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    locked_systems = locked_manifest["roles"]["memory_system"]["systems"]
    locked_questions = locked_manifest["question_ids"]
    latest = _latest_predictions(read_jsonl(predictions_path))
    expected = {
        (system, question_id)
        for system in locked_systems
        for question_id in locked_questions
    }
    if set(latest) != expected or any(
        latest[key].get("quality_status") != "OK" for key in expected & set(latest)
    ):
        failed = sum(
            latest[key].get("quality_status") != "OK"
            for key in expected & set(latest)
        )
        print(
            f"MEM-1D1 chunk recorded; terminal freeze deferred; "
            f"predictions={len(expected & set(latest))}/{len(expected)}; infra_rows={failed}"
        )
        return None

    metrics = _finalize_generation(
        run_dir,
        locked_manifest,
        locked_systems,
        locked_questions,
        config.embedding_model,
    )
    print(f"Prediction artifact frozen: {metrics['prediction_frozen']}")
    return metrics


def generate(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["HC_MEMORY_TRACK"] = "main_local_only"
    os.environ["HC_MEM1_RUN_DIR"] = str(args.run_dir.resolve())
    split, model_protocol, local_protocol, question_ids = _load_locked_inputs(args)
    final_reader_contract_binding = None
    if args.answer_track == "context_controlled":
        contract, contract_sha, _, _ = load_final_reader_contract()
        final_reader_contract_binding = contract_binding(contract, contract_sha)
    if args.mem1d1_frozen_10:
        _verify_frozen_evidence(args.run_dir, require_ledgers=True)
        if not (args.run_dir / "predictions.sha256").exists():
            latest_existing = _latest_predictions(
                read_jsonl(args.run_dir / "predictions.jsonl")
            )
            if any(row.get("quality_status") != "OK" for row in latest_existing.values()):
                raise RuntimeError(
                    "MEM-1D1 run contains an unresolved infrastructure row; "
                    "do not retry it under the same frozen code identity"
                )
    compatibility = json.loads((ROOT / "docs/research/memory/baseline_compatibility_matrix.json").read_text(encoding="utf-8"))
    patch_sha = compatibility["patch_provenance"]["patch_sha256"]
    verify_pinned_patch(args.memeval_root, args.patch, patch_sha)

    system_names = list(dict.fromkeys(args.system))
    simplemem_source = None
    if "simplemem" in system_names:
        simplemem_root = ROOT.parent / "external" / "memory" / "SimpleMem"
        simplemem_source = verify_official_simplemem_source(simplemem_root)
        os.environ["HC_SIMPLEMEM_OFFICIAL_ROOT"] = str(simplemem_root.resolve())
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
    records_by_id = {
        conversation["qa"][0]["question_id"]: conversation
        for conversation in conversations
    }
    conversations = [records_by_id[question_id] for question_id in question_ids]
    source_hash = _source_prompt_hash(args.memeval_root / "src")
    roles = _provider_manifest(
        config, system_names, systems, reader_sha, embedding_artifact, slot_context,
        local_protocol["roles"]["memory_internal_llm"]["generation"],
        local_protocol["local_execution_constraints"]["reader_runtime"],
    )
    runner_code_sha = sha256_bytes(canonical_json({
        "runner": sha256_file(__file__),
        "artifact_helpers": sha256_file(Path(__file__).with_name("mem1_artifacts.py")),
        "reader_contract_module": sha256_file(Path(__file__).with_name("final_reader_contract.py")),
        "local_protocol": sha256_file(LOCAL_PROTOCOL_PATH),
    }))
    cache_code_hash = sha256_bytes(canonical_json({
        "memeval_patch_sha256": patch_sha,
        "healthcopilot_runner_sha256": runner_code_sha,
    }))
    config_hashes = {
        name: sha256_bytes(canonical_json({
            "system": name,
            "answer_track": args.answer_track,
            "architecture": systems[name].get("architecture"),
            "infrastructure": systems[name].get("infrastructure"),
            "reader_model": config.reader_model,
            "reader_generation": local_protocol["roles"]["reader_answer_model"]["generation"],
            "reader_runtime": local_protocol["local_execution_constraints"]["reader_runtime"],
            "memory_internal_generation": local_protocol["roles"]["memory_internal_llm"]["generation"],
            "embedding_model": config.embedding_model if name != "fullcontext" else None,
            "embedding_artifact_sha256": embedding_artifact["model_sha256"] if name != "fullcontext" else None,
            "architecture_source": simplemem_source if name == "simplemem" else None,
            "final_reader_contract_sha256": (
                final_reader_contract_binding["sha256"]
                if final_reader_contract_binding is not None else None
            ),
        }))
        for name in system_names
    }
    roles["memory_system"]["system_config_sha256"] = config_hashes
    roles["memory_system"]["prompt_source_sha256"] = source_hash
    if simplemem_source is not None:
        roles["memory_system"]["implementation_sources"] = {"simplemem": simplemem_source}
    roles["reader_answer_model"]["generation"] = local_protocol["roles"]["reader_answer_model"]["generation"]
    roles["embedding_model"]["runtime_settings"] = {
        key: embedding_artifact[key] for key in (
            "repo", "revision", "model_sha256", "weights_sha256", "dimensions",
            "normalized", "query_instruction", "document_instruction", "batch_size",
            "max_batch_tokens", "device", "dtype", "torch_version", "cuda_version",
            "device_name", "max_length", "truncation",
        )
    }
    run_config_hash = sha256_bytes(canonical_json(config_hashes))
    manifest = write_run_manifest(
        args.run_dir / "run_manifest.json",
        run_id=args.run_dir.name,
        track=(
            "main_local_only_context_controlled"
            if args.answer_track == "context_controlled"
            else "main_local_only_native_audit"
        ),
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
        created_at=datetime.now(UTC).isoformat(),
        selection_manifest_sha256=args.d1_selection_sha256,
        final_reader_contract=final_reader_contract_binding,
    )

    predictions_path = args.run_dir / "predictions.jsonl"
    sidecar_path = args.run_dir / "predictions.sha256"
    if sidecar_path.exists():
        if final_reader_contract_binding is not None:
            frozen_rows = read_jsonl(predictions_path)
            if not frozen_rows or any(
                row.get("final_reader_contract_sha256")
                != final_reader_contract_binding["sha256"]
                for row in frozen_rows
            ):
                raise RuntimeError("Frozen predictions do not match the locked final reader contract")
        _finalize_generation(args.run_dir, manifest, system_names, question_ids, config.embedding_model)
        from agents_memory.healthcopilot_provider import release_local_embedding_runtimes
        release_local_embedding_runtimes()
        return
    if args.answer_track == "context_controlled":
        execution_question_ids = set(args.execution_question_ids)
        execution_conversations = [
            conversation for conversation in conversations
            if conversation["qa"][0]["question_id"] in execution_question_ids
        ]
        execution_systems = {
            name: systems[name] for name in args.execution_systems
        }
        try:
            _run_context_controlled(
                args=args,
                manifest=manifest,
                systems=execution_systems,
                conversations=execution_conversations,
                question_ids=question_ids,
                split=split,
                source_hash=source_hash,
                cache_code_hash=cache_code_hash,
                config_hashes=config_hashes,
                config=config,
                reader_sha=reader_sha,
                embedding_artifact=embedding_artifact,
                local_protocol=local_protocol,
                category_names=CATEGORY_NAMES,
            )
        finally:
            from agents_memory.healthcopilot_provider import release_local_embedding_runtimes
            release_local_embedding_runtimes()
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
            except Exception as error:  # noqa: BLE001 - isolate failures per requested prediction
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
                result["reader_prompt_tokens"] = _sum_captured(answer_calls, "prompt_tokens")
            if isinstance(result.get("predicted"), str):
                extra_metrics = _answer_metrics(
                    result["predicted"], str(result.get("ground_truth", ""))
                )
                extra_metrics.pop("f1")
                result.update(extra_metrics)
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
    from agents_memory.healthcopilot_provider import release_local_embedding_runtimes
    release_local_embedding_runtimes()


def main() -> None:
    args = _parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if len(args.system) != len(set(args.system)):
        raise SystemExit("Duplicate --system arguments are not allowed")
    generate(args)


if __name__ == "__main__":
    main()
