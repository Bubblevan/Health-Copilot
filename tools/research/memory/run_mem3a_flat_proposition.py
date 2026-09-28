"""Frozen-ten M10-FlatProp proposition-memory diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_mem1d3_reader as reader_runtime
import run_mem2a_m10_base as mem2a
import run_mem2b_rank_aware_projection as mem2b
import run_mem2c_rawspan as mem2c
import run_mem2d_semantic_retrieval as mem2d
from final_reader_contract import build_reader_messages, load_final_reader_contract
from local_qwen3_embedding import (
    BATCH_SIZE,
    DEVICE,
    DIMENSIONS,
    DTYPE,
    MAX_BATCH_TOKENS,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_TREE_SHA256,
    QUERY_INSTRUCTION,
    WEIGHTS_SHA256,
    LocalQwen3Embedding,
    sha256_file,
    verify_local_model,
)
from mem1_artifacts import read_jsonl

from health_ai_copilot.runtime.context_manager import (
    ContextItemCategory,
    ContextManager,
    DeterministicTokenEstimator,
)
from health_ai_copilot.runtime.memory import (
    InMemoryMemoryStore,
    MemoryKind,
    MemoryOperation,
    MemoryOperationType,
    MemoryQuery,
    MemorySensitivity,
    MemorySourceType,
    MemoryStatus,
    SQLiteMemoryStore,
)

RUN_ID = "mem3a-flat-proposition-10-20260928"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
RUNS_ROOT = ROOT / "runs" / "memory" / "mem2"
MEM2C_DIR = RUNS_ROOT / "mem2c-rawspan-10-20260928"
MEM2D_DIR = RUNS_ROOT / "mem2d-semantic-retrieval-10-20260928"
MEM2C_INVENTORY = MEM2C_DIR / "memory_inventory.jsonl"
MEM2D_MANIFEST = MEM2D_DIR / "run_manifest.json"
DATASET_PATH = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
PROMPT_PATH = ROOT / "docs/research/memory/flat_proposition_extractor_v1.txt"
EXTRACTOR_CONTRACT_PATH = ROOT / "docs/research/memory/flat_proposition_extractor_contract.json"
RETRIEVAL_CONTRACT_PATH = (
    ROOT / "docs/research/memory/flat_proposition_retrieval_view_contract.json"
)
PROJECTION_CONTRACT_PATH = ROOT / "docs/research/memory/rank_aware_projection_contract.json"
READER_CONTRACT_PATH = ROOT / "docs/research/memory/final_reader_contract.json"
CASE_REVIEW_PATH = ROOT / "docs/research/memory/mem_3a_case_review.json"
PROTOCOL_PATH = ROOT / "docs/research/memory/mem_3a_flat_proposition_memory.md"
RUN_REPORT_PATH = RUN_DIR / "report.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
PREFLIGHT_PATH = RUN_DIR / "writer_preflight.json"
EXTRACTION_MANIFEST_PATH = RUN_DIR / "session_extraction_manifest.json"
EXTRACTIONS_PATH = RUN_DIR / "session_extractions.jsonl"
FLAT_PROPOSITIONS_PATH = RUN_DIR / "flat_propositions.jsonl"
WRITER_LEDGER_PATH = RUN_DIR / "writer_call_ledger.jsonl"
MATERIALIZATION_LEDGER_PATH = RUN_DIR / "materialization_ledger.jsonl"
CANDIDATE_GROUPS_PATH = RUN_DIR / "candidate_revision_groups.json"
EMBEDDING_MANIFEST_PATH = RUN_DIR / "embedding_manifest.json"
DENSE_TOP8_PATH = RUN_DIR / "dense_top8.jsonl"
CONTEXT_PLANS_PATH = RUN_DIR / "context_plans.jsonl"
CONTEXT_BUNDLES_PATH = RUN_DIR / "context_bundles.jsonl"
PREDICTIONS_PATH = RUN_DIR / "predictions.jsonl"
READER_LEDGER_PATH = RUN_DIR / "reader_call_ledger.jsonl"
METRICS_PATH = RUN_DIR / "deterministic_metrics.json"
EFFICIENCY_PATH = RUN_DIR / "efficiency.json"
COMPARISON_PATH = RUN_DIR / "comparison_mem2d_dense_vs_mem3a_flat.json"
QUESTION_IDS = tuple(mem2c.QUESTION_IDS)
PINNED_BASE_COMMIT = "14d8c3ff9bf127d10ecb25cf5bdd30d949d7a5f4"
PINNED_DATASET_SHA256 = mem2c.PINNED_DATASET_SHA256
PINNED_MEM2C_INVENTORY_SHA256 = "93414251555c003e3acaa51968f5cf07a85ba3a189743a66301dd20ed7e09c26"
PINNED_READER_SHA256 = "57d3df897a1cf20a6ab0277e4dca3b6ad58cc0348e2057aacfffb1a8184535e3"
PINNED_PROJECTION_SHA256 = "ee16902373d695307db42797f33a1f4a531484096b36c29639b98abfd58d52ed"
PINNED_WRITER_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
WRITER_MAX_OUTPUT = 4096
READER_MAX_OUTPUT = 256
MAX_CONTEXT_TOKENS = 131072
MEMORY_BUDGET = 1024
TOP_K = 8
READER_ENDPOINT = "http://127.0.0.1:8081/v1"
READER_MODEL = "health-memory-qwen3-8b"
WRITER_ROLE = "memory_write_extract"
KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
FORBIDDEN_WRITER_KEYS = {
    "question_id",
    "question",
    "question_date",
    "answer",
    "answer_session_ids",
    "question_type",
    "has_answer",
    "retrieval_results",
    "previous_prediction_correctness",
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["propositions"],
    "additionalProperties": False,
    "properties": {
        "propositions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "proposition_text",
                    "entity_key_candidate",
                    "attribute_key_candidate",
                    "value_text",
                    "source_role",
                    "source_turn_indices",
                    "evidence_quotes",
                ],
                "additionalProperties": False,
                "properties": {
                    "proposition_text": {"type": "string", "minLength": 1},
                    "entity_key_candidate": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
                    },
                    "attribute_key_candidate": {
                        "type": "string",
                        "pattern": "^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
                    },
                    "value_text": {"type": "string", "minLength": 1},
                    "source_role": {"type": "string", "enum": ["user", "assistant", "mixed"]},
                    "source_turn_indices": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "integer", "minimum": 0},
                    },
                    "evidence_quotes": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        }
    },
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _sidecar_candidates(path: Path) -> list[Path]:
    return [_sidecar(path), path.with_name(f"{path.name}.sha256")]


def _existing_sidecars(path: Path) -> list[Path]:
    return [candidate for candidate in _sidecar_candidates(path) if candidate.is_file()]


def _freeze(path: Path) -> str:
    digest = sha256_file(path)
    _sidecar(path).write_text(f"{digest}  {path.name}\n", encoding="ascii", newline="\n")
    return digest


def _verify_frozen(path: Path) -> bool:
    if not path.is_file():
        return False
    markers = _existing_sidecars(path)
    return bool(markers) and all(
        marker.read_text(encoding="ascii").strip().split() == [sha256_file(path), path.name]
        for marker in markers
    )


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _write_json(path: Path, value: Any, *, freeze: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temp, path)
    return _freeze(path) if freeze else sha256_file(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]], *, freeze: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(_jsonl_bytes(rows))
    os.replace(temp, path)
    return _freeze(path) if freeze else sha256_file(path)


def _write_or_verify(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if not _verify_frozen(path) or path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Frozen MEM-3A artifact differs: {path.name}")
        return sha256_file(path)
    return _write_json(path, value)


def _write_or_verify_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = _jsonl_bytes(rows)
    if path.exists():
        if not _verify_frozen(path) or path.read_bytes() != payload:
            raise RuntimeError(f"Frozen MEM-3A artifact differs: {path.name}")
        return sha256_file(path)
    return _write_jsonl(path, rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_input(path: Path, expected_sha: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise RuntimeError(f"Frozen upstream input hash mismatch: {path}")
    if not _verify_frozen(path):
        raise RuntimeError(f"Frozen upstream input sidecar invalid: {path.name}")


def _contract_sha_freeze() -> dict[str, str]:
    paths = (PROMPT_PATH, EXTRACTOR_CONTRACT_PATH, RETRIEVAL_CONTRACT_PATH)
    result = {}
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"MEM-3A contract file is missing: {path}")
        markers = _existing_sidecars(path)
        if markers and not _verify_frozen(path):
            raise RuntimeError(f"MEM-3A contract sidecar exists but does not match: {path.name}")
        result[path.relative_to(ROOT).as_posix()] = sha256_file(path) if markers else _freeze(path)
    contract = _read_json(EXTRACTOR_CONTRACT_PATH)
    if (
        contract.get("contract_id") != "flat-proposition-extractor-v1"
        or contract.get("output_schema") != OUTPUT_SCHEMA
    ):
        raise RuntimeError(
            "Frozen extractor contract does not match the runner's strict output schema"
        )
    retrieval = _read_json(RETRIEVAL_CONTRACT_PATH)
    if (
        retrieval.get("contract_id") != "flat_proposition_retrieval_view_v1"
        or retrieval.get("document_text") != "exact proposition_text only"
    ):
        raise RuntimeError("Frozen proposition retrieval view contract changed")
    return result


def _verify_upstream() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != PINNED_BASE_COMMIT:
        raise RuntimeError(f"MEM-3A must start on pinned base {PINNED_BASE_COMMIT}; found {head}")
    _verify_input(MEM2C_INVENTORY, PINNED_MEM2C_INVENTORY_SHA256)
    if sha256_file(DATASET_PATH) != PINNED_DATASET_SHA256:
        raise RuntimeError("Pinned LongMemEval-S dataset bytes changed")
    m2d = _read_json(MEM2D_MANIFEST)
    if (
        not _verify_frozen(MEM2D_MANIFEST)
        or m2d.get("status") != "COMPLETE"
        or m2d.get("completion_gate_marker") != "MEM2D_SEMANTIC_RETRIEVAL_FROZEN_10_DIAGNOSTIC=YES"
        or m2d.get("gate", {}).get("passed") is not True
        or m2d.get("question_ids") != list(QUESTION_IDS)
        or m2d.get("test_access") is not False
        or m2d.get("102_dev_run") is not False
    ):
        raise RuntimeError("Required MEM-2D frozen-ten diagnostic gate failed")
    if (
        m2d.get("embedding_model_revision") != MODEL_REVISION
        or m2d.get("embedding_model_tree_sha256") != MODEL_TREE_SHA256
        or m2d.get("embedding_weights_sha256") != WEIGHTS_SHA256
    ):
        raise RuntimeError("MEM-2D embedding identity differs from the frozen local adapter")
    mem2d._verify_upstream()
    reader_contract, reader_sha, _, _ = load_final_reader_contract()
    if reader_sha != PINNED_READER_SHA256:
        raise RuntimeError("Frozen final reader contract SHA changed")
    if sha256_file(PROJECTION_CONTRACT_PATH) != PINNED_PROJECTION_SHA256:
        raise RuntimeError("Frozen rank-aware projection contract SHA changed")
    if not _verify_frozen(PROJECTION_CONTRACT_PATH):
        raise RuntimeError("Frozen projection contract sidecar failed")
    return {
        "mem2d_manifest_sha256": sha256_file(MEM2D_MANIFEST),
        "mem2c_inventory_sha256": PINNED_MEM2C_INVENTORY_SHA256,
        "dataset_sha256": PINNED_DATASET_SHA256,
        "reader_contract_sha256": reader_sha,
        "projection_contract_sha256": PINNED_PROJECTION_SHA256,
        "reader_contract": reader_contract,
    }


def _source_sessions() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    """Rebuild exact turns from the frozen RawSpan partition without reading question labels."""
    scoped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(MEM2C_INVENTORY):
        qid = row.get("question_id")
        if qid not in QUESTION_IDS or row.get("scope_id") != f"longmemeval:{qid}":
            raise RuntimeError("Frozen RawSpan inventory escaped the ten DEV scopes")
        value = row.get("value")
        if not isinstance(value, dict) or set(value) != {
            "role",
            "session_date",
            "content",
            "source_turn_index",
            "source_span_index",
        }:
            raise RuntimeError("Frozen RawSpan value schema changed")
        sid = row.get("source_session_id")
        if not isinstance(sid, str) or not sid:
            raise RuntimeError("RawSpan source session identity is missing")
        session = scoped.setdefault(
            (qid, sid),
            {
                "session_date": value["session_date"],
                "valid_from": row["valid_from"],
                "turns": defaultdict(list),
                "first_seen": len(scoped),
            },
        )
        if (
            session["session_date"] != value["session_date"]
            or session["valid_from"] != row["valid_from"]
        ):
            raise RuntimeError("Source session has inconsistent date anchors")
        session["turns"][int(value["source_turn_index"])].append(
            (
                int(row["char_start"]),
                int(row["char_end"]),
                int(value["source_span_index"]),
                value["role"],
                value["content"],
            )
        )
    unique: dict[str, dict[str, Any]] = {}
    refs: dict[str, dict[str, str]] = {qid: {} for qid in QUESTION_IDS}
    for (qid, sid), raw in scoped.items():
        turns = []
        for turn_index, parts in sorted(raw["turns"].items()):
            parts.sort(key=lambda item: (item[0], item[2]))
            if [item[2] for item in parts] != list(range(len(parts))):
                raise RuntimeError(f"RawSpan indices are not contiguous: {sid}/{turn_index}")
            cursor = 0
            roles = set()
            fragments = []
            for start, end, _, role, content in parts:
                if (
                    start != cursor
                    or end - start != len(content)
                    or role not in {"user", "assistant"}
                ):
                    raise RuntimeError(
                        f"RawSpan source partition is not lossless: {sid}/{turn_index}"
                    )
                cursor = end
                roles.add(role)
                fragments.append(content)
            if len(roles) != 1:
                raise RuntimeError("RawSpan pieces of one source turn disagree on role")
            turns.append(
                {"turn_index": turn_index, "role": next(iter(roles)), "content": "".join(fragments)}
            )
        if not turns:
            raise RuntimeError(f"Source session has no reconstructible turns: {sid}")
        identity = _session_identity_sha256(raw["session_date"], turns)
        turns_sha = _sha_json(turns)
        session = unique.setdefault(
            identity,
            {
                "session_identity_sha256": identity,
                "session_date": raw["session_date"],
                "valid_from": raw["valid_from"],
                "turns": turns,
                "source_turns_sha256": turns_sha,
                "source_session_ids": [],
                "_refs": [],
            },
        )
        if (
            session["session_date"] != raw["session_date"]
            or session["source_turns_sha256"] != turns_sha
        ):
            raise RuntimeError("SHA collision in source session identity")
        if sid not in session["source_session_ids"]:
            session["source_session_ids"].append(sid)
        session["_refs"].append({"question_id": qid, "source_session_id": sid})
        refs[qid][identity] = sid
    if len(scoped) != 477 or len(unique) != 477:
        raise RuntimeError(
            f"Frozen MEM-3A source session count changed: scoped={len(scoped)}, unique={len(unique)}"
        )
    ordered = dict(
        sorted(
            unique.items(),
            key=lambda item: min(
                (QUESTION_IDS.index(ref["question_id"]), ref["source_session_id"])
                for ref in item[1]["_refs"]
            ),
        )
    )
    return ordered, refs


def _writer_messages(session: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    # Only the single session's date and turns enter this function; IDs/questions/labels do not.
    payload = {"session_date": session["session_date"], "ordered_turns": session["turns"]}
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        },
    ]


def _session_identity_sha256(session_date: str, turns: list[dict[str, Any]]) -> str:
    identity_body = {
        "session_date": session_date,
        "turns": [
            {
                "turn_index": turn["turn_index"],
                "role": turn["role"],
                "content": turn["content"],
            }
            for turn in turns
        ],
    }
    return _sha_json(identity_body)


def _proposition_retrieval_document(row: dict[str, Any]) -> str:
    text = row.get("proposition_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("retrieval_document_must_be_nonempty_proposition_text")
    return text


def _writer_request(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "model": READER_MODEL,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": WRITER_MAX_OUTPUT,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "flat_proposition_packet",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            },
        },
    }


def _validate_packet(raw_text: str, session: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise ValueError("malformed_json") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"propositions"}
        or not isinstance(payload["propositions"], list)
    ):
        raise ValueError("schema_root_invalid")
    turns = {int(turn["turn_index"]): turn for turn in session["turns"]}
    output = []
    required = {
        "proposition_text",
        "entity_key_candidate",
        "attribute_key_candidate",
        "value_text",
        "source_role",
        "source_turn_indices",
        "evidence_quotes",
    }
    for index, proposition in enumerate(payload["propositions"]):
        if not isinstance(proposition, dict) or set(proposition) != required:
            raise ValueError(f"schema_proposition_invalid:{index}")
        for field in (
            "proposition_text",
            "entity_key_candidate",
            "attribute_key_candidate",
            "value_text",
        ):
            if not isinstance(proposition[field], str) or not proposition[field].strip():
                raise ValueError(f"schema_{field}_invalid:{index}")
        if not KEY_PATTERN.fullmatch(
            proposition["entity_key_candidate"]
        ) or not KEY_PATTERN.fullmatch(proposition["attribute_key_candidate"]):
            raise ValueError(f"candidate_key_not_lower_snake_case:{index}")
        source_indices = proposition["source_turn_indices"]
        quotes = proposition["evidence_quotes"]
        if (
            not isinstance(source_indices, list)
            or not source_indices
            or any(type(value) is not int or value not in turns for value in source_indices)
            or len(set(source_indices)) != len(source_indices)
        ):
            raise ValueError(f"source_turn_indices_invalid:{index}")
        if (
            not isinstance(quotes, list)
            or not quotes
            or any(not isinstance(quote, str) or not quote for quote in quotes)
        ):
            raise ValueError(f"evidence_quotes_invalid:{index}")
        declared_roles = {turns[turn_index]["role"] for turn_index in source_indices}
        expected_role = next(iter(declared_roles)) if len(declared_roles) == 1 else "mixed"
        if proposition["source_role"] != expected_role:
            raise ValueError(f"source_role_mismatch:{index}")
        for quote in quotes:
            if not any(quote in turns[turn_index]["content"] for turn_index in source_indices):
                raise ValueError(f"evidence_quote_not_exact_source_substring:{index}")
        output.append(dict(proposition))
    return output


def _session_cache_identity(
    session: dict[str, Any], contract_sha: str, prompt_sha: str, request_sha: str
) -> dict[str, Any]:
    body = {
        "stage": RUN_ID,
        "session_identity_sha256": session["session_identity_sha256"],
        "source_turns_sha256": session["source_turns_sha256"],
        "extractor_contract_sha256": contract_sha,
        "writer_model_sha256": PINNED_WRITER_MODEL_SHA256,
        "prompt_sha256": prompt_sha,
        "request_sha256": request_sha,
        "writer_role": WRITER_ROLE,
    }
    if FORBIDDEN_WRITER_KEYS.intersection(body):
        raise AssertionError("Question/label leaked into writer cache identity")
    return {"identity": body, "identity_sha256": _sha_json(body)}


def _preflight(*, save: bool) -> dict[str, Any]:
    upstream = _verify_upstream()
    contract_shas = _contract_sha_freeze()
    sessions, _ = _source_sessions()
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    if not system_prompt:
        raise RuntimeError("Frozen proposition writer prompt is empty")
    m2a_manifest = _read_json(mem2c.MEM2A_DIR / "run_manifest.json")
    rows = []
    saved = (
        _read_json(PREFLIGHT_PATH)
        if PREFLIGHT_PATH.exists() and _verify_frozen(PREFLIGHT_PATH)
        else None
    )
    saved_by_identity = (
        {row["session_identity_sha256"]: row for row in saved.get("sessions", [])} if saved else {}
    )
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        runtime = mem2b._verify_reader(client, m2a_manifest)
        if (
            not runtime.get("loopback_only")
            or runtime.get("endpoint") != READER_ENDPOINT
            or runtime.get("server_build") != "llama.cpp 10068 (571d0d540)"
            or runtime.get("model_sha256") != PINNED_WRITER_MODEL_SHA256
            or runtime.get("context_tokens") != MAX_CONTEXT_TOKENS
        ):
            raise RuntimeError("Frozen local Qwen writer runtime identity failed")
        for session in sessions.values():
            messages = _writer_messages(session, system_prompt)
            request = _writer_request(messages)
            prompt_sha = _sha_bytes(mem2a.canonical_json(messages).encode("utf-8"))
            request_sha = _sha_bytes(mem2a.canonical_json(request).encode("utf-8"))
            if not save:
                previous = saved_by_identity.get(session["session_identity_sha256"])
                if (
                    previous is None
                    or previous["source_turns_sha256"] != session["source_turns_sha256"]
                    or previous["prompt_sha256"] != prompt_sha
                    or previous["request_sha256"] != request_sha
                ):
                    raise RuntimeError(
                        f"Frozen writer preflight differs from current source/prompt: {session['session_identity_sha256']}"
                    )
                prompt_tokens = previous["prompt_tokens"]
                rendered_sha = previous["rendered_prompt_sha256"]
                preflight_ms = previous["tokenizer_preflight_latency_ms"]
            else:
                preflight_started = time.perf_counter()
                prompt_tokens, rendered_sha = reader_runtime._render_and_tokenize(client, messages)
                preflight_ms = round((time.perf_counter() - preflight_started) * 1000, 3)
            if prompt_tokens + WRITER_MAX_OUTPUT > MAX_CONTEXT_TOKENS:
                raise RuntimeError(
                    f"Writer prompt exceeds frozen 131072 context bound: {session['session_identity_sha256']}"
                )
            cache_identity = _session_cache_identity(
                session,
                contract_shas[EXTRACTOR_CONTRACT_PATH.relative_to(ROOT).as_posix()],
                prompt_sha,
                request_sha,
            )
            rows.append(
                {
                    "session_identity_sha256": session["session_identity_sha256"],
                    "source_session_ids": session["source_session_ids"],
                    "session_date": session["session_date"],
                    "turn_count": len(session["turns"]),
                    "source_turns_sha256": session["source_turns_sha256"],
                    "prompt_sha256": prompt_sha,
                    "rendered_prompt_sha256": rendered_sha,
                    "request_sha256": request_sha,
                    "cache_identity": cache_identity,
                    "prompt_tokens": prompt_tokens,
                    "output_reserve": WRITER_MAX_OUTPUT,
                    "max_context_tokens": MAX_CONTEXT_TOKENS,
                    "truncated": False,
                    "tokenizer_preflight_latency_ms": preflight_ms,
                }
            )
    expected_order = list(sessions)
    if [row["session_identity_sha256"] for row in rows] != expected_order or len(rows) != 477:
        raise RuntimeError(
            "Writer preflight did not cover every unique source session in stable order"
        )
    preflight = {
        "schema_version": 1,
        "stage": RUN_ID,
        "base_commit_sha": PINNED_BASE_COMMIT,
        "scope": "frozen ten DEV cases only; no labels used",
        "question_ids": list(QUESTION_IDS),
        "unique_source_sessions": len(rows),
        "extractor_contract_sha256": contract_shas[
            EXTRACTOR_CONTRACT_PATH.relative_to(ROOT).as_posix()
        ],
        "extractor_prompt_sha256": contract_shas[PROMPT_PATH.relative_to(ROOT).as_posix()],
        "retrieval_view_contract_sha256": contract_shas[
            RETRIEVAL_CONTRACT_PATH.relative_to(ROOT).as_posix()
        ],
        "writer_role": WRITER_ROLE,
        "writer_model_sha256": PINNED_WRITER_MODEL_SHA256,
        "writer_runtime_identity": runtime,
        "local_only": True,
        "hosted_calls": 0,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "output_reserve": WRITER_MAX_OUTPUT,
        "all_prompt_tokens_plus_reserve_fit": all(
            row["prompt_tokens"] + WRITER_MAX_OUTPUT <= MAX_CONTEXT_TOKENS
            and row["truncated"] is False
            for row in rows
        ),
        "sessions": rows,
        "upstream": upstream,
    }
    if not preflight["all_prompt_tokens_plus_reserve_fit"]:
        raise RuntimeError("At least one writer prompt failed full-history preflight")
    if save:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        _write_or_verify(PREFLIGHT_PATH, preflight)
    elif saved != preflight:
        raise RuntimeError("Frozen writer preflight identity or runtime changed")
    return preflight


def _append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _canonical(row) + b"\n"
    with path.open("ab") as target:
        target.write(line)
        target.flush()
        os.fsync(target.fileno())


def _load_writer_journal(path: Path) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return events
    with path.open("rb") as source:
        for line_no, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Writer journal is truncated/corrupt at line {line_no}; no retry is safe"
                ) from error
            identity = row.get("cache_identity_sha256")
            if not isinstance(identity, str):
                raise TypeError("Writer journal entry has no cache identity")
            events[identity].append(row)
    return events


def _build_packet(
    session: dict[str, Any], propositions: list[dict[str, Any]], contract_sha: str
) -> dict[str, Any]:
    return {
        "session_identity_sha256": session["session_identity_sha256"],
        "source_session_ids": session["source_session_ids"],
        "session_date": session["session_date"],
        "valid_from": session["valid_from"],
        "source_turns_sha256": session["source_turns_sha256"],
        "extractor_contract_sha256": contract_sha,
        "writer_model_sha256": PINNED_WRITER_MODEL_SHA256,
        "propositions": propositions,
    }


def _extract_all(preflight: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sessions, _ = _source_sessions()
    contract_sha = preflight["extractor_contract_sha256"]
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    work_root = (
        ROOT.parent
        / ".cache"
        / "health-copilot"
        / "mem3a-writer"
        / preflight["writer_runtime_identity"].get("runtime_props_sha256", "runtime")
        / contract_sha
    )
    work_root.mkdir(parents=True, exist_ok=True)
    journal = work_root / "writer_journal.jsonl"
    events = _load_writer_journal(journal)
    rows_by_identity = {row["session_identity_sha256"]: row for row in preflight["sessions"]}
    session_packets: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        for identity, session in sessions.items():
            preflight_row = rows_by_identity[identity]
            messages = _writer_messages(session, system_prompt)
            request = _writer_request(messages)
            actual_prompt_sha = _sha_bytes(mem2a.canonical_json(messages).encode("utf-8"))
            actual_request_sha = _sha_bytes(mem2a.canonical_json(request).encode("utf-8"))
            if (
                actual_prompt_sha != prompt_sha_for_session(preflight_row)
                or actual_request_sha != preflight_row["request_sha256"]
            ):
                raise RuntimeError(
                    f"Writer prompt/request identity changed after preflight: {identity}"
                )
            cache_identity = preflight_row["cache_identity"]
            cache_sha = cache_identity["identity_sha256"]
            prior_events = events.get(cache_sha, [])
            if prior_events:
                if (
                    len(prior_events) != 2
                    or prior_events[0].get("event") != "STARTED"
                    or prior_events[1].get("event") != "COMPLETE"
                ):
                    raise RuntimeError(
                        f"Writer call has a started/incomplete prior attempt; refusing retry: {identity}"
                    )
                done = prior_events[1]
                if (
                    done.get("request_sha256") != actual_request_sha
                    or done.get("status") != "SUCCESS"
                ):
                    raise RuntimeError(
                        f"Prior writer attempt failed or changed; no retry permitted: {identity}"
                    )
                propositions = _validate_packet(done["raw_content"], session)
                packet = _build_packet(session, propositions, contract_sha)
                if done.get("packet_sha256") != _sha_json(packet):
                    raise RuntimeError(f"Cached writer packet digest mismatch: {identity}")
                session_packets.append(packet)
                ledger.append(done["call"])
                continue

            if not urlsplit(READER_ENDPOINT).hostname == "127.0.0.1":
                raise RuntimeError("Writer endpoint is not loopback")
            started_utc = datetime.now(UTC).isoformat()
            _append_journal(
                journal,
                {
                    "event": "STARTED",
                    "cache_identity_sha256": cache_sha,
                    "request_sha256": actual_request_sha,
                    "started_utc": started_utc,
                },
            )
            started = time.perf_counter()
            response = None
            payload: dict[str, Any] = {}
            raw_content: str | None = None
            error_type = None
            try:
                response = client.post(f"{READER_ENDPOINT}/chat/completions", json=request)
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices")
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                ):
                    raise RuntimeError("unexpected_choice_shape")
                raw_content = choices[0].get("message", {}).get("content")
                if not isinstance(raw_content, str):
                    raise TypeError("missing_response_content")
                if choices[0].get("finish_reason") == "length":
                    raise RuntimeError("finish_reason_length")
                propositions = _validate_packet(raw_content, session)
                packet = _build_packet(session, propositions, contract_sha)
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                error_type = (
                    str(error)
                    if str(error)
                    in {
                        "finish_reason_length",
                        "unexpected_choice_shape",
                        "missing_response_content",
                    }
                    else type(error).__name__
                )
                latency_ms = round((time.perf_counter() - started) * 1000, 3)
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                choice = (
                    payload.get("choices", [{}])[0]
                    if isinstance(payload.get("choices"), list) and payload.get("choices")
                    else {}
                )
                failure_call = {
                    "role": WRITER_ROLE,
                    "provider": "local_qwen",
                    "endpoint": READER_ENDPOINT,
                    "loopback_only": True,
                    "model": READER_MODEL,
                    "session_identity_sha256": identity,
                    "cache_identity_sha256": cache_sha,
                    "prompt_sha256": actual_prompt_sha,
                    "request_sha256": actual_request_sha,
                    "prompt_tokens": preflight_row["prompt_tokens"],
                    "completion_tokens": usage.get("completion_tokens"),
                    "latency_ms": latency_ms,
                    "finish_reason": choice.get("finish_reason"),
                    "success": False,
                    "failure_status": "EXTRACTION_FAILURE",
                    "error_type": error_type,
                    "retry_count": 0,
                    "hosted_call": False,
                }
                _append_journal(
                    journal,
                    {
                        "event": "COMPLETE",
                        "cache_identity_sha256": cache_sha,
                        "request_sha256": actual_request_sha,
                        "status": "FAILURE",
                        "call": failure_call,
                        "error_type": error_type,
                        "raw_content_sha256": _sha_bytes(raw_content.encode("utf-8"))
                        if raw_content is not None
                        else None,
                    },
                )
                raise RuntimeError(
                    f"EXTRACTION_FAILURE for source session {identity}: {error_type}; no retry issued"
                ) from error
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            choice = payload["choices"][0]
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            call = {
                "role": WRITER_ROLE,
                "provider": "local_qwen",
                "endpoint": READER_ENDPOINT,
                "loopback_only": True,
                "model": READER_MODEL,
                "session_identity_sha256": identity,
                "cache_identity_sha256": cache_sha,
                "prompt_sha256": actual_prompt_sha,
                "request_sha256": actual_request_sha,
                "prompt_tokens": preflight_row["prompt_tokens"],
                "completion_tokens": usage.get("completion_tokens"),
                "latency_ms": latency_ms,
                "finish_reason": choice.get("finish_reason"),
                "success": True,
                "failure_status": None,
                "error_type": None,
                "retry_count": 0,
                "hosted_call": False,
            }
            _append_journal(
                journal,
                {
                    "event": "COMPLETE",
                    "cache_identity_sha256": cache_sha,
                    "request_sha256": actual_request_sha,
                    "status": "SUCCESS",
                    "raw_content": raw_content,
                    "packet_sha256": _sha_json(packet),
                    "call": call,
                },
            )
            session_packets.append(packet)
            ledger.append(call)
            print(f"writer_sessions={len(session_packets)}/{len(sessions)}", flush=True)
    if (
        len(session_packets) != len(sessions)
        or len(ledger) != len(sessions)
        or any(
            row.get("hosted_call") is not False or row.get("success") is not True for row in ledger
        )
    ):
        raise RuntimeError(
            "Writer extraction did not complete exactly once for every unique source session"
        )
    return session_packets, ledger


def prompt_sha_for_session(preflight_row: dict[str, Any]) -> str:
    return preflight_row["prompt_sha256"]


def _materialize(
    session_packets: list[dict[str, Any]], refs: dict[str, dict[str, str]], contract_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    packet_by_identity = {packet["session_identity_sha256"]: packet for packet in session_packets}
    flat_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    records_by_question: dict[str, dict[str, dict[str, Any]]] = {qid: {} for qid in QUESTION_IDS}
    for qid in QUESTION_IDS:
        scope_id = f"longmemeval:{qid}"
        store = InMemoryMemoryStore()
        for identity, source_sid in refs[qid].items():
            packet = packet_by_identity[identity]
            for proposition_index, proposition in enumerate(packet["propositions"]):
                provenance = {
                    "source_turn_indices": proposition["source_turn_indices"],
                    "evidence_quotes": proposition["evidence_quotes"],
                }
                provenance_sha = _sha_json(provenance)
                proposition_sha = _sha_bytes(proposition["proposition_text"].encode("utf-8"))
                id_body = {
                    "extractor_contract_sha256": contract_sha,
                    "session_identity_sha256": identity,
                    "proposition_index": proposition_index,
                    "proposition_text_sha256": proposition_sha,
                    "provenance_sha256": provenance_sha,
                }
                memory_digest = _sha_json(id_body)
                memory_id = f"m10flat-{memory_digest}"
                key = f"flat_proposition:{memory_digest}"
                value = {
                    "proposition": proposition["proposition_text"],
                    "observed_at": packet["valid_from"],
                    "source_role": proposition["source_role"],
                }
                op = MemoryOperation.add(
                    scope_id=scope_id,
                    key=key,
                    kind=MemoryKind.SESSION_NOTE,
                    value=value,
                    source_type=MemorySourceType.SESSION_DERIVED,
                    sensitivity=MemorySensitivity.NON_SENSITIVE,
                    memory_id=memory_id,
                    valid_from=packet["valid_from"],
                    valid_until=None,
                    expires_at=None,
                    supersedes_id=None,
                    source_session_id=source_sid,
                    source_event_ids=(f"longmemeval-flat-proposition-{memory_digest}",),
                )
                if op.operation != MemoryOperationType.ADD:
                    raise RuntimeError("Non-ADD MemoryOperation is forbidden in MEM-3A")
                record = store.apply(op, now=packet["valid_from"])
                if (
                    record is None
                    or record.status != MemoryStatus.ACTIVE
                    or record.version != 1
                    or record.supersedes_id is not None
                ):
                    raise RuntimeError(
                        "Flat proposition materialized outside ACTIVE version-1 invariant"
                    )
                row = {
                    "question_id": qid,
                    "scope_id": scope_id,
                    "memory_id": memory_id,
                    "key": key,
                    "session_identity_sha256": identity,
                    "source_session_id": source_sid,
                    "session_date": packet["session_date"],
                    "valid_from": packet["valid_from"],
                    "proposition_index": proposition_index,
                    "proposition_text": proposition["proposition_text"],
                    "proposition_text_sha256": proposition_sha,
                    "entity_key_candidate": proposition["entity_key_candidate"],
                    "attribute_key_candidate": proposition["attribute_key_candidate"],
                    "value_text": proposition["value_text"],
                    "source_role": proposition["source_role"],
                    "source_turn_indices": proposition["source_turn_indices"],
                    "evidence_quotes": proposition["evidence_quotes"],
                    "provenance_sha256": provenance_sha,
                    "operation": "ADD",
                    "kind": "session_note",
                    "source_type": "session_derived",
                    "status": "active",
                    "version": 1,
                    "valid_until": None,
                    "expires_at": None,
                    "supersedes_id": None,
                    "reader_value": value,
                    "retrieval_document_sha256": proposition_sha,
                }
                flat_rows.append(row)
                operation_rows.append(
                    {
                        "question_id": qid,
                        "operation": "ADD",
                        "memory_id": memory_id,
                        "scope_id": scope_id,
                        "key": key,
                        "kind": record.kind.value,
                        "source_type": record.source_type.value,
                        "status": record.status.value,
                        "version": record.version,
                        "valid_from": record.valid_from,
                        "valid_until": record.valid_until,
                        "expires_at": record.expires_at,
                        "supersedes_id": record.supersedes_id,
                        "source_session_id": record.source_session_id,
                        "value_sha256": record.value_sha256,
                    }
                )
                records_by_question[qid][memory_id] = {"record": record, "row": row}
        if len(store.history(scope_id)) != len(records_by_question[qid]) or any(
            event.operation != MemoryOperationType.ADD for event in store.history(scope_id)
        ):
            raise RuntimeError(f"M10 store history contains a non-ADD operation: {qid}")
        store.close()
    flat_rows.sort(
        key=lambda row: (
            QUESTION_IDS.index(row["question_id"]),
            row["session_identity_sha256"],
            row["proposition_index"],
        )
    )
    operation_rows.sort(key=lambda row: (QUESTION_IDS.index(row["question_id"]), row["memory_id"]))
    return flat_rows, operation_rows, records_by_question


def _candidate_groups(flat_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in flat_rows:
        groups[(row["entity_key_candidate"], row["attribute_key_candidate"])].append(row)
    group_rows = []
    for (entity, attribute), rows in sorted(groups.items()):
        values = sorted({row["value_text"] for row in rows})
        group_rows.append(
            {
                "entity_key_candidate": entity,
                "attribute_key_candidate": attribute,
                "proposition_count": len(rows),
                "distinct_value_text_count": len(values),
                "candidate_revision_group": len(values) > 1,
                "values": values,
                "observed_timestamps_ordered": sorted({row["valid_from"] for row in rows}),
                "source_sessions": sorted({row["source_session_id"] for row in rows}),
                "question_scopes": sorted(
                    {row["question_id"] for row in rows}, key=QUESTION_IDS.index
                ),
            }
        )
    pair_counts = Counter(
        (row["entity_key_candidate"], row["attribute_key_candidate"]) for row in flat_rows
    )
    reused_props = sum(count for count in pair_counts.values() if count > 1)
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "revision_semantics": False,
        "candidate_key_pairs": len(pair_counts),
        "proposition_count": len(flat_rows),
        "exact_key_pair_reuse_rate": reused_props / len(flat_rows) if flat_rows else 0.0,
        "same_key_multiple_value_group_count": sum(
            group["candidate_revision_group"] for group in group_rows
        ),
        "groups": group_rows,
        "surface_attribute_similarity_audit": _surface_key_audit(flat_rows),
        "interpretation_boundary": "Key/value groupings are unnormalized extractor diagnostics; multiple values are not labeled contradiction, stale, current, or superseded.",
    }


def _surface_key_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stop = {
        "the",
        "user",
        "is",
        "has",
        "have",
        "had",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "with",
        "their",
        "they",
        "i",
        "my",
        "now",
        "currently",
    }
    signatures = []
    for row in rows:
        tokens = set(re.findall(r"[a-z]+", row["proposition_text"].casefold())) - stop
        tokens = {token for token in tokens if not token.isdigit()}
        signatures.append((row, tokens))
    examples = []
    by_entity: dict[str, list[tuple[dict[str, Any], set[str]]]] = defaultdict(list)
    for row, tokens in signatures:
        by_entity[row["entity_key_candidate"]].append((row, tokens))
    seen = set()
    for entity, candidates in sorted(by_entity.items()):
        for i, (left, left_tokens) in enumerate(candidates):
            if not left_tokens:
                continue
            for right, right_tokens in candidates[i + 1 :]:
                if (
                    left["attribute_key_candidate"] == right["attribute_key_candidate"]
                    or not right_tokens
                ):
                    continue
                union = left_tokens | right_tokens
                similarity = len(left_tokens & right_tokens) / len(union) if union else 0.0
                if similarity < 0.6:
                    continue
                key = tuple(sorted((left["memory_id"], right["memory_id"])))
                if key in seen:
                    continue
                seen.add(key)
                examples.append(
                    {
                        "entity_key_candidate": entity,
                        "left_attribute_key_candidate": left["attribute_key_candidate"],
                        "left_proposition_text": left["proposition_text"],
                        "right_attribute_key_candidate": right["attribute_key_candidate"],
                        "right_proposition_text": right["proposition_text"],
                        "lexical_jaccard": similarity,
                        "heuristic": True,
                    }
                )
                if len(examples) >= 50:
                    break
            if len(examples) >= 50:
                break
        if len(examples) >= 50:
            break
    return {
        "method": "same-entity proposition-token Jaccard >= 0.6; manual review required",
        "heuristic": True,
        "different_key_pair_example_count": len(examples),
        "examples": examples,
    }


def _source_diagnostics(
    inventory_rows: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    scoped_ids = {(row["question_id"], row["source_session_id"]) for row in inventory_rows}
    raw_turns = {
        (row["question_id"], row["source_session_id"], row["source_turn_index"])
        for row in inventory_rows
    }
    return {
        "source_sessions": len(scoped_ids),
        "unique_session_content_identities": len(sessions),
        "raw_turns": len(raw_turns),
        "raw_spans": len(inventory_rows),
    }


def _materialize_sqlite(
    records_by_question: dict[str, dict[str, dict[str, Any]]],
    flat_rows: list[dict[str, Any]],
    cache_identity: str,
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    root = ROOT.parent / ".cache" / "health-copilot" / "mem3a-memory" / cache_identity
    root.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    matches_by_question: dict[str, list[dict[str, Any]]] = {}
    flat_by_id = {(row["question_id"], row["memory_id"]): row for row in flat_rows}
    questions = mem2d._load_frozen_inputs()["questions"]
    for qid in QUESTION_IDS:
        path = root / f"{qid}.sqlite"
        existed_before = path.exists()
        expected = {
            memory_id: item["record"].to_dict()
            for memory_id, item in records_by_question[qid].items()
        }
        store = SQLiteMemoryStore(path)
        existing = store.active_records(f"longmemeval:{qid}", now="9999-01-01T00:00:00Z")
        if existing:
            actual = {record.memory_id: record.to_dict() for record in existing}
            if actual != expected:
                store.close()
                raise RuntimeError(
                    f"Persistent M10-FlatProp SQLite cache differs from frozen inventory: {qid}"
                )
        else:
            if existed_before:
                store.close()
                raise RuntimeError(f"Unidentified partial SQLite database cannot be resumed: {qid}")
            for item in records_by_question[qid].values():
                record = item["record"]
                op = MemoryOperation.add(
                    scope_id=record.scope_id,
                    key=record.key,
                    kind=record.kind,
                    value=record.value,
                    source_type=record.source_type,
                    sensitivity=record.sensitivity,
                    memory_id=record.memory_id,
                    valid_from=record.valid_from,
                    valid_until=record.valid_until,
                    expires_at=record.expires_at,
                    source_event_ids=record.source_event_ids,
                    source_session_id=record.source_session_id,
                )
                store.apply(op, now=record.created_at)
            actual = {
                record.memory_id: record.to_dict()
                for record in store.active_records(f"longmemeval:{qid}", now="9999-01-01T00:00:00Z")
            }
            if actual != expected:
                store.close()
                raise RuntimeError(
                    f"M10 SQLite materialization did not reproduce the logical inventory: {qid}"
                )
        dense_rows = []
        started = time.perf_counter()
        question = questions[qid]
        matches = store.matches(
            MemoryQuery(
                scope_id=f"longmemeval:{qid}",
                text=question["question"],
                now=question["query_now"],
                top_k=len(expected),
            )
        )
        m10_filter_ms = round((time.perf_counter() - started) * 1000, 3)
        for match in matches:
            row = flat_by_id.get((qid, match.record.memory_id))
            if row is None:
                raise RuntimeError("M10 SQLite returned a memory outside this question scope")
            dense_rows.append({"record": match.record, "row": row})
        matches_by_question[qid] = dense_rows
        store.close()
        sizes[qid] = path.stat().st_size
        matches_by_question[qid + "__latency"] = m10_filter_ms  # type: ignore[assignment]
    return sizes, matches_by_question


def _embedding_and_retrieval(
    flat_rows: list[dict[str, Any]],
    records_by_question: dict[str, dict[str, dict[str, Any]]],
    refs: dict[str, dict[str, str]],
    preflight: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int], dict[str, list[dict[str, Any]]]]:
    import numpy as np

    model_path = Path(r"E:\Health-Copilot-Models\models\Qwen3-Embedding-0.6B")
    model_identity = verify_local_model(model_path)
    if (
        model_identity["revision"] != MODEL_REVISION
        or model_identity["model_tree_sha256"] != MODEL_TREE_SHA256
        or model_identity["weights_sha256"] != WEIGHTS_SHA256
    ):
        raise RuntimeError("Local Qwen3-Embedding-0.6B identity is not frozen")
    doc_text_by_sha: dict[str, str] = {}
    for row in flat_rows:
        text = _proposition_retrieval_document(row)
        digest = _sha_bytes(text.encode("utf-8"))
        if row.get("retrieval_document_sha256") != digest:
            raise RuntimeError(
                "Frozen proposition inventory does not carry the exact retrieval document SHA"
            )
        previous = doc_text_by_sha.setdefault(digest, text)
        if previous != text:
            raise RuntimeError("Proposition document SHA collision")
    ordered_doc_shas = sorted(doc_text_by_sha)
    unique_texts = [doc_text_by_sha[digest] for digest in ordered_doc_shas]
    frozen_questions = mem2d._load_frozen_inputs()["questions"]
    adapter = LocalQwen3Embedding(model_path)
    try:
        doc_token_counts = adapter.count_tokens(unique_texts, "document")
        if any(count > MAX_LENGTH for count in doc_token_counts):
            raise RuntimeError("At least one proposition embedding document would be truncated")
        cache_identity_body = {
            "stage": RUN_ID,
            "representation": "flat-proposition-v1",
            "flat_propositions_sha256": sha256_file(FLAT_PROPOSITIONS_PATH),
            "retrieval_view_contract_sha256": preflight["retrieval_view_contract_sha256"],
            "embedding_model_id": MODEL_ID,
            "embedding_revision": MODEL_REVISION,
            "embedding_tree_sha256": MODEL_TREE_SHA256,
            "embedding_weights_sha256": WEIGHTS_SHA256,
            "embedding_adapter_sha256": sha256_file(TOOLS_DIR / "local_qwen3_embedding.py"),
            "query_instruction": QUERY_INSTRUCTION,
            "document_instruction": "none",
            "dimensions": DIMENSIONS,
            "device": DEVICE,
            "dtype": DTYPE,
            "batch_size": BATCH_SIZE,
            "max_batch_tokens": MAX_BATCH_TOKENS,
            "max_length": MAX_LENGTH,
            "ordered_document_sha_root": _sha_bytes(
                ("\n".join(ordered_doc_shas) + "\n").encode("ascii")
            ),
        }
        cache_identity = {
            "identity": cache_identity_body,
            "identity_sha256": _sha_json(cache_identity_body),
        }
        cache_root = ROOT.parent / ".cache" / "health-copilot" / "mem3a-embedding"
        vectors, cache_stats = mem2d._embed_document_corpus(
            adapter=adapter,
            unique_doc_shas=ordered_doc_shas,
            unique_texts=unique_texts,
            token_counts=doc_token_counts,
            cache_identity=cache_identity,
            cache_root=cache_root,
        )
        query_texts = [frozen_questions[qid]["question"] for qid in QUESTION_IDS]
        query_tokens = adapter.count_tokens(query_texts, "query")
        if any(count > MAX_LENGTH for count in query_tokens):
            raise RuntimeError("At least one benchmark query embedding would be truncated")
        query_batches = list(
            adapter.encode_batches(query_texts, "query", token_counts=query_tokens)
        )
        if (
            len(query_batches) != 1
            or query_batches[0].vectors.shape != (len(QUESTION_IDS), DIMENSIONS)
            or query_batches[0].truncated_count
        ):
            raise RuntimeError("Frozen query embedding batch was incomplete or truncated")
        query_vectors = query_batches[0].vectors
        query_vector_by_qid = {qid: query_vectors[index] for index, qid in enumerate(QUESTION_IDS)}
        memory_index = {digest: index for index, digest in enumerate(ordered_doc_shas)}
        by_qid_flat: dict[str, dict[str, dict[str, Any]]] = {qid: {} for qid in QUESTION_IDS}
        for row in flat_rows:
            by_qid_flat[row["question_id"]][row["memory_id"]] = row
        all_ranked = []
        retrieval_latency: dict[str, float] = {}
        for qid in QUESTION_IDS:
            session = InMemoryMemoryStore()
            expected = records_by_question[qid]
            session._records = {mid: item["record"] for mid, item in expected.items()}
            question = frozen_questions[qid]
            started = time.perf_counter()
            eligible = session.matches(
                MemoryQuery(
                    scope_id=question["scope_id"],
                    text=question["question"],
                    now=question["query_now"],
                    top_k=len(expected),
                )
            )
            eligible_ids = {match.record.memory_id for match in eligible}
            if len(eligible_ids) != len(eligible):
                raise RuntimeError("M10 validity filter emitted duplicate proposition IDs")
            candidates = []
            for memory_id in eligible_ids:
                row = by_qid_flat[qid][memory_id]
                vector = vectors[memory_index[row["retrieval_document_sha256"]]]
                score = float(np.dot(query_vector_by_qid[qid], vector))
                candidates.append(
                    {
                        "question_id": qid,
                        "memory_id": memory_id,
                        "cosine_similarity": score,
                        "source_session_id": row["source_session_id"],
                        "source_turn_indices": row["source_turn_indices"],
                        "valid_from": row["valid_from"],
                        "retrieval_document_sha256": row["retrieval_document_sha256"],
                        "proposition_text_sha256": row["proposition_text_sha256"],
                    }
                )
            candidates.sort(key=lambda row: (-row["cosine_similarity"], row["memory_id"]))
            ranked = [dict(row, rank=rank) for rank, row in enumerate(candidates[:TOP_K], 1)]
            if len(ranked) != TOP_K:
                raise RuntimeError(f"Dense retrieval produced fewer than eight candidates: {qid}")
            retrieval_latency[qid] = round((time.perf_counter() - started) * 1000, 3)
            all_ranked.extend(ranked)
        embedding_manifest = {
            "schema_version": 1,
            "role": "embedding_model",
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "model_path": str(model_path),
            "tree_sha256": model_identity["model_tree_sha256"],
            "weights_sha256": model_identity["weights_sha256"],
            "dimensions": DIMENSIONS,
            "device": adapter.device,
            "device_name": adapter.device_name,
            "dtype": adapter.dtype,
            "pooling": "true final non-padding token under left padding",
            "normalization": "L2 normalization in float32",
            "batch_size": BATCH_SIZE,
            "max_batch_tokens": MAX_BATCH_TOKENS,
            "max_length": MAX_LENGTH,
            "query_instruction": QUERY_INSTRUCTION,
            "document_instruction": "none",
            "document_count": len(flat_rows),
            "unique_document_count": len(unique_texts),
            "document_input_tokens": sum(doc_token_counts),
            "query_input_tokens": sum(query_tokens),
            "proposition_token_median": statistics.median(doc_token_counts)
            if doc_token_counts
            else None,
            "proposition_token_p95": np_percentile(doc_token_counts, 0.95)
            if doc_token_counts
            else None,
            "document_token_counts_sha256": _sha_json(sorted(doc_token_counts)),
            "document_truncations": 0,
            "query_truncations": 0,
            "cache": cache_stats,
            "query_embedding_wall_ms": round(query_batches[0].latency_ms, 3),
            "retrieval_latency_ms": retrieval_latency,
            "local_only": True,
            "hosted_calls": 0,
        }
    finally:
        adapter.close()
    vectors._mmap.close()
    return (
        all_ranked,
        embedding_manifest,
        query_tokens_by_qid(query_tokens, QUESTION_IDS),
        retrieval_latency,
    )


def query_tokens_by_qid(values: list[int], qids: tuple[str, ...]) -> dict[str, int]:
    return dict(zip(qids, values, strict=True))


def _project_contexts(
    ranked: list[dict[str, Any]],
    records_by_question: dict[str, dict[str, dict[str, Any]]],
    questions: dict[str, dict[str, Any]],
    client: httpx.Client,
    reader_contract: tuple[dict[str, Any], str, str, str],
    retrieval_latency: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        grouped[row["question_id"]].append(row)
    plan_rows = []
    bundle_rows = []
    item_rows_by_qid = {}
    budget = mem2c._memory_budget()
    for qid in QUESTION_IDS:
        question = questions[qid]
        candidates = grouped[qid]
        records = [records_by_question[qid][row["memory_id"]]["record"] for row in candidates]
        rank_by_id = {f"memory-{row['memory_id']}": row["rank"] for row in candidates}
        manager = ContextManager(
            budget=budget, estimator=DeterministicTokenEstimator(), history_window=24
        )
        started = time.perf_counter()
        plan = manager.build_plan(
            session_id=question["scope_id"],
            session_revision=question["session_revision"],
            current_user=question["question"],
            history=(),
            memory_records=records,
            retrieval_query=question["question"],
            selection_rank_hints=rank_by_id,
        )
        projection_latency = round((time.perf_counter() - started) * 1000, 3)
        if plan.memory_tokens > MEMORY_BUDGET:
            raise RuntimeError(f"Frozen rank-aware projection exceeds 1024: {qid}")
        rank_by_memory = {row["memory_id"]: row["rank"] for row in candidates}
        items = [item for item in plan.items if item.category == ContextItemCategory.MEMORY]
        selected_ranks = [rank_by_memory[item.provenance["memory_id"]] for item in items]
        if selected_ranks != sorted(selected_ranks):
            raise RuntimeError(f"Rank-aware projection changed native dense order: {qid}")
        context_items = []
        content_only = []
        for display_rank, item in enumerate(items, 1):
            memory_id = str(item.provenance["memory_id"])
            source = records_by_question[qid][memory_id]["row"]
            native = rank_by_memory[memory_id]
            retrieval_row = candidates[native - 1]
            context_items.append(
                {
                    "text": mem2a.canonical_json(item.content),
                    "rank": display_rank,
                    "kind": item.category.value,
                    "memory_id": memory_id,
                    "native_retrieval_rank": native,
                    "retrieval_score": retrieval_row["cosine_similarity"],
                    "source_session_ids": [source["source_session_id"]],
                    "source_turn_indices": source["source_turn_indices"],
                    "status": source["status"],
                    "version": source["version"],
                    "estimated_tokens": item.estimated_tokens,
                    "proposition_text": source["proposition_text"],
                    "candidate_key_metadata_reader_visible": False,
                }
            )
            content_only.append(source["proposition_text"])
        serialized = "\n\n".join(
            f"[Context item {item['rank']} | {item['kind']}]\n{item['text']}"
            for item in context_items
        )
        context_tokens = mem2c._local_token_count(client, serialized) if serialized else 0
        content_only_tokens = (
            mem2c._local_token_count(client, "\n\n".join(content_only)) if content_only else 0
        )
        bundle_base = {
            "schema_version": 3,
            "system": "healthcopilot_m10_flatprop",
            "question_id": qid,
            "representation": "M10-FlatProp",
            "items": [
                {key: value for key, value in item.items() if key != "estimated_tokens"}
                for item in context_items
            ],
            "serialized_context": serialized,
            "context_embedding_tokens": None,
            "context_embedding_tokenizer": "NOT_APPLICABLE_CONTEXT_IS_READER_VIEW",
            "context_reader_tokens": context_tokens,
            "content_only_reader_tokens": content_only_tokens,
            "metadata_overhead_ratio": max(0, context_tokens - content_only_tokens)
            / max(1, context_tokens),
            "provenance_available": True,
            "ingestion_latency_ms": None,
            "retrieval_latency_ms": retrieval_latency[qid],
        }
        bundle_sha = _sha_json(bundle_base)
        bundle = {**bundle_base, "context_bundle_sha256": bundle_sha}
        _, contract_sha, system_template, user_template = reader_contract
        messages = build_reader_messages(
            question["question"],
            question["question_date"],
            serialized,
            system_template=system_template,
            user_template=user_template,
        )
        prompt_tokens, rendered_sha = reader_runtime._render_and_tokenize(client, messages)
        if prompt_tokens + READER_MAX_OUTPUT > MAX_CONTEXT_TOKENS:
            raise RuntimeError(f"Shared-reader prompt exceeds frozen context bound: {qid}")
        message_sha = _sha_bytes(mem2a.canonical_json(messages).encode("utf-8"))
        bundle_rows.append(
            {
                "question_id": qid,
                "context_bundle": bundle,
                "context_bundle_sha256": bundle_sha,
                "reader_prompt_sha256": message_sha,
                "rendered_prompt_sha256": rendered_sha,
                "reader_prompt_tokens_preflight": prompt_tokens,
                "max_model_length": MAX_CONTEXT_TOKENS,
                "output_reserve": READER_MAX_OUTPUT,
                "truncated": False,
                "shared_reader_contract_sha256": contract_sha,
                "reader_messages": messages,
                "projection_latency_ms": projection_latency,
            }
        )
        plan_rows.append(
            {
                "question_id": qid,
                "memory_budget_tokens": MEMORY_BUDGET,
                "plan": plan.to_dict(include_content=True),
                "selected_native_ranks": selected_ranks,
                "selected_memory_ids": list(plan.selected_memory_ids),
                "projection_latency_ms": projection_latency,
            }
        )
        item_rows_by_qid[qid] = context_items
    return plan_rows, bundle_rows, item_rows_by_qid


def _append_reader_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_reader(
    pre_reader_sha: str,
    preflight: dict[str, Any],
    bundle_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    m2a_manifest = _read_json(mem2c.MEM2A_DIR / "run_manifest.json")
    predictions = []
    ledger = []
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        runtime = mem2b._verify_reader(client, m2a_manifest)
        if runtime != preflight["writer_runtime_identity"]:
            raise RuntimeError("Frozen Qwen runtime identity changed after writer preflight")
        for bundle_row in bundle_rows:
            qid = bundle_row["question_id"]
            request = {
                "model": READER_MODEL,
                "messages": bundle_row["reader_messages"],
                "temperature": 0,
                "seed": 42,
                "max_tokens": READER_MAX_OUTPUT,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_sha = _sha_bytes(mem2a.canonical_json(request).encode("utf-8"))
            cache_identity_body = {
                "stage": RUN_ID,
                "pre_reader_freeze_sha256": pre_reader_sha,
                "question_id": qid,
                "context_bundle_sha256": bundle_row["context_bundle_sha256"],
                "reader_contract_sha256": PINNED_READER_SHA256,
                "reader_model_sha256": PINNED_WRITER_MODEL_SHA256,
                "reader_runtime_props_sha256": runtime["runtime_props_sha256"],
                "request_sha256": request_sha,
            }
            cache_identity = {
                "identity": cache_identity_body,
                "identity_sha256": _sha_json(cache_identity_body),
            }
            state_path = RUN_DIR / "calls" / "reader" / f"{qid}.json"
            if state_path.exists():
                state = _read_json(state_path)
                if (
                    state.get("cache_identity") != cache_identity
                    or state.get("status") != "COMPLETE"
                ):
                    raise RuntimeError(
                        f"Incomplete/mismatched reader cache; refusing repeat call: {qid}"
                    )
                if (
                    state["prediction"].get("quality_status") != "OK"
                    or state["call"].get("success") is not True
                ):
                    raise RuntimeError(f"Prior reader call failed; no retry permitted: {qid}")
                predictions.append(state["prediction"])
                ledger.append(state["call"])
                continue
            _append_reader_cache(
                state_path,
                {
                    "cache_identity": cache_identity,
                    "status": "STARTED",
                    "request_sha256": request_sha,
                },
            )
            started = time.perf_counter()
            response = None
            payload: dict[str, Any] = {}
            choice: dict[str, Any] = {}
            answer = None
            error_type = None
            try:
                response = client.post(f"{READER_ENDPOINT}/chat/completions", json=request)
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices", [])
                if not isinstance(choices, list) or len(choices) != 1:
                    raise RuntimeError("unexpected_choice_count")
                choice = choices[0]
                answer = choice.get("message", {}).get("content")
                if not isinstance(answer, str):
                    raise TypeError("missing_answer")
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                error_type = (
                    str(error)
                    if str(error) in {"unexpected_choice_count", "missing_answer"}
                    else type(error).__name__
                )
            latency = round((time.perf_counter() - started) * 1000, 3)
            usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
            server_prompt_tokens = usage.get("prompt_tokens")
            prompt_match = server_prompt_tokens == bundle_row["reader_prompt_tokens_preflight"]
            quality_status = "OK" if answer is not None and prompt_match else "INFRA_FAILURE"
            call = {
                "role": "reader_answer",
                "provider": "local_qwen",
                "endpoint": READER_ENDPOINT,
                "loopback_only": True,
                "model": READER_MODEL,
                "question_id": qid,
                "context_bundle_sha256": bundle_row["context_bundle_sha256"],
                "cache_identity_sha256": cache_identity["identity_sha256"],
                "prompt_sha256": bundle_row["reader_prompt_sha256"],
                "rendered_prompt_sha256": bundle_row["rendered_prompt_sha256"],
                "request_sha256": request_sha,
                "temperature": 0,
                "seed": 42,
                "enable_thinking": False,
                "max_new_tokens": READER_MAX_OUTPUT,
                "prompt_tokens_preflight": bundle_row["reader_prompt_tokens_preflight"],
                "prompt_tokens_server": server_prompt_tokens,
                "completion_tokens_server": usage.get("completion_tokens"),
                "prompt_tokens_match": prompt_match,
                "finish_reason": choice.get("finish_reason"),
                "latency_ms": latency,
                "success": answer is not None,
                "quality_status": quality_status,
                "error_type": error_type,
                "retry_count": 0,
                "hosted_call": False,
            }
            prediction = {
                "system": "m10_flatprop_dense",
                "question_id": qid,
                "predicted": answer.strip() if isinstance(answer, str) else None,
                "quality_status": quality_status,
                "reader_prompt_tokens_preflight": bundle_row["reader_prompt_tokens_preflight"],
                "reader_prompt_tokens_server": server_prompt_tokens,
                "completion_tokens_server": usage.get("completion_tokens"),
                "prompt_tokens_match": prompt_match,
                "reader_latency_ms": latency,
                "finish_reason": choice.get("finish_reason"),
                "output_hit_token_cap": choice.get("finish_reason") == "length",
                "input_truncated": False if answer is not None and prompt_match else None,
                "context_bundle_sha256": bundle_row["context_bundle_sha256"],
                "final_reader_contract_sha256": PINNED_READER_SHA256,
                "shared_reader_prompt_sha256": bundle_row["reader_prompt_sha256"],
                "rendered_prompt_sha256": bundle_row["rendered_prompt_sha256"],
                "cache_identity": cache_identity,
            }
            _append_reader_cache(
                state_path,
                {
                    "cache_identity": cache_identity,
                    "status": "COMPLETE",
                    "prediction": prediction,
                    "call": call,
                },
            )
            if quality_status != "OK":
                raise RuntimeError(f"Reader INFRA_FAILURE for {qid}; no retry permitted")
            predictions.append(prediction)
            ledger.append(call)
            print(f"reader_calls={len(predictions)}/{len(bundle_rows)}", flush=True)
    if len(predictions) != 10 or len(ledger) != 10:
        raise RuntimeError("MEM-3A requires exactly ten new reader calls")
    return predictions, ledger


def _labels_after_freeze() -> dict[str, dict[str, Any]]:
    if not _verify_frozen(PREDICTIONS_PATH) or not _verify_frozen(READER_LEDGER_PATH):
        raise RuntimeError(
            "Prediction and reader-call sidecars must freeze before benchmark label access"
        )
    predictions = read_jsonl(PREDICTIONS_PATH)
    calls = read_jsonl(READER_LEDGER_PATH)
    if (
        [row["question_id"] for row in predictions] != list(QUESTION_IDS)
        or len(calls) != 10
        or any(
            call.get("success") is not True or call.get("hosted_call") is not False
            for call in calls
        )
    ):
        raise RuntimeError("Frozen prediction/call artifacts do not pass the label-access gate")
    labels = {}
    for row in mem2c.iter_json_array(DATASET_PATH, include_gold=True):
        qid = row.get("question_id")
        if qid in QUESTION_IDS:
            labels[qid] = {
                "answer": row.get("answer", ""),
                "answer_session_ids": row.get("answer_session_ids", []),
                "question_type": row.get("question_type"),
                "has_answer": row.get("has_answer"),
            }
    if set(labels) != set(QUESTION_IDS):
        raise RuntimeError(
            "Post-freeze label join did not return exactly the ten frozen DEV questions"
        )
    return labels


def _recall(rows: list[dict[str, Any]], answer_sessions: set[str], k: int) -> float | None:
    if not answer_sessions:
        return None
    selected = rows[:k]
    hit = next(
        (
            index
            for index, row in enumerate(selected, 1)
            if row["source_session_id"] in answer_sessions
        ),
        None,
    )
    return float(hit is not None)


def _retrieval_session_metrics(
    rows: list[dict[str, Any]], answer_sessions: set[str]
) -> dict[str, Any]:
    if not answer_sessions:
        return {"answer_session_recall_at_5": None, "answer_session_recall_at_8": None, "mrr": None}
    hit_ranks = [row["rank"] for row in rows if row["source_session_id"] in answer_sessions]
    recall_at = lambda top: (
        len({row["source_session_id"] for row in rows if row["rank"] <= top} & answer_sessions)
        / len(answer_sessions)
    )
    return {
        "answer_session_recall_at_5": recall_at(5),
        "answer_session_recall_at_8": recall_at(8),
        "mrr": 1.0 / min(hit_ranks) if hit_ranks else 0.0,
    }


def _candidate_exposure(selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        grouped[(row["entity_key_candidate"], row["attribute_key_candidate"])].append(row)
    result = []
    for (entity, attribute), group in sorted(grouped.items()):
        values = {row["value_text"] for row in group}
        if len(values) > 1:
            result.append(
                {
                    "entity_key_candidate": entity,
                    "attribute_key_candidate": attribute,
                    "values_in_selected_context": [
                        {
                            "value_text": row["value_text"],
                            "observed_at": row["valid_from"],
                            "source_session_id": row["source_session_id"],
                            "memory_id": row["memory_id"],
                        }
                        for row in group
                    ],
                }
            )
    return result


def _metrics(
    predictions: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pred_by = {row["question_id"]: row for row in predictions}
    bundle_by = {row["question_id"]: row for row in bundles}
    plan_by = {row["question_id"]: row for row in plans}
    props = {row["memory_id"]: row for row in flat_rows}
    ranked_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        ranked_by[row["question_id"]].append(row)
    per_question = []
    for qid in QUESTION_IDS:
        label = labels[qid]
        answer_sessions = set(label["answer_session_ids"] or [])
        retrieval = ranked_by[qid]
        bundle = bundle_by[qid]["context_bundle"]
        selected_items = bundle["items"]
        selected_props = [props[item["memory_id"]] for item in selected_items]
        projected_session_rows = [
            {"rank": i, "source_session_id": row["source_session_id"]}
            for i, row in enumerate(selected_props, 1)
        ]
        projected_metrics = _retrieval_session_metrics(projected_session_rows, answer_sessions)
        native_metrics = _retrieval_session_metrics(retrieval, answer_sessions)
        context = bundle["serialized_context"]
        coverage, exact = mem2a._memory_gold_coverage(label["answer"], context)
        full_gold = bool(
            label["answer"]
            and mem2a._memory_gold_coverage(
                label["answer"], "\n\n".join(row["proposition_text"] for row in selected_props)
            )[1]
        )
        prediction = pred_by[qid]["predicted"] or ""
        answer_metrics = mem2a._answer_metrics(prediction, label["answer"])
        turn_set = {
            (row["source_session_id"], turn)
            for row in selected_props
            for turn in row["source_turn_indices"]
        }
        selected_sessions = {row["source_session_id"] for row in selected_props}
        per_question.append(
            {
                "question_id": qid,
                "question_type": label["question_type"],
                "has_answer": label["has_answer"],
                "answer_session_ids_count": len(answer_sessions),
                "native_answer_session_recall_at_5": native_metrics["answer_session_recall_at_5"],
                "native_answer_session_recall_at_8": native_metrics["answer_session_recall_at_8"],
                "native_mrr": native_metrics["mrr"],
                "projected_answer_session_recall_at_8": projected_metrics[
                    "answer_session_recall_at_8"
                ],
                "projected_mrr": projected_metrics["mrr"],
                "unique_source_sessions_at_8": len({row["source_session_id"] for row in retrieval}),
                "unique_source_turns_at_8": len(
                    {
                        (row["source_session_id"], turn)
                        for row in retrieval
                        for turn in row["source_turn_indices"]
                    }
                ),
                "projected_unique_source_sessions": len(selected_sessions),
                "projected_unique_source_turns": len(turn_set),
                "selected_record_count_native": len(retrieval),
                "selected_record_count_projected": len(selected_items),
                "context_reader_tokens": bundle["context_reader_tokens"],
                "content_only_reader_tokens": bundle["content_only_reader_tokens"],
                "metadata_overhead_ratio": bundle["metadata_overhead_ratio"],
                "estimated_memory_tokens": plan_by[qid]["plan"]["memory_tokens"],
                "gold_token_coverage": coverage,
                "exact_normalized_gold_sequence_present": exact,
                "selected_proposition_contains_full_gold_string": full_gold,
                "selected_proposition_provenance_intersects_answer_session": any(
                    row["source_session_id"] in answer_sessions for row in selected_props
                ),
                "token_precision": answer_metrics["token_precision"],
                "token_recall": answer_metrics["token_recall"],
                "token_f1": answer_metrics["f1"],
                "normalized_em": answer_metrics["normalized_exact_match"],
                "deterministic_abstention_accuracy": float(
                    mem2a._is_deterministic_refusal(prediction)
                )
                if label["has_answer"] is False
                else None,
                "candidate_value_conflicts_in_selected_context": _candidate_exposure(
                    selected_props
                ),
                "reader_latency_ms": pred_by[qid]["reader_latency_ms"],
                "reader_prompt_tokens": pred_by[qid]["reader_prompt_tokens_server"],
                "reader_completion_tokens": pred_by[qid]["completion_tokens_server"],
                "retrieval_latency_ms": bundle.get("retrieval_latency_ms"),
                "writer_calls": sum(call["success"] for call in ledger),
            }
        )
    metrics_names = [
        "native_answer_session_recall_at_5",
        "native_answer_session_recall_at_8",
        "native_mrr",
        "projected_answer_session_recall_at_8",
        "projected_mrr",
        "unique_source_sessions_at_8",
        "unique_source_turns_at_8",
        "context_reader_tokens",
        "content_only_reader_tokens",
        "estimated_memory_tokens",
        "gold_token_coverage",
        "exact_normalized_gold_sequence_present",
        "selected_proposition_contains_full_gold_string",
        "token_precision",
        "token_recall",
        "token_f1",
        "normalized_em",
        "deterministic_abstention_accuracy",
        "reader_latency_ms",
        "reader_prompt_tokens",
        "reader_completion_tokens",
    ]

    def mean(items: list[dict[str, Any]], name: str) -> float | None:
        values = [
            float(item[name])
            for item in items
            if isinstance(item.get(name), (int, float)) and not isinstance(item.get(name), bool)
        ]
        return statistics.fmean(values) if values else None

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        by_category[str(row["question_type"])].append(row)
    return {
        "schema_version": 1,
        "scope": "same frozen ten DEV diagnostic cases",
        "arms": {
            "m10_flatprop_dense": {
                "n": len(per_question),
                "means": {name: mean(per_question, name) for name in metrics_names},
                "by_question_type": {
                    category: {
                        "n": len(items),
                        "means": {name: mean(items, name) for name in metrics_names},
                    }
                    for category, items in sorted(by_category.items())
                },
                "per_question": per_question,
            }
        },
        "answer_metrics": "deterministic token-set metrics from the pinned MEM-2A implementation; no judge",
        "answer_session_hit_is_not_answer_bearing_proposition_hit": True,
        "abstention_accuracy": "reported only for has_answer=false; deterministic refusal semantics",
        "descriptive_only_no_ranking_or_superiority_claim": True,
        "judge_calls": 0,
    }


def _case_review(
    flat_rows: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    metrics: dict[str, Any],
    labels: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    question_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    focus = {
        "1cea1afa": "knowledge update diagnostic: Instagram follower values",
        "c4ea545c": "knowledge update diagnostic: gym frequency",
        "1c549ce4": "multi-session composition diagnostic: $120 plus $20",
        "gpt4_e061b84g": "temporal diagnostic: soccer event",
        "gpt4_f420262c": "temporal diagnostic: airline chronology",
        "778164c6": "assistant-origin diagnostic: snapper recommendation",
        "fca70973": "preference diagnostic: theme parks",
        "06878be2": "preference diagnostic: photography accessories",
    }
    questions = []
    by_qid = defaultdict(list)
    for row in flat_rows:
        by_qid[row["question_id"]].append(row)
    ranked_by = defaultdict(list)
    for row in ranked:
        ranked_by[row["question_id"]].append(row)
    metric_by = {
        row["question_id"]: row for row in metrics["arms"]["m10_flatprop_dense"]["per_question"]
    }
    pred_by = {row["question_id"]: row for row in predictions}
    for qid, note in focus.items():
        answer_sessions = set(labels[qid]["answer_session_ids"] or [])
        answer_session_propositions = [
            {
                "source_session_id": row["source_session_id"],
                "source_turn_indices": row["source_turn_indices"],
                "proposition_text": row["proposition_text"],
                "evidence_quotes": row["evidence_quotes"],
                "candidate_key": [row["entity_key_candidate"], row["attribute_key_candidate"]],
                "value_text": row["value_text"],
                "valid_from": row["valid_from"],
                "answer_session_membership_only_not_answer_bearing_turn": True,
            }
            for row in by_qid[qid]
            if row["source_session_id"] in answer_sessions
        ]
        questions.append(
            {
                "question_id": qid,
                "question": question_data[qid]["question"],
                "question_date": question_data[qid]["question_date"],
                "reflection_target": note,
                "reference_answer": labels[qid]["answer"],
                "answer_session_ids": labels[qid]["answer_session_ids"],
                "predicted": pred_by[qid]["predicted"],
                "metrics": metric_by[qid],
                "top8": [
                    {
                        "rank": row["rank"],
                        "memory_id": row["memory_id"],
                        "source_session_id": row["source_session_id"],
                        "source_turn_indices": row["source_turn_indices"],
                        "cosine_similarity": row["cosine_similarity"],
                        "proposition_text": by_id["proposition_text"],
                    }
                    for row in ranked_by[qid]
                    for by_id in [
                        next(item for item in by_qid[qid] if item["memory_id"] == row["memory_id"])
                    ]
                ],
                "answer_session_propositions": answer_session_propositions,
                "session_proposition_count": len(by_qid[qid]),
            }
        )
    return {
        "schema_version": 1,
        "stage": "MEM-3A",
        "diagnostic_only": True,
        "causal_labels_assigned": False,
        "no_revision_semantics": True,
        "focus_cases": questions,
    }


def _render_protocol(preflight: dict[str, Any], run_manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# MEM-3A - M10-FlatProp Proposition Memory",
            "",
            "Completion gate: `MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=YES`.",
            "",
            "## Research Question",
            "",
            "Can query-independent atomic proposition memories reduce raw conversational retrieval noise while deliberately preserving every historical proposition as an independent ACTIVE record? This is the write-time compression ablation before revision materialization.",
            "",
            "## Frozen Boundary",
            "",
            "M10-FlatProp uses one local, question-independent Qwen3-8B extraction request per unique source-session content identity, exact-quote provenance validation, deterministic ADD-only M10 materialization, proposition-text-only Qwen3-Embedding retrieval, Dense cosine top-8, the frozen rank-aware projection (1024 estimated Memory tokens), and the frozen shared reader. No UPDATE, DELETE, supersession, contradiction suppression, current/as-of/change routing, recency, reranking, diversity, tuning, judge, API, 102 DEV, or TEST was used.",
            "",
            f"- Frozen sessions: {preflight['unique_source_sessions']}; writer calls: {run_manifest.get('writer_calls', 'not run')}; local hosted calls: 0.",
            f"- Main reader/answer model: local Qwen3-8B Q4_K_M SHA `{PINNED_WRITER_MODEL_SHA256}`; writer role is separately logged as `{WRITER_ROLE}`.",
            f"- Embedding model: `{MODEL_ID}` revision `{MODEL_REVISION}`, tree `{MODEL_TREE_SHA256}`, weights `{WEIGHTS_SHA256}`, 1024-d, CUDA FP16, final non-padding token pooling.",
            f"- Projection contract `{PINNED_PROJECTION_SHA256}`; memory budget {MEMORY_BUDGET}; reader contract `{PINNED_READER_SHA256}`; ten reader calls only.",
            "- Data split: frozen ten diagnostic DEV IDs only; remaining 92 DEV and 398 held-out public TEST not accessed.",
            "",
            "## Reading the Result",
            "",
            "All metrics are descriptive evidence from the same ten frozen cases, not a ranking or public performance claim. Answer-session hit is not answer-bearing proposition hit. Multiple values under one candidate key are only candidate revision groups, never labeled stale/current/contradictory in MEM-3A. The lexical surface-attribute audit is heuristic and requires human review.",
            "",
            "## Reproduction Artifacts",
            "",
            "Compact extraction packets, flat proposition inventory, writer and reader ledgers, candidate-key groups, Dense top-8, projected contexts, deterministic metrics, efficiency, and MEM-2D paired diagnostic are frozen under `runs/memory/mem3/mem3a-flat-proposition-10-20260928/`. Raw source corpus and raw embedding vectors are not committed; the working SQLite databases and embedding cache remain outside Git.",
            "",
            "## Interpretation Boundary",
            "",
            "If FlatProp reduces context noise, the supported statement is semantic write-time compression on this diagnostic. If both old and new values remain selected, that exposes the state-resolution problem for a later stage; it does not itself justify an outcome claim. No revision materialization is implemented here.",
            "",
        ]
    )


def _finalize(
    upstream: dict[str, Any],
    preflight: dict[str, Any],
    session_packets: list[dict[str, Any]],
    writer_ledger: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    operation_rows: list[dict[str, Any]],
    candidate_groups: dict[str, Any],
    embedding_manifest: dict[str, Any],
    ranked: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    bundle_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    reader_ledger: list[dict[str, Any]],
    sqlite_sizes: dict[str, int],
    retrieval_latency: dict[str, float],
    writer_runtime_identity: str,
) -> dict[str, Any]:
    session_manifest = {
        "schema_version": 1,
        "question_independent": True,
        "extractor_contract_sha256": preflight["extractor_contract_sha256"],
        "writer_prompt_sha256": preflight["extractor_prompt_sha256"],
        "writer_model_sha256": PINNED_WRITER_MODEL_SHA256,
        "unique_source_sessions": len(session_packets),
        "source_sessions": [
            {
                "session_identity_sha256": packet["session_identity_sha256"],
                "source_session_ids": packet["source_session_ids"],
                "session_date": packet["session_date"],
                "source_turns_sha256": packet["source_turns_sha256"],
                "proposition_count": len(packet["propositions"]),
                "packet_sha256": _sha_json(packet),
            }
            for packet in session_packets
        ],
    }
    _write_or_verify(EXTRACTION_MANIFEST_PATH, session_manifest)
    _write_or_verify_jsonl(EXTRACTIONS_PATH, session_packets)
    _write_or_verify_jsonl(FLAT_PROPOSITIONS_PATH, flat_rows)
    _write_or_verify_jsonl(WRITER_LEDGER_PATH, writer_ledger)
    _write_or_verify_jsonl(MATERIALIZATION_LEDGER_PATH, operation_rows)
    _write_or_verify(CANDIDATE_GROUPS_PATH, candidate_groups)
    _write_or_verify(EMBEDDING_MANIFEST_PATH, embedding_manifest)
    _write_or_verify_jsonl(DENSE_TOP8_PATH, ranked)
    _write_or_verify_jsonl(CONTEXT_PLANS_PATH, plan_rows)
    _write_or_verify_jsonl(CONTEXT_BUNDLES_PATH, bundle_rows)

    frozen_inputs = {
        path.name: sha256_file(path)
        for path in (
            EXTRACTION_MANIFEST_PATH,
            EXTRACTIONS_PATH,
            FLAT_PROPOSITIONS_PATH,
            WRITER_LEDGER_PATH,
            MATERIALIZATION_LEDGER_PATH,
            CANDIDATE_GROUPS_PATH,
            EMBEDDING_MANIFEST_PATH,
            DENSE_TOP8_PATH,
            CONTEXT_PLANS_PATH,
            CONTEXT_BUNDLES_PATH,
        )
    }
    pre_reader_sha = _sha_json(frozen_inputs)
    # Reader outputs are frozen before this function is allowed to open benchmark labels.
    predictions, reader_ledger = _run_reader(pre_reader_sha, preflight, bundle_rows)
    _write_or_verify_jsonl(PREDICTIONS_PATH, predictions)
    _write_or_verify_jsonl(READER_LEDGER_PATH, reader_ledger)
    labels = _labels_after_freeze()
    metrics = _metrics(
        predictions, reader_ledger, bundle_rows, plan_rows, ranked, flat_rows, labels
    )
    _write_or_verify(METRICS_PATH, metrics)

    m2d_metrics_path = MEM2D_DIR / "deterministic_metrics.json"
    m2d_eff_path = MEM2D_DIR / "efficiency.json"
    if not mem2d._verify_sidecar(m2d_metrics_path) or not mem2d._verify_sidecar(m2d_eff_path):
        raise RuntimeError(
            "Frozen MEM-2D metric/efficiency sidecars failed after prediction freeze"
        )
    m2d_metrics = _read_json(m2d_metrics_path)
    baseline_rows = {
        row["question_id"]: row for row in m2d_metrics["arms"]["dense"]["per_question"]
    }
    flat_metric_rows = {
        row["question_id"]: row for row in metrics["arms"]["m10_flatprop_dense"]["per_question"]
    }
    comparison = {
        "schema_version": 1,
        "baseline": "MEM-2D Dense RawSpan",
        "candidate": "MEM-3A M10-FlatProp Dense",
        "fixed": [
            "ten frozen DEV questions",
            "Qwen3-Embedding-0.6B",
            "query instruction",
            "Dense cosine descending + memory_id ascending",
            "native top_k=8",
            "M10 scope/time validity",
            "ContextManager",
            "1024 Memory budget",
            "rank-aware projection",
            "final reader contract",
        ],
        "changed_component": "query-independent memory write/representation layer",
        "per_question": [
            {
                "question_id": qid,
                "baseline": baseline_rows[qid],
                "flatprop": flat_metric_rows[qid],
                "delta_token_f1": flat_metric_rows[qid]["token_f1"]
                - baseline_rows[qid]["token_f1"],
                "delta_context_reader_tokens": flat_metric_rows[qid]["context_reader_tokens"]
                - baseline_rows[qid]["context_reader_tokens"],
            }
            for qid in QUESTION_IDS
        ],
        "interpretation": "Descriptive ten-case diagnostic only; no ranking, superiority claim, or statistical inference.",
        "answer_session_hit_is_not_answer_bearing_proposition_hit": True,
    }
    _write_or_verify(COMPARISON_PATH, comparison)

    inventory = read_jsonl(MEM2C_INVENTORY)
    source_diag = _source_diagnostics(
        inventory, {packet["session_identity_sha256"]: packet for packet in session_packets}
    )
    prop_session = len({(row["question_id"], row["source_session_id"]) for row in flat_rows})
    emb_manifest = embedding_manifest
    p95 = embedding_manifest["proposition_token_p95"]
    context_counts = {
        row["question_id"]: row["context_bundle"]["context_reader_tokens"] for row in bundle_rows
    }
    efficiency = {
        "schema_version": 1,
        "representation": {
            **source_diag,
            "flat_propositions": len(flat_rows),
            "propositions_per_session": len(flat_rows) / max(1, prop_session),
            "propositions_per_raw_turn": len(flat_rows) / max(1, source_diag["raw_turns"]),
            "rawspan_records_per_flat_proposition": source_diag["raw_spans"]
            / max(1, len(flat_rows)),
            "proposition_token_median": embedding_manifest["proposition_token_median"],
            "proposition_token_p95": p95,
            "serialized_inventory_bytes": FLAT_PROPOSITIONS_PATH.stat().st_size,
            "sqlite_bytes_by_question": sqlite_sizes,
            "sqlite_total_bytes": sum(sqlite_sizes.values()),
        },
        "writer": {
            "calls": len(writer_ledger),
            "prompt_tokens": sum(row["prompt_tokens"] for row in writer_ledger),
            "completion_tokens": sum(row["completion_tokens"] or 0 for row in writer_ledger),
            "latency_ms_total": sum(row["latency_ms"] for row in writer_ledger),
            "latency_ms_mean": statistics.fmean(row["latency_ms"] for row in writer_ledger),
            "hosted_calls": sum(row["hosted_call"] for row in writer_ledger),
            "failures": sum(not row["success"] for row in writer_ledger),
        },
        "embedding": {
            "document_count": len(flat_rows),
            "unique_document_count": emb_manifest["unique_document_count"],
            "document_input_tokens": emb_manifest["document_input_tokens"],
            "query_input_tokens": emb_manifest["query_input_tokens"],
            "truncations": emb_manifest["document_truncations"] + emb_manifest["query_truncations"],
            "cache": emb_manifest["cache"],
        },
        "retrieval": {
            "p50_latency_ms": statistics.median(retrieval_latency.values()),
            "p95_latency_ms": np_percentile(list(retrieval_latency.values()), 0.95),
            "by_question_latency_ms": retrieval_latency,
        },
        "projection": {
            "memory_budget_tokens": MEMORY_BUDGET,
            "context_reader_tokens_by_question": context_counts,
            "mean_context_reader_tokens": statistics.fmean(context_counts.values()),
            "mean_content_only_reader_tokens": statistics.fmean(
                row["context_bundle"]["content_only_reader_tokens"] for row in bundle_rows
            ),
            "mean_metadata_overhead_ratio": statistics.fmean(
                row["context_bundle"]["metadata_overhead_ratio"] for row in bundle_rows
            ),
        },
        "reader": {
            "calls": len(reader_ledger),
            "latency_ms_mean": statistics.fmean(row["latency_ms"] for row in reader_ledger),
            "prompt_tokens_total": sum(row["prompt_tokens_server"] for row in reader_ledger),
            "completion_tokens_total": sum(
                row["completion_tokens_server"] or 0 for row in reader_ledger
            ),
            "hosted_calls": 0,
        },
        "no_quality_cost_winner_claim": True,
    }
    _write_or_verify(EFFICIENCY_PATH, efficiency)
    case_review = _case_review(
        flat_rows, ranked, metrics, labels, predictions, mem2d._load_frozen_inputs()["questions"]
    )
    _write_or_verify(CASE_REVIEW_PATH, case_review)
    protocol = _render_protocol(preflight, {"writer_calls": len(writer_ledger)})
    if PROTOCOL_PATH.exists() and _verify_frozen(PROTOCOL_PATH):
        if PROTOCOL_PATH.read_text(encoding="utf-8") != protocol:
            raise RuntimeError("Frozen MEM-3A protocol report changed on resume")
    else:
        PROTOCOL_PATH.write_text(protocol, encoding="utf-8", newline="\n")
        _freeze(PROTOCOL_PATH)
    report = _render_report(
        preflight, efficiency, metrics, candidate_groups, comparison, sqlite_sizes
    )
    _write_or_verify(RUN_REPORT_PATH, report)

    sidecar_paths = [
        EXTRACTION_MANIFEST_PATH,
        EXTRACTIONS_PATH,
        FLAT_PROPOSITIONS_PATH,
        WRITER_LEDGER_PATH,
        MATERIALIZATION_LEDGER_PATH,
        CANDIDATE_GROUPS_PATH,
        EMBEDDING_MANIFEST_PATH,
        DENSE_TOP8_PATH,
        CONTEXT_PLANS_PATH,
        CONTEXT_BUNDLES_PATH,
        PREDICTIONS_PATH,
        READER_LEDGER_PATH,
        METRICS_PATH,
        EFFICIENCY_PATH,
        COMPARISON_PATH,
        CASE_REVIEW_PATH,
        PROTOCOL_PATH,
        RUN_REPORT_PATH,
        PROMPT_PATH,
        EXTRACTOR_CONTRACT_PATH,
        RETRIEVAL_CONTRACT_PATH,
    ]
    sidecars_valid = all(_verify_frozen(path) for path in sidecar_paths)
    no_revision = len(operation_rows) == len(flat_rows) and all(
        row["operation"] == "ADD"
        and row["status"] == "active"
        and row["version"] == 1
        and row["supersedes_id"] is None
        for row in operation_rows
    )
    run_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "COMPLETE",
        "base_commit_sha": PINNED_BASE_COMMIT,
        "question_ids": list(QUESTION_IDS),
        "test_access": False,
        "remaining_92_dev_run": False,
        "reader_model_role": "reader_answer",
        "memory_writer_role": WRITER_ROLE,
        "embedding_model_role": "embedding_model",
        "judge_model": None,
        "writer_calls": len(writer_ledger),
        "writer_failures": sum(not row["success"] for row in writer_ledger),
        "reader_calls_successful": len(reader_ledger),
        "reader_calls_failed": sum(not row["success"] for row in reader_ledger),
        "hosted_calls": sum(bool(row["hosted_call"]) for row in writer_ledger + reader_ledger),
        "judge_calls": 0,
        "memory_operations": len(operation_rows),
        "memory_records": len(flat_rows),
        "completion_gate_marker": "MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=YES",
        "gate": {
            "upstream_gates_verified": True,
            "writer_contract_frozen_before_calls": True,
            "writer_question_independent": True,
            "one_packet_per_unique_source_session": len(session_packets)
            == preflight["unique_source_sessions"],
            "strict_schema_and_quote_validation": True,
            "writer_hosted_calls_zero": sum(row["hosted_call"] for row in writer_ledger) == 0,
            "all_operations_add": no_revision,
            "all_records_active_v1": no_revision,
            "no_supersession": no_revision,
            "embedding_contract_frozen_local": emb_manifest["local_only"]
            and emb_manifest["document_truncations"] == 0
            and emb_manifest["query_truncations"] == 0,
            "dense_top8_frozen_ten": len(ranked) == 80,
            "projection_contract_unchanged": PINNED_PROJECTION_SHA256
            == upstream["projection_contract_sha256"],
            "memory_budget_1024": MEMORY_BUDGET == 1024,
            "reader_contract_unchanged": PINNED_READER_SHA256 == upstream["reader_contract_sha256"],
            "exactly_ten_reader_calls": len(reader_ledger) == 10
            and all(row["success"] and row["quality_status"] == "OK" for row in reader_ledger),
            "judge_calls_zero": True,
            "test_access_false": True,
            "remaining_92_dev_not_run": True,
            "all_compact_artifact_sidecars_valid": sidecars_valid,
            "passed": False,
        },
        "upstream": upstream,
        "artifact_sha256": {path.name: sha256_file(path) for path in sidecar_paths},
        "pre_reader_freeze_sha256": pre_reader_sha,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    run_manifest["gate"]["passed"] = all(
        value for key, value in run_manifest["gate"].items() if key != "passed"
    )
    if not run_manifest["gate"]["passed"]:
        run_manifest["completion_gate_marker"] = "MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO"
    _write_json(RUN_MANIFEST_PATH, run_manifest)
    _freeze(RUN_MANIFEST_PATH)
    if run_manifest["gate"]["passed"]:
        print("MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=YES", flush=True)
    else:
        raise RuntimeError("MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO")
    return run_manifest


def np_percentile(values: list[float] | list[int], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _render_report(
    preflight: dict[str, Any],
    efficiency: dict[str, Any],
    metrics: dict[str, Any],
    groups: dict[str, Any],
    comparison: dict[str, Any],
    sqlite_sizes: dict[str, int],
) -> str:
    means = metrics["arms"]["m10_flatprop_dense"]["means"]
    lines = [
        "# MEM-3A - M10-FlatProp Frozen Ten-Case Diagnostic",
        "",
        "Completion gate: `MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=YES`.",
        "",
        "## Scope",
        "",
        "M10-FlatProp performs question-independent semantic proposition extraction and lossless historical retention, then compares Dense retrieval against MEM-2D RawSpan + Dense on the same ten frozen DEV questions. All findings are descriptive; this is not a ranking or public performance claim.",
        "",
        f"- Source sessions: {efficiency['representation']['source_sessions']} scoped / {efficiency['representation']['unique_session_content_identities']} unique content identities; raw turns {efficiency['representation']['raw_turns']}; RawSpans {efficiency['representation']['raw_spans']}; propositions {efficiency['representation']['flat_propositions']}.",
        f"- Compression: {efficiency['representation']['propositions_per_session']:.2f} propositions/session; {efficiency['representation']['propositions_per_raw_turn']:.3f} propositions/raw turn; RawSpan/FlatProp {efficiency['representation']['rawspan_records_per_flat_proposition']:.2f}.",
        f"- Local writer calls {efficiency['writer']['calls']}; input/output tokens {efficiency['writer']['prompt_tokens']}/{efficiency['writer']['completion_tokens']}; writer latency total {efficiency['writer']['latency_ms_total'] / 1000:.1f}s; hosted calls {efficiency['writer']['hosted_calls']}.",
        f"- Embedding documents {efficiency['embedding']['document_count']} ({efficiency['embedding']['unique_document_count']} unique); local CUDA; truncations {efficiency['embedding']['truncations']}.",
        f"- Temporary M10 SQLite sizes total {sum(sqlite_sizes.values()):,} bytes (kept in local cache, not committed).",
        "",
        "## Diagnostic Results",
        "",
        "| Measure | M10-FlatProp Dense mean |",
        "|---|---:|",
    ]
    for key, label in (
        ("native_answer_session_recall_at_5", "Native answer-session Recall@5"),
        ("native_answer_session_recall_at_8", "Native answer-session Recall@8"),
        ("native_mrr", "Native answer-session MRR"),
        ("projected_answer_session_recall_at_8", "Projected answer-session Recall@8"),
        ("gold_token_coverage", "Normalized gold-token coverage"),
        ("exact_normalized_gold_sequence_present", "Exact normalized gold sequence present"),
        (
            "selected_proposition_contains_full_gold_string",
            "Selected proposition contains full gold string",
        ),
        ("token_precision", "Token precision"),
        ("token_recall", "Token recall"),
        ("token_f1", "Token F1"),
        ("normalized_em", "Normalized EM"),
        ("context_reader_tokens", "Reader-visible memory context tokens"),
        ("estimated_memory_tokens", "ContextManager estimated memory tokens"),
    ):
        lines.append(f"| {label} | {means.get(key)} |")
    lines.extend(
        [
            "",
            "## Candidate Key Diagnostics",
            "",
            f"- Distinct entity/attribute key pairs: {groups['candidate_key_pairs']}; exact-pair reuse rate: {groups['exact_key_pair_reuse_rate']:.3f}; same-key/multiple-value groups: {groups['same_key_multiple_value_group_count']}.",
            "- Multiple values remain independent ACTIVE version-1 records. These groups are candidates only, not contradiction/current/stale classifications.",
            f"- Surface-attribute / different-key examples: {groups['surface_attribute_similarity_audit']['different_key_pair_example_count']} (lexical heuristic; manual review required).",
            "- Full per-question top-8 value exposures and the Instagram/gym reflection targets are retained in `mem_3a_case_review.json`.",
            "",
            "## Paired Diagnostic",
            "",
            "Only the memory representation/write layer changes from MEM-2D RawSpan + Dense; the embedding model, query, M10 scope/time eligibility, Dense top-8, projection, budget, and reader remain fixed. Per-question paired values are in `comparison_mem2d_dense_vs_mem3a_flat.json`.",
            "",
            "| Measure | Mean |",
            "|---|---:|",
        ]
    )
    lines.extend(
        [
            f"| Retrieval p50 / p95 latency (ms) | {efficiency['retrieval']['p50_latency_ms']:.1f} / {efficiency['retrieval']['p95_latency_ms']:.1f} |",
            f"| Reader-visible context tokens | {efficiency['projection']['mean_context_reader_tokens']:.1f} |",
            f"| Content-only tokens | {efficiency['projection']['mean_content_only_reader_tokens']:.1f} |",
            f"| Metadata overhead ratio | {efficiency['projection']['mean_metadata_overhead_ratio']:.3f} |",
            "",
            "## Interpretation Boundary",
            "",
            "This ten-case development diagnostic does not support a general quality or cost winner claim. Answer-session hit is not answer-bearing proposition hit. FlatProp deliberately does not resolve revisions: older and newer values may both be retrieved and remain ACTIVE. No 102-case DEV, remaining 92 DEV, or 398-case TEST was run; no revision materialization was implemented.",
            "",
            f"- Extractor preflight SHA: `{sha256_file(PREFLIGHT_PATH)}`.",
            f"- Local embedding contract SHA: `{preflight['retrieval_view_contract_sha256']}`.",
            f"- Reader contract SHA: `{PINNED_READER_SHA256}`; projection SHA: `{PINNED_PROJECTION_SHA256}`.",
            "- Hosted calls: 0; judge calls: 0; TEST access: false.",
            "",
        ]
    )
    return "\n".join(lines)


def _run() -> dict[str, Any]:
    preflight = _read_json(PREFLIGHT_PATH)
    if not _verify_frozen(PREFLIGHT_PATH):
        raise RuntimeError("Writer preflight sidecar failed; no model calls are allowed")
    current = _preflight(save=False)
    if current != preflight:
        raise RuntimeError("Live source/runtime no longer matches frozen writer preflight")
    upstream = _verify_upstream()
    _, refs = _source_sessions()
    session_packets, writer_ledger = _extract_all(preflight)
    contract_sha = preflight["extractor_contract_sha256"]
    flat_rows, operation_rows, records_by_question = _materialize(
        session_packets, refs, contract_sha
    )
    # Freeze the proposition inventory before initializing dense embeddings; retrieval embeds text only.
    _write_or_verify_jsonl(FLAT_PROPOSITIONS_PATH, flat_rows)
    _write_or_verify_jsonl(EXTRACTIONS_PATH, session_packets)
    _write_or_verify_jsonl(WRITER_LEDGER_PATH, writer_ledger)
    _write_or_verify_jsonl(MATERIALIZATION_LEDGER_PATH, operation_rows)
    _write_or_verify(
        EXTRACTION_MANIFEST_PATH,
        {
            "schema_version": 1,
            "question_independent": True,
            "extractor_contract_sha256": contract_sha,
            "writer_prompt_sha256": preflight["extractor_prompt_sha256"],
            "writer_model_sha256": PINNED_WRITER_MODEL_SHA256,
            "unique_source_sessions": len(session_packets),
            "source_sessions": [
                {
                    "session_identity_sha256": packet["session_identity_sha256"],
                    "source_session_ids": packet["source_session_ids"],
                    "session_date": packet["session_date"],
                    "source_turns_sha256": packet["source_turns_sha256"],
                    "proposition_count": len(packet["propositions"]),
                    "packet_sha256": _sha_json(packet),
                }
                for packet in session_packets
            ],
        },
    )
    candidate_groups = _candidate_groups(flat_rows)
    _write_or_verify(CANDIDATE_GROUPS_PATH, candidate_groups)
    started = time.perf_counter()
    ranked, embedding_manifest, query_tokens, retrieval_latency = _embedding_and_retrieval(
        flat_rows, records_by_question, refs, preflight
    )
    embedding_manifest["query_token_counts"] = query_tokens
    _write_or_verify(EMBEDDING_MANIFEST_PATH, embedding_manifest)
    m2d_inputs = mem2d._load_frozen_inputs()
    questions = m2d_inputs["questions"]
    sqlite_sizes, sqlite_matches = _materialize_sqlite(
        records_by_question,
        flat_rows,
        _sha_json({"propositions": sha256_file(FLAT_PROPOSITIONS_PATH), "contract": contract_sha}),
    )
    # SQLite-backed M10 scope/time matches are audited against the pure M10 filtering used above.
    for qid in QUESTION_IDS:
        sqlite_ids = {item["record"].memory_id for item in sqlite_matches[qid]}
        expected = {row["memory_id"] for row in ranked if row["question_id"] == qid}
        if not expected.issubset(sqlite_ids):
            raise RuntimeError(
                f"SQLite M10 scope/time filter disagrees with dense eligible set: {qid}"
            )
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        _, reader_sha, system_template, user_template = load_final_reader_contract()
        if reader_sha != PINNED_READER_SHA256:
            raise RuntimeError("Reader contract changed before context projection")
        plans, bundles, _ = _project_contexts(
            ranked,
            records_by_question,
            questions,
            client,
            (upstream["reader_contract"], reader_sha, system_template, user_template),
        )
    _write_or_verify_jsonl(DENSE_TOP8_PATH, ranked)
    _write_or_verify_jsonl(CONTEXT_PLANS_PATH, plans)
    _write_or_verify_jsonl(CONTEXT_BUNDLES_PATH, bundles)
    if len(ranked) != 80 or any(
        len([item for item in ranked if item["question_id"] == qid]) != 8 for qid in QUESTION_IDS
    ):
        raise RuntimeError("Frozen Dense top-8 artifact is incomplete")
    # Freeze pre-reader state; labels are loaded only after predictions and call ledger freeze.
    artifact_paths = [
        EXTRACTION_MANIFEST_PATH,
        EXTRACTIONS_PATH,
        FLAT_PROPOSITIONS_PATH,
        WRITER_LEDGER_PATH,
        MATERIALIZATION_LEDGER_PATH,
        CANDIDATE_GROUPS_PATH,
        EMBEDDING_MANIFEST_PATH,
        DENSE_TOP8_PATH,
        CONTEXT_PLANS_PATH,
        CONTEXT_BUNDLES_PATH,
    ]
    if not all(_verify_frozen(path) for path in artifact_paths):
        raise RuntimeError("A pre-reader artifact failed SHA freeze")
    pre_reader_sha = _sha_json({path.name: sha256_file(path) for path in artifact_paths})
    run_manifest = _finalize(
        upstream,
        preflight,
        session_packets,
        writer_ledger,
        flat_rows,
        operation_rows,
        candidate_groups,
        embedding_manifest,
        ranked,
        plans,
        bundles,
        [],
        [],
        sqlite_sizes,
        retrieval_latency,
        pre_reader_sha,
    )
    run_manifest["elapsed_wall_seconds"] = round(time.perf_counter() - started, 3)
    _write_json(RUN_MANIFEST_PATH, run_manifest)
    _freeze(RUN_MANIFEST_PATH)
    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        preflight = _preflight(save=True)
        print(f"writer_preflight={PREFLIGHT_PATH}", flush=True)
        print(f"unique_source_sessions={preflight['unique_source_sessions']}", flush=True)
        print("MEM3A_WRITER_PREFLIGHT=YES", flush=True)
        return 0
    try:
        manifest = _run()
        print(manifest["completion_gate_marker"], flush=True)
        return 0
    except Exception as error:  # noqa: BLE001 - freeze an explicit failed stage record before stopping.
        failed = {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": "FAILED",
            "completion_gate_marker": "MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO",
            "failure_type": type(error).__name__,
            "failure": str(error),
            "test_access": False,
            "judge_calls": 0,
            "hosted_calls": 0,
        }
        _write_json(RUN_MANIFEST_PATH, failed)
        print(
            f"MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
