"""Run MEM-3A.2 minimal FlatProp writer and frozen-ten diagnostic locally."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import flat_proposition_writer_v3 as writer_v3
import run_mem1d3_reader as reader_runtime
import run_mem2b_rank_aware_projection as mem2b
import run_mem2d_semantic_retrieval as mem2d
import run_mem3a1_extractor_v2_qualification as v2_runtime
import run_mem3a_flat_proposition as flat
from flat_proposition_writer_v2 import (
    WriterQualificationFailure,
    WriterRecoveryError,
    atomic_write_bytes,
    build_source_span_catalog,
    canonical_json,
    catalog_sha256,
    execute_or_resume,
    sha256_bytes,
)
from mem1_artifacts import read_jsonl

sha256_file = v2_runtime.sha256_file

BASE_COMMIT = "165e0b123552e19e6aa2469a36a89dde944e9524"
RUN_ID = "mem3a2-minimal-flatprop-frozen-10-20260929"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
PROMPT_PATH = ROOT / "docs/research/memory/flat_proposition_extractor_v3.txt"
CONTRACT_PATH = ROOT / "docs/research/memory/flat_proposition_extractor_v3_contract.json"
OLD_V2_PROMPT_SHA = "0aea4338d5de1e8dac753e3aae98caf36461cc15b1cf8d782a36f298f63cf627"
OLD_V2_CONTRACT_SHA = "fee5ae49c13d7b55d75c29af38146c2a9e079ac7ca24e5aeff151dee7a31b666"
EXPECTED_OLD_GATE = "MEM3A_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO"
EXPECTED_OLD_V2_GATE = "MEM3A1_EXTRACTOR_V2_QUALIFIED=NO"
EXPECTED_OLD_V2R_GATE = "MEM3A1R_EXTRACTOR_V2_QUALIFIED=NO"
EXPECTED_OLD_MEM3A2 = "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NOT_RUN"
OLD_FLAT_MANIFEST = ROOT / "runs/memory/mem3/mem3a-flat-proposition-10-20260928/run_manifest.json"
OLD_V2_MANIFEST = (
    ROOT / "runs/memory/mem3/mem3a1-extractor-v2-qualification-20260928/run_manifest.json"
)
OLD_V2R_MANIFEST = (
    ROOT / "runs/memory/mem3/mem3a1r-extractor-v2-repaired-20260928/run_manifest.json"
)
OLD_MEM3A2_MANIFEST = (
    ROOT / "runs/memory/mem3/mem3a2-flat-proposition-10-20260928/run_manifest.json"
)
OLD_SELECTION = (
    ROOT
    / "runs/memory/mem3/mem3a1-extractor-v2-qualification-20260928/qualification_selection.json"
)
LOCAL_CACHE_ROOT = ROOT / ".cache/health-copilot/mem3a2-minimal-flatprop-v3-writer"
OUTPUT_RESERVE = 4096
CONTEXT_LIMIT = 131072
EXPECTED_FOCUS = (
    "1cea1afa",
    "c4ea545c",
    "1c549ce4",
    "gpt4_e061b84g",
    "gpt4_f420262c",
    "778164c6",
    "fca70973",
    "06878be2",
)
TRANSIENT_HINT = re.compile(
    r"\b(can you help|help me|looking for advice|seeking advice|asks? for advice|"
    r"asked for help|wants? to know|how can i|what should i|one[- ]off request)\b",
    re.IGNORECASE,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    atomic_write_bytes(path, payload)
    return flat._freeze(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)
    return flat._freeze(path)


def _write_or_verify_json(path: Path, value: Any) -> str:
    expected = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    if path.exists():
        if not flat._verify_frozen(path) or path.read_bytes() != expected:
            raise RuntimeError(f"Frozen v3 artifact differs: {path.name}")
        return sha256_file(path)
    return _write_json(path, value)


def _write_or_verify_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    expected = b"".join(canonical_json(row) + b"\n" for row in rows)
    if path.exists():
        if not flat._verify_frozen(path) or path.read_bytes() != expected:
            raise RuntimeError(f"Frozen v3 artifact differs: {path.name}")
        return sha256_file(path)
    return _write_jsonl(path, rows)


def _freeze_source(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = flat._sidecar(path)
    expected = f"{digest}  {path.name}\n"
    if sidecar.exists():
        if sidecar.read_text(encoding="ascii") != expected:
            raise RuntimeError(f"Existing v3 source SHA sidecar differs: {path.name}")
    else:
        atomic_write_bytes(sidecar, expected.encode("ascii"))
    if not flat._verify_frozen(path):
        raise RuntimeError(f"V3 source artifact failed SHA verification: {path.name}")
    return digest


def _configure_flat_paths() -> None:
    flat.RUN_ID = RUN_ID
    flat.RUN_DIR = RUN_DIR
    flat.PINNED_BASE_COMMIT = BASE_COMMIT
    flat.PROMPT_PATH = PROMPT_PATH
    flat.EXTRACTOR_CONTRACT_PATH = CONTRACT_PATH
    flat.RUN_REPORT_PATH = RUN_DIR / "report.md"
    flat.RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
    flat.PREFLIGHT_PATH = RUN_DIR / "writer_preflight_v3.json"
    flat.EXTRACTION_MANIFEST_PATH = RUN_DIR / "session_extraction_manifest.json"
    flat.EXTRACTIONS_PATH = RUN_DIR / "session_extractions.jsonl"
    flat.FLAT_PROPOSITIONS_PATH = RUN_DIR / "flat_propositions.jsonl"
    flat.WRITER_LEDGER_PATH = RUN_DIR / "writer_call_ledger.jsonl"
    flat.MATERIALIZATION_LEDGER_PATH = RUN_DIR / "materialization_ledger.jsonl"
    flat.CANDIDATE_GROUPS_PATH = RUN_DIR / "revision_semantics_not_in_scope.json"
    flat.EMBEDDING_MANIFEST_PATH = RUN_DIR / "embedding_manifest.json"
    flat.DENSE_TOP8_PATH = RUN_DIR / "dense_top8.jsonl"
    flat.CONTEXT_PLANS_PATH = RUN_DIR / "context_plans.jsonl"
    flat.CONTEXT_BUNDLES_PATH = RUN_DIR / "context_bundles.jsonl"
    flat.PREDICTIONS_PATH = RUN_DIR / "predictions.jsonl"
    flat.READER_LEDGER_PATH = RUN_DIR / "reader_call_ledger.jsonl"
    flat.METRICS_PATH = RUN_DIR / "deterministic_metrics.json"
    flat.EFFICIENCY_PATH = RUN_DIR / "efficiency.json"
    flat.COMPARISON_PATH = RUN_DIR / "comparison_mem2d_dense_vs_mem3a2_flat.json"
    flat.CASE_REVIEW_PATH = RUN_DIR / "mem_3a2_case_review.json"
    flat.PROTOCOL_PATH = RUN_DIR / "protocol_v3.md"


def _verify_historical_gates() -> dict[str, Any]:
    expected = (
        (OLD_FLAT_MANIFEST, EXPECTED_OLD_GATE),
        (OLD_V2_MANIFEST, EXPECTED_OLD_V2_GATE),
        (OLD_V2R_MANIFEST, EXPECTED_OLD_V2R_GATE),
        (OLD_MEM3A2_MANIFEST, EXPECTED_OLD_MEM3A2),
    )
    results = {}
    for path, marker in expected:
        if not flat._verify_frozen(path):
            raise RuntimeError(f"Historical gate record is not SHA-frozen: {path}")
        item = _json(path)
        if item.get("completion_gate_marker") != marker:
            raise RuntimeError(f"Historical gate changed: {path}")
        results[path.parent.name] = {"marker": marker, "sha256": sha256_file(path)}
    old_prompt = ROOT / "docs/research/memory/flat_proposition_extractor_v2.txt"
    old_contract = ROOT / "docs/research/memory/flat_proposition_extractor_v2_contract.json"
    if (
        sha256_file(old_prompt) != OLD_V2_PROMPT_SHA
        or sha256_file(old_contract) != OLD_V2_CONTRACT_SHA
    ):
        raise RuntimeError("Historical extractor-v2 source bytes changed")
    return results


def _verify_upstream() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != BASE_COMMIT:
        raise RuntimeError(f"MEM-3A.2 requires base {BASE_COMMIT}; found {head}")
    flat.PINNED_BASE_COMMIT = BASE_COMMIT
    upstream = flat._verify_upstream()
    return {
        **upstream,
        "historical_gates": _verify_historical_gates(),
        "base_commit_sha": BASE_COMMIT,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
    }


def _source_sessions_with_catalog() -> tuple[
    dict[str, dict[str, Any]], dict[str, dict[str, str]], list[dict[str, Any]]
]:
    sessions, refs = flat._source_sessions()
    inventory = read_jsonl(flat.MEM2C_INVENTORY)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        qid = row["question_id"]
        sid = row["source_session_id"]
        grouped[(qid, sid)].append(row)
    for identity, session in sessions.items():
        scoped_refs = [
            (qid, sid) for qid in flat.QUESTION_IDS if (sid := refs[qid].get(identity)) is not None
        ]
        if len(scoped_refs) != 1:
            raise RuntimeError(
                f"Expected one frozen scope per unique source-session identity: {identity}"
            )
        qid, sid = scoped_refs[0]
        rows = grouped.get((qid, sid), [])
        raw_spans = [
            {
                "source_turn_index": int(row["value"]["source_turn_index"]),
                "source_span_index": int(row["value"]["source_span_index"]),
                "char_start": int(row["char_start"]),
                "char_end": int(row["char_end"]),
                "role": row["value"]["role"],
                "content": row["value"]["content"],
            }
            for row in rows
        ]
        catalog = build_source_span_catalog(raw_spans)
        turns: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for span in catalog:
            turns[span["source_turn_index"]].append(span)
        reconstructed = [
            {
                "turn_index": turn_index,
                "role": turn_spans[0]["role"],
                "content": "".join(span["content"] for span in turn_spans),
            }
            for turn_index, turn_spans in sorted(turns.items())
        ]
        if canonical_json(reconstructed) != canonical_json(session["turns"]):
            raise RuntimeError(f"Harness RawSpan catalog differs from frozen session: {identity}")
        session["catalog"] = catalog
        session["catalog_sha256"] = catalog_sha256(catalog)
        session["source_session_ids"] = [sid]
    if len(sessions) != 477:
        raise RuntimeError(f"Expected 477 unique source sessions, got {len(sessions)}")
    return sessions, refs, inventory


def _build_preflight(*, tokenize: bool) -> dict[str, Any]:
    upstream = _verify_upstream()
    prompt_sha = _freeze_source(PROMPT_PATH)
    contract_sha = _freeze_source(CONTRACT_PATH)
    retrieval_contract_sha = _freeze_source(flat.RETRIEVAL_CONTRACT_PATH)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    if not prompt:
        raise RuntimeError("v3 prompt is empty")
    sessions, _, _ = _source_sessions_with_catalog()
    m2a_manifest = _json(flat.mem2c.MEM2A_DIR / "run_manifest.json")
    rows = []
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        runtime = mem2b._verify_reader(client, m2a_manifest)
        if (
            not runtime.get("loopback_only")
            or runtime.get("endpoint") != flat.READER_ENDPOINT
            or runtime.get("server_build") != "llama.cpp 10068 (571d0d540)"
            or runtime.get("model_sha256") != flat.PINNED_WRITER_MODEL_SHA256
            or runtime.get("context_tokens") != CONTEXT_LIMIT
        ):
            raise RuntimeError("Frozen local Qwen3-8B writer runtime identity failed")
        if urlsplit(flat.READER_ENDPOINT).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError("Writer endpoint is not loopback")
        for identity, session in sessions.items():
            request = writer_v3.writer_request(
                session_date=session["session_date"],
                catalog=session["catalog"],
                system_prompt=prompt,
                model_alias=flat.READER_MODEL,
            )
            if tokenize:
                tokenizer_started = time.perf_counter()
                prompt_tokens, rendered_sha = reader_runtime._render_and_tokenize(
                    client, request["messages"]
                )
                tokenizer_ms = round((time.perf_counter() - tokenizer_started) * 1000, 3)
            else:
                raise AssertionError("non-tokenizing preflight is reconstructed separately")
            if prompt_tokens + OUTPUT_RESERVE > CONTEXT_LIMIT:
                raise RuntimeError(f"v3 prompt exceeds frozen context reserve: {identity}")
            schema_sha = writer_v3.schema_sha256(session["catalog"])
            rows.append(
                {
                    "session_identity_sha256": identity,
                    "source_session_ids": session["source_session_ids"],
                    "session_date": session["session_date"],
                    "source_turns_sha256": session["source_turns_sha256"],
                    "catalog_sha256": session["catalog_sha256"],
                    "request_sha256": sha256_bytes(canonical_json(request)),
                    "dynamic_schema_sha256": schema_sha,
                    "rendered_prompt_sha256": rendered_sha,
                    "prompt_tokens": prompt_tokens,
                    "output_reserve": OUTPUT_RESERVE,
                    "max_context_tokens": CONTEXT_LIMIT,
                    "truncated": False,
                    "tokenizer_preflight_latency_ms": tokenizer_ms,
                }
            )
    if len(rows) != 477 or len({row["session_identity_sha256"] for row in rows}) != 477:
        raise RuntimeError("v3 tokenizer preflight did not cover 477 unique identities")
    identity_body = {
        "stage": RUN_ID,
        "base_commit_sha": BASE_COMMIT,
        "prompt_sha256": prompt_sha,
        "contract_sha256": contract_sha,
        "writer_request_source_sha256": sha256_file(Path(writer_v3.__file__)),
        "writer_validator_source_sha256": sha256_file(Path(writer_v3.__file__)),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "flatprop_runtime_source_sha256": sha256_file(Path(flat.__file__)),
        "dense_embedding_adapter_source_sha256": sha256_file(
            TOOLS_DIR / "local_qwen3_embedding.py"
        ),
        "transport_unwrap_source_sha256": sha256_file(
            Path(v2_runtime.__file__).with_name("flat_proposition_writer_v2.py")
        ),
        "writer_model_sha256": runtime["model_sha256"],
        "endpoint": runtime["endpoint"],
        "runtime_props_sha256": runtime["runtime_props_sha256"],
        "source_count": 477,
        "preflight_request_root_sha256": sha256_bytes(
            ("\n".join(row["request_sha256"] for row in rows) + "\n").encode("ascii")
        ),
    }
    extraction_identity = {
        "identity": identity_body,
        "identity_sha256": sha256_bytes(canonical_json(identity_body)),
    }
    preflight = {
        "schema_version": 1,
        "stage": RUN_ID,
        "base_commit_sha": BASE_COMMIT,
        "contract_id": "flat-proposition-extractor-v3-minimal",
        "extractor_prompt_sha256": prompt_sha,
        "extractor_contract_sha256": contract_sha,
        "retrieval_view_contract_sha256": retrieval_contract_sha,
        "writer_role": "memory_write_extract",
        "writer_model_role": "reader_answer_and_memory_internal_llm_same_frozen_qwen3_8b",
        "writer_model_sha256": runtime["model_sha256"],
        "writer_runtime_identity": runtime,
        "max_context_tokens": CONTEXT_LIMIT,
        "output_reserve": OUTPUT_RESERVE,
        "unique_source_sessions": len(rows),
        "requests": rows,
        "all_prompt_tokens_plus_reserve_fit": all(
            row["prompt_tokens"] + OUTPUT_RESERVE <= CONTEXT_LIMIT and not row["truncated"]
            for row in rows
        ),
        "truncated_requests": sum(row["truncated"] for row in rows),
        "local_only": True,
        "hosted_calls": 0,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
        "upstream": upstream,
        "extraction_identity": extraction_identity,
    }
    if not preflight["all_prompt_tokens_plus_reserve_fit"] or preflight["truncated_requests"]:
        raise RuntimeError("v3 477-request preflight failed fit/truncation gate")
    return preflight


def _preflight_core(preflight: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in preflight.items() if key not in {"requests"}} | {
        "requests": [
            {key: value for key, value in row.items() if key != "tokenizer_preflight_latency_ms"}
            for row in preflight["requests"]
        ]
    }


def _preflight(*, save: bool) -> dict[str, Any]:
    current = _build_preflight(tokenize=True)
    path = RUN_DIR / "writer_preflight_v3.json"
    if save:
        if path.exists():
            old = _json(path)
            if not flat._verify_frozen(path):
                raise RuntimeError("Existing writer_preflight_v3 sidecar is invalid")
            if _preflight_core(old) != _preflight_core(current):
                if LOCAL_CACHE_ROOT.exists() and any(LOCAL_CACHE_ROOT.rglob("journal.jsonl")):
                    raise RuntimeError(
                        "v3 preflight differs after a writer journal exists; refusing to change identity"
                    )
                _write_json(path, current)
                _write_json(
                    RUN_DIR / "writer_extraction_identity.json", current["extraction_identity"]
                )
                return current
            return old
        _write_json(path, current)
        _write_or_verify_json(
            RUN_DIR / "writer_extraction_identity.json", current["extraction_identity"]
        )
        return current
    if not path.exists() or not flat._verify_frozen(path):
        raise RuntimeError("Frozen writer_preflight_v3.json missing or invalid")
    saved = _json(path)
    if _preflight_core(saved) != _preflight_core(current):
        raise RuntimeError("Live source/model/request differs from frozen v3 preflight")
    return saved


def _normalized_session_packet(
    session: dict[str, Any], normalized: dict[str, Any]
) -> dict[str, Any]:
    propositions = []
    for index, prop in enumerate(normalized["propositions"]):
        propositions.append({**prop, "proposition_index": index})
    return {
        "session_identity_sha256": session["session_identity_sha256"],
        "session_date": session["session_date"],
        "valid_from": session["valid_from"],
        "source_session_ids": session["source_session_ids"],
        "source_turns_sha256": session["source_turns_sha256"],
        "catalog_sha256": session["catalog_sha256"],
        "propositions": propositions,
    }


def _extract_all(
    preflight: dict[str, Any], sessions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = preflight["extractor_prompt_sha256"]
    contract_sha = preflight["extractor_contract_sha256"]
    runtime = preflight["writer_runtime_identity"]
    preflight_by_id = {row["session_identity_sha256"]: row for row in preflight["requests"]}
    outcomes: dict[str, dict[str, Any]] = {}
    ledger_by_id: dict[str, dict[str, Any]] = {}
    provider_counter = {"calls": 0, "last_latency_ms": None}
    unwrap_sha = sha256_file(Path(__file__).with_name("flat_proposition_writer_v2.py"))
    validator_sha = sha256_file(Path(writer_v3.__file__))
    schema_source_sha = sha256_file(Path(writer_v3.__file__))
    manifest_path = RUN_DIR / "session_extraction_manifest.json"
    _write_json(
        RUN_DIR / "run_manifest.json",
        {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": "WRITER_IN_PROGRESS",
            "base_commit_sha": BASE_COMMIT,
            "completion_gate_marker": "MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=PENDING",
            "expected_unique_source_sessions": 477,
            "completed_unique_source_sessions": 0,
            "provider_calls_this_process": 0,
            "retries": 0,
            "hosted_calls": 0,
            "labels_loaded": False,
            "test_access": False,
            "102_dev_access": False,
        },
    )
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        for index, (identity, session) in enumerate(sessions.items(), 1):
            frozen = preflight_by_id.get(identity)
            if frozen is None or frozen["catalog_sha256"] != session["catalog_sha256"]:
                raise RuntimeError(f"v3 preflight identity mismatch before writer call: {identity}")
            request = writer_v3.writer_request(
                session_date=session["session_date"],
                catalog=session["catalog"],
                system_prompt=prompt,
                model_alias=flat.READER_MODEL,
            )
            request_sha = sha256_bytes(canonical_json(request))
            if request_sha != frozen["request_sha256"]:
                raise RuntimeError(f"v3 frozen request SHA changed: {identity}")

            def provider(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
                provider_counter["calls"] += 1
                started = time.perf_counter()
                try:
                    response = client.post(f"{flat.READER_ENDPOINT}/chat/completions", json=req)
                    return (
                        response.status_code,
                        response.content,
                        response.headers.get("content-type"),
                    )
                finally:
                    provider_counter["last_latency_ms"] = round(
                        (time.perf_counter() - started) * 1000, 3
                    )

            provider_counter["last_latency_ms"] = None
            try:
                normalized, call = execute_or_resume(
                    request=request,
                    catalog=session["catalog"],
                    session_identity_sha256=identity,
                    prompt_sha256=prompt_sha,
                    contract_sha256=contract_sha,
                    local_cache_root=LOCAL_CACHE_ROOT,
                    provider=provider,
                    stage_identity="MEM-3A.2/minimal-flatprop-v3",
                    model_sha256=runtime["model_sha256"],
                    dynamic_schema_sha256=frozen["dynamic_schema_sha256"],
                    unwrap_source_sha256=unwrap_sha,
                    packet_validator=writer_v3.validate_packet,
                    packet_validator_sha256=validator_sha,
                )
            except (
                WriterQualificationFailure,
                WriterRecoveryError,
                OSError,
                httpx.HTTPError,
            ) as exc:
                error = exc.error if isinstance(exc, WriterQualificationFailure) else None
                failure_code = (
                    (error or {}).get("code")
                    if isinstance(exc, WriterQualificationFailure)
                    else str(exc).split(":", 1)[0]
                    if isinstance(exc, WriterRecoveryError)
                    else "LOCAL_RESPONSE_CAPTURE_FAILURE"
                )
                failed_ledger = (
                    dict(exc.ledger) if isinstance(exc, WriterQualificationFailure) else {}
                )
                failed_ledger.update(
                    {
                        "session_identity_sha256": identity,
                        "request_sha256": request_sha,
                        "success": False,
                        "failure_code": failure_code,
                        "provider": "local_llama_cpp",
                        "hosted_call": False,
                        "retry_count": 0,
                        "latency_ms": failed_ledger.get("provider_duration_ms"),
                    }
                )
                ledger_by_id[identity] = failed_ledger
                failure = {
                    "session_identity_sha256": identity,
                    "session_index": index,
                    "code": failure_code,
                    "detail": error or str(exc),
                }
                _write_jsonl(
                    RUN_DIR / "writer_call_ledger.partial.jsonl", list(ledger_by_id.values())
                )
                _write_json(
                    RUN_DIR / "run_manifest.json",
                    {
                        "schema_version": 1,
                        "stage": RUN_ID,
                        "status": "FAILED",
                        "base_commit_sha": BASE_COMMIT,
                        "completion_gate_marker": "MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO",
                        "expected_unique_source_sessions": 477,
                        "completed_unique_source_sessions": len(outcomes),
                        "provider_calls_this_process": provider_counter["calls"],
                        "retries": 0,
                        "hosted_calls": 0,
                        "failure": failure,
                        "labels_loaded": False,
                        "test_access": False,
                        "102_dev_access": False,
                    },
                )
                return [], list(ledger_by_id.values()), {"failure": failure}

            packet = _normalized_session_packet(session, normalized)
            outcomes[identity] = packet
            call.update(
                {
                    "session_identity_sha256": identity,
                    "request_sha256": request_sha,
                    "dynamic_schema_sha256": frozen["dynamic_schema_sha256"],
                    "writer_prompt_sha256": prompt_sha,
                    "validator_source_sha256": schema_source_sha,
                    "success": call.get("validation") == "passed",
                    "provider": "local_llama_cpp",
                    "endpoint": flat.READER_ENDPOINT,
                    "loopback_only": True,
                    "writer_role": "memory_write_extract",
                    "hosted_call": False,
                    "retry_count": 0,
                    "latency_ms": call.get("provider_duration_ms"),
                    "latency_measured_this_process": call.get("provider_calls_this_resume") == 1,
                    "prompt_tokens_preflight": frozen["prompt_tokens"],
                    "prompt_tokens_match_preflight": call.get("prompt_tokens")
                    == frozen["prompt_tokens"],
                    "completion_tokens": call.get("completion_tokens"),
                }
            )
            if not call["success"]:
                raise RuntimeError(
                    f"v3 packet validator returned non-success without failure: {identity}"
                )
            ledger_by_id[identity] = call
            if index % 5 == 0 or index == 477:
                print(
                    f"MEM3A.2 v3 writer {index}/477 propositions={sum(len(item['propositions']) for item in outcomes.values())} provider_calls={provider_counter['calls']}",
                    flush=True,
                )

    packets = [outcomes[identity] for identity in sessions]
    ledger = [ledger_by_id[identity] for identity in sessions]
    if len(packets) != 477 or len(ledger) != 477 or any(not row["success"] for row in ledger):
        raise RuntimeError("MEM3A.2 v3 writer outcomes are incomplete")
    manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "contract_id": "flat-proposition-extractor-v3-minimal",
        "status": "WRITER_COMPLETE",
        "unique_source_sessions": len(packets),
        "structurally_valid_outcomes": sum(row["success"] for row in ledger),
        "provider_calls_unique": sum(row.get("provider_calls", 0) for row in ledger),
        "provider_calls_this_process": provider_counter["calls"],
        "retries": sum(row.get("retry_count", 0) for row in ledger),
        "hosted_calls": sum(bool(row.get("hosted_call")) for row in ledger),
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
        "failure": None,
        "session_packets": [
            {
                "session_identity_sha256": packet["session_identity_sha256"],
                "source_session_ids": packet["source_session_ids"],
                "session_date": packet["session_date"],
                "catalog_sha256": packet["catalog_sha256"],
                "proposition_count": len(packet["propositions"]),
                "packet_sha256": sha256_bytes(canonical_json(packet)),
            }
            for packet in packets
        ],
    }
    _write_json(manifest_path, manifest)
    _write_jsonl(flat.EXTRACTIONS_PATH, packets)
    _write_jsonl(flat.WRITER_LEDGER_PATH, ledger)
    _write_json(
        RUN_DIR / "run_manifest.json",
        {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": "WRITER_COMPLETE",
            "base_commit_sha": BASE_COMMIT,
            "completion_gate_marker": "MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=PENDING",
            "expected_unique_source_sessions": 477,
            "completed_unique_source_sessions": len(packets),
            "provider_calls_unique": manifest["provider_calls_unique"],
            "provider_calls_this_process": provider_counter["calls"],
            "retries": manifest["retries"],
            "hosted_calls": manifest["hosted_calls"],
            "labels_loaded": False,
            "test_access": False,
            "102_dev_access": False,
        },
    )
    return packets, ledger, manifest


def _materialize(
    packets: list[dict[str, Any]], refs: dict[str, dict[str, str]], contract_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    packet_by_id = {packet["session_identity_sha256"]: packet for packet in packets}
    rows: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    records_by_question: dict[str, dict[str, dict[str, Any]]] = {
        qid: {} for qid in flat.QUESTION_IDS
    }
    for qid in flat.QUESTION_IDS:
        scope = f"longmemeval:{qid}"
        store = flat.InMemoryMemoryStore()
        for identity, source_sid in refs[qid].items():
            packet = packet_by_id[identity]
            for proposition in packet["propositions"]:
                evidence = proposition["evidence"]
                provenance = {"evidence_refs": proposition["evidence_refs"], "evidence": evidence}
                provenance_sha = sha256_bytes(canonical_json(provenance))
                text_sha = sha256_bytes(proposition["proposition_text"].encode("utf-8"))
                id_body = {
                    "extractor_v3_contract_sha256": contract_sha,
                    "session_identity_sha256": identity,
                    "proposition_index": proposition["proposition_index"],
                    "proposition_text_sha256": text_sha,
                    "evidence_refs_provenance_sha256": provenance_sha,
                }
                digest = sha256_bytes(canonical_json(id_body))
                memory_id = f"m10flat3-{digest}"
                value = {
                    "proposition": proposition["proposition_text"],
                    "observed_at": packet["valid_from"],
                    "source_authority": proposition["source_authority"],
                }
                op = flat.MemoryOperation.add(
                    scope_id=scope,
                    key=f"flat_proposition:{digest}",
                    kind=flat.MemoryKind.SESSION_NOTE,
                    value=value,
                    source_type=flat.MemorySourceType.SESSION_DERIVED,
                    sensitivity=flat.MemorySensitivity.NON_SENSITIVE,
                    memory_id=memory_id,
                    valid_from=packet["valid_from"],
                    valid_until=None,
                    expires_at=None,
                    supersedes_id=None,
                    source_session_id=source_sid,
                    source_event_ids=(f"longmemeval-flat-proposition-v3-{digest}",),
                )
                if op.operation != flat.MemoryOperationType.ADD:
                    raise RuntimeError("MEM-3A.2 v3 permits ADD only")
                record = store.apply(op, now=packet["valid_from"])
                if (
                    record is None
                    or record.status != flat.MemoryStatus.ACTIVE
                    or record.version != 1
                    or record.supersedes_id is not None
                ):
                    raise RuntimeError("v3 FlatProp violated ACTIVE version-1 invariant")
                row = {
                    "question_id": qid,
                    "scope_id": scope,
                    "memory_id": memory_id,
                    "key": f"flat_proposition:{digest}",
                    "session_identity_sha256": identity,
                    "source_session_id": source_sid,
                    "session_date": packet["session_date"],
                    "valid_from": packet["valid_from"],
                    "valid_until": None,
                    "expires_at": None,
                    "proposition_index": proposition["proposition_index"],
                    "proposition_text": proposition["proposition_text"],
                    "proposition_text_sha256": text_sha,
                    "source_authority": proposition["source_authority"],
                    "source_turn_indices": list(
                        dict.fromkeys(e["source_turn_index"] for e in evidence)
                    ),
                    "evidence_refs": proposition["evidence_refs"],
                    "evidence": evidence,
                    "provenance_sha256": provenance_sha,
                    "operation": "ADD",
                    "kind": "session_note",
                    "source_type": "session_derived",
                    "status": "active",
                    "version": 1,
                    "supersedes_id": None,
                    "reader_value": value,
                    "retrieval_document_sha256": text_sha,
                }
                rows.append(row)
                operations.append(
                    {
                        "question_id": qid,
                        "operation": "ADD",
                        "memory_id": memory_id,
                        "scope_id": scope,
                        "key": row["key"],
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
                        "provenance_sha256": provenance_sha,
                    }
                )
                records_by_question[qid][memory_id] = {"record": record, "row": row}
        if len(store.history(scope)) != len(records_by_question[qid]) or any(
            event.operation != flat.MemoryOperationType.ADD for event in store.history(scope)
        ):
            raise RuntimeError(f"Non-ADD history found in v3 scope {qid}")
        store.close()
    rows.sort(
        key=lambda row: (
            flat.QUESTION_IDS.index(row["question_id"]),
            row["session_identity_sha256"],
            row["proposition_index"],
        )
    )
    operations.sort(key=lambda row: (flat.QUESTION_IDS.index(row["question_id"]), row["memory_id"]))
    return rows, operations, records_by_question


def _mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(row[name])
        for row in rows
        if isinstance(row.get(name), (int, float)) and not isinstance(row.get(name), bool)
    ]
    return statistics.fmean(values) if values else None


def _metrics(
    predictions: list[dict[str, Any]],
    reader_ledger: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pred_by = {row["question_id"]: row for row in predictions}
    bundle_by = {row["question_id"]: row["context_bundle"] for row in bundles}
    plan_by = {row["question_id"]: row["plan"] for row in plans}
    row_by_id = {row["memory_id"]: row for row in flat_rows}
    ranked_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        ranked_by[row["question_id"]].append(row)
    per_question = []
    for qid in flat.QUESTION_IDS:
        label = labels[qid]
        answer_sessions = set(label.get("answer_session_ids") or [])
        candidates = ranked_by[qid]
        retrieval = flat._retrieval_session_metrics(candidates, answer_sessions)
        bundle = bundle_by[qid]
        selected_props = [row_by_id[item["memory_id"]] for item in bundle["items"]]
        projected = flat._retrieval_session_metrics(
            [
                {"rank": i, "source_session_id": prop["source_session_id"]}
                for i, prop in enumerate(selected_props, 1)
            ],
            answer_sessions,
        )
        context = bundle["serialized_context"]
        coverage, exact = flat.mem2a._memory_gold_coverage(label.get("answer", ""), context)
        answer_metrics = flat.mem2a._answer_metrics(
            pred_by[qid].get("predicted") or "", label.get("answer", "")
        )
        selected_turns = {
            (prop["source_session_id"], turn)
            for prop in selected_props
            for turn in prop["source_turn_indices"]
        }
        per_question.append(
            {
                "question_id": qid,
                "question_type": label["question_type"],
                "has_answer": label["has_answer"],
                "answer_session_ids_count": len(answer_sessions),
                "answer_session_recall_at_5": retrieval["answer_session_recall_at_5"],
                "answer_session_recall_at_8": retrieval["answer_session_recall_at_8"],
                "mrr": retrieval["mrr"],
                "projected_answer_session_recall_at_8": projected["answer_session_recall_at_8"],
                "projected_mrr": projected["mrr"],
                "unique_source_sessions_at_8": len(
                    {row["source_session_id"] for row in candidates}
                ),
                "unique_source_turns_at_8": len(
                    {
                        (row["source_session_id"], turn)
                        for row in candidates
                        for turn in row["source_turn_indices"]
                    }
                ),
                "selected_memory_count": len(selected_props),
                "estimated_memory_tokens": plan_by[qid]["memory_tokens"],
                "reader_visible_context_tokens": bundle["context_reader_tokens"],
                "content_only_reader_tokens": bundle["content_only_reader_tokens"],
                "metadata_overhead_ratio": bundle["metadata_overhead_ratio"],
                "gold_token_coverage": coverage,
                "exact_normalized_gold_sequence_present": exact,
                "projected_unique_source_sessions": len(
                    {row["source_session_id"] for row in selected_props}
                ),
                "projected_unique_source_turns": len(selected_turns),
                "token_precision": answer_metrics["token_precision"],
                "token_recall": answer_metrics["token_recall"],
                "token_f1": answer_metrics["f1"],
                "normalized_em": answer_metrics["normalized_exact_match"],
                "deterministic_abstention_accuracy": float(
                    flat.mem2a._is_deterministic_refusal(pred_by[qid].get("predicted") or "")
                )
                if label["has_answer"] is False
                else None,
                "retrieval_latency_ms": bundle["retrieval_latency_ms"],
                "reader_latency_ms": pred_by[qid]["reader_latency_ms"],
                "reader_prompt_tokens": pred_by[qid]["reader_prompt_tokens_server"],
                "reader_completion_tokens": pred_by[qid]["completion_tokens_server"],
            }
        )
    names = [
        "answer_session_recall_at_5",
        "answer_session_recall_at_8",
        "mrr",
        "projected_answer_session_recall_at_8",
        "projected_mrr",
        "unique_source_sessions_at_8",
        "unique_source_turns_at_8",
        "selected_memory_count",
        "estimated_memory_tokens",
        "reader_visible_context_tokens",
        "content_only_reader_tokens",
        "gold_token_coverage",
        "exact_normalized_gold_sequence_present",
        "token_precision",
        "token_recall",
        "token_f1",
        "normalized_em",
        "deterministic_abstention_accuracy",
        "retrieval_latency_ms",
        "reader_latency_ms",
        "reader_prompt_tokens",
        "reader_completion_tokens",
    ]
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_question:
        categories[row["question_type"]].append(row)
    return {
        "schema_version": 1,
        "scope": "ten frozen DEV diagnostic cases; labels joined after all prediction and reader artifacts froze",
        "system": "MEM-3A.2 Minimal FlatProp Dense",
        "means": {name: _mean(per_question, name) for name in names},
        "by_question_type": {
            category: {"n": len(rows), "means": {name: _mean(rows, name) for name in names}}
            for category, rows in sorted(categories.items())
        },
        "per_question": per_question,
        "answer_metrics": "deterministic token precision/recall/F1 and normalized EM; no LLM judge",
        "abstention_accuracy": "reported only for has_answer=false using the shared deterministic refusal helper",
        "judge_calls": 0,
        "descriptive_only_no_performance_ranking_claim": True,
    }


def _writer_diagnostics(
    packets: list[dict[str, Any]], ledger: list[dict[str, Any]], inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    props = [prop for packet in packets for prop in packet["propositions"]]
    authority = Counter(prop["source_authority"] for prop in props)
    selection = _json(OLD_SELECTION)
    if not flat._verify_frozen(OLD_SELECTION) or selection.get("labels_used") is not False:
        raise RuntimeError("Frozen v2 qualification identity selection failed verification")
    qualification_ids = {row["session_identity_sha256"] for row in selection["session_identities"]}
    if len(qualification_ids) != 12:
        raise RuntimeError("Expected the previously frozen twelve qualification identities")
    packet_by_id = {packet["session_identity_sha256"]: packet for packet in packets}
    if not qualification_ids.issubset(packet_by_id):
        raise RuntimeError("V3 diagnostic packets are missing frozen qualification identities")
    selected_props = [
        prop for identity in qualification_ids for prop in packet_by_id[identity]["propositions"]
    ]
    span_counts = [len(prop["evidence"]) for prop in props]
    turn_counts = [len({item["source_turn_index"] for item in prop["evidence"]}) for prop in props]
    transient = [prop for prop in selected_props if TRANSIENT_HINT.search(prop["proposition_text"])]
    return {
        "source_sessions": len(packets),
        "sessions_with_zero_propositions": sum(not packet["propositions"] for packet in packets),
        "total_propositions": len(props),
        "propositions_per_session": len(props) / max(1, len(packets)),
        "rawspan_count": len(inventory),
        "flatprop_count_scoped": len(props),
        "rawspan_to_flatprop_ratio": len(inventory) / max(1, len(props)),
        "writer_input_tokens": sum(row["prompt_tokens"] or 0 for row in ledger),
        "writer_completion_tokens": sum(row["completion_tokens"] or 0 for row in ledger),
        "writer_latency_ms_total_measured_this_process": sum(
            row["latency_ms"] or 0 for row in ledger
        ),
        "writer_latency_ms_mean_measured_this_process": statistics.fmean(
            [row["latency_ms"] for row in ledger if row.get("latency_ms") is not None]
        )
        if any(row.get("latency_ms") is not None for row in ledger)
        else None,
        "authority_counts": {
            "user_only_propositions": authority.get("user", 0),
            "assistant_only_propositions": authority.get("assistant", 0),
            "mixed_source_propositions": authority.get("mixed", 0),
        },
        "mean_evidence_refs_per_proposition": statistics.fmean(span_counts) if span_counts else 0.0,
        "multi_span_propositions": sum(count > 1 for count in span_counts),
        "multi_turn_propositions": sum(count > 1 for count in turn_counts),
        "frozen_12_identity_diagnostics": {
            "identity_count": len(qualification_ids),
            "proposition_count": len(selected_props),
            "apparent_transient_request_propositions": len(transient),
            "apparent_transient_request_rate": len(transient) / max(1, len(selected_props)),
            "transient_classifier": "heuristic keyword patterns over proposition text; diagnostic only, no deletion or gate",
            "assistant_origin_propositions": sum(
                prop["source_authority"] == "assistant" for prop in selected_props
            ),
            "user_origin_propositions": sum(
                prop["source_authority"] == "user" for prop in selected_props
            ),
            "mixed_origin_propositions": sum(
                prop["source_authority"] == "mixed" for prop in selected_props
            ),
        },
        "semantic_quality_used_as_gate": False,
        "revision_keys_extracted": False,
    }


def _case_review(
    flat_rows: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    prop_by_id = {row["memory_id"]: row for row in flat_rows}
    rank_by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        rank_by_q[row["question_id"]].append(row)
    pred_by = {row["question_id"]: row for row in predictions}
    cases = []
    for qid in EXPECTED_FOCUS:
        if qid not in questions or qid not in labels:
            raise RuntimeError(f"Required critical case is absent: {qid}")
        retrieved = []
        for candidate in rank_by_q[qid]:
            prop = prop_by_id[candidate["memory_id"]]
            retrieved.append(
                {
                    "rank": candidate["rank"],
                    "cosine_similarity": candidate["cosine_similarity"],
                    "memory_id": prop["memory_id"],
                    "proposition_text": prop["proposition_text"],
                    "observed_at": prop["valid_from"],
                    "source_authority": prop["source_authority"],
                    "source_session_id": prop["source_session_id"],
                    "exact_harness_evidence": prop["evidence"],
                }
            )
        cases.append(
            {
                "question_id": qid,
                "question": questions[qid]["question"],
                "question_date": questions[qid]["question_date"],
                "retrieved_top8": retrieved,
                "prediction": pred_by[qid]["predicted"],
                "gold_answer": labels[qid]["answer"],
                "answer_session_ids": labels[qid]["answer_session_ids"],
            }
        )
    instagram = []
    for row in flat_rows:
        if row["question_id"] != "1cea1afa":
            continue
        combined = " ".join(
            [row["proposition_text"], *(item["evidence_quote"] for item in row["evidence"])]
        ).casefold()
        for count in (500, 600):
            if "instagram" in combined and re.search(rf"\b{count}\b", combined):
                instagram.append(
                    {
                        "expected_observation": f"{count} Instagram followers",
                        "captured_in_proposition": bool(
                            "instagram" in row["proposition_text"].casefold()
                            and re.search(rf"\b{count}\b", row["proposition_text"])
                        ),
                        "proposition_text": row["proposition_text"],
                        "observed_at": row["valid_from"],
                        "source_authority": row["source_authority"],
                        "source_session_id": row["source_session_id"],
                        "memory_id": row["memory_id"],
                        "exact_harness_evidence": row["evidence"],
                        "retrieved_top8": any(
                            item["memory_id"] == row["memory_id"] for item in rank_by_q["1cea1afa"]
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "no_revision_semantics": True,
        "case_ids": list(EXPECTED_FOCUS),
        "cases": cases,
        "instagram_500_to_600_checkpoint": {
            "detected_observations": instagram,
            "has_500": any(row["expected_observation"].startswith("500 ") for row in instagram),
            "has_600": any(row["expected_observation"].startswith("600 ") for row in instagram),
            "resolved_or_suppressed": False,
        },
        "provenance_source": "Harness reconstructed from frozen RawSpan catalog; model emitted evidence refs only.",
    }


def _comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    baseline_path = flat.MEM2D_DIR / "deterministic_metrics.json"
    efficiency_path = flat.MEM2D_DIR / "efficiency.json"
    if not mem2d._verify_sidecar(baseline_path) or not mem2d._verify_sidecar(efficiency_path):
        raise RuntimeError("Frozen MEM-2D Dense baseline artifact SHA failed")
    baseline = _json(baseline_path)["arms"]["dense"]["per_question"]
    base_by_q = {row["question_id"]: row for row in baseline}
    candidate_by_q = {row["question_id"]: row for row in metrics["per_question"]}
    per_question = []
    for qid in flat.QUESTION_IDS:
        old, new = base_by_q[qid], candidate_by_q[qid]
        per_question.append(
            {
                "question_id": qid,
                "rawspan_dense": old,
                "minimal_flatprop_dense": new,
                "delta_token_f1": new["token_f1"] - old["token_f1"],
                "delta_reader_visible_context_tokens": new["reader_visible_context_tokens"]
                - old["context_reader_tokens"],
            }
        )
    return {
        "schema_version": 1,
        "baseline": "MEM-2D RawSpan + Dense",
        "candidate": "MEM-3A.2 Minimal FlatProp + Dense",
        "fixed": [
            "same ten frozen DEV questions and question dates",
            "Qwen/Qwen3-Embedding-0.6B and frozen query instruction",
            "Dense cosine descending; memory_id ascending tie-break; top_k=8",
            "question scope and validity filtering",
            "m10-rank-aware-projection-v1 with 1024-token memory budget",
            "same final reader contract and local Qwen3-8B reader",
        ],
        "changed_component": "memory representation and write-time semantic compression only",
        "per_question": per_question,
        "interpretation": "Descriptive frozen-ten diagnostic only; no ranking, superiority claim, or statistical inference.",
        "labels_used_after_freeze": True,
        "judge_calls": 0,
    }


def _efficiency(
    packets: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    embedding: dict[str, Any],
    bundles: list[dict[str, Any]],
    reader_ledger: list[dict[str, Any]],
    writer_diag: dict[str, Any],
) -> dict[str, Any]:
    latencies = [row["retrieval_latency_ms"] for row in bundles]
    return {
        "schema_version": 1,
        "representation": writer_diag,
        "writer": {
            "outcomes": len(ledger),
            "provider_calls_unique": sum(row.get("provider_calls", 0) for row in ledger),
            "input_tokens": writer_diag["writer_input_tokens"],
            "completion_tokens": writer_diag["writer_completion_tokens"],
            "latency_ms_total_measured_this_process": writer_diag[
                "writer_latency_ms_total_measured_this_process"
            ],
            "latency_ms_mean_measured_this_process": writer_diag[
                "writer_latency_ms_mean_measured_this_process"
            ],
            "hosted_calls": sum(bool(row["hosted_call"]) for row in ledger),
            "failures": sum(not row["success"] for row in ledger),
        },
        "embedding": {
            "model_id": embedding["model_id"],
            "revision": embedding["revision"],
            "dimensions": embedding["dimensions"],
            "device": embedding["device"],
            "dtype": embedding["dtype"],
            "documents": embedding["document_count"],
            "unique_documents": embedding["unique_document_count"],
            "document_tokens": embedding["document_input_tokens"],
            "query_tokens": embedding["query_input_tokens"],
            "truncations": embedding["document_truncations"] + embedding["query_truncations"],
            "cache": embedding["cache"],
            "local_only": embedding["local_only"],
            "hosted_calls": embedding["hosted_calls"],
        },
        "retrieval": {
            "p50_latency_ms": statistics.median(latencies) if latencies else None,
            "p95_latency_ms": flat.np_percentile(latencies, 0.95) if latencies else None,
            "by_question_latency_ms": {
                row["question_id"]: row["retrieval_latency_ms"] for row in bundles
            },
        },
        "projection": {
            "memory_budget_tokens": flat.MEMORY_BUDGET,
            "reader_visible_context_tokens_by_question": {
                row["question_id"]: row["context_bundle"]["context_reader_tokens"]
                for row in bundles
            },
            "mean_reader_visible_context_tokens": statistics.fmean(
                row["context_bundle"]["context_reader_tokens"] for row in bundles
            ),
            "mean_content_only_tokens": statistics.fmean(
                row["context_bundle"]["content_only_reader_tokens"] for row in bundles
            ),
            "mean_metadata_overhead_ratio": statistics.fmean(
                row["context_bundle"]["metadata_overhead_ratio"] for row in bundles
            ),
        },
        "reader": {
            "calls": len(reader_ledger),
            "mean_latency_ms": statistics.fmean(row["latency_ms"] for row in reader_ledger),
            "prompt_tokens_total": sum(row["prompt_tokens_server"] for row in reader_ledger),
            "completion_tokens_total": sum(
                row["completion_tokens_server"] or 0 for row in reader_ledger
            ),
            "hosted_calls": sum(bool(row["hosted_call"]) for row in reader_ledger),
        },
        "no_quality_cost_winner_claim": True,
    }


def _render_report(
    gate: str,
    writer_diag: dict[str, Any],
    embedding: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    means = metrics["means"]
    lines = [
        "# MEM-3A.2 - Minimal FlatProp v3 Frozen-Ten Diagnostic",
        "",
        f"Completion gate: `{gate}`.",
        "",
        "Controlled descriptive comparison: MEM-2D RawSpan + Dense versus minimal FlatProp + Dense. The only intended change is memory representation/write-time semantic compression. No benchmark ranking or broad performance claim is made from ten DEV questions.",
        "",
        "## Writer and Materialization",
        "",
        f"- Writer outcomes: {manifest['writer_outcomes']}/477; unique local calls: {manifest['writer_provider_calls_unique']}; retries: {manifest['retries']}; hosted calls: {manifest['hosted_calls']}.",
        f"- Source sessions: {writer_diag['source_sessions']}; zero-proposition sessions: {writer_diag['sessions_with_zero_propositions']}; FlatProps: {writer_diag['total_propositions']} ({writer_diag['propositions_per_session']:.3f}/session).",
        f"- RawSpan inventory rows: {writer_diag['rawspan_count']}; scoped FlatProps: {writer_diag['flatprop_count_scoped']}; RawSpan/FlatProp ratio: {writer_diag['rawspan_to_flatprop_ratio']:.3f}.",
        f"- Authority counts: user {writer_diag['authority_counts']['user_only_propositions']}, assistant {writer_diag['authority_counts']['assistant_only_propositions']}, mixed {writer_diag['authority_counts']['mixed_source_propositions']}.",
        f"- Mean evidence refs/proposition: {writer_diag['mean_evidence_refs_per_proposition']:.3f}; multi-span: {writer_diag['multi_span_propositions']}; multi-turn: {writer_diag['multi_turn_propositions']}.",
        f"- Writer tokens: {writer_diag['writer_input_tokens']} input / {writer_diag['writer_completion_tokens']} completion; local latency mean: {writer_diag['writer_latency_ms_mean_measured_this_process']} ms.",
        "- Every reader-visible memory value carries proposition, observed_at, and Harness-derived source_authority. All materialized operations are ADD, ACTIVE, version 1, with no supersession.",
        "",
        "## Retrieval and Reader",
        "",
        f"- Embedding: `{embedding['model_id']}` revision `{embedding['revision']}`, {embedding['dimensions']} dimensions, {embedding['device']} / {embedding['dtype']}; hosted calls {embedding['hosted_calls']}; truncations {embedding['document_truncations'] + embedding['query_truncations']}.",
        f"- Top-8 Dense retrieval rows: {manifest['dense_rows']}; rank-aware projection budget: {flat.MEMORY_BUDGET}; successful shared-reader calls: {manifest['reader_calls']}.",
        f"- Mean answer-session Recall@5 / @8 / MRR: {means['answer_session_recall_at_5']} / {means['answer_session_recall_at_8']} / {means['mrr']}.",
        f"- Mean projected answer-session Recall@8: {means['projected_answer_session_recall_at_8']}; mean reader-visible context tokens: {means['reader_visible_context_tokens']}.",
        f"- Deterministic token precision / recall / F1 / normalized EM: {means['token_precision']} / {means['token_recall']} / {means['token_f1']} / {means['normalized_em']}.",
        "",
        "## RawSpan-Dense Comparison",
        "",
        "Per-question deltas and complete case details are in `comparison_mem2d_dense_vs_mem3a2_flat.json` and `mem_3a2_case_review.json`. The comparison holds the ten questions, dates, embedding, Dense top-8/tie-break, projection, budget, and reader fixed.",
        f"- Mean token-F1 delta: {statistics.fmean(row['delta_token_f1'] for row in comparison['per_question']):.4f}.",
        f"- Mean reader-visible context-token delta: {statistics.fmean(row['delta_reader_visible_context_tokens'] for row in comparison['per_question']):.1f}.",
        "- These are diagnostic observations, not evidence of superiority or generalization.",
        "",
        "## Gate and Scope",
        "",
        f"- `MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC={gate}`.",
        "- Test access: false; 102-case DEV: not run; judge calls: zero; hosted calls: zero.",
        "- No revision keys, revision materialization, or state reconciliation were run. The Instagram 500/600 checkpoint reports capture status only and does not resolve either value.",
        "",
    ]
    return "\n".join(lines)


def _downstream(
    upstream: dict[str, Any],
    preflight: dict[str, Any],
    packets: list[dict[str, Any]],
    writer_ledger: list[dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
    refs: dict[str, dict[str, str]],
    inventory: list[dict[str, Any]],
    writer_manifest: dict[str, Any],
) -> dict[str, Any]:
    if len(packets) != 477 or len(writer_ledger) != 477:
        raise RuntimeError("MEM-3A.2 downstream requires 477 frozen v3 writer outcomes")
    contract_sha = preflight["extractor_contract_sha256"]
    rows, operations, records = _materialize(packets, refs, contract_sha)
    _write_jsonl(flat.FLAT_PROPOSITIONS_PATH, rows)
    _write_jsonl(flat.MATERIALIZATION_LEDGER_PATH, operations)
    embedding_started = time.perf_counter()
    ranked, embedding, query_tokens, retrieval_latency = flat._embedding_and_retrieval(
        rows, records, refs, preflight
    )
    embedding["query_token_counts"] = query_tokens
    embedding["representation"] = "minimal_flatprop_v3_proposition_text_only"
    embedding["retrieval_algorithm"] = (
        "scope/time-valid Dense cosine; top_k=8; memory_id ascending tie-break"
    )
    _write_json(flat.EMBEDDING_MANIFEST_PATH, embedding)
    m2d_inputs = mem2d._load_frozen_inputs()
    questions = m2d_inputs["questions"]
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        reader_contract, reader_sha, system_template, user_template = (
            flat.load_final_reader_contract()
        )
        if (
            reader_sha != flat.PINNED_READER_SHA256
            or reader_sha != upstream["reader_contract_sha256"]
        ):
            raise RuntimeError("Frozen final reader contract changed")
        plans, bundles, _ = flat._project_contexts(
            ranked,
            records,
            questions,
            client,
            (reader_contract, reader_sha, system_template, user_template),
            retrieval_latency,
        )
    _write_jsonl(flat.DENSE_TOP8_PATH, ranked)
    _write_jsonl(flat.CONTEXT_PLANS_PATH, plans)
    _write_jsonl(flat.CONTEXT_BUNDLES_PATH, bundles)
    pre_reader_paths = [
        flat.PREFLIGHT_PATH,
        RUN_DIR / "writer_extraction_identity.json",
        RUN_DIR / "session_extraction_manifest.json",
        flat.EXTRACTIONS_PATH,
        flat.WRITER_LEDGER_PATH,
        flat.FLAT_PROPOSITIONS_PATH,
        flat.MATERIALIZATION_LEDGER_PATH,
        flat.EMBEDDING_MANIFEST_PATH,
        flat.DENSE_TOP8_PATH,
        flat.CONTEXT_PLANS_PATH,
        flat.CONTEXT_BUNDLES_PATH,
    ]
    if len(ranked) != 80 or not all(flat._verify_frozen(path) for path in pre_reader_paths):
        raise RuntimeError("v3 pre-reader artifacts did not freeze or Dense top-8 is incomplete")
    pre_reader_sha = sha256_bytes(
        canonical_json({path.name: sha256_file(path) for path in pre_reader_paths})
    )
    preflight_for_reader = dict(preflight)
    preflight_for_reader["writer_runtime_identity"] = preflight["writer_runtime_identity"]
    predictions, reader_ledger = flat._run_reader(pre_reader_sha, preflight_for_reader, bundles)
    predictions = [{**row, "system": "mem3a2_minimal_flatprop_dense"} for row in predictions]
    _write_jsonl(flat.PREDICTIONS_PATH, predictions)
    _write_jsonl(flat.READER_LEDGER_PATH, reader_ledger)
    if not all(
        flat._verify_frozen(path) for path in (flat.PREDICTIONS_PATH, flat.READER_LEDGER_PATH)
    ):
        raise RuntimeError("Prediction and reader ledger failed SHA freeze")

    # Gold answers and answer-session IDs enter only after every prediction/call and context input froze.
    labels = flat._labels_after_freeze()
    metrics = _metrics(predictions, reader_ledger, bundles, plans, ranked, rows, labels)
    _write_json(flat.METRICS_PATH, metrics)
    writer_diag = _writer_diagnostics(packets, writer_ledger, inventory)
    comparison = _comparison(metrics)
    _write_json(flat.COMPARISON_PATH, comparison)
    efficiency = _efficiency(
        packets, rows, writer_ledger, embedding, bundles, reader_ledger, writer_diag
    )
    _write_json(flat.EFFICIENCY_PATH, efficiency)
    case_review = _case_review(rows, ranked, predictions, labels, questions)
    _write_json(flat.CASE_REVIEW_PATH, case_review)

    flat_no_revision = all(
        row["operation"] == "ADD"
        and row["status"] == "active"
        and row["version"] == 1
        and row["supersedes_id"] is None
        and row["valid_until"] is None
        and row["expires_at"] is None
        for row in operations
    )
    provenance_valid = all(
        prop["source_authority"] in {"user", "assistant", "mixed"}
        and all(
            set(item)
            == {
                "evidence_ref",
                "source_turn_index",
                "source_span_index",
                "source_role",
                "char_start",
                "char_end",
                "evidence_quote",
                "content_sha256",
            }
            for item in prop["evidence"]
        )
        for packet in packets
        for prop in packet["propositions"]
    )
    report_gate = {
        "upstream_gates_verified": True,
        "v3_contract_and_prompt_frozen_before_writer_calls": flat._verify_frozen(PROMPT_PATH)
        and flat._verify_frozen(CONTRACT_PATH),
        "477_source_sessions_preflighted": len(preflight["requests"]) == 477
        and preflight["truncated_requests"] == 0,
        "477_structurally_valid_writer_outcomes": len(packets) == 477
        and all(row["success"] for row in writer_ledger),
        "writer_calls_one_per_identity_zero_retries": writer_manifest["provider_calls_unique"]
        == 477
        and writer_manifest["retries"] == 0,
        "hosted_calls_zero": writer_manifest["hosted_calls"] == 0
        and all(not row["hosted_call"] for row in writer_ledger + reader_ledger),
        "all_provenance_harness_derived": provenance_valid,
        "flat_add_active_v1_no_revision": flat_no_revision,
        "frozen_qwen3_embedding_local_no_truncation": embedding["local_only"]
        and embedding["hosted_calls"] == 0
        and embedding["model_id"] == "Qwen/Qwen3-Embedding-0.6B"
        and str(embedding["device"]).startswith("cuda")
        and embedding["dtype"] == "float16"
        and embedding["document_truncations"] == 0
        and embedding["query_truncations"] == 0,
        "dense_top8_all_ten": len(ranked) == 80,
        "rank_aware_projection_sha_unchanged": flat.PINNED_PROJECTION_SHA256
        == sha256_file(flat.PROJECTION_CONTRACT_PATH),
        "memory_budget_1024": flat.MEMORY_BUDGET == 1024,
        "ten_shared_reader_calls_successful": len(reader_ledger) == 10
        and all(row["success"] and row["quality_status"] == "OK" for row in reader_ledger),
        "judge_calls_zero": True,
        "test_access_false": True,
        "102_case_dev_not_run": True,
        "prediction_and_reader_sha_frozen_before_labels": True,
        "critical_case_packet_complete": len(case_review["cases"]) == 8,
        "all_compact_artifact_sha_sidecars_valid": True,
    }
    gate = "YES" if all(report_gate.values()) else "NO"
    manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "COMPLETE" if gate == "YES" else "FAILED",
        "base_commit_sha": BASE_COMMIT,
        "question_ids": list(flat.QUESTION_IDS),
        "reader_model_role": "reader_answer; local Qwen3-8B Q4_K_M",
        "memory_internal_llm_role": "memory_ingest; same frozen local Qwen3-8B",
        "embedding_model_role": "Qwen/Qwen3-Embedding-0.6B local CUDA FP16",
        "judge_model": None,
        "contract_id": "flat-proposition-extractor-v3-minimal",
        "writer_outcomes": len(packets),
        "writer_provider_calls_unique": writer_manifest["provider_calls_unique"],
        "writer_calls_this_process": writer_manifest["provider_calls_this_process"],
        "writer_failures": sum(not row["success"] for row in writer_ledger),
        "retries": writer_manifest["retries"],
        "reader_calls": len(reader_ledger),
        "hosted_calls": writer_manifest["hosted_calls"]
        + sum(bool(row["hosted_call"]) for row in reader_ledger),
        "judge_calls": 0,
        "dense_rows": len(ranked),
        "memory_operations": len(operations),
        "memory_records": len(rows),
        "labels_loaded_after_freeze": True,
        "test_access": False,
        "102_dev_access": False,
        "completion_gate_marker": f"MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC={gate}",
        "gate": report_gate,
        "upstream": upstream,
        "writer_extraction_manifest_sha256": sha256_file(flat.EXTRACTION_MANIFEST_PATH),
        "pre_reader_freeze_sha256": pre_reader_sha,
        "embedding_and_projection_wall_seconds": round(time.perf_counter() - embedding_started, 3),
        "artifact_sha256": {},
    }
    report = _render_report(gate, writer_diag, embedding, metrics, comparison, manifest)
    # The report is plain Markdown so it remains easy to inspect alongside the JSON artifacts.
    atomic_write_bytes(flat.RUN_REPORT_PATH, report.encode("utf-8"))
    flat._freeze(flat.RUN_REPORT_PATH)
    artifact_paths = [
        flat.PREFLIGHT_PATH,
        RUN_DIR / "writer_extraction_identity.json",
        flat.EXTRACTION_MANIFEST_PATH,
        flat.EXTRACTIONS_PATH,
        flat.FLAT_PROPOSITIONS_PATH,
        flat.WRITER_LEDGER_PATH,
        flat.MATERIALIZATION_LEDGER_PATH,
        flat.EMBEDDING_MANIFEST_PATH,
        flat.DENSE_TOP8_PATH,
        flat.CONTEXT_PLANS_PATH,
        flat.CONTEXT_BUNDLES_PATH,
        flat.PREDICTIONS_PATH,
        flat.READER_LEDGER_PATH,
        flat.METRICS_PATH,
        flat.EFFICIENCY_PATH,
        flat.COMPARISON_PATH,
        flat.CASE_REVIEW_PATH,
        flat.RUN_REPORT_PATH,
        PROMPT_PATH,
        CONTRACT_PATH,
    ]
    manifest["artifact_sha256"] = {path.name: sha256_file(path) for path in artifact_paths}
    manifest["gate"]["all_compact_artifact_sha_sidecars_valid"] = all(
        flat._verify_frozen(path) for path in artifact_paths
    )
    passed = all(manifest["gate"].values())
    manifest["completion_gate_marker"] = (
        "MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=YES"
        if passed
        else "MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO"
    )
    manifest["status"] = "COMPLETE" if passed else "FAILED"
    _write_json(flat.RUN_MANIFEST_PATH, manifest)
    if not passed:
        raise RuntimeError("MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO")
    print(manifest["completion_gate_marker"], flush=True)
    return manifest


def run(*, preflight_only: bool) -> dict[str, Any]:
    _configure_flat_paths()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if preflight_only:
        preflight = _preflight(save=True)
        print(f"writer_preflight={flat.PREFLIGHT_PATH}", flush=True)
        print(f"preflighted_sessions={len(preflight['requests'])}", flush=True)
        print("MEM3A2_V3_WRITER_PREFLIGHT=YES", flush=True)
        return preflight
    preflight = _preflight(save=False)
    upstream = _verify_upstream()
    sessions, refs, inventory = _source_sessions_with_catalog()
    packets, writer_ledger, writer_manifest = _extract_all(preflight, sessions)
    if writer_manifest.get("failure") or len(packets) != 477:
        raise RuntimeError("MEM3A2_MINIMAL_FLATPROP_WRITER=NO; downstream not started")
    return _downstream(
        upstream,
        preflight,
        packets,
        writer_ledger,
        sessions,
        refs,
        inventory,
        writer_manifest,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        run(preflight_only=args.preflight_only)
        return 0
    except Exception as exc:  # Freeze visible failure state; do not retry model requests.
        _configure_flat_paths()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        _write_json(
            RUN_DIR / "run_failure.json",
            {
                "stage": RUN_ID,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completion_gate_marker": "MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO",
                "hosted_calls": 0,
                "judge_calls": 0,
                "test_access": False,
                "102_dev_access": False,
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
