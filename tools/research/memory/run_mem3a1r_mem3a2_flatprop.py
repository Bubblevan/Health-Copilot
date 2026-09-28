"""Repair MEM-3A.1 transport, then run the frozen 477-session FlatProp diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
import time
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

import run_mem3a1_extractor_v2_qualification as v2
import run_mem3a_flat_proposition as flat
from flat_proposition_writer_v2 import (
    WriterQualificationFailure,
    WriterRecoveryError,
    atomic_write_bytes,
    canonical_json,
    catalog_sha256,
    execute_or_resume,
    sha256_bytes,
    writer_request,
)
from mem1_artifacts import read_jsonl

BASE_COMMIT = "3d04255bae39cb51ab80a8837f7335c2b408748e"
PROMPT_SHA256 = "0aea4338d5de1e8dac753e3aae98caf36461cc15b1cf8d782a36f298f63cf627"
CONTRACT_SHA256 = "fee5ae49c13d7b55d75c29af38146c2a9e079ac7ca24e5aeff151dee7a31b666"
HISTORICAL_IDENTITY = "25267ea359060788693f6abf37b1731af749145149afb1927a3f39f538ff19a1"
QUALIFICATION_RUN_ID = "mem3a1r-extractor-v2-repaired-20260928"
FLATPROP_RUN_ID = "mem3a2-flat-proposition-10-20260928"
QUALIFICATION_RUN_DIR = ROOT / "runs/memory/mem3" / QUALIFICATION_RUN_ID
FLATPROP_RUN_DIR = ROOT / "runs/memory/mem3" / FLATPROP_RUN_ID
LEGACY_RUN_DIR = ROOT / "runs/memory/mem3/mem3a1-extractor-v2-qualification-20260928"
LOCAL_CACHE_ROOT = ROOT / ".cache/health-copilot/mem3a1r-writer"
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    atomic_write_bytes(path, payload)
    return v2.write_sidecar(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(canonical_json(row) + b"\n" for row in rows)
    atomic_write_bytes(path, payload)
    return v2.write_sidecar(path)


def _gate_manifest(path: Path, marker: str) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("completion_gate_marker") != marker:
        raise RuntimeError(f"Required historical gate is not {marker}: {path}")
    return manifest


def _verify_base_and_upstream() -> dict[str, Any]:
    head = (
        __import__("subprocess")
        .run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True)
        .stdout.strip()
    )
    if head != BASE_COMMIT:
        raise RuntimeError(f"MEM-3A.1R requires base {BASE_COMMIT}; found {head}")
    historical = _gate_manifest(
        LEGACY_RUN_DIR / "run_manifest.json", "MEM3A1_EXTRACTOR_V2_QUALIFIED=NO"
    )
    if historical.get("successful_packets") != 0:
        raise RuntimeError("Historical MEM-3A.1 evidence no longer matches the failed gate")
    v2.PINNED_BASE_COMMIT = BASE_COMMIT
    upstream = v2._verify_upstream()
    if upstream["mem2c_inventory_sha256"] != flat.PINNED_MEM2C_INVENTORY_SHA256:
        raise RuntimeError("MEM-2C RawSpan identity differs between frozen adapters")
    prompt_sha = v2.sha256_file(v2.PROMPT_PATH)
    contract_sha = v2.sha256_file(v2.CONTRACT_PATH)
    if prompt_sha != PROMPT_SHA256 or contract_sha != CONTRACT_SHA256:
        raise RuntimeError("Extractor-v2 prompt or contract changed")
    if (
        v2.verify_sidecar(v2.PROMPT_PATH) != prompt_sha
        or v2.verify_sidecar(v2.CONTRACT_PATH) != contract_sha
    ):
        raise RuntimeError("Extractor-v2 prompt/contract SHA sidecar is invalid")
    return {**upstream, "historical_mem3a1_gate": historical["completion_gate_marker"]}


def _historical_capture(session: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    ledger_path = LEGACY_RUN_DIR / "writer_call_ledger.jsonl"
    v2.verify_sidecar(ledger_path)
    ledger_rows = read_jsonl(ledger_path)
    matches = [
        row for row in ledger_rows if row.get("session_identity_sha256") == HISTORICAL_IDENTITY
    ]
    if len(matches) != 1:
        raise RuntimeError("Historical failed response ledger identity is not unique")
    old = matches[0]
    request_sha = sha256_bytes(canonical_json(request))
    old_identity = sha256_bytes(
        canonical_json(
            {
                "stage": "MEM-3A.1",
                "contract_sha256": CONTRACT_SHA256,
                "prompt_sha256": PROMPT_SHA256,
                "session_identity_sha256": HISTORICAL_IDENTITY,
                "catalog_sha256": catalog_sha256(session["catalog"]),
                "request_sha256": request_sha,
            }
        )
    )
    if (
        old.get("session_identity_sha256") != HISTORICAL_IDENTITY
        or old.get("request_sha256") != request_sha
        or old.get("cache_identity_sha256") != old_identity
        or old.get("http_status") != 200
    ):
        raise RuntimeError(
            "Historical response identity/request/status does not match the committed ledger"
        )
    raw_path = Path(old["response_cache_path"]).resolve()
    expected_root = (ROOT / ".cache/health-copilot/mem3a1-writer").resolve()
    if expected_root not in raw_path.parents or raw_path.name != "raw_response.txt":
        raise RuntimeError("Historical response is not under the expected local ignored cache")
    raw_body = raw_path.read_bytes()
    envelope_sha = sha256_bytes(raw_body)
    meta_path = raw_path.parent / "response_meta.json"
    metadata = _read_json(meta_path)
    if (
        envelope_sha != old.get("response_sha256")
        or envelope_sha != metadata.get("response_sha256")
        or metadata.get("http_status") != old.get("http_status")
    ):
        raise RuntimeError("Historical response bytes/status do not match old ledger and metadata")
    return {
        "http_status": 200,
        "raw_http_body": raw_body,
        "content_type": metadata.get("content_type"),
        "imported_from_stage": "mem3a1-extractor-v2-qualification-20260928",
        "import_reason": "HARNESS_RESPONSE_UNWRAPPING_DEFECT",
        "old_cache_identity_sha256": old_identity,
        "old_request_sha256": request_sha,
        "old_http_envelope_sha256": envelope_sha,
        "old_response_cache_path": str(raw_path),
    }


def _freeze_or_verify_protocol(identity: dict[str, Any]) -> str:
    path = FLATPROP_RUN_DIR / "extraction_identity.json"
    if path.exists():
        v2.verify_sidecar(path)
        if _read_json(path) != identity:
            raise RuntimeError("Existing MEM-3A.2 extraction identity differs; refusing a new call")
        return v2.verify_sidecar(path)
    return _write_json(path, identity)


def _writer_identity(
    session: dict[str, Any], prompt: str, model_alias: str
) -> tuple[dict[str, Any], str, str]:
    request = writer_request(
        session_date=session["session_date"],
        catalog=session["catalog"],
        system_prompt=prompt,
        model_alias=model_alias,
    )
    schema_sha = sha256_bytes(canonical_json(request["response_format"]["json_schema"]["schema"]))
    return request, schema_sha, sha256_bytes(canonical_json(request))


def _normalized_session_packet(
    session: dict[str, Any], normalized: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    propositions = []
    for proposition in normalized["propositions"]:
        evidence = proposition["evidence"]
        roles = sorted({item["source_role"] for item in evidence})
        propositions.append(
            {
                **proposition,
                "source_role": roles[0] if len(roles) == 1 else "mixed",
                "source_turn_indices": list(
                    dict.fromkeys(item["source_turn_index"] for item in evidence)
                ),
                "evidence_quotes": [item["evidence_quote"] for item in evidence],
            }
        )
    return {
        "session_identity_sha256": session["session_identity_sha256"],
        "session_date": session["session_date"],
        "valid_from": source["valid_from"],
        "source_session_ids": source["source_session_ids"],
        "source_turns_sha256": source["source_turns_sha256"],
        "catalog_sha256": session["catalog_sha256"],
        "propositions": propositions,
    }


def _process_writer_call(
    *,
    identity: str,
    session: dict[str, Any],
    prompt: str,
    client: httpx.Client,
    imported_capture: dict[str, Any] | None,
    model_sha: str,
    unwrap_sha: str,
    provider_counter: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request, schema_sha, request_sha = _writer_identity(session, prompt, v2.READER_MODEL)
    started = time.perf_counter()
    provider_counter["last_latency_ms"] = 0.0

    def provider(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
        provider_counter["calls"] += 1
        provider_started = time.perf_counter()
        try:
            response = client.post(f"{v2.READER_ENDPOINT}/chat/completions", json=req)
            return response.status_code, response.content, response.headers.get("content-type")
        finally:
            provider_counter["last_latency_ms"] = round(
                (time.perf_counter() - provider_started) * 1000, 3
            )

    normalized, ledger = execute_or_resume(
        request=request,
        catalog=session["catalog"],
        session_identity_sha256=identity,
        prompt_sha256=PROMPT_SHA256,
        contract_sha256=CONTRACT_SHA256,
        local_cache_root=LOCAL_CACHE_ROOT,
        provider=provider,
        stage_identity="MEM-3A.1R/3A.2",
        model_sha256=model_sha,
        dynamic_schema_sha256=schema_sha,
        unwrap_source_sha256=unwrap_sha,
        imported_capture=imported_capture,
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    ledger.update(
        {
            "request_sha256": request_sha,
            "dynamic_schema_sha256": schema_sha,
            "unwrap_source_sha256": unwrap_sha,
            "latency_ms": elapsed_ms,
            "success": ledger.get("validation") == "passed",
            "hosted_call": False,
            "provider": "local_llama_cpp",
            "writer_role": "memory_write_extract",
            "prompt_tokens": ledger.get("prompt_tokens"),
            "completion_tokens": ledger.get("completion_tokens"),
        }
    )
    return normalized, ledger


def _write_qualification_result(
    packets: list[dict[str, Any]], ledger: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    _write_jsonl(QUALIFICATION_RUN_DIR / "qualification_packets.jsonl", packets)
    _write_jsonl(QUALIFICATION_RUN_DIR / "writer_call_ledger.jsonl", ledger)
    _write_json(QUALIFICATION_RUN_DIR / "run_manifest.json", manifest)
    report_lines = [
        "# MEM-3A.1R - Repaired Extractor-v2 Qualification",
        "",
        f"Gate: `{manifest['completion_gate_marker']}`",
        "",
        "The historical MEM-3A.1 `NO` record remains unchanged and is classified as a response-unwrapping Harness defect. The v2 prompt/schema are unchanged. Qualification validates exact assistant content bytes extracted from the frozen OpenAI-compatible HTTP envelope, including Harness-derived evidence references.",
        "",
        f"- Qualification packets: {manifest['successful_packets']}/12.",
        f"- New provider calls in this repaired qualification: {manifest['provider_calls_this_run']} (historical envelope imported with zero provider calls).",
        f"- Retries: {manifest['retries']}; hosted calls: {manifest['hosted_calls']}; provenance reconstruction failures: {manifest['provenance_failures']}; envelope failures: {manifest['envelope_failures']}; extractor failures: {manifest['extractor_failures']}.",
        f"- Prompt SHA: `{PROMPT_SHA256}`; contract SHA: `{CONTRACT_SHA256}`.",
        "",
        "## First-Response Import",
        "",
        f"- Imported from `{manifest['historical_import']['imported_from_stage']}` for `{manifest['historical_import']['import_reason']}`.",
        f"- Historical HTTP envelope SHA256: `{manifest['historical_import']['http_envelope_sha256']}`; assistant content SHA256 after unwrap: `{manifest['historical_import'].get('assistant_content_sha256')}`.",
        "- The old failed journal/ledger and raw cache remain unmodified.",
        "",
        "No benchmark labels or question answers were used. Structural qualification auto-advances to the full extraction only when the 12/12 gate passes.",
        "",
    ]
    failure = manifest.get("failure")
    if failure:
        report_lines.extend(
            [
                "## First Failure",
                "",
                f"- Code: `{failure.get('code')}`; session: `{failure.get('session_identity_sha256')}`; proposition index: `{failure.get('proposition_index')}`; field: `{failure.get('field')}`.",
                f"- Detail: {failure.get('detail')}",
                "- The assistant content was successfully unwrapped and the Harness evidence reference resolved. This is a genuine extractor-v2 packet validation failure, not an HTTP-envelope or provenance-reconstruction failure.",
                f"- Generated packet detail: {json.dumps(manifest.get('failure_diagnostic_summary', {}), ensure_ascii=False, sort_keys=True)}.",
                "- No retry was issued; the full 477-session extraction and all MEM-3A.2 downstream stages were not started.",
                "- Compact proposition/evidence-role diagnostics are in `qualification_failure_diagnostics.jsonl`; raw HTTP envelopes remain local and ignored.",
                "",
            ]
        )
    report = "\n".join(report_lines).rstrip() + "\n"
    report_path = QUALIFICATION_RUN_DIR / "mem_3a1r_repaired_qualification.md"
    atomic_write_bytes(report_path, report.encode("utf-8"))
    v2.write_sidecar(report_path)


def _build_failure_diagnostics(
    failure: dict[str, Any],
    ledger_rows: list[dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    identity = failure.get("session_identity_sha256")
    row = next(
        (
            item
            for item in ledger_rows
            if item.get("session_identity_sha256") == identity and not item.get("success")
        ),
        None,
    )
    if row is None:
        return []
    propositions = []
    assistant_path = row.get("assistant_content_cache_path")
    if assistant_path and Path(assistant_path).is_file():
        payload = json.loads(Path(assistant_path).read_text(encoding="utf-8"))
        raw_props = payload.get("propositions", []) if isinstance(payload, dict) else []
        index = failure.get("proposition_index")
        if isinstance(index, int) and 0 <= index < len(raw_props):
            proposition = raw_props[index]
            catalog = {span["evidence_ref"]: span for span in sessions[identity]["catalog"]}
            propositions.append(
                {
                    "proposition_index": index,
                    "proposition_text": proposition.get("proposition_text"),
                    "memory_kind": proposition.get("memory_kind"),
                    "entity_key_candidate": proposition.get("entity_key_candidate"),
                    "attribute_key_candidate": proposition.get("attribute_key_candidate"),
                    "value_text": proposition.get("value_text"),
                    "evidence": [
                        {
                            "evidence_ref": ref,
                            "source_role": catalog[ref]["role"],
                            "source_turn_index": catalog[ref]["source_turn_index"],
                            "source_span_index": catalog[ref]["source_span_index"],
                            "content_sha256": catalog[ref]["content_sha256"],
                        }
                        for ref in proposition.get("evidence_refs", [])
                        if ref in catalog
                    ],
                }
            )
    return [
        {
            "session_identity_sha256": identity,
            "failure_code": failure.get("code"),
            "field": failure.get("field"),
            "proposition_index": failure.get("proposition_index"),
            "detail": failure.get("detail"),
            "http_envelope_sha256": row.get("http_envelope_sha256"),
            "assistant_content_sha256": row.get("assistant_content_sha256"),
            "classification": "GENUINE_EXTRACTOR_V2_PACKET_VALIDATION_FAILURE",
            "propositions": propositions,
        }
    ]


def _refresh_failed_qualification_artifacts() -> dict[str, Any]:
    manifest_path = QUALIFICATION_RUN_DIR / "run_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("completion_gate_marker") != "MEM3A1R_EXTRACTOR_V2_QUALIFIED=NO":
        raise RuntimeError("Failure report refresh requires an already-failed qualification gate")
    ledger_path = QUALIFICATION_RUN_DIR / "writer_call_ledger.jsonl"
    v2.verify_sidecar(ledger_path)
    ledger_rows = read_jsonl(ledger_path)
    failed_row = next((row for row in ledger_rows if not row.get("success")), None)
    if failed_row is None:
        raise RuntimeError("Failed qualification manifest has no structured failed call row")
    failure = dict(manifest.get("failure") or {})
    failure.setdefault("session_identity_sha256", failed_row["session_identity_sha256"])
    failure.setdefault("code", failed_row.get("failure_code"))
    manifest["failure"] = failure
    # The captured timestamps provide call latency without touching the provider again.
    cache_path = Path(failed_row.get("response_cache_path", ""))
    if cache_path.is_file():
        cache_dir = cache_path.parent
        events = read_jsonl(cache_dir / "journal.jsonl")
        started_at = next(
            (event.get("started_at_utc") for event in events if event.get("state") == "STARTED"),
            None,
        )
        metadata = _read_json(cache_dir / "response_meta.json")
        captured_at = metadata.get("captured_at_utc")
        if started_at and captured_at:
            from datetime import datetime

            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(captured_at)
            failed_row["latency_ms"] = round(max(0.0, (end - start).total_seconds() * 1000), 3)
    sessions = v2._read_source_sessions()
    diagnostics = _build_failure_diagnostics(failure, ledger_rows, sessions)
    diagnostic_path = QUALIFICATION_RUN_DIR / "qualification_failure_diagnostics.jsonl"
    diagnostic_sha = _write_jsonl(diagnostic_path, diagnostics)
    summary = {}
    if diagnostics and diagnostics[0].get("propositions"):
        proposition = diagnostics[0]["propositions"][0]
        summary = {
            "proposition_index": proposition["proposition_index"],
            "memory_kind": proposition["memory_kind"],
            "proposition_text": proposition["proposition_text"],
            "evidence": [
                {
                    "evidence_ref": item["evidence_ref"],
                    "source_role": item["source_role"],
                    "source_turn_index": item["source_turn_index"],
                }
                for item in proposition["evidence"]
            ],
        }
    manifest.update(
        {
            "qualification_attempted_outcomes": len(ledger_rows),
            "unique_historical_outcomes": len(ledger_rows),
            "provider_calls_unique_new": sum(row.get("provider_calls", 0) for row in ledger_rows),
            "provider_calls_this_run": sum(row.get("provider_calls", 0) for row in ledger_rows),
            "failure_diagnostics_sha256": diagnostic_sha,
            "failure_diagnostic_summary": summary,
            "auto_advanced_to_full_extraction": False,
            "full_extraction_started": False,
            "test_access": False,
            "labels_loaded": False,
        }
    )
    _write_qualification_result(
        read_jsonl(QUALIFICATION_RUN_DIR / "qualification_packets.jsonl"),
        ledger_rows,
        manifest,
    )
    _write_json(
        FLATPROP_RUN_DIR / "run_manifest.json",
        {
            "run_id": FLATPROP_RUN_ID,
            "stage": "MEM-3A.2 FlatProp full extraction and ten-case diagnostic",
            "status": "NOT_STARTED",
            "completion_gate_marker": "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NOT_RUN",
            "qualification_gate": manifest["completion_gate_marker"],
            "qualification_run_manifest_sha256": v2.sha256_file(manifest_path),
            "extraction_identity_sha256": manifest.get("extraction_identity_sha256"),
            "full_extraction_started": False,
            "writer_calls": 0,
            "embedding": False,
            "retrieval": False,
            "reader_calls": 0,
            "judge_calls": 0,
            "hosted_calls": 0,
            "labels_loaded": False,
            "test_access": False,
            "reason_not_started": "MEM3A1R_EXTRACTOR_V2_QUALIFIED=NO",
        },
    )
    return _read_json(manifest_path)


def _run_qualification_and_full_extraction(
    upstream: dict[str, Any],
    sessions: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    preflight_sha: str,
    selection_sha: str,
    source_sessions: dict[str, dict[str, Any]],
    refs: dict[str, dict[str, str]],
    runtime: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt = v2.PROMPT_PATH.read_text(encoding="utf-8")
    unwrap_sha = v2.sha256_file(Path(__file__).with_name("flat_proposition_writer_v2.py"))
    model_sha = runtime["model_sha256"]
    selected_ids = [row["session_identity_sha256"] for row in selection["session_identities"]]
    if selected_ids[0] != HISTORICAL_IDENTITY:
        raise RuntimeError(
            "Frozen 12-session selection no longer begins with the historical failure"
        )
    identity = {
        "schema_version": 1,
        "stage": "MEM-3A.1R to MEM-3A.2",
        "base_commit_sha": BASE_COMMIT,
        "session_count": 477,
        "prompt_sha256": PROMPT_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "writer_model_sha256": model_sha,
        "writer_endpoint": v2.READER_ENDPOINT,
        "source_span_catalog_algorithm_sha256": v2.sha256_file(
            Path(__file__).with_name("flat_proposition_writer_v2.py")
        ),
        "response_unwrapper_source_sha256": unwrap_sha,
        "preflight_sha256": preflight_sha,
        "selection_sha256": selection_sha,
        "model_runtime": runtime,
        "local_only": True,
        "labels_loaded": False,
        "test_access": False,
    }
    extraction_identity_sha = _freeze_or_verify_protocol(identity)
    source_index = source_sessions
    qualification_packets: list[dict[str, Any]] = []
    qualification_ledger: list[dict[str, Any]] = []
    provider_counter = {"calls": 0}
    historical_request, _, _ = _writer_identity(
        sessions[HISTORICAL_IDENTITY], prompt, v2.READER_MODEL
    )
    imported = _historical_capture(sessions[HISTORICAL_IDENTITY], historical_request)
    run1_manifest: dict[str, Any] = {
        "run_id": QUALIFICATION_RUN_ID,
        "status": "IN_PROGRESS",
        "completion_gate_marker": "MEM3A1R_EXTRACTOR_V2_QUALIFIED=PENDING",
        "base_commit_sha": BASE_COMMIT,
        "prompt_sha256": PROMPT_SHA256,
        "contract_sha256": CONTRACT_SHA256,
        "preflight_sha256": preflight_sha,
        "selection_sha256": selection_sha,
        "provider_calls_this_run": 0,
        "successful_packets": 0,
        "retries": 0,
        "hosted_calls": 0,
        "provenance_failures": 0,
        "envelope_failures": 0,
        "extractor_failures": 0,
        "test_access": False,
        "labels_loaded": False,
        "historical_import": {
            "imported_from_stage": imported["imported_from_stage"],
            "import_reason": imported["import_reason"],
            "session_identity_sha256": HISTORICAL_IDENTITY,
            "request_sha256": imported["old_request_sha256"],
            "http_envelope_sha256": imported["old_http_envelope_sha256"],
            "http_status": imported["http_status"],
            "assistant_content_sha256": None,
            "provider_calls_this_run": 0,
        },
        "failure": None,
    }
    QUALIFICATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (FLATPROP_RUN_DIR / "calls").mkdir(parents=True, exist_ok=True)
    _write_qualification_result(qualification_packets, qualification_ledger, run1_manifest)
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        for index, session_identity in enumerate(selected_ids, 1):
            session = sessions[session_identity]
            print(f"MEM3A1R qualification {index}/12 {session_identity}", flush=True)
            try:
                normalized, ledger_row = _process_writer_call(
                    identity=session_identity,
                    session=session,
                    prompt=prompt,
                    client=client,
                    imported_capture=imported if index == 1 else None,
                    model_sha=model_sha,
                    unwrap_sha=unwrap_sha,
                    provider_counter=provider_counter,
                )
            except WriterQualificationFailure as exc:
                row = dict(exc.ledger)
                row.update(
                    {
                        "request_sha256": _writer_identity(session, prompt, v2.READER_MODEL)[2],
                        "success": False,
                        "hosted_call": False,
                        "latency_ms": provider_counter.get("last_latency_ms", 0.0),
                        "provider": "local_llama_cpp",
                    }
                )
                qualification_ledger.append(row)
                code = (exc.error or {}).get("code", "WRITER_FAILURE")
                failure = {"code": code, **(exc.error or {})}
                failure.update(
                    {
                        "session_identity_sha256": session_identity,
                        "qualification_index": index,
                    }
                )
                if code.startswith("COMPLETION_"):
                    run1_manifest["envelope_failures"] += 1
                elif code in {
                    "PROVENANCE_CONTENT_HASH_MISMATCH",
                    "PROVENANCE_OFFSET_MISMATCH",
                    "UNKNOWN_EVIDENCE_REF",
                    "DUPLICATE_EVIDENCE_REF",
                }:
                    run1_manifest["provenance_failures"] += 1
                else:
                    run1_manifest["extractor_failures"] += 1
                run1_manifest["failure"] = failure
                break
            except (WriterRecoveryError, OSError, httpx.HTTPError) as exc:
                code = (
                    str(exc).split(":", 1)[0]
                    if isinstance(exc, WriterRecoveryError)
                    else "LOCAL_RESPONSE_CAPTURE_FAILURE"
                )
                failure = {
                    "code": code,
                    "detail": str(exc),
                    "session_identity_sha256": session_identity,
                }
                qualification_ledger.append(
                    {
                        "session_identity_sha256": session_identity,
                        "request_sha256": _writer_identity(session, prompt, v2.READER_MODEL)[2],
                        "validation": "failed",
                        "failure_code": code,
                        "success": False,
                        "provider_calls": 0,
                        "provider_calls_this_resume": 0,
                        "hosted_call": False,
                        "latency_ms": provider_counter.get("last_latency_ms", 0.0),
                        "provider": "local_llama_cpp",
                    }
                )
                run1_manifest["failure"] = failure
                break
            source = source_index[session_identity]
            qualification_packets.append(_normalized_session_packet(session, normalized, source))
            qualification_ledger.append(ledger_row)
            if index == 1:
                run1_manifest["historical_import"]["assistant_content_sha256"] = ledger_row.get(
                    "assistant_content_sha256"
                )
            _write_jsonl(
                QUALIFICATION_RUN_DIR / "qualification_packets.jsonl", qualification_packets
            )
            _write_jsonl(QUALIFICATION_RUN_DIR / "writer_call_ledger.jsonl", qualification_ledger)
            run1_manifest["provider_calls_this_run"] = provider_counter["calls"]
            run1_manifest["successful_packets"] = len(qualification_packets)
            _write_qualification_result(qualification_packets, qualification_ledger, run1_manifest)
            print(
                f"MEM3A1R qualification {index}/12 valid propositions={len(normalized['propositions'])} envelope_sha256={ledger_row['http_envelope_sha256']}",
                flush=True,
            )

    qualified = len(qualification_packets) == 12 and run1_manifest["failure"] is None
    run1_manifest.update(
        {
            "status": "COMPLETE" if qualified else "FAILED",
            "completion_gate_marker": "MEM3A1R_EXTRACTOR_V2_QUALIFIED=YES"
            if qualified
            else "MEM3A1R_EXTRACTOR_V2_QUALIFIED=NO",
            "provider_calls_this_run": provider_counter["calls"],
            "successful_packets": len(qualification_packets),
            "qualification_attempted_outcomes": len(qualification_ledger),
            "historical_original_provider_call": 1,
            "unique_historical_outcomes": len(qualification_ledger),
            "provider_calls_unique_new": sum(
                row.get("provider_calls", 0) for row in qualification_ledger
            ),
            "auto_advanced_to_full_extraction": qualified,
            "retries": 0,
            "hosted_calls": 0,
            "reused_historical_captured_envelope": True,
            "historical_envelope_sha256": imported["old_http_envelope_sha256"],
            "historical_assistant_content_sha256": run1_manifest["historical_import"].get(
                "assistant_content_sha256"
            ),
            "extraction_identity_sha256": extraction_identity_sha,
            "failure": run1_manifest["failure"],
        }
    )
    _write_qualification_result(qualification_packets, qualification_ledger, run1_manifest)
    if not qualified:
        return [], [], run1_manifest

    print(
        "MEM3A1R_EXTRACTOR_V2_QUALIFIED=YES; auto-advancing to 477-session extraction", flush=True
    )
    provider_calls_before_full = provider_counter["calls"]
    full_packets: list[dict[str, Any]] = []
    full_ledger: list[dict[str, Any]] = []
    packets_by_identity: dict[str, dict[str, Any]] = {}
    ledger_by_identity: dict[str, dict[str, Any]] = {}
    qualification_set = set(selected_ids)
    # Cache identity is shared with qualification; these twelve are only read, never recalled.
    for row, ledger_row in zip(qualification_packets, qualification_ledger, strict=True):
        packets_by_identity[row["session_identity_sha256"]] = row
        ledger_by_identity[row["session_identity_sha256"]] = ledger_row
    extraction_manifest: dict[str, Any] = {
        "run_id": FLATPROP_RUN_ID,
        "stage": "MEM-3A.2 FlatProp full extraction",
        "status": "IN_PROGRESS",
        "completion_gate_marker": "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=PENDING",
        "base_commit_sha": BASE_COMMIT,
        "extraction_identity_sha256": extraction_identity_sha,
        "expected_unique_sessions": 477,
        "completed_unique_sessions": 12,
        "expected_new_provider_calls": 465,
        "provider_calls_before_full_extraction": provider_calls_before_full,
        "provider_calls_this_process": provider_counter["calls"],
        "retries": 0,
        "hosted_calls": 0,
        "test_access": False,
        "102_dev_access": False,
        "failure": None,
    }
    _write_json(FLATPROP_RUN_DIR / "run_manifest.json", extraction_manifest)
    _write_jsonl(FLATPROP_RUN_DIR / "session_extractions.jsonl", qualification_packets)
    _write_jsonl(FLATPROP_RUN_DIR / "writer_call_ledger.jsonl", qualification_ledger)
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        remaining = [identity for identity in sessions if identity not in qualification_set]
        for index, session_identity in enumerate(remaining, 13):
            session = sessions[session_identity]
            print(f"MEM3A.2 extraction {index}/477 {session_identity}", flush=True)
            try:
                normalized, ledger_row = _process_writer_call(
                    identity=session_identity,
                    session=session,
                    prompt=prompt,
                    client=client,
                    imported_capture=None,
                    model_sha=model_sha,
                    unwrap_sha=unwrap_sha,
                    provider_counter=provider_counter,
                )
            except WriterQualificationFailure as exc:
                code = (exc.error or {}).get("code", "WRITER_FAILURE")
                failed_row = dict(exc.ledger)
                failed_row.update(
                    {
                        "success": False,
                        "hosted_call": False,
                        "provider": "local_llama_cpp",
                        "failure_code": code,
                        "latency_ms": provider_counter.get("last_latency_ms", 0.0),
                    }
                )
                ledger_by_identity[session_identity] = failed_row
                extraction_manifest["failure"] = {
                    "code": code,
                    **(exc.error or {}),
                    "session_identity_sha256": session_identity,
                    "session_index": index,
                }
                extraction_manifest["status"] = "FAILED"
                extraction_manifest["completion_gate_marker"] = (
                    "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO"
                )
                extraction_manifest["provider_calls_this_process"] = provider_counter["calls"]
                extraction_manifest["completed_unique_sessions"] = len(packets_by_identity)
                _write_json(FLATPROP_RUN_DIR / "run_manifest.json", extraction_manifest)
                _write_jsonl(
                    FLATPROP_RUN_DIR / "session_extractions.jsonl",
                    list(packets_by_identity.values()),
                )
                _write_jsonl(
                    FLATPROP_RUN_DIR / "writer_call_ledger.jsonl",
                    [ledger_by_identity[item] for item in sessions if item in ledger_by_identity],
                )
                break
            except (WriterRecoveryError, OSError, httpx.HTTPError) as exc:
                code = (
                    str(exc).split(":", 1)[0]
                    if isinstance(exc, WriterRecoveryError)
                    else "LOCAL_RESPONSE_CAPTURE_FAILURE"
                )
                ledger_by_identity[session_identity] = {
                    "session_identity_sha256": session_identity,
                    "validation": "failed",
                    "failure_code": code,
                    "success": False,
                    "provider_calls": 0,
                    "provider_calls_this_resume": 0,
                    "hosted_call": False,
                    "latency_ms": provider_counter.get("last_latency_ms", 0.0),
                    "provider": "local_llama_cpp",
                }
                extraction_manifest["failure"] = {
                    "code": code,
                    "detail": str(exc),
                    "session_identity_sha256": session_identity,
                    "session_index": index,
                }
                extraction_manifest["status"] = "FAILED"
                extraction_manifest["completion_gate_marker"] = (
                    "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO"
                )
                extraction_manifest["provider_calls_this_process"] = provider_counter["calls"]
                extraction_manifest["completed_unique_sessions"] = len(packets_by_identity)
                _write_json(FLATPROP_RUN_DIR / "run_manifest.json", extraction_manifest)
                _write_jsonl(
                    FLATPROP_RUN_DIR / "session_extractions.jsonl",
                    list(packets_by_identity.values()),
                )
                _write_jsonl(
                    FLATPROP_RUN_DIR / "writer_call_ledger.jsonl",
                    [ledger_by_identity[item] for item in sessions if item in ledger_by_identity],
                )
                break
            packets_by_identity[session_identity] = _normalized_session_packet(
                session, normalized, source_index[session_identity]
            )
            ledger_by_identity[session_identity] = ledger_row
            if len(packets_by_identity) % 5 == 0 or len(packets_by_identity) == 477:
                current_packets = [
                    packets_by_identity[item] for item in sessions if item in packets_by_identity
                ]
                current_ledger = [
                    ledger_by_identity[item] for item in sessions if item in ledger_by_identity
                ]
                _write_jsonl(FLATPROP_RUN_DIR / "session_extractions.jsonl", current_packets)
                _write_jsonl(FLATPROP_RUN_DIR / "writer_call_ledger.jsonl", current_ledger)
                extraction_manifest["completed_unique_sessions"] = len(packets_by_identity)
                extraction_manifest["provider_calls_this_process"] = provider_counter["calls"]
                _write_json(FLATPROP_RUN_DIR / "run_manifest.json", extraction_manifest)
    full_packets = [
        packets_by_identity[item] for item in source_index if item in packets_by_identity
    ]
    full_ledger = [ledger_by_identity[item] for item in source_index if item in ledger_by_identity]
    if extraction_manifest["failure"] is not None or len(full_packets) != 477:
        extraction_manifest["completed_unique_sessions"] = len(full_packets)
        extraction_manifest["provider_calls_this_process"] = provider_counter["calls"]
        _write_json(FLATPROP_RUN_DIR / "run_manifest.json", extraction_manifest)
        _write_jsonl(FLATPROP_RUN_DIR / "session_extractions.jsonl", full_packets)
        _write_jsonl(FLATPROP_RUN_DIR / "writer_call_ledger.jsonl", full_ledger)
        run1_manifest["auto_advanced_to_full_extraction"] = True
        _write_qualification_result(qualification_packets, qualification_ledger, run1_manifest)
        return full_packets, full_ledger, extraction_manifest

    run1_manifest["auto_advanced_to_full_extraction"] = True
    run1_manifest["full_unique_sessions_frozen"] = len(full_packets)
    _write_qualification_result(qualification_packets, qualification_ledger, run1_manifest)
    extraction_manifest.update(
        {
            "status": "EXTRACTION_COMPLETE",
            "completion_gate_marker": "MEM3A2_EXTRACTION_477_FROZEN=YES",
            "completed_unique_sessions": len(full_packets),
            "provider_calls_this_process": provider_counter["calls"],
            "provider_calls_unique_total": sum(row.get("provider_calls", 0) for row in full_ledger),
            "provider_calls_after_qualification": sum(
                ledger_by_identity[item].get("provider_calls", 0)
                for item in sessions
                if item not in qualification_set and item in ledger_by_identity
            ),
            "historical_import_calls": 0,
            "qualification_cache_reuses": 12,
            "retries": 0,
            "hosted_calls": 0,
            "all_packets_structurally_valid": True,
        }
    )
    _write_json(FLATPROP_RUN_DIR / "run_manifest.json", extraction_manifest)
    _write_jsonl(FLATPROP_RUN_DIR / "session_extractions.jsonl", full_packets)
    _write_jsonl(FLATPROP_RUN_DIR / "writer_call_ledger.jsonl", full_ledger)
    return full_packets, full_ledger, extraction_manifest


def _flatprop_paths() -> None:
    flat.RUN_ID = FLATPROP_RUN_ID
    flat.RUN_DIR = FLATPROP_RUN_DIR
    flat.PINNED_BASE_COMMIT = BASE_COMMIT
    flat.PROMPT_PATH = v2.PROMPT_PATH
    flat.EXTRACTOR_CONTRACT_PATH = v2.CONTRACT_PATH
    flat.RUN_REPORT_PATH = FLATPROP_RUN_DIR / "mem_3a2_flatprop_frozen_10_diagnostic.md"
    flat.RUN_MANIFEST_PATH = FLATPROP_RUN_DIR / "run_manifest.json"
    flat.PREFLIGHT_PATH = FLATPROP_RUN_DIR / "writer_preflight.json"
    flat.EXTRACTION_MANIFEST_PATH = FLATPROP_RUN_DIR / "session_extraction_manifest.json"
    flat.EXTRACTIONS_PATH = FLATPROP_RUN_DIR / "session_extractions.jsonl"
    flat.FLAT_PROPOSITIONS_PATH = FLATPROP_RUN_DIR / "flat_propositions.jsonl"
    flat.WRITER_LEDGER_PATH = FLATPROP_RUN_DIR / "writer_call_ledger.jsonl"
    flat.MATERIALIZATION_LEDGER_PATH = FLATPROP_RUN_DIR / "materialization_ledger.jsonl"
    flat.CANDIDATE_GROUPS_PATH = FLATPROP_RUN_DIR / "candidate_revision_groups.json"
    flat.EMBEDDING_MANIFEST_PATH = FLATPROP_RUN_DIR / "embedding_manifest.json"
    flat.DENSE_TOP8_PATH = FLATPROP_RUN_DIR / "dense_top8.jsonl"
    flat.CONTEXT_PLANS_PATH = FLATPROP_RUN_DIR / "context_plans.jsonl"
    flat.CONTEXT_BUNDLES_PATH = FLATPROP_RUN_DIR / "context_bundles.jsonl"
    flat.PREDICTIONS_PATH = FLATPROP_RUN_DIR / "predictions.jsonl"
    flat.READER_LEDGER_PATH = FLATPROP_RUN_DIR / "reader_call_ledger.jsonl"
    flat.METRICS_PATH = FLATPROP_RUN_DIR / "deterministic_metrics.json"
    flat.EFFICIENCY_PATH = FLATPROP_RUN_DIR / "efficiency.json"
    flat.COMPARISON_PATH = FLATPROP_RUN_DIR / "comparison_mem2d_dense_vs_mem3a2_flat.json"
    flat.CASE_REVIEW_PATH = FLATPROP_RUN_DIR / "reflection_cases.json"
    flat.PROTOCOL_PATH = FLATPROP_RUN_DIR / "protocol.md"


def _with_value_sequences(
    groups: dict[str, Any], flat_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in flat_rows:
        by_key.setdefault((row["entity_key_candidate"], row["attribute_key_candidate"]), []).append(
            row
        )
    for group in groups["groups"]:
        rows = sorted(
            by_key[(group["entity_key_candidate"], group["attribute_key_candidate"])],
            key=lambda row: (row["valid_from"], row["source_session_id"], row["proposition_index"]),
        )
        group["value_sequence"] = [
            {
                "value_text": row["value_text"],
                "observed_at": row["valid_from"],
                "source_session_id": row["source_session_id"],
                "question_id": row["question_id"],
                "proposition_text": row["proposition_text"],
            }
            for row in rows
        ]
    return groups


def _materialize_v2(
    session_packets: list[dict[str, Any]], refs: dict[str, dict[str, str]], contract_sha: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    packet_by_identity = {packet["session_identity_sha256"]: packet for packet in session_packets}
    flat_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    records_by_question: dict[str, dict[str, dict[str, Any]]] = {
        qid: {} for qid in flat.QUESTION_IDS
    }
    for qid in flat.QUESTION_IDS:
        scope_id = f"longmemeval:{qid}"
        store = flat.InMemoryMemoryStore()
        for identity, source_session_id in refs[qid].items():
            packet = packet_by_identity[identity]
            for proposition_index, proposition in enumerate(packet["propositions"]):
                evidence = [
                    {**item, "source_session_id": source_session_id}
                    for item in proposition["evidence"]
                ]
                provenance = {
                    "evidence_refs": proposition["evidence_refs"],
                    "evidence": evidence,
                }
                provenance_sha = sha256_bytes(canonical_json(provenance))
                proposition_sha = sha256_bytes(proposition["proposition_text"].encode("utf-8"))
                digest = sha256_bytes(
                    canonical_json(
                        {
                            "extractor_contract_sha256": contract_sha,
                            "session_identity_sha256": identity,
                            "proposition_index": proposition_index,
                            "proposition_text_sha256": proposition_sha,
                            "provenance_sha256": provenance_sha,
                        }
                    )
                )
                memory_id = f"m10flat-{digest}"
                key = f"flat_proposition:{digest}"
                source_roles = sorted({item["source_role"] for item in evidence})
                source_role = source_roles[0] if len(source_roles) == 1 else "mixed"
                value = {
                    "proposition": proposition["proposition_text"],
                    "observed_at": packet["valid_from"],
                    "source_role": source_role,
                }
                operation = flat.MemoryOperation.add(
                    scope_id=scope_id,
                    key=key,
                    kind=flat.MemoryKind.SESSION_NOTE,
                    value=value,
                    source_type=flat.MemorySourceType.SESSION_DERIVED,
                    sensitivity=flat.MemorySensitivity.NON_SENSITIVE,
                    memory_id=memory_id,
                    valid_from=packet["valid_from"],
                    valid_until=None,
                    expires_at=None,
                    supersedes_id=None,
                    source_session_id=source_session_id,
                    source_event_ids=(f"longmemeval-flat-proposition-{digest}",),
                )
                if operation.operation != flat.MemoryOperationType.ADD:
                    raise RuntimeError("MEM-3A.2 permits only ADD operations")
                record = store.apply(operation, now=packet["valid_from"])
                if (
                    record is None
                    or record.status != flat.MemoryStatus.ACTIVE
                    or record.version != 1
                    or record.supersedes_id is not None
                ):
                    raise RuntimeError("FlatProp record violated ACTIVE version-1 invariants")
                row = {
                    "question_id": qid,
                    "scope_id": scope_id,
                    "memory_id": memory_id,
                    "key": key,
                    "session_identity_sha256": identity,
                    "source_session_id": source_session_id,
                    "session_date": packet["session_date"],
                    "valid_from": packet["valid_from"],
                    "proposition_index": proposition_index,
                    "proposition_text": proposition["proposition_text"],
                    "proposition_text_sha256": proposition_sha,
                    "entity_key_candidate": proposition["entity_key_candidate"],
                    "attribute_key_candidate": proposition["attribute_key_candidate"],
                    "value_text": proposition["value_text"],
                    "memory_kind": proposition["memory_kind"],
                    "source_role": source_role,
                    "source_turn_indices": list(
                        dict.fromkeys(item["source_turn_index"] for item in evidence)
                    ),
                    "evidence_refs": proposition["evidence_refs"],
                    "evidence_quotes": [item["evidence_quote"] for item in evidence],
                    "evidence": evidence,
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
                        "provenance_sha256": provenance_sha,
                    }
                )
                records_by_question[qid][memory_id] = {"record": record, "row": row}
        if len(store.history(scope_id)) != len(records_by_question[qid]) or any(
            item.operation != flat.MemoryOperationType.ADD for item in store.history(scope_id)
        ):
            raise RuntimeError(f"Non-ADD event found in FlatProp M10 history: {qid}")
        store.close()
    flat_rows.sort(
        key=lambda row: (
            flat.QUESTION_IDS.index(row["question_id"]),
            row["session_identity_sha256"],
            row["proposition_index"],
        )
    )
    operation_rows.sort(
        key=lambda row: (flat.QUESTION_IDS.index(row["question_id"]), row["memory_id"])
    )
    return flat_rows, operation_rows, records_by_question


def _downstream(
    upstream: dict[str, Any],
    full_packets: list[dict[str, Any]],
    writer_ledger: list[dict[str, Any]],
    runtime: dict[str, Any],
    extraction_manifest: dict[str, Any],
) -> dict[str, Any]:
    _flatprop_paths()
    _write_json(FLATPROP_RUN_DIR / "writer_extraction_run_manifest.json", extraction_manifest)
    source_sessions, refs = flat._source_sessions()
    if len(source_sessions) != 477 or len(full_packets) != 477:
        raise RuntimeError("FlatProp materialization requires exactly 477 unique session packets")
    _write_jsonl(flat.EXTRACTIONS_PATH, full_packets)
    _write_jsonl(flat.WRITER_LEDGER_PATH, writer_ledger)
    prompt_sha = PROMPT_SHA256
    contract_sha = CONTRACT_SHA256
    retrieval_contract_sha = v2.sha256_file(flat.RETRIEVAL_CONTRACT_PATH)
    preflight = {
        "schema_version": 1,
        "stage": FLATPROP_RUN_ID,
        "base_commit_sha": BASE_COMMIT,
        "scope": "ten frozen DEV question diagnostic; extractor input is question-independent source sessions only",
        "question_ids": list(flat.QUESTION_IDS),
        "unique_source_sessions": len(full_packets),
        "extractor_contract_sha256": contract_sha,
        "extractor_prompt_sha256": prompt_sha,
        "retrieval_view_contract_sha256": retrieval_contract_sha,
        "writer_role": "memory_write_extract",
        "writer_model_sha256": runtime["model_sha256"],
        "writer_runtime_identity": runtime,
        "local_only": True,
        "hosted_calls": 0,
        "max_context_tokens": 131072,
        "output_reserve": 4096,
        "all_prompt_tokens_plus_reserve_fit": True,
        "preflight_reference": {
            "path": str(LEGACY_RUN_DIR / "writer_preflight_v2.json"),
            "sha256": v2.verify_sidecar(LEGACY_RUN_DIR / "writer_preflight_v2.json"),
        },
        "qualification_selection_sha256": v2.verify_sidecar(
            LEGACY_RUN_DIR / "qualification_selection.json"
        ),
        "full_session_extraction_sha256": v2.sha256_file(
            FLATPROP_RUN_DIR / "session_extractions.jsonl"
        ),
        "upstream": upstream,
    }
    _write_json(flat.PREFLIGHT_PATH, preflight)
    packets_by_id = {packet["session_identity_sha256"]: packet for packet in full_packets}
    normalized_packets = []
    for identity, session in source_sessions.items():
        packet = packets_by_id.get(identity)
        if packet is None:
            raise RuntimeError(f"Missing frozen FlatProp packet: {identity}")
        if (
            packet["valid_from"] != session["valid_from"]
            or packet["source_session_ids"] != session["source_session_ids"]
            or packet["source_turns_sha256"] != session["source_turns_sha256"]
        ):
            raise RuntimeError(f"Frozen packet provenance does not match RawSpan: {identity}")
        normalized_packets.append(packet)
    flat_rows, operation_rows, records_by_question = _materialize_v2(
        normalized_packets, refs, contract_sha
    )
    _write_jsonl(flat.FLAT_PROPOSITIONS_PATH, flat_rows)
    _write_jsonl(flat.MATERIALIZATION_LEDGER_PATH, operation_rows)
    groups = _with_value_sequences(flat._candidate_groups(flat_rows), flat_rows)
    _write_json(flat.CANDIDATE_GROUPS_PATH, groups)
    ranked, embedding_manifest, query_tokens, retrieval_latency = flat._embedding_and_retrieval(
        flat_rows,
        records_by_question,
        refs,
        preflight,
    )
    embedding_manifest["query_token_counts"] = query_tokens
    _write_json(flat.EMBEDDING_MANIFEST_PATH, embedding_manifest)
    sqlite_sizes, sqlite_matches = flat._materialize_sqlite(
        records_by_question,
        flat_rows,
        sha256_bytes(
            canonical_json(
                {
                    "propositions_sha256": v2.sha256_file(flat.FLAT_PROPOSITIONS_PATH),
                    "contract_sha256": contract_sha,
                }
            )
        ),
    )
    for qid in flat.QUESTION_IDS:
        sqlite_ids = {item["record"].memory_id for item in sqlite_matches[qid]}
        expected = {row["memory_id"] for row in ranked if row["question_id"] == qid}
        if not expected.issubset(sqlite_ids):
            raise RuntimeError(f"SQLite scope/time filter disagrees with dense retrieval: {qid}")
    m2d_inputs = flat.mem2d._load_frozen_inputs()
    questions = m2d_inputs["questions"]
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        reader_contract, reader_sha, system_template, user_template = (
            flat.load_final_reader_contract()
        )
        if reader_sha != flat.PINNED_READER_SHA256:
            raise RuntimeError("Frozen final reader contract changed")
        plans, bundles, _ = flat._project_contexts(
            ranked,
            records_by_question,
            questions,
            client,
            (reader_contract, reader_sha, system_template, user_template),
            retrieval_latency,
        )
    _write_jsonl(flat.DENSE_TOP8_PATH, ranked)
    _write_jsonl(flat.CONTEXT_PLANS_PATH, plans)
    _write_jsonl(flat.CONTEXT_BUNDLES_PATH, bundles)
    pre_reader_paths = (
        flat.PREFLIGHT_PATH,
        flat.EXTRACTIONS_PATH,
        flat.FLAT_PROPOSITIONS_PATH,
        flat.WRITER_LEDGER_PATH,
        flat.MATERIALIZATION_LEDGER_PATH,
        flat.CANDIDATE_GROUPS_PATH,
        flat.EMBEDDING_MANIFEST_PATH,
        flat.DENSE_TOP8_PATH,
        flat.CONTEXT_PLANS_PATH,
        flat.CONTEXT_BUNDLES_PATH,
    )
    if not all(flat._verify_frozen(path) for path in pre_reader_paths):
        raise RuntimeError("A frozen pre-reader artifact failed its SHA sidecar")
    pre_reader_sha = sha256_bytes(
        canonical_json({path.name: v2.sha256_file(path) for path in pre_reader_paths})
    )
    full_upstream = {
        **upstream,
        "projection_contract_sha256": flat.PINNED_PROJECTION_SHA256,
        "reader_contract": reader_contract,
        "base_commit_sha": BASE_COMMIT,
    }
    final = flat._finalize(
        full_upstream,
        preflight,
        normalized_packets,
        writer_ledger,
        flat_rows,
        operation_rows,
        groups,
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
    final["extraction_gate"] = {
        "all_477_unique_packets": len(full_packets) == 477,
        "exactly_465_new_calls_after_qualification": extraction_manifest.get(
            "provider_calls_after_qualification"
        )
        == 465,
        "qualification_12_reused_without_duplicate_calls": extraction_manifest.get(
            "qualification_cache_reuses"
        )
        == 12,
        "retries_zero": extraction_manifest.get("retries") == 0,
        "hosted_calls_zero": extraction_manifest.get("hosted_calls") == 0,
        "all_provenance_harness_derived": all(
            "evidence" in proposition
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
                for item in proposition["evidence"]
            )
            for packet in full_packets
            for proposition in packet["propositions"]
        ),
        "all_operations_add_active_v1": all(
            row["operation"] == "ADD"
            and row["status"] == "active"
            and row["version"] == 1
            and row["supersedes_id"] is None
            for row in operation_rows
        ),
        "dense_top8_all_10_questions": len(ranked) == 80,
        "projection_1024_unchanged": flat.PINNED_PROJECTION_SHA256
        == v2.sha256_file(flat.PROJECTION_CONTRACT_PATH),
        "ten_reader_calls_successful": len(read_jsonl(flat.READER_LEDGER_PATH)) == 10,
        "test_access_false": True,
        "102_dev_access_false": True,
        "no_revision_semantics": True,
    }
    final["gate"]["passed"] = all(
        value for key, value in final["gate"].items() if key != "passed"
    ) and all(final["extraction_gate"].values())
    final["completion_gate_marker"] = (
        "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=YES"
        if final["gate"]["passed"]
        else "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO"
    )
    _write_json(flat.RUN_MANIFEST_PATH, final)
    focus_labels = flat._labels_after_freeze()
    predictions = read_jsonl(flat.PREDICTIONS_PATH)
    prediction_by = {row["question_id"]: row for row in predictions}
    flat_rows_by_q = {
        qid: [row for row in flat_rows if row["question_id"] == qid] for qid in EXPECTED_FOCUS
    }
    ranked_by_q = {
        qid: [row for row in ranked if row["question_id"] == qid] for qid in EXPECTED_FOCUS
    }
    row_by_id = {row["memory_id"]: row for row in flat_rows}
    reflection = {
        "schema_version": 1,
        "diagnostic_only": True,
        "revision_semantics": False,
        "case_ids": list(EXPECTED_FOCUS),
        "cases": [],
        "instagram_500_to_600_audit": [],
        "candidate_revision_groups": groups,
        "provenance_note": "Evidence offsets, roles, quotes, and hashes are copied from the frozen RawSpan catalog by the Harness; model output provides evidence references only.",
    }
    for qid in EXPECTED_FOCUS:
        by_top = []
        for retrieval in ranked_by_q[qid]:
            row = row_by_id[retrieval["memory_id"]]
            by_top.append(
                {
                    "rank": retrieval["rank"],
                    "cosine_similarity": retrieval["cosine_similarity"],
                    "memory_id": row["memory_id"],
                    "source_session_id": row["source_session_id"],
                    "observed_at": row["valid_from"],
                    "proposition_text": row["proposition_text"],
                    "candidate_key": [row["entity_key_candidate"], row["attribute_key_candidate"]],
                    "value_text": row["value_text"],
                    "evidence_quotes": row["evidence_quotes"],
                }
            )
        question = questions[qid]
        reflection["cases"].append(
            {
                "question_id": qid,
                "question": question["question"],
                "question_date": question["question_date"],
                "reference_answer": focus_labels[qid]["answer"],
                "answer_session_ids": focus_labels[qid]["answer_session_ids"],
                "prediction": prediction_by[qid]["predicted"],
                "metrics": next(
                    item
                    for item in _read_json(flat.METRICS_PATH)["arms"]["m10_flatprop_dense"][
                        "per_question"
                    ]
                    if item["question_id"] == qid
                ),
                "dense_top8": by_top,
                "all_scoped_propositions": [
                    {
                        "source_session_id": row["source_session_id"],
                        "observed_at": row["valid_from"],
                        "proposition_text": row["proposition_text"],
                        "candidate_key": [
                            row["entity_key_candidate"],
                            row["attribute_key_candidate"],
                        ],
                        "value_text": row["value_text"],
                        "evidence_quotes": row["evidence_quotes"],
                    }
                    for row in flat_rows_by_q[qid]
                ],
            }
        )
    for row in flat_rows:
        text = " ".join(
            str(row[key]).casefold()
            for key in (
                "entity_key_candidate",
                "attribute_key_candidate",
                "value_text",
                "proposition_text",
            )
        )
        if row["question_id"] == "1cea1afa" and (
            "instagram" in text or "500" in text or "600" in text
        ):
            reflection["instagram_500_to_600_audit"].append(
                {
                    "question_id": row["question_id"],
                    "candidate_key": [row["entity_key_candidate"], row["attribute_key_candidate"]],
                    "value_text": row["value_text"],
                    "observed_at": row["valid_from"],
                    "source_session_id": row["source_session_id"],
                    "proposition_text": row["proposition_text"],
                    "retrieved_top8": any(
                        retrieval["question_id"] == row["question_id"]
                        and retrieval["memory_id"] == row["memory_id"]
                        for retrieval in ranked
                    ),
                }
            )
    _write_json(flat.RUN_DIR / "reflection_cases.json", reflection)
    manifest = _read_json(flat.RUN_MANIFEST_PATH)
    manifest["artifact_sha256"]["reflection_cases.json"] = v2.sha256_file(
        flat.RUN_DIR / "reflection_cases.json"
    )
    manifest["artifact_sha256"]["writer_extraction_run_manifest.json"] = v2.sha256_file(
        FLATPROP_RUN_DIR / "writer_extraction_run_manifest.json"
    )
    manifest["gate"]["reflection_cases_frozen"] = flat._verify_frozen(
        flat.RUN_DIR / "reflection_cases.json"
    )
    manifest["gate"]["writer_extraction_manifest_frozen"] = flat._verify_frozen(
        FLATPROP_RUN_DIR / "writer_extraction_run_manifest.json"
    )
    manifest["gate"]["passed"] = all(
        value for key, value in manifest["gate"].items() if key != "passed"
    ) and all(manifest["extraction_gate"].values())
    manifest["completion_gate_marker"] = (
        "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=YES"
        if manifest["gate"]["passed"]
        else "MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO"
    )
    report_path = flat.RUN_REPORT_PATH
    report = report_path.read_text(encoding="utf-8")
    report += _render_reflection_summary(reflection, groups, full_packets, flat_rows, writer_ledger)
    atomic_write_bytes(report_path, report.encode("utf-8"))
    flat._freeze(report_path)
    manifest["artifact_sha256"][report_path.name] = v2.sha256_file(report_path)
    _write_json(flat.RUN_MANIFEST_PATH, manifest)
    if not manifest["gate"]["passed"]:
        raise RuntimeError("MEM3A2_FLAT_PROPOSITION_FROZEN_10_DIAGNOSTIC=NO")
    return manifest


def _render_reflection_summary(
    reflection: dict[str, Any],
    groups: dict[str, Any],
    packets: list[dict[str, Any]],
    flat_rows: list[dict[str, Any]],
    writer_ledger: list[dict[str, Any]],
) -> str:
    props = sum(len(packet["propositions"]) for packet in packets)
    raw_spans = len(read_jsonl(flat.MEM2C_INVENTORY))
    instagram = reflection["instagram_500_to_600_audit"]
    lines = [
        "",
        "## Reflection Detail",
        "",
        f"- Unique source sessions: {len(packets)}; RawSpan records: {raw_spans}; FlatProp propositions: {props}; RawSpan/FlatProp ratio: {raw_spans / max(1, props):.3f}.",
        f"- Writer outcomes: {len(writer_ledger)}; local provider calls across qualification and full stage: {sum(row.get('provider_calls', 0) for row in writer_ledger)}; writer prompt/completion tokens: {sum(row.get('prompt_tokens') or 0 for row in writer_ledger)}/{sum(row.get('completion_tokens') or 0 for row in writer_ledger)}.",
        f"- Candidate keys reused: {groups['candidate_key_pairs']}; same-key multiple-value groups: {groups['same_key_multiple_value_group_count']}. No candidate group is materialized as a revision.",
        "- Full evidence and retrieval context for the eight requested reflection cases is in `reflection_cases.json`.",
        "",
        "### Instagram 500 → 600",
        "",
        f"- Extracted propositions matching Instagram / 500 / 600: {len(instagram)}.",
    ]
    for row in instagram:
        lines.append(
            f"- `{row['question_id']}` {row['candidate_key'][0]}.{row['candidate_key'][1]} = {row['value_text']} ({row['observed_at']}, {row['source_session_id']}); retrieved top-8: {row['retrieved_top8']}. {row['proposition_text']}"
        )
    lines.extend(
        [
            "",
            "### Diagnostic Boundary",
            "",
            "These ten frozen question cases are descriptive only. A source-session hit is not proof that the answer-bearing proposition was retrieved. If the writer retains both Instagram values under a reusable key and both enter context, any remaining reader error is visible as a state-resolution motivation, not evidence that this stage resolved it. FlatProp remains ADD-only, ACTIVE v1, without UPDATE, DELETE, SUPERSEDED, or temporal query modes.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    QUALIFICATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    FLATPROP_RUN_DIR.mkdir(parents=True, exist_ok=True)
    upstream = _verify_base_and_upstream()
    sessions = v2._read_source_sessions()
    source_sessions, refs = flat._source_sessions()
    if set(sessions) != set(source_sessions) or len(sessions) != 477:
        raise RuntimeError("MEM-3A.1R and MEM-3A.2 RawSpan identities differ")
    selection_rows, selection, preflight_sha, selection_sha = v2._frozen_preparation(
        sessions,
        v2.PROMPT_PATH.read_text(encoding="utf-8"),
        PROMPT_SHA256,
        CONTRACT_SHA256,
        upstream["mem2c_inventory_sha256"],
    )
    if len(selection_rows) != 477 or len(selection["session_identities"]) != 12:
        raise RuntimeError("Frozen qualification/preflight identities are incomplete")
    m2a_manifest = v2.read_json(v2.MEM2A_DIR / "run_manifest.json")
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
        trust_env=False,
    ) as client:
        runtime = v2._runtime_manifest(client, m2a_manifest)
    endpoint = urlsplit(runtime["endpoint"])
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "::1"}:
        raise RuntimeError("Writer endpoint is not loopback-only")
    packets, ledger, extraction_manifest = _run_qualification_and_full_extraction(
        upstream,
        sessions,
        selection,
        preflight_sha,
        selection_sha,
        source_sessions,
        refs,
        runtime,
    )
    if extraction_manifest.get("completion_gate_marker") != "MEM3A2_EXTRACTION_477_FROZEN=YES":
        if extraction_manifest.get("completion_gate_marker") == "MEM3A1R_EXTRACTOR_V2_QUALIFIED=NO":
            _refresh_failed_qualification_artifacts()
            return _read_json(QUALIFICATION_RUN_DIR / "run_manifest.json")
        return extraction_manifest
    return _downstream(upstream, packets, ledger, runtime, extraction_manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--refresh-failure-report", action="store_true")
    args = parser.parse_args()
    if args.refresh_failure_report:
        try:
            manifest = _refresh_failed_qualification_artifacts()
        except Exception as exc:  # noqa: BLE001 - report deterministic artifact refresh failures.
            print(f"FAILURE REPORT REFRESH FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(manifest["completion_gate_marker"], flush=True)
        return 0 if manifest["completion_gate_marker"].endswith("=NO") else 1
    try:
        manifest = run()
    except Exception as exc:  # noqa: BLE001 - fail closed and report the exact first-stage exception.
        failure = {"code": type(exc).__name__, "detail": str(exc)}
        print(
            f"MEM-3A.1R/MEM-3A.2 STOP: {json.dumps(failure, ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(manifest["completion_gate_marker"], flush=True)
    return 0 if manifest["completion_gate_marker"].endswith("=YES") else 1


if __name__ == "__main__":
    raise SystemExit(main())
