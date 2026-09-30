"""Run MEM-3B0R with keyed identity maps and local-only recovery."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import flat_proposition_writer_v2 as writer_v2  # noqa: E402
from tools.research.memory import revision_identity as semantic_v1  # noqa: E402
from tools.research.memory import revision_identity_wire_v2 as wire_v2  # noqa: E402
from tools.research.memory import run_mem2b_rank_aware_projection as mem2b  # noqa: E402
from tools.research.memory import run_mem3a_flat_proposition as flat  # noqa: E402
from tools.research.memory import run_mem3b0_revision_identity as v1  # noqa: E402

RUN_ID = "mem3b0r-revision-identity-keyed-map-20260930"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
CONTRACT_PATH = RUN_DIR / "identity_contract.json"
PREFLIGHT_PATH = RUN_DIR / "identity_preflight.json"
IDENTITIES_PATH = RUN_DIR / "revision_identity_records.jsonl"
CALL_LEDGER_PATH = RUN_DIR / "identity_call_ledger.jsonl"
LINEAGE_PATH = RUN_DIR / "recovery_lineage.jsonl"
CORE_MANIFEST_PATH = RUN_DIR / "identity_run_manifest.json"
GROUPS_PATH = RUN_DIR / "candidate_revision_groups.json"
STATS_PATH = RUN_DIR / "identity_statistics.json"
FRAGMENTATION_PATH = RUN_DIR / "key_fragmentation_diagnostics.json"
COLLISION_PATH = RUN_DIR / "key_collision_diagnostics.json"
CASE_REVIEW_PATH = RUN_DIR / "critical_case_review.json"
FINAL_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
FAILURE_PATH = RUN_DIR / "execution_failure.json"
REPORT_PATH = RUN_DIR / "report.md"
REPORT_DOC_PATH = ROOT / "docs" / "research" / "memory" / "mem_3b0r_revision_identity_review.md"
REPORT_PROTOCOL_PATH = (
    ROOT / "docs" / "research" / "memory" / "mem_3b0r_keyed_identity_map_protocol.md"
)
PREVIOUS_CONTRACT_PATH = v1.CONTRACT_PATH
READER_ENDPOINT = v1.READER_ENDPOINT
READER_MODEL = v1.READER_MODEL
READER_MODEL_SHA256 = v1.READER_MODEL_SHA256
INITIAL_BATCH_SIZE = 32
MAX_COMPLETION_TOKENS = 8192
CONTEXT_TOKENS = 131072
FRAGMENTATION_THRESHOLD = 0.82
COLLISION_JACCARD_THRESHOLD = 0.08
GATE_NAME = "MEM3B0R_REVISION_IDENTITY_OVERLAY_COMPLETE"
PROMPT_SHA256 = writer_v2.sha256_bytes(semantic_v1.SYSTEM_PROMPT.encode("utf-8"))
STAGE_CACHE_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Health-Copilot" / RUN_ID
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _sha_file(path: Path) -> str:
    return writer_v2.sha256_bytes(path.read_bytes())


def _freeze(path: Path, payload: bytes) -> str:
    return v1._freeze(path, payload)


def _freeze_json(path: Path, value: Any) -> str:
    return v1._freeze_json(path, value)


def _freeze_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(_canonical_json(row) + b"\n" for row in rows)
    return _freeze(path, payload)


def _contract_components() -> tuple[dict[str, Any], dict[str, str]]:
    if not v1._verify_frozen(PREVIOUS_CONTRACT_PATH):
        raise wire_v2.GlobalIdentityFailure("historical_semantic_contract_sidecar_invalid")
    semantic_sha = _sha_file(PREVIOUS_CONTRACT_PATH)
    if semantic_sha != "7636f96f90e18eb014d9afa83116569335f4cbd219fe52916549bae3bc5bfad0":
        raise wire_v2.GlobalIdentityFailure("historical_semantic_contract_sha_mismatch")

    wire_contract = {
        "contract_id": wire_v2.WIRE_CONTRACT_ID,
        "root_shape": {"identities": "object keyed by the exact current batch IDs"},
        "inner_payload_fields": ["revision_kind", "subject_key", "attribute_key", "value_text"],
        "memory_id_in_inner_payload": False,
        "required_keys_equal_current_batch_ids": True,
        "additional_properties": False,
        "raw_duplicate_object_keys": "reject before dict construction",
    }
    recovery_contract = {
        "local_failure_codes": sorted(wire_v2.LOCAL_FAILURE_CODES),
        "batch_size_gt_one": "discard parent semantic output; split contiguous canonical input in half; left before right",
        "single_item_local_output_failure": "UNKNOWN / unknown / unknown with original frozen proposition text and HARNESS_UNKNOWN_FALLBACK origin",
        "global_failures": [
            "frozen source SHA or inventory corruption",
            "duplicate frozen input IDs",
            "MemoryStore mutation",
            "provider/model/runtime identity mismatch",
            "benchmark label leakage",
            "cache or artifact integrity failure",
            "code/protocol identity mismatch",
            "provider response unavailable",
        ],
        "same_request_retries": 0,
        "fallback_for_infrastructure_or_source_failure": False,
    }
    source_hashes = {
        "validator_source_sha256": _sha_file(Path(wire_v2.__file__)),
        "recovery_implementation_sha256": _sha_file(Path(wire_v2.__file__)),
        "recovery_contract_sha256": writer_v2.sha256_bytes(_canonical_json(recovery_contract)),
        "wire_contract_sha256": writer_v2.sha256_bytes(_canonical_json(wire_contract)),
        "semantic_identity_contract_sha256": semantic_sha,
        "model_sha256": READER_MODEL_SHA256,
    }
    validator_binding_sha = writer_v2.sha256_bytes(_canonical_json(source_hashes))
    contract = {
        "schema_version": 2,
        "contract_id": "revision-identity-v2-keyed-map-local-recovery",
        "validator_id": wire_v2.VALIDATOR_ID,
        "semantic_contract_unchanged": True,
        "semantic_prompt_sha256": PROMPT_SHA256,
        "input_fields": list(semantic_v1.INPUT_FIELDS),
        "revision_kinds": list(semantic_v1.REVISION_KINDS),
        "output_fields": [
            "memory_id",
            "revision_kind",
            "subject_key",
            "attribute_key",
            "value_text",
            "identity_origin",
            "fallback_reason",
        ],
        "wire_contract": wire_contract,
        "recovery_contract": recovery_contract,
        "validator_identity_binding": source_hashes,
        "validator_identity_sha256": validator_binding_sha,
        "model": {
            "role": "identity proposer",
            "model": "Qwen3-8B Q4_K_M frozen local artifact",
            "model_sha256": READER_MODEL_SHA256,
            "endpoint": READER_ENDPOINT,
            "temperature": 0,
            "seed": 42,
            "thinking": False,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "hosted_fallback": False,
        },
        "batch": {
            "initial_size": INITIAL_BATCH_SIZE,
            "order": "scope_id then memory_id ascending",
            "same_request_retries": 0,
        },
    }
    hashes = {
        **source_hashes,
        "validator_identity_sha256": validator_binding_sha,
        "contract_sha256": writer_v2.sha256_bytes(_json_bytes(contract)),
    }
    return contract, hashes


def _source_code_identity() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "wire_validator_and_recovery": Path(wire_v2.__file__),
        "semantic_identity_v1": Path(semantic_v1.__file__),
        "historical_v1_runner": Path(v1.__file__),
        "response_journal": Path(writer_v2.__file__),
        "runtime_verifier": Path(mem2b.__file__),
        "reader_adapter": Path(flat.__file__),
        "server_launcher": ROOT / "tools" / "research" / "memory" / "start_mem1_local_reader.ps1",
        "protocol": REPORT_PROTOCOL_PATH,
    }
    return {name: _sha_file(path) for name, path in paths.items()}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_counts(base: str, head: str) -> tuple[int, int]:
    left, right = _git("rev-list", "--left-right", "--count", f"{base}...{head}").split()
    return int(right), int(left)


def _token_count(client: httpx.Client, text: str) -> int:
    response = client.post(
        "http://127.0.0.1:8081/tokenize",
        json={"content": text, "add_special": False, "parse_special": True},
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise wire_v2.GlobalIdentityFailure("local_tokenizer_response_malformed")
    return len(tokens)


def _render_template(client: httpx.Client, request: dict[str, Any]) -> str:
    response = client.post(
        "http://127.0.0.1:8081/apply-template",
        json={
            "messages": request["messages"],
            "chat_template_kwargs": request["chat_template_kwargs"],
            "add_generation_prompt": True,
        },
    )
    response.raise_for_status()
    prompt = response.json().get("prompt")
    if not isinstance(prompt, str):
        raise wire_v2.GlobalIdentityFailure("local_chat_template_response_malformed")
    return prompt


def _distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0.0}
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 3),
    }


def _batch_profile(
    client: httpx.Client, projected: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batches = [
        projected[i : i + INITIAL_BATCH_SIZE] for i in range(0, len(projected), INITIAL_BATCH_SIZE)
    ]
    rows = []
    prompt_counts = []
    schema_counts = []
    request_sizes = []
    for batch in batches:
        request = wire_v2.build_request(batch, model=READER_MODEL, max_tokens=MAX_COMPLETION_TOKENS)
        rendered = _render_template(client, request)
        prompt_tokens = _token_count(client, rendered)
        schema_text = json.dumps(
            request["response_format"]["json_schema"]["schema"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        schema_tokens = _token_count(client, schema_text)
        request_bytes = len(_canonical_json(request))
        fits = prompt_tokens + schema_tokens + MAX_COMPLETION_TOKENS <= CONTEXT_TOKENS
        batch_id = wire_v2.batch_id(batch)
        rows.append(
            {
                "batch_id": batch_id,
                "batch_size": len(batch),
                "first_memory_id": batch[0]["memory_id"],
                "last_memory_id": batch[-1]["memory_id"],
                "prompt_tokens": prompt_tokens,
                "dynamic_schema_tokens": schema_tokens,
                "schema_tokens_in_fit_reserve": schema_tokens,
                "request_size_bytes": request_bytes,
                "completion_reserve_tokens": MAX_COMPLETION_TOKENS,
                "context_tokens": CONTEXT_TOKENS,
                "fits_frozen_context": fits,
            }
        )
        prompt_counts.append(prompt_tokens)
        schema_counts.append(schema_tokens)
        request_sizes.append(request_bytes)
    profile = {
        "initial_batch_count": len(batches),
        "batch_size": INITIAL_BATCH_SIZE,
        "prompt_token_distribution": _distribution(prompt_counts),
        "dynamic_schema_token_distribution": _distribution(schema_counts),
        "request_size_bytes_distribution": _distribution(request_sizes),
        "maximum_request_size_bytes": max(request_sizes, default=0),
        "all_requests_fit_frozen_context": all(row["fits_frozen_context"] for row in rows),
        "context_check": "conservative rendered prompt tokens + dynamic schema tokens + 8192 completion reserve <= 131072",
        "tokenizer_endpoint": "http://127.0.0.1:8081/tokenize",
        "template_endpoint": "http://127.0.0.1:8081/apply-template",
        "tokenizer_execution_local": True,
        "generation_calls_during_preflight": 0,
        "batch_profiles": rows,
    }
    return rows, profile


def _preflight(
    *, write: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if _git("branch", "--show-current") != v1.TOPIC_BRANCH:
        raise wire_v2.GlobalIdentityFailure("unexpected_memory_topic_branch")
    if not v1._verify_frozen(REPORT_PROTOCOL_PATH):
        raise wire_v2.GlobalIdentityFailure("mem3b0r_protocol_sidecar_invalid")
    rows, projected, frozen_input = v1._load_source()
    if len(rows) != 8112:
        raise wire_v2.GlobalIdentityFailure("frozen_flatprop_input_count_mismatch")

    audit_text = v1.COMPATIBILITY_AUDIT.read_text(encoding="utf-8")
    if "MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES" not in audit_text:
        raise wire_v2.GlobalIdentityFailure("medmemorybench_compatibility_gate_missing")
    if "LONGMEM_MEMORY_MECHANISM_FROZEN=NO" not in audit_text:
        raise wire_v2.GlobalIdentityFailure("longmem_memory_mechanism_gate_changed")

    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    merge_base = _git("merge-base", "HEAD", "origin/main")
    ahead, behind = _git_counts("origin/main", "HEAD")
    tracked_clean = (
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False).returncode
        == 0
    )
    staged_clean = (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
    )
    if not tracked_clean or not staged_clean or behind != 0:
        raise wire_v2.GlobalIdentityFailure(
            "preflight_requires_clean_tracked_branch_at_latest_main"
        )
    untracked = sorted(_git("ls-files", "--others", "--exclude-standard").splitlines())

    contract, contract_hashes = _contract_components()
    contract_bytes = _json_bytes(contract)
    contract_sha = writer_v2.sha256_bytes(contract_bytes)
    if contract_sha != contract_hashes["contract_sha256"]:
        raise wire_v2.GlobalIdentityFailure("identity_contract_hash_internal_mismatch")

    with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0), trust_env=False) as client:
        runtime = v1._reader_runtime(client)
        runtime.pop("server_pid", None)
        runtime.pop("answer_calls", None)
        if runtime.get("local_only") is not True or runtime.get("loopback_only") is not True:
            raise wire_v2.GlobalIdentityFailure("identity_reader_not_local_loopback")
        batch_rows, profile = _batch_profile(client, projected)
    if len(batch_rows) != 254 or not profile["all_requests_fit_frozen_context"]:
        raise wire_v2.GlobalIdentityFailure("full_run_batch_or_context_preflight_failed")
    _freeze(CONTRACT_PATH, contract_bytes)

    code_identity = _source_code_identity()
    preflight = {
        "schema_version": 1,
        "stage": RUN_ID,
        "branch": branch,
        "topic_head_sha": head,
        "origin_main_sha": origin_main,
        "base_commit_sha": merge_base,
        "ahead_of_origin_main": ahead,
        "behind_origin_main": behind,
        "tracked_worktree_clean": tracked_clean,
        "staged_worktree_clean": staged_clean,
        "untracked_worktree_path_count": len(untracked),
        "untracked_worktree_roots": sorted({path.split("/")[0] for path in untracked}),
        "untracked_worktree_paths_sha256": writer_v2.sha256_bytes(
            "\n".join(untracked).encode("utf-8")
        ),
        "frozen_input": frozen_input,
        "historical_v1_failure": {
            "gate": "MEM3B0_REVISION_IDENTITY_FROZEN_DIAGNOSTIC=NO",
            "interpretation": "array ID coverage schema insufficient; not semantic-quality evidence",
            "run_dir_unchanged": True,
        },
        "identity_contract_sha256": contract_sha,
        "validator_identity_binding": contract["validator_identity_binding"],
        "validator_identity_sha256": contract["validator_identity_sha256"],
        "identity_model_runtime": runtime,
        "full_request_preflight": profile,
        "source_code_identity_sha256": code_identity,
        "label_leakage": {
            "benchmark_question_supplied": False,
            "question_date_supplied": False,
            "gold_answer_supplied": False,
            "answer_session_ids_supplied": False,
            "question_type_supplied": False,
            "reader_prediction_supplied": False,
            "retrieval_rank_supplied": False,
            "correctness_supplied": False,
            "observed_at_supplied": False,
        },
        "calls_during_preflight": {
            "local_tokenizer_requests": len(batch_rows) * 2,
            "identity_generation_requests": 0,
            "embedding_calls": 0,
            "retrieval_calls": 0,
            "reader_calls": 0,
            "judge_calls": 0,
            "hosted_calls": 0,
        },
        "102_dev_access": False,
        "test_access": False,
        "medmemorybench_runs": 0,
        "all_source_memories_active_version_1_no_supersedes": frozen_input[
            "all_source_memories_active_version_1_no_supersedes"
        ],
    }
    if write:
        _freeze_json(PREFLIGHT_PATH, preflight)
    return rows, projected, preflight


def _validate_saved_preflight() -> tuple[
    list[dict[str, Any]], list[dict[str, str]], dict[str, Any]
]:
    if not v1._verify_frozen(CONTRACT_PATH) or not v1._verify_frozen(PREFLIGHT_PATH):
        raise wire_v2.GlobalIdentityFailure("identity_contract_or_preflight_not_frozen")
    if not v1._verify_frozen(REPORT_PROTOCOL_PATH):
        raise wire_v2.GlobalIdentityFailure("mem3b0r_protocol_sidecar_invalid")
    rows, projected, frozen_input = v1._load_source()
    preflight = v1._read_json(PREFLIGHT_PATH)
    _contract, hashes = _contract_components()
    if _sha_file(CONTRACT_PATH) != hashes["contract_sha256"]:
        raise wire_v2.GlobalIdentityFailure("frozen_identity_contract_sha_mismatch")
    if preflight["frozen_input"] != frozen_input:
        raise wire_v2.GlobalIdentityFailure("frozen_flatprop_input_changed_after_preflight")
    if preflight["identity_contract_sha256"] != hashes["contract_sha256"]:
        raise wire_v2.GlobalIdentityFailure("identity_contract_changed_after_preflight")
    if preflight["source_code_identity_sha256"] != _source_code_identity():
        raise wire_v2.GlobalIdentityFailure("identity_source_code_changed_after_preflight")
    if _git("branch", "--show-current") != v1.TOPIC_BRANCH:
        raise wire_v2.GlobalIdentityFailure("unexpected_memory_topic_branch")
    if _git("rev-parse", "HEAD") != preflight["topic_head_sha"]:
        raise wire_v2.GlobalIdentityFailure("topic_head_changed_after_preflight")
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0), trust_env=False) as client:
        runtime = v1._reader_runtime(client)
        runtime.pop("server_pid", None)
        runtime.pop("answer_calls", None)
    if runtime != preflight["identity_model_runtime"]:
        raise wire_v2.GlobalIdentityFailure("identity_runtime_changed_after_preflight")
    if preflight["full_request_preflight"]["all_requests_fit_frozen_context"] is not True:
        raise wire_v2.GlobalIdentityFailure("frozen_context_preflight_not_passed")
    return rows, projected, preflight


def _ledger_fields(ledger: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "session_identity_sha256",
        "cache_identity_sha256",
        "request_sha256",
        "http_envelope_sha256",
        "assistant_content_sha256",
        "http_status",
        "provider_calls",
        "provider_calls_this_resume",
        "provider_duration_ms",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "model",
        "validation",
        "failure_code",
        "reused_frozen_result",
    )
    return {key: ledger.get(key) for key in fields}


def _run_identity_calls(
    projected: list[dict[str, str]],
    contract_sha: str,
    code_identity: dict[str, str],
    expected_runtime: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if urlsplit(READER_ENDPOINT).hostname != "127.0.0.1":
        raise wire_v2.GlobalIdentityFailure("hosted_identity_fallback_forbidden")
    STAGE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    provider_sends = 0
    started = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0), trust_env=False) as client:
        runtime_at_start = v1._reader_runtime(client)
        runtime_at_start.pop("server_pid", None)
        runtime_at_start.pop("answer_calls", None)
        if runtime_at_start != expected_runtime:
            raise wire_v2.GlobalIdentityFailure("identity_runtime_changed_before_first_batch")

        def invoke(batch: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            nonlocal provider_sends
            current_batch_id = wire_v2.batch_id(batch)
            request = wire_v2.build_request(
                batch, model=READER_MODEL, max_tokens=MAX_COMPLETION_TOKENS
            )
            request_sha = writer_v2.sha256_bytes(_canonical_json(request))

            def provider(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
                nonlocal provider_sends
                if req != request:
                    raise wire_v2.GlobalIdentityFailure("provider_request_identity_mismatch")
                if urlsplit(READER_ENDPOINT).hostname != "127.0.0.1":
                    raise wire_v2.GlobalIdentityFailure("provider_endpoint_not_loopback")
                provider_sends += 1
                response = client.post(f"{READER_ENDPOINT}/chat/completions", json=req)
                return response.status_code, response.content, response.headers.get("content-type")

            def packet_validator(raw: bytes, catalog: list[dict[str, Any]]) -> dict[str, Any]:
                try:
                    records = wire_v2.validate_response(raw, catalog)
                except wire_v2.KeyedIdentityContractError as exc:
                    raise writer_v2.ExtractionValidationError(exc.code, detail=exc.detail) from exc
                return {"identities": records}

            try:
                packet, ledger = writer_v2.execute_or_resume(
                    request=request,
                    catalog=batch,
                    session_identity_sha256=current_batch_id,
                    prompt_sha256=PROMPT_SHA256,
                    contract_sha256=contract_sha,
                    local_cache_root=STAGE_CACHE_ROOT,
                    provider=provider,
                    stage_identity=RUN_ID,
                    model_sha256=READER_MODEL_SHA256,
                    dynamic_schema_sha256=writer_v2.sha256_bytes(
                        _canonical_json(request["response_format"])
                    ),
                    unwrap_source_sha256=code_identity["response_journal"],
                    packet_validator=packet_validator,
                    packet_validator_sha256=code_identity["wire_validator_and_recovery"],
                )
            except writer_v2.WriterQualificationFailure as exc:
                failure_code = exc.ledger.get("failure_code") or "UNKNOWN_WRITER_FAILURE"
                row = {
                    **_ledger_fields(exc.ledger),
                    "batch_id": current_batch_id,
                    "input_memory_ids": [item["memory_id"] for item in batch],
                    "request_sha256": request_sha,
                    "hosted_call": False,
                }
                detail = exc.error.get("detail") if isinstance(exc.error, dict) else failure_code
                if failure_code in wire_v2.LOCAL_FAILURE_CODES:
                    raise wire_v2.LocalBatchFailure(failure_code, row, str(detail)) from exc
                raise wire_v2.GlobalIdentityFailure(
                    f"nonrecoverable_provider_or_envelope_failure:{failure_code}:{detail}"
                ) from exc
            except writer_v2.WriterRecoveryError as exc:
                raise wire_v2.GlobalIdentityFailure(
                    f"response_cache_integrity_failure:{exc}"
                ) from exc

            if ledger.get("model") != str(v1.READER_MODEL_PATH):
                raise wire_v2.GlobalIdentityFailure("identity_provider_response_model_mismatch")

            output = packet.get("identities") if isinstance(packet, dict) else None
            if not isinstance(output, list) or len(output) != len(batch):
                raise wire_v2.GlobalIdentityFailure(
                    f"identity_packet_shape_mismatch:{current_batch_id}"
                )
            row = {
                **_ledger_fields(ledger),
                "batch_id": current_batch_id,
                "input_memory_ids": [item["memory_id"] for item in batch],
                "request_sha256": request_sha,
                "hosted_call": False,
            }
            return output, row

        for start in range(0, len(projected), INITIAL_BATCH_SIZE):
            batch = projected[start : start + INITIAL_BATCH_SIZE]
            records, batch_calls, batch_lineage = wire_v2.recover_batch(batch, invoke)
            calls.extend(batch_calls)
            lineage.extend(batch_lineage)
            for row in records:
                memory_id = row["memory_id"]
                if memory_id in completed:
                    raise wire_v2.GlobalIdentityFailure(f"duplicate_terminal_identity:{memory_id}")
                completed[memory_id] = row
            if len(completed) % 320 == 0 or len(completed) == len(projected):
                print(
                    f"identity_progress={len(completed)}/{len(projected)} "
                    f"model_calls={sum(row.get('provider_calls') or 0 for row in calls)} "
                    f"fallbacks={sum(row.get('identity_origin') == 'HARNESS_UNKNOWN_FALLBACK' for row in completed.values())}",
                    flush=True,
                )

    identities = [completed[row["memory_id"]] for row in projected]
    wire_v2.require_exact_terminal_coverage([row["memory_id"] for row in projected], identities)
    call_summary = {
        "runtime_at_start": runtime_at_start,
        "initial_batch_count": (len(projected) + INITIAL_BATCH_SIZE - 1) // INITIAL_BATCH_SIZE,
        "model_calls_unique": sum(row.get("provider_calls") or 0 for row in calls),
        "provider_requests_this_execution": provider_sends,
        "successful_initial_batches": sum(
            row["depth"] == 0 and row["outcome"] == "MODEL_VALIDATED" for row in lineage
        ),
        "recovered_parent_batches": sum(
            row["outcome"] == "LOCAL_FAILURE_SUBDIVIDED" for row in lineage
        ),
        "subdivision_calls": sum(row.get("parent_batch_id") is not None for row in calls),
        "maximum_recovery_depth": max((row["depth"] for row in calls), default=0),
        "single_item_unknown_fallback_count": sum(
            row["identity_origin"] == "HARNESS_UNKNOWN_FALLBACK" for row in identities
        ),
        "retry_count": sum(row.get("retry_count", 0) for row in calls),
        "hosted_calls": sum(bool(row.get("hosted_call")) for row in calls),
        "embedding_calls": 0,
        "retrieval_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "all_requests_loopback": True,
    }
    return identities, calls, lineage, call_summary


def _enrich_candidate_groups(
    groups: list[dict[str, Any]],
    identities: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    identity_by_id = {row["memory_id"]: row for row in identities}
    source_by_id = {row["memory_id"]: row for row in source_rows}
    output = []
    for group in groups:
        observations = []
        for memory_id in group["memory_ids"]:
            identity_row = identity_by_id[memory_id]
            source = source_by_id[memory_id]
            reader_value = source.get("reader_value")
            observed = reader_value.get("observed_at") if isinstance(reader_value, dict) else None
            observations.append(
                {
                    "memory_id": memory_id,
                    "value_text": identity_row["value_text"],
                    "proposition_text": source["proposition_text"],
                    "observed_at": observed,
                    "source_session_id": source.get("source_session_id"),
                    "source_authority": source["source_authority"],
                }
            )
        output.append(
            {
                **group,
                "observed_timestamps": sorted(
                    {row["observed_at"] for row in observations if row["observed_at"]}
                ),
                "source_sessions": sorted(
                    {row["source_session_id"] for row in observations if row["source_session_id"]}
                ),
                "source_authorities": sorted({row["source_authority"] for row in observations}),
                "observations": observations,
            }
        )
    return output


def _kind_counts(identities: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["revision_kind"] for row in identities)
    return {kind: counts.get(kind, 0) for kind in semantic_v1.REVISION_KINDS}


def _render_report(
    identities: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    statistics_row: dict[str, Any],
    call_summary: dict[str, Any],
    case_review: dict[str, Any],
    gate: bool,
) -> str:
    fallback_count = call_summary["single_item_unknown_fallback_count"]
    model_unknown = statistics_row["model_unknown_count"]
    fallback_reasons = statistics_row["fallback_reasons"]
    lines = [
        "# MEM-3B0R - Keyed Identity Map Review",
        "",
        f"Completion gate: `{GATE_NAME}={'YES' if gate else 'NO'}`.",
        "",
        "This stage changes only the identity wire format and batch recovery. It does not change the v1 semantic prompt or decide currentness, supersession, or MemoryStore operations.",
        "",
        "## Coverage and Recovery",
        "",
        f"- Terminal identities: {len(identities)} / 8,112; model-validated: {len(identities) - fallback_count}; conservative UNKNOWN fallbacks: {fallback_count} ({fallback_count / len(identities):.4%}).",
        f"- Model-returned UNKNOWN identities: {model_unknown} ({statistics_row['model_unknown_rate']:.4%} of model-validated identities).",
        f"- Fallback reasons: `{json.dumps(fallback_reasons, sort_keys=True)}`; UNKNOWN among all terminal records: {statistics_row['unknown_rate_all_terminal_records']:.4%}.",
        f"- Initial batches: {call_summary['initial_batch_count']}; unique local model requests: {call_summary['model_calls_unique']}; successful initial batches: {call_summary['successful_initial_batches']}.",
        f"- Recovered parent batches: {call_summary['recovered_parent_batches']}; subdivision calls: {call_summary['subdivision_calls']}; maximum recovery depth: {call_summary['maximum_recovery_depth']}.",
        f"- Retries: {call_summary['retry_count']}; hosted calls: {call_summary['hosted_calls']}; elapsed seconds: {call_summary['elapsed_seconds']:.3f}.",
        "- Embedding, retrieval, reader-answer, and judge calls: 0 each.",
        "- No LongMemEval 102 DEV, TEST, or MedMemoryBench run was performed.",
        "",
        "## Identity Statistics",
        "",
        f"- Revision kinds: `{json.dumps(_kind_counts(identities), sort_keys=True)}`.",
        f"- Unique subject / attribute keys: {statistics_row['unique_subject_keys']} / {statistics_row['unique_attribute_keys']}.",
        f"- Candidate singleton-state revision groups: {len(groups)}; groups with multiple values: {statistics_row['singleton_groups_with_multiple_values']}; cross-session groups: {statistics_row['singleton_groups_spanning_multiple_sessions']}.",
        f"- Singleton orphan rate: {statistics_row['singleton_orphan_rate']:.4f}; fallback fraction: {fallback_count / len(identities):.4f}.",
        "",
        "## Critical Cases",
        "",
        f"`INSTAGRAM_REVISION_SLOT_IDENTIFIED={case_review['instagram_revision_slot_identified']}`",
        "",
        f"`GYM_REVISION_SLOT_IDENTIFIED={case_review['gym_revision_slot_identified']}`",
        "",
        "Case history and timestamps were joined only after identity records, call ledger, recovery lineage, and core identity manifest were frozen. No gold answer or reader output was loaded.",
        "",
        "## Interpretation",
        "",
        "Structural completion and semantic quality are separate. UNKNOWN fallbacks cannot enter candidate singleton groups. Fragmentation/collision flags are deterministic lexical heuristics only; no keys were merged, no groups split, and no MemoryStore side effects occurred.",
        "",
        "Human Reflection decides whether slot quality supports MEM-3B1, whether fragmentation warrants MEM-3B0.1, or whether collision warrants revising the semantic identity layer. No benchmark performance metric was computed.",
        "",
        f"`{GATE_NAME}={'YES' if gate else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def _freeze_core(
    identities: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    call_summary: dict[str, Any],
    preflight: dict[str, Any],
    contract_sha: str,
) -> dict[str, Any]:
    core_sha = {
        "revision_identity_records.jsonl": _freeze_jsonl(IDENTITIES_PATH, identities),
        "identity_call_ledger.jsonl": _freeze_jsonl(CALL_LEDGER_PATH, calls),
        "recovery_lineage.jsonl": _freeze_jsonl(LINEAGE_PATH, lineage),
    }
    core_paths = [IDENTITIES_PATH, CALL_LEDGER_PATH, LINEAGE_PATH]
    core_verify = v1._verify_all_sidecars(core_paths)
    if not core_verify["all_valid"]:
        raise wire_v2.GlobalIdentityFailure("identity_core_sidecar_verification_failed")
    identity_counts = _kind_counts(identities)
    core_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "CORE_OVERLAY_FROZEN_BEFORE_METADATA_JOIN",
        "completion_gate_marker": None,
        "input_inventory_sha256": preflight["frozen_input"]["inventory_sha256"],
        "input_count": 8112,
        "identity_contract_sha256": contract_sha,
        "preflight_sha256": _sha_file(PREFLIGHT_PATH),
        "model_roles": {
            "identity_proposer": "frozen local Qwen3-8B Q4_K_M",
            "memory_system": "frozen FlatProp; not modified",
            "reader_answer_model": None,
            "embedding_model": None,
            "judge_model": None,
        },
        "core_artifact_sha256": core_sha,
        "core_sidecars_valid": True,
        "identity_records_frozen_before_metadata_join": True,
        "identity_call_ledger_frozen_before_metadata_join": True,
        "recovery_lineage_frozen_before_metadata_join": True,
        "revision_kind_counts": identity_counts,
        "model_unknown_count": sum(
            row["identity_origin"] == "MODEL_VALIDATED" and row["revision_kind"] == "UNKNOWN"
            for row in identities
        ),
        "fallback_count": sum(
            row["identity_origin"] == "HARNESS_UNKNOWN_FALLBACK" for row in identities
        ),
        "call_summary": call_summary,
        "diagnostic_case_history_not_inspected": True,
    }
    _freeze_json(CORE_MANIFEST_PATH, core_manifest)
    if not v1._verify_frozen(CORE_MANIFEST_PATH):
        raise wire_v2.GlobalIdentityFailure("core_identity_run_manifest_freeze_failed")
    return core_manifest


def run() -> dict[str, Any]:
    rows, projected, preflight = _validate_saved_preflight()
    contract, contract_hashes = _contract_components()
    contract_sha = contract_hashes["contract_sha256"]
    if _sha_file(CONTRACT_PATH) != contract_sha:
        raise wire_v2.GlobalIdentityFailure("identity_contract_artifact_hash_mismatch")
    code_identity = _source_code_identity()
    identities, calls, lineage, call_summary = _run_identity_calls(
        projected,
        contract_sha,
        code_identity,
        preflight["identity_model_runtime"],
    )
    wire_v2.require_exact_terminal_coverage([row["memory_id"] for row in projected], identities)
    if len(identities) != 8112:
        raise wire_v2.GlobalIdentityFailure("identity_overlay_record_count_mismatch")

    # Freeze identity-only records, batch ledger, recovery lineage, then core manifest.
    core_manifest = _freeze_core(identities, calls, lineage, call_summary, preflight, contract_sha)

    # Only after core freeze do timestamps, source sessions, and case IDs join.
    eligible = [row for row in identities if wire_v2.is_candidate_singleton(row)]
    sparse_groups = semantic_v1.candidate_groups(eligible, projected)
    groups = _enrich_candidate_groups(sparse_groups, identities, rows)
    _freeze_json(GROUPS_PATH, groups)
    statistics_row = semantic_v1.identity_statistics(identities, rows, groups)
    statistics_row.pop("candidate_groups", None)
    model_identities = [
        row for row in identities if row["identity_origin"] == "MODEL_VALIDATED"
    ]
    statistics_row["model_unknown_count"] = core_manifest["model_unknown_count"]
    statistics_row["unknown_fallback_count"] = core_manifest["fallback_count"]
    statistics_row["fallback_fraction"] = core_manifest["fallback_count"] / len(identities)
    statistics_row["unique_subject_keys"] = len(
        {row["subject_key"] for row in model_identities}
    )
    statistics_row["unique_attribute_keys"] = len(
        {row["attribute_key"] for row in model_identities}
    )
    statistics_row["model_unknown_rate"] = (
        core_manifest["model_unknown_count"] / len(model_identities)
        if model_identities
        else 0.0
    )
    statistics_row["unknown_rate_all_terminal_records"] = (
        sum(row["revision_kind"] == "UNKNOWN" for row in identities) / len(identities)
    )
    statistics_row["fallback_reasons"] = dict(
        sorted(
            Counter(
                row["fallback_reason"]
                for row in identities
                if row["identity_origin"] == "HARNESS_UNKNOWN_FALLBACK"
            ).items()
        )
    )
    _freeze_json(STATS_PATH, statistics_row)
    fragmentation = semantic_v1.fragmentation_diagnostics(
        identities, rows, threshold=FRAGMENTATION_THRESHOLD
    )
    collisions = semantic_v1.collision_diagnostics(groups, threshold=COLLISION_JACCARD_THRESHOLD)
    _freeze_json(FRAGMENTATION_PATH, fragmentation)
    _freeze_json(COLLISION_PATH, collisions)
    case_review = v1._case_review(identities, rows, groups, fragmentation, collisions)
    _freeze_json(CASE_REVIEW_PATH, case_review)

    leakage_clear = not any(preflight["label_leakage"].values())
    memory_store_unchanged = (
        _sha_file(v1.INVENTORY_PATH) == preflight["frozen_input"]["inventory_sha256"]
        and _sha_file(v1.MATERIALIZATION_PATH)
        == preflight["frozen_input"]["materialization_ledger_sha256"]
        and v1._verify_frozen(v1.INVENTORY_PATH)
        and v1._verify_frozen(v1.MATERIALIZATION_PATH)
    )
    local_calls_clear = (
        all(
            call_summary[key] == 0
            for key in (
                "hosted_calls",
                "retry_count",
                "embedding_calls",
                "retrieval_calls",
                "reader_calls",
                "judge_calls",
            )
        )
        and call_summary["all_requests_loopback"]
    )
    other_eval_clear = (
        preflight["102_dev_access"] is False
        and preflight["test_access"] is False
        and preflight["medmemorybench_runs"] == 0
    )
    pre_manifest_paths = [
        CONTRACT_PATH,
        PREFLIGHT_PATH,
        IDENTITIES_PATH,
        CALL_LEDGER_PATH,
        LINEAGE_PATH,
        CORE_MANIFEST_PATH,
        GROUPS_PATH,
        STATS_PATH,
        FRAGMENTATION_PATH,
        COLLISION_PATH,
        CASE_REVIEW_PATH,
        REPORT_PROTOCOL_PATH,
    ]
    report_gates = {
        "frozen_flatprop_sha_valid": preflight["frozen_input"]["inventory_sidecar_valid"],
        "terminal_identity_coverage_8112": len(identities) == 8112,
        "exactly_one_terminal_record_per_id": len({row["memory_id"] for row in identities}) == 8112,
        "all_local_failures_resolved": len(lineage) == len(calls)
        and all(
            row["outcome"]
            in {"MODEL_VALIDATED", "HARNESS_UNKNOWN_FALLBACK", "LOCAL_FAILURE_SUBDIVIDED"}
            for row in lineage
        ),
        "zero_retries_and_hosted_calls": call_summary["retry_count"] == 0
        and call_summary["hosted_calls"] == 0,
        "no_benchmark_metadata_leakage": leakage_clear,
        "memory_store_unchanged": memory_store_unchanged,
        "zero_embedding_retrieval_reader_judge": local_calls_clear,
        "zero_102_dev_test_medmemorybench": other_eval_clear,
        "core_frozen_before_metadata_join": core_manifest[
            "identity_records_frozen_before_metadata_join"
        ]
        and core_manifest["identity_call_ledger_frozen_before_metadata_join"]
        and core_manifest["recovery_lineage_frozen_before_metadata_join"],
        "all_pre_report_sha_sidecars_valid": v1._verify_all_sidecars(pre_manifest_paths)[
            "all_valid"
        ],
    }
    gate = all(report_gates.values())
    report = _render_report(identities, groups, statistics_row, call_summary, case_review, gate)
    report_bytes = report.encode("utf-8")
    _freeze(REPORT_PATH, report_bytes)
    _freeze(REPORT_DOC_PATH, report_bytes)
    final_paths = [*pre_manifest_paths, REPORT_PATH, REPORT_DOC_PATH]
    final_sidecars = v1._verify_all_sidecars(final_paths)
    gate = gate and final_sidecars["all_valid"]

    final_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "COMPLETE" if gate else "FAILED",
        "completion_gate_marker": f"{GATE_NAME}={'YES' if gate else 'NO'}",
        "base_commit_sha": preflight["base_commit_sha"],
        "topic_head_at_preflight": preflight["topic_head_sha"],
        "historical_v1_gate": "MEM3B0_REVISION_IDENTITY_FROZEN_DIAGNOSTIC=NO",
        "historical_v1_failure_interpretation": "array ID coverage schema insufficient; not semantic-quality evidence",
        "validator_identity_binding": contract["validator_identity_binding"],
        "validator_identity_sha256": contract["validator_identity_sha256"],
        "gate": report_gates
        | {"all_required_artifact_sha_sidecars_valid": final_sidecars["all_valid"]},
        "identity_counts": statistics_row,
        "call_summary": call_summary,
        "critical_case_results": {
            "INSTAGRAM_REVISION_SLOT_IDENTIFIED": case_review["instagram_revision_slot_identified"],
            "GYM_REVISION_SLOT_IDENTIFIED": case_review["gym_revision_slot_identified"],
        },
        "diagnostic_scope": {
            "102_dev_access": False,
            "test_access": False,
            "medmemorybench_runs": 0,
            "benchmark_performance_metrics": False,
            "memory_store_writes": 0,
            "embedding_calls": 0,
            "retrieval_calls": 0,
            "reader_calls": 0,
            "judge_calls": 0,
        },
        "artifact_sha256": {
            path.relative_to(ROOT).as_posix(): _sha_file(path) for path in final_paths
        },
    }
    _freeze_json(FINAL_MANIFEST_PATH, final_manifest)
    if not v1._verify_frozen(FINAL_MANIFEST_PATH):
        raise wire_v2.GlobalIdentityFailure("final_run_manifest_sidecar_invalid")
    return final_manifest


def _write_execution_failure(exc: Exception) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cached_journals = (
        list(STAGE_CACHE_ROOT.rglob("journal.jsonl")) if STAGE_CACHE_ROOT.is_dir() else []
    )
    failure = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "FAILED_GLOBAL_INTEGRITY_OR_PROVIDER",
        "completion_gate_marker": f"{GATE_NAME}=NO",
        "failure_type": type(exc).__name__,
        "failure": str(exc),
        "preflight_frozen": v1._verify_frozen(PREFLIGHT_PATH),
        "identity_generation_started": bool(cached_journals),
        "local_journal_count": len(cached_journals),
    }
    try:
        _freeze_json(FAILURE_PATH, failure)
    except RuntimeError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight:
            rows, _projected, result = _preflight(write=True)
            print(
                json.dumps(
                    {
                        "preflight": "PASS",
                        "input_rows": len(rows),
                        "projected_fields": list(semantic_v1.INPUT_FIELDS),
                        "batches": result["full_request_preflight"]["initial_batch_count"],
                        "all_context_fit": result["full_request_preflight"][
                            "all_requests_fit_frozen_context"
                        ],
                        "max_prompt_tokens": result["full_request_preflight"][
                            "prompt_token_distribution"
                        ]["max"],
                        "max_schema_tokens": result["full_request_preflight"][
                            "dynamic_schema_token_distribution"
                        ]["max"],
                        "max_request_bytes": result["full_request_preflight"][
                            "maximum_request_size_bytes"
                        ],
                    },
                    indent=2,
                )
            )
            return 0
        result = run()
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "completion_gate_marker": result["completion_gate_marker"],
                    "identity_counts": result["identity_counts"],
                    "call_summary": result["call_summary"],
                },
                indent=2,
            )
        )
        return 0 if result["completion_gate_marker"].endswith("=YES") else 1
    except (wire_v2.GlobalIdentityFailure, writer_v2.WriterRecoveryError, httpx.HTTPError) as exc:
        _write_execution_failure(exc)
        print(json.dumps({"status": "FAILED", "failure": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
