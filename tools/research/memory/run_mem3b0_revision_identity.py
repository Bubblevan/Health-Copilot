"""Run the side-effect-free MEM-3B0 revision identity diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import re
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

from tools.research.memory import flat_proposition_writer_v2 as writer_v2
from tools.research.memory import revision_identity as identity
from tools.research.memory import run_mem2b_rank_aware_projection as mem2b
from tools.research.memory import run_mem3a_flat_proposition as flat

RUN_ID = "mem3b0-revision-identity-frozen-flatprop-20260930"
TOPIC_BRANCH = "mem3b0-revision-identity-20260930"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
SOURCE_RUN_DIR = ROOT / "runs" / "memory" / "mem3" / "mem3a3r-recursive-flatprop-frozen-10-20260929"
INVENTORY_PATH = SOURCE_RUN_DIR / "flatprop_inventory.jsonl"
MATERIALIZATION_PATH = SOURCE_RUN_DIR / "materialization_ledger.jsonl"
UPSTREAM_MANIFEST_PATH = SOURCE_RUN_DIR / "run_manifest.json"
COMPATIBILITY_AUDIT = (
    ROOT / "docs" / "research" / "memory" / "medmemorybench_compatibility_audit.md"
)
CONTRACT_PATH = RUN_DIR / "revision_identity_contract.json"
PREFLIGHT_PATH = RUN_DIR / "identity_preflight.json"
IDENTITIES_PATH = RUN_DIR / "revision_identity_records.jsonl"
CALL_LEDGER_PATH = RUN_DIR / "identity_call_ledger.jsonl"
GROUPS_PATH = RUN_DIR / "candidate_revision_groups.json"
STATS_PATH = RUN_DIR / "identity_statistics.json"
FRAGMENTATION_PATH = RUN_DIR / "key_fragmentation_diagnostics.json"
COLLISION_PATH = RUN_DIR / "key_collision_diagnostics.json"
CASE_REVIEW_PATH = RUN_DIR / "critical_case_review.json"
CORE_MANIFEST_PATH = RUN_DIR / "identity_run_manifest.json"
FINAL_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
REPORT_PATH = RUN_DIR / "report.md"
REPORT_DOC_PATH = ROOT / "docs" / "research" / "memory" / "mem_3b0_revision_identity_review.md"
REPORT_PROTOCOL_PATH = (
    ROOT / "docs" / "research" / "memory" / "mem_3b0_revision_identity_protocol.md"
)
READER_ENDPOINT = "http://127.0.0.1:8081/v1"
READER_MODEL = "health-memory-qwen3-8b"
READER_MODEL_PATH = Path(r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
READER_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
READER_SERVER_SHA256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
READER_SERVER_VERSION = "llama.cpp version 10068 (571d0d540)"
BATCH_SIZE = 32
MAX_COMPLETION_TOKENS = 8192
FRAGMENTATION_THRESHOLD = 0.82
COLLISION_JACCARD_THRESHOLD = 0.08
GATE_NAME = "MEM3B0_REVISION_IDENTITY_FROZEN_DIAGNOSTIC"
STAGE_CACHE_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Health-Copilot" / RUN_ID
)


def _sha_file(path: Path) -> str:
    return writer_v2.sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid_jsonl:{path.name}:{index}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"invalid_jsonl_row:{path.name}:{index}")
        rows.append(row)
    return rows


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _verify_frozen(path: Path) -> bool:
    sidecar = _sidecar_path(path)
    if not path.is_file() or not sidecar.is_file():
        return False
    return sidecar.read_text(encoding="ascii").strip().split() == [_sha_file(path), path.name]


def _freeze(path: Path, payload: bytes, *, exact_existing: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_path(path)
    if path.exists() or sidecar.exists():
        if not _verify_frozen(path) or (exact_existing and path.read_bytes() != payload):
            raise RuntimeError(f"frozen_artifact_mismatch:{path}")
        return _sha_file(path)
    writer_v2.atomic_write_bytes(path, payload)
    writer_v2.atomic_write_bytes(sidecar, f"{_sha_file(path)}  {path.name}\n".encode("ascii"))
    if not _verify_frozen(path):
        raise RuntimeError(f"freeze_verification_failed:{path}")
    return _sha_file(path)


def _freeze_json(path: Path, value: Any) -> str:
    return _freeze(path, _json_bytes(value))


def _freeze_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    payload = b"".join(identity.canonical_json(row) + b"\n" for row in rows)
    return _freeze(path, payload)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_counts(base: str, head: str) -> tuple[int, int]:
    left, right = _git("rev-list", "--left-right", "--count", f"{base}...{head}").split()
    return int(right), int(left)


def _hash_sidecar_input(path: Path) -> tuple[str, bool]:
    digest = _sha_file(path)
    sidecar = _sidecar_path(path)
    if not sidecar.is_file():
        return digest, False
    fields = sidecar.read_text(encoding="ascii").strip().split()
    return digest, fields == [digest, path.name]


def _key_value(command_line: str, key: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(key)}(?:\s+|=)(?:\"([^\"]+)\"|(\S+))", command_line)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _reader_process() -> dict[str, Any]:
    script = (
        "$p=@(Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | "
        "Where-Object { $_.CommandLine -match '--host\\s+127\\.0\\.0\\.1' -and "
        "$_.CommandLine -match '--port\\s+8081' } | "
        "Select-Object ProcessId,ExecutablePath,CommandLine); "
        "ConvertTo-Json -InputObject $p -Compress -Depth 4"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = result.stdout.strip()
    parsed = json.loads(raw) if raw else []
    processes = parsed if isinstance(parsed, list) else [parsed]
    if len(processes) != 1:
        raise RuntimeError(f"expected_one_loopback_reader_process:{len(processes)}")
    return processes[0]


def _reader_runtime(client: httpx.Client) -> dict[str, Any]:
    if urlsplit(READER_ENDPOINT).hostname != "127.0.0.1":
        raise RuntimeError("reader_endpoint_not_loopback")
    if _sha_file(READER_MODEL_PATH) != READER_MODEL_SHA256:
        raise RuntimeError("frozen_qwen3_8b_artifact_sha_mismatch")
    mem2a_manifest_path = flat.mem2c.MEM2A_DIR / "run_manifest.json"
    if not flat._verify_frozen(mem2a_manifest_path):
        raise RuntimeError("frozen_mem2a_reader_manifest_invalid")
    runtime = mem2b._verify_reader(client, _read_json(mem2a_manifest_path))
    if (
        runtime.get("endpoint") != READER_ENDPOINT
        or runtime.get("loopback_only") is not True
        or runtime.get("model_sha256") != READER_MODEL_SHA256
        or runtime.get("server_build") != "llama.cpp 10068 (571d0d540)"
        or runtime.get("context_tokens") != 131072
    ):
        raise RuntimeError("frozen_reader_runtime_identity_invalid")

    process = _reader_process()
    executable = Path(process.get("ExecutablePath") or "")
    command_line = process.get("CommandLine") or ""
    binary_sha = _sha_file(executable)
    version = subprocess.check_output(
        [str(executable), "--version"], text=True, stderr=subprocess.STDOUT
    ).strip()
    expected_args = {
        "--host": "127.0.0.1",
        "--port": "8081",
        "--ctx-size": "131072",
        "--rope-scaling": "yarn",
        "--rope-scale": "4",
        "--cache-type-k": "q4_0",
        "--cache-type-v": "q4_0",
        "--n-gpu-layers": "99",
        "--flash-attn": "on",
        "--parallel": "1",
        "-m": str(READER_MODEL_PATH),
    }
    actual_args = {key: _key_value(command_line, key) for key in expected_args}
    if binary_sha != READER_SERVER_SHA256:
        raise RuntimeError("frozen_llama_server_binary_sha_mismatch")
    if "10068" not in version or "571d0d540" not in version:
        raise RuntimeError(f"frozen_llama_server_version_mismatch:{version}")
    if actual_args != expected_args or "--no-kv-offload" in command_line:
        raise RuntimeError(f"frozen_llama_server_arguments_mismatch:{actual_args}")

    models_response = client.get(f"{READER_ENDPOINT}/models")
    models_response.raise_for_status()
    models = models_response.json().get("data", [])
    model_ids = [row.get("id") for row in models if isinstance(row, dict)]
    model_path_normalized = str(READER_MODEL_PATH).replace("/", "\\").casefold()
    if not any(
        str(value).replace("/", "\\").casefold() == model_path_normalized for value in model_ids
    ):
        raise RuntimeError("loopback_reader_does_not_serve_frozen_gguf")
    props_response = client.get("http://127.0.0.1:8081/props")
    props_response.raise_for_status()
    props = props_response.json()
    if (
        props.get("model_path") != str(READER_MODEL_PATH)
        or props.get("model_ftype") != "Q4_K - Medium"
        or props.get("build_info") != "b10068-571d0d540"
        or props.get("default_generation_settings", {}).get("n_ctx") != 131072
        or props.get("total_slots") != 1
    ):
        raise RuntimeError("loopback_reader_props_mismatch")
    slots_response = client.get("http://127.0.0.1:8081/slots")
    slots_response.raise_for_status()
    slots = slots_response.json()
    if not isinstance(slots, list) or len(slots) != 1 or slots[0].get("is_processing"):
        raise RuntimeError("loopback_reader_slot_busy_or_invalid")

    return {
        **runtime,
        "server_pid": int(process["ProcessId"]),
        "server_executable": str(executable),
        "server_binary_sha256": binary_sha,
        "server_version_output": version,
        "server_arguments": actual_args,
        "model_artifact": str(READER_MODEL_PATH),
        "local_only": True,
        "proxy_environment_used": False,
        "thinking": False,
        "temperature": 0,
        "seed": 42,
    }


def _load_source() -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    if not _verify_frozen(UPSTREAM_MANIFEST_PATH):
        raise RuntimeError("frozen_mem3a3r_run_manifest_sidecar_invalid")
    upstream = _read_json(UPSTREAM_MANIFEST_PATH)
    if (
        upstream.get("completion_gate_marker")
        != "MEM3A3R_RECURSIVE_FLATPROP_FROZEN_10_DIAGNOSTIC=YES"
        or upstream.get("status") != "COMPLETE"
        or upstream.get("artifact_sha256", {}).get("flatprop_inventory.jsonl")
        != _sha_file(INVENTORY_PATH)
    ):
        raise RuntimeError("frozen_mem3a3r_source_gate_invalid")
    inventory_sha, sidecar_valid = _hash_sidecar_input(INVENTORY_PATH)
    if not sidecar_valid:
        raise RuntimeError("frozen_flatprop_inventory_sidecar_invalid")
    materialization_sha, materialization_sidecar_valid = _hash_sidecar_input(MATERIALIZATION_PATH)
    if not materialization_sidecar_valid:
        raise RuntimeError("frozen_materialization_ledger_sidecar_invalid")
    rows = _read_jsonl(INVENTORY_PATH)
    if len(rows) != 8112:
        raise RuntimeError(f"unexpected_flatprop_count:{len(rows)}")
    projected = [identity.project_input_row(row) for row in rows]
    ids = [row["memory_id"] for row in projected]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate_flatprop_memory_id")
    projected.sort(key=lambda row: (row["scope_id"], row["memory_id"]))
    by_id = {row["memory_id"]: row for row in rows}
    rows = [by_id[row["memory_id"]] for row in projected]
    invalid_state = [
        row["memory_id"]
        for row in rows
        if row.get("status") != "active"
        or row.get("version") != 1
        or row.get("supersedes_id") is not None
    ]
    if invalid_state:
        raise RuntimeError(f"frozen_flatprop_state_contract_invalid:{len(invalid_state)}")
    if len(projected) != 8112:
        raise RuntimeError("projected_identity_input_coverage_invalid")
    scope_counts = Counter(row["scope_id"] for row in projected)
    data = {
        "inventory_path": INVENTORY_PATH.relative_to(ROOT).as_posix(),
        "inventory_sha256": inventory_sha,
        "upstream_run_manifest_sha256": _sha_file(UPSTREAM_MANIFEST_PATH),
        "inventory_sidecar_valid": sidecar_valid,
        "inventory_rows": len(rows),
        "unique_memory_ids": len(set(ids)),
        "projected_identity_input_sha256": identity.sha256_bytes(
            b"\n".join(identity.canonical_json(row) for row in projected)
        ),
        "canonical_order": "scope_id then memory_id ascending",
        "scope_count": len(scope_counts),
        "scope_row_counts": dict(sorted(scope_counts.items())),
        "materialization_ledger_path": MATERIALIZATION_PATH.relative_to(ROOT).as_posix(),
        "materialization_ledger_sha256": materialization_sha,
        "materialization_ledger_sidecar_valid": materialization_sidecar_valid,
        "all_source_memories_active_version_1_no_supersedes": True,
    }
    return rows, projected, data


def _contract_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": "revision-identity-v1",
        "research_question": "Can a query-independent semantic identity layer identify candidate mutable state slots in frozen FlatProp?",
        "input_fields": list(identity.INPUT_FIELDS),
        "forbidden_input_metadata": [
            "benchmark_question",
            "question_date",
            "gold_answer",
            "answer_session_ids",
            "question_type",
            "reader_prediction",
            "retrieval_rank",
            "correctness",
            "observed_at",
        ],
        "output_fields": list(identity.OUTPUT_FIELDS),
        "revision_kinds": list(identity.REVISION_KINDS),
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
            "initial_size": BATCH_SIZE,
            "order": "scope_id then memory_id ascending",
            "response_id_enum": "exact IDs from the current batch",
            "on_finish_reason_length": "discard incomplete parent output and recursively bisect contiguously, left before right",
            "single_item_length": "fatal IRREDUCIBLE_IDENTITY_OUTPUT_FAILURE",
            "retry_policy": "zero retries; subdivision is not a retry",
        },
        "key_normalization": {
            "unicode": "NFKC",
            "surface": "strip, lowercase, whitespace and hyphens to underscore, collapse repeated underscores",
            "syntax": "^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
            "forbidden_temporal_key_tokens": sorted(identity.FORBIDDEN_KEY_TOKENS),
            "alias_dictionary": False,
            "fuzzy_merge": False,
        },
        "candidate_grouping": {
            "kinds": ["SINGLETON_STATE"],
            "key": ["scope_id", "subject_key", "attribute_key"],
            "materialization_side_effects": False,
            "all_source_memories_remain_active_version_1": True,
        },
        "diagnostic_heuristics": {
            "slot_fragmentation_similarity": "difflib.SequenceMatcher over subject_key + attribute_key within same scope",
            "fragmentation_threshold": FRAGMENTATION_THRESHOLD,
            "collision_similarity": "content-token Jaccard over proposition text for distinct values in one exact group",
            "collision_threshold_at_or_below": COLLISION_JACCARD_THRESHOLD,
            "heuristic_only_no_automatic_merge_or_split": True,
        },
        "calls_forbidden": ["embedding", "retrieval", "reader_answer", "judge", "hosted_api"],
        "system_prompt": identity.SYSTEM_PROMPT,
        "system_prompt_sha256": identity.sha256_bytes(identity.SYSTEM_PROMPT.encode("utf-8")),
    }


def _source_code_identity() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "identity_module": Path(identity.__file__),
        "response_journal": Path(writer_v2.__file__),
        "runtime_verifier": Path(mem2b.__file__),
        "reader_adapter": Path(flat.__file__),
        "server_launcher": ROOT / "tools" / "research" / "memory" / "start_mem1_local_reader.ps1",
        "protocol_doc": REPORT_PROTOCOL_PATH,
    }
    return {name: _sha_file(path) for name, path in paths.items()}


def _enrich_candidate_groups(
    groups: list[dict[str, Any]],
    identities: list[dict[str, str]],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    identity_by_id = {row["memory_id"]: row for row in identities}
    source_by_id = {row["memory_id"]: row for row in source_rows}
    enriched = []
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
        enriched.append(
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
    return enriched


def _preflight(
    *, write: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if _git("branch", "--show-current") != TOPIC_BRANCH:
        raise RuntimeError(f"unexpected_mem3b0_topic_branch:{_git('branch', '--show-current')}")
    if not _verify_frozen(REPORT_PROTOCOL_PATH):
        raise RuntimeError("mem3b0_protocol_sha_sidecar_invalid")
    rows, projected, input_data = _load_source()
    audit_text = COMPATIBILITY_AUDIT.read_text(encoding="utf-8")
    if "MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES" not in audit_text:
        raise RuntimeError("required_medmemorybench_compatibility_gate_missing")
    if "LONGMEM_MEMORY_MECHANISM_FROZEN=NO" not in audit_text:
        raise RuntimeError("expected_longmem_memory_mechanism_gate_changed")
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    origin_main = _git("rev-parse", "origin/main")
    base_main = _git("merge-base", "HEAD", "origin/main")
    ahead, behind = _git_counts("origin/main", "HEAD")
    tracked_clean = (
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False).returncode
        == 0
    )
    staged_clean = (
        subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False).returncode
        == 0
    )
    if not tracked_clean or not staged_clean:
        raise RuntimeError("mem3b0_preflight_requires_clean_tracked_and_staged_worktree")
    untracked_paths = _git("ls-files", "--others", "--exclude-standard").splitlines()
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=5.0), trust_env=False) as client:
        runtime = _reader_runtime(client)
    runtime.pop("server_pid", None)
    contract = _contract_payload()
    contract_bytes = _json_bytes(contract)
    contract_sha = identity.sha256_bytes(contract_bytes)
    existing_contract_sha = _freeze(CONTRACT_PATH, contract_bytes)
    preflight = {
        "schema_version": 1,
        "stage": RUN_ID,
        "base_commit_sha": base_main,
        "branch": branch,
        "topic_head_sha": head,
        "origin_main_sha": origin_main,
        "merge_base_sha": base_main,
        "ahead_of_origin_main": ahead,
        "behind_origin_main": behind,
        "tracked_worktree_clean": tracked_clean,
        "staged_worktree_clean": staged_clean,
        "untracked_worktree_paths": untracked_paths,
        "working_tree_clean": tracked_clean and staged_clean and not untracked_paths,
        "tracked_worktree_clean_required": True,
        "global_gates": {
            "medmemorybench_compatibility_audited": True,
            "longmem_memory_mechanism_frozen": False,
        },
        "frozen_input": input_data,
        "identity_contract_sha256": contract_sha,
        "identity_contract_frozen_sha256": existing_contract_sha,
        "identity_model_runtime": runtime,
        "source_code_identity_sha256": _source_code_identity(),
        "call_separation": {
            "identity_proposer_calls": "local loopback Qwen3-8B only",
            "embedding_calls": 0,
            "retrieval_calls": 0,
            "reader_calls": 0,
            "judge_calls": 0,
            "hosted_calls": 0,
        },
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
        "102_dev_access": False,
        "test_access": False,
        "medmemorybench_runs": 0,
    }
    batch = projected[:BATCH_SIZE]
    request = identity.build_request(batch, model=READER_MODEL, max_tokens=MAX_COMPLETION_TOKENS)
    user_payload = json.loads(request["messages"][1]["content"])
    if set(user_payload) != {"memories"} or any(
        set(row) != set(identity.INPUT_FIELDS) for row in user_payload["memories"]
    ):
        raise RuntimeError("identity_request_projection_invalid")
    forbidden = {
        "question",
        "question_id",
        "question_date",
        "gold",
        "answer",
        "answer_session_ids",
        "question_type",
        "reader_prediction",
        "retrieval_rank",
        "correctness",
        "observed_at",
    }
    if forbidden.intersection(user_payload) or any(
        forbidden.intersection(row) for row in user_payload["memories"]
    ):
        raise RuntimeError("benchmark_metadata_leaked_into_identity_request")
    preflight["sample_request_sha256"] = identity.sha256_bytes(identity.canonical_json(request))
    preflight["sample_request_field_projection_valid"] = True
    preflight["source_identity_complete"] = True
    if write:
        _freeze_json(PREFLIGHT_PATH, preflight)
    return rows, projected, preflight


def _validate_saved_preflight() -> tuple[
    list[dict[str, Any]], list[dict[str, str]], dict[str, Any]
]:
    if not _verify_frozen(CONTRACT_PATH) or not _verify_frozen(PREFLIGHT_PATH):
        raise RuntimeError("identity_contract_or_preflight_not_frozen")
    rows, projected, current = _preflight(write=False)
    frozen = _read_json(PREFLIGHT_PATH)
    # _preflight verifies exact frozen files and only allows runtime PID to vary.
    runtime_before = frozen["identity_model_runtime"]
    runtime_now = current["identity_model_runtime"]
    stable_fields = set(runtime_before) - {"server_pid"}
    if any(runtime_before.get(field) != runtime_now.get(field) for field in stable_fields):
        raise RuntimeError("identity_runtime_changed_after_preflight")
    if frozen["frozen_input"] != current["frozen_input"]:
        raise RuntimeError("frozen_flatprop_input_changed_after_preflight")
    if frozen["identity_contract_sha256"] != current["identity_contract_sha256"]:
        raise RuntimeError("identity_contract_changed_after_preflight")
    if frozen["source_code_identity_sha256"] != current["source_code_identity_sha256"]:
        raise RuntimeError("identity_source_code_changed_after_preflight")
    return rows, projected, frozen


def _batch_id(batch: list[dict[str, str]]) -> str:
    return identity.sha256_bytes(identity.canonical_json([row["memory_id"] for row in batch]))


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
    projected: list[dict[str, str]], contract_sha: str, source_code_sha: dict[str, str]
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    if urlsplit(READER_ENDPOINT).hostname != "127.0.0.1":
        raise RuntimeError("hosted_identity_fallback_forbidden")
    STAGE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, str]] = {}
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()
    provider_sends = 0
    with httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0), trust_env=False) as client:
        runtime_at_start = _reader_runtime(client)

        def invoke(batch: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
            batch_id = _batch_id(batch)
            request = identity.build_request(
                batch, model=READER_MODEL, max_tokens=MAX_COMPLETION_TOKENS
            )
            request_sha = identity.sha256_bytes(identity.canonical_json(request))
            if set(request) != {
                "model",
                "messages",
                "temperature",
                "seed",
                "max_tokens",
                "stream",
                "chat_template_kwargs",
                "response_format",
            }:
                raise RuntimeError("identity_request_envelope_unexpected_fields")

            def provider(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
                nonlocal provider_sends
                if req != request:
                    raise RuntimeError("identity_provider_request_identity_mismatch")
                if urlsplit(READER_ENDPOINT).hostname != "127.0.0.1":
                    raise RuntimeError("identity_provider_endpoint_not_loopback")
                provider_sends += 1
                response = client.post(f"{READER_ENDPOINT}/chat/completions", json=req)
                return response.status_code, response.content, response.headers.get("content-type")

            def packet_validator(raw: bytes, catalog: list[dict[str, str]]) -> dict[str, Any]:
                try:
                    return identity.validate_identity_response(raw, catalog)
                except identity.IdentityContractError as exc:
                    raise writer_v2.ExtractionValidationError(
                        "IDENTITY_CONTRACT_FAILURE", detail=str(exc)
                    ) from exc

            batch_identity_sha = identity.sha256_bytes(
                identity.canonical_json([row["memory_id"] for row in batch])
            )
            try:
                packet, ledger = writer_v2.execute_or_resume(
                    request=request,
                    catalog=batch,
                    session_identity_sha256=batch_identity_sha,
                    prompt_sha256=identity.sha256_bytes(identity.SYSTEM_PROMPT.encode("utf-8")),
                    contract_sha256=contract_sha,
                    local_cache_root=STAGE_CACHE_ROOT,
                    provider=provider,
                    stage_identity=RUN_ID,
                    model_sha256=READER_MODEL_SHA256,
                    dynamic_schema_sha256=identity.sha256_bytes(
                        identity.canonical_json(request["response_format"])
                    ),
                    unwrap_source_sha256=source_code_sha["response_journal"],
                    packet_validator=packet_validator,
                    packet_validator_sha256=source_code_sha["identity_module"],
                )
            except writer_v2.WriterQualificationFailure as exc:
                ledger_row = {
                    **_ledger_fields(exc.ledger),
                    "batch_id": batch_id,
                    "memory_ids": [row["memory_id"] for row in batch],
                    "request_sha256": request_sha,
                    "hosted_call": False,
                }
                if (
                    exc.ledger.get("finish_reason") == "length"
                    and exc.ledger.get("failure_code") == "COMPLETION_TRUNCATED"
                ):
                    raise identity.IdentityLengthStop(ledger_row) from exc
                raise RuntimeError(
                    f"fatal_identity_batch_failure:{batch_id}:{exc.ledger.get('failure_code')}"
                ) from exc
            rows_out = packet.get("identities") if isinstance(packet, dict) else None
            if not isinstance(rows_out, list) or len(rows_out) != len(batch):
                raise RuntimeError(f"identity_packet_shape_mismatch:{batch_id}")
            ledger_row = {
                **_ledger_fields(ledger),
                "batch_id": batch_id,
                "memory_ids": [row["memory_id"] for row in batch],
                "request_sha256": request_sha,
                "hosted_call": False,
            }
            return rows_out, ledger_row

        for start in range(0, len(projected), BATCH_SIZE):
            batch = projected[start : start + BATCH_SIZE]
            rows_out, batch_attempts = identity.execute_with_bisection(batch, invoke)
            for row in rows_out:
                if row["memory_id"] in completed:
                    raise RuntimeError(f"duplicate_terminal_identity:{row['memory_id']}")
                completed[row["memory_id"]] = row
            attempts.extend(batch_attempts)
            if len(attempts) % 10 == 0 or len(completed) == len(projected):
                print(
                    f"identity_progress={len(completed)}/{len(projected)} "
                    f"attempts={len(attempts)} elapsed_seconds={time.perf_counter() - started:.1f}",
                    flush=True,
                )

    identities = [completed[row["memory_id"]] for row in projected]
    if len(identities) != len(projected) or set(completed) != {
        row["memory_id"] for row in projected
    }:
        raise RuntimeError("identity_coverage_incomplete")
    call_summary = {
        "runtime_at_start": runtime_at_start,
        "initial_batches": (len(projected) + BATCH_SIZE - 1) // BATCH_SIZE,
        "total_provider_requests_unique": sum(row.get("provider_calls") or 0 for row in attempts),
        "provider_requests_this_execution": provider_sends,
        "overflow_parent_count": sum(row["status"] == "OVERFLOW_PARENT" for row in attempts),
        "max_bisection_depth": max((row["depth"] for row in attempts), default=0),
        "retry_count": sum(row.get("retry_count", 0) for row in attempts),
        "hosted_calls": sum(bool(row.get("hosted_call")) for row in attempts),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "all_requests_loopback": True,
        "embedding_calls": 0,
        "retrieval_calls": 0,
        "reader_calls": 0,
        "judge_calls": 0,
    }
    return identities, attempts, call_summary


def _sample_by_kind(
    identities: list[dict[str, str]], source_rows: list[dict[str, Any]], kind: str, limit: int = 5
) -> list[dict[str, Any]]:
    source_by_id = {row["memory_id"]: row for row in source_rows}
    candidates = [row for row in identities if row["revision_kind"] == kind]
    candidates.sort(key=lambda row: (source_by_id[row["memory_id"]]["scope_id"], row["memory_id"]))
    selected: list[dict[str, Any]] = []
    seen_scopes: set[str] = set()
    for row in candidates:
        source = source_by_id[row["memory_id"]]
        scope = source["scope_id"]
        if scope not in seen_scopes:
            selected.append(_review_example(row, source))
            seen_scopes.add(scope)
            if len(selected) == limit:
                return selected
    selected_ids = {row["memory_id"] for row in selected}
    for row in candidates:
        if row["memory_id"] not in selected_ids:
            selected.append(_review_example(row, source_by_id[row["memory_id"]]))
            if len(selected) == limit:
                break
    return selected


def _review_example(identity_row: dict[str, str], source: dict[str, Any]) -> dict[str, Any]:
    return {
        **identity_row,
        "scope_id": source["scope_id"],
        "proposition_text": source["proposition_text"],
        "source_authority": source["source_authority"],
        "source_session_id": source.get("source_session_id"),
        "observed_at": source.get("reader_value", {}).get("observed_at")
        if isinstance(source.get("reader_value"), dict)
        else None,
    }


def _case_review(
    identities: list[dict[str, str]],
    source_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    fragmentation: dict[str, Any],
    collisions: dict[str, Any],
) -> dict[str, Any]:
    identity_by_id = {row["memory_id"]: row for row in identities}
    group_by_id = {memory_id: group for group in groups for memory_id in group["memory_ids"]}
    instagram = []
    gym = []
    for source in source_rows:
        qid = source.get("question_id")
        text = source.get("proposition_text", "")
        lowered = text.casefold()
        if (
            qid == "1cea1afa"
            and "instagram" in lowered
            and "follower" in lowered
            and re.search(r"\b(?:500|600)\b", lowered)
        ):
            instagram.append(_review_example(identity_by_id[source["memory_id"]], source))
        if qid == "c4ea545c" and any(
            marker in lowered
            for marker in (
                "gym",
                "workout",
                "tuesday",
                "thursday",
                "saturday",
                "four times",
                "per week",
            )
        ):
            gym.append(_review_example(identity_by_id[source["memory_id"]], source))

    instagram_groups = {
        group_by_id[row["memory_id"]]["group_id"]
        for row in instagram
        if row["memory_id"] in group_by_id
    }
    instagram_values = {
        str(value)
        for row in instagram
        for value in re.findall(r"\b(?:500|600)\b", row["value_text"])
    }
    instagram_kinds = {row["revision_kind"] for row in instagram}
    instagram_slots = {(row["subject_key"], row["attribute_key"]) for row in instagram}
    instagram_values = instagram_values | {
        str(value)
        for row in instagram
        for value in re.findall(r"\b(?:500|600)\b", row["proposition_text"])
    }
    if (
        {"500", "600"}.issubset(instagram_values)
        and len(instagram_groups) == 1
        and instagram_kinds == {"SINGLETON_STATE"}
        and len(instagram_slots) == 1
        and len({row["value_text"] for row in instagram}) >= 2
    ):
        instagram_result = "YES"
    elif instagram and instagram_values:
        instagram_result = "PARTIAL"
    else:
        instagram_result = "NO"

    gym_by_text = {
        "schedule": [
            row
            for row in gym
            if all(
                word in row["proposition_text"].casefold()
                for word in ("tuesday", "thursday", "saturday")
            )
        ],
        "frequency": [
            row
            for row in gym
            if "four times" in row["proposition_text"].casefold()
            and (
                "week" in row["proposition_text"].casefold()
                or "weekly" in row["proposition_text"].casefold()
            )
        ],
    }
    gym_group_ids = {
        group_by_id[row["memory_id"]]["group_id"]
        for examples in gym_by_text.values()
        for row in examples
        if row["memory_id"] in group_by_id
    }
    gym_kind_set = {row["revision_kind"] for examples in gym_by_text.values() for row in examples}
    gym_values = {row["value_text"] for examples in gym_by_text.values() for row in examples}
    gym_slots = {
        (row["subject_key"], row["attribute_key"])
        for examples in gym_by_text.values()
        for row in examples
    }
    if (
        all(gym_by_text.values())
        and len(gym_group_ids) == 1
        and gym_kind_set == {"SINGLETON_STATE"}
        and len(gym_slots) == 1
        and len(gym_values) >= 2
    ):
        gym_result = "YES"
    elif any(gym_by_text.values()):
        gym_result = "PARTIAL"
    else:
        gym_result = "NO"

    def matching_examples(patterns: tuple[str, ...], limit: int = 8) -> list[dict[str, Any]]:
        matches = []
        for source in source_rows:
            text = source.get("proposition_text", "").casefold()
            if all(pattern in text for pattern in patterns):
                matches.append(_review_example(identity_by_id[source["memory_id"]], source))
        matches.sort(key=lambda row: (row["scope_id"], row["memory_id"]))
        return matches[:limit]

    multi_value_groups = [row for row in groups if row["distinct_value_text_count"] > 1]
    multi_value_groups.sort(
        key=lambda row: (
            -row["proposition_count"],
            row["scope_id"],
            row["subject_key"],
            row["attribute_key"],
        )
    )
    fragmentation_pairs = fragmentation["flagged_pairs"][:10]
    collision_groups = collisions["flagged_groups"][:10]
    return {
        "schema_version": 1,
        "labels_joined_after_core_freeze": True,
        "instagram_case_id": "1cea1afa",
        "instagram_revision_slot_identified": instagram_result,
        "instagram_observations": instagram,
        "gym_case_id": "c4ea545c",
        "gym_revision_slot_identified": gym_result,
        "gym_observations": gym,
        "gym_target_phrase_matches": gym_by_text,
        "multi_value_singleton_group_examples": multi_value_groups[:10],
        "suspected_fragmented_slot_pairs": fragmentation_pairs,
        "suspected_collision_groups": collision_groups,
        "set_state_examples": _sample_by_kind(identities, source_rows, "SET_STATE"),
        "event_examples": _sample_by_kind(identities, source_rows, "EVENT"),
        "unknown_examples": _sample_by_kind(identities, source_rows, "UNKNOWN"),
        "negative_controls": {
            "instagram_follower_count": matching_examples(("instagram", "follower")),
            "instagram_content_strategy": matching_examples(("instagram", "content")),
            "gym_frequency": matching_examples(("gym", "week")),
            "exercise_form_advice": matching_examples(("form", "exercise")),
            "water_change_frequency": matching_examples(("water", "change")),
            "interpretation": "keyword-selected diagnostic candidates only; lexical overlap is not a semantic verdict",
        },
        "event_control": {
            "representative_events": _sample_by_kind(identities, source_rows, "EVENT"),
            "event_groups_entered_candidate_singleton_groups": any(
                identity_by_id[memory_id]["revision_kind"] == "EVENT"
                for group in groups
                for memory_id in group["memory_ids"]
            ),
        },
        "heuristics_are_noncausal": True,
        "automatic_key_merges": 0,
        "automatic_group_splits": 0,
    }


def _render_report(
    identities: list[dict[str, str]],
    groups: list[dict[str, Any]],
    statistics: dict[str, Any],
    calls: dict[str, Any],
    case_review: dict[str, Any],
    gate: bool,
) -> str:
    kind_counts = statistics["counts_by_revision_kind"]
    lines = [
        "# MEM-3B0 - Revision Identity Review",
        "",
        f"Completion gate: `{GATE_NAME}={'YES' if gate else 'NO'}`.",
        "",
        "This is a query-independent semantic slot-identity diagnostic over frozen FlatProp. It assigns no current/old status, performs no supersession, and does not modify MemoryStore. The output is an evaluator-side overlay only.",
        "",
        "## Frozen Input and Calls",
        "",
        f"- Frozen FlatProp propositions: {len(identities)}; identity records: {len(identities)}.",
        f"- Kind counts: `SINGLETON_STATE={kind_counts['SINGLETON_STATE']}`, `SET_STATE={kind_counts['SET_STATE']}`, `EVENT={kind_counts['EVENT']}`, `NON_REVISIONAL={kind_counts['NON_REVISIONAL']}`, `UNKNOWN={kind_counts['UNKNOWN']}`.",
        f"- Candidate singleton groups: {len(groups)}; groups with multiple values: {statistics['singleton_groups_with_multiple_values']}; groups spanning multiple sessions: {statistics['singleton_groups_spanning_multiple_sessions']}.",
        f"- Singleton orphan rate: {statistics['singleton_orphan_rate']:.4f}; UNKNOWN rate: {statistics['unknown_rate']:.4f}.",
        f"- Local Qwen proposer requests: {calls['total_provider_requests_unique']} unique; overflow parents: {calls['overflow_parent_count']}; recursive depth: {calls['max_bisection_depth']}; retries: {calls['retry_count']}; hosted calls: {calls['hosted_calls']}.",
        "- Embedding, retrieval, reader-answer, and judge calls: 0 each.",
        "- No LongMemEval 102 DEV, TEST, or MedMemoryBench run was performed.",
        "",
        "## Critical Cases",
        "",
        f"`INSTAGRAM_REVISION_SLOT_IDENTIFIED={case_review['instagram_revision_slot_identified']}`",
        "",
        f"Instagram target proposition rows found: {len(case_review['instagram_observations'])}.",
        "",
        f"`GYM_REVISION_SLOT_IDENTIFIED={case_review['gym_revision_slot_identified']}`",
        "",
        f"Gym target-candidate rows found: {len(case_review['gym_observations'])}.",
        "",
        "The case review joins proposition-side question IDs, timestamps, and source sessions only after identity records, candidate groups, call ledger, and core identity manifest are frozen. No gold answers or reader outputs are loaded.",
        "",
        "## Human Review Packet",
        "",
        f"- Multi-value SINGLETON_STATE groups supplied: {len(case_review['multi_value_singleton_group_examples'])}.",
        f"- Suspected fragmented-key pairs supplied: {len(case_review['suspected_fragmented_slot_pairs'])} (surface-similarity heuristic only).",
        f"- Suspected collision groups supplied: {len(case_review['suspected_collision_groups'])} (content-token Jaccard heuristic only).",
        f"- SET_STATE / EVENT / UNKNOWN examples supplied: {len(case_review['set_state_examples'])} / {len(case_review['event_examples'])} / {len(case_review['unknown_examples'])}.",
        "- Fragmentation candidates were not merged; collision candidates were not split.",
        "- Event examples are historical occurrences and do not enter candidate singleton groups.",
        "- Preference examples are not forced to SINGLETON_STATE; SET_STATE is allowed.",
        "",
        "The JSON review packet and full diagnostics are in `critical_case_review.json`, `key_fragmentation_diagnostics.json`, and `key_collision_diagnostics.json` within this run directory. Lexical diagnostics are hints, not causal attribution or semantic adjudication. No LLM judge or second-pass identity model was used.",
        "",
        "## Interpretation Boundary",
        "",
        "This stage measures the identity layer only. Semantic quality is not a structural completion gate; Human Reflection decides whether to proceed to MEM-3B1, run MEM-3B0.1 canonicalization over the frozen output, or revise the identity contract. No materializer was implemented or invoked.",
        "",
        f"`{GATE_NAME}={'YES' if gate else 'NO'}`",
        "",
    ]
    return "\n".join(lines)


def _verify_all_sidecars(paths: list[Path]) -> dict[str, Any]:
    failures = [str(path.relative_to(ROOT)) for path in paths if not _verify_frozen(path)]
    return {"required_file_count": len(paths), "invalid_paths": failures, "all_valid": not failures}


def run() -> dict[str, Any]:
    rows, projected, preflight = _validate_saved_preflight()
    contract_sha = _sha_file(CONTRACT_PATH)
    source_code_sha = _source_code_identity()
    if contract_sha != preflight["identity_contract_sha256"]:
        raise RuntimeError("frozen_contract_sha_mismatch")
    identities, attempts, call_summary = _run_identity_calls(
        projected, contract_sha, source_code_sha
    )

    # Core artifacts are frozen before diagnostic case IDs/history are inspected.
    core_groups = identity.candidate_groups(identities, projected)
    core_sha = {
        "revision_identity_records.jsonl": _freeze_jsonl(IDENTITIES_PATH, identities),
        "identity_call_ledger.jsonl": _freeze_jsonl(CALL_LEDGER_PATH, attempts),
        "candidate_revision_groups.json": _freeze_json(GROUPS_PATH, core_groups),
    }
    groups = _read_json(GROUPS_PATH)
    groups_for_diagnostics = _enrich_candidate_groups(groups, identities, rows)
    stats = identity.identity_statistics(identities, rows, groups_for_diagnostics)
    stats.pop("candidate_groups", None)
    fragmentation = identity.fragmentation_diagnostics(
        identities, rows, threshold=FRAGMENTATION_THRESHOLD
    )
    collisions = identity.collision_diagnostics(
        groups_for_diagnostics, threshold=COLLISION_JACCARD_THRESHOLD
    )
    _freeze_json(STATS_PATH, stats)
    _freeze_json(FRAGMENTATION_PATH, fragmentation)
    _freeze_json(COLLISION_PATH, collisions)
    core_paths = [IDENTITIES_PATH, CALL_LEDGER_PATH, GROUPS_PATH]
    core_sidecars = _verify_all_sidecars(core_paths)
    if not core_sidecars["all_valid"]:
        raise RuntimeError("identity_core_artifact_freeze_failed")

    core_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "CORE_FROZEN_BEFORE_DIAGNOSTIC_INSPECTION",
        "completion_gate_marker": None,
        "input_inventory_sha256": preflight["frozen_input"]["inventory_sha256"],
        "input_count": len(projected),
        "identity_contract_sha256": contract_sha,
        "identity_preflight_sha256": _sha_file(PREFLIGHT_PATH),
        "core_artifact_sha256": core_sha,
        "core_artifact_sidecars_valid": True,
        "identity_records_frozen_before_case_inspection": True,
        "candidate_groups_frozen_before_case_inspection": True,
        "call_ledger_frozen_before_case_inspection": True,
        "model_roles": {
            "reader_answer_model": None,
            "memory_system": "frozen FlatProp input; not modified",
            "identity_proposer": "frozen local Qwen3-8B Q4_K_M",
            "embedding_model": None,
            "judge_model": None,
        },
        "call_summary": call_summary,
        "diagnostic_artifacts_not_yet_inspected": True,
    }
    _freeze_json(CORE_MANIFEST_PATH, core_manifest)
    if not _verify_frozen(CORE_MANIFEST_PATH):
        raise RuntimeError("identity_run_manifest_freeze_failed")

    # Only now join frozen identities to case history and source timestamps.
    case_review = _case_review(identities, rows, groups, fragmentation, collisions)
    _freeze_json(CASE_REVIEW_PATH, case_review)
    report_gate_inputs = {
        "flatprop_input_sha_valid": preflight["frozen_input"]["inventory_sidecar_valid"],
        "identity_records_exactly_8112": len(identities) == 8112,
        "all_local_calls_complete": len(identities) == len(projected),
        "zero_retries": call_summary["retry_count"] == 0,
        "zero_hosted_calls": call_summary["hosted_calls"] == 0,
        "question_metadata_never_supplied": not any(preflight["label_leakage"].values()),
        "candidate_groups_frozen": _verify_frozen(GROUPS_PATH),
        "memory_store_unchanged": _sha_file(MATERIALIZATION_PATH)
        == preflight["frozen_input"]["materialization_ledger_sha256"]
        and _sha_file(INVENTORY_PATH) == preflight["frozen_input"]["inventory_sha256"],
        "zero_embedding_retrieval_reader_judge_calls": all(
            call_summary[field] == 0
            for field in ("embedding_calls", "retrieval_calls", "reader_calls", "judge_calls")
        ),
        "102_dev_test_medmemorybench_false": preflight["102_dev_access"] is False
        and preflight["test_access"] is False
        and preflight["medmemorybench_runs"] == 0,
        "core_frozen_before_diagnostics": _verify_frozen(CORE_MANIFEST_PATH)
        and core_manifest["identity_records_frozen_before_case_inspection"],
        "all_pre_report_sidecars_valid": _verify_all_sidecars(
            [
                CONTRACT_PATH,
                PREFLIGHT_PATH,
                IDENTITIES_PATH,
                CALL_LEDGER_PATH,
                GROUPS_PATH,
                STATS_PATH,
                FRAGMENTATION_PATH,
                COLLISION_PATH,
                CASE_REVIEW_PATH,
                CORE_MANIFEST_PATH,
            ]
        )["all_valid"],
    }
    gate = all(report_gate_inputs.values())
    report = _render_report(identities, groups, stats, call_summary, case_review, gate)
    report_bytes = report.encode("utf-8")
    _freeze(REPORT_PATH, report_bytes)
    _freeze(REPORT_DOC_PATH, report_bytes)
    final_artifacts = [
        CONTRACT_PATH,
        PREFLIGHT_PATH,
        IDENTITIES_PATH,
        CALL_LEDGER_PATH,
        GROUPS_PATH,
        STATS_PATH,
        FRAGMENTATION_PATH,
        COLLISION_PATH,
        CASE_REVIEW_PATH,
        CORE_MANIFEST_PATH,
        REPORT_PATH,
        REPORT_DOC_PATH,
        REPORT_PROTOCOL_PATH,
    ]
    final_sidecars = _verify_all_sidecars(final_artifacts)
    gate = gate and final_sidecars["all_valid"]
    final_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "base_commit_sha": preflight["base_commit_sha"],
        "topic_head_at_preflight": preflight["topic_head_sha"],
        "upstream_gate": "MEM3A3R_RECURSIVE_FLATPROP_FROZEN_10_DIAGNOSTIC=YES",
        "status": "COMPLETE" if gate else "FAILED",
        "completion_gate_marker": f"{GATE_NAME}={'YES' if gate else 'NO'}",
        "gate": report_gate_inputs
        | {"all_required_artifact_sha_sidecars_valid": final_sidecars["all_valid"]},
        "identity_statistics": stats,
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
            "embedding_calls": 0,
            "retrieval_calls": 0,
            "reader_calls": 0,
            "judge_calls": 0,
        },
        "artifact_sha256": {
            str(path.relative_to(ROOT).as_posix()): _sha_file(path) for path in final_artifacts
        },
    }
    _freeze_json(FINAL_MANIFEST_PATH, final_manifest)
    final_artifacts.append(FINAL_MANIFEST_PATH)
    verified = _verify_all_sidecars(final_artifacts)
    if not verified["all_valid"]:
        raise RuntimeError(f"final_artifact_sidecar_failure:{verified['invalid_paths']}")
    return final_manifest


def _render_protocol(
    contract: dict[str, Any],
    preflight: dict[str, Any],
    stats: dict[str, Any],
    calls: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "# MEM-3B0 - Revision Identity Protocol",
            "",
            f"Base: `{preflight['base_commit_sha']}`.",
            f"Contract: `revision-identity-v1`, SHA-256 `{preflight['identity_contract_sha256']}`.",
            f"Frozen FlatProp input: `{preflight['frozen_input']['inventory_path']}`, SHA-256 `{preflight['frozen_input']['inventory_sha256']}`, {preflight['frozen_input']['inventory_rows']} propositions.",
            "",
            "## Locked Method",
            "",
            f"- Initial batch size: {contract['batch']['initial_size']}; sorted by scope then memory ID.",
            f"- Model: {contract['model']['model']} ({contract['model']['model_sha256']}); local loopback endpoint only.",
            f"- Generation: temperature {contract['model']['temperature']}, seed {contract['model']['seed']}, thinking disabled, {contract['model']['max_tokens']} max tokens.",
            "- Request fields are only memory_id, proposition_text, source_authority, and scope_id. No timestamp or benchmark question/gold metadata is sent.",
            "- Only finish_reason=length triggers deterministic recursive bisection; incomplete parent output is discarded. No retries or cap increase.",
            "- Candidate groups include SINGLETON_STATE only and use exact (scope_id, subject_key, attribute_key). No supersession or MemoryStore writes.",
            "- Fragmentation and collision detectors are deterministic lexical heuristics, flagged but never auto-merged or auto-split.",
            "",
            "## Run Totals",
            "",
            f"- Kind counts: `{json.dumps(stats['counts_by_revision_kind'], sort_keys=True)}`.",
            f"- Candidate singleton groups: {stats['candidate_singleton_groups']}; multi-value: {stats['singleton_groups_with_multiple_values']}; multi-session: {stats['singleton_groups_spanning_multiple_sessions']}.",
            f"- Local proposer requests: {calls['total_provider_requests_unique']}; overflow parents: {calls['overflow_parent_count']}; retries: {calls['retry_count']}.",
            "- Embedding/retrieval/reader/judge/hosted calls: 0.",
            "",
            "This stage stops for Human Reflection. No MEM-3B1 materializer or MEM-3B0.1 canonicalization is included.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        rows, _projected, preflight = _preflight(write=True)
        print(
            json.dumps(
                {
                    "preflight": "PASS",
                    "stage_base": preflight["base_commit_sha"],
                    "topic_head": preflight["topic_head_sha"],
                    "input_rows": len(rows),
                    "projected_fields": list(identity.INPUT_FIELDS),
                    "runtime_model_sha256": preflight["identity_model_runtime"]["model_sha256"],
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
                "identity_statistics": result["identity_statistics"],
                "artifact_sha256": result["artifact_sha256"],
            },
            indent=2,
        )
    )
    return 0 if result["completion_gate_marker"].endswith("=YES") else 1


if __name__ == "__main__":
    raise SystemExit(main())
