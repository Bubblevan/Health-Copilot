"""Run the frozen, local-only MEM-3B0P pairwise revision admission study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import revision_pairwise_admission as admission

RUN_ID = "mem3b0p-pairwise-admission-20260930"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
FLATPROP_PATH = (
    ROOT
    / "runs"
    / "memory"
    / "mem3"
    / "mem3a3r-recursive-flatprop-frozen-10-20260929"
    / "flatprop_inventory.jsonl"
)
IDENTITY_PATH = (
    ROOT
    / "runs"
    / "memory"
    / "mem3"
    / "mem3b0r-revision-identity-keyed-map-20260930"
    / "revision_identity_records.jsonl"
)
SCORECARD_JSON_PATH = (
    ROOT / "docs" / "research" / "memory" / "memory_module_scorecard_contract.json"
)
SCORECARD_MD_PATH = (
    ROOT / "docs" / "research" / "memory" / "memory_module_scorecard_contract.md"
)
PROTOCOL_PATH = (
    ROOT / "docs" / "research" / "memory" / "mem_3b0p_pairwise_admission_protocol.md"
)
MODEL_PATH = Path(r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
EXPECTED_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
EXPECTED_FLATPROP_SHA256 = "250a24a10b532cb01bd841b965511df019fc0e597ec1e85ee12d7830c166c5fe"
EXPECTED_IDENTITY_SHA256 = "49af354bb535909a572df580cf02483ab56022dcf31873266154caf0086b4f75"
EXPECTED_SERVER_SHA256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
EXPECTED_SERVER_BUILD = "10068 (571d0d540)"
BASE_URL = "http://127.0.0.1:8081"
API_BASE = f"{BASE_URL}/v1"
MAX_CONTEXT = 131072
MAX_COMPLETION = 32
TEMPERATURE = 0
SEED = 42
SCORECARD_CONTRACT_ID = "health-copilot-memory-module-scorecard-v1"
FALSE_SAFE_LABELS = (
    "retro_game_night_interest",
    "value_reported_on_5_05_2021",
    "interest_in_documentary_series",
    "q_and_a_session_preparation",
    "current_planner_search",
    "aunt_and_uncle_relationship_support",
    "work_experience",
)


class IntegrityFailure(RuntimeError):
    """Frozen inputs, runtime identity, or artifact invariants failed."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(admission.canonical_json_bytes(row) + b"\n" for row in rows)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _freeze(path: Path, payload: bytes) -> str:
    _atomic_write(path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write(path.with_suffix(".sha256"), f"{digest}  {path.name}\n".encode())
    return digest


def _verify_sidecar(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise IntegrityFailure(f"missing_artifact:{path}")
    digest = _sha_file(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise IntegrityFailure(f"missing_artifact_sidecar:{sidecar}")
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != path.name or parts[0] != digest:
        raise IntegrityFailure(f"artifact_sidecar_mismatch:{path}")
    if expected is not None and digest != expected:
        raise IntegrityFailure(f"artifact_sha_mismatch:{path}")
    return digest


def _load_projected_jsonl(path: Path, allowed_fields: frozenset[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = admission.jsonl_projection(line, allowed_fields)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise IntegrityFailure(f"invalid_projected_jsonl:{path}:{line_number}") from exc
            if not set(row).issubset(allowed_fields):
                raise IntegrityFailure(f"projection_allowlist_violation:{path}:{line_number}")
            rows.append(row)
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise IntegrityFailure(f"non_object_jsonl:{path}:{line_number}")
            rows.append(row)
    return rows


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _source_code_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "admission_module": Path(admission.__file__),
        "protocol": PROTOCOL_PATH,
        "scorecard_json": SCORECARD_JSON_PATH,
        "scorecard_markdown": SCORECARD_MD_PATH,
    }
    return {name: _sha_file(path) for name, path in paths.items()}


def _pairwise_contract() -> dict[str, Any]:
    value = admission.pairwise_contract()
    value["runtime"].update(
        {
            "endpoint": API_BASE,
            "loopback_only": True,
            "llama_cpp_build": EXPECTED_SERVER_BUILD,
            "llama_server_sha256": EXPECTED_SERVER_SHA256,
            "server_arguments": {
                "host": "127.0.0.1",
                "port": 8081,
                "ctx_size": MAX_CONTEXT,
                "rope_scaling": "yarn",
                "rope_scale": 4,
                "yarn_orig_ctx": 32768,
                "cache_type_k": "q4_0",
                "cache_type_v": "q4_0",
                "n_gpu_layers": 99,
                "flash_attn": "on",
                "parallel": 1,
            },
        }
    )
    value["generation"].update(
        {"temperature": TEMPERATURE, "seed": SEED, "max_tokens": MAX_COMPLETION}
    )
    return value


def _code_commit_guard() -> dict[str, str]:
    scorecard_commit = _git("log", "-1", "--format=%H", "--", str(SCORECARD_MD_PATH.relative_to(ROOT)))
    tracked_status = _git("status", "--porcelain", "--", str(SCORECARD_MD_PATH.relative_to(ROOT)), str(SCORECARD_JSON_PATH.relative_to(ROOT)))
    if tracked_status:
        raise IntegrityFailure("scorecard_contract_has_uncommitted_changes")
    if _git("merge-base", "--is-ancestor", scorecard_commit, "HEAD") != "":
        raise IntegrityFailure("scorecard_lock_commit_not_ancestor_of_head")
    return {"scorecard_lock_commit_sha": scorecard_commit, "scorecard_commit_subject": _git("show", "-s", "--format=%s", scorecard_commit)}


def _initial_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, str]:
    _verify_sidecar(FLATPROP_PATH, EXPECTED_FLATPROP_SHA256)
    _verify_sidecar(IDENTITY_PATH, EXPECTED_IDENTITY_SHA256)
    scorecard_bytes = SCORECARD_JSON_PATH.read_bytes()
    scorecard = json.loads(scorecard_bytes)
    if scorecard.get("contract_id") != SCORECARD_CONTRACT_ID:
        raise IntegrityFailure("memory_scorecard_contract_id_mismatch")
    scorecard_md_sha = _sha_file(SCORECARD_MD_PATH)

    source_rows = _load_projected_jsonl(FLATPROP_PATH, admission.PROPOSAL_SOURCE_FIELDS)
    identity_rows = _load_projected_jsonl(IDENTITY_PATH, admission.PROPOSAL_IDENTITY_FIELDS)
    if len(source_rows) != 8112 or len(identity_rows) != 8112:
        raise IntegrityFailure("frozen_input_row_count_mismatch")

    universe, decisions, stats = admission.build_candidate_universe(identity_rows, source_rows)
    pairwise_contract = _pairwise_contract()
    grounding_contract = admission.grounding_contract()
    admission_contract = admission.admission_contract()
    pairwise_sha = admission.sha256_bytes(admission.canonical_json_bytes(pairwise_contract))
    admission_sha = admission.sha256_bytes(admission.canonical_json_bytes(admission_contract))
    pairs = admission.pairwise_records(universe, pairwise_sha)
    stats.update(
        {
            "pairwise_contract_canonical_sha256": pairwise_sha,
            "admission_contract_canonical_sha256": admission_sha,
            "unordered_pair_count": len(pairs),
            "timestamp_fields_loaded": 0,
            "timestamp_fields_used": 0,
            "full_identity_calls": 0,
            "memory_store_mutations": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "SUPERSEDED": 0},
        }
    )
    return (
        {"scorecard": scorecard, "scorecard_bytes": scorecard_bytes, "scorecard_md_sha256": scorecard_md_sha},
        {"universe": universe, "decisions": decisions, "pairs": pairs, "statistics": stats},
        pairwise_contract,
        {"grounding": grounding_contract, "admission": admission_contract},
        pairwise_sha,
        admission_sha,
    )


def prepare() -> dict[str, Any]:
    if RUN_DIR.exists():
        raise IntegrityFailure(f"run_directory_already_exists:{RUN_DIR}")
    code_guard = _code_commit_guard()
    scorecard_bundle, data_bundle, pairwise_contract, policy_contracts, pairwise_sha, admission_sha = _initial_artifacts()
    RUN_DIR.mkdir(parents=True)

    artifact_hashes: dict[str, str] = {}

    def freeze_json(name: str, value: Any) -> None:
        artifact_hashes[name] = _freeze(RUN_DIR / name, _json_bytes(value))

    def freeze_jsonl(name: str, rows: list[dict[str, Any]]) -> None:
        artifact_hashes[name] = _freeze(RUN_DIR / name, _jsonl_bytes(rows))

    artifact_hashes["memory_module_scorecard_contract.json"] = _freeze(
        RUN_DIR / "memory_module_scorecard_contract.json", scorecard_bundle["scorecard_bytes"]
    )
    artifact_hashes["proposal_grounding_contract.json"] = _freeze(
        RUN_DIR / "proposal_grounding_contract.json", _json_bytes(policy_contracts["grounding"])
    )
    artifact_hashes["pairwise_verifier_contract.json"] = _freeze(
        RUN_DIR / "pairwise_verifier_contract.json", _json_bytes(pairwise_contract)
    )
    artifact_hashes["revision_admission_contract.json"] = _freeze(
        RUN_DIR / "revision_admission_contract.json", _json_bytes(policy_contracts["admission"])
    )
    freeze_jsonl("proposal_grounding_decisions.jsonl", data_bundle["decisions"])
    freeze_json("candidate_groups.json", data_bundle["universe"])
    freeze_jsonl("pair_manifest.jsonl", data_bundle["pairs"])
    artifact_hashes["pairwise_call_ledger.jsonl"] = _freeze(
        RUN_DIR / "pairwise_call_ledger.jsonl", b""
    )
    freeze_json("safety_statistics.json", data_bundle["statistics"])

    manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "CANDIDATES_FROZEN_AWAITING_LOCAL_PAIRWISE_VERIFICATION",
        "completion_gate_marker": "MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=NO",
        "readiness_gate_marker": "MEM3B0P_MEM3B1_READY=NO",
        "base_commit_sha": _git("rev-parse", "HEAD"),
        "topic_branch": _git("branch", "--show-current"),
        **code_guard,
        "scorecard": {
            "contract_id": SCORECARD_CONTRACT_ID,
            "json_sha256": artifact_hashes["memory_module_scorecard_contract.json"],
            "markdown_sha256": scorecard_bundle["scorecard_md_sha256"],
            "committed_before_any_model_call": True,
        },
        "upstream": {
            "flatprop_path": str(FLATPROP_PATH.relative_to(ROOT)).replace("\\", "/"),
            "flatprop_sha256": EXPECTED_FLATPROP_SHA256,
            "flatprop_count": len(data_bundle["decisions"]),
            "identity_path": str(IDENTITY_PATH.relative_to(ROOT)).replace("\\", "/"),
            "identity_sha256": EXPECTED_IDENTITY_SHA256,
            "identity_count": len(data_bundle["decisions"]),
            "identity_role": "SEMANTIC_CANDIDATE_HINTS_NOT_AUTHORITATIVE_STATE_SCHEMA",
        },
        "contract_sha256": {
            "pairwise_contract_canonical_sha256": pairwise_sha,
            "admission_contract_canonical_sha256": admission_sha,
            "grounding_contract_file_sha256": artifact_hashes["proposal_grounding_contract.json"],
        },
        "source_code_sha256": _source_code_hashes(),
        "candidate_statistics": data_bundle["statistics"],
        "frozen_artifacts": artifact_hashes,
        "execution_policy": {
            "pairwise_calls_allowed": len(data_bundle["pairs"]),
            "full_identity_calls": 0,
            "same_request_retries": 0,
            "hosted_calls": 0,
            "memory_store_mutations": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "SUPERSEDED": 0},
            "embedding_calls": 0,
            "retrieval_calls": 0,
            "reader_answer_calls": 0,
            "judge_calls": 0,
            "benchmark_runs": 0,
            "timestamp_fields_loaded": 0,
            "timestamp_fields_used": 0,
        },
    }
    _freeze(RUN_DIR / "run_manifest.json", _json_bytes(manifest))
    _refresh_manifest_sidecar()
    return {"prepared": True, **data_bundle["statistics"], "run_dir": str(RUN_DIR)}


def _refresh_manifest_sidecar() -> str:
    path = RUN_DIR / "run_manifest.json"
    digest = _sha_file(path)
    _atomic_write(path.with_suffix(".sha256"), f"{digest}  {path.name}\n".encode())
    return digest


def _read_manifest() -> dict[str, Any]:
    path = RUN_DIR / "run_manifest.json"
    _verify_sidecar(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != RUN_ID:
        raise IntegrityFailure("run_manifest_stage_mismatch")
    return value


def _read_frozen(path_name: str) -> Any:
    path = RUN_DIR / path_name
    _verify_sidecar(path)
    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _argument(command_line: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}\s+(?:\"([^\"]+)\"|(\S+))", command_line)
    return (match.group(1) or match.group(2)) if match else None


def _listener_process() -> dict[str, Any]:
    script = (
        "$rows=@(netstat -ano -p tcp | Select-String '^\\s*TCP\\s+127\\.0\\.0\\.1:8081\\s+\\S+\\s+LISTENING\\s+(\\d+)\\s*$'); "
        "$p=@(); foreach($row in $rows) { if($row.Line -match '(\\d+)\\s*$') { "
        "$pidValue=[int]$Matches[1]; $p+=@(Get-CimInstance Win32_Process -Filter \"ProcessId=$pidValue\" | "
        "Select-Object ProcessId,ExecutablePath,CommandLine) } }; "
        "ConvertTo-Json -InputObject $p -Compress -Depth 4"
    )
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", script], text=True
    ).strip()
    if not output:
        raise IntegrityFailure("frozen_loopback_reader_not_running")
    parsed = json.loads(output)
    rows = parsed if isinstance(parsed, list) else [parsed]
    if len(rows) != 1:
        raise IntegrityFailure("expected_exactly_one_loopback_reader")
    return rows[0]


def _verify_runtime(client: httpx.Client) -> dict[str, Any]:
    model_sha = _sha_file(MODEL_PATH)
    if model_sha != EXPECTED_MODEL_SHA256:
        raise IntegrityFailure("frozen_qwen_model_sha_mismatch")
    process = _listener_process()
    command_line = process.get("CommandLine") or ""
    executable = Path(process.get("ExecutablePath") or "")
    if not executable.is_file():
        raise IntegrityFailure("reader_executable_missing")
    binary_sha = _sha_file(executable)
    if binary_sha != EXPECTED_SERVER_SHA256:
        raise IntegrityFailure("frozen_llama_server_sha_mismatch")
    version_result = subprocess.run(
        [str(executable), "--version"], capture_output=True, text=True, check=False
    )
    version = "\n".join(part.strip() for part in (version_result.stdout, version_result.stderr) if part.strip())
    if version_result.returncode != 0 or "10068" not in version or "571d0d540" not in version:
        raise IntegrityFailure("frozen_llama_server_build_mismatch")

    required_args = {
        "--host": "127.0.0.1",
        "--port": "8081",
        "--ctx-size": "131072",
        "--rope-scaling": "yarn",
        "--rope-scale": "4",
        "--yarn-orig-ctx": "32768",
        "--cache-type-k": "q4_0",
        "--cache-type-v": "q4_0",
        "--n-gpu-layers": "99",
        "--flash-attn": "on",
        "--parallel": "1",
    }
    for name, expected in required_args.items():
        if _argument(command_line, name) != expected:
            raise IntegrityFailure(f"frozen_reader_argument_mismatch:{name}")
    model_arg = _argument(command_line, "-m")
    if not model_arg or model_arg.replace("/", "\\").casefold() != str(MODEL_PATH).replace("/", "\\").casefold():
        raise IntegrityFailure("reader_not_serving_frozen_model")
    if "--no-kv-offload" in command_line:
        raise IntegrityFailure("gpu_kv_offload_disabled")

    models_response = client.get(f"{API_BASE}/models")
    models_response.raise_for_status()
    model_rows = models_response.json().get("data", [])
    model_ids = [row.get("id") for row in model_rows if isinstance(row, dict) and isinstance(row.get("id"), str)]
    if len(model_ids) != 1:
        raise IntegrityFailure("expected_one_local_model_endpoint")
    model_id = model_ids[0]
    if "qwen3-8b-q4_k_m.gguf" not in model_id.replace("\\", "/").casefold() and model_id != "health-memory-qwen3-8b":
        raise IntegrityFailure("endpoint_model_identifier_not_frozen_qwen")

    slots_response = client.get(f"{BASE_URL}/slots")
    slots_response.raise_for_status()
    slots = slots_response.json()
    contexts = [slot.get("n_ctx") for slot in slots if isinstance(slot, dict) and isinstance(slot.get("n_ctx"), int)]
    if not contexts or min(contexts) != MAX_CONTEXT:
        raise IntegrityFailure("reader_context_length_mismatch")
    if any(slot.get("is_processing") for slot in slots if isinstance(slot, dict)):
        raise IntegrityFailure("reader_slot_busy_before_run")
    return {
        "provider": "local_qwen_loopback",
        "endpoint": API_BASE,
        "loopback_only": True,
        "model_id": model_id,
        "model_artifact": str(MODEL_PATH),
        "model_sha256": model_sha,
        "server_pid": int(process["ProcessId"]),
        "server_executable": str(executable),
        "server_binary_sha256": binary_sha,
        "server_version": version,
        "server_build": EXPECTED_SERVER_BUILD,
        "context_tokens": min(contexts),
        "generation": {"temperature": TEMPERATURE, "seed": SEED, "enable_thinking": False, "max_tokens": MAX_COMPLETION},
        "proxy_environment_used": False,
        "hosted_calls": 0,
    }


def _rendered_prompt_tokens(client: httpx.Client, messages: list[dict[str, str]]) -> int:
    rendered = client.post(
        f"{BASE_URL}/apply-template",
        json={"messages": messages, "chat_template_kwargs": {"enable_thinking": False}, "add_generation_prompt": True},
    )
    rendered.raise_for_status()
    prompt = rendered.json().get("prompt")
    if not isinstance(prompt, str):
        raise TypeError("template_endpoint_returned_malformed_prompt")
    response = client.post(
        f"{BASE_URL}/tokenize",
        json={"content": prompt, "add_special": False, "parse_special": True},
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise TypeError("tokenizer_endpoint_returned_malformed_tokens")
    return len(tokens)


def _ledger_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in result:
            raise IntegrityFailure("duplicate_or_missing_pair_ledger_id")
        result[pair_id] = row
    return result


def _write_ledger(ledger: dict[str, dict[str, Any]], order: list[str]) -> str:
    rows = [ledger[pair_id] for pair_id in order if pair_id in ledger]
    return _freeze(RUN_DIR / "pairwise_call_ledger.jsonl", _jsonl_bytes(rows))


def _record_global_failure(manifest: dict[str, Any], error: Exception) -> None:
    failure = {
        "schema_version": 1,
        "failure_class": "GLOBAL_INFRA_FAILURE",
        "reason": type(error).__name__,
        "detail": str(error),
        "hosted_calls": 0,
        "same_request_retries": 0,
    }
    failure_sha = _freeze(RUN_DIR / "execution_failure.json", _json_bytes(failure))
    manifest["status"] = "INFRA_FAILURE"
    manifest["execution_failure"] = failure
    manifest.setdefault("frozen_artifacts", {})["execution_failure.json"] = failure_sha
    _freeze(RUN_DIR / "run_manifest.json", _json_bytes(manifest))


def verify_pairs() -> dict[str, Any]:
    manifest = _read_manifest()
    if manifest.get("status") not in {
        "CANDIDATES_FROZEN_AWAITING_LOCAL_PAIRWISE_VERIFICATION",
        "PAIRWISE_VERIFICATION_IN_PROGRESS",
    }:
        raise IntegrityFailure(f"invalid_pairwise_run_status:{manifest.get('status')}")
    if _source_code_hashes() != manifest.get("source_code_sha256"):
        raise IntegrityFailure("source_code_changed_after_candidate_freeze")
    universe = _read_frozen("candidate_groups.json")
    pair_manifest = _read_frozen("pair_manifest.jsonl")
    contract = _read_frozen("pairwise_verifier_contract.json")
    pairwise_sha = admission.sha256_bytes(admission.canonical_json_bytes(contract))
    if pairwise_sha != manifest.get("contract_sha256", {}).get("pairwise_contract_canonical_sha256"):
        raise IntegrityFailure("pairwise_contract_hash_mismatch")
    expected_pairs = admission.pairwise_records(universe, pairwise_sha)
    if pair_manifest != expected_pairs:
        raise IntegrityFailure("pair_manifest_not_reproducible_from_frozen_candidates")

    ledger_path = RUN_DIR / "pairwise_call_ledger.jsonl"
    _verify_sidecar(ledger_path)
    ledger = _ledger_index(_load_jsonl(ledger_path))
    pair_order = [pair["pair_id"] for pair in pair_manifest]
    pair_by_id = {pair["pair_id"]: pair for pair in pair_manifest}
    if set(ledger) - set(pair_by_id):
        raise IntegrityFailure("ledger_contains_unknown_pair_id")

    timeout = httpx.Timeout(connect=10, read=120, write=30, pool=30)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        try:
            runtime = _verify_runtime(client)
        except Exception as exc:
            _record_global_failure(manifest, exc)
            raise
        for pair_index, pair in enumerate(pair_manifest, 1):
            pair_id = pair["pair_id"]
            existing = ledger.get(pair_id)
            if existing and existing.get("state") == "TERMINAL":
                continue
            if existing and existing.get("state") == "PENDING":
                ledger[pair_id] = {
                    **existing,
                    "state": "TERMINAL",
                    "verdict": "UNKNOWN",
                    "failure_code": "INTERRUPTED_PENDING_NEVER_RESEND",
                    "provider_call_uncertain": True,
                    "hosted_calls": 0,
                    "retry_count": 0,
                }
                _write_ledger(ledger, pair_order)
                continue

            projection = admission.pairwise_request_projection(pair)
            messages = [
                {"role": "system", "content": contract["system_prompt"]},
                {"role": "user", "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
            ]
            request = {
                "model": runtime["model_id"],
                "messages": messages,
                "temperature": TEMPERATURE,
                "seed": SEED,
                "max_tokens": MAX_COMPLETION,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "mem3b0p_pairwise_verdict", "strict": True, "schema": contract["response_schema"]},
                },
            }
            request_sha = admission.sha256_bytes(admission.canonical_json_bytes(request))
            try:
                prompt_tokens = _rendered_prompt_tokens(client, messages)
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                ledger[pair_id] = {
                    "pair_id": pair_id,
                    "group_id": pair["group_id"],
                    "request_sha256": request_sha,
                    "state": "TERMINAL",
                    "verdict": "UNKNOWN",
                    "failure_code": "LOCAL_TEMPLATE_OR_TOKENIZER_ANOMALY",
                    "provider_calls": 0,
                    "provider_call_uncertain": False,
                    "hosted_calls": 0,
                    "retry_count": 0,
                    "prompt_tokens": None,
                }
                _write_ledger(ledger, pair_order)
                continue
            if prompt_tokens + MAX_COMPLETION > MAX_CONTEXT:
                ledger[pair_id] = {
                    "pair_id": pair_id,
                    "group_id": pair["group_id"],
                    "request_sha256": request_sha,
                    "state": "TERMINAL",
                    "verdict": "UNKNOWN",
                    "failure_code": "PAIR_PROMPT_EXCEEDS_CONTEXT",
                    "provider_calls": 0,
                    "provider_call_uncertain": False,
                    "hosted_calls": 0,
                    "retry_count": 0,
                    "prompt_tokens": prompt_tokens,
                }
                _write_ledger(ledger, pair_order)
                continue

            ledger[pair_id] = {
                "pair_id": pair_id,
                "group_id": pair["group_id"],
                "memory_id_a": pair["memory_id_a"],
                "memory_id_b": pair["memory_id_b"],
                "request_sha256": request_sha,
                "request_projection_sha256": admission.sha256_bytes(admission.canonical_json_bytes(projection)),
                "prompt_tokens": prompt_tokens,
                "state": "PENDING",
                "verdict": None,
                "failure_code": None,
                "provider_calls": 1,
                "provider_call_uncertain": True,
                "hosted_calls": 0,
                "retry_count": 0,
                "latency_ms": None,
            }
            _write_ledger(ledger, pair_order)
            started = time.perf_counter()
            try:
                response = client.post(f"{API_BASE}/chat/completions", json=request)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                content = (choice.get("message") or {}).get("content")
                finish_reason = choice.get("finish_reason")
                verdict, failure_code = admission.parse_pairwise_output(content, finish_reason)
            except (httpx.HTTPError, ValueError, TypeError, KeyError, IndexError):
                content = None
                finish_reason = None
                verdict, failure_code = "UNKNOWN", "PAIRWISE_TRANSPORT_OR_CONTENT_ANOMALY"
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            ledger[pair_id] = {
                **ledger[pair_id],
                "state": "TERMINAL",
                "verdict": verdict,
                "failure_code": failure_code,
                "finish_reason": finish_reason,
                "raw_completion": content,
                "provider_call_uncertain": False,
                "latency_ms": latency_ms,
            }
            _write_ledger(ledger, pair_order)
            if pair_index % 20 == 0 or pair_index == len(pair_manifest):
                print(f"pairwise_progress={pair_index}/{len(pair_manifest)}", flush=True)

    if set(ledger) != set(pair_order) or any(row.get("state") != "TERMINAL" for row in ledger.values()):
        manifest["status"] = "PAIRWISE_VERIFICATION_IN_PROGRESS"
        manifest["runtime"] = runtime
        manifest["pairwise_completed_count"] = sum(row.get("state") == "TERMINAL" for row in ledger.values())
        manifest["frozen_artifacts"]["pairwise_call_ledger.jsonl"] = _verify_sidecar(ledger_path)
        _freeze(RUN_DIR / "run_manifest.json", _json_bytes(manifest))
        _refresh_manifest_sidecar()
        raise IntegrityFailure("pairwise_terminal_coverage_incomplete")

    manifest["status"] = "PAIRWISE_CALLS_TERMINAL"
    manifest["runtime"] = runtime
    manifest["pairwise_provider_calls"] = sum(int(row.get("provider_calls", 0)) for row in ledger.values())
    manifest["pairwise_unknown_count"] = sum(row.get("verdict") == "UNKNOWN" for row in ledger.values())
    manifest["same_request_retries"] = 0
    manifest["hosted_calls"] = 0
    manifest["frozen_artifacts"]["pairwise_call_ledger.jsonl"] = _verify_sidecar(ledger_path)
    _freeze(RUN_DIR / "run_manifest.json", _json_bytes(manifest))
    _refresh_manifest_sidecar()
    return _freeze_semantic_decisions(manifest, universe, pair_manifest, ledger)


def _freeze_semantic_decisions(
    manifest: dict[str, Any],
    universe: dict[str, Any],
    pair_manifest: list[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairwise_sha = manifest["contract_sha256"]["pairwise_contract_canonical_sha256"]
    admission_sha = manifest["contract_sha256"]["admission_contract_canonical_sha256"]
    pair_rows = []
    for pair in pair_manifest:
        row = ledger[pair["pair_id"]]
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "group_id": pair["group_id"],
                "memory_id_a": pair["memory_id_a"],
                "memory_id_b": pair["memory_id_b"],
                "verdict": row["verdict"],
                "failure_code": row.get("failure_code"),
                "pairwise_contract_sha256": pairwise_sha,
                "request_projection_sha256": row.get("request_projection_sha256"),
            }
        )
    overlay, slot_manifest = admission.build_admission_overlay(universe, pair_rows, admission_sha)
    for payload in (pair_rows, overlay, slot_manifest):
        if _contains_forbidden(payload):
            raise IntegrityFailure("semantic_artifact_contains_timestamp_or_benchmark_field")

    artifact_hashes = manifest.setdefault("frozen_artifacts", {})
    artifact_hashes["pairwise_verdicts.jsonl"] = _freeze(
        RUN_DIR / "pairwise_verdicts.jsonl", _jsonl_bytes(pair_rows)
    )
    artifact_hashes["revision_admission_overlay.jsonl"] = _freeze(
        RUN_DIR / "revision_admission_overlay.jsonl", _jsonl_bytes(overlay)
    )
    artifact_hashes["revision_slot_manifest.json"] = _freeze(
        RUN_DIR / "revision_slot_manifest.json", _json_bytes(slot_manifest)
    )
    stats = json.loads((RUN_DIR / "safety_statistics.json").read_text(encoding="utf-8"))
    verdict_counts = Counter(row["verdict"] for row in pair_rows)
    group_counts = Counter(row["admission_status"] for row in overlay)
    stats.update(
        {
            "pairwise_verdict_counts": dict(sorted(verdict_counts.items())),
            "repeated_candidate_groups_admitted": group_counts.get("REVISION_ADMITTED_PAIRWISE_ALL_SAME", 0),
            "repeated_candidate_groups_blocked": group_counts.get("REVISION_BLOCKED_PAIRWISE_INCONSISTENCY", 0),
            "singleton_groups_no_revision_history": group_counts.get("NO_REVISION_HISTORY", 0),
            "opaque_revision_slots_created": len(slot_manifest),
            "pairwise_provider_calls": sum(int(row.get("provider_calls", 0)) for row in ledger.values()),
            "pairwise_unknown_count": verdict_counts.get("UNKNOWN", 0),
            "same_request_retries": 0,
            "hosted_calls": 0,
            "timestamp_fields_loaded": 0,
            "timestamp_fields_used": 0,
            "memory_store_mutations": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "SUPERSEDED": 0},
        }
    )
    artifact_hashes["safety_statistics.json"] = _freeze(
        RUN_DIR / "safety_statistics.json", _json_bytes(stats)
    )
    semantic_marker = {
        "schema_version": 1,
        "status": "SEMANTIC_DECISIONS_FROZEN",
        "pairwise_contract_canonical_sha256": pairwise_sha,
        "admission_contract_canonical_sha256": admission_sha,
        "pairwise_verdicts_sha256": artifact_hashes["pairwise_verdicts.jsonl"],
        "revision_admission_overlay_sha256": artifact_hashes["revision_admission_overlay.jsonl"],
        "revision_slot_manifest_sha256": artifact_hashes["revision_slot_manifest.json"],
        "pairwise_call_ledger_sha256": _verify_sidecar(RUN_DIR / "pairwise_call_ledger.jsonl"),
        "timestamp_fields_loaded_before_freeze": 0,
        "timestamp_fields_used_before_freeze": 0,
        "timestamp_fields_loaded_in_mem3b0p": 0,
        "benchmark_labels_loaded": False,
    }
    artifact_hashes["semantic_freeze.json"] = _freeze(
        RUN_DIR / "semantic_freeze.json", _json_bytes(semantic_marker)
    )
    manifest["status"] = "SEMANTIC_DECISIONS_FROZEN_AWAITING_HUMAN_REFLECTION"
    manifest["semantic_freeze"] = semantic_marker
    manifest["frozen_artifacts"] = artifact_hashes
    _freeze(RUN_DIR / "run_manifest.json", _json_bytes(manifest))
    _refresh_manifest_sidecar()
    return {
        "semantic_decisions_frozen": True,
        "pairwise_verdict_counts": dict(sorted(verdict_counts.items())),
        "group_admission_counts": dict(sorted(group_counts.items())),
        "opaque_revision_slots_created": len(slot_manifest),
        "run_dir": str(RUN_DIR),
    }


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        if set(value).intersection(admission.TEMPORAL_OR_BENCHMARK_FIELDS):
            return True
        return any(_contains_forbidden(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden(child) for child in value)
    return False


def _load_semantic_freeze() -> dict[str, Any]:
    path = RUN_DIR / "semantic_freeze.json"
    _verify_sidecar(path)
    marker = admission.require_semantic_freeze(path.read_bytes())
    manifest = _read_manifest()
    if marker.get("pairwise_verdicts_sha256") != _verify_sidecar(RUN_DIR / "pairwise_verdicts.jsonl"):
        raise IntegrityFailure("semantic_freeze_pair_verdict_hash_mismatch")
    if marker.get("revision_admission_overlay_sha256") != _verify_sidecar(RUN_DIR / "revision_admission_overlay.jsonl"):
        raise IntegrityFailure("semantic_freeze_admission_overlay_hash_mismatch")
    if marker.get("revision_slot_manifest_sha256") != _verify_sidecar(RUN_DIR / "revision_slot_manifest.json"):
        raise IntegrityFailure("semantic_freeze_slot_manifest_hash_mismatch")
    if manifest.get("semantic_freeze") != marker:
        raise IntegrityFailure("semantic_freeze_manifest_mismatch")
    return marker


def build_human_review_packet() -> dict[str, Any]:
    _load_semantic_freeze()
    universe = _read_frozen("candidate_groups.json")
    overlay = _read_frozen("revision_admission_overlay.jsonl")
    pair_rows = _read_frozen("pairwise_verdicts.jsonl")
    overlay_by_id = {row["group_id"]: row for row in overlay}
    pairs_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pairs_by_group[row["group_id"]].append(row)
    groups_by_id = {
        row["group_id"]: row
        for row in universe["repeated_candidate_groups"]
    }
    review_rows = []
    for slot in _read_frozen("revision_slot_manifest.json"):
        group_id = next(
            group_id
            for group_id, row in overlay_by_id.items()
            if row["revision_slot_id"] == slot["revision_slot_id"]
        )
        group = groups_by_id[group_id]
        review_rows.append(
            {
                "revision_slot_id": slot["revision_slot_id"],
                "group_id": group_id,
                "scope_id": slot["scope_id"],
                "diagnostic_subject_key": slot["diagnostic_subject_key"],
                "diagnostic_attribute_key": slot["diagnostic_attribute_key"],
                "members": group["propositions"],
                "pair_verdicts": [
                    {
                        "memory_id_a": row["memory_id_a"],
                        "memory_id_b": row["memory_id_b"],
                        "verdict": row["verdict"],
                    }
                    for row in sorted(pairs_by_group[group_id], key=lambda item: item["pair_id"])
                ],
                "human_review": {"decision": "PENDING", "notes": ""},
            }
        )
    packet = {
        "schema_version": 1,
        "review_scope": "every admitted group; no timestamps",
        "decision_enum": ["true_singleton_slot", "false_revision_merge", "uncertain"],
        "reviewed_groups": review_rows,
    }
    _freeze(RUN_DIR / "admitted_groups_human_review_packet.json", _json_bytes(packet))
    return {"review_packet_created": True, "admitted_group_count": len(review_rows)}


def _identity_source_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    _verify_sidecar(FLATPROP_PATH, EXPECTED_FLATPROP_SHA256)
    _verify_sidecar(IDENTITY_PATH, EXPECTED_IDENTITY_SHA256)
    # Still allowlist-projected: timestamps are never materialized, even post-freeze.
    source_rows = _load_projected_jsonl(FLATPROP_PATH, admission.PROPOSAL_SOURCE_FIELDS)
    identity_rows = _load_projected_jsonl(IDENTITY_PATH, admission.PROPOSAL_IDENTITY_FIELDS)
    return (
        {row["memory_id"]: row for row in source_rows},
        {row["memory_id"]: row for row in identity_rows},
    )


def _human_reviewed_rows(decisions_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packet = _read_frozen("admitted_groups_human_review_packet.json")
    decision_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions = decision_payload.get("reviews")
    if not isinstance(decisions, list):
        raise IntegrityFailure("human_review_decisions_missing_reviews")
    decision_by_slot: dict[str, dict[str, Any]] = {}
    for row in decisions:
        if not isinstance(row, dict) or row.get("decision") not in {
            "true_singleton_slot",
            "false_revision_merge",
            "uncertain",
        }:
            raise IntegrityFailure("invalid_human_review_decision")
        slot_id = row.get("revision_slot_id")
        if not isinstance(slot_id, str) or slot_id in decision_by_slot:
            raise IntegrityFailure("duplicate_or_missing_human_review_slot")
        decision_by_slot[slot_id] = row
    packet_rows = packet["reviewed_groups"]
    expected_slots = {row["revision_slot_id"] for row in packet_rows}
    if set(decision_by_slot) != expected_slots:
        raise IntegrityFailure("human_review_not_exhaustive")
    final_rows = []
    for row in packet_rows:
        decision = decision_by_slot[row["revision_slot_id"]]
        final_rows.append(
            {
                **row,
                "human_review": {
                    "decision": decision["decision"],
                    "notes": str(decision.get("notes", "")),
                },
            }
        )
    return packet, final_rows


def _critical_cases(
    decisions: list[dict[str, Any]],
    universe: dict[str, Any],
    overlay: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    identity_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decision_by_id = {row["memory_id"]: row for row in decisions}
    overlay_by_id = {row["group_id"]: row for row in overlay}
    false_safe_results = []
    for label in FALSE_SAFE_LABELS:
        matched_ids = []
        for memory_id, identity in identity_by_id.items():
            key_values = [identity.get("subject_key"), identity.get("attribute_key")]
            if any(isinstance(value, str) and label.casefold() in value.casefold() for value in key_values):
                matched_ids.append(memory_id)
        if not matched_ids:
            false_safe_results.append(
                {"historical_case": label, "status": "NOT_FOUND_IN_FROZEN_IDENTITY_HINTS", "member_memory_ids": [], "all_blocked": False}
            )
            continue
        statuses = []
        current_group_ids = set()
        for memory_id in matched_ids:
            decision = decision_by_id[memory_id]
            group_id = decision.get("candidate_group_id")
            if group_id:
                current_group_ids.add(group_id)
                current = overlay_by_id.get(group_id)
                statuses.append(current["admission_status"] if current else "NO_TERMINAL_GROUP_STATUS")
            elif decision.get("grounding_status") == "BLOCKED":
                statuses.append(decision.get("grounding_reason"))
            else:
                statuses.append(decision.get("eligibility_reason", "INELIGIBLE"))
        all_blocked = all(status != "REVISION_ADMITTED_PAIRWISE_ALL_SAME" for status in statuses)
        false_safe_results.append(
            {
                "historical_case": label,
                "member_memory_ids": sorted(matched_ids),
                "current_group_ids": sorted(current_group_ids),
                "current_outcomes": sorted(set(statuses)),
                "status": "BLOCKED_OR_NOT_ADMITTED" if all_blocked else "STILL_ADMITTED",
                "all_blocked": all_blocked,
            }
        )

    instagram_group = None
    for group in universe["repeated_candidate_groups"]:
        texts = [row["proposition_text"] for row in group["propositions"]]
        first = any("instagram" in text.casefold() and "500" in admission.normalized_tokens(text) for text in texts)
        second = any("instagram" in text.casefold() and "600" in admission.normalized_tokens(text) for text in texts)
        if first and second:
            group_outcome = overlay_by_id[group["group_id"]]
            matching_pairs = [
                row for row in pair_rows
                if row["group_id"] == group["group_id"]
                and {row["memory_id_a"], row["memory_id_b"]}.issubset(set(group["member_memory_ids"]))
            ]
            instagram_group = {
                "group_id": group["group_id"],
                "member_memory_ids": group["member_memory_ids"],
                "pairwise_verdicts": [row["verdict"] for row in matching_pairs],
                "admission_status": group_outcome["admission_status"],
                "revision_slot_id": group_outcome["revision_slot_id"],
            }
            break
    instagram_yes = bool(
        instagram_group
        and instagram_group["admission_status"] == "REVISION_ADMITTED_PAIRWISE_ALL_SAME"
        and "SAME_MUTABLE_SLOT" in instagram_group["pairwise_verdicts"]
        and instagram_group["revision_slot_id"]
    )

    gym_related = []
    for group in universe["repeated_candidate_groups"] + universe["singletons_without_revision_history"]:
        blob = " ".join(
            [group["diagnostic_subject_key"], group["diagnostic_attribute_key"]]
            + [row["proposition_text"] for row in group["propositions"]]
        ).casefold()
        if "gym" in blob or "workout" in blob or "work out" in blob:
            current = overlay_by_id.get(group["group_id"], {})
            gym_related.append(
                {
                    "group_id": group["group_id"],
                    "diagnostic_attribute_key": group["diagnostic_attribute_key"],
                    "member_memory_ids": group["member_memory_ids"],
                    "admission_status": current.get("admission_status", "NO_REVISION_HISTORY"),
                }
            )
    gym_state = "PARTIAL" if gym_related and len({row["diagnostic_attribute_key"] for row in gym_related}) > 1 else "NO"

    return {
        "historical_b0s_false_safe_examples": false_safe_results,
        "all_historical_b0s_false_safe_examples_blocked": bool(false_safe_results)
        and all(row["all_blocked"] for row in false_safe_results),
        "instagram_positive_control": instagram_group,
        "instagram_pairwise_revision_admission": "YES" if instagram_yes else "NO",
        "gym_revision_admission": gym_state,
        "gym_related_exact_groups": gym_related,
        "no_timestamp_values_in_review": True,
    }


def closeout(decisions_path: Path) -> dict[str, Any]:
    semantic_marker = _load_semantic_freeze()
    manifest = _read_manifest()
    if decisions_path.resolve().parent != RUN_DIR.resolve():
        raise IntegrityFailure("human_review_decisions_must_be_inside_run_directory")
    _verify_sidecar(decisions_path)
    universe = _read_frozen("candidate_groups.json")
    overlay = _read_frozen("revision_admission_overlay.jsonl")
    pair_rows = _read_frozen("pairwise_verdicts.jsonl")
    decisions = _read_frozen("proposal_grounding_decisions.jsonl")
    packet, human_review = _human_reviewed_rows(decisions_path)
    if len(human_review) != len(_read_frozen("revision_slot_manifest.json")):
        raise IntegrityFailure("exhaustive_human_review_group_count_mismatch")
    if any(row["human_review"]["decision"] == "PENDING" for row in human_review):
        raise IntegrityFailure("human_review_pending")

    source_by_id, identity_by_id = _identity_source_maps()
    cases = _critical_cases(decisions, universe, overlay, pair_rows, source_by_id, identity_by_id)
    human_clear = all(row["human_review"]["decision"] == "true_singleton_slot" for row in human_review)
    pairwise_admitted_complete = all(
        all(verdict == "SAME_MUTABLE_SLOT" for verdict in row["pairwise_verdicts"])
        for row in overlay
        if row["admission_status"] == "REVISION_ADMITTED_PAIRWISE_ALL_SAME"
    )
    readiness = all(
        [
            cases["instagram_pairwise_revision_admission"] == "YES",
            cases["all_historical_b0s_false_safe_examples_blocked"],
            human_clear,
            pairwise_admitted_complete,
            semantic_marker["timestamp_fields_loaded_before_freeze"] == 0,
        ]
    )
    stats = _read_frozen("safety_statistics.json")
    ledger = _read_frozen("pairwise_call_ledger.jsonl")
    pair_ids = {row["pair_id"] for row in _read_frozen("pair_manifest.jsonl")}
    if {row["pair_id"] for row in pair_rows} != pair_ids:
        raise IntegrityFailure("pairwise_terminal_coverage_mismatch")
    if any(row["state"] != "TERMINAL" for row in ledger):
        raise IntegrityFailure("pairwise_call_ledger_has_nonterminal_row")
    if any(row.get("retry_count", 0) != 0 for row in ledger):
        raise IntegrityFailure("same_request_retry_detected")
    if any(row.get("hosted_calls", 0) != 0 for row in ledger):
        raise IntegrityFailure("hosted_call_detected")

    review_payload = {
        **packet,
        "reviewed_groups": human_review,
        "review_summary": {
            "true_singleton_slot": sum(row["human_review"]["decision"] == "true_singleton_slot" for row in human_review),
            "false_revision_merge": sum(row["human_review"]["decision"] == "false_revision_merge" for row in human_review),
            "uncertain": sum(row["human_review"]["decision"] == "uncertain" for row in human_review),
        },
    }
    _freeze(RUN_DIR / "admitted_groups_human_review.json", _json_bytes(review_payload))
    _freeze(RUN_DIR / "critical_case_review.json", _json_bytes(cases))

    provider_calls = sum(int(row.get("provider_calls", 0)) for row in ledger)
    latencies = [row["latency_ms"] for row in ledger if isinstance(row.get("latency_ms"), (int, float))]
    prompt_tokens = [row["prompt_tokens"] for row in ledger if isinstance(row.get("prompt_tokens"), int)]
    final_stats = {
        **stats,
        "pairwise_provider_calls": provider_calls,
        "pairwise_prompt_tokens_total": sum(prompt_tokens),
        "pairwise_prompt_tokens_p50": _percentile(prompt_tokens, 0.5),
        "pairwise_latency_ms_p50": _percentile(latencies, 0.5),
        "pairwise_latency_ms_p95": _percentile(latencies, 0.95),
        "same_request_retries": 0,
        "hosted_calls": 0,
        "timestamp_fields_loaded": 0,
        "timestamp_fields_used": 0,
        "full_identity_calls": 0,
        "memory_store_mutations": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "SUPERSEDED": 0},
        "completion_marker": "MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=YES",
        "readiness_marker": f"MEM3B0P_MEM3B1_READY={'YES' if readiness else 'NO'}",
    }
    _freeze(RUN_DIR / "safety_statistics.json", _json_bytes(final_stats))
    _refresh_manifest_sidecar()

    report = _render_report(manifest, final_stats, cases, review_payload, readiness)
    _freeze(RUN_DIR / "report.md", report.encode("utf-8"))
    manifest["status"] = "COMPLETE"
    manifest["completion_gate_marker"] = "MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=YES"
    manifest["readiness_gate_marker"] = f"MEM3B0P_MEM3B1_READY={'YES' if readiness else 'NO'}"
    manifest["completion"] = {
        "structural_complete": True,
        "readiness": "YES" if readiness else "NO",
        "human_review_summary": review_payload["review_summary"],
        "critical_case_review_sha256": _verify_sidecar(RUN_DIR / "critical_case_review.json"),
        "human_review_sha256": _verify_sidecar(RUN_DIR / "admitted_groups_human_review.json"),
        "report_sha256": _verify_sidecar(RUN_DIR / "report.md"),
    }
    manifest.setdefault("frozen_artifacts", {}).update(
        {
            "critical_case_review.json": _verify_sidecar(RUN_DIR / "critical_case_review.json"),
            "admitted_groups_human_review.json": _verify_sidecar(RUN_DIR / "admitted_groups_human_review.json"),
            "safety_statistics.json": _verify_sidecar(RUN_DIR / "safety_statistics.json"),
            "report.md": _verify_sidecar(RUN_DIR / "report.md"),
        }
    )
    _freeze(RUN_DIR / "run_manifest.json", _json_bytes(manifest))
    _refresh_manifest_sidecar()
    validate_artifacts()
    return {
        "structural_complete": True,
        "readiness": "YES" if readiness else "NO",
        "false_safe_examples_blocked": cases["all_historical_b0s_false_safe_examples_blocked"],
        "instagram_pairwise_revision_admission": cases["instagram_pairwise_revision_admission"],
        "gym_revision_admission": cases["gym_revision_admission"],
        "admitted_group_count": len(human_review),
        "run_dir": str(RUN_DIR),
    }


def _percentile(values: list[int] | list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction + 0.999999) - 1)))
    return round(float(ordered[index]), 3)


def _render_report(
    manifest: dict[str, Any],
    stats: dict[str, Any],
    cases: dict[str, Any],
    review: dict[str, Any],
    readiness: bool,
) -> str:
    verdicts = stats["pairwise_verdict_counts"]
    lines = [
        "# MEM-3B0P Pairwise Revision Admission Closeout",
        "",
        "This is a structural/safety admission experiment, not a QA benchmark or performance comparison.",
        "",
        "## Frozen Inputs",
        "",
        f"- Base commit: `{manifest['base_commit_sha']}`.",
        f"- FlatProp rows/hash: {manifest['upstream']['flatprop_count']} / `{manifest['upstream']['flatprop_sha256']}`.",
        f"- B0R identity rows/hash: {manifest['upstream']['identity_count']} / `{manifest['upstream']['identity_sha256']}`; hints only.",
        f"- Scorecard contract commit: `{manifest['scorecard_lock_commit_sha']}`; committed before all B0P model requests.",
        "",
        "## Grounding And Admission",
        "",
        f"- Grounding outcomes: `{json.dumps(stats['proposal_grounding_counts'], sort_keys=True)}`.",
        f"- Repeated candidate groups: {stats['repeated_candidate_groups']}; unordered pair requests: {stats['unordered_pair_count']}.",
        f"- Pair verdicts: `{json.dumps(verdicts, sort_keys=True)}`.",
        f"- Admitted / blocked / no-history groups: {stats['repeated_candidate_groups_admitted']} / {stats['repeated_candidate_groups_blocked']} / {stats['singleton_groups_no_revision_history']}.",
        f"- Opaque Harness revision slots: {stats['opaque_revision_slots_created']}.",
        "- Group admission required every unordered pair to be `SAME_MUTABLE_SLOT`; no transitive clustering was used.",
        "",
        "## Critical Controls",
        "",
        f"- Instagram 500/600 positive control: `{cases['instagram_pairwise_revision_admission']}`.",
        f"- Historical B0S false-safe examples all blocked/not admitted: `{cases['all_historical_b0s_false_safe_examples_blocked']}`.",
        f"- Gym cross-key recall diagnostic: `{cases['gym_revision_admission']}`; no cross-key merge was introduced.",
        f"- Exhaustively human-reviewed admitted groups: {review['review_summary']}.",
        "",
        "## Runtime And Safety",
        "",
        f"- Runtime model/server: `{manifest.get('runtime', {}).get('model_sha256', 'not recorded')}` / `{manifest.get('runtime', {}).get('server_build', 'not recorded')}`.",
        f"- Pairwise local provider calls: {stats['pairwise_provider_calls']}; hosted calls: {stats['hosted_calls']}; same-request retries: {stats['same_request_retries']}.",
        f"- Prompt tokens total / p50: {stats['pairwise_prompt_tokens_total']} / {stats['pairwise_prompt_tokens_p50']}; latency p50 / p95 ms: {stats['pairwise_latency_ms_p50']} / {stats['pairwise_latency_ms_p95']}.",
        "- Timestamp fields loaded/used: `0/0`; no timestamp values appear in verifier prompts, decisions, human review, or this report.",
        "- MemoryStore mutations `ADD/UPDATE/DELETE/SUPERSEDED`: `0/0/0/0`; embeddings, retrieval, answer-reader, judge, and benchmark calls: `0`.",
        "",
        "## Outcome",
        "",
        "`MEM3B0P_PAIRWISE_REVISION_ADMISSION_COMPLETE=YES`.",
        f"`MEM3B0P_MEM3B1_READY={'YES' if readiness else 'NO'}`.",
        "",
        "No LongMemEval or MedMemoryBench performance claim is made from this stage.",
    ]
    return "\n".join(lines) + "\n"


def validate_artifacts() -> dict[str, Any]:
    manifest = _read_manifest()
    if manifest.get("status") != "COMPLETE":
        raise IntegrityFailure("run_not_structurally_complete")
    sidecars = sorted(RUN_DIR.glob("*.sha256"))
    for sidecar in sidecars:
        name = sidecar.read_text(encoding="utf-8").strip().split()
        if len(name) != 2:
            raise IntegrityFailure(f"invalid_sidecar_format:{sidecar.name}")
        target = RUN_DIR / name[1]
        _verify_sidecar(target)
    required = {
        "run_manifest.json",
        "memory_module_scorecard_contract.json",
        "proposal_grounding_contract.json",
        "proposal_grounding_decisions.jsonl",
        "pairwise_verifier_contract.json",
        "pairwise_verdicts.jsonl",
        "pairwise_call_ledger.jsonl",
        "revision_admission_overlay.jsonl",
        "revision_slot_manifest.json",
        "safety_statistics.json",
        "critical_case_review.json",
        "admitted_groups_human_review.json",
        "report.md",
        "semantic_freeze.json",
    }
    missing = sorted(name for name in required if not (RUN_DIR / name).is_file())
    if missing:
        raise IntegrityFailure(f"required_artifact_missing:{','.join(missing)}")
    return {"validated_sha256_sidecars": len(sidecars), "required_artifacts": len(required)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "review-packet", "closeout", "validate"))
    parser.add_argument("--review-decisions", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "verify":
        result = verify_pairs()
    elif args.command == "review-packet":
        result = build_human_review_packet()
    elif args.command == "closeout":
        if args.review_decisions is None:
            parser.error("closeout requires --review-decisions")
        result = closeout(args.review_decisions)
    else:
        result = validate_artifacts()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
