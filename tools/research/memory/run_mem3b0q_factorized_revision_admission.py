"""Run the staged local-only MEM-3B0Q factorized revision-admission study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import factorized_revision_admission as factorized
from tools.research.memory import revision_pairwise_admission as b0p
from tools.research.memory import run_mem3b0p_pairwise_admission as b0p_runner
from tools.research.memory.local_qwen3_embedding import LocalQwen3Embedding

RUN_ID = "mem3b0q-factorized-admission-20260930"
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
SCORECARD_PATH = ROOT / "docs" / "research" / "memory" / "memory_module_scorecard_contract.json"
PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem_3b0q_factorized_admission_protocol.md"
PROTOCOL_SHA_PATH = PROTOCOL_PATH.with_suffix(".sha256")
EMBEDDING_PATH = Path(r"E:\Health-Copilot-Models\models\Qwen3-Embedding-0.6B")
EXPECTED_FLATPROP_SHA256 = "250a24a10b532cb01bd841b965511df019fc0e597ec1e85ee12d7830c166c5fe"
EXPECTED_IDENTITY_SHA256 = "49af354bb535909a572df580cf02483ab56022dcf31873266154caf0086b4f75"
EXPECTED_SCORECARD_SHA256 = "457efc4d3bb58766fafa4d2486b34a881326eef1983687b9fbd113857ab36568"
EXPECTED_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
EXPECTED_EMBEDDING_TREE_SHA256 = "9d2d790d6448ef2c0911ffeb03f959d035c71ac3d2b14b7d586f2d2b39fb0efa"
EXPECTED_EMBEDDING_WEIGHTS_SHA256 = (
    "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
)
API_BASE = "http://127.0.0.1:8081/v1"
BASE_URL = "http://127.0.0.1:8081"
MAX_CONTEXT = 131072
MAX_COMPLETION = 64
TEMPERATURE = 0
SEED = 42
SOURCE_FIELDS = b0p.PROPOSAL_SOURCE_FIELDS
IDENTITY_FIELDS = b0p.PROPOSAL_IDENTITY_FIELDS


class IntegrityFailure(RuntimeError):
    """A frozen input, runtime, or stage invariant failed."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(factorized.canonical_json_bytes(row) + b"\n" for row in rows)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _freeze(path: Path, content: bytes) -> str:
    _atomic_write(path, content)
    digest = hashlib.sha256(content).hexdigest()
    _atomic_write(path.with_suffix(".sha256"), f"{digest}  {path.name}\n".encode())
    return digest


def _verify_sidecar(path: Path, expected: str | None = None) -> str:
    if not path.is_file():
        raise IntegrityFailure(f"missing_artifact:{path}")
    digest = _sha_file(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise IntegrityFailure(f"missing_sidecar:{sidecar}")
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != path.name or parts[0] != digest:
        raise IntegrityFailure(f"sidecar_mismatch:{path}")
    if expected is not None and digest != expected:
        raise IntegrityFailure(f"frozen_hash_mismatch:{path}")
    return digest


def _project_jsonl(path: Path, allowlist: frozenset[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = b0p.jsonl_projection(line, allowlist)
            if not set(row).issubset(allowlist):
                raise IntegrityFailure(f"allowlist_violation:{path}:{line_number}")
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("memory_id", "")))
    return rows


def _read_frozen_jsonl(name: str) -> list[dict[str, Any]]:
    path = RUN_DIR / name
    _verify_sidecar(path)
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise IntegrityFailure(f"non_object_artifact_row:{name}:{line_number}")
            rows.append(row)
    return rows


def _read_frozen_json(name: str) -> dict[str, Any]:
    path = RUN_DIR / name
    _verify_sidecar(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityFailure(f"non_object_json_artifact:{name}")
    return value


def _write_manifest(manifest: dict[str, Any]) -> None:
    _freeze(RUN_DIR / "run_manifest.json", _json_bytes(manifest))


def _record(manifest: dict[str, Any], path: Path, payload: bytes) -> str:
    digest = _freeze(path, payload)
    manifest.setdefault("frozen_artifacts", {})[path.name] = digest
    return digest


def _read_manifest() -> dict[str, Any]:
    path = RUN_DIR / "run_manifest.json"
    _verify_sidecar(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("stage") != RUN_ID:
        raise IntegrityFailure("run_manifest_identity_mismatch")
    return manifest


def _assert_code_freeze(manifest: dict[str, Any]) -> None:
    if _source_code_hashes() != manifest.get("source_code_sha256"):
        raise IntegrityFailure("source_code_changed_after_protocol_freeze")


def _self_test_forbidden_metadata_skip() -> dict[str, Any]:
    sentinel = "9" * 5000
    raw = (
        '{"memory_id":"sentinel","scope_id":"scope","proposition_text":"p",'
        '"source_authority":"user","observed_at":{"nested":[{"sentinel":' + sentinel + "}]}}"
    )
    projected = b0p.jsonl_projection(raw, SOURCE_FIELDS)
    expected = {
        "memory_id": "sentinel",
        "scope_id": "scope",
        "proposition_text": "p",
        "source_authority": "user",
    }
    if projected != expected:
        raise IntegrityFailure("forbidden_metadata_sentinel_projection_failed")
    return {"forbidden_metadata_decode_count": 0, "sentinel_projection_passed": True}


def _source_code_hashes() -> dict[str, str]:
    return {
        "runner": _sha_file(Path(__file__)),
        "factorized_module": _sha_file(Path(factorized.__file__)),
        "source_reader_module": _sha_file(Path(b0p.__file__)),
        "b0s_eligibility_module": _sha_file(
            Path(b0p_runner.__file__).with_name("revision_safety_overlay.py")
        ),
        "protocol": _sha_file(PROTOCOL_PATH),
        "scorecard": _sha_file(SCORECARD_PATH),
    }


def _candidate_contracts() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], str, str, str, str
]:
    candidate = factorized.candidate_generation_contract()
    candidate["candidate_sources"]["SEMANTIC_NEIGHBOR"].update(
        {
            "model_tree_sha256": EXPECTED_EMBEDDING_TREE_SHA256,
            "weights_sha256": EXPECTED_EMBEDDING_WEIGHTS_SHA256,
            "device": "cuda:0",
            "dtype": "float16",
            "batch_size": 8,
            "max_length": 8192,
            "max_batch_tokens": 8192,
            "dimension": 1024,
            "normalization": "L2 float32 output",
            "document_instruction": None,
        }
    )
    factor_contract = factorized.factorized_contract()
    factor_contract["runtime"] = {
        "endpoint": API_BASE,
        "loopback_only": True,
        "llama_cpp_build": b0p_runner.EXPECTED_SERVER_BUILD,
        "llama_server_sha256": b0p_runner.EXPECTED_SERVER_SHA256,
        "context_tokens": MAX_CONTEXT,
        "max_completion_tokens": MAX_COMPLETION,
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
    admission_contract = {
        "schema_version": 1,
        "contract_id": "mem3b0q-harness-graph-clique-slot-admission-v1",
        "positive_edge": "same_state_dimension=YES AND state_cardinality=SINGLE_VALUE_AT_A_TIME; value_relation does not encode chronology",
        "closure": "all missing intra-component pairs for preliminary positive components of size 2..16; larger components blocked without quadratic expansion",
        "grouping": "deterministic maximal cliques of size >=2; all memberships of any vertex in multiple maximal cliques are ambiguous and blocked",
        "slot_id": "SHA256(UTF8(scope_id + concat(sorted(member_memory_ids)) + admission_contract_sha256))",
        "slot_id_authority": "Harness only; B0R subject/attribute keys excluded",
        "state_mutations": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "SUPERSEDED": 0},
    }
    candidate_sha = factorized.sha256_bytes(factorized.canonical_json_bytes(candidate))
    factor_sha = factorized.sha256_bytes(factorized.canonical_json_bytes(factor_contract))
    admission_sha = factorized.sha256_bytes(factorized.canonical_json_bytes(admission_contract))
    pair_identity_sha = factorized.sha256_bytes(
        factorized.canonical_json_bytes([candidate_sha, factor_sha])
    )
    return (
        candidate,
        factor_contract,
        admission_contract,
        candidate_sha,
        factor_sha,
        admission_sha,
        pair_identity_sha,
    )


FACTOR_CONTRACT = factorized.factorized_contract()
FACTOR_CONTRACT["runtime"] = {
    "endpoint": API_BASE,
    "loopback_only": True,
    "llama_cpp_build": b0p_runner.EXPECTED_SERVER_BUILD,
    "llama_server_sha256": b0p_runner.EXPECTED_SERVER_SHA256,
    "context_tokens": MAX_CONTEXT,
    "max_completion_tokens": MAX_COMPLETION,
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


def prepare() -> dict[str, Any]:
    if RUN_DIR.exists():
        raise IntegrityFailure(f"run_directory_already_exists:{RUN_DIR}")
    if not PROTOCOL_PATH.is_file() or not SCORECARD_PATH.is_file():
        raise IntegrityFailure("frozen_protocol_or_scorecard_missing")
    if _verify_sidecar(FLATPROP_PATH, EXPECTED_FLATPROP_SHA256) != EXPECTED_FLATPROP_SHA256:
        raise IntegrityFailure("flatprop_frozen_input_hash_mismatch")
    if _verify_sidecar(IDENTITY_PATH, EXPECTED_IDENTITY_SHA256) != EXPECTED_IDENTITY_SHA256:
        raise IntegrityFailure("b0r_identity_frozen_input_hash_mismatch")
    if _sha_file(SCORECARD_PATH) != EXPECTED_SCORECARD_SHA256:
        raise IntegrityFailure("frozen_scorecard_hash_mismatch")
    protocol_parts = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip().split()
    if (
        len(protocol_parts) != 2
        or protocol_parts[1] != PROTOCOL_PATH.name
        or protocol_parts[0] != _sha_file(PROTOCOL_PATH)
    ):
        raise IntegrityFailure("protocol_sidecar_mismatch")
    if (
        json.loads(SCORECARD_PATH.read_text(encoding="utf-8")).get("contract_id")
        != "health-copilot-memory-module-scorecard-v1"
    ):
        raise IntegrityFailure("scorecard_contract_id_mismatch")
    reader_test = _self_test_forbidden_metadata_skip()
    code_hashes = _source_code_hashes()
    scorecard_lock_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(SCORECARD_PATH.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
    ).strip()
    if (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", scorecard_lock_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise IntegrityFailure("scorecard_lock_not_in_current_branch")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty:
        raise IntegrityFailure("working_tree_must_be_committed_before_source_projection_freeze")

    (
        candidate_contract,
        factor_contract,
        admission_contract,
        candidate_sha,
        factor_sha,
        admission_sha,
        pair_identity_sha,
    ) = _candidate_contracts()
    RUN_DIR.mkdir(parents=True)
    source_rows = _project_jsonl(FLATPROP_PATH, SOURCE_FIELDS)
    if len(source_rows) != 8112:
        raise IntegrityFailure("flatprop_row_count_mismatch")
    source_sha = _freeze(
        RUN_DIR / "revision_semantic_source_projection.jsonl", _jsonl_bytes(source_rows)
    )
    identity_rows = _project_jsonl(IDENTITY_PATH, IDENTITY_FIELDS)
    if len(identity_rows) != len(source_rows):
        raise IntegrityFailure("identity_row_count_mismatch")
    _freeze(RUN_DIR / "b0r_identity_hints_projection.jsonl", _jsonl_bytes(identity_rows))
    candidate_contract_file_sha = _freeze(
        RUN_DIR / "candidate_generation_contract.json", _json_bytes(candidate_contract)
    )
    factor_contract_file_sha = _freeze(
        RUN_DIR / "factorized_verifier_contract.json", _json_bytes(factor_contract)
    )
    admission_contract_file_sha = _freeze(
        RUN_DIR / "slot_admission_contract.json", _json_bytes(admission_contract)
    )
    manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "SOURCE_AND_PROTOCOL_FROZEN",
        "base_commit_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "canonical_main_at_stage_start": "5b0dc5c4d32b4fcbea55f697766f521d3551b175",
        "source_code_sha256": code_hashes,
        "scorecard_lock_commit_sha": scorecard_lock_commit,
        "scorecard_sha256": EXPECTED_SCORECARD_SHA256,
        "upstream": {
            "flatprop_rows": len(source_rows),
            "flatprop_path": str(FLATPROP_PATH.relative_to(ROOT)),
            "flatprop_sha256": EXPECTED_FLATPROP_SHA256,
            "semantic_source_projection_sha256": source_sha,
            "b0r_identity_rows": len(identity_rows),
            "b0r_identity_path": str(IDENTITY_PATH.relative_to(ROOT)),
            "b0r_identity_sha256": EXPECTED_IDENTITY_SHA256,
            "b0r_hint_role": "diagnostic candidate hints only; no authority for identity, groups, supersession, or currentness",
        },
        "contracts": {
            "candidate_generation_contract_file_sha256": candidate_contract_file_sha,
            "candidate_generation_contract_canonical_sha256": candidate_sha,
            "factorized_verifier_contract_file_sha256": factor_contract_file_sha,
            "factorized_verifier_contract_canonical_sha256": factor_sha,
            "slot_admission_contract_file_sha256": admission_contract_file_sha,
            "slot_admission_contract_canonical_sha256": admission_sha,
            "candidate_pair_identity_contract_sha256": pair_identity_sha,
        },
        "runtime_policy": {
            "reader_model": "Qwen3-8B Q4_K_M",
            "reader_model_sha256": EXPECTED_MODEL_SHA256,
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "hosted_calls": 0,
            "same_request_retries": 0,
            "judge": "NONE",
            "api_key_required": False,
            "memory_store_mutations": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "SUPERSEDED": 0},
        },
        "timestamp_barrier": {
            "forbidden_metadata_decode_count": 0,
            "timestamp_fields_decoded_before_semantic_freeze": 0,
            "timestamp_fields_used_before_semantic_freeze": 0,
            "sentinel_test": reader_test,
        },
        "scope_guards": {"dev_opened": False, "test_opened": False, "medmemorybench_run": False},
        "execution": {"embedding_calls": 0, "seed_pair_calls": 0, "closure_pair_calls": 0},
        "frozen_artifacts": {},
    }
    for name in (
        "revision_semantic_source_projection.jsonl",
        "b0r_identity_hints_projection.jsonl",
        "candidate_generation_contract.json",
        "factorized_verifier_contract.json",
        "slot_admission_contract.json",
    ):
        manifest["frozen_artifacts"][name] = _verify_sidecar(RUN_DIR / name)
    _write_manifest(manifest)
    return {
        "status": manifest["status"],
        "source_projection_rows": len(source_rows),
        "source_projection_sha256": source_sha,
        "forbidden_metadata_decode_count": 0,
        "candidate_generation_started": False,
        "model_calls_started": False,
        "run_dir": str(RUN_DIR),
    }


def _load_eligible() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_manifest()
    if manifest.get("status") not in {
        "SOURCE_AND_PROTOCOL_FROZEN",
        "CANDIDATE_PAIRS_FROZEN",
        "SEED_VERIFICATION_IN_PROGRESS",
        "SEED_PAIRS_TERMINAL",
        "CLOSURE_PAIRS_FROZEN",
        "CLOSURE_VERIFICATION_IN_PROGRESS",
        "PAIRWISE_VERDICTS_TERMINAL",
        "SEMANTIC_DECISIONS_FROZEN",
    }:
        raise IntegrityFailure(f"invalid_pre_candidate_status:{manifest.get('status')}")
    source_rows = _read_frozen_jsonl("revision_semantic_source_projection.jsonl")
    identity_rows = _read_frozen_jsonl("b0r_identity_hints_projection.jsonl")
    eligible, decisions, stats = factorized.eligible_records(source_rows, identity_rows)
    if len(source_rows) != manifest["upstream"]["flatprop_rows"]:
        raise IntegrityFailure("sanitized_projection_row_count_changed")
    return eligible, decisions, stats


def generate_candidates() -> dict[str, Any]:
    manifest = _read_manifest()
    if manifest.get("status") != "SOURCE_AND_PROTOCOL_FROZEN":
        raise IntegrityFailure(f"candidate_generation_not_allowed_from:{manifest.get('status')}")
    if _source_code_hashes() != manifest.get("source_code_sha256"):
        raise IntegrityFailure("source_code_changed_after_protocol_freeze")
    eligible, decisions, eligibility_stats = _load_eligible()
    if not eligible:
        raise IntegrityFailure("no_eligible_records")
    embedding_contract = _read_frozen_json("candidate_generation_contract.json")[
        "candidate_sources"
    ]["SEMANTIC_NEIGHBOR"]
    if embedding_contract.get("top_k") != 4:
        raise IntegrityFailure("frozen_semantic_top_k_mismatch")
    model_path = EMBEDDING_PATH
    if not model_path.is_dir():
        raise IntegrityFailure(f"local_embedding_model_missing:{model_path}")
    import numpy as np
    import torch

    if (
        torch.__version__ != "2.10.0+cu128"
        or torch.version.cuda != "12.8"
        or not torch.cuda.is_available()
    ):
        raise IntegrityFailure("frozen_mem1_cuda_embedding_runtime_mismatch")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 Laptop GPU":
        raise IntegrityFailure("frozen_mem1_gpu_mismatch")
    pre_free, pre_total = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats(0)
    adapter = LocalQwen3Embedding(model_path)
    try:
        if (
            adapter.identity["model_tree_sha256"] != EXPECTED_EMBEDDING_TREE_SHA256
            or adapter.identity["weights_sha256"] != EXPECTED_EMBEDDING_WEIGHTS_SHA256
        ):
            raise IntegrityFailure("embedding_model_identity_mismatch")
        texts = [row["proposition_text"] for row in eligible]
        token_counts = adapter.count_tokens(texts, "document")
        if any(count > adapter.max_length for count in token_counts):
            raise IntegrityFailure("semantic_neighbor_document_truncation_forbidden")
        started = time.perf_counter()
        vectors = np.empty(
            (len(eligible), adapter.identity.get("dimension", 1024)), dtype=np.float32
        )
        batch_rows = list(adapter.encode_batches(texts, "document", token_counts=token_counts))
        for batch in batch_rows:
            vectors[batch.start : batch.end] = batch.vectors
        embedding_ms = round((time.perf_counter() - started) * 1000, 3)
        if vectors.shape != (len(eligible), 1024):
            raise IntegrityFailure("embedding_output_dimension_mismatch")
        vector_by_id = {row["memory_id"]: vectors[index] for index, row in enumerate(eligible)}
        factor_contract_sha = manifest["contracts"]["candidate_pair_identity_contract_sha256"]
        pairs, pair_stats = factorized.candidate_pairs(
            eligible,
            vector_by_id,
            factor_contract_sha,
            top_k=int(embedding_contract["top_k"]),
        )
        post_free, _ = torch.cuda.mem_get_info(0)
        embedding_runtime = {
            **adapter.identity,
            "model_id": adapter.identity["model_id"],
            "device": adapter.device,
            "device_name": adapter.device_name,
            "dtype": adapter.dtype,
            "dimensions": 1024,
            "normalization": "L2 float32 output",
            "document_instruction": None,
            "batch_size": adapter.batch_size,
            "max_length": adapter.max_length,
            "max_batch_tokens": adapter.max_batch_tokens,
            "document_count": len(texts),
            "input_tokens": sum(token_counts),
            "truncated_count": 0,
            "batch_count": len(batch_rows),
            "wall_ms": embedding_ms,
            "torch_version": adapter.torch_version,
            "cuda_version": adapter.cuda_version,
            "gpu_memory_before_bytes": {"free": pre_free, "total": pre_total},
            "gpu_memory_after_bytes": {"free": post_free, "total": pre_total},
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        }
    finally:
        adapter.close()
        torch.cuda.empty_cache()

    identity_sha = _freeze(RUN_DIR / "eligibility_decisions.jsonl", _jsonl_bytes(decisions))
    eligible_sha = _freeze(RUN_DIR / "eligible_records.jsonl", _jsonl_bytes(eligible))
    pair_sha = _freeze(RUN_DIR / "candidate_pair_manifest.jsonl", _jsonl_bytes(pairs))
    generation_stats = {
        **eligibility_stats,
        **pair_stats,
        "embedding_runtime": embedding_runtime,
        "forbidden_metadata_decode_count": 0,
        "timestamp_fields_used": 0,
    }
    stats_sha = _freeze(
        RUN_DIR / "candidate_generation_statistics.json", _json_bytes(generation_stats)
    )
    manifest.setdefault("frozen_artifacts", {}).update(
        {
            "eligibility_decisions.jsonl": identity_sha,
            "eligible_records.jsonl": eligible_sha,
            "candidate_pair_manifest.jsonl": pair_sha,
            "candidate_generation_statistics.json": stats_sha,
        }
    )
    manifest["status"] = "CANDIDATE_PAIRS_FROZEN"
    manifest["embedding_runtime"] = embedding_runtime
    manifest["candidate_statistics"] = {
        key: value for key, value in generation_stats.items() if key != "embedding_runtime"
    }
    manifest["execution"]["embedding_calls"] = len(texts)
    _write_manifest(manifest)
    return {
        "eligible_records": len(eligible),
        "exact_hint_seed_pairs": pair_stats["exact_hint_seed_pairs"],
        "semantic_neighbor_seed_pairs": pair_stats["semantic_neighbor_seed_pairs"],
        "semantic_neighbor_retrievals_directed": pair_stats[
            "semantic_neighbor_retrievals_directed"
        ],
        "union_seed_pairs": len(pairs),
        "embedding_wall_ms": embedding_ms,
        "status": manifest["status"],
    }


def _runtime_verify(client: httpx.Client) -> dict[str, Any]:
    runtime = b0p_runner._verify_runtime(client)
    if (
        runtime.get("model_sha256") != EXPECTED_MODEL_SHA256
        or runtime.get("context_tokens") != MAX_CONTEXT
    ):
        raise IntegrityFailure("frozen_qwen_reader_identity_mismatch")
    runtime["generation"] = {
        "temperature": TEMPERATURE,
        "seed": SEED,
        "enable_thinking": False,
        "max_tokens": MAX_COMPLETION,
    }
    runtime["hosted_calls"] = 0
    return runtime


def _ledger_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in result:
            raise IntegrityFailure("duplicate_or_missing_pair_ledger_id")
        result[pair_id] = row
    return result


def _write_ledger(ledger: dict[str, dict[str, Any]], pair_order: list[str]) -> str:
    rows = [ledger[pair_id] for pair_id in pair_order if pair_id in ledger]
    return _freeze(RUN_DIR / "pairwise_call_ledger.jsonl", _jsonl_bytes(rows))


def _record_infra_failure(manifest: dict[str, Any], error: Exception) -> None:
    payload = {
        "failure_class": "GLOBAL_INFRA_FAILURE",
        "reason": type(error).__name__,
        "detail": str(error),
        "hosted_calls": 0,
        "same_request_retries": 0,
    }
    digest = _freeze(RUN_DIR / "execution_failure.json", _json_bytes(payload))
    manifest.setdefault("frozen_artifacts", {})["execution_failure.json"] = digest
    manifest["status"] = "INFRA_FAILURE"
    manifest["execution_failure"] = payload
    _write_manifest(manifest)


def _verify_pairs(
    pair_name: str, result_name: str, expected_status: str, in_progress_status: str, phase: str
) -> dict[str, Any]:
    manifest = _read_manifest()
    if manifest.get("status") not in {expected_status, in_progress_status}:
        raise IntegrityFailure(f"pair_verification_invalid_status:{manifest.get('status')}")
    _assert_code_freeze(manifest)
    frozen_contract = _read_frozen_json("factorized_verifier_contract.json")
    if frozen_contract != FACTOR_CONTRACT:
        raise IntegrityFailure("factorized_verifier_contract_changed")
    if (
        factorized.sha256_bytes(factorized.canonical_json_bytes(frozen_contract))
        != manifest["contracts"]["factorized_verifier_contract_canonical_sha256"]
    ):
        raise IntegrityFailure("factorized_verifier_contract_sha_mismatch")
    pairs = _read_frozen_jsonl(pair_name)
    pair_order = [str(row["pair_id"]) for row in pairs]
    if len(pair_order) != len(set(pair_order)):
        raise IntegrityFailure("duplicate_pair_in_frozen_manifest")
    forbidden_pair_fields = b0p.TEMPORAL_OR_BENCHMARK_FIELDS | {
        "scope_id",
        "subject_key",
        "attribute_key",
        "value_text",
        "source_authority",
        "revision_kind",
        "identity_origin",
        "question",
        "question_id",
        "gold",
    }
    if any(forbidden_pair_fields.intersection(pair) for pair in pairs):
        raise IntegrityFailure("candidate_pair_manifest_contains_forbidden_metadata")
    ledger_path = RUN_DIR / "pairwise_call_ledger.jsonl"
    if ledger_path.exists():
        _verify_sidecar(ledger_path)
        ledger = _ledger_index(_read_frozen_jsonl("pairwise_call_ledger.jsonl"))
    else:
        ledger = {}
    ledger_write_order = [
        row["pair_id"] for row in _read_frozen_jsonl("candidate_pair_manifest.jsonl")
    ]
    closure_path = RUN_DIR / "closure_pair_manifest.jsonl"
    if closure_path.exists():
        ledger_write_order.extend(
            row["pair_id"] for row in _read_frozen_jsonl("closure_pair_manifest.jsonl")
        )
    if set(ledger) - set(ledger_write_order):
        raise IntegrityFailure("call_ledger_contains_pair_outside_frozen_manifests")

    timeout = httpx.Timeout(connect=10, read=180, write=30, pool=30)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        try:
            runtime = _runtime_verify(client)
        except Exception as exc:
            _record_infra_failure(manifest, exc)
            raise
        for index, pair in enumerate(pairs, 1):
            pid = pair["pair_id"]
            existing = ledger.get(pid)
            if existing and existing.get("state") == "TERMINAL":
                continue
            if existing and existing.get("state") == "PENDING":
                ledger[pid] = {
                    **existing,
                    **factorized.UNKNOWN_VERDICT,
                    "state": "TERMINAL",
                    "failure_code": "INTERRUPTED_PENDING_NEVER_RESEND",
                    "provider_call_uncertain": True,
                    "same_request_retries": 0,
                }
                _write_ledger(ledger, ledger_write_order)
                continue

            projection = factorized.request_projection(pair)
            messages = [
                {"role": "system", "content": FACTOR_CONTRACT["system_prompt"]},
                {
                    "role": "user",
                    "content": json.dumps(projection, ensure_ascii=False, separators=(",", ":")),
                },
            ]
            prompt_tokens = b0p_runner._rendered_prompt_tokens(client, messages)
            if prompt_tokens + MAX_COMPLETION > MAX_CONTEXT:
                raise IntegrityFailure(f"factorized_pair_prompt_exceeds_context:{pid}")
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
                    "json_schema": {
                        "name": "mem3b0q_factorized_pair_semantics",
                        "strict": True,
                        "schema": FACTOR_CONTRACT["output_schema"],
                    },
                },
            }
            request_sha = factorized.sha256_bytes(factorized.canonical_json_bytes(request))
            ledger[pid] = {
                "pair_id": pid,
                "phase": phase,
                "request_sha256": request_sha,
                "request_projection_sha256": factorized.sha256_bytes(
                    factorized.canonical_json_bytes(projection)
                ),
                "prompt_tokens": prompt_tokens,
                "state": "PENDING",
                "provider_calls": 1,
                "provider_call_uncertain": True,
                "hosted_calls": 0,
                "same_request_retries": 0,
                "latency_ms": None,
            }
            _write_ledger(ledger, ledger_write_order)
            started = time.perf_counter()
            try:
                response = client.post(f"{API_BASE}/chat/completions", json=request)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                message = choice.get("message") or {}
                verdict, failure_code = factorized.parse_factorized_output(
                    message.get("content"), choice.get("finish_reason")
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError) as exc:
                _record_infra_failure(manifest, exc)
                raise IntegrityFailure(f"local_pair_transport_or_protocol_failure:{pid}") from exc
            ledger[pid] = {
                **ledger[pid],
                **verdict,
                "state": "TERMINAL",
                "failure_code": failure_code,
                "provider_call_uncertain": False,
                "raw_completion": message.get("content"),
                "finish_reason": choice.get("finish_reason"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            _write_ledger(ledger, ledger_write_order)
            if index % 50 == 0 or index == len(pairs):
                print(f"{phase}_progress={index}/{len(pairs)}", flush=True)

    if set(ledger).intersection(pair_order) != set(pair_order) or any(
        ledger[pid].get("state") != "TERMINAL" for pid in pair_order
    ):
        manifest["status"] = in_progress_status
        manifest["runtime"] = runtime
        manifest["execution"][f"{phase}_pair_calls"] = sum(
            row.get("phase") == phase and row.get("provider_calls", 0) for row in ledger.values()
        )
        manifest.setdefault("frozen_artifacts", {})["pairwise_call_ledger.jsonl"] = _verify_sidecar(
            ledger_path
        )
        _write_manifest(manifest)
        raise IntegrityFailure(f"{phase}_pair_terminal_coverage_incomplete")

    result_rows = []
    for pair in pairs:
        call = ledger[pair["pair_id"]]
        result_rows.append(
            {
                "pair_id": pair["pair_id"],
                "memory_id_a": pair["memory_id_a"],
                "memory_id_b": pair["memory_id_b"],
                "candidate_sources": pair.get("candidate_sources", ["CLOSURE"]),
                "same_state_dimension": call.get("same_state_dimension", "UNKNOWN"),
                "state_cardinality": call.get("state_cardinality", "UNKNOWN"),
                "value_relation": call.get("value_relation", "UNKNOWN"),
                "pair_origin": call.get("pair_origin", "HARNESS_UNKNOWN_FALLBACK"),
                "failure_code": call.get("failure_code"),
                "request_sha256": call["request_sha256"],
            }
        )
    result_sha = _freeze(RUN_DIR / result_name, _jsonl_bytes(result_rows))
    manifest.setdefault("frozen_artifacts", {})[result_name] = result_sha
    manifest["runtime"] = runtime
    manifest["execution"][f"{phase}_pair_calls"] = sum(
        row.get("phase") == phase and row.get("provider_calls", 0) for row in ledger.values()
    )
    manifest.setdefault("frozen_artifacts", {})["pairwise_call_ledger.jsonl"] = _verify_sidecar(
        ledger_path
    )
    manifest["status"] = "SEED_PAIRS_TERMINAL" if phase == "seed" else "PAIRWISE_VERDICTS_TERMINAL"
    _write_manifest(manifest)
    return {
        "status": manifest["status"],
        "pair_count": len(result_rows),
        "unknown_count": sum(
            row["pair_origin"] == "HARNESS_UNKNOWN_FALLBACK" for row in result_rows
        ),
        "hosted_calls": 0,
        "same_request_retries": 0,
    }


def verify_seeds() -> dict[str, Any]:
    return _verify_pairs(
        "candidate_pair_manifest.jsonl",
        "seed_pairwise_verdicts.jsonl",
        "CANDIDATE_PAIRS_FROZEN",
        "SEED_VERIFICATION_IN_PROGRESS",
        "seed",
    )


def prepare_closure() -> dict[str, Any]:
    manifest = _read_manifest()
    _assert_code_freeze(manifest)
    if manifest.get("status") != "SEED_PAIRS_TERMINAL":
        raise IntegrityFailure("seed_pairs_must_be_terminal_before_closure_planning")
    eligible = _read_frozen_jsonl("eligible_records.jsonl")
    seed_pairs = _read_frozen_jsonl("candidate_pair_manifest.jsonl")
    seed_verdict_rows = _read_frozen_jsonl("seed_pairwise_verdicts.jsonl")
    seed_verdicts = {row["pair_id"]: row for row in seed_verdict_rows}
    closure, oversized = factorized.closure_pairs(
        eligible,
        seed_pairs,
        seed_verdicts,
        manifest["contracts"]["candidate_pair_identity_contract_sha256"],
    )
    edges = [
        [row["memory_id_a"], row["memory_id_b"]]
        for row in seed_verdict_rows
        if factorized.is_positive_singleton_edge(row)
    ]
    components = factorized.connected_components(
        [row["memory_id"] for row in eligible], [tuple(edge) for edge in edges]
    )
    pregraph = {
        "vertices": sorted(row["memory_id"] for row in eligible),
        "positive_edges": edges,
        "positive_components": components,
        "oversized_components_blocked": oversized,
    }
    graph_sha = _freeze(RUN_DIR / "preliminary_positive_graph.json", _json_bytes(pregraph))
    closure_sha = _freeze(RUN_DIR / "closure_pair_manifest.jsonl", _jsonl_bytes(closure))
    manifest.setdefault("frozen_artifacts", {}).update(
        {
            "preliminary_positive_graph.json": graph_sha,
            "closure_pair_manifest.jsonl": closure_sha,
        }
    )
    manifest["status"] = "CLOSURE_PAIRS_FROZEN"
    manifest["closure_statistics"] = {
        "preliminary_positive_edges": len(edges),
        "preliminary_non_singleton_components": sum(len(component) > 1 for component in components),
        "closure_pairs": len(closure),
        "oversized_components": len(oversized),
    }
    _write_manifest(manifest)
    if not closure:
        _freeze(RUN_DIR / "closure_pairwise_verdicts.jsonl", b"")
        manifest.setdefault("frozen_artifacts", {})["closure_pairwise_verdicts.jsonl"] = (
            _verify_sidecar(RUN_DIR / "closure_pairwise_verdicts.jsonl")
        )
        manifest["status"] = "PAIRWISE_VERDICTS_TERMINAL"
        _write_manifest(manifest)
    return {
        "status": manifest["status"],
        "closure_pairs": len(closure),
        "oversized_components": len(oversized),
    }


def verify_closure() -> dict[str, Any]:
    manifest = _read_manifest()
    if manifest.get("status") == "PAIRWISE_VERDICTS_TERMINAL" and not _read_frozen_jsonl(
        "closure_pair_manifest.jsonl"
    ):
        return {"status": "PAIRWISE_VERDICTS_TERMINAL", "pair_count": 0, "unknown_count": 0}
    return _verify_pairs(
        "closure_pair_manifest.jsonl",
        "closure_pairwise_verdicts.jsonl",
        "CLOSURE_PAIRS_FROZEN",
        "CLOSURE_VERIFICATION_IN_PROGRESS",
        "closure",
    )


def closeout() -> dict[str, Any]:
    manifest = _read_manifest()
    _assert_code_freeze(manifest)
    if manifest.get("status") != "PAIRWISE_VERDICTS_TERMINAL":
        raise IntegrityFailure("all_seed_and_closure_pairs_must_be_terminal")
    eligible = _read_frozen_jsonl("eligible_records.jsonl")
    seed_rows = _read_frozen_jsonl("seed_pairwise_verdicts.jsonl")
    closure_rows = _read_frozen_jsonl("closure_pairwise_verdicts.jsonl")
    all_verdicts = sorted(seed_rows + closure_rows, key=lambda row: row["pair_id"])
    expected_count = len(_read_frozen_jsonl("candidate_pair_manifest.jsonl")) + len(
        _read_frozen_jsonl("closure_pair_manifest.jsonl")
    )
    if (
        len(all_verdicts) != expected_count
        or len({row["pair_id"] for row in all_verdicts}) != expected_count
    ):
        raise IntegrityFailure("pairwise_verdict_coverage_incomplete_or_duplicate")
    verdict_by_id = {row["pair_id"]: row for row in all_verdicts}
    seed_manifest = _read_frozen_jsonl("candidate_pair_manifest.jsonl")
    closure_manifest = _read_frozen_jsonl("closure_pair_manifest.jsonl")
    all_pairs = sorted(seed_manifest + closure_manifest, key=lambda row: row["pair_id"])
    if set(verdict_by_id) != {row["pair_id"] for row in all_pairs}:
        raise IntegrityFailure("verdict_pair_manifest_mismatch")
    graph, slots, graph_stats = factorized.build_slot_manifest(
        eligible,
        all_pairs,
        verdict_by_id,
        manifest["contracts"]["slot_admission_contract_canonical_sha256"],
    )
    for pair in all_verdicts:
        if pair[
            "pair_origin"
        ] == "HARNESS_UNKNOWN_FALLBACK" and factorized.is_positive_singleton_edge(pair):
            raise IntegrityFailure("unknown_fallback_created_positive_edge")
    verdict_sha = _freeze(RUN_DIR / "pairwise_verdicts.jsonl", _jsonl_bytes(all_verdicts))
    graph_sha = _freeze(RUN_DIR / "positive_slot_graph.json", _json_bytes(graph))
    slots_sha = _freeze(
        RUN_DIR / "revision_slot_manifest.json", _json_bytes({"schema_version": 1, "slots": slots})
    )
    stats = {
        **manifest.get("candidate_statistics", {}),
        **manifest.get("closure_statistics", {}),
        **graph_stats,
        "seed_pair_count": len(seed_manifest),
        "closure_pair_count": len(closure_manifest),
        "factorized_pair_call_count": sum(
            row.get("provider_calls", 0) for row in _read_frozen_jsonl("pairwise_call_ledger.jsonl")
        ),
        "harness_unknown_pair_count": sum(
            row["pair_origin"] == "HARNESS_UNKNOWN_FALLBACK" for row in all_verdicts
        ),
        "harness_unknown_pair_rate": round(
            sum(row["pair_origin"] == "HARNESS_UNKNOWN_FALLBACK" for row in all_verdicts)
            / max(1, len(all_verdicts)),
            6,
        ),
        "hosted_calls": 0,
        "same_request_retries": 0,
        "forbidden_metadata_decode_count": 0,
        "timestamp_fields_used_before_semantic_freeze": 0,
        "memory_store_mutations": {"ADD": 0, "UPDATE": 0, "DELETE": 0, "SUPERSEDED": 0},
        "dev_opened": False,
        "test_opened": False,
        "medmemorybench_run": False,
    }
    stats_sha = _freeze(RUN_DIR / "safety_statistics.json", _json_bytes(stats))
    ledger_rows = _read_frozen_jsonl("pairwise_call_ledger.jsonl")
    structural_failures = []
    expected_pair_ids = {row["pair_id"] for row in all_pairs}
    if {row["pair_id"] for row in ledger_rows} != expected_pair_ids:
        structural_failures.append("PAIRWISE_LEDGER_COVERAGE_MISMATCH")
    if any(row.get("state") != "TERMINAL" for row in ledger_rows):
        structural_failures.append("NONTERMINAL_PAIRWISE_LEDGER_ROW")
    if any(row.get("hosted_calls", 0) != 0 for row in ledger_rows):
        structural_failures.append("HOSTED_CALL_DETECTED")
    if any(row.get("same_request_retries", 0) != 0 for row in ledger_rows):
        structural_failures.append("SAME_REQUEST_RETRY_DETECTED")
    if manifest.get("runtime", {}).get("loopback_only") is not True:
        structural_failures.append("LOCAL_READER_RUNTIME_UNVERIFIED")
    embedding_runtime = manifest.get("embedding_runtime", {})
    if (
        embedding_runtime.get("model_tree_sha256") != EXPECTED_EMBEDDING_TREE_SHA256
        or embedding_runtime.get("weights_sha256") != EXPECTED_EMBEDDING_WEIGHTS_SHA256
        or embedding_runtime.get("device") != "cuda:0"
        or embedding_runtime.get("dtype") != "float16"
        or embedding_runtime.get("truncated_count") != 0
    ):
        structural_failures.append("LOCAL_EMBEDDING_RUNTIME_UNVERIFIED")
    if manifest.get("timestamp_barrier", {}).get("forbidden_metadata_decode_count") != 0:
        structural_failures.append("FORBIDDEN_METADATA_DECODE_DETECTED")
    if any(slot.get("slot_owner") != "Harness" for slot in slots):
        structural_failures.append("NON_HARNESS_SLOT_IDENTITY")
    if manifest.get("scope_guards") != {
        "dev_opened": False,
        "test_opened": False,
        "medmemorybench_run": False,
    }:
        structural_failures.append("OUT_OF_SCOPE_EVALUATION_OPENED")
    if any(manifest.get("runtime_policy", {}).get("memory_store_mutations", {}).values()):
        structural_failures.append("MEMORY_STORE_MUTATION_DETECTED")
    for name in (
        "revision_semantic_source_projection.jsonl",
        "candidate_pair_manifest.jsonl",
        "seed_pairwise_verdicts.jsonl",
        "closure_pair_manifest.jsonl",
        "closure_pairwise_verdicts.jsonl",
        "pairwise_verdicts.jsonl",
        "positive_slot_graph.json",
        "revision_slot_manifest.json",
        "safety_statistics.json",
    ):
        _verify_sidecar(RUN_DIR / name)
    if structural_failures:
        raise IntegrityFailure(f"structural_completion_gate_failed:{','.join(structural_failures)}")
    manifest.setdefault("frozen_artifacts", {}).update(
        {
            "pairwise_verdicts.jsonl": verdict_sha,
            "positive_slot_graph.json": graph_sha,
            "revision_slot_manifest.json": slots_sha,
            "safety_statistics.json": stats_sha,
        }
    )
    semantic_members = [
        "revision_semantic_source_projection.jsonl",
        "b0r_identity_hints_projection.jsonl",
        "candidate_pair_manifest.jsonl",
        "seed_pairwise_verdicts.jsonl",
        "closure_pair_manifest.jsonl",
        "closure_pairwise_verdicts.jsonl",
        "pairwise_verdicts.jsonl",
        "pairwise_call_ledger.jsonl",
        "positive_slot_graph.json",
        "revision_slot_manifest.json",
        "safety_statistics.json",
    ]
    semantic_sha = {name: _verify_sidecar(RUN_DIR / name) for name in semantic_members}
    freeze = {
        "schema_version": 1,
        "status": "SEMANTIC_DECISIONS_FROZEN",
        "frozen_artifacts": semantic_sha,
        "forbidden_metadata_decode_count": 0,
        "timestamp_fields_opened_before_freeze": 0,
        "timestamp_fields_used_before_freeze": 0,
        "structural_gate_failures": [],
        "candidate_generation_deterministic": True,
        "graph_and_clique_construction_deterministic": True,
    }
    freeze_sha = _freeze(RUN_DIR / "semantic_freeze.json", _json_bytes(freeze))
    manifest.setdefault("frozen_artifacts", {})["semantic_freeze.json"] = freeze_sha
    manifest["status"] = "SEMANTIC_DECISIONS_FROZEN"
    manifest["semantic_freeze_sha256"] = freeze_sha
    manifest["completion"] = {
        "structural_complete": True,
        "completion_marker": "MEM3B0Q_FACTORIZED_REVISION_ADMISSION_COMPLETE=YES",
        "readiness": "PENDING_HUMAN_REFLECTION",
        "readiness_marker": "MEM3B0Q_MEM3B1_READY=NO",
        "semantic_quality_not_part_of_structural_gate": True,
    }
    _write_manifest(manifest)
    return {
        "structural_complete": True,
        "completion_marker": "MEM3B0Q_FACTORIZED_REVISION_ADMISSION_COMPLETE=YES",
        "readiness": "PENDING_HUMAN_REFLECTION",
        "eligible_records": len(eligible),
        "seed_pairs": len(seed_manifest),
        "closure_pairs": len(closure_manifest),
        "positive_edges": graph_stats["positive_singleton_edges"],
        "maximal_cliques": graph_stats["maximal_clique_count"],
        "admitted_slots": len(slots),
    }


def _load_temporal_history_after_freeze() -> dict[str, dict[str, Any]]:
    freeze = _read_frozen_json("semantic_freeze.json")
    if freeze.get("status") != "SEMANTIC_DECISIONS_FROZEN":
        raise IntegrityFailure("semantic_freeze_required_before_timestamp_access")
    for name, expected in freeze.get("frozen_artifacts", {}).items():
        _verify_sidecar(RUN_DIR / name, expected)
    temporal_allowlist = frozenset({"memory_id", "source_session_id", "observed_at", "session_id"})
    result: dict[str, dict[str, Any]] = {}
    with FLATPROP_PATH.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = b0p.jsonl_projection(line, temporal_allowlist)
            memory_id = row.get("memory_id")
            if isinstance(memory_id, str):
                result[memory_id] = row
    return result


def critical_review() -> dict[str, Any]:
    manifest = _read_manifest()
    _assert_code_freeze(manifest)
    if manifest.get("status") != "SEMANTIC_DECISIONS_FROZEN":
        raise IntegrityFailure("critical_review_requires_semantic_freeze")
    _load_temporal_history_after_freeze()
    slots_payload = _read_frozen_json("revision_slot_manifest.json")
    source_by_id = {
        row["memory_id"]: row
        for row in _read_frozen_jsonl("revision_semantic_source_projection.jsonl")
    }
    packet = {
        "schema_version": 1,
        "reviewer_role": "PENDING_USER_HUMAN_REFLECTION",
        "labels": ["true_singleton_slot", "false_revision_merge", "uncertain"],
        "no_timestamps": True,
        "slots": [
            {
                "revision_slot_id": slot["revision_slot_id"],
                "member_memory_ids": slot["member_memory_ids"],
                "propositions": [
                    {
                        "memory_id": memory_id,
                        "proposition_text": source_by_id[memory_id]["proposition_text"],
                    }
                    for memory_id in slot["member_memory_ids"]
                ],
                "decision": "PENDING",
                "notes": "",
            }
            for slot in slots_payload["slots"]
        ],
    }
    packet_sha = _freeze(RUN_DIR / "admitted_slot_human_review_packet.json", _json_bytes(packet))
    decisions_template = {
        "schema_version": 1,
        "packet_sha256": packet_sha,
        "decisions": [
            {
                "revision_slot_id": row["revision_slot_id"],
                "decision": "PENDING",
                "notes": "",
            }
            for row in packet["slots"]
        ],
    }
    decisions_sha = _freeze(
        RUN_DIR / "human_review_decisions_template.json", _json_bytes(decisions_template)
    )
    manifest.setdefault("frozen_artifacts", {})["admitted_slot_human_review_packet.json"] = (
        packet_sha
    )
    manifest["frozen_artifacts"]["human_review_decisions_template.json"] = decisions_sha
    manifest["completion"]["readiness"] = "PENDING_HUMAN_REFLECTION"
    manifest["completion"]["readiness_marker"] = "MEM3B0Q_MEM3B1_READY=NO"
    _write_manifest(manifest)
    return {
        "review_packet_sha256": packet_sha,
        "admitted_slots": len(packet["slots"]),
        "human_decisions_pending": True,
    }


def finalize_human_review(decisions_path: Path) -> dict[str, Any]:
    manifest = _read_manifest()
    _assert_code_freeze(manifest)
    if manifest.get("status") != "SEMANTIC_DECISIONS_FROZEN":
        raise IntegrityFailure("human_review_requires_semantic_freeze")
    packet = _read_frozen_json("admitted_slot_human_review_packet.json")
    decision_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    if not isinstance(decision_payload, dict) or decision_payload.get(
        "packet_sha256"
    ) != _verify_sidecar(RUN_DIR / "admitted_slot_human_review_packet.json"):
        raise IntegrityFailure("human_review_packet_identity_mismatch")
    decisions_rows = decision_payload.get("decisions", [])
    expected_slots = {row["revision_slot_id"] for row in packet.get("slots", [])}
    if {row.get("revision_slot_id") for row in decisions_rows} != expected_slots:
        raise IntegrityFailure("human_review_slot_coverage_mismatch")
    if any(row.get("decision") == "PENDING" for row in decisions_rows):
        raise IntegrityFailure("human_reflection_incomplete")
    decisions = [row.get("decision") for row in decisions_rows]
    if any(
        decision not in {"true_singleton_slot", "false_revision_merge", "uncertain"}
        for decision in decisions
    ):
        raise IntegrityFailure("illegal_human_reflection_label")
    reviewed_packet = {
        **packet,
        "reviewer_role": "USER_HUMAN_REFLECTION",
        "decisions": decisions_rows,
    }
    critical = _critical_controls_after_freeze(reviewed_packet)
    ready = (
        critical.get("instagram_revision_slot_admitted") == "YES"
        and critical.get("b0p_false_purchase_event_blocked") is True
        and critical.get("historical_b0s_false_safe_examples_blocked") is True
        and all(decision != "false_revision_merge" for decision in decisions)
    )
    review_summary = {
        label: decisions.count(label)
        for label in ("true_singleton_slot", "false_revision_merge", "uncertain")
    }
    review_sha = _freeze(
        RUN_DIR / "admitted_slot_human_review.json",
        _json_bytes({**reviewed_packet, "review_summary": review_summary}),
    )
    critical_sha = _freeze(RUN_DIR / "critical_case_review.json", _json_bytes(critical))
    report = _render_report(manifest, critical, review_summary, ready)
    report_sha = _freeze(RUN_DIR / "report.md", report.encode("utf-8"))
    manifest.setdefault("frozen_artifacts", {}).update(
        {
            "admitted_slot_human_review.json": review_sha,
            "critical_case_review.json": critical_sha,
            "report.md": report_sha,
        }
    )
    manifest["completion"]["readiness"] = "YES" if ready else "NO"
    manifest["completion"]["readiness_marker"] = f"MEM3B0Q_MEM3B1_READY={'YES' if ready else 'NO'}"
    manifest["completion"]["human_review_summary"] = review_summary
    manifest["completion"]["critical_controls"] = critical
    _write_manifest(manifest)
    return {
        "structural_complete": True,
        "completion_marker": "MEM3B0Q_FACTORIZED_REVISION_ADMISSION_COMPLETE=YES",
        "readiness": "YES" if ready else "NO",
        "review_summary": review_summary,
        "critical_controls": critical,
    }


def _critical_controls_after_freeze(packet: dict[str, Any]) -> dict[str, Any]:
    # Control labels are resolved only after the semantic freeze; this function never mutates slots.
    b0p_dir = ROOT / "runs" / "memory" / "mem3" / "mem3b0p-pairwise-admission-20260930"
    historical: dict[str, Any] = {}
    old_critical_path = b0p_dir / "critical_case_review.json"
    if old_critical_path.exists():
        old_critical = json.loads(old_critical_path.read_text(encoding="utf-8"))
        historical = old_critical if isinstance(old_critical, dict) else {}
    slot_member_sets = [
        set(row["member_memory_ids"])
        for row in _read_frozen_json("revision_slot_manifest.json")["slots"]
    ]
    all_slots = set().union(*slot_member_sets) if slot_member_sets else set()
    instagram_ids = _collect_control_memory_ids(historical, ("instagram", "500", "600"))
    purchase_ids = _collect_control_memory_ids(historical, ("purchase", "necklace"))
    instagram_admitted = any(
        instagram_ids and instagram_ids.issubset(slot) for slot in slot_member_sets
    )
    purchase_blocked = not any(
        purchase_ids and purchase_ids.issubset(slot) for slot in slot_member_sets
    )
    historical_false_safe = historical.get("all_historical_b0s_false_safe_examples_blocked") is True
    return {
        "instagram_revision_slot_admitted": "YES" if instagram_admitted else "NO",
        "instagram_control_memory_ids": sorted(instagram_ids),
        "b0p_false_purchase_event_blocked": purchase_blocked,
        "b0p_purchase_control_memory_ids": sorted(purchase_ids),
        "historical_b0s_false_safe_examples_blocked": historical_false_safe
        and not any(
            set(row.get("memory_ids", [])) & all_slots
            for row in historical.get("historical_b0s_false_safe_examples", [])
        ),
        "gym_candidate_recall": "UNRESOLVED_CONTROL_SOURCE_REVIEW_REQUIRED",
        "human_review_slots": len(packet.get("slots", [])),
        "control_source": "historical B0P critical-case artifact inspected after B0Q semantic freeze",
    }


def _collect_control_memory_ids(value: Any, terms: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False).casefold()
        if all(term in text for term in terms):
            for key in ("memory_id", "memory_ids", "member_memory_ids"):
                item = value.get(key)
                if isinstance(item, str):
                    found.add(item)
                elif isinstance(item, list):
                    found.update(candidate for candidate in item if isinstance(candidate, str))
        for child in value.values():
            found.update(_collect_control_memory_ids(child, terms))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_control_memory_ids(child, terms))
    return found


def _render_report(
    manifest: dict[str, Any], critical: dict[str, Any], review_summary: dict[str, int], ready: bool
) -> str:
    stats = _read_frozen_json("safety_statistics.json")
    lines = [
        "# MEM-3B0Q Factorized Revision Admission Closeout",
        "",
        "This is a Harness safety/admission experiment, not a benchmark or performance claim. B0P's failed structural marker remains unchanged.",
        "",
        "## Candidate Graph",
        "",
        f"- Eligible records: {stats.get('eligible_records')}; exact-hint seed pairs: {stats.get('exact_hint_seed_pairs')}; semantic-neighbor seed pairs: {stats.get('semantic_neighbor_seed_pairs')} ({stats.get('semantic_neighbor_retrievals_directed')} directed top-k hits); union pairs: {stats.get('union_seed_pairs')}.",
        f"- Seed/closure calls: {stats.get('seed_pair_count')} / {stats.get('closure_pair_count')}; positive edges: {stats.get('positive_singleton_edges')}; positive components >16 blocked: {stats.get('oversized_component_count')}.",
        f"- Maximal cliques: {stats.get('maximal_clique_count')}; overlapping cliques blocked: {stats.get('overlapping_ambiguous_clique_count')}; admitted slots: {stats.get('admitted_slot_count')}.",
        f"- Harness UNKNOWN fallback pairs: {stats.get('harness_unknown_pair_count')} / {stats.get('seed_pair_count', 0) + stats.get('closure_pair_count', 0)} ({stats.get('harness_unknown_pair_rate')}).",
        "",
        "## Controls And Review",
        "",
        f"- Instagram positive control admitted: `{critical.get('instagram_revision_slot_admitted')}`.",
        f"- B0P completed-purchase event blocked: `{critical.get('b0p_false_purchase_event_blocked')}`.",
        f"- Historical B0S false-safe examples blocked: `{critical.get('historical_b0s_false_safe_examples_blocked')}`.",
        f"- Gym: `{critical.get('gym_candidate_recall')}`.",
        f"- Frozen-ten human reflection: `{review_summary}`.",
        "",
        "## Structural Safety",
        "",
        f"- Timestamp metadata decoded before semantic freeze: `{stats.get('forbidden_metadata_decode_count')}`; hosted calls: `{stats.get('hosted_calls')}`; same-request retries: `{stats.get('same_request_retries')}`.",
        f"- MemoryStore mutations ADD/UPDATE/DELETE/SUPERSEDED: `{stats.get('memory_store_mutations')}`.",
        f"- DEV opened: `{stats.get('dev_opened')}`; TEST opened: `{stats.get('test_opened')}`; MedMemoryBench run: `{stats.get('medmemorybench_run')}`.",
        "",
        "## Outcome",
        "",
        "`MEM3B0Q_FACTORIZED_REVISION_ADMISSION_COMPLETE=YES`.",
        f"`MEM3B0Q_MEM3B1_READY={'YES' if ready else 'NO'}`.",
        "",
        "No LongMemEval or MedMemoryBench performance claim is made from this stage.",
    ]
    return "\n".join(lines) + "\n"


def validate() -> dict[str, Any]:
    manifest = _read_manifest()
    sidecars = sorted(RUN_DIR.glob("*.sha256"))
    for sidecar in sidecars:
        parts = sidecar.read_text(encoding="utf-8").strip().split()
        if len(parts) != 2:
            raise IntegrityFailure(f"invalid_sidecar:{sidecar.name}")
        _verify_sidecar(RUN_DIR / parts[1], parts[0])
    for name, expected in manifest.get("frozen_artifacts", {}).items():
        _verify_sidecar(RUN_DIR / name, expected)
    marker = manifest.get("completion", {}).get("completion_marker")
    if marker == "MEM3B0Q_FACTORIZED_REVISION_ADMISSION_COMPLETE=YES":
        required = {
            "revision_semantic_source_projection.jsonl",
            "candidate_pair_manifest.jsonl",
            "pairwise_call_ledger.jsonl",
            "pairwise_verdicts.jsonl",
            "closure_pair_manifest.jsonl",
            "positive_slot_graph.json",
            "revision_slot_manifest.json",
            "safety_statistics.json",
            "semantic_freeze.json",
        }
        missing = sorted(name for name in required if not (RUN_DIR / name).is_file())
        if missing:
            raise IntegrityFailure(f"required_artifacts_missing:{missing}")
    return {
        "validated_sidecars": len(sidecars),
        "frozen_artifacts": len(manifest.get("frozen_artifacts", {})),
        "status": manifest.get("status"),
        "completion_marker": marker,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "candidates",
            "verify-seeds",
            "prepare-closure",
            "verify-closure",
            "closeout",
            "critical-review",
            "finalize-human-review",
            "validate",
        ),
    )
    parser.add_argument("--review-decisions", type=Path)
    args = parser.parse_args()
    commands = {
        "prepare": prepare,
        "candidates": generate_candidates,
        "verify-seeds": verify_seeds,
        "prepare-closure": prepare_closure,
        "verify-closure": verify_closure,
        "closeout": closeout,
        "critical-review": critical_review,
        "validate": validate,
    }
    if args.command == "finalize-human-review":
        if args.review_decisions is None:
            parser.error("finalize-human-review requires --review-decisions")
        result = finalize_human_review(args.review_decisions)
    else:
        result = commands[args.command]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
