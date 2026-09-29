"""Run MEM-3A.3 with a frozen, deterministic RawSpan chunk plan."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
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
_JIEBA_SITE = Path(r"D:\Anaconda\envs\yolov10\Lib\site-packages")
_JIEBA_PACKAGE = _JIEBA_SITE / "jieba"
if "jieba" not in sys.modules and (_JIEBA_PACKAGE / "__init__.py").is_file():
    _jieba_spec = importlib.util.spec_from_file_location(
        "jieba",
        _JIEBA_PACKAGE / "__init__.py",
        submodule_search_locations=[str(_JIEBA_PACKAGE)],
    )
    if _jieba_spec is None or _jieba_spec.loader is None:
        raise RuntimeError("Could not load the frozen local jieba package")
    _jieba_module = importlib.util.module_from_spec(_jieba_spec)
    sys.modules["jieba"] = _jieba_module
    _jieba_spec.loader.exec_module(_jieba_module)

from tools.research.memory import flatprop_writer_v4_chunked as writer_v4  # noqa: E402
from tools.research.memory import run_mem2b_rank_aware_projection as mem2b  # noqa: E402
from tools.research.memory import run_mem2d_semantic_retrieval as mem2d  # noqa: E402
from tools.research.memory import run_mem3a2_minimal_flatprop as base  # noqa: E402
from tools.research.memory.flatprop_chunking import (  # noqa: E402
    aggregate_chunk_propositions,
    plan_session_chunks,
)
from tools.research.memory.flat_proposition_writer_v2 import (  # noqa: E402
    WriterQualificationFailure,
    WriterRecoveryError,
    atomic_write_bytes,
    canonical_json,
    execute_or_resume,
    sha256_bytes,
)

flat = base.flat
RUN_ID = "mem3a3-chunked-flatprop-frozen-10-20260929"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
SOURCE_ROOT = ROOT.parent / "Health-Copilot"
SOURCE_DATASET_PATH = SOURCE_ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
CHUNKING_CONTRACT_PATH = ROOT / "docs" / "research" / "memory" / "flatprop_chunking_v1.json"
STAGE_PROTOCOL_PATH = ROOT / "docs" / "research" / "memory" / "mem_3a3_chunked_flatprop.md"
NORMALIZATION_PATH = (
    ROOT / "docs" / "research" / "memory" / "evidence_ref_set_normalization_v1.json"
)
NORMALIZATION_SHA256 = "d718ae5895976e4736fd4aed63019d656cf60b863a1632c4b15631fc55b5862e"
V3_PROMPT_SHA256 = "0c0211bac51cf48e8e01663bbee7d9f9db69815afc7d40c1e8d6877f4a0f2376"
V3_CONTRACT_SHA256 = "708da9fce6990c3b410f1bc610009e76dc39a993a786f59bfeac9f5798fd1545"
MEM3A2S_HEAD = "ba5456c7233427c57c14d1efa0cb9c04bac00f18"
MAX_PROMPT_TOKENS = 6144
MAX_COMPLETION_TOKENS = 8192
HIGH_SIMILARITY_THRESHOLD = 0.90
CONTEXT_LIMIT = 131072
OUTPUT_RESERVE = 8192
SCHEMA_PREFIX = "\n\n<FLATPROP_DYNAMIC_SCHEMA>\n"
SCHEMA_SUFFIX = "\n</FLATPROP_DYNAMIC_SCHEMA>"
RISK_SESSION_ID = "de43030f_1"
GATE = "MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC"
LOCAL_CACHE_ROOT = ROOT / ".cache" / "health-copilot" / "mem3a3-chunked-v4-writer"


def _sha_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _json(path: Path) -> Any:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _verify(path: Path) -> bool:
    marker = _sidecar(path)
    if not path.is_file() or not marker.is_file():
        return False
    return marker.read_text(encoding="ascii").strip().split() == [_sha_file(path), path.name]


def _freeze(path: Path) -> str:
    digest = _sha_file(path)
    atomic_write_bytes(_sidecar(path), f"{digest}  {path.name}\n".encode("ascii"))
    return digest


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    atomic_write_bytes(path, payload)
    return _freeze(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    atomic_write_bytes(path, b"".join(canonical_json(row) + b"\n" for row in rows))
    return _freeze(path)


def _write_or_verify(path: Path, payload: bytes) -> str:
    if path.exists():
        if not _verify(path) or path.read_bytes() != payload:
            raise RuntimeError(f"Frozen MEM-3A.3 artifact differs: {path.name}")
        return _sha_file(path)
    atomic_write_bytes(path, payload)
    return _freeze(path)


def _configure_paths() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    base.RUN_ID = RUN_ID
    base.RUN_DIR = RUN_DIR
    base.BASE_COMMIT = head
    base.PROMPT_PATH = ROOT / "docs" / "research" / "memory" / "flat_proposition_extractor_v3.txt"
    base.CONTRACT_PATH = (
        ROOT / "docs" / "research" / "memory" / "flat_proposition_extractor_v3_contract.json"
    )
    base.LOCAL_CACHE_ROOT = LOCAL_CACHE_ROOT
    flat.DATASET_PATH = SOURCE_DATASET_PATH
    flat.mem2c.DATASET_PATH = SOURCE_DATASET_PATH
    base._configure_flat_paths()
    flat.RUN_ID = RUN_ID
    flat.RUN_DIR = RUN_DIR
    flat.PINNED_BASE_COMMIT = head
    flat.PREFLIGHT_PATH = RUN_DIR / "chunking_preflight.json"
    flat.EXTRACTION_MANIFEST_PATH = RUN_DIR / "session_extraction_manifest.json"
    flat.EXTRACTIONS_PATH = RUN_DIR / "session_extractions.jsonl"
    flat.FLAT_PROPOSITIONS_PATH = RUN_DIR / "flatprop_inventory.jsonl"
    flat.WRITER_LEDGER_PATH = RUN_DIR / "chunk_writer_ledger.jsonl"
    flat.MATERIALIZATION_LEDGER_PATH = RUN_DIR / "materialization_ledger.jsonl"
    flat.EMBEDDING_MANIFEST_PATH = RUN_DIR / "embedding_manifest.json"
    flat.DENSE_TOP8_PATH = RUN_DIR / "dense_top8.jsonl"
    flat.CONTEXT_PLANS_PATH = RUN_DIR / "context_plans.jsonl"
    flat.CONTEXT_BUNDLES_PATH = RUN_DIR / "context_bundles.jsonl"
    flat.PREDICTIONS_PATH = RUN_DIR / "predictions.jsonl"
    flat.READER_LEDGER_PATH = RUN_DIR / "reader_call_ledger.jsonl"
    flat.METRICS_PATH = RUN_DIR / "deterministic_metrics.json"
    flat.EFFICIENCY_PATH = RUN_DIR / "efficiency.json"
    flat.COMPARISON_PATH = RUN_DIR / "comparison_mem2d_rawspan_vs_mem3a3.json"
    flat.CASE_REVIEW_PATH = RUN_DIR / "mem_3a3_case_review.json"
    flat.PROTOCOL_PATH = RUN_DIR / "mem_3a3_protocol.md"
    flat.CANDIDATE_GROUPS_PATH = RUN_DIR / "revision_semantics_not_in_scope.json"
    return head


def _verify_upstream(head: str) -> dict[str, Any]:
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", MEM3A2S_HEAD, head], cwd=ROOT, capture_output=True
    )
    if ancestor.returncode != 0:
        raise RuntimeError("MEM-3A.2S lineage commit is not an ancestor of the stage checkout")
    parent_manifest_path = (
        ROOT
        / "runs"
        / "memory"
        / "mem3"
        / "mem3a2s-flatprop-setrefs-frozen-10-20260929-final"
        / "run_manifest.json"
    )
    if not flat._verify_frozen(parent_manifest_path):
        raise RuntimeError("Historical MEM-3A.2S run manifest sidecar is invalid")
    parent_manifest = _json(parent_manifest_path)
    if parent_manifest.get("completion_gate_marker") != (
        "MEM3A2S_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO"
    ):
        raise RuntimeError("MEM-3A.2S historical gate record changed")
    upstream = flat._verify_upstream()
    historical = base._verify_historical_gates()
    memory_audit = (
        ROOT / "docs" / "research" / "memory" / "medmemorybench_compatibility_audit.md"
    ).read_text(encoding="utf-8")
    if "MEDMEMORYBENCH_COMPATIBILITY_AUDITED=YES" not in memory_audit or (
        "LONGMEM_MEMORY_MECHANISM_FROZEN=NO" not in memory_audit
    ):
        raise RuntimeError("Historical benchmark-routing gates changed")
    return {
        **upstream,
        "historical_gates": historical,
        "mem3a2s_head_sha256": MEM3A2S_HEAD,
        "mem3a2s_run_manifest_sha256": _sha_file(parent_manifest_path),
        "convergence_head_sha256": head,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
        "medmemorybench_runs": 0,
    }


def _runtime(client: httpx.Client) -> dict[str, Any]:
    manifest = _json(flat.mem2c.MEM2A_DIR / "run_manifest.json")
    runtime = mem2b._verify_reader(client, manifest)
    if (
        runtime.get("endpoint") != flat.READER_ENDPOINT
        or runtime.get("loopback_only") is not True
        or runtime.get("model_sha256") != flat.PINNED_WRITER_MODEL_SHA256
        or runtime.get("server_build") != "llama.cpp 10068 (571d0d540)"
        or runtime.get("context_tokens") != CONTEXT_LIMIT
        or urlsplit(runtime["endpoint"]).hostname not in {"127.0.0.1", "localhost", "::1"}
    ):
        raise RuntimeError("Frozen local Qwen3-8B llama.cpp runtime identity failed")
    return runtime


def _tokenize(client: httpx.Client, text: str) -> int:
    response = client.post(
        "http://127.0.0.1:8081/tokenize",
        json={"content": text, "add_special": False},
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise TypeError("Frozen llama.cpp /tokenize returned no token IDs")
    return len(tokens)


def _measure_request(
    *,
    client: httpx.Client,
    session: dict[str, Any],
    catalog: list[dict[str, Any]],
    prompt: str,
    schema_prefix: str,
    schema_suffix: str,
    detailed: bool,
) -> dict[str, Any]:
    request = writer_v4.writer_request(
        session_date=session["session_date"],
        catalog=catalog,
        system_prompt=prompt,
        model_alias=flat.READER_MODEL,
    )
    user_payload = request["messages"][1]["content"]
    schema = request["response_format"]["json_schema"]["schema"]
    schema_text = canonical_json(schema).decode("utf-8")
    rendered = client.post(
        "http://127.0.0.1:8081/apply-template",
        json={
            "messages": request["messages"],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    rendered.raise_for_status()
    rendered_prompt = rendered.json().get("prompt")
    if not isinstance(rendered_prompt, str):
        raise TypeError("llama.cpp /apply-template returned no rendered writer prompt")
    budget_view = f"{rendered_prompt}{schema_prefix}{schema_text}{schema_suffix}"
    row = {
        "request_budget_tokens": _tokenize(client, budget_view),
        "rendered_chat_prompt_sha256": sha256_bytes(rendered_prompt.encode("utf-8")),
        "request_budget_view_sha256": sha256_bytes(budget_view.encode("utf-8")),
        "rendered_source_payload_sha256": sha256_bytes(user_payload.encode("utf-8")),
        "dynamic_schema_sha256": writer_v4.schema_sha256(catalog),
        "request_sha256": sha256_bytes(canonical_json(request)),
    }
    if detailed:
        row.update(
            {
                "rendered_chat_prompt_tokens": _tokenize(client, rendered_prompt),
                "dynamic_schema_tokens": _tokenize(client, schema_text),
            }
        )
    return row


def _percentiles(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "p50": float(flat.np_percentile(values, 0.50)),
        "p90": float(flat.np_percentile(values, 0.90)),
        "p95": float(flat.np_percentile(values, 0.95)),
        "p99": float(flat.np_percentile(values, 0.99)),
        "max": max(values),
    }


def _build_preflight(head: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    upstream = _verify_upstream(head)
    contract = _json(CHUNKING_CONTRACT_PATH)
    if (
        contract.get("method_identity") != writer_v4.CONTRACT_ID
        or contract.get("max_rendered_chunk_prompt_tokens") != MAX_PROMPT_TOKENS
        or contract.get("chunk_max_completion_tokens") != MAX_COMPLETION_TOKENS
        or contract.get("turn_overlap") != 1
        or contract.get("tokenizer_budget_view", {}).get("schema_prefix") != SCHEMA_PREFIX
        or contract.get("tokenizer_budget_view", {}).get("schema_suffix") != SCHEMA_SUFFIX
        or contract.get("duplicate_diagnostics", {}).get("high_similarity_threshold")
        != HIGH_SIMILARITY_THRESHOLD
    ):
        raise RuntimeError("Chunking contract differs from the runner's frozen constants")
    prompt_sha = _sha_file(base.PROMPT_PATH)
    v3_contract_sha = _sha_file(base.CONTRACT_PATH)
    if prompt_sha != V3_PROMPT_SHA256 or v3_contract_sha != V3_CONTRACT_SHA256:
        raise RuntimeError("Frozen v3 prompt/contract changed")
    if (
        not flat._verify_frozen(NORMALIZATION_PATH)
        or _sha_file(NORMALIZATION_PATH) != NORMALIZATION_SHA256
    ):
        raise RuntimeError("Evidence-ref set-normalization contract is not frozen")
    embedding_identity = flat.verify_local_model(
        Path(r"E:\Health-Copilot-Models\models\Qwen3-Embedding-0.6B")
    )
    if (
        embedding_identity.get("revision") != flat.MODEL_REVISION
        or embedding_identity.get("model_tree_sha256") != flat.MODEL_TREE_SHA256
        or embedding_identity.get("weights_sha256") != flat.WEIGHTS_SHA256
        or flat.MODEL_ID != "Qwen/Qwen3-Embedding-0.6B"
        or flat.DEVICE != "cuda:0"
        or flat.DTYPE != "float16"
    ):
        raise RuntimeError("Frozen local Qwen3-Embedding-0.6B CUDA FP16 identity failed")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA is unavailable for the frozen embedding adapter")
    chunking_contract_sha = _sha_file(CHUNKING_CONTRACT_PATH)
    method_identity_body = {
        "contract_id": writer_v4.CONTRACT_ID,
        "chunking_contract_sha256": chunking_contract_sha,
        "v3_extractor_prompt_sha256": prompt_sha,
        "v3_extractor_contract_sha256": v3_contract_sha,
        "evidence_ref_set_normalization_sha256": NORMALIZATION_SHA256,
        "semantic_fields": ["proposition_text", "evidence_refs"],
        "writer_max_completion_tokens": MAX_COMPLETION_TOKENS,
        "embedding_model_identity": embedding_identity,
        "embedding_device": flat.DEVICE,
        "embedding_dtype": flat.DTYPE,
        "writer_request_source_sha256": _sha_file(Path(writer_v4.__file__)),
        "chunk_planner_source_sha256": _sha_file(TOOLS_DIR / "flatprop_chunking.py"),
        "packet_validator_source_sha256": _sha_file(Path(base.writer_v3.__file__)),
        "transport_unwrap_source_sha256": _sha_file(
            Path(base.__file__).with_name("flat_proposition_writer_v2.py")
        ),
        "runner_source_sha256": _sha_file(Path(__file__).resolve()),
        "flatprop_runtime_source_sha256": _sha_file(Path(flat.__file__)),
        "embedding_adapter_source_sha256": _sha_file(TOOLS_DIR / "local_qwen3_embedding.py"),
    }
    method_contract_sha = sha256_bytes(canonical_json(method_identity_body))
    method_path = RUN_DIR / "writer_method_identity.json"
    _write_or_verify(
        method_path,
        json.dumps(method_identity_body, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        + b"\n",
    )
    protocol_copy = RUN_DIR / "mem_3a3_protocol.md"
    _write_or_verify(protocol_copy, STAGE_PROTOCOL_PATH.read_bytes())
    normalization_copy = RUN_DIR / "evidence_ref_set_normalization_v1.json"
    _write_or_verify(normalization_copy, NORMALIZATION_PATH.read_bytes())

    sessions, refs, inventory = base._source_sessions_with_catalog()
    if len(sessions) != 477 or len(inventory) != 52703:
        raise RuntimeError("Frozen source corpus differs from 477 sessions / 52,703 RawSpans")
    risk_matches = [
        identity
        for identity, session in sessions.items()
        if RISK_SESSION_ID in session["source_session_ids"]
    ]
    if len(risk_matches) != 1:
        raise RuntimeError("Risk-first session de43030f_1 did not resolve uniquely")

    prompt = base.PROMPT_PATH.read_text(encoding="utf-8")
    schema_cfg = contract["tokenizer_budget_view"]
    preflight_started = time.perf_counter()
    chunk_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        runtime = _runtime(client)
        for session_identity, session in sessions.items():
            measure_cache: dict[tuple[str, ...], dict[str, Any]] = {}

            def measure(spans: list[dict[str, Any]]) -> dict[str, Any]:
                key = tuple(span["evidence_ref"] for span in spans)
                if key not in measure_cache:
                    measure_cache[key] = _measure_request(
                        client=client,
                        session=session,
                        catalog=spans,
                        prompt=prompt,
                        schema_prefix=schema_cfg["schema_prefix"],
                        schema_suffix=schema_cfg["schema_suffix"],
                        detailed=False,
                    )
                return measure_cache[key]

            plan = plan_session_chunks(
                session_identity_sha256=session_identity,
                catalog=session["catalog"],
                chunking_contract_sha256=chunking_contract_sha,
                max_prompt_tokens=MAX_PROMPT_TOKENS,
                measure_spans=measure,
            )
            session_chunk_ids = []
            for chunk in plan["chunks"]:
                visible_catalog = [session["catalog"][i] for i in chunk["visible_span_ordinals"]]
                detail = _measure_request(
                    client=client,
                    session=session,
                    catalog=visible_catalog,
                    prompt=prompt,
                    schema_prefix=schema_cfg["schema_prefix"],
                    schema_suffix=schema_cfg["schema_suffix"],
                    detailed=True,
                )
                if detail["request_budget_tokens"] != chunk["request_budget_tokens"]:
                    raise RuntimeError(
                        f"Nondeterministic token budget on chunk {chunk['chunk_id']}"
                    )
                if detail["request_budget_tokens"] > MAX_PROMPT_TOKENS:
                    raise RuntimeError(
                        f"Rendered writer request exceeds 6,144 tokens: {chunk['chunk_id']}"
                    )
                row = {
                    **chunk,
                    **detail,
                    "session_identity_sha256": session_identity,
                    "source_session_ids": session["source_session_ids"],
                    "session_date": session["session_date"],
                    "valid_from": session["valid_from"],
                    "source_turns_sha256": session["source_turns_sha256"],
                    "catalog_sha256": session["catalog_sha256"],
                }
                session_chunk_ids.append(row["chunk_id"])
                chunk_rows.append(row)
            session_rows.append(
                {
                    "session_identity_sha256": session_identity,
                    "source_session_ids": session["source_session_ids"],
                    "session_date": session["session_date"],
                    "source_turns_sha256": session["source_turns_sha256"],
                    "catalog_sha256": session["catalog_sha256"],
                    "source_turn_count": plan["source_turn_count"],
                    "source_span_count": plan["source_span_count"],
                    "chunk_count": plan["chunk_count"],
                    "oversized_turn_split_count": plan["oversized_turn_split_count"],
                    "primary_span_coverage_exactly_once": plan[
                        "primary_span_coverage_exactly_once"
                    ],
                    "primary_turn_coverage_exactly_once_or_oversized_span_partition": plan[
                        "primary_turn_coverage_exactly_once_or_oversized_span_partition"
                    ],
                    "chunk_ids": session_chunk_ids,
                }
            )

    chunk_count_by_session = [row["chunk_count"] for row in session_rows]
    prompt_budgets = [row["request_budget_tokens"] for row in chunk_rows]
    all_primary = defaultdict(list)
    all_overlaps = defaultdict(list)
    for row in chunk_rows:
        all_primary[row["session_identity_sha256"]].extend(row["primary_span_ids"])
        all_overlaps[row["session_identity_sha256"]].extend(row["overlap_span_ids"])
    coverage_valid = all(
        sorted(all_primary[identity]) == sorted(span["evidence_ref"] for span in session["catalog"])
        and len(all_primary[identity]) == len(set(all_primary[identity]))
        for identity, session in sessions.items()
    )
    if (
        len(session_rows) != 477
        or len({row["session_identity_sha256"] for row in session_rows}) != 477
        or not coverage_valid
        or not all(
            row["primary_turn_coverage_exactly_once_or_oversized_span_partition"]
            for row in session_rows
        )
        or any(value > MAX_PROMPT_TOKENS for value in prompt_budgets)
        or sum(len(row["primary_span_ids"]) for row in chunk_rows) != len(inventory)
    ):
        raise RuntimeError("Full-corpus deterministic chunk preflight failed coverage/ceiling")
    manifest_path = RUN_DIR / "chunk_manifest.jsonl"
    preflight_path = RUN_DIR / "chunking_preflight.json"
    chunk_sha = _write_jsonl(manifest_path, chunk_rows)
    preflight = {
        "schema_version": 1,
        "stage": RUN_ID,
        "stage_base_commit_sha": head,
        "method_contract_sha256": method_contract_sha,
        "chunking_contract_sha256": chunking_contract_sha,
        "v3_extractor_prompt_sha256": prompt_sha,
        "v3_extractor_contract_sha256": v3_contract_sha,
        "normalization_contract_sha256": NORMALIZATION_SHA256,
        "source_dataset_sha256": _sha_file(SOURCE_DATASET_PATH),
        "source_inventory_sha256": _sha_file(flat.MEM2C_INVENTORY),
        "retrieval_view_contract_sha256": _sha_file(flat.RETRIEVAL_CONTRACT_PATH),
        "unique_source_sessions": len(session_rows),
        "source_rawspan_count": len(inventory),
        "source_turn_count": sum(row["source_turn_count"] for row in session_rows),
        "total_chunks": len(chunk_rows),
        "chunks_per_session": _percentiles(chunk_count_by_session),
        "request_budget_prompt_tokens_per_chunk": _percentiles(prompt_budgets),
        "primary_span_coverage_exactly_once": coverage_valid,
        "primary_turn_coverage_exactly_once": all(
            row["primary_turn_coverage_exactly_once_or_oversized_span_partition"]
            for row in session_rows
        ),
        "primary_span_count": sum(len(row["primary_span_ids"]) for row in chunk_rows),
        "overlap_span_emissions": sum(len(items) for items in all_overlaps.values()),
        "oversized_turn_split_count": sum(
            row["oversized_turn_split_count"] for row in session_rows
        ),
        "risk_first_session_identity_sha256": risk_matches[0],
        "risk_first_session_chunk_count": next(
            row["chunk_count"]
            for row in session_rows
            if row["session_identity_sha256"] == risk_matches[0]
        ),
        "rendered_request_budget_ceiling": MAX_PROMPT_TOKENS,
        "all_rendered_request_budgets_fit": all(
            value <= MAX_PROMPT_TOKENS for value in prompt_budgets
        ),
        "chunk_max_completion_tokens": MAX_COMPLETION_TOKENS,
        "tokenizer": "frozen llama.cpp /apply-template + /tokenize; add_special=false",
        "rendered_chat_prompt_tokens_and_dynamic_schema_tokens_recorded_separately": all(
            isinstance(row.get("rendered_chat_prompt_tokens"), int)
            and isinstance(row.get("dynamic_schema_tokens"), int)
            for row in chunk_rows
        ),
        "chunk_manifest_sha256": chunk_sha,
        "writer_runtime_identity": runtime,
        "upstream": upstream,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
        "medmemorybench_runs": 0,
        "preflight_wall_seconds": round(time.perf_counter() - preflight_started, 3),
        "session_rows": session_rows,
        "method_identity": method_identity_body,
    }
    _write_json(preflight_path, preflight)
    _write_json(
        RUN_DIR / "writer_extraction_identity.json",
        method_identity_body | {"identity_sha256": method_contract_sha},
    )
    return (
        preflight,
        chunk_rows,
        {"upstream": upstream, "sessions": sessions, "refs": refs, "inventory": inventory},
    )


def _load_or_build_preflight(
    head: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    preflight_path = RUN_DIR / "chunking_preflight.json"
    manifest_path = RUN_DIR / "chunk_manifest.jsonl"
    if not preflight_path.exists() and not manifest_path.exists():
        return _build_preflight(head)
    if not _verify(preflight_path) or not _verify(manifest_path):
        raise RuntimeError("Existing chunk plan artifacts are incomplete or SHA-invalid")
    preflight = _json(preflight_path)
    if (
        preflight.get("stage_base_commit_sha") != head
        or preflight.get("chunking_contract_sha256") != _sha_file(CHUNKING_CONTRACT_PATH)
        or preflight.get("source_dataset_sha256") != _sha_file(SOURCE_DATASET_PATH)
        or preflight.get("source_inventory_sha256") != _sha_file(flat.MEM2C_INVENTORY)
    ):
        raise RuntimeError("Frozen chunk plan identity differs from live source or method")
    method_identity = preflight.get("method_identity")
    if (
        not isinstance(method_identity, dict)
        or method_identity.get("runner_source_sha256") != _sha_file(Path(__file__).resolve())
        or method_identity.get("chunk_planner_source_sha256")
        != _sha_file(TOOLS_DIR / "flatprop_chunking.py")
        or sha256_bytes(canonical_json(method_identity)) != preflight.get("method_contract_sha256")
    ):
        raise RuntimeError("Frozen chunk method identity differs from live implementation")
    frozen_stage_sources = (
        RUN_DIR / "writer_method_identity.json",
        RUN_DIR / "writer_extraction_identity.json",
        RUN_DIR / "mem_3a3_protocol.md",
        RUN_DIR / "evidence_ref_set_normalization_v1.json",
    )
    if not all(_verify(path) for path in frozen_stage_sources):
        raise RuntimeError("Frozen MEM-3A.3 source/contract copies are missing or SHA-invalid")
    if (
        _json(frozen_stage_sources[0]) != method_identity
        or _json(frozen_stage_sources[1])
        != method_identity | {"identity_sha256": preflight["method_contract_sha256"]}
        or frozen_stage_sources[2].read_bytes() != STAGE_PROTOCOL_PATH.read_bytes()
        or frozen_stage_sources[3].read_bytes() != NORMALIZATION_PATH.read_bytes()
    ):
        raise RuntimeError("Frozen method identity copy differs from chunk preflight")
    chunk_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(chunk_rows) != preflight.get("total_chunks"):
        raise RuntimeError("Frozen chunk manifest row count differs from preflight")
    if (
        preflight.get("primary_span_coverage_exactly_once") is not True
        or preflight.get("primary_turn_coverage_exactly_once") is not True
        or not all(
            row.get("primary_turn_coverage_exactly_once_or_oversized_span_partition") is True
            for row in preflight.get("session_rows", [])
        )
    ):
        raise RuntimeError("Frozen chunk preflight does not prove full primary source coverage")
    upstream = _verify_upstream(head)
    sessions, refs, inventory = base._source_sessions_with_catalog()
    if len(sessions) != 477 or len(inventory) != 52703:
        raise RuntimeError("Frozen live source corpus changed after chunk planning")
    primary_refs: dict[str, list[str]] = defaultdict(list)
    for chunk in chunk_rows:
        identity = chunk.get("session_identity_sha256")
        if (
            identity not in sessions
            or chunk.get("request_budget_tokens", MAX_PROMPT_TOKENS + 1) > MAX_PROMPT_TOKENS
        ):
            raise RuntimeError("Frozen chunk references an unknown session or exceeds the budget")
        primary_refs[identity].extend(chunk.get("primary_span_ids", []))
        expected_primary = [
            sessions[identity]["catalog"][ordinal]["evidence_ref"]
            for ordinal in chunk.get("primary_span_ordinals", [])
        ]
        if chunk.get("primary_span_ids") != expected_primary:
            raise RuntimeError(f"Frozen primary span mapping changed: {chunk.get('chunk_id')}")
    if set(primary_refs) != set(sessions) or any(
        sorted(primary_refs[identity])
        != sorted(span["evidence_ref"] for span in session["catalog"])
        or len(primary_refs[identity]) != len(set(primary_refs[identity]))
        for identity, session in sessions.items()
    ):
        raise RuntimeError("Frozen full-corpus primary RawSpan coverage failed on resume")
    with httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False) as client:
        runtime = _runtime(client)
    if runtime != preflight.get("writer_runtime_identity"):
        raise RuntimeError("Frozen writer runtime identity changed after chunk preflight")
    return (
        preflight,
        chunk_rows,
        {
            "upstream": upstream,
            "sessions": sessions,
            "refs": refs,
            "inventory": inventory,
        },
    )


def _write_progress(
    preflight: dict[str, Any], chunk_ledger: list[dict[str, Any]], status: str
) -> None:
    _write_json(
        RUN_DIR / "writer_progress.json",
        {
            "schema_version": 1,
            "stage": RUN_ID,
            "status": status,
            "planned_chunks": preflight["total_chunks"],
            "completed_chunks": sum(row.get("success") is True for row in chunk_ledger),
            "provider_calls": sum(row.get("provider_calls", 0) for row in chunk_ledger),
            "provider_calls_this_process": sum(
                row.get("provider_calls_this_resume", 0) for row in chunk_ledger
            ),
            "retries": 0,
            "hosted_calls": 0,
            "labels_loaded": False,
        },
    )


def _extract_chunks(
    preflight: dict[str, Any],
    chunk_rows: list[dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prompt = base.PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = preflight["v3_extractor_prompt_sha256"]
    method_sha = preflight["method_contract_sha256"]
    runtime = preflight["writer_runtime_identity"]
    validator_source_sha = _sha_file(Path(base.writer_v3.__file__))
    unwrap_source_sha = _sha_file(Path(base.__file__).with_name("flat_proposition_writer_v2.py"))
    ordered_rows = sorted(
        chunk_rows,
        key=lambda row: (
            0 if RISK_SESSION_ID in row["source_session_ids"] else 1,
            list(sessions).index(row["session_identity_sha256"]),
            row["chunk_index"],
        ),
    )
    chunk_outputs: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    output_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    new_provider_calls = 0
    call_started_at = time.perf_counter()
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=1800.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        if _runtime(client) != runtime:
            raise RuntimeError("Writer runtime changed immediately before chunk calls")
        for index, chunk in enumerate(ordered_rows, 1):
            session = sessions[chunk["session_identity_sha256"]]
            visible_catalog = [
                session["catalog"][ordinal] for ordinal in chunk["visible_span_ordinals"]
            ]
            request = writer_v4.writer_request(
                session_date=session["session_date"],
                catalog=visible_catalog,
                system_prompt=prompt,
                model_alias=flat.READER_MODEL,
            )
            request_sha = sha256_bytes(canonical_json(request))
            if request_sha != chunk["request_sha256"]:
                raise RuntimeError(f"Frozen request changed for chunk {chunk['chunk_id']}")

            def provider(req: dict[str, Any]) -> tuple[int, bytes, str | None]:
                nonlocal new_provider_calls
                if req != request:
                    raise RuntimeError("Provider request differs from frozen chunk request")
                new_provider_calls += 1
                response = client.post(f"{flat.READER_ENDPOINT}/chat/completions", json=req)
                return response.status_code, response.content, response.headers.get("content-type")

            try:
                normalized, call = execute_or_resume(
                    request=request,
                    catalog=visible_catalog,
                    session_identity_sha256=chunk["chunk_id"],
                    prompt_sha256=prompt_sha,
                    contract_sha256=method_sha,
                    local_cache_root=LOCAL_CACHE_ROOT,
                    provider=provider,
                    stage_identity="MEM-3A.3/flat-proposition-extractor-v4-chunked",
                    model_sha256=runtime["model_sha256"],
                    dynamic_schema_sha256=chunk["dynamic_schema_sha256"],
                    unwrap_source_sha256=unwrap_source_sha,
                    packet_validator=base.writer_v3.validate_packet,
                    packet_validator_sha256=validator_source_sha,
                )
            except (WriterQualificationFailure, WriterRecoveryError) as exc:
                failure_ledger = getattr(exc, "ledger", {})
                failure_row = {
                    **chunk,
                    **failure_ledger,
                    "success": False,
                    "failure": getattr(exc, "error", str(exc)),
                    "hosted_call": False,
                    "retry_count": 0,
                }
                ledger_rows.append(failure_row)
                _write_jsonl(RUN_DIR / "chunk_writer_ledger.partial.jsonl", ledger_rows)
                _write_progress(preflight, ledger_rows, "FAILED_STOP_NO_RETRY")
                _write_json(
                    RUN_DIR / "writer_failure.json",
                    {
                        "schema_version": 1,
                        "stage": RUN_ID,
                        "gate": f"{GATE}=NO",
                        "failure_type": type(exc).__name__,
                        "failure": failure_row["failure"],
                        "failed_chunk_id": chunk["chunk_id"],
                        "failed_source_session_ids": chunk["source_session_ids"],
                        "primary_turn_indices": chunk["primary_turn_indices"],
                        "primary_span_ids": chunk["primary_span_ids"],
                        "request_sha256": failure_ledger.get("request_sha256", request_sha),
                        "http_envelope_sha256": failure_ledger.get("http_envelope_sha256"),
                        "assistant_content_sha256": failure_ledger.get("assistant_content_sha256"),
                        "finish_reason": failure_ledger.get("finish_reason"),
                        "prompt_tokens": failure_ledger.get("prompt_tokens"),
                        "completion_tokens": failure_ledger.get("completion_tokens"),
                        "provider_calls_this_process": new_provider_calls,
                        "retries": 0,
                        "hosted_calls": 0,
                        "labels_loaded": False,
                        "downstream_started": False,
                    },
                )
                raise RuntimeError(
                    f"{GATE}=NO; stopped on exact failed chunk {chunk['chunk_id']}"
                ) from exc
            if (
                call.get("finish_reason") == "length"
                and call.get("completion_tokens") == MAX_COMPLETION_TOKENS
            ):
                raise RuntimeError(f"{GATE}=NO; completion cap reached for {chunk['chunk_id']}")
            chunk_output = {
                "chunk_index": chunk["chunk_index"],
                "chunk_id": chunk["chunk_id"],
                "session_identity_sha256": chunk["session_identity_sha256"],
                "source_session_ids": chunk["source_session_ids"],
                "normalized_packet": normalized,
            }
            chunk_outputs.append(chunk_output)
            output_by_session[chunk["session_identity_sha256"]].append(chunk_output)
            ledger_rows.append(
                {
                    **chunk,
                    **call,
                    "normalized_proposition_count": len(normalized["propositions"]),
                    "success": call.get("validation") == "passed",
                    "hosted_call": False,
                    "retry_count": 0,
                }
            )
            if index % 10 == 0 or index == len(ordered_rows):
                _write_jsonl(RUN_DIR / "chunk_writer_ledger.partial.jsonl", ledger_rows)
                _write_progress(preflight, ledger_rows, "WRITER_IN_PROGRESS")

    if len(chunk_outputs) != len(chunk_rows) or any(
        row.get("success") is not True for row in ledger_rows
    ):
        raise RuntimeError(f"{GATE}=NO; not all chunk writer outcomes completed")
    chunk_outputs.sort(
        key=lambda row: (list(sessions).index(row["session_identity_sha256"]), row["chunk_index"])
    )
    _write_jsonl(RUN_DIR / "chunk_writer_ledger.jsonl", ledger_rows)
    _write_jsonl(RUN_DIR / "chunk_normalized_outputs.jsonl", chunk_outputs)

    session_packets = []
    aggregate_diagnostics = {}
    for identity, session in sessions.items():
        outputs = output_by_session[identity]
        props, diagnostics = aggregate_chunk_propositions(
            catalog=session["catalog"], chunk_outputs=outputs
        )
        aggregate_diagnostics[identity] = diagnostics
        session_packets.append(
            {
                "session_identity_sha256": identity,
                "session_date": session["session_date"],
                "valid_from": session["valid_from"],
                "source_session_ids": session["source_session_ids"],
                "source_turns_sha256": session["source_turns_sha256"],
                "catalog_sha256": session["catalog_sha256"],
                "propositions": props,
                "source_chunk_ids": [row["chunk_id"] for row in outputs],
                "exact_cross_chunk_duplicates_removed": diagnostics[
                    "exact_cross_chunk_duplicates_removed"
                ],
            }
        )
    if len(session_packets) != 477:
        raise RuntimeError("Session aggregation did not cover all 477 source sessions")
    _write_jsonl(flat.EXTRACTIONS_PATH, session_packets)
    extraction_manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "method_contract_sha256": method_sha,
        "planned_sessions": 477,
        "aggregated_sessions": len(session_packets),
        "planned_chunks": len(chunk_rows),
        "successful_chunks": len(chunk_outputs),
        "provider_calls_unique": sum(row.get("provider_calls", 0) for row in ledger_rows),
        "provider_calls_this_process": new_provider_calls,
        "retries": 0,
        "hosted_calls": 0,
        "session_extractions_sha256": _sha_file(flat.EXTRACTIONS_PATH),
        "chunk_ledger_sha256": _sha_file(RUN_DIR / "chunk_writer_ledger.jsonl"),
        "chunk_outputs_sha256": _sha_file(RUN_DIR / "chunk_normalized_outputs.jsonl"),
        "aggregate_diagnostics": aggregate_diagnostics,
        "labels_loaded": False,
        "test_access": False,
        "102_dev_access": False,
    }
    _write_json(flat.EXTRACTION_MANIFEST_PATH, extraction_manifest)
    _write_progress(preflight, ledger_rows, "WRITER_COMPLETE")
    manifest = {
        "chunk_calls": len(ledger_rows),
        "provider_calls_unique": extraction_manifest["provider_calls_unique"],
        "provider_calls_this_process": new_provider_calls,
        "retries": 0,
        "hosted_calls": 0,
        "wall_seconds": round(time.perf_counter() - call_started_at, 3),
        "output_by_session": output_by_session,
        "chunk_outputs": chunk_outputs,
        "aggregate_diagnostics": aggregate_diagnostics,
    }
    return session_packets, ledger_rows, manifest


def _writer_diagnostics(
    packets: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
    chunk_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    props = [prop for packet in packets for prop in packet["propositions"]]
    high_similarity_pairs = []
    same_text_different_refs = 0
    props_by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chunk_outputs:
        props_by_identity[row["session_identity_sha256"]].extend(
            {**prop, "chunk_id": row["chunk_id"]}
            for prop in row["normalized_packet"]["propositions"]
        )
    for identity, rows in props_by_identity.items():
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                if left["chunk_id"] == right["chunk_id"]:
                    continue
                if (
                    left["proposition_text"] == right["proposition_text"]
                    and left["evidence_refs"] != right["evidence_refs"]
                ):
                    same_text_different_refs += 1
                left_text = left["proposition_text"].casefold()
                right_text = right["proposition_text"].casefold()
                if (
                    min(len(left_text), len(right_text))
                    / max(1, max(len(left_text), len(right_text)))
                    < HIGH_SIMILARITY_THRESHOLD
                ):
                    continue
                ratio = SequenceMatcher(
                    None,
                    left_text,
                    right_text,
                    autojunk=False,
                ).ratio()
                if ratio >= HIGH_SIMILARITY_THRESHOLD:
                    high_similarity_pairs.append(
                        {
                            "session_identity_sha256": identity,
                            "chunk_ids": [left["chunk_id"], right["chunk_id"]],
                            "similarity": round(ratio, 4),
                            "left_text_sha256": sha256_bytes(
                                left["proposition_text"].encode("utf-8")
                            ),
                            "right_text_sha256": sha256_bytes(
                                right["proposition_text"].encode("utf-8")
                            ),
                            "retained": True,
                        }
                    )
    authority = defaultdict(int)
    for prop in props:
        authority[prop["source_authority"]] += 1
    chunks_per_session = [
        sum(row["session_identity_sha256"] == packet["session_identity_sha256"] for row in ledger)
        for packet in packets
    ]
    return {
        "source_sessions": len(packets),
        "rawspan_count": len(inventory),
        "total_chunks": len(ledger),
        "chunks_per_session": _percentiles(chunks_per_session),
        "request_budget_tokens_per_chunk": _percentiles(
            [row["request_budget_tokens"] for row in ledger]
        ),
        "rendered_chat_prompt_tokens_per_chunk": _percentiles(
            [row["rendered_chat_prompt_tokens"] for row in ledger]
        ),
        "dynamic_schema_tokens_per_chunk": _percentiles(
            [row["dynamic_schema_tokens"] for row in ledger]
        ),
        "completion_tokens_per_chunk": _percentiles(
            [row["completion_tokens"] for row in ledger if row.get("completion_tokens") is not None]
        ),
        "total_propositions": len(props),
        "propositions_per_chunk": _percentiles(
            [row["normalized_proposition_count"] for row in ledger]
        ),
        "propositions_per_session": _percentiles(
            [len(packet["propositions"]) for packet in packets]
        ),
        "exact_overlap_duplicates_removed": sum(
            packet["exact_cross_chunk_duplicates_removed"] for packet in packets
        ),
        "same_text_different_evidence_pairs_retained": same_text_different_refs,
        "high_similarity_cross_chunk_pairs_retained": high_similarity_pairs,
        "high_similarity_definition": f"SequenceMatcher ratio >= {HIGH_SIMILARITY_THRESHOLD:.2f} on casefolded proposition text; heuristic diagnostic only; no merge",
        "authority_counts": {
            "user_only_propositions": authority["user"],
            "assistant_only_propositions": authority["assistant"],
            "mixed_propositions": authority["mixed"],
        },
        "writer_wall_seconds": sum(row.get("provider_duration_ms") or 0 for row in ledger) / 1000,
        "writer_wall_seconds_this_process": sum(
            row.get("provider_duration_ms") or 0
            for row in ledger
            if row.get("provider_calls_this_resume") == 1
        )
        / 1000,
        "provider_calls": sum(row.get("provider_calls", 0) for row in ledger),
        "provider_calls_this_process": sum(
            row.get("provider_calls_this_resume", 0) for row in ledger
        ),
        "retries": sum(row.get("retry_count", 0) for row in ledger),
        "hosted_calls": sum(bool(row.get("hosted_call")) for row in ledger),
        "writer_input_tokens_server_reported": sum(row.get("prompt_tokens") or 0 for row in ledger),
        "writer_completion_tokens": sum(row.get("completion_tokens") or 0 for row in ledger),
        "semantic_quality_used_as_gate": False,
        "revision_keys_extracted": False,
    }


def _efficiency(
    chunk_diagnostics: dict[str, Any],
    embedding: dict[str, Any],
    bundles: list[dict[str, Any]],
    reader_ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    latencies = [row["retrieval_latency_ms"] for row in bundles]
    return {
        "schema_version": 1,
        "writer": chunk_diagnostics,
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


def _comparison(metrics: dict[str, Any]) -> dict[str, Any]:
    result = base._comparison(metrics)
    result["candidate"] = "MEM-3A.3 Deterministic Chunked FlatProp + Dense"
    result["changed_component"] = (
        "writer execution granularity only; proposition semantics and retrieval are unchanged"
    )
    result["interpretation"] = (
        "Descriptive frozen-ten diagnostic only; no ranking, superiority claim, or statistical inference."
    )
    return result


def _case_review(
    flat_rows: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    questions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    review = base._case_review(flat_rows, ranked, predictions, labels, questions)
    instagram_rows = []
    for row in flat_rows:
        if row["question_id"] != "1cea1afa":
            continue
        combined = " ".join(
            [row["proposition_text"], *(item["evidence_quote"] for item in row["evidence"])]
        )
        if "instagram" not in combined.casefold():
            continue
        for count in (500, 600):
            if re.search(rf"\b{count}\b", combined):
                instagram_rows.append(
                    {
                        "follower_count": count,
                        "memory_id": row["memory_id"],
                        "source_session_id": row["source_session_id"],
                        "proposition_text": row["proposition_text"],
                        "evidence_refs": row["evidence_refs"],
                        "valid_from": row["valid_from"],
                    }
                )
    checkpoint = review["instagram_500_to_600_checkpoint"]
    checkpoint.update(
        {
            "flatprop_observations": instagram_rows,
            "flatprop_has_500": any(row["follower_count"] == 500 for row in instagram_rows),
            "flatprop_has_600": any(row["follower_count"] == 600 for row in instagram_rows),
            "multiple_historical_values_survive_simultaneously": (
                any(row["follower_count"] == 500 for row in instagram_rows)
                and any(row["follower_count"] == 600 for row in instagram_rows)
            ),
            "resolved_or_suppressed": False,
        }
    )
    for case in review["cases"]:
        if case["question_id"] in {"1cea1afa", "c4ea545c"}:
            case["historical_state_observations"] = [
                row for row in flat_rows if row["question_id"] == case["question_id"]
            ]
            case["state_conflicts_resolved"] = False
    review["critical_conflict_cases"] = ["1cea1afa", "c4ea545c"]
    review["conflict_packet_is_diagnostic_only"] = True
    return review


def _render_report(
    gate: str,
    diagnostics: dict[str, Any],
    embedding: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    manifest: dict[str, Any],
    case_review: dict[str, Any],
) -> str:
    means = metrics["means"]
    instagram = case_review["instagram_500_to_600_checkpoint"]
    lines = [
        "# MEM-3A.3 - Deterministic Chunked FlatProp Frozen-Ten Diagnostic",
        "",
        f"Completion gate: `{GATE}={gate}`.",
        "",
        "This is a controlled frozen-ten mechanism diagnostic, not a public benchmark result or performance ranking. It compares MEM-2D RawSpan + Dense with chunked FlatProp + Dense while retaining the v3 proposition semantics and changing only writer execution granularity.",
        "",
        "## Chunk Writer",
        "",
        f"- Source sessions / RawSpans: {manifest['source_sessions']} / {manifest['source_rawspans']}; chunks: {diagnostics['total_chunks']}; chunks/session P50/P90/P95/P99/max: {diagnostics['chunks_per_session']}.",
        f"- Request-budget tokens/chunk P50/P90/P95/P99/max: {diagnostics['request_budget_tokens_per_chunk']}; hard ceiling 6,144.",
        f"- Rendered chat tokens/chunk: {diagnostics['rendered_chat_prompt_tokens_per_chunk']}; schema tokens/chunk: {diagnostics['dynamic_schema_tokens_per_chunk']} (measured separately).",
        f"- Completion tokens/chunk: {diagnostics['completion_tokens_per_chunk']}; proposition counts/chunk and/session: {diagnostics['propositions_per_chunk']} / {diagnostics['propositions_per_session']}.",
        f"- Exact cross-chunk duplicates removed: {diagnostics['exact_overlap_duplicates_removed']}; same-text/different-evidence pairs retained: {diagnostics['same_text_different_evidence_pairs_retained']}; high-similarity cross-chunk pairs retained: {len(diagnostics['high_similarity_cross_chunk_pairs_retained'])} (heuristic).",
        f"- Authority counts: user-only {diagnostics['authority_counts']['user_only_propositions']}, assistant-only {diagnostics['authority_counts']['assistant_only_propositions']}, mixed {diagnostics['authority_counts']['mixed_propositions']}.",
        f"- Local writer provider calls: {diagnostics['provider_calls']}; retries {diagnostics['retries']}; hosted calls {diagnostics['hosted_calls']}; provider latency sum {diagnostics['writer_wall_seconds']:.3f}s; stage wall time {diagnostics['writer_stage_elapsed_wall_seconds']:.3f}s.",
        "- Every chunk output is retained with chunk/request/envelope/content hashes, finish reason, token counts, normalized propositions, and Harness-derived provenance. Historical 3A.2/3A.2S whole-session responses were not imported.",
        "",
        "## RawSpan Comparison",
        "",
        f"- Local embedding: `{embedding['model_id']}` revision `{embedding['revision']}`, {embedding['dimensions']} dimensions, {embedding['device']} / {embedding['dtype']}; hosted calls {embedding['hosted_calls']}; truncations {embedding['document_truncations'] + embedding['query_truncations']}.",
        f"- Dense top-8 rows: {manifest['dense_rows']}; projection: `m10-rank-aware-projection-v1`, 1,024 tokens; shared reader calls: {manifest['reader_calls']}/10; judge calls: 0.",
        f"- Mean answer-session Recall@5 / @8 / MRR: {means['answer_session_recall_at_5']} / {means['answer_session_recall_at_8']} / {means['mrr']}; projected Recall@8: {means['projected_answer_session_recall_at_8']}.",
        f"- Mean context reader tokens: {means['reader_visible_context_tokens']}; deterministic token precision / recall / F1 / normalized EM: {means['token_precision']} / {means['token_recall']} / {means['token_f1']} / {means['normalized_em']}.",
        "- Per-question metrics and deltas are in `comparison_mem2d_rawspan_vs_mem3a3.json`; all values are descriptive and do not establish superiority or generalization.",
        "",
        "## State Conflicts",
        "",
        f"- Instagram 500 and 600 observations captured: {instagram['flatprop_has_500']} / {instagram['flatprop_has_600']}; both survive in the FlatProp inventory: {instagram['multiple_historical_values_survive_simultaneously']}; no value was resolved or suppressed.",
        "- Frozen update cases `1cea1afa` and `c4ea545c` include retrieved propositions and Harness evidence in `mem_3a3_case_review.json`; conflict packet is diagnostic only.",
        "",
        "## Scope",
        "",
        f"- Primary span coverage exactly once: {manifest['gate'].get('primary_source_coverage_exactly_once')}; all request budgets <= 6,144: {manifest['gate'].get('all_rendered_chunk_prompts_within_ceiling')}.",
        "- No revision semantics, no LongMemEval 102 DEV, no TEST, no MedMemoryBench run, no judge, no hosted provider, and no benchmark-level claim.",
        f"- `MEM3A3_FLAT_NO_REVISION={'YES' if manifest.get('flat_no_revision_gate_marker') == 'MEM3A3_FLAT_NO_REVISION=YES' else 'NO'}`.",
        "- Full-context/full-session writer is closed; no output cap increase or retry was attempted.",
        "",
        f"`{GATE}={gate}`",
        "",
    ]
    return "\n".join(lines)


def _downstream(
    upstream: dict[str, Any],
    preflight: dict[str, Any],
    packets: list[dict[str, Any]],
    chunk_ledger: list[dict[str, Any]],
    sessions: dict[str, dict[str, Any]],
    refs: dict[str, dict[str, str]],
    inventory: list[dict[str, Any]],
    writer_manifest: dict[str, Any],
    chunk_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(packets) != 477 or len(chunk_ledger) != preflight["total_chunks"]:
        raise RuntimeError("MEM-3A.3 downstream requires all 477 aggregates and all chunk calls")
    rows, operations, records = base._materialize(packets, refs, V3_CONTRACT_SHA256)
    _write_jsonl(flat.FLAT_PROPOSITIONS_PATH, rows)
    _write_jsonl(flat.MATERIALIZATION_LEDGER_PATH, operations)
    embedding_started = time.perf_counter()
    ranked, embedding, query_tokens, retrieval_latency = flat._embedding_and_retrieval(
        rows, records, refs, preflight
    )
    embedding["query_token_counts"] = query_tokens
    embedding["representation"] = "flatprop_v4_chunked_proposition_text_only"
    embedding["retrieval_algorithm"] = (
        "scope/time-valid Dense cosine; top_k=8; memory_id ascending tie-break"
    )
    _write_json(flat.EMBEDDING_MANIFEST_PATH, embedding)
    frozen = mem2d._load_frozen_inputs()
    questions = frozen["questions"]
    with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
        reader_contract = flat.load_final_reader_contract()
        if (
            reader_contract[1] != flat.PINNED_READER_SHA256
            or reader_contract[1] != upstream["reader_contract_sha256"]
        ):
            raise RuntimeError("Frozen shared-reader contract changed")
        plans, bundles, _ = flat._project_contexts(
            ranked,
            records,
            questions,
            client,
            reader_contract,
            retrieval_latency,
        )
    _write_jsonl(flat.DENSE_TOP8_PATH, ranked)
    _write_jsonl(flat.CONTEXT_PLANS_PATH, plans)
    _write_jsonl(flat.CONTEXT_BUNDLES_PATH, bundles)
    pre_reader_paths = [
        flat.PREFLIGHT_PATH,
        RUN_DIR / "writer_extraction_identity.json",
        RUN_DIR / "writer_method_identity.json",
        RUN_DIR / "mem_3a3_protocol.md",
        RUN_DIR / "evidence_ref_set_normalization_v1.json",
        RUN_DIR / "chunk_manifest.jsonl",
        RUN_DIR / "chunk_normalized_outputs.jsonl",
        flat.EXTRACTION_MANIFEST_PATH,
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
        raise RuntimeError("MEM-3A.3 pre-reader artifacts failed freeze/top-8 completeness")
    pre_reader_sha = sha256_bytes(
        canonical_json({path.name: _sha_file(path) for path in pre_reader_paths})
    )
    predictions, reader_ledger = flat._run_reader(pre_reader_sha, preflight, bundles)
    predictions = [{**row, "system": "mem3a3_chunked_flatprop_dense"} for row in predictions]
    _write_jsonl(flat.PREDICTIONS_PATH, predictions)
    _write_jsonl(flat.READER_LEDGER_PATH, reader_ledger)
    if not all(
        flat._verify_frozen(path) for path in (flat.PREDICTIONS_PATH, flat.READER_LEDGER_PATH)
    ):
        raise RuntimeError("Frozen-ten prediction or reader ledger SHA failed")

    labels = flat._labels_after_freeze()
    metrics = base._metrics(predictions, reader_ledger, bundles, plans, ranked, rows, labels)
    _write_json(flat.METRICS_PATH, metrics)
    diagnostics = _writer_diagnostics(packets, chunk_ledger, inventory, chunk_outputs)
    diagnostics["writer_stage_elapsed_wall_seconds"] = writer_manifest["wall_seconds"]
    _write_json(RUN_DIR / "writer_chunk_diagnostics.json", diagnostics)
    comparison = _comparison(metrics)
    _write_json(flat.COMPARISON_PATH, comparison)
    efficiency = _efficiency(diagnostics, embedding, bundles, reader_ledger)
    _write_json(flat.EFFICIENCY_PATH, efficiency)
    case_review = _case_review(rows, ranked, predictions, labels, questions)
    _write_json(flat.CASE_REVIEW_PATH, case_review)

    no_revision = all(
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
    primary_exact = preflight["primary_span_coverage_exactly_once"]
    turn_exact = preflight["primary_turn_coverage_exactly_once"]
    all_chunk_success = len(chunk_ledger) == preflight["total_chunks"] and all(
        row["success"] and row.get("validation") == "passed" for row in chunk_ledger
    )
    all_artifacts = [
        *pre_reader_paths,
        flat.PREDICTIONS_PATH,
        flat.READER_LEDGER_PATH,
        flat.METRICS_PATH,
        flat.EFFICIENCY_PATH,
        flat.COMPARISON_PATH,
        flat.CASE_REVIEW_PATH,
        RUN_DIR / "writer_chunk_diagnostics.json",
        RUN_DIR / "mem_3a3_protocol.md",
        RUN_DIR / "evidence_ref_set_normalization_v1.json",
    ]
    gate = {
        "upstream_historical_gates_verified": True,
        "477_sessions_chunk_planned": preflight["unique_source_sessions"] == 477,
        "primary_source_coverage_exactly_once": primary_exact,
        "primary_turn_coverage_exactly_once": turn_exact,
        "all_rendered_chunk_prompts_within_ceiling": preflight["all_rendered_request_budgets_fit"],
        "all_chunk_writer_calls_complete": all_chunk_success,
        "exactly_one_local_provider_call_per_chunk": writer_manifest["provider_calls_unique"]
        == preflight["total_chunks"]
        and all(row.get("provider_calls") == 1 for row in chunk_ledger),
        "finish_reason_length_zero": all(
            row.get("finish_reason") != "length" for row in chunk_ledger
        ),
        "zero_retries": writer_manifest["retries"] == 0
        and all(row.get("retry_count") == 0 for row in chunk_ledger),
        "zero_hosted_calls": writer_manifest["hosted_calls"] == 0
        and all(not row.get("hosted_call") for row in chunk_ledger + reader_ledger),
        "all_provenance_harness_derived": provenance_valid,
        "flat_add_active_v1_no_revision": no_revision,
        "frozen_qwen3_embedding_local_cuda_no_truncation": embedding["local_only"]
        and embedding["hosted_calls"] == 0
        and embedding["model_id"] == "Qwen/Qwen3-Embedding-0.6B"
        and str(embedding["device"]).startswith("cuda")
        and embedding["dtype"] == "float16"
        and embedding["document_truncations"] == 0
        and embedding["query_truncations"] == 0,
        "dense_top8_complete_for_ten": len(ranked) == 80,
        "rank_aware_projection_sha_unchanged": _sha_file(flat.PROJECTION_CONTRACT_PATH)
        == flat.PINNED_PROJECTION_SHA256,
        "memory_budget_1024": flat.MEMORY_BUDGET == 1024,
        "shared_reader_successful_10_of_10": len(reader_ledger) == 10
        and all(row["success"] and row["quality_status"] == "OK" for row in reader_ledger),
        "judge_calls_zero": True,
        "test_access_false": True,
        "102_dev_not_run": True,
        "medmemorybench_runs_zero": True,
        "historical_whole_session_outputs_not_imported": True,
        "prediction_reader_ledgers_frozen_before_labels": True,
        "critical_conflict_packet_complete": len(case_review["cases"]) == 8
        and case_review["critical_conflict_cases"] == ["1cea1afa", "c4ea545c"]
        and case_review["instagram_500_to_600_checkpoint"].get("resolved_or_suppressed") is False
        and isinstance(case_review["instagram_500_to_600_checkpoint"].get("flatprop_has_500"), bool)
        and isinstance(
            case_review["instagram_500_to_600_checkpoint"].get("flatprop_has_600"), bool
        ),
        "all_compact_artifact_sha_sidecars_valid": all(
            flat._verify_frozen(path) for path in all_artifacts
        ),
    }
    completion = "YES" if all(gate.values()) else "NO"
    manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "COMPLETE" if completion == "YES" else "FAILED",
        "base_commit_sha": preflight["stage_base_commit_sha"],
        "question_ids": list(flat.QUESTION_IDS),
        "reader_model_role": "reader_answer; local Qwen3-8B Q4_K_M",
        "memory_internal_llm_role": "memory_ingest; same frozen local Qwen3-8B",
        "embedding_model_role": "Qwen/Qwen3-Embedding-0.6B local CUDA FP16",
        "judge_model": None,
        "method_identity": writer_v4.CONTRACT_ID,
        "method_contract_sha256": preflight["method_contract_sha256"],
        "source_sessions": 477,
        "source_rawspans": len(inventory),
        "planned_chunks": preflight["total_chunks"],
        "successful_chunk_outcomes": sum(row["success"] for row in chunk_ledger),
        "writer_provider_calls_unique": writer_manifest["provider_calls_unique"],
        "writer_calls_this_process": writer_manifest["provider_calls_this_process"],
        "writer_failures": sum(not row["success"] for row in chunk_ledger),
        "retries": writer_manifest["retries"],
        "hosted_calls": writer_manifest["hosted_calls"]
        + sum(bool(row["hosted_call"]) for row in reader_ledger),
        "reader_calls": len(reader_ledger),
        "judge_calls": 0,
        "dense_rows": len(ranked),
        "memory_operations": len(operations),
        "memory_records": len(rows),
        "labels_loaded_after_freeze": True,
        "test_access": False,
        "102_dev_access": False,
        "medmemorybench_runs": 0,
        "gate": gate,
        "completion_gate_marker": f"{GATE}={completion}",
        "flat_no_revision_gate_marker": "MEM3A3_FLAT_NO_REVISION=YES"
        if no_revision
        else "MEM3A3_FLAT_NO_REVISION=NO",
        "upstream": upstream,
        "pre_reader_freeze_sha256": pre_reader_sha,
        "embedding_and_projection_wall_seconds": round(time.perf_counter() - embedding_started, 3),
        "artifact_sha256": {},
    }
    report = _render_report(
        completion, diagnostics, embedding, metrics, comparison, manifest, case_review
    )
    atomic_write_bytes(flat.RUN_REPORT_PATH, report.encode("utf-8"))
    flat._freeze(flat.RUN_REPORT_PATH)
    all_artifacts.append(flat.RUN_REPORT_PATH)
    manifest["artifact_sha256"] = {path.name: _sha_file(path) for path in all_artifacts}
    manifest["gate"]["all_compact_artifact_sha_sidecars_valid"] = all(
        flat._verify_frozen(path) for path in all_artifacts
    )
    completion = "YES" if all(manifest["gate"].values()) else "NO"
    if completion != gate:
        report = _render_report(
            completion, diagnostics, embedding, metrics, comparison, manifest, case_review
        )
        atomic_write_bytes(flat.RUN_REPORT_PATH, report.encode("utf-8"))
        flat._freeze(flat.RUN_REPORT_PATH)
        manifest["artifact_sha256"][flat.RUN_REPORT_PATH.name] = _sha_file(flat.RUN_REPORT_PATH)
    manifest["completion_gate_marker"] = f"{GATE}={completion}"
    manifest["status"] = "COMPLETE" if completion == "YES" else "FAILED"
    _write_json(flat.RUN_MANIFEST_PATH, manifest)
    if completion != "YES":
        raise RuntimeError(f"{GATE}=NO")
    print(f"{GATE}=YES", flush=True)
    return manifest


def run(*, preflight_only: bool = False) -> dict[str, Any]:
    head = _configure_paths()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    preflight, chunk_rows, source = _load_or_build_preflight(head)
    print(f"frozen_sessions={preflight['unique_source_sessions']}", flush=True)
    print(f"frozen_chunks={preflight['total_chunks']}", flush=True)
    print(
        f"max_request_budget_tokens={preflight['request_budget_prompt_tokens_per_chunk']['max']}",
        flush=True,
    )
    if preflight_only:
        print("MEM3A3_FULL_CHUNK_PREFLIGHT=YES", flush=True)
        return preflight
    packets, chunk_ledger, writer_manifest = _extract_chunks(
        preflight, chunk_rows, source["sessions"]
    )
    if len(packets) != 477 or len(chunk_ledger) != preflight["total_chunks"]:
        raise RuntimeError(f"{GATE}=NO; chunk writer output is incomplete; downstream not started")
    return _downstream(
        source["upstream"],
        preflight,
        packets,
        chunk_ledger,
        source["sessions"],
        source["refs"],
        source["inventory"],
        writer_manifest,
        writer_manifest["chunk_outputs"],
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
    except Exception as exc:
        if RUN_DIR.exists():
            try:
                report_path = RUN_DIR / "report.md"
                labels_loaded = (RUN_DIR / "deterministic_metrics.json").exists()
                if not report_path.exists():
                    label_note = (
                        "Labels were loaded only after prediction and reader artifacts froze."
                        if labels_loaded
                        else "No benchmark labels were loaded."
                    )
                    report = (
                        "# MEM-3A.3 - Deterministic Chunked FlatProp Diagnostic\n\n"
                        f"Completion gate: `{GATE}=NO`.\n\n"
                        f"Stopped before closeout: `{type(exc).__name__}: {exc}`.\n\n"
                        f"{label_note} "
                        "See `writer_failure.json` for the exact chunk when a writer request failed.\n"
                    )
                    atomic_write_bytes(report_path, report.encode("utf-8"))
                    _freeze(report_path)
                _write_json(
                    RUN_DIR / "run_failure.json",
                    {
                        "schema_version": 1,
                        "stage": RUN_ID,
                        "status": "FAILED",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "completion_gate_marker": f"{GATE}=NO",
                        "hosted_calls": 0,
                        "judge_calls": 0,
                        "labels_loaded": labels_loaded,
                        "downstream_started": (RUN_DIR / "flatprop_inventory.jsonl").exists(),
                    },
                )
                failure_manifest_path = RUN_DIR / "run_manifest.json"
                if not failure_manifest_path.exists():
                    _write_json(
                        failure_manifest_path,
                        {
                            "schema_version": 1,
                            "stage": RUN_ID,
                            "status": "FAILED",
                            "completion_gate_marker": f"{GATE}=NO",
                            "failure_type": type(exc).__name__,
                            "failure": str(exc),
                            "labels_loaded": labels_loaded,
                            "hosted_calls": 0,
                            "judge_calls": 0,
                            "test_access": False,
                            "102_dev_access": False,
                            "medmemorybench_runs": 0,
                        },
                    )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
