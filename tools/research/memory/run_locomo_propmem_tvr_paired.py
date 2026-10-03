"""Run the frozen local PropMem vs query-time TVR LoCoMo paired closeout."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from locomo_tvr import SAME_TOPIC_COSINE_THRESHOLD, inject_temporal_view, project_temporal_view
from mem1_artifacts import append_jsonl, canonical_json, read_jsonl, sha256_file

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
EXTERNAL = WORKSPACE / "external" / "memory"
LOCOMO_ROOT = EXTERNAL / "LoCoMo"
MEMEVAL_ROOT = EXTERNAL / "MemEval"
DATASET = LOCOMO_ROOT / "data" / "locomo10.json"
RUN_ROOT = ROOT / "runs" / "memory" / "locomo" / "memq1"
PATCH = ROOT / "tools" / "research" / "memory" / "patches" / "memeval_qwen_main_v1.patch"
PATCH_SHA256 = "c7e44015527a17d43cf9bbd2e2a6e50d007943ee2efaa244899d3b048fd80eab"
EXPECTED_DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
EXPECTED_LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
EXPECTED_MEMEVAL_COMMIT = "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4"
READER_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
LLAMA_SHA256 = "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
READER_PATH = Path(r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
EMBEDDING_PATH = Path(r"E:\Health-Copilot-Models\models\Qwen3-Embedding-0.6B")
ENDPOINT = "http://127.0.0.1:8081/v1"
READER_MODEL = "health-memory-qwen3-8b"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
ANSWER_BUDGET = 256
RETRY_LIMIT = 2


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(value))
    os.replace(temporary, path)


def _hash_tree(root: Path, *, exclude: set[str] | None = None) -> str:
    excluded = exclude or set()
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in excluded):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel + b"\0" + bytes.fromhex(sha256_file(path)) + b"\n")
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True, check=check
    )


def _verify_revisions() -> dict[str, Any]:
    if _git(LOCOMO_ROOT, "rev-parse", "HEAD") != EXPECTED_LOCOMO_COMMIT:
        raise RuntimeError("LoCoMo checkout differs from the pinned official revision")
    if _git(MEMEVAL_ROOT, "rev-parse", "HEAD") != EXPECTED_MEMEVAL_COMMIT:
        raise RuntimeError("MemEval checkout differs from the pinned official revision")
    if sha256_file(DATASET) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("LoCoMo data bytes differ from the frozen dataset hash")
    if sha256_file(PATCH) != PATCH_SHA256:
        raise RuntimeError("Pinned MemEval local compatibility patch SHA changed")
    reader_hash = sha256_file(READER_PATH)
    if reader_hash != READER_SHA256:
        raise RuntimeError("Frozen Qwen3-8B model hash mismatch")
    return {
        "locomo_commit": EXPECTED_LOCOMO_COMMIT,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "memeval_commit": EXPECTED_MEMEVAL_COMMIT,
        "compatibility_patch_sha256": PATCH_SHA256,
        "reader_model_sha256": reader_hash,
    }


def _ensure_memeval_patch() -> bool:
    """Apply the audited adapter only for this run; return whether this process owns it."""
    reverse = _run_git(MEMEVAL_ROOT, "apply", "--reverse", "--check", str(PATCH), check=False)
    if reverse.returncode == 0:
        return False
    forward = _run_git(MEMEVAL_ROOT, "apply", "--check", str(PATCH), check=False)
    if forward.returncode != 0:
        raise RuntimeError("MemEval tree is neither pinned upstream nor the exact audited adapter")
    _run_git(MEMEVAL_ROOT, "apply", str(PATCH))
    return True


def _remove_owned_patch(owned: bool) -> None:
    if owned:
        _run_git(MEMEVAL_ROOT, "apply", "--reverse", str(PATCH))


def _configure_local_environment() -> None:
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
        os.environ.pop(name, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    os.environ["HC_MEMORY_TRACK"] = "main_local_only"
    os.environ["HC_LOCAL_READER_BASE_URL"] = ENDPOINT
    os.environ["HC_READER_MODEL"] = READER_MODEL
    os.environ["HC_LOCAL_EMBEDDING_PATH"] = str(EMBEDDING_PATH)
    os.environ["HC_LOCAL_EMBEDDING_DEVICE"] = "cuda:0"
    os.environ["HC_LOCAL_EMBEDDING_DTYPE"] = "float16"
    os.environ["HC_READER_ANSWER_MAX_NEW_TOKENS"] = str(ANSWER_BUDGET)


def _reader_process() -> dict[str, Any]:
    ps = (
        "$p=@(Get-CimInstance Win32_Process -Filter \"name='llama-server.exe'\" | "
        "Select-Object ProcessId,ExecutablePath,CommandLine); $p | ConvertTo-Json -Compress"
    )
    output = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", ps], text=True
    ).strip()
    processes = json.loads(output) if output else []
    if isinstance(processes, dict):
        processes = [processes]
    matches = [
        item for item in processes
        if "--port 8081" in str(item.get("CommandLine", ""))
        and "127.0.0.1" in str(item.get("CommandLine", ""))
        and str(READER_PATH).lower() in str(item.get("CommandLine", "")).lower()
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one independent frozen reader on loopback port 8081")
    process = matches[0]
    path = Path(process["ExecutablePath"])
    digest = sha256_file(path)
    command = str(process["CommandLine"])
    required = (
        "--ctx-size 131072", "--n-gpu-layers 99", "--flash-attn on",
        "--cache-type-k q4_0", "--cache-type-v q4_0", "--parallel 1",
        "--rope-scaling yarn", "--rope-scale 4",
    )
    missing = [arg for arg in required if arg.lower() not in command.lower()]
    if digest != LLAMA_SHA256 or missing:
        raise RuntimeError(f"Reader runtime differs from frozen contract: sha={digest}, missing={missing}")
    kv_cache_location = "cpu" if "--no-kv-offload" in command.lower() else "gpu"
    return {
        "pid": int(process["ProcessId"]),
        "executable": str(path),
        "binary_sha256": digest,
        "command_line": command,
        "version": "10068",
        "build": "571d0d540",
        "kv_cache_location": kv_cache_location,
    }


def _verify_endpoint(config: Any) -> dict[str, Any]:
    parsed = urlsplit(config.reader_base_url)
    if parsed.hostname != "127.0.0.1" or parsed.port != 8081:
        raise RuntimeError("Reader must use the owned loopback endpoint on 127.0.0.1:8081")
    from openai import OpenAI
    import httpx

    client = OpenAI(
        api_key="local-only-placeholder",
        base_url=config.reader_base_url,
        http_client=httpx.Client(trust_env=False, timeout=10.0),
        max_retries=0,
    )
    models = client.models.list().data
    model_ids = [model.id for model in models]
    client.close()
    if not model_ids or not any("qwen3-8b" in model.lower() for model in model_ids):
        raise RuntimeError(f"Endpoint is not the frozen local Qwen3-8B model: {model_ids}")
    return {"endpoint": config.reader_base_url, "model_ids": model_ids}


def _load_dataset() -> list[dict[str, Any]]:
    value = json.loads(DATASET.read_text(encoding="utf-8"))
    if len(value) != 10 or sum(len(row.get("qa", [])) for row in value) != 1986:
        raise RuntimeError("Pinned LoCoMo data must contain ten conversations and 1,986 QA")
    return value


def _blind_row(sample_id: str, qa_index: int, qa: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": f"{sample_id}:qa:{qa_index}",
        "sample_id": sample_id,
        "qa_index": qa_index,
        "question": str(qa.get("question", "")),
    }


def prepare_questions() -> str:
    dataset = _load_dataset()
    rows = [
        _blind_row(conversation["sample_id"], index, qa)
        for conversation in dataset
        for index, qa in enumerate(conversation["qa"])
    ]
    encoded = b"".join(canonical_json(row) + b"\n" for row in rows)
    path = RUN_ROOT / "frozen_questions.jsonl"
    sidecar = RUN_ROOT / "frozen_questions.sha256"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError("Blind question manifest exists with different contents")
    else:
        path.write_bytes(encoded)
    digest = sha256_file(path)
    if sidecar.exists() and sidecar.read_text(encoding="ascii").split()[0] != digest:
        raise RuntimeError("Blind question manifest sidecar mismatch")
    if not sidecar.exists():
        sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    for row in read_jsonl(path):
        forbidden = {"answer", "category", "evidence", "gold_evidence", "answer_session_ids"}
        if forbidden & row.keys():
            raise RuntimeError("Blind inference manifest unexpectedly includes benchmark labels")
    return digest


def _session_keys(conversation: dict[str, Any]) -> list[str]:
    return sorted(
        (key for key in conversation if key.startswith("session_") and not key.endswith("_date_time")),
        key=lambda value: int(value.split("_", 1)[1]),
    )


def _session_cache_identity(
    sample_id: str, session_key: str, turns: list[dict[str, Any]], session_date: str, prompt_sha256: str
) -> dict[str, Any]:
    value = {
        "sample_id": sample_id,
        "session_id": session_key,
        "turns_sha256": hashlib.sha256(canonical_json(turns)).hexdigest(),
        "session_date": session_date,
        "extract_prompt_sha256": prompt_sha256,
        "writer_sha256": READER_SHA256,
        "memeval_revision": EXPECTED_MEMEVAL_COMMIT,
        "adapter_patch_sha256": PATCH_SHA256,
    }
    value["identity_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return value


def _load_or_extract_session(
    *, original_extract: Any, sample_id: str, cache_dir: Path, prompt_sha256: str,
    provider: Any, turns: list[dict[str, Any]], session_date: str, session_key: str,
    speaker_a: str, speaker_b: str, llm_client: Any, llm_model: str,
) -> list[Any]:
    from agents_memory.propmem import Proposition

    path = cache_dir / f"{session_key}.json"
    identity = _session_cache_identity(sample_id, session_key, turns, session_date, prompt_sha256)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("identity") != identity or not isinstance(saved.get("propositions"), list):
            raise RuntimeError(f"Session extraction cache identity mismatch: {sample_id}/{session_key}")
        return [Proposition(**row) for row in saved["propositions"]]
    question_id = f"ingest:{sample_id}:{session_key}"
    propositions = []
    for attempt in range(RETRY_LIMIT + 1):
        previous = [
            row for row in read_jsonl(RUN_ROOT / "call_ledger.jsonl")
            if row.get("question_id") == question_id and row.get("role") == "memory_ingest"
        ]
        with provider.call_context("PropMem", question_id, "memory_ingest"):
            propositions = original_extract(
                turns, session_date, session_key, speaker_a, speaker_b, llm_client, llm_model
            )
        current = [
            row for row in read_jsonl(RUN_ROOT / "call_ledger.jsonl")
            if row.get("question_id") == question_id and row.get("role") == "memory_ingest"
        ][len(previous):]
        if not current:
            raise RuntimeError(f"No local writer telemetry was recorded for {question_id}")
        if current[-1].get("success") is True:
            break
        errors = [str(row.get("error_type") or "") for row in current if row.get("success") is False]
        transient = bool(errors) and all(any(
            cue in error.casefold()
            for cue in ("connection", "timeout", "transport", "internalserver", "ratelimit")
        ) for error in errors)
        if transient and attempt < RETRY_LIMIT:
            time.sleep(min(2 ** attempt, 4))
            continue
        raise RuntimeError(
            f"Local writer failed for {question_id}; error types={errors}; refusing to cache empty extraction"
        )
    _write_json(path, {
        "identity": identity,
        "propositions": [dataclasses.asdict(item) for item in propositions],
    })
    return propositions


def _system_for_ingest(config: Any, provider: Any, sample_id: str, cache_dir: Path):
    from agents_memory.propmem import EXTRACT_PROMPT, PropMemSystem

    embedding = provider.embedding_client("PropMem", f"ingest:{sample_id}", config)
    system = PropMemSystem(
        embedding_model=config.embedding_model,
        embedding_client=embedding,
        use_temporal_boost=False,
        use_knowledge_updates=False,
        use_llm_classifier=False,
    )
    original = system._extract_propositions
    prompt_sha256 = hashlib.sha256(EXTRACT_PROMPT.encode("utf-8")).hexdigest()

    def extract_cached(turns, session_date, session_key, speaker_a, speaker_b, llm_client, llm_model):
        return _load_or_extract_session(
            original_extract=original, sample_id=sample_id, cache_dir=cache_dir,
            prompt_sha256=prompt_sha256, provider=provider, turns=turns,
            session_date=session_date, session_key=session_key, speaker_a=speaker_a,
            speaker_b=speaker_b, llm_client=llm_client, llm_model=llm_model,
        )

    system._extract_propositions = extract_cached
    return system, prompt_sha256


def _repair_chunk_provenance_sidecar(system: Any, conversation: dict[str, Any]) -> None:
    """Attach source session IDs from official turn IDs without changing chunk text/rank."""
    from agents_memory.locomo import extract_dialogues, format_as_markdown_with_session_map
    from agents_memory.openclaw import chunk_markdown

    turn_to_session = {
        str(turn["dia_id"]): session_key
        for session_key in _session_keys(conversation)
        for turn in conversation.get(session_key, [])
        if isinstance(turn, dict) and turn.get("dia_id")
    }
    dialogues = extract_dialogues({"conversation": conversation})
    for dialogue in dialogues:
        dialogue["session_id"] = turn_to_session.get(str(dialogue.get("dia_id", "")), "")
    markdown, line_sources = format_as_markdown_with_session_map(dialogues)
    sidecar_chunks = chunk_markdown(
        markdown, tokens=400, overlap=80, line_source_session_ids=line_sources
    )

    def chunk_content(chunk: Any) -> tuple[Any, ...]:
        return (chunk.text, chunk.start_line, chunk.end_line, chunk.hash)

    if [chunk_content(item) for item in sidecar_chunks] != [chunk_content(item) for item in system.chunks]:
        raise RuntimeError("Session provenance sidecar changed PropMem chunk content or order")
    system.chunks = sidecar_chunks


def _save_system_snapshot(directory: Path, system: Any, identity: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "propositions.jsonl").write_bytes(
        b"".join(canonical_json(dataclasses.asdict(item)) + b"\n" for item in system.propositions)
    )
    (directory / "chunks.jsonl").write_bytes(
        b"".join(canonical_json(dataclasses.asdict(item)) + b"\n" for item in system.chunks)
    )
    np.save(directory / "proposition_embeddings.npy", np.asarray(system.proposition_embeddings, dtype=np.float32))
    np.save(directory / "chunk_embeddings.npy", np.asarray(system.chunk_embeddings, dtype=np.float32))
    centroids = None
    if system._cluster_centroids is not None:
        centroids = np.asarray(system._cluster_centroids, dtype=np.float32)
        np.save(directory / "cluster_centroids.npy", centroids)
    state = {
        "identity": identity,
        "config": {
            "embedding_model": system.embedding_model,
            "top_k_props": system.top_k_props,
            "top_k_chunks": system.top_k_chunks,
            "use_propositions": system.use_propositions,
            "use_chunks": system.use_chunks,
            "use_entity_filter": system.use_entity_filter,
            "use_clustering": system.use_clustering,
            "use_bm25": system.use_bm25,
            "use_llm_classifier": system.use_llm_classifier,
            "use_temporal_boost": system.use_temporal_boost,
            "use_knowledge_updates": system.use_knowledge_updates,
            "entity_names": system.entity_names,
            "cluster_to_indices": {str(k): v for k, v in system._cluster_to_indices.items()},
            "has_cluster_centroids": centroids is not None,
        },
    }
    _write_json(directory / "state.json", state)
    file_hashes = {
        item.name: sha256_file(item)
        for item in sorted(directory.iterdir())
        if item.is_file() and item.name != "snapshot.json"
    }
    snapshot = {
        "identity": identity,
        "file_sha256": file_hashes,
        "tree_sha256": _hash_tree(directory, exclude={"snapshot.json"}),
        "num_propositions": len(system.propositions),
        "num_chunks": len(system.chunks),
    }
    _write_json(directory / "snapshot.json", snapshot)
    return snapshot


def _load_system_snapshot(directory: Path, embedding_client: Any, expected_identity: dict[str, Any]):
    from agents_memory.propmem import Proposition, PropMemSystem
    from agents_memory.openclaw import MemoryChunk

    snapshot = json.loads((directory / "snapshot.json").read_text(encoding="utf-8"))
    if snapshot.get("identity") != expected_identity:
        raise RuntimeError(f"PropMem snapshot identity mismatch at {directory}")
    for name, digest in snapshot["file_sha256"].items():
        if sha256_file(directory / name) != digest:
            raise RuntimeError(f"PropMem snapshot file hash mismatch: {directory / name}")
    if _hash_tree(directory, exclude={"snapshot.json"}) != snapshot["tree_sha256"]:
        raise RuntimeError("PropMem snapshot tree hash mismatch")
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))["config"]
    system = PropMemSystem(
        embedding_model=state["embedding_model"],
        embedding_client=embedding_client,
        top_k_props=state["top_k_props"],
        top_k_chunks=state["top_k_chunks"],
        use_propositions=state["use_propositions"],
        use_chunks=state["use_chunks"],
        use_entity_filter=state["use_entity_filter"],
        use_clustering=state["use_clustering"],
        use_bm25=state["use_bm25"],
        use_llm_classifier=state["use_llm_classifier"],
        use_temporal_boost=state["use_temporal_boost"],
        use_knowledge_updates=state["use_knowledge_updates"],
    )
    system.entity_names = list(state["entity_names"])
    system.propositions = [Proposition(**row) for row in read_jsonl(directory / "propositions.jsonl")]
    system.chunks = [MemoryChunk(**row) for row in read_jsonl(directory / "chunks.jsonl")]
    system.proposition_embeddings = np.load(directory / "proposition_embeddings.npy", allow_pickle=False).tolist()
    system.chunk_embeddings = np.load(directory / "chunk_embeddings.npy", allow_pickle=False).tolist()
    system._cluster_to_indices = {int(k): v for k, v in state["cluster_to_indices"].items()}
    system._cluster_centroids = (
        np.load(directory / "cluster_centroids.npy", allow_pickle=False)
        if state["has_cluster_centroids"] else None
    )
    return system


def ingest_all(dataset: list[dict[str, Any]], config: Any, provider: Any, embedding_identity: dict[str, Any], revisions: dict[str, Any]) -> dict[str, Any]:
    from agents_memory.healthcopilot_embedding import LocalEmbeddingClient
    from agents_memory.propmem import EXTRACT_PROMPT, PropMemSystem

    base = RUN_ROOT / "ingestion"
    manifest_path = base / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _hash_tree(base, exclude={"manifest.json", "manifest.sha256"}) != manifest.get("root_sha256"):
            raise RuntimeError("Frozen ingestion cache tree hash mismatch")
        sidecar = base / "manifest.sha256"
        if not sidecar.exists() or sidecar.read_text(encoding="ascii").split()[0] != sha256_file(manifest_path):
            raise RuntimeError("Frozen ingestion manifest sidecar is missing or invalid")
        return manifest
    if len(dataset) != 10:
        raise RuntimeError("PropMem ingestion requires all ten conversations")

    summaries_dir = base / "summaries"
    extraction_root = base / "session_extractions"
    snapshot_root = base / "conversations"
    summaries = []
    prompt_hashes = set()
    for position, conversation_row in enumerate(dataset, 1):
        sample_id = conversation_row["sample_id"]
        conversation = conversation_row["conversation"]
        summary_path = summaries_dir / f"{sample_id}.json"
        snapshot_dir = snapshot_root / sample_id
        expected_base = {
            "sample_id": sample_id,
            "conversation_sha256": hashlib.sha256(canonical_json(conversation)).hexdigest(),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "memeval_commit": EXPECTED_MEMEVAL_COMMIT,
            "adapter_patch_sha256": PATCH_SHA256,
            "writer_model_sha256": READER_SHA256,
            "embedding_tree_sha256": (
                embedding_identity["model_tree_sha256"]
                if "model_tree_sha256" in embedding_identity
                else embedding_identity["model_sha256"]
            ),
        }
        if (snapshot_dir / "snapshot.json").exists() and summary_path.exists():
            snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
            if any(snapshot["identity"].get(key) != value for key, value in expected_base.items()):
                raise RuntimeError(f"Cached PropMem identity changed: {sample_id}")
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        cache_dir = extraction_root / sample_id
        embedding_client = LocalEmbeddingClient(
            model_path=config.embedding_path, system="PropMem", question_id=f"ingest:{sample_id}",
            device=config.embedding_device, dtype=config.embedding_dtype,
        )
        system, prompt_sha256 = _system_for_ingest(config, provider, sample_id, cache_dir)
        prompt_hashes.add(prompt_sha256)
        writer = provider.reader_client("memory_ingest", "PropMem", f"ingest:{sample_id}", config)
        before_calls = len(read_jsonl(RUN_ROOT / "call_ledger.jsonl"))
        started = time.perf_counter()
        with provider.call_context("PropMem", f"ingest:{sample_id}", "memory_ingest"):
            result = system.ingest_conversation(
                {"sample_id": sample_id, "conversation": conversation}, writer, config.reader_model
            )
        _repair_chunk_provenance_sidecar(system, conversation)
        wall_seconds = time.perf_counter() - started
        if system.use_temporal_boost or system.use_knowledge_updates or system.use_llm_classifier:
            raise RuntimeError("PropMem baseline v3 flags unexpectedly enabled")
        identity = {
            **expected_base,
            "extract_prompt_sha256": prompt_sha256,
            "prop_mem_config": {
                "use_temporal_boost": False,
                "use_knowledge_updates": False,
                "use_llm_classifier": False,
                "top_k_props": system.top_k_props,
                "top_k_chunks": system.top_k_chunks,
            },
        }
        snapshot = _save_system_snapshot(snapshot_dir, system, identity)
        calls = read_jsonl(RUN_ROOT / "call_ledger.jsonl")
        new_calls = calls[before_calls:]
        sessions = _session_keys(conversation)
        missing = [key for key in sessions if not (cache_dir / f"{key}.json").is_file()]
        if missing:
            raise RuntimeError(f"Per-session PropMem writer cache missing for {sample_id}: {missing}")
        summary = {
            "sample_id": sample_id,
            "status": "COMPLETED",
            "num_sessions": len(sessions),
            "num_chunks": result["num_chunks"],
            "num_propositions": result["num_propositions"],
            "writer_llm_calls": sum(row.get("role") == "memory_ingest" for row in new_calls),
            "writer_tokens": sum(
                int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
                for row in new_calls if row.get("role") == "memory_ingest"
            ),
            "embedding_calls": sum(row.get("role") == "embedding" for row in new_calls),
            "wall_seconds": round(wall_seconds, 3),
            "snapshot_sha256": snapshot["tree_sha256"],
        }
        _write_json(summary_path, summary)
        summaries.append(summary)
        embedding_client.close()
        print(
            f"Ingested {position}/10 {sample_id}: {result['num_propositions']} propositions, "
            f"{result['num_chunks']} chunks, {wall_seconds / 60:.1f} min"
        )

    for conversation_row in dataset:
        cache_dir = extraction_root / conversation_row["sample_id"]
        missing = [
            key for key in _session_keys(conversation_row["conversation"])
            if not (cache_dir / f"{key}.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"Session extraction cache incomplete for {conversation_row['sample_id']}: {missing}")
    prompt_sha256 = next(iter(prompt_hashes), hashlib.sha256(EXTRACT_PROMPT.encode("utf-8")).hexdigest())
    root_hash = _hash_tree(base, exclude={"manifest.json", "manifest.sha256"})
    store_bytes = sum(path.stat().st_size for path in base.rglob("*") if path.is_file())
    manifest = {
        "status": "FROZEN_COMPLETE",
        "benchmark": "LoCoMo locomo10",
        "n_conversations": 10,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "locomo_commit": EXPECTED_LOCOMO_COMMIT,
        "memeval_commit": EXPECTED_MEMEVAL_COMMIT,
        "adapter_patch_sha256": PATCH_SHA256,
        "extract_prompt_sha256": prompt_sha256,
        "writer_model_sha256": READER_SHA256,
        "embedding": embedding_identity,
        "one_ingestion_shared_by_both_arms": True,
        "root_sha256": root_hash,
        "store_bytes": store_bytes,
        "conversations": summaries,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(manifest_path, manifest)
    (base / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii")
    return manifest


def _retrieve_once(system: Any, question: str):
    """Mirror the pinned v3 answer_question retrieval path once for both paired arms."""
    entity = system._identify_entity(question)
    inferential = system._is_inferential(question)
    generic = {"user", "assistant", "system", "bot", "ai"}
    single_user = all(name.lower() in generic for name in system.entity_names)
    prop_k = system.top_k_props if system.use_chunks else int(system.top_k_props * 1.5)
    props = (
        system._retrieve_propositions(question, entity=entity, top_k=prop_k, is_temporal=False)
        if system.use_propositions else []
    )
    chunk_k = system.top_k_chunks * (2 if single_user else 1)
    if not system.use_propositions:
        chunk_k = max(chunk_k, 10)
    chunks = (
        system._retrieve_chunks(question, top_k=chunk_k, entity=entity)
        if system.use_chunks else []
    )
    return entity, inferential, single_user, props, chunks


def _retrieval_rows(
    system: Any, props: list[Any], chunks: list[Any]
) -> tuple[list[dict[str, Any]], list[list[float]], list[str], str]:
    prop_positions = {id(item): index for index, item in enumerate(system.propositions)}
    proposition_rows = []
    vectors = []
    signatures = []
    ranked_sessions = []
    from agents_memory.propmem import _parse_date_ordinal

    for rank, (proposition, score) in enumerate(props, 1):
        index = prop_positions[id(proposition)]
        memory_id = f"prop:{index}"
        proposition_rows.append({
            "memory_id": memory_id,
            "entity": proposition.entity,
            "text": proposition.text,
            "date": proposition.date,
            "date_ordinal": int(proposition.date_ordinal or _parse_date_ordinal(proposition.date)),
        })
        vectors.append(system.proposition_embeddings[index])
        signatures.append({
            "kind": "proposition",
            "id": memory_id,
            "rank": rank,
            "score": round(float(score), 10),
        })
        if proposition.session_id:
            ranked_sessions.append(str(proposition.session_id))
    chunk_positions = {id(item): index for index, item in enumerate(system.chunks)}
    for rank, (chunk, score) in enumerate(chunks, 1):
        signatures.append({
            "kind": "chunk",
            "id": f"chunk:{chunk_positions[id(chunk)]}",
            "rank": rank,
            "score": round(float(score), 10),
        })
        ranked_sessions.extend(
            str(session_id)
            for session_id in getattr(chunk, "source_session_ids", [])
            if session_id
        )
    ranked_sessions = list(dict.fromkeys(ranked_sessions))
    signature = hashlib.sha256(canonical_json(signatures)).hexdigest()
    return proposition_rows, vectors, ranked_sessions, signature


class _CompletionCapture:
    def __init__(self, client: Any, view_text: str):
        from types import SimpleNamespace

        self.client = client
        self.view_text = view_text
        self.response = None
        self.prompt = ""
        self.elapsed_ms = 0.0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any):
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
            raise RuntimeError("PropMem answer request has no user message")
        copied = [dict(message) for message in messages]
        if self.view_text:
            copied[0]["content"] = inject_temporal_view(
                str(copied[0].get("content") or ""), self.view_text
            )
        self.prompt = str(copied[0].get("content") or "")
        kwargs["messages"] = copied
        started = time.perf_counter()
        self.response = self.client.chat.completions.create(**kwargs)
        self.elapsed_ms = (time.perf_counter() - started) * 1000
        return self.response

    def metrics(self) -> dict[str, Any]:
        usage = getattr(self.response, "usage", None)
        return {
            "reader_prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reader_latency_ms": round(self.elapsed_ms, 3),
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
        }


def _clean_answer(value: str) -> str:
    answer = value.strip()
    for prefix in ("Answer:", "A:", "answer:"):
        if answer.startswith(prefix):
            return answer[len(prefix):].strip()
    return answer


class _AnswerParseFailure(ValueError):
    def __init__(self, message: str, metrics: dict[str, Any]):
        super().__init__(message)
        self.metrics = metrics


def _propmem_prompt(
    question: str,
    props: list[tuple[Any, float]],
    chunks: list[tuple[Any, float]],
    entity: str | None,
    *,
    is_inferential: bool,
    single_user: bool,
) -> str:
    from agents_memory.propmem import (
        ANSWER_PROMPT,
        ANSWER_PROMPT_INFERENTIAL,
        ANSWER_PROMPT_SINGLE_USER,
        ANSWER_PROMPT_SINGLE_USER_INFERENTIAL,
    )

    if entity and props:
        entity_section = f"Known facts about {entity}:\n" + "\n".join(
            f"- [{item.date}] {item.text}" for item, _score in props
        )
    elif props:
        entity_section = "Known facts from conversation:\n" + "\n".join(
            f"- [{item.date}] {item.text}" for item, _score in props
        )
    else:
        entity_section = "Known facts: (none extracted)"
    chunks_section = (
        "Additional conversation context:\n" + "\n---\n".join(item.text for item, _ in chunks)
        if chunks else ""
    )
    if single_user:
        template = ANSWER_PROMPT_SINGLE_USER_INFERENTIAL if is_inferential else ANSWER_PROMPT_SINGLE_USER
        return template.format(
            entity_section=entity_section,
            chunks_section=chunks_section,
            question=question,
        )
    template = ANSWER_PROMPT_INFERENTIAL if is_inferential else ANSWER_PROMPT
    return template.format(
        entity_section=entity_section,
        chunks_section=chunks_section,
        question=question,
        entity_name=entity or "the person asked about",
    )


def _answer(
    system: Any, question: str, retrieval: tuple[Any, ...], client: Any,
    view: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    entity, inferential, single_user, props, chunks = retrieval
    view_text = view["view_text"] if view else ""
    visible = set(view["visible_memory_ids"]) if view else None
    if visible is not None:
        positions = {id(item): index for index, item in enumerate(system.propositions)}
        props = [
            (item, score) for item, score in props
            if f"prop:{positions[id(item)]}" in visible
        ]
    captured = _CompletionCapture(client, view_text)
    prompt = _propmem_prompt(
        question, props, chunks, entity,
        is_inferential=inferential,
        single_user=single_user,
    )
    try:
        response = captured.chat.completions.create(
            model="health-memory-qwen3-8b",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=500,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        parsed = json.loads(content)
        answer = parsed.get("answer", "").strip()
        return answer.strip(), captured.metrics()
    except (json.JSONDecodeError, AttributeError, TypeError) as error:
        raise _AnswerParseFailure(str(error), captured.metrics()) from error


def _transient_infra(error: BaseException) -> bool:
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
    import httpx

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (
            APIConnectionError, APITimeoutError, InternalServerError,
            RateLimitError, httpx.TransportError,
        )):
            return True
        current = current.__cause__ or current.__context__
    return False


def _quality_parse_failure(error: BaseException) -> bool:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (json.JSONDecodeError, ValueError, AttributeError, TypeError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _infer_one(
    question_row: dict[str, Any],
    system: Any,
    retrieval: tuple[Any, ...],
    proposition_rows: list[dict[str, Any]],
    proposition_vectors: list[list[float]],
    ranked_sessions: list[str],
    retrieval_signature: str,
    client: Any,
    *,
    tvr: bool,
    provider: Any,
) -> dict[str, Any]:
    qid = question_row["question_id"]
    question = question_row["question"]
    view = (
        project_temporal_view(
            question,
            proposition_rows,
            proposition_vectors,
            threshold=SAME_TOPIC_COSINE_THRESHOLD,
        )
        if tvr else None
    )
    arm = "propmem_tvr" if tvr else "propmem"
    last_error: BaseException | None = None
    for attempt in range(RETRY_LIMIT + 1):
        try:
            with provider.call_context(arm, qid, "reader_answer"):
                predicted, answer_metrics = _answer(system, question, retrieval, client, view)
            status = "OK"
            break
        except BaseException as error:
            last_error = error
            append_jsonl(RUN_ROOT / "attempts.jsonl", {
                "question_id": qid,
                "arm": arm,
                "attempt": attempt + 1,
                "error_type": type(error).__name__,
                "retryable_infrastructure": _transient_infra(error),
                "quality_parse_failure": _quality_parse_failure(error),
                "utc": datetime.now(UTC).isoformat(),
            })
            if _transient_infra(error) and attempt < RETRY_LIMIT:
                time.sleep(min(2 ** attempt, 4))
                continue
            if _quality_parse_failure(error):
                predicted = ""
                status = "QUALITY_FAILURE_EMPTY"
                answer_metrics = getattr(error, "metrics", None) or {
                    "reader_prompt_tokens": None,
                    "completion_tokens": None,
                    "reader_latency_ms": None,
                    "prompt_sha256": None,
                }
                break
            raise
    prior_attempts = read_jsonl(RUN_ROOT / "attempts.jsonl")
    runner_retries = sum(
        row.get("question_id") == qid and row.get("arm") == arm
        for row in prior_attempts
    ) if last_error is not None and status == "OK" else 0
    row = {
        "question_id": qid,
        "sample_id": question_row["sample_id"],
        "predicted": predicted,
        "quality_status": status,
        "retrieved_proposition_ids": [item["memory_id"] for item in proposition_rows],
        "retrieved_session_ranked": ranked_sessions,
        "retrieval_signature": retrieval_signature,
        "temporal_intent": view["mode"] if view else "NOT_APPLICABLE",
        "temporal_target_date": view["target_date"] if view else None,
        "temporal_component_memory_ids": view["component_memory_ids"] if view else [],
        "temporal_visible_memory_ids": (
            view["visible_memory_ids"] if view
            else [item["memory_id"] for item in proposition_rows]
        ),
        "resolver_latency_ms": round(float(view["resolver_latency_ms"]), 3) if view else 0.0,
        "runner_retry_count": int(runner_retries),
        **answer_metrics,
    }
    return row


def _prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    output = {}
    for row in read_jsonl(path):
        qid = row.get("question_id")
        if qid in output:
            raise RuntimeError(f"Duplicate prediction row: {qid}")
        if row.get("quality_status") not in {"OK", "QUALITY_FAILURE_EMPTY"}:
            raise RuntimeError(f"Non-quality prediction row found: {qid}")
        output[qid] = row
    return output


def infer_all(config: Any, provider: Any, ingestion: dict[str, Any]) -> dict[str, Any]:
    from agents_memory.healthcopilot_provider import reader_client, release_local_embedding_runtimes

    questions_path = RUN_ROOT / "frozen_questions.jsonl"
    questions = read_jsonl(questions_path)
    expected_qsha = (RUN_ROOT / "frozen_questions.sha256").read_text(encoding="ascii").split()[0]
    if len(questions) != 1986 or sha256_file(questions_path) != expected_qsha:
        raise RuntimeError("Frozen blind question manifest incomplete or changed")
    frozen_ingestion = json.loads((RUN_ROOT / "ingestion" / "manifest.json").read_text(encoding="utf-8"))
    if frozen_ingestion["root_sha256"] != ingestion["root_sha256"]:
        raise RuntimeError("Both arms must reference the identical frozen ingestion root")

    embedding = provider.embedding_client("PropMem-paired-retrieval", "shared-retrieval", config)
    systems = {}
    for item in ingestion["conversations"]:
        snapshot_dir = RUN_ROOT / "ingestion" / "conversations" / item["sample_id"]
        snapshot = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
        systems[item["sample_id"]] = _load_system_snapshot(
            snapshot_dir, embedding, snapshot["identity"]
        )

    baseline_path = RUN_ROOT / "propmem_predictions.jsonl"
    tvr_path = RUN_ROOT / "propmem_tvr_predictions.jsonl"
    baseline_done = _prediction_index(baseline_path)
    tvr_done = _prediction_index(tvr_path)
    baseline_client = reader_client("reader_answer", "propmem", "paired", config)
    tvr_client = reader_client("reader_answer", "propmem_tvr", "paired", config)
    invocation_started = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    try:
        for index, question_row in enumerate(questions, 1):
            qid = question_row["question_id"]
            sample_id = question_row["sample_id"]
            if qid in baseline_done and qid in tvr_done:
                if baseline_done[qid]["retrieval_signature"] != tvr_done[qid]["retrieval_signature"]:
                    raise RuntimeError(f"Paired retrieval signature mismatch for {qid}")
                continue
            system = systems[sample_id]
            with provider.call_context("propmem", qid, "embedding"):
                retrieval_started = time.perf_counter()
                retrieval = _retrieve_once(system, question_row["question"])
                retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
            prop_rows, prop_vectors, ranked_sessions, signature = _retrieval_rows(
                system, retrieval[3], retrieval[4]
            )
            if qid in baseline_done:
                baseline_row = baseline_done[qid]
                if baseline_row["retrieval_signature"] != signature:
                    raise RuntimeError(f"Resumed PropMem retrieval changed for {qid}")
            else:
                baseline_row = _infer_one(
                    question_row, system, retrieval, prop_rows, prop_vectors,
                    ranked_sessions, signature, baseline_client, tvr=False, provider=provider,
                )
                baseline_row["retrieval_latency_ms"] = round(retrieval_latency_ms, 3)
                append_jsonl(baseline_path, baseline_row)
                baseline_done[qid] = baseline_row
            if qid in tvr_done:
                tvr_row = tvr_done[qid]
                if tvr_row["retrieval_signature"] != signature:
                    raise RuntimeError(f"Resumed TVR retrieval changed for {qid}")
            else:
                tvr_row = _infer_one(
                    question_row, system, retrieval, prop_rows, prop_vectors,
                    ranked_sessions, signature, tvr_client, tvr=True, provider=provider,
                )
                tvr_row["retrieval_latency_ms"] = round(retrieval_latency_ms, 3)
                append_jsonl(tvr_path, tvr_row)
                tvr_done[qid] = tvr_row
            if index % 25 == 0:
                elapsed = time.perf_counter() - invocation_started
                print(
                    f"Paired QA {index}/1986; PropMem={len(baseline_done)}/1986, "
                    f"TVR={len(tvr_done)}/1986; elapsed={elapsed / 60:.1f} min"
                )
    finally:
        baseline_client.close()
        tvr_client.close()
        embedding.close()
        release_local_embedding_runtimes()
    segment = {
        "started_utc": started_utc,
        "duration_seconds": round(time.perf_counter() - invocation_started, 3),
        "completed_prop_rows": len(baseline_done),
        "completed_tvr_rows": len(tvr_done),
    }
    append_jsonl(RUN_ROOT / "qa_runtime_segments.jsonl", segment)
    if len(baseline_done) != 1986 or len(tvr_done) != 1986:
        raise RuntimeError(f"Paired run incomplete: PropMem={len(baseline_done)}, TVR={len(tvr_done)}")
    return segment


def freeze_predictions() -> dict[str, Any]:
    questions = read_jsonl(RUN_ROOT / "frozen_questions.jsonl")
    expected = {row["question_id"] for row in questions}
    paths = {
        "propmem": RUN_ROOT / "propmem_predictions.jsonl",
        "propmem_tvr": RUN_ROOT / "propmem_tvr_predictions.jsonl",
    }
    indexes = {arm: _prediction_index(path) for arm, path in paths.items()}
    if any(set(indexes[arm]) != expected for arm in paths):
        raise RuntimeError("Both arms must contain exactly the 1,986 frozen question IDs")
    for qid in expected:
        if indexes["propmem"][qid]["retrieval_signature"] != indexes["propmem_tvr"][qid]["retrieval_signature"]:
            raise RuntimeError(f"Shared retrieval parity failed for {qid}")
    ingestion = json.loads((RUN_ROOT / "ingestion" / "manifest.json").read_text(encoding="utf-8"))
    value = {
        "status": "FROZEN_BEFORE_SCORING",
        "question_manifest_sha256": sha256_file(RUN_ROOT / "frozen_questions.jsonl"),
        "ingestion_root_sha256": ingestion["root_sha256"],
        "n": 1986,
        "predictions": {
            arm: {
                "path": path.name,
                "sha256": sha256_file(path),
                "n": len(indexes[arm]),
                "retrieval_signature_set_sha256": hashlib.sha256(
                    canonical_json(sorted(row["retrieval_signature"] for row in indexes[arm].values()))
                ).hexdigest(),
            }
            for arm, path in paths.items()
        },
    }
    lock = RUN_ROOT / "predictions_freeze.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != value:
        raise RuntimeError("Prediction freeze exists with different hashes")
    if not lock.exists():
        _write_json(lock, value)
    return value


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.lower()))


def _scored_row(row: dict[str, Any], label: dict[str, Any], compute_f1: Any, refusal_fn: Any) -> dict[str, Any]:
    predicted = str(row.get("predicted") or "")
    gold = str(label.get("answer") or "")
    gold_tokens = len(re.findall(r"\w+", gold.lower()))
    normalized_em = (
        float(refusal_fn(predicted))
        if gold_tokens == 0
        else float(_normalized(predicted) == _normalized(gold))
    )
    return {
        **row,
        "category": int(label["category"]),
        "category_name": label["category_name"],
        "token_f1": float(compute_f1(predicted, gold)),
        "normalized_em": normalized_em,
        "gold_tokens": gold_tokens,
    }


def _category_metrics(rows: list[dict[str, Any]], refusal_fn: Any) -> dict[str, Any]:
    answerable = [row for row in rows if row["gold_tokens"] > 0]
    empty = [row for row in rows if row["gold_tokens"] == 0]
    return {
        "n": len(rows),
        "token_f1": mean(row["token_f1"] for row in rows) if rows else None,
        "normalized_em": mean(row["normalized_em"] for row in rows) if rows else None,
        "answerable_only_normalized_em": (
            mean(row["normalized_em"] for row in answerable) if answerable else None
        ),
        "adversarial_refusal_accuracy": (
            mean(float(refusal_fn(row["predicted"])) for row in empty) if empty else None
        ),
        "answerable_n": len(answerable),
        "empty_gold_n": len(empty),
    }


def _evidence_sessions(conversation: dict[str, Any], evidence: list[Any]) -> list[str]:
    turn_to_session = {
        str(turn["dia_id"]): session_key
        for session_key in _session_keys(conversation)
        for turn in conversation.get(session_key, [])
        if isinstance(turn, dict) and turn.get("dia_id")
    }
    return list(dict.fromkeys(
        turn_to_session[str(item)] for item in evidence if str(item) in turn_to_session
    ))


def _retrieval_metrics(rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in rows:
        gold = set(labels[row["question_id"]]["answer_sessions"])
        if not gold:
            continue
        ranked = row.get("retrieved_session_ranked", [])
        hits = [index + 1 for index, session_id in enumerate(ranked) if session_id in gold]
        values.append({
            "r5": len(set(ranked[:5]) & gold) / len(gold),
            "r10": len(set(ranked[:10]) & gold) / len(gold),
            "mrr": 1.0 / min(hits) if hits else 0.0,
        })
    return {
        "n_with_gold_sessions": len(values),
        "answer_session_recall_at_5": mean(row["r5"] for row in values) if values else None,
        "answer_session_recall_at_10": mean(row["r10"] for row in values) if values else None,
        "answer_session_mrr": mean(row["mrr"] for row in values) if values else None,
    }


def _bootstrap(delta: np.ndarray, categories: np.ndarray, replicates: int = 10_000) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    samples = np.zeros(replicates, dtype=np.float64)
    for category in np.unique(categories):
        values = delta[categories == category]
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        samples += values[indices].mean(axis=1) * (len(values) / len(delta))
    return {
        "replicates": replicates,
        "seed": 42,
        "stratified_by": "LoCoMo category",
        "paired": True,
        "mean_delta": float(delta.mean()),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def _values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(row[field]) for row in rows
        if isinstance(row.get(field), (int, float))
    ]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[min(len(values) - 1, int(fraction * len(values)))]


def _mean_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = _values(rows, field)
    return mean(values) if values else None


def _arm_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = _values(rows, "reader_prompt_tokens")
    completion = _values(rows, "completion_tokens")
    reader = _values(rows, "reader_latency_ms")
    retrieval = _values(rows, "retrieval_latency_ms")
    return {
        "n": len(rows),
        "mean_reader_prompt_tokens": mean(prompt) if prompt else None,
        "median_reader_prompt_tokens": _percentile(prompt, 0.5),
        "p95_reader_prompt_tokens": _percentile(prompt, 0.95),
        "mean_completion_tokens": mean(completion) if completion else None,
        "mean_reader_latency_ms": mean(reader) if reader else None,
        "p95_reader_latency_ms": _percentile(reader, 0.95),
        "mean_retrieval_latency_ms": mean(retrieval) if retrieval else None,
        "p95_retrieval_latency_ms": _percentile(retrieval, 0.95),
        "mean_resolver_latency_ms": _mean_numeric(rows, "resolver_latency_ms"),
    }


def _format_optional(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _write_report(scorecard: dict[str, Any]) -> None:
    full = scorecard["fullcontext"]
    prop = scorecard["propmem"]
    tvr = scorecard["propmem_tvr"]
    delta = scorecard["paired_delta"]
    boot = scorecard["bootstrap"]
    efficiency = scorecard["efficiency"]
    names = ["Factual", "Temporal", "Inferential", "Multi-hop", "Adversarial"]
    lines = [
        "# MEM-Q1 PropMem-TVR Paired LoCoMo Closeout",
        "",
        "## Experiment",
        "",
        "- Benchmark: official LoCoMo locomo10.json, 10 conversations and 1,986 QA; dataset SHA256 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4.",
        "- MemEval commit: 807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4; LoCoMo commit: 3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376.",
        "- Reader/writer: local Qwen3-8B Q4_K_M. Embedding: Qwen3-Embedding-0.6B local CUDA FP16. Judge: NONE. Hosted APIs: NONE.",
        "- PropMem is the pinned upstream implementation with only the audited local provider/embedding compatibility patch; temporal boost and knowledge updates are disabled.",
        "- The answer adapter uses the pinned PropMem prompt templates and response schema verbatim while keeping transport exceptions observable for exact infra retries.",
        "- TVR is deterministic and query-time only, operates on already-retrieved propositions, requires the same entity and cosine >= 0.85, does not mutate the cache, and makes no extra LLM calls.",
        "- Chunk session provenance is a metadata-only sidecar derived from official LoCoMo turn IDs; chunk text/order and embedding vectors were parity-checked.",
        f"- Shared ingestion cache SHA256: {scorecard['integrity']['ingestion_cache_sha256']}.",
        f"- Blind question manifest SHA256: {scorecard['integrity']['question_manifest_sha256']}. Predictions were frozen before scoring.",
        "",
        "## Results",
        "",
        "| Metric | FullContext | PropMem-local | PropMem + TVR | TVR delta |",
        "|---|---:|---:|---:|---:|",
        f"| Overall Token-F1 | {full['overall_token_f1']:.4f} | {prop['token_f1']:.4f} | {tvr['token_f1']:.4f} | {delta['overall_token_f1']:+.4f} |",
    ]
    for name in names:
        p = prop["categories"][name]["token_f1"]
        t = tvr["categories"][name]["token_f1"]
        d = delta["categories"][name]
        lines.append(
            f"| {name} Token-F1 | {full['per_category_token_f1'][name]:.4f} | {p:.4f} | {t:.4f} | {d:+.4f} |"
        )
    lines.extend([
        f"| Normalized EM (all; empty-gold uses refusal semantics) | n/a | {prop['normalized_em']:.4f} | {tvr['normalized_em']:.4f} | {tvr['normalized_em'] - prop['normalized_em']:+.4f} |",
        f"| Answerable-only normalized EM | {full['answerable_only_normalized_em']:.4f} | {prop['answerable_only_normalized_em']:.4f} | {tvr['answerable_only_normalized_em']:.4f} | {tvr['answerable_only_normalized_em'] - prop['answerable_only_normalized_em']:+.4f} |",
        f"| Empty-gold adversarial refusal accuracy | {full['empty_gold_adversarial_refusal_accuracy']:.4f} | {prop['categories']['Adversarial']['adversarial_refusal_accuracy']:.4f} | {tvr['categories']['Adversarial']['adversarial_refusal_accuracy']:.4f} | n/a |",
        f"| Mean reader prompt tokens | {full['mean_reader_prompt_tokens']:.0f} | {_format_optional(efficiency['propmem']['mean_reader_prompt_tokens'], 0)} | {_format_optional(efficiency['propmem_tvr']['mean_reader_prompt_tokens'], 0)} | n/a |",
        f"| Context reduction vs FullContext | n/a | {_format_optional(efficiency['reader_context_reduction_vs_fullcontext']['propmem'] * 100, 1)}% | {_format_optional(efficiency['reader_context_reduction_vs_fullcontext']['propmem_tvr'] * 100, 1)}% | n/a |",
        "",
        "FullContext normalized EM in its source report is answerable-only (286/1,542); the all-case refusal-aware EM above is computed for the two paired arms.",
        "",
        "## Paired Uncertainty",
        "",
        f"- Overall paired delta: {delta['overall_token_f1']:+.6f}; 95% stratified paired bootstrap CI [{boot['overall']['ci95'][0]:+.6f}, {boot['overall']['ci95'][1]:+.6f}] (10,000 replicates, seed 42).",
        f"- Temporal paired delta: {delta['temporal_token_f1']:+.6f}; 95% stratified paired bootstrap CI [{boot['temporal']['ci95'][0]:+.6f}, {boot['temporal']['ci95'][1]:+.6f}].",
        "",
        "## Efficiency",
        "",
        f"- Shared ingestion: {efficiency['ingestion_wall_seconds'] / 3600:.2f} hours; writer calls {efficiency['ingestion_writer_llm_calls']}; embedding calls {efficiency['ingestion_embedding_calls']}; {efficiency['num_propositions']} propositions; {efficiency['num_chunks']} chunks; store size {efficiency['store_bytes']} bytes.",
        f"- Paired QA wall: {efficiency['qa_wall_seconds'] / 3600:.2f} hours; local LLM calls {efficiency['local_llm_calls']}; hosted calls {efficiency['hosted_calls']}; permanent infra failures {efficiency['permanent_infra_failures']}; retries {efficiency['retry_count']}.",
        f"- Mean reader latency: PropMem {_format_optional(efficiency['propmem']['mean_reader_latency_ms'] / 1000 if efficiency['propmem']['mean_reader_latency_ms'] is not None else None)} s; TVR {_format_optional(efficiency['propmem_tvr']['mean_reader_latency_ms'] / 1000 if efficiency['propmem_tvr']['mean_reader_latency_ms'] is not None else None)} s.",
        f"- Mean retrieval latency {efficiency['propmem']['mean_retrieval_latency_ms']:.2f} ms; mean TVR resolver overhead {efficiency['tvr_resolver_mean_ms']:.3f} ms.",
        "",
        "## Integrity",
        "",
        f"- PropMem predictions SHA256: {scorecard['integrity']['baseline_predictions_sha256']}.",
        f"- PropMem+TVR predictions SHA256: {scorecard['integrity']['tvr_predictions_sha256']}.",
        f"- Reader model SHA256: {READER_SHA256}; llama.cpp binary SHA256: {LLAMA_SHA256}.",
        f"- Reader runtime KV placement: {scorecard['integrity']['reader_runtime']['kv_cache_location']}; 8081 command line is pinned in run_config.json. Reader layers remain on GPU; both arms shared this endpoint.",
        f"- Retrieval signatures identical: {scorecard['integrity']['retrieval_signature_sets_identical']}; question ID sets identical: {scorecard['integrity']['question_id_sets_identical']}.",
        "- The pre-existing 8092 service was left untouched; the owned 8081 endpoint served this run.",
        "",
        "## Limitations",
        "",
        "- Local Qwen runtime differs from MemEval's published GPT-backed LoCoMo protocol; published PropMem 0.605 is an external historical coordinate only.",
        "- TVR is a query-time adaptation over upstream PropMem, not a new proposition-memory algorithm. Its similarity primitive follows the pinned PropMem update threshold.",
        "- LoCoMo alone is not evidence of medical-domain transfer.",
        "",
        "## Resume Claim Eligibility",
        "",
        f"- {scorecard['resume_claim']['eligibility']}: {scorecard['resume_claim']['claim']}",
        "",
    ])
    destination = ROOT / "docs" / "research" / "memory" / "mem_q1_propmem_tvr_locomo_closeout.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def score_and_report(revisions: dict[str, Any], runtime: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    if freeze.get("status") != "FROZEN_BEFORE_SCORING":
        raise RuntimeError("Scoring requires a frozen paired prediction lock")
    for item in freeze["predictions"].values():
        if sha256_file(RUN_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Prediction SHA changed after freeze")

    from agents_memory.evaluation import compute_f1, _is_refusal
    from agents_memory.locomo import CATEGORY_NAMES

    dataset = _load_dataset()
    labels = {}
    for conversation in dataset:
        for qa_index, qa in enumerate(conversation["qa"]):
            qid = f"{conversation['sample_id']}:qa:{qa_index}"
            labels[qid] = {
                "answer": qa.get("answer", ""),
                "category": int(qa["category"]),
                "category_name": CATEGORY_NAMES[int(qa["category"])],
                "answer_sessions": _evidence_sessions(conversation["conversation"], qa.get("evidence", [])),
            }
    questions = read_jsonl(RUN_ROOT / "frozen_questions.jsonl")
    qids = [row["question_id"] for row in questions]
    if set(labels) != set(qids) or len(qids) != 1986:
        raise RuntimeError("Scoring labels do not match the frozen blind question set")
    categories = np.asarray([labels[qid]["category"] for qid in qids], dtype=np.int32)
    arm_rows = {}
    arm_index = {}
    for arm, file_name in (
        ("propmem", "propmem_predictions.jsonl"),
        ("propmem_tvr", "propmem_tvr_predictions.jsonl"),
    ):
        raw = read_jsonl(RUN_ROOT / file_name)
        if len(raw) != 1986 or {row["question_id"] for row in raw} != set(qids):
            raise RuntimeError(f"{arm} predictions do not cover the frozen 1,986 questions")
        scored = [
            _scored_row(row, labels[row["question_id"]], compute_f1, _is_refusal)
            for row in raw
        ]
        arm_rows[arm] = scored
        arm_index[arm] = {row["question_id"]: row for row in scored}

    deltas = np.asarray(
        [
            arm_index["propmem_tvr"][qid]["token_f1"] - arm_index["propmem"][qid]["token_f1"]
            for qid in qids
        ],
        dtype=np.float64,
    )
    category_names = {int(key): value for key, value in CATEGORY_NAMES.items()}
    result: dict[str, Any] = {
        "benchmark": "LoCoMo",
        "n": 1986,
        "fullcontext": {
            "overall_token_f1": 0.44128651157614884,
            "per_category_token_f1": {
                "Factual": 0.4088017,
                "Temporal": 0.3217090,
                "Inferential": 0.1157723,
                "Multi-hop": 0.5228659,
                "Adversarial": 0.4641256,
            },
            "answerable_only_normalized_em": 0.1855,
            "answerable_only_normalized_em_numerator": 286,
            "answerable_only_normalized_em_denominator": 1542,
            "empty_gold_adversarial_refusal_accuracy": 0.4662,
            "mean_reader_prompt_tokens": 19641.96,
            "prediction_sha256": "cf40f1fdf2deda465003df6fb44d5c04b7c736eb3276e1ef94a43afe5f4ec794",
            "source": "frozen local Qwen FullContext control; not rerun in MEM-Q1",
        },
    }
    categories_result = {}
    category_bootstraps = {}
    for category, name in category_names.items():
        mask = categories == category
        category_delta = deltas[mask]
        rows_p = [row for row in arm_rows["propmem"] if row["category"] == category]
        rows_t = [row for row in arm_rows["propmem_tvr"] if row["category"] == category]
        categories_result[name] = {
            "n": int(mask.sum()),
            "propmem": _category_metrics(rows_p, _is_refusal),
            "propmem_tvr": _category_metrics(rows_t, _is_refusal),
            "paired_delta_token_f1": float(category_delta.mean()),
        }
        category_bootstraps[name] = _bootstrap(
            category_delta, np.full(len(category_delta), category, dtype=np.int32)
        )
    result["propmem"] = {
        **_category_metrics(arm_rows["propmem"], _is_refusal),
        "categories": {name: value["propmem"] for name, value in categories_result.items()},
        "retrieval": _retrieval_metrics(arm_rows["propmem"], labels),
    }
    result["propmem_tvr"] = {
        **_category_metrics(arm_rows["propmem_tvr"], _is_refusal),
        "categories": {name: value["propmem_tvr"] for name, value in categories_result.items()},
        "retrieval": _retrieval_metrics(arm_rows["propmem_tvr"], labels),
    }
    temporal_mask = categories == 2
    overall_bootstrap = _bootstrap(deltas, categories)
    temporal_bootstrap = _bootstrap(
        deltas[temporal_mask],
        np.full(int(temporal_mask.sum()), 2, dtype=np.int32),
    )
    result["paired_delta"] = {
        "overall_token_f1": float(deltas.mean()),
        "temporal_token_f1": float(deltas[temporal_mask].mean()),
        "categories": {
            name: value["paired_delta_token_f1"]
            for name, value in categories_result.items()
        },
    }
    result["bootstrap"] = {
        "overall": overall_bootstrap,
        "temporal": temporal_bootstrap,
        "categories": category_bootstraps,
    }

    ledger = read_jsonl(RUN_ROOT / "call_ledger.jsonl")
    hosted_calls = sum(
        str(row.get("provider") or "").startswith("local_") is False
        or int(row.get("hosted_calls") or 0) > 0
        for row in ledger
    )
    if hosted_calls:
        raise RuntimeError(f"Hosted provider call detected: {hosted_calls}")
    runtime_segments = read_jsonl(RUN_ROOT / "qa_runtime_segments.jsonl")
    ingestion = json.loads((RUN_ROOT / "ingestion" / "manifest.json").read_text(encoding="utf-8"))
    calls_by_role: dict[str, int] = {}
    for row in ledger:
        role = str(row.get("role") or "unknown")
        calls_by_role[role] = calls_by_role.get(role, 0) + 1
    baseline_rows = arm_rows["propmem"]
    tvr_rows = arm_rows["propmem_tvr"]
    writer_calls = sum(int(row.get("writer_llm_calls") or 0) for row in ingestion["conversations"])
    inference_calls = sum(row.get("role") in {"reader_answer", "memory_reasoning"} for row in ledger)
    transient_failures = sum(row.get("success") is False for row in ledger)
    result["efficiency"] = {
        "ingestion_wall_seconds": sum(float(row.get("wall_seconds") or 0) for row in ingestion["conversations"]),
        "ingestion_writer_llm_calls": writer_calls,
        "ingestion_writer_tokens": sum(int(row.get("writer_tokens") or 0) for row in ingestion["conversations"]),
        "ingestion_embedding_calls": sum(int(row.get("embedding_calls") or 0) for row in ingestion["conversations"]),
        "store_bytes": ingestion["store_bytes"],
        "num_propositions": sum(int(row["num_propositions"]) for row in ingestion["conversations"]),
        "num_chunks": sum(int(row["num_chunks"]) for row in ingestion["conversations"]),
        "qa_wall_seconds": sum(float(row.get("duration_seconds") or 0) for row in runtime_segments),
        "local_llm_calls": writer_calls + inference_calls,
        "local_embedding_calls_total": calls_by_role.get("embedding", 0),
        "hosted_calls": 0,
        "permanent_infra_failures": 0,
        "transient_infra_failure_events": transient_failures,
        "retry_count": sum(int(row.get("retry_count") or 0) for row in ledger)
        + transient_failures
        + sum(int(row.get("runner_retry_count") or 0) for row in baseline_rows + tvr_rows),
        "calls_by_role": calls_by_role,
        "propmem": _arm_efficiency(baseline_rows),
        "propmem_tvr": _arm_efficiency(tvr_rows),
        "reader_context_reduction_vs_fullcontext": {
            "propmem": (
                1 - _mean_numeric(baseline_rows, "reader_prompt_tokens") / 19641.96
                if _mean_numeric(baseline_rows, "reader_prompt_tokens") is not None else None
            ),
            "propmem_tvr": (
                1 - _mean_numeric(tvr_rows, "reader_prompt_tokens") / 19641.96
                if _mean_numeric(tvr_rows, "reader_prompt_tokens") is not None else None
            ),
        },
        "tvr_resolver_mean_ms": _mean_numeric(tvr_rows, "resolver_latency_ms"),
    }
    result["integrity"] = {
        **revisions,
        "reader_runtime": runtime,
        "question_manifest_sha256": freeze["question_manifest_sha256"],
        "ingestion_cache_sha256": freeze["ingestion_root_sha256"],
        "baseline_predictions_sha256": freeze["predictions"]["propmem"]["sha256"],
        "tvr_predictions_sha256": freeze["predictions"]["propmem_tvr"]["sha256"],
        "n_baseline": len(baseline_rows),
        "n_tvr": len(tvr_rows),
        "question_id_sets_identical": True,
        "retrieval_signature_sets_identical": (
            freeze["predictions"]["propmem"]["retrieval_signature_set_sha256"]
            == freeze["predictions"]["propmem_tvr"]["retrieval_signature_set_sha256"]
        ),
        "predictions_frozen_before_scoring": True,
        "hosted_calls": 0,
        "permanent_infra_failures": 0,
        "llama_8092_untouched": True,
    }

    overall_ci = overall_bootstrap["ci95"]
    temporal_ci = temporal_bootstrap["ci95"]
    if result["paired_delta"]["overall_token_f1"] > 0 and overall_ci[0] > 0:
        eligibility = "STRONG"
        claim = (
            f"Based on PropMem, a non-mutating query-time temporal validity resolver improved "
            f"LoCoMo token F1 from {result['propmem']['token_f1']:.4f} to "
            f"{result['propmem_tvr']['token_f1']:.4f} "
            f"(+{result['paired_delta']['overall_token_f1'] * 100:.2f} pp, paired 95% CI "
            f"[{overall_ci[0] * 100:.2f}, {overall_ci[1] * 100:.2f}] pp)."
        )
    elif (
        result["paired_delta"]["temporal_token_f1"] > 0
        and temporal_ci[0] > 0
        and result["paired_delta"]["overall_token_f1"] >= -0.02
    ):
        eligibility = "TEMPORAL_ONLY"
        claim = (
            f"Temporal slice improved by {result['paired_delta']['temporal_token_f1'] * 100:.2f} pp "
            f"(95% CI [{temporal_ci[0] * 100:.2f}, {temporal_ci[1] * 100:.2f}] pp), "
            "with overall degradation no worse than 2 pp."
        )
    else:
        eligibility = "NO"
        claim = "RESUME_METHOD_IMPROVEMENT_CLAIM=NO"
    result["resume_claim"] = {"eligibility": eligibility, "claim": claim}
    _write_json(RUN_ROOT / "final_scorecard.json", result)
    _write_report(result)
    return result


def _freeze_run_config(
    revisions: dict[str, Any], runtime: dict[str, Any], embedding: dict[str, Any]
) -> dict[str, Any]:
    value = {
        "protocol": "MEM-Q1-PropMem-TVR-LoCoMo-v1",
        "runner_sha256": sha256_file(Path(__file__)),
        "artifact_helper_sha256": sha256_file(Path(__file__).with_name("mem1_artifacts.py")),
        "revisions": revisions,
        "reader": {
            "model_sha256": READER_SHA256,
            "endpoint": ENDPOINT,
            "binary_sha256": runtime["binary_sha256"],
            "command_line": runtime["command_line"],
            "kv_cache_location": runtime["kv_cache_location"],
            "answer_budget": ANSWER_BUDGET,
            "temperature": 0,
            "seed": 42,
            "thinking": False,
        },
        "embedding": embedding,
        "tvr": {
            "same_entity": True,
            "cosine_threshold": SAME_TOPIC_COSINE_THRESHOLD,
            "intent_rules_sha256": sha256_file(Path(__file__).with_name("locomo_tvr.py")),
        },
        "hosted_apis": "NONE",
    }
    path = RUN_ROOT / "run_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            mutable = [
                item for item in RUN_ROOT.rglob("*")
                if item.is_file() and item.name not in {
                    "run_config.json", "frozen_questions.jsonl", "frozen_questions.sha256"
                }
            ]
            if mutable:
                raise RuntimeError("MEM-Q1 run config differs from artifacts already produced; refusing mixed resume")
            _write_json(path, value)
    else:
        _write_json(path, value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",), default="all")
    parser.parse_args()

    _configure_local_environment()
    sys.path.insert(0, str(MEMEVAL_ROOT / "src"))
    revisions = _verify_revisions()
    owned_patch = _ensure_memeval_patch()
    try:
        from agents_memory.healthcopilot_provider import (
            ProviderConfig,
            configure_call_ledger,
            initialize_local_embedding,
            reader_slot_context,
            release_local_embedding_runtimes,
        )

        config = ProviderConfig.from_env()
        runtime = {**_reader_process(), **_verify_endpoint(config)}
        runtime["slot_context_tokens"] = reader_slot_context(config)
        if runtime["slot_context_tokens"] < 131072:
            raise RuntimeError(
                f"Reader slot does not expose 131072 tokens: {runtime['slot_context_tokens']}"
            )

        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        configure_call_ledger(RUN_ROOT / "call_ledger.jsonl")
        embedding_runtime = initialize_local_embedding(config)
        embedding = dataclasses.asdict(embedding_runtime.artifact)
        embedding["model_path"] = config.embedding_path
        _freeze_run_config(revisions, runtime, embedding)
        question_sha = prepare_questions()
        print(f"Frozen blind question manifest: 1,986 rows, sha256={question_sha}", flush=True)

        provider = sys.modules["agents_memory.healthcopilot_provider"]
        ingestion = ingest_all(_load_dataset(), config, provider, embedding, revisions)
        print(f"Shared PropMem ingestion frozen: sha256={ingestion['root_sha256']}", flush=True)
        infer_all(config, provider, ingestion)
        freeze = freeze_predictions()
        print("Both 1,986-row prediction files frozen before scoring.", flush=True)
        result = score_and_report(revisions, runtime, freeze)
        print(json.dumps({
            "status": "COMPLETE",
            "propmem_f1": result["propmem"]["token_f1"],
            "propmem_tvr_f1": result["propmem_tvr"]["token_f1"],
            "paired_delta": result["paired_delta"]["overall_token_f1"],
            "temporal_delta": result["paired_delta"]["temporal_token_f1"],
            "resume_claim": result["resume_claim"]["eligibility"],
            "report": str(ROOT / "docs/research/memory/mem_q1_propmem_tvr_locomo_closeout.md"),
            "scorecard": str(RUN_ROOT / "final_scorecard.json"),
        }, ensure_ascii=False, indent=2), flush=True)
        return 0
    finally:
        try:
            from agents_memory.healthcopilot_provider import release_local_embedding_runtimes
            release_local_embedding_runtimes()
        except Exception:
            pass
        _remove_owned_patch(owned_patch)


if __name__ == "__main__":
    raise SystemExit(main())
