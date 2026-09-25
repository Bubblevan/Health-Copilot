"""Generate frozen same-Qwen GAR views for DEV or a locked TEST method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_crb_data import (
    PARTITIONS,
    SOURCE_MANIFEST_PATH,
    load_partition_inputs,
    load_source_manifest,
)
from eval.r2med_gar_generation import (
    GENERATION_CONFIG,
    METHODS,
    LocalLlamaCppClient,
    R2MedGARGenerator,
    load_upstream_prompt_catalog,
    prompt_sha256,
)

E_ROOT = Path(r"E:\Health-Copilot-RAG")
QWEN_PATH = E_ROOT / "models/qwen3-8b/Qwen3-8B-Q4_K_M.gguf"
QWEN_BYTES = 5_027_783_488
QWEN_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
DEFAULT_UPSTREAM = Path(r"D:\MyLab\Jianli\external\rag\R2MED")
DEFAULT_SOURCE_ROOT = Path(r"E:\Health-Copilot-E1.2\sources")
DEFAULT_SERVER = r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
PORT = 8089
OUTPUT_ROOT = E_ROOT / "r2med/generated"
RANKING_ROOT = E_ROOT / "r2med/rankings"
RUN_MANIFEST_ROOT = ROOT / "runs/rag_r2med_crb/generation"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_rankings(path: Path) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            docs = row["ranking"]
            if not isinstance(docs, list) or len(docs) > 100:
                raise ValueError(f"invalid top-100 ranking at {path.name}:{line_number}")
            rankings[query_id] = [str(item["doc_id"]) for item in docs[:10]]
    return rankings


def _feedback_for_query(
    query_id: str,
    ranked_ids: Mapping[str, Sequence[str]],
    documents: Mapping[str, str],
) -> list[str]:
    if query_id not in ranked_ids:
        raise ValueError(f"BM25 feedback ranking is missing query {query_id}")
    feedback = []
    for doc_id in ranked_ids[query_id][:10]:
        if doc_id not in documents:
            raise ValueError(f"BM25 top-10 references an unknown corpus document: {doc_id}")
        feedback.append(" ".join(documents[doc_id].replace("\n", " ").split()[:512]))
    return feedback


def _write_line(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _check_qwen() -> dict[str, Any]:
    if not QWEN_PATH.is_file():
        raise FileNotFoundError(f"required Qwen GGUF is not installed: {QWEN_PATH}")
    if QWEN_PATH.stat().st_size != QWEN_BYTES:
        raise ValueError("Qwen GGUF byte count does not match the frozen model identity")
    digest = hashlib.sha256()
    with QWEN_PATH.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    actual_hash = digest.hexdigest()
    if actual_hash != QWEN_SHA256:
        raise ValueError("Qwen GGUF SHA-256 does not match the frozen model identity")
    return {"path": str(QWEN_PATH), "bytes": QWEN_BYTES, "sha256": actual_hash}


def _llama_version(executable: Path) -> str:
    result = subprocess.run([str(executable), "--version"], check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip().splitlines()[0]


def _server_ready(port: int, timeout_seconds: int = 240) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/health", method="GET")
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("local llama.cpp server did not become healthy on loopback")


def _start_server(executable: Path, port: int, log_path: Path) -> tuple[subprocess.Popen, Any]:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise OSError(f"port {port} is already serving another process; refusing model ambiguity")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    args = [
        str(executable),
        "--model", str(QWEN_PATH),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--ctx-size", "16384",
        "--n-gpu-layers", "99",
        "--reasoning", "off",
        "--chat-template-kwargs", '{"enable_thinking":false}',
        "--no-webui",
    ]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(args, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=flags)
    if process.poll() is not None:
        log_handle.close()
        raise RuntimeError(f"llama.cpp server exited with code {process.returncode}")
    return process, log_handle


def _manifest_rows(input_subsets, output_files: Mapping[str, Path]) -> list[dict[str, Any]]:
    identity = load_source_manifest(SOURCE_MANIFEST_PATH)
    result = []
    for subset in input_subsets:
        entry = next(item for item in identity["datasets"]["DEV"] + identity["datasets"]["TEST"] if item["name"] == subset.name)
        query_order = "\n".join(query.query_id for query in subset.queries)
        output_path = output_files[subset.name]
        data = output_path.read_bytes()
        result.append(
            {
                "subset": subset.name,
                "upstream_prompt_family": subset.upstream_prompt_family,
                "query_count": len(subset.queries),
                "query_file_sha256": entry["files"]["query.jsonl"]["sha256"],
                "corpus_file_sha256": entry["files"]["corpus.jsonl"]["sha256"],
                "query_order_sha256": _hash_text(query_order),
                "artifact": str(output_path),
                "artifact_sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return result


def generate_method(
    partition: str,
    method: str,
    *,
    upstream_root: Path,
    source_root: Path,
    server_executable: Path,
    port: int = PORT,
) -> dict[str, Any]:
    if partition not in PARTITIONS:
        raise ValueError("partition must be DEV or TEST")
    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    if partition == "TEST":
        lock_path = ROOT / "runs/rag_r2med_crb/final_method_lock.json"
        if not lock_path.is_file():
            raise FileNotFoundError("TEST generation requires committed final_method_lock.json")
        if subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", str(lock_path.relative_to(ROOT))], capture_output=True, text=True, check=True).stdout.strip():
            raise ValueError("TEST generation requires a clean committed final_method_lock.json")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("status") != "FROZEN_BEFORE_TEST" or lock.get("crb_variant") not in {"crb_q", "crb_prf"}:
            raise ValueError("TEST generation refused: final method lock is invalid")
        if method.startswith("crb_") and method != lock["crb_variant"]:
            raise ValueError("TEST generation refused: CRB variant differs from the frozen lock")

    manifest = load_source_manifest(SOURCE_MANIFEST_PATH)
    qwen = _check_qwen()
    if not server_executable.is_file():
        raise FileNotFoundError(f"llama.cpp server executable not found: {server_executable}")
    method_manifest_path = RUN_MANIFEST_ROOT / partition.lower() / method / "generation_manifest.json"
    if method_manifest_path.exists():
        raise FileExistsError(f"generation manifest already exists; refusing duplicate run: {method_manifest_path}")
    upstream_prompts = load_upstream_prompt_catalog(upstream_root)
    inputs = load_partition_inputs(
        partition, source_root=source_root, source_manifest_path=SOURCE_MANIFEST_PATH
    )
    output_files = {
        subset.name: OUTPUT_ROOT / partition.lower() / method / f"{subset.name}.jsonl"
        for subset in inputs
    }
    feedback_by_subset: dict[str, dict[str, list[str]]] = {}
    if method in {"lamer", "crb_prf"}:
        for subset in inputs:
            bm25_path = RANKING_ROOT / partition.lower() / subset.name / "bm25_original.jsonl"
            top_ids = _read_rankings(bm25_path)
            docs = {document.doc_id: document.text for document in subset.documents}
            feedback_by_subset[subset.name] = {
                query.query_id: _feedback_for_query(query.query_id, top_ids, docs)
                for query in subset.queries
            }
    model_root = E_ROOT / "models/qwen3-8b"
    logs = E_ROOT / "cache/llama-cpp"
    model_root.mkdir(parents=True, exist_ok=True)
    prompt_hashes: set[str] = set()
    pending = False
    for subset in inputs:
        final_path = output_files[subset.name]
        partial_path = final_path.with_suffix(final_path.suffix + ".partial")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            rows = [json.loads(line) for line in final_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if [row.get("query_id") for row in rows] != [query.query_id for query in subset.queries]:
                raise ValueError(f"completed generation artifact has wrong query order: {final_path}")
        else:
            pending = True
            if partial_path.exists():
                rows = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                expected_prefix = [query.query_id for query in subset.queries[: len(rows)]]
                if [row.get("query_id") for row in rows] != expected_prefix:
                    raise ValueError(f"partial generation artifact is not a valid query prefix: {partial_path}")

        if method in {"crb_q", "crb_prf"}:
            from eval.r2med_gar_generation import CRB_FEEDBACK_SUFFIX, CRB_PROMPT

            prompt_hashes.add(prompt_sha256(CRB_PROMPT + (CRB_FEEDBACK_SUFFIX if method == "crb_prf" else "")))
        else:
            prompt_hashes.add(prompt_sha256(upstream_prompts[method][subset.upstream_prompt_family]))

    process = None
    log_handle = None
    api_client = LocalLlamaCppClient(f"http://127.0.0.1:{port}/v1")
    if pending:
        process, log_handle = _start_server(
            server_executable,
            port,
            logs / f"{partition.lower()}-{method}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log",
        )
    try:
        if pending:
            _server_ready(port)
        for subset in inputs:
            final_path = output_files[subset.name]
            partial_path = final_path.with_suffix(final_path.suffix + ".partial")
            if final_path.exists():
                continue
            generator = R2MedGARGenerator(subset.upstream_prompt_family, api_client, upstream_prompts)
            completed_rows = (
                [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if partial_path.exists()
                else []
            )
            with partial_path.open("a" if partial_path.exists() else "x", encoding="utf-8", newline="\n") as handle:
                for query in subset.queries[len(completed_rows) :]:
                    feedback = feedback_by_subset.get(subset.name, {}).get(query.query_id)
                    view = generator.generate(query.query_id, query.text, method, feedback)
                    row = view.to_dict()
                    row["subset"] = subset.name
                    _write_line(handle, row)
            partial_path.replace(final_path)
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_handle is not None:
            log_handle.close()

    generated_rows = [
        json.loads(line)
        for path in output_files.values()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calls = len(generated_rows)
    valid_outputs = sum(bool(row.get("valid")) for row in generated_rows)
    completed_outputs = sum(bool(row.get("completed")) for row in generated_rows)
    failures = sum(not bool(row.get("valid")) for row in generated_rows)
    truncations = sum(bool(row.get("truncated")) for row in generated_rows)
    total_output_tokens = sum(int(row.get("output_tokens", 0)) for row in generated_rows)

    method_root = RUN_MANIFEST_ROOT / partition.lower() / method
    method_root.mkdir(parents=True, exist_ok=True)
    manifest_path = method_manifest_path
    summary = {
        "schema_version": "r2med-gar-generation-manifest-v1",
        "partition": partition,
        "method": method,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "generator": {
            **qwen,
            "llama_cpp_version": _llama_version(server_executable),
            "endpoint_bind": "127.0.0.1",
        },
        "prompt_sha256": sorted(prompt_hashes),
        "generation_config": GENERATION_CONFIG,
        "query_count": calls,
        "call_count": calls,
        "failure_count": failures,
        "valid_output_count": valid_outputs,
        "completion_count": completed_outputs,
        "truncation_count": truncations,
        "average_output_tokens": total_output_tokens / calls if calls else 0.0,
        "dataset_source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST_PATH.read_bytes()).hexdigest(),
        "upstream_commit": manifest["upstream"]["commit"],
        "subsets": _manifest_rows(inputs, output_files),
    }
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite generation manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def smoke_local_generator(server_executable: Path, *, port: int = PORT) -> dict[str, Any]:
    """Validate the exact Qwen/llama.cpp JSON-constrained path on synthetic text."""
    qwen = _check_qwen()
    if not server_executable.is_file():
        raise FileNotFoundError(f"llama.cpp server executable not found: {server_executable}")
    log_path = E_ROOT / "cache/llama-cpp" / f"smoke-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
    process, log_handle = _start_server(server_executable, port, log_path)
    try:
        _server_ready(port)
        generator = R2MedGARGenerator(
            "synthetic", LocalLlamaCppClient(f"http://127.0.0.1:{port}/v1"), {}
        )
        view = generator.generate(
            "synthetic-smoke-query",
            "Synthetic smoke test: describe general public evidence about sodium intake and blood pressure.",
            "crb_q",
        )
        if not view.valid or not view.completed or view.truncated:
            raise RuntimeError(f"local Qwen CRB JSON smoke test failed: {view.error or view.finish_reason}")
        return {
            "model_sha256": qwen["sha256"],
            "llama_cpp_version": _llama_version(server_executable),
            "output_tokens": view.output_tokens,
            "structured_fields": sorted(view.structured or {}),
            "json_valid": view.valid,
            "completed": view.completed,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("DEV", "TEST"), default="DEV")
    parser.add_argument("--method", choices=(*METHODS, "all"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--llama-server", type=Path, default=Path(DEFAULT_SERVER))
    args = parser.parse_args()
    if args.smoke_only:
        print(json.dumps(smoke_local_generator(args.llama_server), indent=2))
        return
    if args.method is None:
        parser.error("--method is required unless --smoke-only is set")
    methods = METHODS if args.method == "all" else (args.method,)
    for method in methods:
        summary = generate_method(
            args.partition,
            method,
            upstream_root=args.upstream_root,
            source_root=args.source_root,
            server_executable=args.llama_server,
        )
        print(
            f"{method}: calls={summary['call_count']} valid={summary['valid_output_count']} "
            f"completed={summary['completion_count']} failures={summary['failure_count']}"
        )


if __name__ == "__main__":
    main()
