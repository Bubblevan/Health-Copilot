"""Freeze the independent E5-B3 reader qualification protocol before model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from eval.rag_e5.e5b3_qualification import build_qualification_fixtures, fixture_set_sha256
from eval.rag_e5.e5b3_reader import (
    READER_STATE_PROJECTION_SHA256,
    READER_V2_PROMPT,
    READER_V2_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = Path(
    r"D:\MyLab\Jianli\external\rag_e5\e5a3\corpus\public_health_plus_guideline"
)
DEFAULT_MODEL = Path(
    r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf"
)
DEFAULT_SERVER = Path(
    r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
)
LOCK_PATH = ROOT / "runs/rag_e5/e5b3_reader_qualification_lock.json"
EXPECTED_MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
EXPECTED_LLAMA_VERSION = (
    "version: 10068 (571d0d540)\nbuilt with Clang 20.1.8 for Windows x86_64"
)
SOURCE_PATHS = (
    "eval/rag_e5/e5b3_reader.py",
    "eval/rag_e5/e5b3_qualification.py",
    "tools/research/rag_e5/freeze_e5b3_qualification.py",
    "tools/research/rag_e5/qualify_e5b3_reader.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_protocol(
    *, corpus_root: Path, model_path: Path, server_executable: Path, lock_path: Path
) -> dict[str, Any]:
    if lock_path.exists():
        raise FileExistsError(f"qualification protocol lock is immutable: {lock_path}")
    chunks_path = corpus_root / "chunks.jsonl"
    if not chunks_path.is_file() or not model_path.is_file() or not server_executable.is_file():
        raise FileNotFoundError("approved corpus, exact model, and llama.cpp executable are required")
    model_hash = _sha256_file(model_path)
    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError("Qwen3-8B GGUF SHA256 differs from the frozen B2 generator")
    version_result = subprocess.run(
        [str(server_executable), "--version"], check=True, capture_output=True, text=True
    )
    llama_version = (version_result.stdout + version_result.stderr).strip()
    if llama_version != EXPECTED_LLAMA_VERSION:
        raise ValueError("llama.cpp version differs from the B2 runtime identity")
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fixtures = build_qualification_fixtures(chunks)
    counts = {
        name: sum(row["stratum"] == name for row in fixtures)
        for name in ("Q0_STATE_ONLY", "Q1_EVIDENCE_ONLY", "Q2_STATE_AND_EVIDENCE")
    }
    if len(fixtures) != 24 or counts != {
        "Q0_STATE_ONLY": 8,
        "Q1_EVIDENCE_ONLY": 8,
        "Q2_STATE_AND_EVIDENCE": 8,
    }:
        raise ValueError("qualification fixture balance differs from the frozen protocol")
    source_hashes = {relative: _sha256_file(ROOT / relative) for relative in SOURCE_PATHS}
    body: dict[str, Any] = {
        "schema_version": "rag-e5-e5b3-reader-qualification-lock-v1",
        "status": "FROZEN_BEFORE_QUALIFICATION_MODEL_CALLS",
        "scope": "independent reader contract qualification; no retrieval or task outcomes",
        "qualification_fixture_count": 24,
        "fixture_strata": counts,
        "fixture_set_sha256": fixture_set_sha256(fixtures),
        "approved_corpus_chunks_sha256": _sha256_file(chunks_path),
        "qualification_passage_ids": sorted(
            {
                passage["chunk_id"]
                for fixture in fixtures
                for passage in fixture["passages"]
            }
        ),
        "q2_passages_per_fixture": 5,
        "q2_passage_char_lengths": sorted(
            len(passage["text"])
            for passage in fixtures[-8:][0]["passages"]
        ),
        "reader_schema_sha256": _canonical_sha256(READER_V2_SCHEMA),
        "reader_prompt_sha256": hashlib.sha256(READER_V2_PROMPT.encode("utf-8")).hexdigest(),
        "reader_state_projection_sha256": READER_STATE_PROJECTION_SHA256,
        "candidate_output_budgets": [256, 384, 512],
        "budget_selection": "smallest budget that passes every locked qualification gate",
        "gates": {
            "valid_reader_v2_json": "24/24",
            "finish_reason_length": "0/24",
            "q0_q2_canonical_state_contract": "16/16",
            "q1_q2_invented_citations": 0,
            "q1_q2_valid_supplied_citation": "at least one for each evidence-required fixture",
            "p95_output_tokens": "<= 75% of candidate output budget",
        },
        "model": {
            "repository": "Qwen/Qwen3-8B-GGUF",
            "revision": "6a569868d07d3bd59e8b97fb001bf8c0b254bb20",
            "file": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": model_hash,
            "temperature": 0.0,
            "reasoning": "disabled",
            "retry_on_error": False,
        },
        "llama_cpp": {
            "version": llama_version,
            "executable": str(server_executable),
            "executable_sha256": _sha256_file(server_executable),
            "server_args": [
                "--host 127.0.0.1",
                "--port 8093",
                "--ctx-size 16384",
                "--n-gpu-layers 0",
                "--reasoning off",
                '--chat-template-kwargs {"enable_thinking":false}',
                "--no-cache-prompt",
                "--parallel 1",
                "--offline",
                "--no-webui",
            ],
            "endpoint": "http://127.0.0.1:8093/v1",
        },
        "b2_or_202608_outputs_read": False,
        "b2_tasks_or_reader_outputs_used": False,
        "retrieval_performed": False,
        "model_calls": 0,
        "source_sha256": source_hashes,
    }
    body["qualification_lock_sha256"] = _canonical_sha256(body)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server-executable", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args()
    lock = freeze_protocol(
        corpus_root=args.corpus_root,
        model_path=args.model,
        server_executable=args.server_executable,
        lock_path=args.lock,
    )
    print(
        json.dumps(
            {
                "qualification_lock_sha256": lock["qualification_lock_sha256"],
                "fixture_set_sha256": lock["fixture_set_sha256"],
                "model_sha256": lock["model"]["sha256"],
                "llama_cpp_version": lock["llama_cpp"]["version"],
                "calls_before_commit": lock["model_calls"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
