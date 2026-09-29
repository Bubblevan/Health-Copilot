"""Run the frozen E5-B3 independent Reader V2 budget qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.rag_e5.e5b3_qualification import (
    build_qualification_fixtures,
    canonical_sha256,
    evaluate_qualification_case,
    fixture_set_sha256,
    summarize_budget,
)
from eval.rag_e5.e5b3_reader import (
    READER_STATE_PROJECTION_SHA256,
    READER_V2_PROMPT,
    READER_V2_SCHEMA,
    render_reader_v2_prompt,
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
REPORT_PATH = ROOT / "runs/rag_e5/e5b3_reader_qualification.json"
EXTERNAL_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b3")
PORT = 8093
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
EXPECTED_LLAMA_VERSION = (
    "version: 10068 (571d0d540)\nbuilt with Clang 20.1.8 for Windows x86_64"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_frozen_protocol(
    *, lock: dict[str, Any], corpus_root: Path, model_path: Path, server_executable: Path
) -> list[dict[str, Any]]:
    body = {key: value for key, value in lock.items() if key != "qualification_lock_sha256"}
    if canonical_sha256(body) != lock.get("qualification_lock_sha256"):
        raise ValueError("reader qualification lock self-hash mismatch")
    if lock.get("status") != "FROZEN_BEFORE_QUALIFICATION_MODEL_CALLS":
        raise ValueError("reader protocol must be frozen before qualification")
    if lock.get("b2_or_202608_outputs_read") is not False or lock.get("retrieval_performed") is not False:
        raise ValueError("qualification lock violates its independence boundary")
    if _sha256_file(corpus_root / "chunks.jsonl") != lock.get("approved_corpus_chunks_sha256"):
        raise ValueError("approved corpus changed after qualification protocol freeze")
    if _sha256_file(model_path) != lock["model"].get("sha256"):
        raise ValueError("Qwen GGUF changed after qualification protocol freeze")
    version_result = subprocess.run(
        [str(server_executable), "--version"], check=True, capture_output=True, text=True
    )
    version = (version_result.stdout + version_result.stderr).strip()
    if version != EXPECTED_LLAMA_VERSION or version != lock["llama_cpp"].get("version"):
        raise ValueError("llama.cpp runtime identity changed after protocol freeze")
    if _sha256_file(server_executable) != lock["llama_cpp"].get("executable_sha256"):
        raise ValueError("llama.cpp executable changed after protocol freeze")
    source_hashes = lock.get("source_sha256", {})
    for relative, expected in source_hashes.items():
        if _sha256_file(ROOT / relative) != expected:
            raise ValueError(f"frozen qualification source changed: {relative}")
    chunks = [
        json.loads(line)
        for line in (corpus_root / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fixtures = build_qualification_fixtures(chunks)
    if fixture_set_sha256(fixtures) != lock.get("fixture_set_sha256"):
        raise ValueError("qualification fixtures differ from the frozen fixture set")
    if lock.get("reader_state_projection_sha256") != READER_STATE_PROJECTION_SHA256:
        raise ValueError("ReaderStateView projection differs from the frozen identity")
    if lock.get("reader_schema_sha256") != canonical_sha256(READER_V2_SCHEMA):
        raise ValueError("Reader V2 schema differs from the frozen identity")
    prompt_hash = hashlib.sha256(READER_V2_PROMPT.encode("utf-8")).hexdigest()
    if lock.get("reader_prompt_sha256") != prompt_hash:
        raise ValueError("Reader V2 prompt differs from the frozen identity")
    return fixtures


def _server_ready(port: int, timeout_seconds: int = 240) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("B3 llama.cpp server did not become healthy on loopback")


def _start_server(server: Path, model: Path, log_path: Path) -> tuple[subprocess.Popen[Any], Any]:
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise OSError(f"loopback port {PORT} is already in use; refusing model ambiguity")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab")
    args = [
        str(server),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--ctx-size",
        "16384",
        "--n-gpu-layers",
        "0",
        "--reasoning",
        "off",
        "--chat-template-kwargs",
        '{"enable_thinking":false}',
        "--no-cache-prompt",
        "--parallel",
        "1",
        "--offline",
        "--no-webui",
    ]
    process = subprocess.Popen(
        args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.poll() is not None:
        log_handle.close()
        raise RuntimeError(f"B3 llama.cpp server exited with code {process.returncode}")
    return process, log_handle


class _QualificationClient:
    def __init__(self, base_url: str, *, budget: int, timeout_seconds: int = 600) -> None:
        if not base_url.startswith("http://127.0.0.1:"):
            raise ValueError("qualification model endpoint must be loopback-only HTTP")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.budget = budget
        self.timeout_seconds = timeout_seconds

    def complete(self, prompt: str) -> dict[str, Any]:
        body = {
            "model": "local-qwen3-8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": self.budget,
            "cache_prompt": False,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "e5_reader_v2",
                    "strict": True,
                    "schema": READER_V2_SCHEMA,
                },
            },
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read())
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("llama.cpp response has invalid choices")
        choice = choices[0]
        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str):
            raise TypeError("llama.cpp response has no text content")
        usage = payload.get("usage") or {}
        return {
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }


def run_qualification(
    *,
    corpus_root: Path,
    model_path: Path,
    server_executable: Path,
    lock_path: Path,
    report_path: Path,
    external_root: Path,
) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"qualification report is immutable: {report_path}")
    lock = _read_json(lock_path)
    fixtures = _validate_frozen_protocol(
        lock=lock,
        corpus_root=corpus_root,
        model_path=model_path,
        server_executable=server_executable,
    )
    external_root.mkdir(parents=True, exist_ok=True)
    ledger_path = external_root / "qualification_call_ledger.jsonl"
    if ledger_path.exists():
        raise FileExistsError(
            "qualification call ledger already exists; no model call may be repeated"
        )
    log_path = external_root / "logs" / "reader-qualification-20260930.log"
    process, log_handle = _start_server(server_executable, model_path, log_path)
    budget_summaries: list[dict[str, Any]] = []
    selected_budget: int | None = None
    total_calls = 0
    try:
        _server_ready(PORT)
        for budget in lock["candidate_output_budgets"]:
            client = _QualificationClient(BASE_URL, budget=budget)
            case_results: list[dict[str, Any]] = []
            for fixture in fixtures:
                prompt = render_reader_v2_prompt(
                    question=fixture["question"],
                    state_view=fixture["state_view"],
                    passages=fixture["passages"],
                )
                prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                call_id = canonical_sha256(
                    [lock["qualification_lock_sha256"], budget, fixture["fixture_id"]]
                )
                _append_jsonl(
                    ledger_path,
                    {
                        "call_id": call_id,
                        "fixture_id": fixture["fixture_id"],
                        "budget": budget,
                        "prompt_sha256": prompt_sha,
                        "status": "STARTED",
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                )
                response = client.complete(prompt)
                evaluation = evaluate_qualification_case(
                    fixture=fixture,
                    text=response["text"],
                    finish_reason=response["finish_reason"],
                    output_tokens=response["output_tokens"],
                    output_cap=budget,
                )
                row = {
                    "fixture_id": fixture["fixture_id"],
                    "stratum": fixture["stratum"],
                    "prompt_sha256": prompt_sha,
                    "input_tokens": response["input_tokens"],
                    "latency_ms": response["latency_ms"],
                    **evaluation,
                    "raw_output": response["text"],
                }
                case_results.append(row)
                _append_jsonl(
                    ledger_path,
                    {
                        "call_id": call_id,
                        "fixture_id": fixture["fixture_id"],
                        "budget": budget,
                        "prompt_sha256": prompt_sha,
                        "status": "COMPLETED",
                        "finish_reason": response["finish_reason"],
                        "input_tokens": response["input_tokens"],
                        "output_tokens": response["output_tokens"],
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    },
                )
                total_calls += 1
            summary = summarize_budget(budget=budget, case_results=case_results)
            budget_summaries.append(summary)
            if summary["passed"]:
                selected_budget = budget
                break
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()

    qualified = selected_budget is not None
    report: dict[str, Any] = {
        "schema_version": "rag-e5-e5b3-reader-qualification-v1",
        "status": "QUALIFIED" if qualified else "NOT_QUALIFIED_STOP",
        "qualification_lock_sha256": lock["qualification_lock_sha256"],
        "fixture_set_sha256": lock["fixture_set_sha256"],
        "approved_corpus_chunks_sha256": lock["approved_corpus_chunks_sha256"],
        "reader_schema_sha256": lock["reader_schema_sha256"],
        "reader_prompt_sha256": lock["reader_prompt_sha256"],
        "reader_state_projection_sha256": READER_STATE_PROJECTION_SHA256,
        "model": lock["model"],
        "llama_cpp": lock["llama_cpp"],
        "qualification_cases": len(fixtures),
        "qualification_model_calls": total_calls,
        "tested_budgets": [row["output_budget"] for row in budget_summaries],
        "budget_results": budget_summaries,
        "selected_output_budget": selected_budget,
        "reader_contract_qualified": qualified,
        "retrieval_calls": 0,
        "bridge_calls": 0,
        "b2_outputs_read": False,
        "202608_opened": False,
        "qualification_code_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "call_ledger_sha256": _sha256_file(ledger_path),
    }
    report["qualification_report_sha256"] = canonical_sha256(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--server-executable", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    args = parser.parse_args()
    result = run_qualification(
        corpus_root=args.corpus_root,
        model_path=args.model,
        server_executable=args.server_executable,
        lock_path=args.lock,
        report_path=args.report,
        external_root=args.external_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "qualification_model_calls": result["qualification_model_calls"],
                "tested_budgets": result["tested_budgets"],
                "selected_output_budget": result["selected_output_budget"],
                "reader_contract_qualified": result["reader_contract_qualified"],
                "qualification_report_sha256": result["qualification_report_sha256"],
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
