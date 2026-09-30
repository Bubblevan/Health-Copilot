"""Freeze the CPU B4 protocol against existing R2MED/E5 identities."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.rag_e5.b4_execution import (DEFAULT_B4_PRIVATE_ROOT,
                                      DEFAULT_PROTOCOL_PATH,
                                      DEFAULT_SERVER_URL, ROOT,
                                      _inspect_server_process, _read_json,
                                      load_verified_b2,
                                      read_and_verify_b3_identity)
from eval.rag_e5.b4_guidance import (CHAT_TEMPLATE_OVERHEAD_RESERVE,
                                     CONTEXT_SIZE, MAX_OUTPUT_TOKENS,
                                     PROMPT_TEMPLATE_SHA256,
                                     SYSTEM_PROMPT_SHA256, LlamaServerClient)
from eval.rag_e5.b4_materializer import classify_runtime_task
from eval.rag_e5.counterfactual import canonical_sha256
from eval.rag_e5.e5b3_recovery import sha256_file

BASE_MAIN = "2e2d62ce93a84378a1cad314994a097a9007a6fb"
BRANCH = "codex/rag-e5-b4-cpu-20260930"
MODEL_PATH = Path(r"E:\Health-Copilot-Models\models\qwen3-8b\Qwen3-8B-Q4_K_M.gguf")
CODE_PATHS = (
    "pyproject.toml",
    "eval/rag_e5/b4_materializer.py",
    "eval/rag_e5/b4_guidance.py",
    "eval/rag_e5/b4_execution.py",
    "eval/rag_e5/b4_evaluator.py",
    "eval/rag_e5/e5b3_recovery.py",
    "eval/rag_e5/counterfactual.py",
    "eval/rag_e5/e5b2_evaluator.py",
    "eval/rag_e5/overlay.py",
    "tools/research/rag_e5/freeze_e5b4_protocol.py",
    "tools/research/rag_e5/run_e5b4_counterfactual.py",
    "tools/research/rag_e5/score_e5b4_counterfactual.py",
    "tests/test_rag_e5_b4.py",
    "docs/research/rag_e5/e5b4_protocol.md",
)
B2_LOCK = ROOT / "runs/rag_e5/e5b2_protocol_lock.json"
B2_EXECUTION = ROOT / "runs/rag_e5/e5b2_execution_manifest.json"
B2_REPORT = ROOT / "runs/rag_e5/e5b2_counterfactual_report.json"
PRIVATE_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b1")
CORPUS_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5a3\corpus\public_health_plus_guideline")
PRIOR_GPU_ATTEMPT_PATH = ROOT / "runs/rag_e5/e5b4_gpu_partial_attempt.json"
CPU_THREADS = 12


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _cpu_snapshot() -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("install the research extra to record CPU runtime identity") from exc
    return {
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "available_memory_bytes_at_freeze": psutil.virtual_memory().available,
        "llama_threads": CPU_THREADS,
        "snapshot_is_not_a_method_variable": True,
    }


def freeze(
    *,
    server_url: str,
    executable_path: Path,
    executable_sha256: str,
    output_path: Path,
    private_root: Path,
    execution_manifest_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError("B4 protocol lock already exists; it is immutable")
    if execution_manifest_path.exists():
        raise FileExistsError("B4 execution manifest path already exists; it is immutable")
    if server_url != DEFAULT_SERVER_URL:
        raise ValueError("CPU B4 must use its isolated loopback server at 127.0.0.1:8082")
    if _git("rev-parse", "origin/main") != BASE_MAIN:
        raise ValueError("B4 must start from the specified origin/main commit")
    if _git("merge-base", "HEAD", "origin/main") != BASE_MAIN:
        raise ValueError("current B4 branch does not descend from the specified base")
    if _git("branch", "--show-current") != BRANCH:
        raise ValueError("B4 protocol must be frozen on its named branch")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("all protocol/code/tests/docs must be committed before freezing")
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            *CODE_PATHS,
            PRIOR_GPU_ATTEMPT_PATH.relative_to(ROOT).as_posix(),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise ValueError("all B4 protocol/code/tests/docs must be tracked before freezing")
    for private_marker in (
        private_root / "call_ledger.jsonl",
        private_root / "execution_plan.json",
        private_root / "smoke" / "smoke_report.json",
    ):
        if private_marker.exists():
            raise FileExistsError("B4 private execution artifacts exist before protocol freeze")
    if not MODEL_PATH.is_file() or not executable_path.is_file():
        raise FileNotFoundError("pinned Qwen3 model and llama-server executable must exist")
    if len(executable_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in executable_sha256
    ):
        raise ValueError("provide the SHA-256 computed for the exact llama-server executable")

    client = LlamaServerClient(server_url)
    client.health()
    props = client.props()
    b2 = _read_json(B2_LOCK)
    port = int(server_url.rsplit(":", 1)[1])
    process = _inspect_server_process(port)
    process_executable = str(Path(process["ExecutablePath"]).resolve())
    requested_executable = str(executable_path.resolve())
    if process_executable.casefold() != requested_executable.casefold():
        raise ValueError("CPU loopback port is not owned by the requested llama-server binary")
    command_line = process.get("CommandLine", "")
    observed_flags = {
        "n_gpu_layers_0": "--n-gpu-layers 0" in command_line,
        "no_op_offload": "--no-op-offload" in command_line,
        "flash_attention": "--flash-attn on" in command_line,
        "parallel_1": "--parallel 1" in command_line,
        "q4_k_cache": "--cache-type-k q4_0" in command_line,
        "q4_v_cache": "--cache-type-v q4_0" in command_line,
        "threads_12": "--threads 12" in command_line,
        "threads_batch_12": "--threads-batch 12" in command_line,
    }
    if not all(observed_flags.values()):
        raise ValueError("CPU server is not the observed zero-GPU-offload B4 runtime")
    actual_model_path = str(props.get("model_path", ""))
    if actual_model_path.replace("/", "\\").casefold() != str(MODEL_PATH).casefold():
        raise ValueError("CPU server model path does not match the frozen Qwen3 GGUF")
    model_sha = sha256_file(MODEL_PATH)
    if model_sha != b2["answer_model"]["sha256"]:
        raise ValueError("Qwen3 model SHA differs from frozen B2")
    if props.get("build_info") != "b10068-571d0d540":
        raise ValueError("shared llama.cpp build differs from the expected B2 version")
    server_context = props.get("default_generation_settings", {}).get("n_ctx")
    if server_context != CONTEXT_SIZE:
        raise ValueError("CPU server context capacity must equal the frozen 32768-token contract")

    prior_attempt = _read_json(PRIOR_GPU_ATTEMPT_PATH)
    prior_root = Path(prior_attempt["private_root"])
    prior_arm = prior_attempt.get("completed_arm", {})
    prior_arm_path = (
        prior_root
        / "arms"
        / prior_arm.get("case_id", "")
        / prior_arm.get("action", "")
        / f"{prior_arm.get('run_id', '')}.json"
    )
    prior_attempt_sha256 = sha256_file(PRIOR_GPU_ATTEMPT_PATH)
    if (
        prior_root.resolve() != Path(r"D:\MyLab\Jianli\external\rag_e5\e5b4").resolve()
        or prior_attempt.get("attempt_id") != "e5b4-gpu-shared-20260930"
        or prior_attempt.get("status") != "ABANDONED_INCOMPLETE_NOT_SCORED"
        or prior_attempt.get("model_calls_completed") != 1
        or prior_attempt.get("completed_arm_count") != 1
        or prior_attempt.get("execution_manifest_created") is not False
        or prior_attempt.get("teacher_opened") is not False
        or prior_attempt.get("scoring_run") is not False
        or sha256_file(ROOT / prior_attempt["protocol_lock_path"])
        != prior_attempt.get("protocol_lock_file_sha256")
        or sha256_file(prior_root / "smoke" / "smoke_report.json")
        != prior_attempt.get("smoke_report_sha256")
        or sha256_file(prior_root / "call_ledger.jsonl") != prior_attempt.get("call_ledger_sha256")
        or sha256_file(prior_root / "execution_plan.json") != prior_attempt.get("execution_plan_file_sha256")
        or sha256_file(prior_arm_path) != prior_attempt.get("completed_arm", {}).get("file_sha256")
    ):
        raise ValueError("the superseded GPU partial attempt changed or is not safely unscored")

    try:
        protocol_relative = output_path.resolve().relative_to(ROOT.resolve()).as_posix()
        manifest_relative = execution_manifest_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("CPU lock and execution manifest must live inside the repository") from exc

    frozen = load_verified_b2({
        "b2_protocol_lock_sha256": b2["counterfactual_lock_sha256"],
        "b2_protocol_lock_file_sha256": sha256_file(B2_LOCK),
        "b2_execution_manifest_file_sha256": sha256_file(B2_EXECUTION),
        "b2_report_file_sha256": sha256_file(B2_REPORT),
        "task_template_sha256": sha256_file(PRIVATE_ROOT / "task_templates.json"),
        "b2_artifact_set_sha256": "e528263a5141e6ac476252a9bfb04a1a95fc53475c5617bec23fb106980168f1",
        "b2_reuse_manifest_sha256": "ef170dc19813c2cb6c5325a7208de1778bebb4a10883f9b4d9a5ab3f055c20d1",
        "task_set_sha256": b2["ordered_case_ids_sha256"],
        "state_set_sha256": b2["state_packet_set_sha256"],
        "corpus_chunks_sha256": b2["external_corpus_chunks_sha256"],
        "runtime_cases_sha256": b2["runtime_cases_file_sha256"],
        "question_set_sha256": b2["question_set_sha256"],
    })
    task_kinds = [classify_runtime_task(row) for row in frozen.runtime_cases]
    task_counts = {kind: task_kinds.count(kind) for kind in sorted(set(task_kinds))}
    if task_counts != {"GUIDANCE_ONLY": 20, "STATE_AND_GUIDANCE": 20, "STATE_ONLY": 20}:
        raise ValueError("B4 runtime-visible 20/20/20 task classification did not pass")
    b3_identity = read_and_verify_b3_identity()

    code_rows = []
    for relative in CODE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required B4 protocol source is absent: {relative}")
        code_rows.append({"path": relative, "sha256": sha256_file(path)})
    b2_report = _read_json(B2_REPORT)
    lock_body = {
        "schema_version": "rag-e5-e5b4-protocol-lock-v1",
        "status": "FROZEN_BEFORE_B4_GUIDANCE_CALLS",
        "execution_id": "e5b4-cpu-20260930",
        "protocol_lock_path": protocol_relative,
        "execution_manifest_path": manifest_relative,
        "superseded_gpu_attempt_path": PRIOR_GPU_ATTEMPT_PATH.relative_to(ROOT).as_posix(),
        "superseded_gpu_attempt_file_sha256": prior_attempt_sha256,
        "base_main": BASE_MAIN,
        "branch": BRANCH,
        "protocol_source_commit": _git("rev-parse", "HEAD"),
        "source_cohort": "202607_only",
        "202608_opened": False,
        "task_set_sha256": frozen.lock["ordered_case_ids_sha256"],
        "state_set_sha256": frozen.lock["state_packet_set_sha256"],
        "runtime_cases_sha256": frozen.lock["runtime_cases_file_sha256"],
        "question_set_sha256": frozen.lock["question_set_sha256"],
        "task_template_sha256": sha256_file(PRIVATE_ROOT / "task_templates.json"),
        "task_kind_counts_runtime_visible": task_counts,
        "b2_protocol_lock_sha256": frozen.lock["counterfactual_lock_sha256"],
        "b2_protocol_lock_file_sha256": frozen.b2_lock_file_sha256,
        "b2_execution_manifest_file_sha256": frozen.b2_execution_manifest_file_sha256,
        "b2_artifact_set_sha256": frozen.artifact_set_sha256,
        "b2_reuse_manifest_sha256": frozen.reuse_manifest_sha256,
        "corpus_chunks_sha256": frozen.lock["external_corpus_chunks_sha256"],
        "b2_report_file_sha256": sha256_file(B2_REPORT),
        "b2_retrieval_coverage_snapshot": b2_report["retrieval_target_matrix"],
        **b3_identity,
        "model": {
            "repository": frozen.lock["answer_model"]["model"],
            "revision": frozen.lock["answer_model"]["revision"],
            "file": frozen.lock["answer_model"]["file"],
            "bytes": frozen.lock["answer_model"]["bytes"],
            "sha256": model_sha,
            "path": str(MODEL_PATH),
            "quantization": "Q4_K_M",
        },
        "runtime": {
            "backend": "cpu",
            "server_url": server_url,
            "server_pid": int(process["ProcessId"]),
            "server_executable": process_executable,
            "server_executable_sha256": executable_sha256.lower(),
            "server_executable_sha256_method": "PowerShell Get-FileHash -Algorithm SHA256",
            "model_path": str(MODEL_PATH),
            "llama_cpp_build_info": props["build_info"],
            "server_context_capacity": server_context,
            "experimental_context_size": CONTEXT_SIZE,
            "max_prompt_tokens": CONTEXT_SIZE - MAX_OUTPUT_TOKENS - CHAT_TEMPLATE_OVERHEAD_RESERVE,
            "chat_template_overhead_reserve": CHAT_TEMPLATE_OVERHEAD_RESERVE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "reasoning": "disabled via chat_template_kwargs.enable_thinking=false",
            "prompt_cache": False,
            "retry": 0,
            "gpu_offload_layers": 0,
            "op_offload": False,
            "cpu_threads": CPU_THREADS,
            "flash_attention": True,
            "cache_type_k": "q4_0",
            "cache_type_v": "q4_0",
            "parallel_slots": 1,
            "shared_existing_server": False,
            "isolated_cpu_server": True,
            "server_process_command_line": command_line,
            "observed_flags": observed_flags,
            "cpu": _cpu_snapshot(),
            "gpu_offload_contract": "n-gpu-layers=0 and no-op-offload; inference tensors remain on CPU",
        },
        "method": {
            "state_materialization": "requested recent_measurement_trends field copied deterministically from LongitudinalStatePacket v4",
            "t0_model_calls": 0,
            "guidance_calls": 120,
            "generation_call_count_per_guidance_arm": 1,
            "new_retrieval_calls": 0,
            "new_bridge_calls": 0,
            "actions": ["OFF", "STANDARD", "STRONG"],
            "execution_order": "sorted_case_id_then_OFF_STANDARD_STRONG",
            "evidence_source": "exact frozen B2 top-five supplied_chunks for STANDARD/STRONG; OFF receives none",
            "plain_text_only": True,
            "json_schema_or_function_calling": False,
            "citation_aliases": "E1-E5 assigned by frozen supplied-chunk rank order; only issued aliases resolve",
            "evaluator_action_blind": True,
        },
        "prompt": {
            "system_prompt_sha256": SYSTEM_PROMPT_SHA256,
            "guidance_prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        },
        "code_sha256": code_rows,
        "teacher_opened_before_generation": False,
        "counterfactual_scoring_after_manifest_freeze": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    lock = dict(lock_body)
    lock["protocol_lock_sha256"] = canonical_sha256(lock_body)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(lock, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--server-executable-sha256", required=True)
    parser.add_argument("--server-executable", type=Path, default=Path(
        r"C:\Users\bubblevan\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
    ))
    parser.add_argument("--output", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_B4_PRIVATE_ROOT)
    parser.add_argument(
        "--manifest-output", type=Path, default=ROOT / "runs/rag_e5/e5b4_cpu_execution_manifest.json"
    )
    args = parser.parse_args()
    result = freeze(
        server_url=args.server_url,
        executable_path=args.server_executable,
        executable_sha256=args.server_executable_sha256,
        output_path=args.output,
        private_root=args.private_root,
        execution_manifest_path=args.manifest_output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
