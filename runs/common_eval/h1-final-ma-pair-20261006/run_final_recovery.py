#!/usr/bin/env python3
"""Recover final no-RAG B2 arms, preserving valid checkpoints and auditing failures."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = Path(__file__).resolve().parent
VENV = Path("/root/gpufree-data/Health-Copilot/training/posttrain/.venv")
PYTHON = VENV / "bin/python"
VLLM = VENV / "bin/vllm"
MODEL_PATH = Path("/root/gpufree-share/data/Qwen3-8B")
MODEL_CONFIG = ROOT / "configs/models/qwen3_8b_base_system_eval.json"
PREPARED_CONFIG = ROOT / "configs/eval/local_prepared_views_h0.json"
TEMPLATE_CONFIG = RUN_ROOT / "vllm-runtime-config.json"
BASE_URL = "http://127.0.0.1:8001"
SERVED_MODEL = "qwen3-8b-system-v1"
EXPECTED = {"diagnosisarena-915": 915, "cmb-common-1024": 1024}
STAGES = (
    ("diagnosisarena-915", "diagnosisarena", "B2", RUN_ROOT / "diagnosisarena-915/B2"),
    ("cmb-common-1024", "cmb-common", "B2", None),
)
MAX_ATTEMPTS = 4


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def service_healthy() -> bool:
    try:
        with urlopen(f"{BASE_URL}/health", timeout=4) as response:
            return response.status == 200
    except (OSError, URLError, TimeoutError):
        return False


def port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", 8001)) == 0


class LocalVllm:
    def __init__(self, state: dict):
        self.state = state
        self.proc: subprocess.Popen | None = None
        self.log_handle = None
        self.generation = int((state.get("vllm") or {}).get("generation", 0))
        self.runtime_config_path: Path | None = None

    def start(self) -> None:
        if self.proc is not None and self.proc.poll() is None and service_healthy():
            return
        if port_open():
            raise RuntimeError("127.0.0.1:8001 is occupied but is not a healthy owned vLLM service")
        self.generation += 1
        log_path = RUN_ROOT / f"vllm-8001-recovery-{self.generation}.log"
        self.log_handle = log_path.open("ab", buffering=0)
        env = os.environ.copy()
        env["PATH"] = f"{VENV / 'bin'}:{env.get('PATH', '')}"
        command = [
            str(VLLM), "serve", str(MODEL_PATH),
            "--served-model-name", SERVED_MODEL,
            "--host", "127.0.0.1", "--port", "8001",
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.5",
            "--max-model-len", "16384",
            "--max-num-seqs", "16",
            "--max-num-batched-tokens", "16384",
            "--kv-cache-dtype", "auto",
            "--no-enable-prefix-caching",
            "--seed", "20261004",
            "--structured-outputs-config", '{"backend":"xgrammar","disable_any_whitespace":true}',
        ]
        self.proc = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=self.log_handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.state["vllm"] = {
            "pid": self.proc.pid, "generation": self.generation,
            "log": str(log_path), "status": "STARTING", "started_at_utc": now(),
        }
        write_json(RUN_ROOT / "final-recovery-status.json", self.state)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited during startup with code {self.proc.returncode}; see {log_path}")
            if service_healthy():
                self.runtime_config_path = self._write_runtime_config()
                self.state["vllm"].update({"status": "HEALTHY", "healthy_at_utc": now()})
                write_json(RUN_ROOT / "final-recovery-status.json", self.state)
                return
            time.sleep(5)
        raise TimeoutError(f"vLLM did not become healthy within 900 seconds; see {log_path}")

    def _write_runtime_config(self) -> Path:
        config = json.loads(TEMPLATE_CONFIG.read_text())
        config["current_server_pid"] = self.proc.pid if self.proc else None
        config["captured_at_utc"] = now()
        config["base_url"] = f"{BASE_URL}/v1"
        config["served_model"] = SERVED_MODEL
        config["recovery_generation"] = self.generation
        config["memory_server_port_8000"] = "not accessed or modified by this run"
        log_path = RUN_ROOT / f"vllm-8001-recovery-{self.generation}.log"
        startup_log = log_path.read_text(encoding="utf-8", errors="replace")
        cache_gib = re.search(r"GPU KV cache size:\s*([0-9.]+)\s*GiB", startup_log)
        cache_tokens = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", startup_log)
        if cache_gib:
            config["available_kv_cache_gib"] = float(cache_gib.group(1))
        if cache_tokens:
            config["kv_cache_tokens"] = int(cache_tokens.group(1).replace(",", ""))
            context = int(config.get("max_model_len", 1))
            config["max_full_context_concurrency"] = round(
                int(config["kv_cache_tokens"]) / context, 2,
            )
        path = RUN_ROOT / f"vllm-runtime-config-recovery-{self.generation}.json"
        write_json(path, config)
        return path

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def flags(row: dict) -> list[str]:
    return [str(x) for x in row.get("response", {}).get("safety_flags", ())]


def has_transport_failure(row: dict) -> bool:
    return any("APIConnectionError" in flag or "APITimeoutError" in flag for flag in flags(row))


def has_any_reasoning_failure(row: dict) -> bool:
    return any(flag.startswith("reasoning_failure:") for flag in flags(row))


def seed_output(source: Path, target: Path, *, initial_buggy_attempt: bool) -> dict:
    source_cases = read_rows(source / "cases.jsonl")
    if not source_cases:
        return {"source": str(source), "reused_cases": 0, "excluded_cases": 0}
    kept = []
    excluded = []
    for row in source_cases:
        failed_route = (row.get("trace_summary") or {}).get("mdt_complexity") == "failed"
        should_exclude = has_transport_failure(row)
        if initial_buggy_attempt:
            should_exclude = should_exclude or has_any_reasoning_failure(row) or failed_route
        if should_exclude:
            excluded.append(row)
        else:
            kept.append(row)
    target.mkdir(parents=True, exist_ok=False)
    with (target / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    trace_ids = {str(row.get("response", {}).get("trace_id")) for row in kept}
    written_traces = 0
    trace_path = source / "traces.jsonl"
    if trace_path.exists():
        with trace_path.open(encoding="utf-8") as incoming, (target / "traces.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as outgoing:
            for line in incoming:
                if not line.strip():
                    continue
                trace = json.loads(line)
                if str(trace.get("trace_id")) in trace_ids:
                    outgoing.write(line if line.endswith("\n") else line + "\n")
                    written_traces += 1
    source_manifest = source / "run_manifest.json"
    provenance = {
        "schema_version": "h1-b2-recovery-seed-v1",
        "created_at_utc": now(),
        "source_output": str(source),
        "source_manifest_sha256": digest(source_manifest) if source_manifest.exists() else None,
        "source_case_rows": len(source_cases),
        "reused_case_rows": len(kept),
        "excluded_case_rows": len(excluded),
        "excluded_reasoning_failure_rows": sum(has_any_reasoning_failure(r) for r in excluded),
        "excluded_transport_failure_rows": sum(has_transport_failure(r) for r in excluded),
        "kept_case_ids_sha256": hashlib.sha256(
            "\n".join(str(r["case_id"]) for r in kept).encode()
        ).hexdigest() if kept else None,
        "written_trace_rows": written_traces,
        "initial_buggy_attempt": initial_buggy_attempt,
        "selection_rule": (
            "reuse only rows with no reasoning failure and no failed MDT route from the invalid first B2 attempt"
            if initial_buggy_attempt else
            "reuse all rows except transport failures from the immediately prior recovery attempt"
        ),
    }
    write_json(target / "recovery_seed.json", provenance)
    return provenance


def checkpoint_state(directory: Path) -> tuple[int, list[dict]]:
    rows = read_rows(directory / "cases.jsonl")
    return len(rows), rows


def run_attempt(
    *, vllm: LocalVllm, state: dict, dataset_dir: str, dataset: str,
    profile: str, attempt: int, output_dir: Path,
) -> tuple[int, list[dict]]:
    assert vllm.runtime_config_path is not None
    env = os.environ.copy()
    env["PATH"] = f"{VENV / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"/root/gpufree-share/venvs/common-kb-v1-packages:{ROOT / 'src'}"
    env["VLLM_BASE_URL"] = f"{BASE_URL}/v1"
    env["VLLM_MODEL_NAME"] = SERVED_MODEL
    command = [
        str(PYTHON), "-u", str(ROOT / "tools/eval/run_common_eval.py"),
        "--dataset", dataset, "--profile", profile,
        "--model-config", str(MODEL_CONFIG),
        "--vllm-runtime-config", str(vllm.runtime_config_path),
        "--prepared-config", str(PREPARED_CONFIG),
        "--output", str(output_dir), "--concurrency", "2", "--resume",
    ]
    if profile == "B2":
        command.append("--capture-adaptive-diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "runner.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        state["active_stage"] = {
            "dataset": dataset_dir, "profile": profile, "attempt": attempt,
            "status": "RUNNING", "output": str(output_dir),
            "pid": process.pid, "started_at_utc": now(),
            "command": command,
        }
        write_json(RUN_ROOT / "final-recovery-status.json", state)
        last_health_check = time.monotonic()
        consecutive_unhealthy = 0
        last_progress_check = 0.0
        while process.poll() is None:
            time.sleep(15)
            if time.monotonic() - last_health_check >= 15:
                last_health_check = time.monotonic()
                if service_healthy():
                    consecutive_unhealthy = 0
                else:
                    consecutive_unhealthy += 1
                    state["active_stage"]["health_failures"] = consecutive_unhealthy
                    state["active_stage"]["last_health_failure_utc"] = now()
                    if consecutive_unhealthy >= 2:
                        state["active_stage"]["status"] = "INTERRUPTED_VLLM_UNHEALTHY"
                        write_json(RUN_ROOT / "final-recovery-status.json", state)
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=20)
                        except (ProcessLookupError, subprocess.TimeoutExpired):
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        break
            if time.monotonic() - last_progress_check >= 60:
                last_progress_check = time.monotonic()
                count = sum(1 for _ in (output_dir / "cases.jsonl").open(encoding="utf-8")) \
                    if (output_dir / "cases.jsonl").exists() else 0
                state["active_stage"]["checkpoint_rows"] = count
                state["active_stage"]["checkpoint_updated_at_utc"] = now()
                write_json(RUN_ROOT / "final-recovery-status.json", state)
        return int(process.returncode if process.returncode is not None else -signal.SIGTERM), read_rows(output_dir / "cases.jsonl")


def run_stage(vllm: LocalVllm, state: dict, dataset_dir: str, dataset: str, profile: str, source: Path | None) -> Path:
    seed_source = source
    initial_buggy_attempt = source is not None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        vllm.start()
        output = RUN_ROOT / dataset_dir / f"{profile}-final-attempt-{attempt}"
        if output.exists():
            raise FileExistsError(f"recovery output already exists: {output}")
        if seed_source is not None:
            seed = seed_output(seed_source, output, initial_buggy_attempt=initial_buggy_attempt)
        else:
            output.mkdir(parents=True)
            seed = None
        state["active_stage"] = {
            "dataset": dataset_dir, "profile": profile, "attempt": attempt,
            "status": "PREPARED", "output": str(output), "seed": seed,
        }
        write_json(RUN_ROOT / "final-recovery-status.json", state)
        returncode, rows = run_attempt(
            vllm=vllm, state=state, dataset_dir=dataset_dir, dataset=dataset,
            profile=profile, attempt=attempt, output_dir=output,
        )
        unique = {str(row.get("case_id")): row for row in rows}
        transport = [row for row in rows if has_transport_failure(row)]
        expected = EXPECTED[dataset_dir]
        status = {
            "dataset": dataset_dir, "profile": profile, "attempt": attempt,
            "output": str(output), "returncode": returncode,
            "case_rows": len(rows), "unique_case_rows": len(unique),
            "expected_case_rows": expected, "transport_failure_rows": len(transport),
            "completed_at_utc": now(),
        }
        state.setdefault("attempts", []).append(status)
        if returncode == 0 and len(rows) == expected and len(unique) == expected and not transport:
            status["status"] = "COMPLETE"
            state["active_stage"] = status
            state.setdefault("final_outputs", {})[dataset_dir] = str(output)
            write_json(RUN_ROOT / "final-recovery-status.json", state)
            return output
        status["status"] = "RETRYING_TRANSPORT_FAILURES" if transport or returncode != 0 else "FAILED_CASE_COUNT"
        state["active_stage"] = status
        write_json(RUN_ROOT / "final-recovery-status.json", state)
        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"{dataset_dir} did not produce a clean complete case set after {MAX_ATTEMPTS} attempts")
        seed_source = output
        initial_buggy_attempt = False
        # A failed EngineCore process is only replaced after the active runner has been stopped.
        if not service_healthy():
            vllm.stop()
            vllm.proc = None
            vllm.start()
    raise AssertionError("unreachable")


def queue_handoff(message: str) -> dict:
    thread_id = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID")
    if not thread_id:
        return {"attempted": False, "reason": "Codex thread/session ID unavailable"}
    try:
        result = subprocess.run(
            ["codex", "queue", "--thread", thread_id, "--message", message],
            capture_output=True, text=True, timeout=60, check=False,
        )
        return {"attempted": True, "returncode": result.returncode, "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:]}
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmb-only", action="store_true",
        help="Run only CMB B2 and preserve the current DiagnosisArena recovery as paused.",
    )
    args = parser.parse_args()
    status_path = RUN_ROOT / "final-recovery-status.json"
    prior_state = json.loads(status_path.read_text()) if status_path.exists() else {}
    if args.cmb_only:
        prior_copy = RUN_ROOT / "diagnosisarena-paused-before-cmb-only.json"
        if not prior_copy.exists():
            write_json(prior_copy, prior_state)
        prior_stage = prior_state.get("active_stage") or {}
        da_output = Path(prior_stage.get("output") or RUN_ROOT / "diagnosisarena-915/B2-final-attempt-1")
        da_rows, _ = checkpoint_state(da_output)
        prior_vllm = prior_state.get("vllm") or {}
        state = {
            "schema_version": "h1-final-ma-recovery-v1",
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "started_at_utc": now(), "status": "RUNNING_CMB_ONLY",
            "operation_mode": "CMB_B2_ONLY",
            "rag": "CLOSED_FOR_COMMON_EVAL_FACTORIAL", "b1_b3": "CANCELLED",
            "diagnosisarena": {
                "status": "PAUSED_BY_USER",
                "output": str(da_output),
                "checkpoint_rows": da_rows,
                "expected_rows": EXPECTED["diagnosisarena-915"],
                "paused_at_utc": now(),
                "resume_guidance": (
                    "Resume the existing DiagnosisArena B2-final-attempt-1 output with the common-eval runner's "
                    "--resume option after validating run identity; do not start a fresh output directory."
                ),
            },
            "prior_status_snapshot": str(prior_copy),
            "prior_status": prior_state.get("status"),
            "vllm": prior_vllm,
            "attempts": [],
        }
    else:
        state = {
            "schema_version": "h1-final-ma-recovery-v1",
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "started_at_utc": now(), "status": "RUNNING",
            "rag": "CLOSED_FOR_COMMON_EVAL_FACTORIAL", "b1_b3": "CANCELLED",
            "parser_fix_test": {"command": "tests/test_adaptive_mdt_adapter.py", "result": "5 passed"},
            "original_b2_attempt": {
                "path": str(RUN_ROOT / "diagnosisarena-915/B2"),
                "status": "INVALID_TRANSPORT_AND_CLASSIFIER_PARSER_FAILURES",
                "reused_valid_cases": 318,
                "retry_cases": 597,
            },
            "attempts": [],
        }
    write_json(status_path, state)
    vllm = LocalVllm(state)
    try:
        vllm.start()
        if not args.cmb_only:
            da_out = run_stage(
                vllm, state, "diagnosisarena-915", "diagnosisarena", "B2",
                RUN_ROOT / "diagnosisarena-915/B2",
            )
            state["diagnosisarena_b2_final"] = str(da_out)
            write_json(status_path, state)
        cmb_out = run_stage(
            vllm, state, "cmb-common-1024", "cmb-common", "B2", None,
        )
        state["cmb_b2_final"] = str(cmb_out)
        state["status"] = "CMB_B2_COMPLETE_DIAGNOSISARENA_PAUSED" if args.cmb_only else "COMPLETE"
        state["completed_at_utc"] = now()
        state["completion_handoff"] = queue_handoff(
            "CMB-only no-RAG B2 run completed. Analyze paired B0 vs B2 on CMB-COMMON-1024, report runtime/parser failures separately, and leave DiagnosisArena paused unless resumed explicitly."
            if args.cmb_only else
            "Final no-RAG B0/B2 recovery completed. Analyze paired B0 vs B2 on DiagnosisArena-915 and CMB-COMMON-1024, report runtime/parser failures separately, then freeze the MA closeout and release GPU to medical SFT/GSPO/GDPO."
        )
        write_json(status_path, state)
        return 0
    except Exception as exc:  # noqa: BLE001
        state["status"] = "FAILED"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["failed_at_utc"] = now()
        state["failure_handoff"] = queue_handoff(
            f"Final B0/B2 recovery stopped: {type(exc).__name__}: {exc}. Check final-recovery-status.json and preserve all existing checkpoints."
        )
        write_json(status_path, state)
        return 1
    finally:
        vllm.stop()
        state["gpu_released_by_recovery_supervisor"] = True
        state["gpu_release_at_utc"] = now()
        write_json(status_path, state)


if __name__ == "__main__":
    raise SystemExit(main())
