#!/usr/bin/env python3
"""Run the final no-RAG B0/B2 pair without invoking B1 or B3."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = Path(__file__).resolve().parent
PYTHON = Path("/root/gpufree-data/Health-Copilot/training/posttrain/.venv/bin/python")
STAGES = (
    ("diagnosisarena-915", "diagnosisarena", "B0", 915, False),
    ("diagnosisarena-915", "diagnosisarena", "B2", 915, True),
    ("cmb-common-1024", "cmb-common", "B2", 1024, True),
)


def save_state(state: dict) -> None:
    temporary = RUN_ROOT / "orchestrator-status.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(RUN_ROOT / "orchestrator-status.json")


def queue_handoff(message: str) -> dict:
    thread_id = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID")
    if not thread_id:
        return {"attempted": False, "error": "Codex thread ID unavailable"}
    try:
        result = subprocess.run(
            ["codex", "queue", "--thread", thread_id, "--message", message],
            text=True, capture_output=True, timeout=60, check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "attempted": True,
        "returncode": result.returncode,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-1000:],
    }


def main() -> int:
    environment = os.environ.copy()
    env_root = Path("/root/gpufree-data/Health-Copilot/training/posttrain/.venv")
    environment["PATH"] = f"{env_root / 'bin'}:{environment.get('PATH', '')}"
    environment["PYTHONPATH"] = f"/root/gpufree-share/venvs/common-kb-v1-packages:{ROOT / 'src'}"
    environment["VLLM_BASE_URL"] = "http://127.0.0.1:8001/v1"
    environment["VLLM_MODEL_NAME"] = "qwen3-8b-system-v1"
    state = {
        "schema_version": "h1-final-ma-pair-orchestrator-v1",
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "RUNNING",
        "stages": [],
        "b1_b3": "CANCELLED",
        "rag": "CLOSED_FOR_COMMON_EVAL_FACTORIAL",
    }
    save_state(state)
    for dataset_dir, dataset, profile, expected_count, diagnostics in STAGES:
        output_dir = RUN_ROOT / dataset_dir / profile
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            existing_count = old.get("dataset", {}).get("expected_case_count")
            if old.get("status") == "COMPLETE" and existing_count == expected_count:
                state["stages"].append({
                    "dataset": dataset_dir, "profile": profile,
                    "status": "REUSED_COMPLETE", "expected_cases": expected_count,
                    "completed_at_utc": old.get("completed_at_utc"),
                })
                save_state(state)
                continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), "-u", str(ROOT / "tools/eval/run_common_eval.py"),
            "--dataset", dataset, "--profile", profile,
            "--model-config", str(ROOT / "configs/models/qwen3_8b_base_system_eval.json"),
            "--vllm-runtime-config", str(RUN_ROOT / "vllm-runtime-config.json"),
            "--prepared-config", str(ROOT / "configs/eval/local_prepared_views_h0.json"),
            "--output", str(output_dir), "--concurrency", "2", "--resume",
        ]
        if diagnostics:
            command.append("--capture-adaptive-diagnostics")
        stage = {
            "dataset": dataset_dir, "profile": profile,
            "expected_cases": expected_count, "status": "RUNNING",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "command": command,
        }
        state["stages"].append(stage)
        save_state(state)
        log_path = output_dir / "runner.log"
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        stage["returncode"] = result.returncode
        stage["finished_at_utc"] = datetime.now(UTC).isoformat()
        stage["status"] = "COMPLETE" if result.returncode == 0 else "FAILED"
        save_state(state)
        if result.returncode != 0:
            state["status"] = "FAILED"
            state["failed_stage"] = stage
            state["failure_handoff"] = queue_handoff(
                f"H1 final B0/B2 run stopped at {dataset_dir} {profile} (exit {result.returncode}). "
                "Check its checkpoint and runner log; do not launch B1/B3 or duplicate completed arms."
            )
            save_state(state)
            return result.returncode
    state["status"] = "COMPLETE"
    state["completed_at_utc"] = datetime.now(UTC).isoformat()
    state["completion_handoff"] = queue_handoff(
        "H1 final no-RAG B0/B2 run is complete for DiagnosisArena-915 and CMB-COMMON-1024. "
        "Analyze paired transitions, McNemar, parse/runtime/safety outcomes, calls/tokens/latency, "
        "and MDT complexity slices. Then freeze the MA closeout and hand the GPU to medical SFT/GSPO/GDPO."
    )
    save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
