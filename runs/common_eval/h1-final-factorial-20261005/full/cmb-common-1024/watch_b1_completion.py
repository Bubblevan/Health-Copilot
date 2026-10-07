#!/usr/bin/env python3
"""Watch B1 completion and queue a continuation into this Codex thread."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
B1_MANIFEST = RUN_ROOT / "B1" / "run_manifest.json"
B2_MANIFEST = RUN_ROOT / "B2" / "run_manifest.json"
NOTICE_PATH = RUN_ROOT / "b1-completion-wake.json"
POLL_SECONDS = 30
B2_START_GRACE_SECONDS = 20 * 60
LOG_PATH = RUN_ROOT / "b1-completion-watcher.log"
RUNNER_MARKER = b"tools/eval/run_common_eval.py"
RUN_ROOT_MARKER = b"h1-final-factorial-20261005/full/cmb-common-1024"


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_notice(value: dict[str, Any]) -> None:
    temporary = NOTICE_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(NOTICE_PATH)


def _log(message: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(UTC).isoformat()} {message}\n")
        stream.flush()


def _runner_pids() -> list[int]:
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            command = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (
            command
            and Path(os.fsdecode(command[0])).name.startswith("python")
            and RUNNER_MARKER in command
            and RUN_ROOT_MARKER in b"\0".join(command)
        ):
            found.append(int(entry.name))
    return found


def main() -> int:
    if NOTICE_PATH.exists():
        _log("existing wake notice found; exiting")
        return 0
    if (RUN_ROOT / "B1" / "coverage_smoke_classification.json").exists():
        _log("B1 was classified as coverage smoke; scheduled B2 wake is disabled")
        return 0

    _log(f"watching {B1_MANIFEST}; pid={os.getpid()}")

    missing_runner_polls = 0
    while True:
        b1 = _read_manifest(B1_MANIFEST)
        b1_status = b1.get("status") if b1 else None
        if b1_status in {"COMPLETE", "FAILED", "ABORTED"}:
            break
        runner_pids = _runner_pids()
        missing_runner_polls = 0 if runner_pids else missing_runner_polls + 1
        if missing_runner_polls >= 3:
            b1_status = "RUNNER_EXITED_WITH_NONTERMINAL_MANIFEST"
            _log("runner process is absent for three polls before B1 reached a terminal status")
            break
        time.sleep(POLL_SECONDS)

    detected_at = datetime.now(UTC).isoformat()
    deadline = time.monotonic() + B2_START_GRACE_SECONDS
    b2: dict[str, Any] | None = None
    while b1_status == "COMPLETE" and time.monotonic() < deadline:
        b2 = _read_manifest(B2_MANIFEST)
        if b2 and b2.get("status") in {"RUNNING", "COMPLETE", "FAILED"}:
            break
        if not _runner_pids():
            break
        time.sleep(POLL_SECONDS)

    b1 = _read_manifest(B1_MANIFEST) or {}
    b2 = _read_manifest(B2_MANIFEST)
    b2_status = b2.get("status") if b2 else "NOT_STARTED_WITHIN_GRACE_PERIOD"
    notice: dict[str, Any] = {
        "schema_version": "h1-b1-completion-wake-v1",
        "detected_at_utc": detected_at,
        "b1_status": b1.get("status") if b1 else "MANIFEST_UNAVAILABLE",
        "b1_completed_at_utc": b1.get("completed_at_utc"),
        "b1_case_count": (b1.get("dataset") or {}).get("expected_case_count"),
        "b2_status_at_detection": b2_status,
        "runner_pids_at_detection": _runner_pids(),
        "queue_attempted": False,
        "queue_succeeded": False,
    }

    thread_id = os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_SESSION_ID")
    if thread_id:
        if b1_status == "COMPLETE":
            message = (
                "Scheduled H1 handoff: the full CMB B1 arm has reached COMPLETE. "
                f"B2 status at detection: {b2_status}. The existing --matrix core runner "
                "should start B2 automatically after B1; do not launch a duplicate. "
                "Check the active runner and B2 manifest, keep the pinned code/model/runtime "
                "identity unchanged, and continue monitoring the factorial."
            )
        else:
            message = (
                "Scheduled H1 alert: B1 did not reach COMPLETE; observed status "
                f"{b1_status or 'unavailable'}. B2 status: {b2_status}. Inspect the run "
                "manifest, console log, and runner state before taking any action."
            )
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            result = subprocess.run(
                ["codex", "queue", "--thread", thread_id, "--message", message],
                capture_output=True,
                text=True,
                timeout=45,
                check=False,
            )
            attempts.append(
                {
                    "attempt": attempt,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-1000:],
                    "at_utc": datetime.now(UTC).isoformat(),
                }
            )
            if result.returncode == 0:
                notice["queue_succeeded"] = True
                _log(f"queued follow-up to thread; b1={b1_status}, b2={b2_status}")
                break
            _log(f"queue attempt {attempt} failed with return code {result.returncode}")
            time.sleep(60)
        notice["queue_attempted"] = True
        notice["queue_attempts"] = attempts
    else:
        notice["queue_error"] = "CODEX_THREAD_ID and CODEX_SESSION_ID were absent"

    _write_notice(notice)
    _log(f"notice written; queue_succeeded={notice['queue_succeeded']}")
    return 0 if notice["queue_succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
