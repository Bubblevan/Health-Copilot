"""One separately recorded, non-qualifying R4-P01 diagnostic repeat."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.research.memory import run_mem3b0q_r4_gate as frozen_gate
from tools.research.memory import run_mem3b0q_r4_guarded_sampler_gate_v1 as strict_overlay


REPEAT_DIR = frozen_gate.RUN_DIR / "posthoc_repeat_r4_p01_20261001"
LOCK_PATH = frozen_gate.MEMORY_DOCS / "mem3b0q_r4_p01_posthoc_repeat_lock_v1.json"
LOCK_SIDECAR_PATH = (
    frozen_gate.MEMORY_DOCS / "mem3b0q_r4_p01_posthoc_repeat_lock_v1.sha256"
)
OVERLAY_LOCK_PATH = strict_overlay.OVERLAY_LOCK_PATH


class PosthocRepeatError(RuntimeError):
    """A post-hoc repeat precondition failed closed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _load_lock() -> tuple[dict[str, Any], str]:
    lock_bytes = LOCK_PATH.read_bytes()
    lock_sha256 = _sha256(lock_bytes)
    sidecar_tokens = LOCK_SIDECAR_PATH.read_text(encoding="ascii").split()
    if not sidecar_tokens or sidecar_tokens[0].lower() != lock_sha256:
        raise PosthocRepeatError("posthoc_lock_sidecar_mismatch")
    lock = json.loads(lock_bytes.decode("utf-8"))
    historical_dir = frozen_gate.GATE_DIR
    expected = {
        "status": "FROZEN",
        "repeat_runner_sha256": _file_sha256(Path(__file__).resolve()),
        "strict_overlay_sha256": _file_sha256(Path(strict_overlay.__file__).resolve()),
        "strict_overlay_lock_sha256": _file_sha256(OVERLAY_LOCK_PATH),
        "frozen_runner_sha256": _file_sha256(Path(frozen_gate.__file__).resolve()),
        "manifest_sha256": frozen_gate.FROZEN_MANIFEST_SHA256,
        "runtime_amendment_sha256": _file_sha256(
            frozen_gate.RUNTIME_PREFLIGHT_AMENDMENT_PATH
        ),
        "historical_gate_attempt_sha256": _file_sha256(
            historical_dir / "gate_attempt.json"
        ),
        "historical_gate_manifest_sha256": _file_sha256(
            historical_dir / "gate_run_manifest.json"
        ),
        "historical_request_sha256": _file_sha256(
            historical_dir / "request_body.json"
        ),
        "historical_response_sha256": _file_sha256(
            historical_dir / "response_body.json"
        ),
        "case_id": "R4-P01",
        "max_completion_posts": 1,
        "qualification_status": "NON_QUALIFYING_POSTHOC_DIAGNOSTIC",
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise PosthocRepeatError(f"posthoc_lock_mismatch:{key}")
    return lock, lock_sha256


def run_posthoc_repeat() -> dict[str, Any]:
    if REPEAT_DIR.exists():
        raise PosthocRepeatError("posthoc_repeat_dir_exists_refusing_retry")
    lock, lock_sha256 = _load_lock()
    _, expected_body = strict_overlay._expected_p01_request()
    request_sha256 = _sha256(expected_body)
    if request_sha256 != lock.get("repeat_request_sha256"):
        raise PosthocRepeatError("repeat_request_not_exact_frozen_p01")
    strict_overlay._load_overlay_lock()

    original_gate_dir = frozen_gate.GATE_DIR
    frozen_gate.GATE_DIR = REPEAT_DIR
    guard = strict_overlay.GuardedSamplerGateOverlay(expected_body)
    try:
        with guard:
            gate_result = frozen_gate.run_gate()
    finally:
        frozen_gate.GATE_DIR = original_gate_dir

    overlay_evidence = guard.evidence()
    overlay_evidence.update(
        {
            "classification": "POSTHOC_EXPLORATORY_REPEAT_NOT_QUALIFICATION",
            "repeat_lock_sha256": lock_sha256,
            "historical_gate_attempt_sha256": lock[
                "historical_gate_attempt_sha256"
            ],
            "historical_request_sha256": lock["historical_request_sha256"],
            "historical_response_sha256": lock["historical_response_sha256"],
            "repeat_request_sha256": request_sha256,
            "completion_posts_forwarded": guard.post_attempts,
            "max_completion_posts": 1,
            "hosted_calls": 0,
            "retry_count": 0,
            "qualification_effect": "NONE_R4_V1_REMAINS_NO",
            "b1_status": "MEM3B0Q_MEM3B1_READY=NO",
        }
    )
    frozen_gate._write_json(REPEAT_DIR / "strict_overlay_manifest.json", overlay_evidence)
    artifact_sha256 = {
        path.name: _file_sha256(path)
        for path in sorted(REPEAT_DIR.iterdir())
        if path.is_file() and path.name != "posthoc_repeat_manifest.json"
    }
    repeat_manifest_path = REPEAT_DIR / "posthoc_repeat_manifest.json"
    frozen_gate._write_json(
        repeat_manifest_path,
        {
            "manifest_id": "mem3b0q-r4-p01-posthoc-repeat-v1",
            "status": "NON_QUALIFYING_POSTHOC_DIAGNOSTIC",
            "user_authorization": "去重试",
            "case_id": "R4-P01",
            "historical_attempt_dir": str(original_gate_dir),
            "repeat_attempt_dir": str(REPEAT_DIR),
            "repeat_request_sha256": request_sha256,
            "completion_posts_forwarded": guard.post_attempts,
            "request_attempts": gate_result.get("request_attempts"),
            "retry_count": 0,
            "hosted_calls": 0,
            "gate_status": gate_result.get("status"),
            "sampler_attribution": gate_result.get("sampler_gate"),
            "quality_status": gate_result.get("quality_status"),
            "quality_failure": gate_result.get("quality_failure"),
            "response_sha256": gate_result.get("response_sha256"),
            "artifact_sha256": artifact_sha256,
            "qualification_effect": "NONE_R4_V1_REMAINS_NO",
        },
    )
    (REPEAT_DIR / "posthoc_repeat_manifest.sha256").write_text(
        f"{_file_sha256(repeat_manifest_path)}  {repeat_manifest_path.name}\n",
        encoding="ascii",
    )
    return {
        "classification": "POSTHOC_EXPLORATORY_REPEAT_NOT_QUALIFICATION",
        "gate_result": gate_result,
        "overlay_evidence": overlay_evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--run-p01-posthoc-repeat",
        action="store_true",
        help="Issue one separately recorded, non-qualifying local R4-P01 diagnostic repeat.",
    )
    args = parser.parse_args()
    if not args.run_p01_posthoc_repeat:
        parser.error("explicit --run-p01-posthoc-repeat is required")
    print(
        json.dumps(
            run_posthoc_repeat(), ensure_ascii=False, sort_keys=True, indent=2
        )
    )


if __name__ == "__main__":
    main()
