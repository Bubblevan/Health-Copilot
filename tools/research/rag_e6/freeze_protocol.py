"""Freeze the strict-output CFEC-v1.3 protocol after the BUILD gate passes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.protocol import create_protocol_lock
from eval.rag_e6.reader_executor import DEFAULT_RUNTIME_CORPUS_ROOT
from eval.rag_e6.scoring import (
    BUILD_ROOT,
    REPORT_JSON,
    SCORED_ROWS,
    SPLIT_MANIFEST,
    U2F_ROOT,
    _read_json,
    _verify_execution_freeze,
)
from eval.rag_e6.split import sha256_file, write_immutable_json

LOCK_PATH = ROOT / "runs/rag_e6/protocol_lock.json"


def freeze_protocol() -> dict[str, object]:
    if LOCK_PATH.exists():
        raise FileExistsError("E6A protocol lock already exists; refusing to replace it")
    frozen_dev_manifest = ROOT / "runs/rag_e6/frozen_dev/frozen_dev_manifest.json"
    if frozen_dev_manifest.exists():
        raise FileExistsError("FROZEN_DEV runtime already exists; protocol freeze is too late")
    if subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"], cwd=ROOT, check=False
    ).returncode != 0:
        raise RuntimeError("commit all method and scoring code before freezing the E6A protocol")

    manifest, _, _, _ = _verify_execution_freeze(
        u2f_root=U2F_ROOT,
        split_manifest_path=SPLIT_MANIFEST,
        corpus_root=DEFAULT_RUNTIME_CORPUS_ROOT,
        build_root=BUILD_ROOT,
        partition="BUILD",
    )
    report = _read_json(REPORT_JSON)
    if (
        report.get("runtime_manifest_sha256") != sha256_file(BUILD_ROOT / "build_manifest.json")
        or report.get("reader_output_sha256") != manifest.get("reader_output_sha256")
        or report.get("generation_call_journal_sha256")
        != manifest.get("generation_call_journal_sha256")
        or report.get("scored_episode_rows_sha256") != sha256_file(SCORED_ROWS)
        or report.get("truth_access", {}).get("frozen_dev_truth_rows_decoded") != 0
    ):
        raise ValueError("BUILD score report is not bound to the complete frozen BUILD run")
    scored_rows_hash = report["scored_episode_rows_sha256"]
    lock = create_protocol_lock(
        build_manifest=manifest,
        build_report=report,
        build_manifest_sha256=sha256_file(BUILD_ROOT / "build_manifest.json"),
        build_report_sha256=sha256_file(REPORT_JSON),
        scored_rows_sha256=scored_rows_hash,
        method_code_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    write_immutable_json(LOCK_PATH, lock)
    return lock


def main() -> None:
    lock = freeze_protocol()
    print(json.dumps({
        "lock_path": str(LOCK_PATH),
        "selected_method": lock["selected_method"],
        "method_code_commit": lock["method_code_commit"],
        "primary_arm": lock["primary_arm"],
        "build_manifest_sha256": lock["build_selection_evidence"]["build_manifest_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
