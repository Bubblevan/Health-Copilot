"""Run every frozen E6A arm on the locked FROZEN_DEV partition."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.rag_e6.reader_executor import (
    DEFAULT_LLAMA_URL,
    DEFAULT_RUNTIME_CORPUS_ROOT,
    DEFAULT_SERVER_MANIFEST,
    execute_partition,
)
from eval.rag_e6.scoring import (
    BUILD_ROOT,
    REPORT_JSON,
    SPLIT_MANIFEST,
    U2F_ROOT,
    _read_json,
)
from eval.rag_e6.split import sha256_file, write_immutable_json

LOCK_PATH = ROOT / "runs/rag_e6/protocol_lock.json"
RUN_ROOT = ROOT / "runs/rag_e6/frozen_dev"
REPORT_PATH = REPORT_JSON
BUILD_MANIFEST_PATH = BUILD_ROOT / "build_manifest.json"


def main() -> None:
    if not LOCK_PATH.exists():
        raise FileNotFoundError("committed E6A protocol lock is required before FROZEN_DEV")
    if (RUN_ROOT / "frozen_dev_manifest.json").exists():
        raise FileExistsError("FROZEN_DEV runtime already exists; refusing a second run")
    lock = _read_json(LOCK_PATH)
    build_report = _read_json(REPORT_PATH)
    if (
        lock.get("schema_version") != "rag-e6a-protocol-lock-v1"
        or lock.get("selected_method") != "CFEC-v1.2"
        or lock.get("build_selection_evidence", {}).get("build_manifest_sha256")
        != sha256_file(BUILD_MANIFEST_PATH)
        or lock.get("build_selection_evidence", {}).get("build_report_sha256")
        != sha256_file(REPORT_PATH)
        or build_report.get("frozen_dev_truth_opened") is not False
    ):
        raise ValueError("protocol lock does not match the frozen, gold-blind BUILD evidence")

    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["method_code_commit"], execution_commit],
        cwd=ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("FROZEN_DEV code commit does not descend from the locked method commit")
    if subprocess.run(
        ["git", "diff-index", "--quiet", "HEAD", "--"], cwd=ROOT, check=False
    ).returncode != 0:
        raise RuntimeError("tracked code changes are not allowed after protocol freeze")

    run_context = {
        "schema_version": "rag-e6a-frozen-dev-run-context-v1",
        "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "method_code_commit": lock["method_code_commit"],
        "execution_code_commit": execution_commit,
        "build_manifest_sha256": sha256_file(BUILD_MANIFEST_PATH),
        "build_report_sha256": sha256_file(REPORT_PATH),
        "frozen_dev_truth_opened_before_execution": False,
    }
    write_immutable_json(RUN_ROOT / "run_context.json", run_context)
    manifest = execute_partition(
        partition="FROZEN_DEV",
        u2f_root=U2F_ROOT,
        split_manifest_path=SPLIT_MANIFEST,
        corpus_root=DEFAULT_RUNTIME_CORPUS_ROOT,
        output_root=RUN_ROOT,
        llama_url=DEFAULT_LLAMA_URL,
        server_manifest_path=DEFAULT_SERVER_MANIFEST,
    )
    print(json.dumps({
        "partition": manifest["partition"],
        "episode_count": manifest["episode_count"],
        "arm_execution_count": manifest["arm_execution_count"],
        "reader_output_sha256": manifest["reader_output_sha256"],
        "generation_calls_attempted": manifest["generator"]["generation_calls_attempted"],
        "evaluator_truth_opened": manifest["evaluator_truth_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
