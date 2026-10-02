"""Run a small local-only PropMem pilot on a category-balanced LoCoMo slice."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mem1_artifacts import read_jsonl, sha256_file
from run_mem1 import _load_system_registry, verify_pinned_patch

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT.parent / "external" / "memory" / "LoCoMo" / "data" / "locomo10.json"
LOCOMO_ROOT = ROOT.parent / "external" / "memory" / "LoCoMo"
EXPECTED_DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
EXPECTED_LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
EXPECTED_MEMEVAL_COMMIT = "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4"
EXPECTED_PATCH_SHA256 = "c7e44015527a17d43cf9bbd2e2a6e50d007943ee2efaa244899d3b048fd80eab"
PATCH = ROOT / "tools" / "research" / "memory" / "patches" / "memeval_qwen_main_v1.patch"
SYSTEM_NAMES = {"openclaw": "OpenClaw", "propmem": "PropMem"}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def _select_questions(
    conversation: dict[str, Any], per_category: int = 4
) -> list[dict[str, Any]]:
    selected = []
    category_ids = sorted({int(row["category"]) for row in conversation["qa"]})
    for category in category_ids:
        category_rows = [
            (index, row)
            for index, row in enumerate(conversation["qa"])
            if int(row["category"]) == category
        ][:per_category]
        if len(category_rows) != per_category:
            raise RuntimeError(f"LoCoMo category {category} has fewer than {per_category} questions")
        for index, row in category_rows:
            selected.append({
                **row,
                "question_id": f"{conversation['sample_id']}:qa:{index}",
                "source_qa_index": index,
            })
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=tuple(SYSTEM_NAMES), default="openclaw")
    parser.add_argument("--per-category", type=int, default=4)
    args = parser.parse_args()
    if args.per_category < 1:
        raise ValueError("--per-category must be positive")
    system_name = args.system
    display_name = SYSTEM_NAMES[system_name]
    run_id = (
        f"{system_name}-conv26-category-balanced-{args.per_category * 5}-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    output = ROOT / "runs" / "memory" / "locomo" / run_id
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing pilot output: {output}")
    if not DATASET.is_file() or sha256_file(DATASET) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("LoCoMo dataset file is missing or differs from its audited SHA256")
    if _git(LOCOMO_ROOT, "remote", "get-url", "origin").rstrip("/").removesuffix(".git") != (
        "https://github.com/snap-research/locomo"
    ):
        raise RuntimeError("LoCoMo checkout origin does not match the official repository")
    locomo_commit = _git(LOCOMO_ROOT, "rev-parse", "HEAD")
    if locomo_commit != EXPECTED_LOCOMO_COMMIT:
        raise RuntimeError(f"LoCoMo checkout differs from the audited commit: {locomo_commit}")
    memeval_root = ROOT.parent / "external" / "memory" / "MemEval"
    if _git(memeval_root, "rev-parse", "HEAD") != EXPECTED_MEMEVAL_COMMIT:
        raise RuntimeError("MemEval checkout is not at its pinned baseline commit")
    if sha256_file(PATCH) != EXPECTED_PATCH_SHA256:
        raise RuntimeError("Local compatibility patch differs from the audited patch SHA256")
    verify_pinned_patch(memeval_root, PATCH, EXPECTED_PATCH_SHA256)

    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ["HC_MEMORY_TRACK"] = "main_local_only"
    os.environ["HC_MEM1_RUN_DIR"] = str(output.resolve())
    os.environ.setdefault("HC_LOCAL_READER_BASE_URL", "http://127.0.0.1:8081/v1")
    os.environ.setdefault(
        "HC_LOCAL_EMBEDDING_PATH",
        r"E:\Health-Copilot-Models\models\Qwen3-Embedding-0.6B",
    )

    sys.path.insert(0, str(memeval_root / "src"))
    from agents_memory.benchmarks.locomo import CATEGORY_NAMES
    from agents_memory.evaluation import compute_f1
    from agents_memory.healthcopilot_provider import (
        ProviderConfig,
        call_context,
        configure_call_ledger,
        initialize_local_embedding,
        release_local_embedding_runtimes,
    )

    config = ProviderConfig.from_env()
    embedding_runtime = initialize_local_embedding(config)
    systems = _load_system_registry(memeval_root, [system_name])
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    conversation = dataset[0]
    questions = _select_questions(conversation, per_category=args.per_category)
    selected = {**conversation, "qa": questions}

    output.mkdir(parents=True, exist_ok=False)
    configure_call_ledger(output / "call_ledger.jsonl")
    log_path = output / "adapter.log"
    started = time.perf_counter()
    error: BaseException | None = None
    rows: list[dict[str, Any]] = []
    try:
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            with call_context(system_name, "locomo:conv-26:pilot"):
                rows = systems[system_name]["fn"](
                    selected,
                    config.reader_model,
                    False,
                    category_names=CATEGORY_NAMES,
                    judge_fn=None,
                )
    except BaseException as caught:  # noqa: BLE001 - preserve infra failure as a distinct status
        error = caught
    finally:
        release_local_embedding_runtimes()

    calls = read_jsonl(output / "call_ledger.jsonl")
    if error is not None:
        status = "INFRA_FAILURE"
    elif any(row.get("success") is False for row in calls):
        status = "INFRA_FAILURE"
    elif len(rows) != len(questions):
        status = "INFRA_FAILURE"
    else:
        status = "COMPLETED_DIAGNOSTIC"

    scored = [
        {
            **row,
            "question_id": questions[index]["question_id"],
            "source_qa_index": questions[index]["source_qa_index"],
            "f1": compute_f1(row.get("predicted", ""), row.get("ground_truth", "")),
        }
        for index, row in enumerate(rows[:len(questions)])
    ]
    category_results = {}
    for category, name in CATEGORY_NAMES.items():
        category_rows = [row for row in scored if row.get("category") == category]
        category_results[name] = {
            "n": len(category_rows),
            "mean_token_f1": (
                sum(float(row["f1"]) for row in category_rows) / len(category_rows)
                if category_rows else None
            ),
        }
    summary = {
        "run_id": run_id,
        "run_dir": str(output),
        "status": status,
        "interpretation": "Exploratory 20-question LoCoMo pilot; not a full-benchmark or performance-ranking claim.",
        "dataset": {
            "id": "locomo10",
            "repository": "https://github.com/snap-research/locomo",
            "revision": locomo_commit,
            "file_sha256": EXPECTED_DATASET_SHA256,
            "license": "CC BY-NC 4.0; non-commercial use only",
            "conversation_id": conversation["sample_id"],
            "question_ids": [row["question_id"] for row in questions],
        },
        "system": f"{display_name} (upstream MemEval implementation; attributed baseline)",
        "memeval_revision": EXPECTED_MEMEVAL_COMMIT,
        "compatibility_patch_sha256": EXPECTED_PATCH_SHA256,
        "reader": {
            "model": config.reader_model,
            "provider": "local_qwen",
            "endpoint": config.reader_base_url,
            "openai_api_key_required": False,
        },
        "embedding": vars(embedding_runtime.artifact),
        "judge": "NONE",
        "hosted_calls": 0,
        "system_wall_time_seconds": round(time.perf_counter() - started, 3),
        "n_questions": len(scored),
        "mean_token_f1": (
            sum(float(row["f1"]) for row in scored) / len(scored) if scored else None
        ),
        "per_category": category_results,
        "provider_calls": len(calls),
        "provider_failure_count": sum(row.get("success") is False for row in calls),
        "adapter_error_type": type(error).__name__ if error is not None else None,
        "adapter_log_sha256": sha256_file(log_path),
    }
    predictions = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scored)
    prediction_bytes = predictions.encode("utf-8")
    (output / "predictions.jsonl").write_bytes(prediction_bytes)
    summary["predictions_sha256"] = hashlib.sha256(prediction_bytes).hexdigest()
    summary["prediction_frozen"] = True
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "COMPLETED_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())
