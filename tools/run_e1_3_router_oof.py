"""Run the frozen 5-fold question-only router OOF evaluation offline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval.e1_3_learned_router import run_oof_predictions
from eval.e1_3_router_dataset import (
    canonical_sha256,
    case_order_sha256,
    load_cases,
    load_historical_arms,
    make_cost_oracle_rows,
    read_jsonl,
    sha256_file,
)
from tools.cache_e1_3_bge_embeddings import question_set_sha256


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        for row in rows:
            temporary.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=Path(r"E:\Health-Copilot-E1.2"))
    parser.add_argument(
        "--cache-root", type=Path, default=Path(r"E:\Health-Copilot-E1.3\router")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    scratch_root = args.scratch_root.resolve()
    cache_root = args.cache_root.resolve()
    artifact_root = repo_root / "runs" / "e1_3"
    protocol = read_json(artifact_root / "router_protocol.json")
    protocol_hash = protocol.get("protocol_sha256")
    if canonical_sha256({k: v for k, v in protocol.items() if k != "protocol_sha256"}) != protocol_hash:
        raise ValueError("Frozen router protocol SHA-256 mismatch")

    input_path = Path(protocol["benchmark"]["input_artifact"])
    if sha256_file(input_path) != protocol["benchmark"]["input_sha256"]:
        raise ValueError("Frozen MIRAGE input changed since protocol freeze")
    cases = load_cases(input_path)
    case_ids = [case.case_id for case in cases]
    if case_order_sha256(case_ids) != protocol["benchmark"]["case_order_sha256"]:
        raise ValueError("MIRAGE case order differs from the frozen protocol")

    fold_path = artifact_root / "router_fold_manifest.json"
    fold_manifest = read_json(fold_path)
    if canonical_sha256({k: v for k, v in fold_manifest.items() if k != "manifest_sha256"}) != fold_manifest.get("manifest_sha256"):
        raise ValueError("Fold manifest SHA-256 mismatch")
    if fold_manifest["manifest_sha256"] != protocol["outer_cv"]["fold_manifest_sha256"]:
        raise ValueError("Fold manifest does not match the frozen protocol")
    if fold_manifest["case_order_sha256"] != protocol["benchmark"]["case_order_sha256"]:
        raise ValueError("Fold manifest case order identity mismatch")

    oracle_path = artifact_root / "router_cost_oracle_v2.jsonl"
    if sha256_file(oracle_path) != protocol["cost_oracle_v2"]["sha256"]:
        raise ValueError("Cost Oracle v2 artifact SHA-256 mismatch")
    oracle_rows = read_jsonl(oracle_path)
    arms = load_historical_arms(scratch_root, case_ids)
    if make_cost_oracle_rows(cases, arms) != oracle_rows:
        raise ValueError("Cost Oracle v2 labels do not reproduce from historical arm outcomes")

    for arm, identity in protocol["historical_arm_sources"].items():
        arm_root = scratch_root / "runs" / "e1_2" / "test" / arm
        if sha256_file(arm_root / "case_results.jsonl") != identity["case_results_sha256"]:
            raise ValueError(f"Historical {arm} result file changed since protocol freeze")
        if sha256_file(arm_root / "manifest.json") != identity["manifest_sha256"]:
            raise ValueError(f"Historical {arm} manifest changed since protocol freeze")

    bge_config = protocol["models"]["bge"]
    model_file = Path(bge_config["model_path"]) / bge_config["weights_file"]
    if sha256_file(model_file) != bge_config["weights_sha256"]:
        raise ValueError("Pinned local BGE weights changed since protocol freeze")
    embedding_path = cache_root / "bge_embeddings.npy"
    embedding_manifest_path = cache_root / "bge_embedding_manifest.json"
    embedding_manifest = read_json(embedding_manifest_path)
    required_embedding_fields = {
        "model_repo_id": bge_config["repo_id"],
        "model_revision": bge_config["revision"],
        "model_weights_sha256": bge_config["weights_sha256"],
        "benchmark_input_sha256": protocol["benchmark"]["input_sha256"],
        "question_set_identity_sha256": question_set_sha256(cases),
        "case_id_order_sha256": protocol["benchmark"]["case_order_sha256"],
        "case_count": len(cases),
        "embedding_shape": [len(cases), 768],
        "dtype": "float32",
        "protocol_sha256": protocol_hash,
    }
    if any(embedding_manifest.get(k) != v for k, v in required_embedding_fields.items()):
        raise ValueError("BGE embedding cache metadata does not match the frozen inputs")
    if sha256_file(embedding_path) != embedding_manifest.get("embedding_file_sha256"):
        raise ValueError("BGE embedding file SHA-256 mismatch")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (len(cases), 768) or embeddings.dtype != np.float32:
        raise ValueError("BGE embedding matrix shape or dtype is incorrect")
    if not np.isfinite(embeddings).all():
        raise ValueError("BGE embedding matrix contains non-finite values")

    output_path = args.output or artifact_root / "router_oof_predictions.jsonl"
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing OOF output: {output_path}")
    rows = run_oof_predictions(
        cases,
        oracle_rows,
        arms,
        fold_manifest,
        protocol,
        embeddings,
    )
    atomic_write_jsonl(output_path, rows)
    run_manifest = {
        "schema_version": "e1-3-router-oof-run-v1",
        "evaluation_status": "EXPOSED_EXPLORATORY_OOF",
        "case_count": len(cases),
        "policy_count": len(protocol["learned_policies"]),
        "prediction_rows": len(rows),
        "protocol_sha256": protocol_hash,
        "fold_manifest_sha256": fold_manifest["manifest_sha256"],
        "cost_oracle_v2_sha256": protocol["cost_oracle_v2"]["sha256"],
        "bge_embedding_sha256": embedding_manifest["embedding_file_sha256"],
        "oof_predictions_sha256": sha256_file(output_path),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "question_text_in_artifact": False,
        "options_or_answers_in_artifact": False,
        "new_answer_inference": False,
        "policies": protocol["learned_policies"],
    }
    run_manifest_path = artifact_root / "router_oof_run_manifest.json"
    if run_manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing OOF run manifest: {run_manifest_path}")
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "OOF_PREDICTIONS_COMPLETE", **{k: run_manifest[k] for k in ("case_count", "policy_count", "prediction_rows", "oof_predictions_sha256")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
