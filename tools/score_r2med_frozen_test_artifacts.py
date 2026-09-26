"""Score already-frozen R2MED TEST rankings after a subset-ID guard erratum.

This is deliberately an evaluation-only continuation. It does not regenerate,
retrieve, rerank, or alter any TEST artifact. The original locked runner stopped
before qrels access because its generation-budget guard treated subset-local
numeric query IDs as globally unique. This tool verifies every frozen artifact
and substitutes only that false global-uniqueness precondition with the proper
(subset, query_id) identity check before calling the original locked scorer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.r2med_final_test import (
    ARTIFACT_ROOT,
    FROZEN_ARMS,
    LOCK_PATH,
    LOCKED_CODE_PATHS,
    REPORT_PATH,
    START_MARKER_PATH,
    TEST_COUNTS,
    TEST_TOTAL,
    read_jsonl_records,
    read_lock,
    sha256_file,
    validate_final_lock,
)
from tools import run_r2med_final_test as locked_runner

ERRATUM_PATH = ROOT / "runs/rag_r2med_final_test/evaluation_preflight_erratum.json"
GENERATION_MANIFESTS = {
    "lamer": ROOT / "runs/rag_r2med_final_test/generation/lamer_manifest.json",
    "compact_crb_q": ROOT / "runs/rag_r2med_final_test/generation/compact_crb_q_manifest.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_subset_scoped_generation_budget(
    lamer_rows: list[dict[str, Any]], compact_rows: list[dict[str, Any]]
) -> None:
    """Validate one generation per query using R2MED's subset-local IDs."""
    for method, rows in (("LameR", lamer_rows), ("compact CRB-Q", compact_rows)):
        if len(rows) != TEST_TOTAL:
            raise ValueError(f"{method} must contain exactly {TEST_TOTAL} rows")
        keys = [(str(row.get("subset", "")), str(row.get("query_id", ""))) for row in rows]
        if any(not subset or not query_id for subset, query_id in keys):
            raise ValueError(f"{method} contains a missing subset/query identity")
        if len(set(keys)) != TEST_TOTAL:
            raise ValueError(f"{method} contains a duplicate (subset, query_id) identity")
        counts = Counter(subset for subset, _ in keys)
        if counts != Counter(TEST_COUNTS):
            raise ValueError(f"{method} subset counts differ from the locked TEST split: {dict(counts)}")


def _verify_frozen_artifacts() -> dict[str, Any]:
    lock_path = ROOT / LOCK_PATH
    lock = read_lock(lock_path)
    validate_final_lock(lock)
    lock_sha = sha256_file(lock_path)

    marker_path = ROOT / START_MARKER_PATH
    marker = _read_json(marker_path)
    if marker.get("status") != "IN_PROGRESS" or marker.get("lock_sha256") != lock_sha:
        raise ValueError("the one-shot marker does not match the committed TEST lock")
    if marker.get("test_accessed") is not True:
        raise ValueError("the frozen TEST execution marker is missing TEST access state")
    locked_paths = [LOCK_PATH.as_posix(), *LOCKED_CODE_PATHS]
    dirty_locked_paths = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *locked_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty_locked_paths:
        raise ValueError("the committed final lock or its bound implementation files are modified")
    for relative, expected_sha in lock["code"]["files_sha256"].items():
        if sha256_file(ROOT / relative) != expected_sha:
            raise ValueError(f"locked implementation hash mismatch: {relative}")

    ranking_freeze_path = ARTIFACT_ROOT / "rankings_frozen_manifest.json"
    ranking_freeze = _read_json(ranking_freeze_path)
    ranking_freeze_sha = sha256_file(ranking_freeze_path)
    if (
        ranking_freeze.get("lock_sha256") != lock_sha
        or ranking_freeze.get("qrels_opened") is not False
        or ranking_freeze.get("query_count_per_arm") != TEST_TOTAL
        or ranking_freeze.get("six_arms") != list(FROZEN_ARMS)
    ):
        raise ValueError("frozen ranking manifest is not the complete pre-qrels artifact for this lock")

    generation_manifests: dict[str, dict[str, Any]] = {}
    generation_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    generation_global_duplicates: dict[str, list[str]] = {}
    expected_methods = {"lamer": "lamer", "compact_crb_q": "crb_q_compact_repair"}
    expected_prompts = {
        "lamer": lock["prompts"]["lamer_upstream_template_sha256_by_subset"],
        "compact_crb_q": {
            subset: lock["prompts"]["compact_crb_template_sha256"] for subset in TEST_COUNTS
        },
    }
    for method, manifest_path in GENERATION_MANIFESTS.items():
        manifest = _read_json(manifest_path)
        if (
            manifest.get("partition") != "TEST"
            or manifest.get("test_status") != "PUBLIC_BENCHMARK_REUSED"
            or manifest.get("method") != method
            or manifest.get("query_count") != TEST_TOTAL
            or manifest.get("call_count") != TEST_TOTAL
            or manifest.get("generation_config", {}).get("retry_on_error") is not False
            or manifest.get("generation_config", {}).get("calls_per_query_per_method") != 1
            or manifest.get("generator", {}).get("identity", {}).get("sha256")
            != lock["models"]["generator"]["sha256"]
        ):
            raise ValueError(f"{method} generation manifest differs from the frozen TEST contract")
        subset_entries = {entry["subset"]: entry for entry in manifest.get("subsets", [])}
        if set(subset_entries) != set(TEST_COUNTS):
            raise ValueError(f"{method} generation manifest has a wrong subset inventory")
        generation_rows[method] = {}
        all_ids: list[str] = []
        for subset, expected_count in TEST_COUNTS.items():
            entry = subset_entries[subset]
            identity = lock["source"]["test_query_corpus_identity"][subset]
            if (
                entry.get("query_count") != expected_count
                or entry.get("query_sha256") != identity["query_sha256"]
                or entry.get("corpus_sha256") != identity["corpus_sha256"]
                or entry.get("prompt_sha256") != expected_prompts[method][subset]
            ):
                raise ValueError(f"{method} locked TEST identity mismatch for {subset}")
            artifact = Path(entry["artifact_path"])
            rows = read_jsonl_records(artifact)
            if sha256_file(artifact) != entry.get("artifact_sha256") or len(rows) != expected_count:
                raise ValueError(f"{method} artifact hash/count mismatch for {subset}")
            ids = [str(row.get("query_id", "")) for row in rows]
            if (
                len(set(ids)) != expected_count
                or any(row.get("subset") != subset or row.get("method") != expected_methods[method] for row in rows)
                or any(row.get("prompt_sha256") != expected_prompts[method][subset] for row in rows)
            ):
                raise ValueError(f"{method} rows are not a complete ordered subset artifact: {subset}")
            # Generation manifests use SHA256 of newline-joined IDs in source order.
            import hashlib

            ids_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
            if ids_sha != entry.get("query_order_sha256"):
                raise ValueError(f"{method} query order hash mismatch for {subset}")
            generation_rows[method][subset] = rows
            all_ids.extend(ids)
        duplicates = sorted(query_id for query_id, count in Counter(all_ids).items() if count > 1)
        generation_global_duplicates[method] = duplicates
        generation_manifests[method] = manifest

    validate_subset_scoped_generation_budget(
        [row for subset_rows in generation_rows["lamer"].values() for row in subset_rows],
        [row for subset_rows in generation_rows["compact_crb_q"].values() for row in subset_rows],
    )

    ranking_hashes = ranking_freeze["ranking_hashes"]
    if set(ranking_hashes) != set(FROZEN_ARMS):
        raise ValueError("frozen ranking manifest does not name all six locked arms")
    rankings: dict[str, dict[str, dict[str, list[str]]]] = {}
    for arm in FROZEN_ARMS:
        if set(ranking_hashes[arm]) != set(TEST_COUNTS):
            raise ValueError(f"frozen ranking manifest has an incomplete subset list for {arm}")
        rankings[arm] = {}
        for subset, expected_count in TEST_COUNTS.items():
            path = ARTIFACT_ROOT / "rankings" / arm / f"{subset}.jsonl"
            rows = read_jsonl_records(path)
            if sha256_file(path) != ranking_hashes[arm][subset] or len(rows) != expected_count:
                raise ValueError(f"frozen ranking artifact hash/count mismatch: {arm}/{subset}")
            query_ids = [str(row.get("query_id", "")) for row in rows]
            if (
                any(row.get("subset") != subset for row in rows)
                or len(set(query_ids)) != expected_count
                or query_ids != [str(row["query_id"]) for row in generation_rows["lamer"][subset]]
                or query_ids != [str(row["query_id"]) for row in generation_rows["compact_crb_q"][subset]]
            ):
                raise ValueError(f"ranking query identities/order differ from generation artifacts: {arm}/{subset}")
            rankings[arm][subset] = {}
            for row in rows:
                doc_ids = [str(item["doc_id"]) for item in row.get("ranking", [])]
                if not doc_ids or len(doc_ids) != len(set(doc_ids)):
                    raise ValueError(f"invalid ranked document list: {arm}/{subset}/{row['query_id']}")
                rankings[arm][subset][str(row["query_id"])] = doc_ids

    return {
        "lock": lock,
        "lock_sha": lock_sha,
        "marker": marker,
        "ranking_freeze": ranking_freeze,
        "ranking_freeze_sha": ranking_freeze_sha,
        "ranking_hashes": ranking_hashes,
        "rankings": rankings,
        "generation_rows": generation_rows,
        "generation_manifests": generation_manifests,
        "global_duplicate_ids": generation_global_duplicates,
    }


def _erratum_record(verified: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "r2med-final-test-evaluation-preflight-erratum-v1",
        "status": "READY_FOR_SCORING_PENDING_COMMIT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "original_final_eval_lock_sha256": verified["lock_sha"],
        "original_lock_commit": verified["marker"].get("lock_commit"),
        "ranking_freeze_manifest_sha256": verified["ranking_freeze_sha"],
        "qrels_opened_before_erratum": False,
        "locked_runner_failure": "validate_generation_budget compared bare numeric query_id values across subsets",
        "observed_test_identity": {
            "subset_counts": TEST_COUNTS,
            "rows_per_generation_method": TEST_TOTAL,
            "shared_bare_query_ids_across_subsets": verified["global_duplicate_ids"],
        },
        "correction": {
            "identity_key": "(subset, query_id)",
            "requires_exactly_one_row_per_query_within_each_subset": True,
            "requires_locked_subset_counts_and_query_order_hashes": True,
            "generation_or_ranking_artifacts_changed": False,
            "method_configs_or_metrics_changed": False,
            "locked_scoring_and_qrels_reader_changed": False,
        },
        "scoring": {
            "status": "NOT_STARTED",
            "evaluator": "tools.run_r2med_final_test._evaluate_and_report",
            "only_qrels_reader": "eval.r2med_crb_evaluator.py",
        },
    }


def prepare() -> None:
    verified = _verify_frozen_artifacts()
    if (ROOT / REPORT_PATH).exists() or (ROOT / "runs/rag_r2med_final_test/candidate_analysis.json").exists():
        raise FileExistsError("scoring outputs already exist; refusing a second evaluation")
    if ERRATUM_PATH.exists():
        raise FileExistsError(f"preflight erratum already exists: {ERRATUM_PATH}")
    erratum = _erratum_record(verified)
    locked_runner._write_json_atomic(ERRATUM_PATH, erratum)
    marker_path = ROOT / START_MARKER_PATH
    marker = _read_json(marker_path)
    marker["evaluation_preflight_erratum"] = {
        "path": str(ERRATUM_PATH.relative_to(ROOT)),
        "sha256": sha256_file(ERRATUM_PATH),
        "qrels_opened": False,
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen artifacts validated; qrels_opened=false; erratum={ERRATUM_PATH}")


def score() -> None:
    verified = _verify_frozen_artifacts()
    if (ROOT / REPORT_PATH).exists():
        raise FileExistsError("final TEST report already exists; refusing a second evaluation")
    if not ERRATUM_PATH.is_file():
        raise FileNotFoundError("prepare and commit the evaluation preflight erratum before scoring")
    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", ERRATUM_PATH.relative_to(ROOT).as_posix()],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("evaluation erratum must be committed before the first qrels read")
    subprocess.run(
        ["git", "-C", str(ROOT), "cat-file", "-e", f"HEAD:{ERRATUM_PATH.relative_to(ROOT).as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    )
    erratum = _read_json(ERRATUM_PATH)
    marker_path = ROOT / START_MARKER_PATH
    marker = _read_json(marker_path)
    if (
        erratum.get("status") != "READY_FOR_SCORING_PENDING_COMMIT"
        or erratum.get("original_final_eval_lock_sha256") != verified["lock_sha"]
        or marker.get("evaluation_preflight_erratum", {}).get("sha256") != sha256_file(ERRATUM_PATH)
    ):
        raise ValueError("preflight erratum, marker, and committed lock identities differ")

    # The frozen scorer remains unchanged; replace only its erroneous global-ID guard.
    locked_runner.validate_generation_budget = validate_subset_scoped_generation_budget
    report = locked_runner._evaluate_and_report(
        lock=verified["lock"],
        lock_sha=verified["lock_sha"],
        lock_commit=str(marker["lock_commit"]),
        inputs=None,
        rankings=verified["rankings"],
        ranking_hashes=verified["ranking_hashes"],
        generations=verified["generation_rows"],
        generation_manifests=verified["generation_manifests"],
        source_root=locked_runner.DEFAULT_SOURCE_ROOT,
    )

    erratum["status"] = "SCORING_COMPLETE"
    erratum["completed_at_utc"] = datetime.now(UTC).isoformat()
    erratum["scoring"].update(
        {
            "status": "COMPLETE",
            "qrels_accessed_after_frozen_ranking_manifest": True,
            "metrics_report_sha256": sha256_file(ROOT / REPORT_PATH),
        }
    )
    locked_runner._write_json_atomic(ERRATUM_PATH, erratum)
    report["evaluation_preflight_correction"] = {
        "path": str(ERRATUM_PATH.relative_to(ROOT)),
        "reason": "subset-local numeric query IDs repeat across TEST subsets; composite identities were validated",
        "frozen_generation_and_ranking_artifacts_changed": False,
        "scoring_or_qrels_reader_changed": False,
    }
    report_path = ROOT / REPORT_PATH
    locked_runner._write_json_atomic(report_path, report)
    marker = _read_json(marker_path)
    marker["report_sha256"] = sha256_file(report_path)
    marker["evaluation_preflight_erratum"]["sha256"] = sha256_file(ERRATUM_PATH)
    marker["evaluation_preflight_erratum"]["qrels_opened"] = True
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Scoring complete; report={report_path}; test_executed_once={report['test_executed_once']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
    else:
        score()


if __name__ == "__main__":
    main()
