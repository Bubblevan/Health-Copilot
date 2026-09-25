"""Audit and summarize exposed E1.2 MIRAGE outcomes without exporting QA content."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_CASES = 5_235
CLEAN_SUBDATASETS = {"medqa", "medmcqa", "mmlu"}
ARMS = {
    "closed_book": "CLOSED_BOOK",
    "random_context": "RANDOM_CONTEXT",
    "rag_bm25": "RAG_BM25",
    "rag_medcpt": "RAG_MEDCPT_CANONICAL",
    "cheap_router": "CHEAP_CAPABILITY_ROUTER",
    "jev_router": "JEV_CAPABILITY_ROUTER",
}
FIXED_ACTIONS = ("closed_book", "rag_bm25", "rag_medcpt")
ACTION_COST = {"closed_book": 0, "rag_bm25": 1, "rag_medcpt": 2}
CODE_PATHS = (
    "tools/run_e1_2_mirage.py",
    "eval/e1_2_runner.py",
    "eval/e1_2_answer_provider.py",
    "eval/e1_2_capability_router.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def git_text(repo_root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Non-object JSONL row in {path.name}:{line_number}")
            rows.append(row)
    return rows


def _component_latency(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return {
        "measurement_coverage": len(values) / len(rows) if rows else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
    }


def _safe_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from eval.e1_2_runner import summarize_results

    metrics = summarize_results(rows)
    metrics["answer_input_tokens_observed_total"] = sum(
        int(row["answer_input_tokens"])
        for row in rows
        if isinstance(row.get("answer_input_tokens"), (int, float))
    )
    metrics["answer_output_tokens_observed_total"] = sum(
        int(row["answer_output_tokens"])
        for row in rows
        if isinstance(row.get("answer_output_tokens"), (int, float))
    )
    metrics["latency_components"] = {
        "retrieval": _component_latency(rows, "retrieval_latency_ms"),
        "answer": _component_latency(rows, "answer_latency_ms"),
        "route": _component_latency(rows, "route_latency_ms"),
        "component_summed_proxy": _component_latency(rows, "component_latency_proxy_ms"),
    }
    return metrics


def _summarize_oracle(
    cases: dict[str, dict[str, Any]],
    arm_rows: dict[str, dict[str, dict[str, Any]]],
    fixed_accuracy: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fallback = max(FIXED_ACTIONS, key=lambda action: (fixed_accuracy[action], -ACTION_COST[action]))
    counts: Counter[str] = Counter()
    no_success = 0
    oracle_correct = 0
    cheap_regret: list[int] = []
    jev_regret: list[int] = []
    cheap_quality_regret = 0
    jev_quality_regret = 0
    cheap_retrieval_when_avoidable = 0
    jev_retrieval_when_avoidable = 0
    labels: list[dict[str, Any]] = []

    for case_id, case_meta in cases.items():
        fixed_correct = {
            action: arm_rows[action][case_id].get("is_correct") is True
            for action in FIXED_ACTIONS
        }
        successful = [action for action in FIXED_ACTIONS if fixed_correct[action]]
        if successful:
            oracle_action = min(successful, key=ACTION_COST.__getitem__)
            oracle_reason = "cheapest_successful_fixed_arm"
            counts[oracle_action] += 1
        else:
            oracle_action = fallback
            oracle_reason = "no_success_fallback"
            no_success += 1
        is_oracle_correct = any(fixed_correct.values())
        oracle_correct += int(is_oracle_correct)

        router_actions: dict[str, str | None] = {}
        for arm, regrets_name in (("cheap_router", "cheap"), ("jev_router", "jev")):
            row = arm_rows[arm][case_id]
            action = row.get("selected_action")
            router_actions[regrets_name] = action if isinstance(action, str) else None
            if action in ACTION_COST:
                regret = ACTION_COST[action] - ACTION_COST[oracle_action]
                (cheap_regret if regrets_name == "cheap" else jev_regret).append(regret)
                route_correct = row.get("is_correct") is True
                if regrets_name == "cheap":
                    cheap_quality_regret += int(is_oracle_correct) - int(route_correct)
                else:
                    jev_quality_regret += int(is_oracle_correct) - int(route_correct)
                if fixed_correct["closed_book"] and action != "closed_book":
                    if regrets_name == "cheap":
                        cheap_retrieval_when_avoidable += 1
                    else:
                        jev_retrieval_when_avoidable += 1

        labels.append(
            {
                "case_id": case_id,
                "subdataset": case_meta["subdataset"],
                "fixed_correct": fixed_correct,
                "oracle_action": oracle_action,
                "oracle_reason": oracle_reason,
                "oracle_correct": is_oracle_correct,
                "cheap_router_action": router_actions["cheap"],
                "cheap_router_correct": arm_rows["cheap_router"][case_id].get("is_correct") is True,
                "jev_router_action": router_actions["jev"],
                "jev_router_correct": arm_rows["jev_router"][case_id].get("is_correct") is True,
            }
        )

    denominator = len(cases)
    if denominator != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} cases for oracle, got {denominator}")
    success_distribution = {
        action: {"cases": counts[action], "rate": counts[action] / denominator}
        for action in FIXED_ACTIONS
    }
    applied_counts = Counter(row["oracle_action"] for row in labels)
    applied_distribution = {
        action: {"cases": applied_counts[action], "rate": applied_counts[action] / denominator}
        for action in FIXED_ACTIONS
    }
    distribution = {
        **success_distribution,
        "no_success_best_fixed_fallback": {
            "cases": no_success,
            "rate": no_success / denominator,
            "fallback_action": fallback,
        },
    }
    oracle_accuracy = oracle_correct / denominator
    best_fixed_accuracy = max(fixed_accuracy.values())
    report = {
        "definition": (
            "Post-hoc per-case minimum-cost successful fixed arm; if no fixed arm is correct, "
            "assign the globally most accurate fixed arm as fallback. This is an exposed-data "
            "ceiling, not an executable policy."
        ),
        "cost_order": list(FIXED_ACTIONS),
        "best_fixed_fallback": fallback,
        "oracle_action_distribution": distribution,
        "oracle_success_action_distribution": success_distribution,
        "oracle_applied_action_distribution_including_fallback": applied_distribution,
        "closed_book_sufficient_rate": success_distribution["closed_book"]["rate"],
        "bm25_needed_rate": success_distribution["rag_bm25"]["rate"],
        "medcpt_needed_rate": success_distribution["rag_medcpt"]["rate"],
        "no_fixed_arm_succeeds": no_success,
        "no_fixed_arm_succeeds_rate": no_success / denominator,
        "oracle_accuracy": oracle_accuracy,
        "best_fixed_accuracy": best_fixed_accuracy,
        "oracle_headroom_absolute": oracle_accuracy - best_fixed_accuracy,
        "avoidable_retrieval_rate_vs_always_bm25": success_distribution["closed_book"]["rate"],
        "router_diagnostics": {
            "cheap_router": {
                "quality_regret_vs_oracle_absolute": cheap_quality_regret / denominator,
                "mean_action_cost_regret_units": sum(cheap_regret) / len(cheap_regret)
                if cheap_regret
                else None,
                "retrieval_cases_where_closed_book_was_sufficient": cheap_retrieval_when_avoidable,
                "retrieval_avoidable_share_of_all_cases": cheap_retrieval_when_avoidable
                / denominator,
            },
            "jev_router": {
                "quality_regret_vs_oracle_absolute": jev_quality_regret / denominator,
                "mean_action_cost_regret_units": sum(jev_regret) / len(jev_regret)
                if jev_regret
                else None,
                "retrieval_cases_where_closed_book_was_sufficient": jev_retrieval_when_avoidable,
                "retrieval_avoidable_share_of_all_cases": jev_retrieval_when_avoidable
                / denominator,
            },
        },
    }
    return report, labels


def _load_historical_exposed_runs(repo_root: Path) -> dict[str, Any]:
    """Read aggregate metadata only for the already-exposed PubMedQA/BioASQ runs."""
    specifications = {
        "closed_book_deepseek": "runs/mirage_textbooks/answer_full_1118/closed_book",
        "bm25_rag_deepseek": "runs/mirage_textbooks/answer_full_1118/rag",
        "harness_rag_deepseek": "runs/mirage_textbooks/answer_full_1118/harness_rag",
        "medcpt_rag_deepseek": "runs/mirage_textbooks/answer_medcpt_full_1118/rag",
        "medcpt_harness_rag_deepseek": "runs/mirage_textbooks/answer_medcpt_full_1118/harness_rag",
    }
    result: dict[str, Any] = {
        "status": "HISTORICAL_EXPOSED",
        "case_count": 1_118,
        "subdatasets": {"pubmedqa": 500, "bioasq": 618},
        "generator_series": "DeepSeek-Flash historical; not comparable to the frozen Qwen3 E1.2 six-arm series",
        "headline_eligible": False,
        "arms": {},
        "raw_case_rows_read": False,
    }
    metric_fields = (
        "cases",
        "completed_cases",
        "failed_cases",
        "answer_coverage",
        "accuracy",
        "invalid_answer_rate",
        "abstain_rate",
        "mean_input_tokens",
        "mean_output_tokens",
        "p50_latency_ms",
        "p95_latency_ms",
        "provider_calls",
    )
    slice_fields = (
        "cases",
        "completed",
        "failed",
        "answer_coverage",
        "accuracy",
        "invalid_answer_rate",
        "abstain_rate",
        "mean_input_tokens",
        "mean_output_tokens",
        "p50_latency_ms",
        "p95_latency_ms",
    )
    for arm_name, relative_dir in specifications.items():
        run_dir = repo_root / relative_dir
        paths = {
            "manifest": run_dir / "manifest.json",
            "config": run_dir / "run_config.json",
            "metrics": run_dir / "metrics.json",
        }
        missing = [name for name, path in paths.items() if not path.is_file()]
        if missing:
            result["arms"][arm_name] = {
                "status": "AGGREGATE_ARTIFACT_MISSING",
                "missing_artifacts": missing,
            }
            continue
        manifest = read_json(paths["manifest"])
        config = read_json(paths["config"])
        metrics = read_json(paths["metrics"])
        by_subdataset = metrics.get("by_subdataset", {})
        result["arms"][arm_name] = {
            "status": manifest.get("status"),
            "manifest_case_count": manifest.get("case_count"),
            "config_identity": config.get("config_identity"),
            "model": config.get("model"),
            "retriever": config.get("retriever", config.get("arm")),
            "evidence_source": config.get("evidence_source"),
            "metrics": {key: metrics.get(key) for key in metric_fields if key in metrics},
            "by_subdataset": {
                subdataset: {key: values.get(key) for key in slice_fields if key in values}
                for subdataset, values in by_subdataset.items()
                if subdataset in {"pubmedqa", "bioasq"} and isinstance(values, dict)
            },
            "artifact_sha256": {name: sha256_file(path) for name, path in paths.items()},
        }
        if (
            manifest.get("status") != "COMPLETED"
            or manifest.get("case_count") != 1_118
            or metrics.get("cases") != 1_118
        ):
            result["arms"][arm_name]["status"] = "INCOMPLETE_OR_IDENTITY_MISMATCH"
    return result


def audit_history(scratch_root: Path, repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config_path = repo_root / "runs/e1_2/frozen_test_config.json"
    split_path = repo_root / "runs/e1_2/mirage_split_manifest.json"
    selection_path = repo_root / "runs/e1_2/generator_selection.json"
    config = read_json(config_path)
    selection = read_json(selection_path)
    split_manifest = read_json(split_path)
    config_hash = sha256_file(config_path)
    split_hash = sha256_file(split_path)
    # The frozen config's raw SHA is the identity written into all E1.2 manifests.
    if not config_hash or not split_hash:
        raise ValueError("Could not hash frozen E1.2 source manifests")

    benchmark = config.get("benchmark", {})
    if split_hash != benchmark.get("split_manifest_sha256"):
        raise ValueError("E1.2 split manifest hash differs from the frozen config")
    expected_count = int(benchmark.get("clean_test_case_count", -1))
    if expected_count != EXPECTED_CASES:
        raise ValueError(f"Frozen E1.2 expected case count changed: {expected_count}")
    split_rows = split_manifest.get("cases", [])
    expected_cases = {
        str(row["case_id"]): {"subdataset": str(row["subdataset"]), "split": str(row["split"])}
        for row in split_rows
        if row.get("split") == "TEST" and row.get("subdataset") in CLEAN_SUBDATASETS
    }
    if len(expected_cases) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} clean TEST case IDs in split manifest")

    generator = config.get("generator_selection", {}).get("primary_candidate", {})
    if not generator:
        generator = config.get("generator_selection", {}).get("candidates", [{}])[0]
    expected_model = str(generator.get("requested_model", ""))
    expected_served = str(generator.get("expected_served_model", ""))
    expected_model_hash = str(generator.get("artifact_sha256", ""))
    expected_model_bytes = int(generator.get("artifact_bytes", 0))
    code_commit = str(config.get("code_base_commit", ""))
    selection_valid = (
        selection.get("selected_candidate") == "QWEN3_LOCAL"
        and selection.get("config_sha256") == config_hash
        and selection.get("pilot", {}).get("served_models") == [expected_served]
        and selection.get("pilot", {}).get("case_count") == 90
        and selection.get("pilot", {}).get("completed") == 90
    )
    code_blobs = {
        path: git_text(repo_root, "rev-parse", f"{code_commit}:{path}") for path in CODE_PATHS
    }
    code_identity_valid = bool(code_commit) and all(code_blobs.values())
    retrieval_config = config.get("retrieval", {})
    medcpt = retrieval_config.get("retriever_arms", {}).get("RAG_MEDCPT", {})
    medcpt_canonical = all(
        medcpt.get(key)
        for key in ("query_encoder", "article_encoder", "cross_encoder", "weights")
    ) and "ARTICLE_ARTICLE" not in json.dumps(medcpt, ensure_ascii=False)
    retrieval_identity = {
        "bm25": retrieval_config.get("retriever_arms", {}).get("RAG_BM25", {}),
        "medcpt_canonical": medcpt,
    }

    all_rows: dict[str, list[dict[str, Any]]] = {}
    arm_audit: dict[str, Any] = {}
    case_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, label in ARMS.items():
        run_dir = scratch_root / "runs" / "e1_2" / "test" / arm
        manifest_path = run_dir / "manifest.json"
        result_path = run_dir / "case_results.jsonl"
        metrics_path = run_dir / "metrics.json"
        if not all(path.is_file() for path in (manifest_path, result_path, metrics_path)):
            arm_audit[arm] = {"label": label, "valid": False, "reason": "required artifact missing"}
            continue
        manifest = read_json(manifest_path)
        rows = load_jsonl(result_path)
        metrics_file = read_json(metrics_path)
        ids = [str(row.get("case_id", "")) for row in rows]
        duplicates = len(ids) - len(set(ids))
        row_map = {str(row.get("case_id", "")): row for row in rows}
        requested_values = sorted(
            {str(row["requested_model"]) for row in rows if row.get("requested_model")}
        )
        served_values = sorted(
            {str(row["served_model"]) for row in rows if row.get("served_model")}
        )
        completed_rows = [row for row in rows if str(row.get("status", "")).lower() == "completed"]
        completed_served_identity_valid = all(
            row.get("served_model") == expected_served for row in completed_rows
        )
        identity = str(manifest.get("result_identity", ""))
        expected_identity = hashlib.sha256(
            json.dumps(
                {
                    "frozen_config_sha256": config_hash,
                    "partition": "TEST",
                    "arm": arm,
                    "model": "QWEN3_LOCAL",
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        IDs_match = set(ids) == set(expected_cases)
        manifest_valid = (
            manifest.get("status") == "COMPLETED"
            and manifest.get("partition") == "TEST"
            and manifest.get("arm") == arm
            and manifest.get("selected_candidate") == "QWEN3_LOCAL"
            and manifest.get("config_sha256") == config_hash
            and manifest.get("case_count") == EXPECTED_CASES
            and manifest.get("failures") == sum(
                str(row.get("status", "")).lower() != "completed" for row in rows
            )
        )
        rows_valid = (
            len(rows) == EXPECTED_CASES
            and duplicates == 0
            and IDs_match
            and all(row.get("result_identity") == identity for row in rows)
            and all(row.get("partition") == "TEST" and row.get("arm") == arm for row in rows)
            and identity == expected_identity
        )
        model_valid = (
            requested_values == [expected_model]
            and served_values == [expected_served]
            and completed_served_identity_valid
            and expected_model_hash
            == "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
            and expected_model_bytes == 5_027_783_488
        )
        if not IDs_match or duplicates or not rows_valid:
            raise ValueError(f"Case identity/coverage validation failed for arm {arm}")
        if not manifest_valid or not model_valid or not selection_valid or not code_identity_valid:
            # Keep all computed diagnostics; the caller can see precisely why history is not reusable.
            pass
        all_rows[arm] = rows
        case_maps[arm] = row_map
        computed = _safe_metrics(rows)
        for field in ("cases", "completed", "accuracy_fixed_denominator", "answer_coverage"):
            if metrics_file.get(field) != computed.get(field):
                raise ValueError(f"Stored metric {field} disagrees with rows in arm {arm}")
        route_counts = dict(sorted(Counter(str(row.get("selected_action", "unknown")) for row in rows).items()))
        arm_audit[arm] = {
            "label": label,
            "valid": bool(manifest_valid and rows_valid and model_valid and selection_valid and code_identity_valid),
            "checks": {
                "manifest_completed_test": manifest_valid,
                "expected_case_count": len(rows) == EXPECTED_CASES,
                "case_id_set_matches_frozen_split": IDs_match,
                "duplicate_case_ids": duplicates,
                "result_identity_matches_config_arm_model": identity == expected_identity,
                "requested_and_served_model_match_frozen_qwen": model_valid,
                "selected_candidate_and_pilot_identity_valid": selection_valid,
                "frozen_code_commit_and_prompt_retrieval_sources_available": code_identity_valid,
                "canonical_medcpt_query_article_crossencoder": medcpt_canonical
                if arm == "rag_medcpt"
                else None,
            },
            "manifest_sha256": sha256_file(manifest_path),
            "case_results_sha256": sha256_file(result_path),
            "metrics_sha256": sha256_file(metrics_path),
            "result_identity": identity,
            "case_count": len(rows),
            "completed": computed["completed"],
            "provider_failures": computed["provider_failures"],
            "retrieval_artifact_failures": computed["retrieval_artifact_failures"],
            "completion_coverage": computed["answer_coverage"],
            "requested_model_ids": requested_values,
            "served_model_ids": [Path(value).name for value in served_values],
            "served_model_measurement_coverage": sum(bool(row.get("served_model")) for row in rows)
            / len(rows)
            if rows
            else 0.0,
            "selected_action_counts": route_counts,
            "metrics": computed,
        }

    all_arm_files_present = len(all_rows) == len(ARMS)
    fixed_valid = all(
        arm_audit[arm]["valid"]
        and (arm != "rag_medcpt" or arm_audit[arm]["checks"]["canonical_medcpt_query_article_crossencoder"])
        for arm in ARMS
        if arm in arm_audit
    ) and all_arm_files_present
    history_reusable = bool(fixed_valid and selection_valid and code_identity_valid)
    if not all_arm_files_present:
        raise ValueError("One or more E1.2 arms are missing required artifacts")

    case_subdatasets = {case_id: value["subdataset"] for case_id, value in expected_cases.items()}
    for arm, row_map in case_maps.items():
        for case_id, row in row_map.items():
            if str(row.get("subdataset")) != case_subdatasets[case_id]:
                raise ValueError(f"Subdataset metadata mismatch in arm {arm}")

    fixed_accuracy = {
        arm: float(arm_audit[arm]["metrics"]["accuracy_fixed_denominator"])
        for arm in FIXED_ACTIONS
    }
    oracle, labels = _summarize_oracle(expected_cases, case_maps, fixed_accuracy)

    audit = {
        "schema_version": "e1-3-mirage-history-audit-v1",
        "benchmark_status": "EXPOSED_EXPLORATORY",
        "audit_status": "REUSED_VALID_HISTORY" if history_reusable else "REUSE_BLOCKED",
        "reuse_history": history_reusable,
        "expected_case_count": EXPECTED_CASES,
        "source_identity": {
            "e1_2_frozen_config_sha256": config_hash,
            "split_manifest": "runs/e1_2/mirage_split_manifest.json",
            "split_manifest_sha256": split_hash,
            "benchmark_raw_sha256": benchmark.get("raw_sha256"),
            "normalized_cases_sha256": benchmark.get("normalized_cases_sha256"),
        },
        "generator_identity": {
            "candidate": "QWEN3_LOCAL",
            "requested_model": expected_model,
            "served_model": Path(expected_served).name,
            "expected_artifact_sha256": expected_model_hash,
            "expected_artifact_bytes": expected_model_bytes,
            "artifact_rehashed_during_this_audit": False,
            "artifact_rehash_note": "The frozen F: model path is not currently mounted; identity is verified against the committed frozen config, DEV pilot manifest, and per-case served_model string.",
            "selection_artifact_sha256": sha256_file(selection_path),
            "selection_pilot_valid": selection_valid,
        },
        "frozen_protocol_identity": {
            "code_base_commit": code_commit,
            "source_blob_ids": code_blobs,
            "code_prompt_identity_available": code_identity_valid,
            "prompt_mode": config.get("answer_protocol", {}).get("prompt_mode_all_arms"),
            "answer_protocol": {
                key: config.get("answer_protocol", {}).get(key)
                for key in ("temperature", "max_output_tokens", "response_format", "reasoning", "retries", "answer_extraction")
            },
            "retrieval_identity": retrieval_identity,
            "medcpt_canonical_identity_valid": medcpt_canonical,
        },
        "arms": arm_audit,
        "reusability_rationale": (
            "All six E1.2 arm manifests bind the same frozen config and candidate; every clean TEST case appears exactly once per arm, "
            "the served-model identity matches the frozen Qwen3-8B-Q4_K_M path, and the frozen code commit is available. "
            "No equivalent arm was rerun."
            if history_reusable
            else "At least one frozen identity, completion, or arm check failed; do not reuse the affected arm as a valid baseline."
        ),
    }
    report = {
        "schema_version": "e1-3-mirage-exploratory-report-v1",
        "evaluation_status": "EXPOSED EXPLORATORY BENCHMARK",
        "notice": "These results are not treated as an untouched confirmatory holdout.",
        "benchmark": "Medical MIRAGE / MedRAG clean TEST subdatasets only",
        "split_status": "EXPOSED_EXPLORATORY",
        "case_count": EXPECTED_CASES,
        "subdataset_case_counts": {
            name: sum(meta["subdataset"] == name for meta in expected_cases.values())
            for name in sorted(CLEAN_SUBDATASETS)
        },
        "history_reused": history_reusable,
        "arms": {
            arm: {
                "label": ARMS[arm],
                "accuracy_fixed_denominator": arm_audit[arm]["metrics"]["accuracy_fixed_denominator"],
                "answer_coverage": arm_audit[arm]["metrics"]["answer_coverage"],
                "invalid_answers": arm_audit[arm]["metrics"]["invalid_answers"],
                "invalid_answer_rate": arm_audit[arm]["metrics"]["invalid_answer_rate"],
                "provider_failures": arm_audit[arm]["metrics"]["provider_failures"],
                "answer_input_tokens_observed_total": arm_audit[arm]["metrics"]["answer_input_tokens_observed_total"],
                "answer_input_token_measurement_coverage": arm_audit[arm]["metrics"]["answer_input_token_measurement_coverage"],
                "answer_output_tokens_observed_total": arm_audit[arm]["metrics"]["answer_output_tokens_observed_total"],
                "answer_output_token_measurement_coverage": arm_audit[arm]["metrics"]["answer_output_token_measurement_coverage"],
                "retrieval_calls": arm_audit[arm]["metrics"]["retrieval_calls"],
                "retrieved_chunks": sum(int(row.get("retrieved_chunks", 0)) for row in all_rows[arm]),
                "context_characters": arm_audit[arm]["metrics"]["context_characters"],
                "latency_components": arm_audit[arm]["metrics"]["latency_components"],
                "selected_action_counts": arm_audit[arm]["selected_action_counts"],
                "random_context_doc_count_match_rate": arm_audit[arm]["metrics"]["random_context_doc_count_match_rate"],
                "random_context_char_budget_match_rate": arm_audit[arm]["metrics"]["random_context_char_budget_match_rate"],
                "by_subdataset": arm_audit[arm]["metrics"]["by_subdataset"],
            }
            for arm in ARMS
        },
        "historical_exposed_subdatasets": _load_historical_exposed_runs(repo_root),
        "oracle_opportunity": oracle,
        "oracle_label_rows": len(labels),
        "privacy_boundary": {
            "question_text_written": False,
            "gold_answer_written": False,
            "prediction_text_written": False,
            "evidence_text_written": False,
            "case_level_outcomes_written": True,
        },
    }
    return audit, report, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, default=Path("E:/Health-Copilot-E1.2"))
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/e1_3")
    args = parser.parse_args()

    audit, report, labels = audit_history(args.scratch_root, args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "mirage_history_audit.json"
    report_path = args.output_dir / "mirage_exploratory_report.json"
    labels_path = args.output_dir / "mirage_oracle_labels.jsonl"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with labels_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report["oracle_labels_sha256"] = sha256_file(labels_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "audit_status": audit["audit_status"],
                "reuse_history": audit["reuse_history"],
                "cases": EXPECTED_CASES,
                "outputs": [str(audit_path), str(report_path), str(labels_path)],
                "oracle_accuracy": report["oracle_opportunity"]["oracle_accuracy"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if audit["reuse_history"] else 2


if __name__ == "__main__":
    sys.exit(main())
