"""One-shot reserved scoring; this is the only E6B runtime truth consumer."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from eval.rag_e6.reader_executor import _write_immutable_bytes
from eval.rag_e6.scoring import (
    _observed_values,
    _paired_cluster_bootstrap,
    _retrieval_contract_failed,
    _score_arm,
)
from eval.rag_e6.split import canonical_json_bytes, sha256_file

from .protocol import (
    ARMS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    OOD_POOLS,
    POOL_SEEDS,
    RESERVED_COMPOSITIONS,
    RUN_ROOT_RELATIVE,
    read_json,
    sha256_bytes,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"expected JSON object at {path.name}:{line_number}")
            rows.append(row)
    return rows


def _read_verified_truth(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    """Read a truth shard once, verifying the exact committed materialization bytes."""
    payload = path.read_bytes()
    actual_sha256 = sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"reserved evaluator truth hash mismatch: {path.name}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"expected JSON object at {path.name}:{line_number}")
        rows.append(row)
    return rows


def _committed_freeze(repository_root: Path, relative: str) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.stdout.strip() or tracked.returncode != 0:
        raise ValueError(f"execution freeze artifact is not committed and clean: {relative}")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=repository_root,
        text=True,
    ).strip()
    if not commit:
        raise ValueError("execution freeze commit could not be identified")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return commit


def _percent(value: float | None) -> float | None:
    return None if value is None else round(100.0 * value, 6)


def _rate(rows: list[dict[str, Any]], arm: str, metric: str) -> dict[str, Any]:
    values = [row["arms"][arm][metric] for row in rows if row["arms"][arm][metric] is not None]
    return {"n": len(values), "rate": sum(map(float, values)) / len(values) if values else None}


def _cluster_ci(
    rows: list[dict[str, Any]], *, metric: str, arm_a: str = ARMS[0], arm_b: str = ARMS[1],
    only_full_evidence: bool = False,
) -> dict[str, Any]:
    result = _paired_cluster_bootstrap(
        rows,
        metric=metric,
        arm_a=arm_a,
        arm_b=arm_b,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
        only_full_evidence=only_full_evidence,
    )
    result["delta_pp"] = _percent(result.get("delta"))
    result["ci95_pp"] = (
        [_percent(item) for item in result["ci95"]] if result.get("ci95") else None
    )
    return result


def _composition_labels(row: dict[str, Any]) -> tuple[str, ...]:
    truth = row["truth_metadata"]
    structure = truth["structural_metadata"]
    capability = row["capability_class"]
    labels: list[str] = []
    if capability == "MEMORY+RAG" and int(structure.get("dependency_depth", 1)) >= 3:
        labels.append("deep MEMORY→RAG serial chains")
    if (
        capability == "MEMORY+RAG"
        and int(structure.get("evidence_branch_count", 0)) >= 2
        and int(structure.get("revision_depth", 0)) >= 1
    ):
        labels.append("multi-source RAG plus memory revision")
    if (
        int(structure.get("independent_dependency_groups", 0)) >= 3
        or int(structure.get("evidence_branch_count", 0)) >= 3
        or int(structure.get("dependency_width", 0)) >= 3
    ):
        labels.append("three-branch compositional tasks")
    if (
        truth.get("history_regime") in {"LONG", "SATURATED"}
        and int(structure.get("external_versions", 0)) >= 2
    ):
        labels.append("long-history plus versioned external evidence")
    return tuple(labels)


def _failure_modes(
    row: dict[str, Any], *, arm: str, visible_ids: set[str],
) -> tuple[str, ...]:
    truth = row["truth_metadata"]
    runtime = row["runtime_arms"][arm]
    score = row["arms"][arm]
    if score["grounded_task_success"]:
        return ()
    required = set(truth.get("required_external_evidence_ids", ()))
    ranked = set(runtime.get("ranked_evidence_ids", ()))
    ledger = runtime.get("rsel_ledger", ())
    modes: list[str] = []
    if arm == "VANILLA_STRONG":
        expected = tuple(str(value) for value in truth.get("answer_values", ()))
        observed = _observed_values(str(runtime.get("answer", "")), truth)
        if len(expected) > 1 and len(observed) < len(expected):
            modes.append("MULTI_VALUE_DROPOUT")
        elif set(observed).issubset(set(expected)) and set(observed) != set(expected):
            modes.append("VALUE_OMISSION")
        elif set(observed) - set(expected):
            modes.append("UNSUPPORTED_EXTRA_VALUE")
        elif observed and set(observed) != set(expected):
            modes.append("WRONG_KEY_VALUE_BINDING")
        if not score["grounding_pass"]:
            modes.append("GROUNDING_FAILURE")
        if not runtime.get("used_evidence_ids") and required:
            modes.append("CITATION_UTILIZATION_FAILURE")
        if not score["output_contract_pass"]:
            modes.append("OUTPUT_CONTRACT_FAILURE")
    else:
        action = runtime.get("rsel_action")
        if truth["capability_requirement_oracle"]["derived_capability_class"] == "INSUFFICIENT":
            modes.append("INSUFFICIENT_EVIDENCE")
        elif row["capability_class"] == "MEMORY+RAG":
            modes.append("COMPOSITION_BEYOND_RSEL_SCOPE")
        elif action == "FALLBACK_NO_MATCH":
            if row["pool"].startswith("OOD_"):
                modes.append("OOD_GRAMMAR_SHIFT")
            modes.append("NO_STRUCTURED_RELATION_MATCH")
        elif len(ledger) > max(1, len(truth.get("answer_values", ()))):
            modes.append("AMBIGUOUS_OR_MULTIPLE_RELATIONS")
        if required - ranked:
            modes.append("RETRIEVAL_MISSING_REQUIRED_EVIDENCE")
        if not required.issubset(visible_ids) and required:
            modes.append("IMPLEMENTATION_CONTRACT_FAILURE")
        if not score["output_contract_pass"]:
            modes.append("IMPLEMENTATION_CONTRACT_FAILURE")
    return tuple(dict.fromkeys(modes or ("OTHER",)))


def _score_pool(
    pool: str,
    runtime_rows: list[dict[str, Any]],
    truth_rows: list[dict[str, Any]],
    visible_by_id: dict[str, set[str]],
) -> list[dict[str, Any]]:
    runtime_by_id = {row["episode_id"]: row for row in runtime_rows}
    truth_by_id = {row["episode_id"]: row for row in truth_rows}
    if set(runtime_by_id) != set(truth_by_id) or len(runtime_by_id) != len(runtime_rows):
        raise ValueError(f"post-freeze truth/runtime IDs are misaligned for {pool}")
    scored: list[dict[str, Any]] = []
    for runtime in runtime_rows:
        truth = truth_by_id[runtime["episode_id"]]
        capability = truth["capability_requirement_oracle"]["derived_capability_class"]
        visible = visible_by_id[runtime["episode_id"]]
        arms = {
            arm: _score_arm(
                truth=truth,
                runtime_arm=runtime["arms"][arm],
                visible_ids=visible,
                retrieval_contract_failure=_retrieval_contract_failed(
                    arm, runtime.get("retrieval_bridge", {})
                ),
            )
            for arm in ARMS
        }
        scored.append({
            "episode_id": runtime["episode_id"],
            "subject_id": runtime["subject_id"],
            "pool": pool,
            "capability_class": capability,
            "scenario_family": truth.get("scenario_family"),
            "available_source_family_count": len(
                runtime.get("visible_source_families", ())
            ),
            "arms": arms,
            "runtime_arms": runtime["arms"],
            "truth_metadata": {
                "answer_type": truth["answer_type"],
                "answer_values": truth["answer_values"],
                "capability_requirement_oracle": truth["capability_requirement_oracle"],
                "required_external_evidence_ids": truth["required_external_evidence_ids"],
                "required_memory_record_ids": truth["required_memory_record_ids"],
                "structural_metadata": truth.get("structural_metadata", {}),
                "history_regime": truth.get("history_regime"),
                "template_family": truth.get("template_family"),
            },
        })
    return scored


def _pool_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rag_rows = [row for row in rows if row["capability_class"] == "RAG"]
    metrics = {
        arm: {
            name: _rate(rag_rows, arm, name)
            for name in (
                "grounded_task_success", "task_success", "answer_value_correct",
                "grounding_pass", "provenance_pass", "external_evidence_coverage",
                "output_contract_pass", "retrieval_contract_pass",
            )
        }
        for arm in ARMS
    }
    # Required-evidence-used coverage is equivalent to the external evidence coverage
    # on answerable RAG cases and is emitted separately for easier interpretation.
    for arm in ARMS:
        metrics[arm]["required_evidence_used_coverage"] = metrics[arm][
            "external_evidence_coverage"
        ]
    result = {
        "all_episodes": len(rows),
        "all_subjects": len({row["subject_id"] for row in rows}),
        "rag_episodes": len(rag_rows),
        "rag_subjects": len({row["subject_id"] for row in rag_rows}),
        "rag_metrics": metrics,
        "grounded_success_delta": _cluster_ci(rag_rows, metric="grounded_task_success"),
        "grounding_delta": _cluster_ci(rag_rows, metric="grounding_pass"),
        "full_evidence_utilization_delta": _cluster_ci(
            rag_rows, metric="task_success", only_full_evidence=True
        ),
        "rsel_action_counts_all": dict(sorted(Counter(
            row["runtime_arms"]["RSEL_STRONG"].get("rsel_action", "MISSING") for row in rows
        ).items())),
        "rsel_action_counts_rag": dict(sorted(Counter(
            row["runtime_arms"]["RSEL_STRONG"].get("rsel_action", "MISSING")
            for row in rag_rows
        ).items())),
    }
    for arm in ARMS:
        for name in metrics[arm]:
            metrics[arm][name]["rate_pp"] = _percent(metrics[arm][name]["rate"])
    return result


def score_reserved(*, repository_root: Path) -> dict[str, Any]:
    """Open reserved truth exactly once, only after the output-freeze commit."""
    repository_root = repository_root.resolve()
    run_root = repository_root / RUN_ROOT_RELATIVE
    reserved_root = run_root / "reserved"
    execution_root = run_root / "execution"
    execution_manifest_path = execution_root / "reserved_execution_manifest.json"
    execution_relative = str(execution_manifest_path.relative_to(repository_root)).replace("\\", "/")
    execution_freeze_commit = _committed_freeze(repository_root, execution_relative)
    execution_manifest = read_json(execution_manifest_path)
    method_freeze = read_json(run_root / "method_freeze.json")
    materialization_path = reserved_root / "reserved_materialization_manifest.json"
    materialization_relative = str(
        materialization_path.relative_to(repository_root)
    ).replace("\\", "/")
    materialization_commit = _committed_freeze(repository_root, materialization_relative)
    materialization = read_json(materialization_path)
    reader_path = execution_root / "reserved_reader_outputs.jsonl"
    journal_path = execution_root / "reserved_generation_calls.jsonl"
    checkpoint_path = execution_root / "reserved_episode_checkpoint.jsonl"
    for artifact in (reader_path, journal_path, checkpoint_path):
        _committed_freeze(
            repository_root,
            str(artifact.relative_to(repository_root)).replace("\\", "/"),
        )
    if execution_manifest.get("materialization_commit") != materialization_commit:
        raise ValueError("execution manifest does not name the committed materialization revision")

    if (
        execution_manifest.get("all_reserved_episodes_executed") is not True
        or execution_manifest.get("all_required_arm_executions_complete") is not True
        or execution_manifest.get("paired_ranked_evidence_identity_pass") is not True
        or execution_manifest.get("paired_evidence_bytes_identity_pass") is not True
        or execution_manifest.get("rsel_no_match_parity_pass") is not True
        or execution_manifest.get("evaluator_truth_opened") is not False
        or execution_manifest.get("extra_rsel_model_calls") != 0
        or sha256_file(reader_path) != execution_manifest.get("reader_output_sha256")
        or sha256_file(journal_path) != execution_manifest.get("call_journal_sha256")
        or materialization.get("evaluator_truth_opened") is not False
        or method_freeze.get("evaluator_truth_opened") is not False
    ):
        raise ValueError("execution freeze failed pre-score audit")
    one_shot_outputs = (
        run_root / "reserved_score_report.json",
        run_root / "reserved_scored_episodes.jsonl",
        run_root / "bootstrap_results.json",
        run_root / "failure_taxonomy.json",
        run_root / "post_training_candidate_manifest.json",
    )
    if any(path.exists() for path in one_shot_outputs) or (run_root / "score_started.json").exists():
        raise FileExistsError("reserved scoring is one-shot and has already started")
    runtime_rows = _read_jsonl(reader_path)
    if len(runtime_rows) != execution_manifest["total_episode_count"]:
        raise ValueError("frozen reader-output count differs from execution manifest")
    runtime_by_pool = {
        pool: [row for row in runtime_rows if row.get("pool") == pool]
        for pool in POOL_SEEDS
    }
    if any(
        len(runtime_by_pool[pool]) != int(materialization["actual_episodes_per_pool"][pool])
        for pool in POOL_SEEDS
    ):
        raise ValueError("frozen reader outputs do not cover every planned pool")
    visible_by_pool: dict[str, dict[str, set[str]]] = {}
    for pool in POOL_SEEDS:
        corpus = _read_jsonl(reserved_root / "runtime" / pool / "runtime_corpus.jsonl")
        visible_by_pool[pool] = {
            row["episode_id"]: {item["doc_id"] for item in row["documents"]}
            for row in corpus
        }
        if len(visible_by_pool[pool]) != len(corpus):
            raise ValueError(f"duplicate runtime corpus episode in {pool}")

    # Persist the one-shot guard before the first read of any evaluator truth file.
    score_started = {
        "schema_version": "rag-e6b-one-shot-score-start-v1",
        "execution_freeze_commit": execution_freeze_commit,
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "truth_access_started": True,
        "truth_opened_before_freeze": False,
        "scoring_seed": 20261002,
    }
    score_started_path = run_root / "score_started.json"
    _write_immutable_bytes(score_started_path, canonical_json_bytes(score_started) + b"\n")

    scored_by_pool: dict[str, list[dict[str, Any]]] = {}
    for pool in POOL_SEEDS:
        truth_path = reserved_root / "evaluator_only" / pool / "evaluator_truth.jsonl"
        expected_truth_sha = materialization["truth_hashes"][pool][
            "evaluator_truth_sha256"
        ]
        truth_rows = _read_verified_truth(truth_path, expected_truth_sha)
        scored_by_pool[pool] = _score_pool(
            pool,
            runtime_by_pool[pool],
            truth_rows,
            visible_by_pool[pool],
        )

    all_scored = [row for pool in POOL_SEEDS for row in scored_by_pool[pool]]
    rag_iid = [row for row in scored_by_pool["IID_TEST"] if row["capability_class"] == "RAG"]
    iid_report = _pool_report(scored_by_pool["IID_TEST"])
    iid_delta = iid_report["grounded_success_delta"]
    iid_grounding_delta = iid_report["grounding_delta"]
    iid_utilization_delta = iid_report["full_evidence_utilization_delta"]
    gate_parts = {
        "grounded_success_delta_at_least_10pp": (
            iid_delta["delta_pp"] is not None and iid_delta["delta_pp"] >= 10.0
        ),
        "paired_cluster_ci_lower_above_zero": (
            iid_delta["ci95_pp"] is not None and iid_delta["ci95_pp"][0] > 0
        ),
        "grounding_degradation_at_least_minus_1pp": (
            iid_grounding_delta["delta_pp"] is not None
            and iid_grounding_delta["delta_pp"] >= -1.0
        ),
        "full_evidence_utilization_delta_at_least_10pp": (
            iid_utilization_delta["delta_pp"] is not None
            and iid_utilization_delta["delta_pp"] >= 10.0
        ),
        "no_extra_rsel_model_calls": execution_manifest["extra_rsel_model_calls"] == 0,
    }
    iid_gate = "PASS" if all(gate_parts.values()) else "FAIL"
    ood_reports = {pool: _pool_report(scored_by_pool[pool]) for pool in OOD_POOLS}
    pooled_ood_rows = [row for pool in OOD_POOLS for row in scored_by_pool[pool]
                       if row["capability_class"] == "RAG"]
    pooled_ood_subjects = [row["subject_id"] for row in pooled_ood_rows]
    pooled_ood = _cluster_ci(pooled_ood_rows, metric="grounded_task_success")
    ood_supported = bool(
        pooled_ood.get("delta_pp") is not None
        and pooled_ood["delta_pp"] > 0
        and pooled_ood.get("ci95_pp")
        and pooled_ood["ci95_pp"][0] > 0
        and all(
            report["grounded_success_delta"].get("delta_pp") is not None
            and report["grounded_success_delta"]["delta_pp"] >= 0
            for report in ood_reports.values()
        )
    )

    action_slices: dict[str, dict[str, Any]] = {}
    rsel_no_match_parity = True
    for label in ("RSEL_MATCH", "RSEL_NO_MATCH"):
        action = (
            "STRUCTURED_RELATION_LEDGER" if label == "RSEL_MATCH" else "FALLBACK_NO_MATCH"
        )
        selected = [row for row in all_scored
                    if row["runtime_arms"]["RSEL_STRONG"].get("rsel_action") == action]
        vanilla_success = _rate(selected, ARMS[0], "grounded_task_success")
        rsel_success = _rate(selected, ARMS[1], "grounded_task_success")
        exact_parity = all(
            all(
                row["runtime_arms"]["VANILLA_STRONG"].get(field)
                == row["runtime_arms"]["RSEL_STRONG"].get(field)
                for field in (
                    "answer", "answer_sha256", "cited_aliases", "used_evidence_ids",
                    "output_contract_failure", "unknown_aliases",
                )
            )
            for row in selected
        ) if label == "RSEL_NO_MATCH" else None
        if label == "RSEL_NO_MATCH":
            rsel_no_match_parity &= bool(exact_parity)
        action_slices[label] = {
            "count": len(selected),
            "vanilla_grounded_success_pp": _percent(vanilla_success["rate"]),
            "rsel_grounded_success_pp": _percent(rsel_success["rate"]),
            "delta_pp": _percent(
                None if vanilla_success["rate"] is None or rsel_success["rate"] is None
                else rsel_success["rate"] - vanilla_success["rate"]
            ),
            "exact_answer_and_evidence_parity": exact_parity,
        }

    action_rate = {
        "IID_TEST": _action_rate(scored_by_pool["IID_TEST"]),
        "OOD": _action_rate([row for pool in OOD_POOLS for row in scored_by_pool[pool]]),
    }

    comp_rows = scored_by_pool["OOD_COMPOSITION"]
    composition_reports: dict[str, Any] = {}
    for name in RESERVED_COMPOSITIONS:
        selected = [row for row in comp_rows if name in _composition_labels(row)]
        rag_selected = [row for row in selected if row["capability_class"] == "RAG"]
        composition_reports[name] = {
            "episodes": len(selected),
            "subjects": len({row["subject_id"] for row in selected}),
            "capability_counts": dict(sorted(Counter(
                row["capability_class"] for row in selected
            ).items())),
            "rag_only_episodes": len(rag_selected),
            "rag_only_subjects": len({row["subject_id"] for row in rag_selected}),
            "rag_only_grounded_success_delta": _cluster_ci(
                rag_selected, metric="grounded_task_success"
            ),
            "is_headline": False,
        }

    class_sanity: dict[str, Any] = {}
    for capability in ("NONE", "INSUFFICIENT"):
        selected = [row for row in all_scored if row["capability_class"] == capability]
        class_sanity[capability] = {
            "episodes": len(selected),
            "subjects": len({row["subject_id"] for row in selected}),
            "metrics": {
                arm: {
                    key: _rate(selected, arm, key)
                    for key in ("task_success", "grounded_task_success", "output_contract_pass")
                }
                for arm in ARMS
            },
            "rsel_action_counts": dict(sorted(Counter(
                row["runtime_arms"]["RSEL_STRONG"].get("rsel_action") for row in selected
            ).items())),
        }

    vanilla_failures: Counter[str] = Counter()
    rsel_failures: Counter[str] = Counter()
    for row in all_scored:
        if row["capability_class"] != "RAG":
            continue
        visible = visible_by_pool[row["pool"]][row["episode_id"]]
        vanilla = row["arms"]["VANILLA_STRONG"]
        rsel = row["arms"]["RSEL_STRONG"]
        if not vanilla["grounded_task_success"] and rsel["grounded_task_success"]:
            vanilla_failures.update(_failure_modes(row, arm="VANILLA_STRONG", visible_ids=visible))
        if not rsel["grounded_task_success"]:
            rsel_failures.update(_failure_modes(row, arm="RSEL_STRONG", visible_ids=visible))

    candidates: list[dict[str, Any]] = []
    for row in all_scored:
        if row["capability_class"] != "RAG":
            continue
        vanilla_runtime = row["runtime_arms"]["VANILLA_STRONG"]
        rsel_runtime = row["runtime_arms"]["RSEL_STRONG"]
        required = set(row["truth_metadata"]["required_external_evidence_ids"])
        retrieved = set(vanilla_runtime["ranked_evidence_ids"])
        if (
            rsel_runtime.get("rsel_action") == "FALLBACK_NO_MATCH"
            and required.issubset(retrieved)
            and not row["arms"]["VANILLA_STRONG"]["grounded_task_success"]
        ):
            candidates.append({
                "episode_id": row["episode_id"],
                "subject_id": row["subject_id"],
                "pool": row["pool"],
                "query_sha256": vanilla_runtime["query_sha256"],
                "candidate_type": "RSEL_NO_MATCH_FULL_EVIDENCE_VANILLA_FAILURE",
                "failure_modes": list(_failure_modes(
                    row,
                    arm="VANILLA_STRONG",
                    visible_ids=visible_by_pool[row["pool"]][row["episode_id"]],
                )),
                "gold_included": False,
            })

    scored_rows = [{
        key: value for key, value in row.items()
        if key not in {"runtime_arms", "truth_metadata"}
    } for row in all_scored]
    scored_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in scored_rows)
    scored_path = run_root / "reserved_scored_episodes.jsonl"
    report = {
        "schema_version": "rag-e6b-reserved-score-v1",
        "one_shot": True,
        "truth_opened_after_execution_freeze": True,
        "execution_freeze_commit": execution_freeze_commit,
        "execution_manifest_sha256": sha256_file(execution_manifest_path),
        "materialization_manifest_sha256": sha256_file(
            reserved_root / "reserved_materialization_manifest.json"
        ),
        "reader_output_sha256": sha256_file(reader_path),
        "call_journal_sha256": sha256_file(journal_path),
        "bootstrap": {
            "method": "paired subject-cluster bootstrap",
            "cluster": "subject_id",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95% CI",
        },
        "reserved_total_episodes": len(all_scored),
        "reserved_total_subjects": len({row["subject_id"] for row in all_scored}),
        "pool_reports": {"IID_TEST": iid_report, **ood_reports},
        "iid_test_rag_population": {
            "episodes": len(rag_iid),
            "subjects": len({row["subject_id"] for row in rag_iid}),
        },
        "iid_gate_components": gate_parts,
        "iid_gate": iid_gate,
        "pooled_ood_rag": {
            "episodes": len(pooled_ood_rows),
            "subjects": len(set(pooled_ood_subjects)),
            "grounded_success_delta": pooled_ood,
        },
        "ood_generalization_supported": ood_supported,
        "rsel_match_no_match": {
            "action_rates": action_rate,
            "slices": action_slices,
            "no_match_parity": "PASS" if rsel_no_match_parity else "FAIL",
        },
        "ood_composition_families": composition_reports,
        "none_insufficient_sanity": class_sanity,
        "top_3_vanilla_failure_modes": vanilla_failures.most_common(3),
        "top_3_remaining_rsel_failure_modes": rsel_failures.most_common(3),
        "failure_taxonomy_counts": {
            "vanilla_fail_rsel_success": dict(sorted(vanilla_failures.items())),
            "remaining_rsel_failures": dict(sorted(rsel_failures.items())),
        },
        "post_training_candidate_count": len(candidates),
        "sft_started": False,
        "opd_started": False,
        "grpo_started": False,
        "method_modified_after_freeze": False,
    }

    # All truth-dependent artifacts are write-once outputs of this sole scorer.
    _write_immutable_bytes(scored_path, scored_bytes)
    bootstrap_path = run_root / "bootstrap_results.json"
    _write_immutable_bytes(
        bootstrap_path,
        canonical_json_bytes({
            "iid_test_rag": iid_report["grounded_success_delta"],
            "ood_pools": {pool: result["grounded_success_delta"]
                          for pool, result in ood_reports.items()},
            "pooled_ood_rag": pooled_ood,
            "seed": BOOTSTRAP_SEED,
            "resamples": BOOTSTRAP_RESAMPLES,
        }) + b"\n",
    )
    taxonomy_path = run_root / "failure_taxonomy.json"
    _write_immutable_bytes(taxonomy_path, canonical_json_bytes(report["failure_taxonomy_counts"]) + b"\n")
    candidate_manifest = {
        "schema_version": "rag-e6b-post-training-candidates-v1",
        "method": "candidate manifest only; no SFT/OPD/GRPO started",
        "selection": (
            "RAG capability; RSEL_NO_MATCH; all required external evidence in shared top-10; "
            "Vanilla grounded-task failure"
        ),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "gold_included": False,
        "rsel_structured_relation_successes_excluded": True,
    }
    candidate_path = run_root / "post_training_candidate_manifest.json"
    _write_immutable_bytes(candidate_path, canonical_json_bytes(candidate_manifest) + b"\n")
    report["artifact_sha256"] = {
        str(path.relative_to(run_root)).replace("\\", "/"): sha256_file(path)
        for path in (
            scored_path,
            bootstrap_path,
            taxonomy_path,
            candidate_path,
        )
    }
    report_path = run_root / "reserved_score_report.json"
    _write_immutable_bytes(
        report_path, canonical_json_bytes(report) + b"\n"
    )
    hashes = dict(report["artifact_sha256"])
    hashes["reserved_score_report.json"] = sha256_file(report_path)
    _write_immutable_bytes(run_root / "score_artifact_hashes.json", canonical_json_bytes(hashes) + b"\n")
    return report


def _action_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["runtime_arms"]["RSEL_STRONG"].get("rsel_action") for row in rows)
    total = len(rows)
    return {
        "episodes": total,
        "counts": dict(sorted(counts.items())),
        "match_rate_pct": round(100 * counts.get("STRUCTURED_RELATION_LEDGER", 0) / total, 6)
        if total else None,
        "no_match_rate_pct": round(100 * counts.get("FALLBACK_NO_MATCH", 0) / total, 6)
        if total else None,
    }
