"""Post-freeze BUILD scoring with opaque skipping of non-BUILD truth rows."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from eval.rag_e6.data import _selected_top_level, load_partition_episodes
from eval.rag_e6.llm import MODEL_SHA256
from eval.rag_e6.reader_executor import (
    ARM_ORDER,
    DEFAULT_BGE_PATH,
    DEFAULT_QWEN_PATH,
    _retrieval_identity,
    _verify_inherited_u3r_runtime_source,
    _verify_server_manifest,
)
from eval.rag_e6.split import canonical_json_bytes, sha256_file, write_immutable_json

ROOT = Path(__file__).resolve().parents[2]
U2F_ROOT = ROOT / "runs/integration/u2f-owned-v1-55955b2eff38"
SPLIT_MANIFEST = ROOT / "runs/rag_e6/split_manifest.json"
CORPUS_ROOT = ROOT / "runs/rag_e6/corpus"
BUILD_ROOT = ROOT / "runs/rag_e6/build"
REPORT_JSON = ROOT / "runs/rag_e6/build/build_score_report.json"
SCORED_ROWS = ROOT / "runs/rag_e6/build/build_scored_episodes.jsonl"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260930
VALUE_TOKEN = re.compile(r"SYNVAL-[0-9A-F]{10}")
NUMBER_TOKEN = re.compile(r"(?<![\w.-])-?\d+(?![\w.])")
ALIAS_TOKEN = re.compile(r"\[E\d+\]")
FLOW_ARMS = ("VANILLA_STRONG", "CFEC_STRONG")
PRIMARY_CLASS = "RAG"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def _read_frozen_outputs(path: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != manifest.get("reader_output_sha256"):
        raise ValueError("BUILD reader outputs differ from their frozen manifest")
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            episode_id = row.get("episode_id")
            if not isinstance(episode_id, str) or episode_id in result:
                raise ValueError(f"malformed or duplicate BUILD output at row {line_number}")
            if row.get("partition") != "BUILD" or set(row.get("arms", {})) != set(ARM_ORDER):
                raise ValueError("BUILD output row has the wrong partition or arm set")
            result[episode_id] = row
    if len(result) != manifest.get("episode_count"):
        raise ValueError("BUILD output row count differs from its frozen manifest")
    return result


def _verify_execution_freeze(
    *, u2f_root: Path, split_manifest_path: Path, corpus_root: Path, build_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = build_root / "build_manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != "rag-e6a-runtime-freeze-v1"
        or manifest.get("partition") != "BUILD"
        or manifest.get("all_partition_episodes_executed") is not True
        or manifest.get("primary_slice_applied_before_execution") is not False
        or manifest.get("episode_count") != 818
        or manifest.get("arm_execution_count") != 818 * len(ARM_ORDER)
        or manifest.get("evaluator_truth_opened") is not False
        or manifest.get("future_train_outcomes_opened") is not False
        or manifest.get("reserved_test_ood_materialized") is not False
        or manifest.get("reserved_test_ood_opened") is not False
    ):
        raise ValueError("BUILD execution manifest is not a complete gold-blind freeze")
    code_hashes = manifest.get("code_sha256")
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise ValueError("BUILD manifest has no frozen code hashes")
    for name, expected_hash in code_hashes.items():
        if sha256_file(ROOT / name) != expected_hash:
            raise ValueError(f"BUILD implementation hash mismatch: {name}")
    frozen_commit = manifest.get("code_commit")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if not isinstance(frozen_commit, str) or subprocess.run(
        ["git", "merge-base", "--is-ancestor", frozen_commit, current_commit],
        cwd=ROOT,
        check=False,
    ).returncode != 0:
        raise ValueError("BUILD execution commit is not an ancestor of scoring HEAD")
    if manifest.get("retrieval") != _retrieval_identity():
        raise ValueError("BUILD retrieval identity differs from the pinned U3-R configuration")
    if manifest.get("inherited_u3r_runtime") != _verify_inherited_u3r_runtime_source(ROOT):
        raise ValueError("BUILD inherited U3-R runtime source identity changed")
    if manifest.get("generator", {}).get("model_sha256") != MODEL_SHA256:
        raise ValueError("BUILD generation model identity differs from the frozen Qwen GGUF")
    if sha256_file(DEFAULT_QWEN_PATH) != MODEL_SHA256:
        raise ValueError("pinned Qwen GGUF changed after BUILD execution")
    if sha256_file(DEFAULT_BGE_PATH / "model.safetensors") != _retrieval_identity()[
        "standard"
    ]["bge_weights_sha256"]:
        raise ValueError("pinned BGE-large weights changed after BUILD execution")
    server_manifest_path = build_root.parent / "runtime/gpu_server_manifest.json"
    server_manifest = _verify_server_manifest(server_manifest_path, qwen_path=DEFAULT_QWEN_PATH)
    if (
        sha256_file(server_manifest_path) != manifest.get("server_manifest_sha256")
        or server_manifest != manifest.get("server_manifest")
    ):
        raise ValueError("BUILD llama.cpp server manifest changed after execution")

    if sha256_file(build_root / "build_reader_outputs.jsonl") != manifest.get(
        "reader_output_sha256"
    ):
        raise ValueError("BUILD reader output hash mismatch")
    if sha256_file(build_root / "build_generation_calls.jsonl") != manifest.get(
        "generation_call_journal_sha256"
    ):
        raise ValueError("BUILD generation-call journal hash mismatch")
    split_manifest = _read_json(split_manifest_path)
    if sha256_file(split_manifest_path) != manifest.get("subject_split_manifest_sha256"):
        raise ValueError("BUILD subject split manifest hash mismatch")
    if split_manifest.get("evaluator_truth_opened") is not False:
        raise ValueError("subject split manifest records premature truth access")
    corpus_path = corpus_root / "build_runtime_corpus.jsonl"
    corpus_manifest_path = corpus_root / "build_corpus_manifest.json"
    corpus_manifest = _read_json(corpus_manifest_path)
    if (
        sha256_file(corpus_manifest_path) != manifest.get("runtime_corpus_manifest_sha256")
        or sha256_file(corpus_path) != manifest.get("runtime_corpus_sha256")
        or corpus_manifest.get("evaluator_truth_opened") is not False
        or corpus_manifest.get("future_train_outcomes_opened") is not False
    ):
        raise ValueError("gold-blind BUILD corpus failed its frozen identity checks")
    rows = _read_frozen_outputs(build_root / "build_reader_outputs.jsonl", manifest)
    episodes = load_partition_episodes(
        u2f_root=u2f_root,
        split_manifest_path=split_manifest_path,
        partition="BUILD",
    )
    expected_ids = {item.episode_id for item in episodes}
    if set(rows) != expected_ids:
        raise ValueError("frozen BUILD output IDs differ from selected runtime episodes")
    if len(episodes) != 818:
        raise ValueError("selected BUILD runtime episode count is not 818")

    journal: dict[str, dict[str, Any]] = {}
    with (build_root / "build_generation_calls.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            event = json.loads(line)
            call_id = event.get("call_id")
            if event.get("event") != "completed" or not isinstance(call_id, str):
                continue
            if call_id in journal:
                raise ValueError(f"duplicate completed generation call at journal row {line_number}")
            journal[call_id] = event
    for output in rows.values():
        bridge = output.get("retrieval_bridge", {})
        bridge_call_id = bridge.get("call_id")
        bridge_record = journal.get(bridge_call_id)
        if (
            bridge_record is None
            or hashlib.sha256(canonical_json_bytes(bridge_record)).hexdigest()
            != bridge.get("call_record_sha256")
        ):
            raise ValueError("LameR retrieval bridge call differs from the frozen call journal")
        for arm in output["arms"].values():
            for call_id, expected_hash in zip(
                arm.get("generation_call_ids", ()),
                arm.get("generation_call_records_sha256", ()),
                strict=True,
            ):
                record = journal.get(call_id)
                if record is None or hashlib.sha256(canonical_json_bytes(record)).hexdigest() != expected_hash:
                    raise ValueError("reader output references a different generation-call record")
        for vanilla_name, cfec_name in (
            ("VANILLA_STANDARD", "CFEC_STANDARD"),
            ("VANILLA_STRONG", "CFEC_STRONG"),
        ):
            vanilla, cfec = output["arms"][vanilla_name], output["arms"][cfec_name]
            for field in ("ranked_evidence_ids", "candidate_union_ids", "channel_ids", "evidence_identity_sha256"):
                if vanilla.get(field) != cfec.get(field):
                    raise ValueError(f"Vanilla and CFEC evidence mismatch for {field}")
        for arm in output["arms"].values():
            if not set(arm.get("used_evidence_ids", ())).issubset(
                set(arm.get("ranked_evidence_ids", ()))
            ):
                raise ValueError("reader used evidence outside its frozen top-10")
    return manifest, split_manifest, rows, {item.episode_id: item for item in episodes}


def _load_build_truth_slice(
    *, u2f_root: Path, split_manifest: dict[str, Any], expected_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selected_subjects = {
        item["subject_id"] for item in split_manifest["subjects_by_partition"]["BUILD"]
    }
    runtime_path = u2f_root / "train/episodes.jsonl"
    truth_path = u2f_root / "train/evaluator_truth.jsonl"
    truths: dict[str, dict[str, Any]] = {}
    selected_hash = hashlib.sha256()
    source_rows = 0
    opaque_truth_rows = 0
    with runtime_path.open(encoding="utf-8") as runtime_handle, truth_path.open(
        encoding="utf-8"
    ) as truth_handle:
        for source_rows, pair in enumerate(zip(runtime_handle, truth_handle, strict=True), start=1):
            runtime_line, truth_line = pair
            runtime_id = _selected_top_level(
                runtime_line, frozenset({"episode_id", "subject_id"})
            )
            if runtime_id.get("subject_id") not in selected_subjects:
                opaque_truth_rows += 1
                continue
            truth = json.loads(truth_line)
            if not isinstance(truth, dict):
                raise TypeError("selected BUILD evaluator-truth row is not an object")
            episode_id = runtime_id.get("episode_id")
            if (
                not isinstance(episode_id, str)
                or truth.get("episode_id") != episode_id
                or episode_id not in expected_ids
                or episode_id in truths
            ):
                raise ValueError("selected BUILD evaluator-truth row is misaligned or duplicated")
            required = {
                "episode_id", "answer_type", "answer_values", "capability_requirement_oracle",
                "required_memory_record_ids", "required_external_evidence_ids",
            }
            if not required.issubset(truth):
                raise ValueError("selected BUILD evaluator-truth row misses required scoring fields")
            selected_hash.update(truth_line.encode("utf-8"))
            truths[episode_id] = truth
    if source_rows != 4096 or set(truths) != expected_ids:
        raise ValueError("selected BUILD truth slice does not align to all 818 episodes")
    return truths, {
        "truth_source_path": str(truth_path),
        "truth_file_rows_aligned": source_rows,
        "build_truth_rows_decoded": len(truths),
        "non_build_truth_rows_skipped_without_json_decode": opaque_truth_rows,
        "build_truth_slice_sha256": selected_hash.hexdigest(),
        "frozen_dev_truth_rows_decoded": 0,
        "future_train_truth_rows_decoded": 0,
        "reserved_test_ood_opened": False,
    }


def _observed_values(answer: str, truth: dict[str, Any]) -> tuple[str, ...]:
    expected = tuple(str(value) for value in truth["answer_values"])
    answer_type = truth["answer_type"]
    without_aliases = ALIAS_TOKEN.sub("", answer)
    if answer_type == "BOOLEAN":
        return tuple(
            item.upper()
            for item in re.findall(r"\b(?:TRUE|FALSE)\b", without_aliases, re.IGNORECASE)
        )
    if expected and all(value in {"UP", "DOWN", "STABLE"} for value in expected):
        return tuple(
            item.upper()
            for item in re.findall(r"\b(?:UP|DOWN|STABLE)\b", without_aliases, re.IGNORECASE)
        )
    values = VALUE_TOKEN.findall(without_aliases)
    if any(value.isdigit() for value in expected):
        values.extend(NUMBER_TOKEN.findall(without_aliases))
    return tuple(values)


def _score_arm(
    *, truth: dict[str, Any], runtime_arm: dict[str, Any], visible_ids: set[str]
) -> dict[str, Any]:
    answer = str(runtime_arm.get("answer", ""))
    used = set(runtime_arm.get("used_evidence_ids", ()))
    ranked = set(runtime_arm.get("ranked_evidence_ids", ()))
    required_memory = set(truth.get("required_memory_record_ids", ()))
    required_external = set(truth.get("required_external_evidence_ids", ()))
    required_resources = required_memory | required_external
    missing_resources = required_resources - used
    oracle = truth["capability_requirement_oracle"]
    answerable = bool(oracle["answerability"])
    expected = tuple(str(value) for value in truth["answer_values"])

    if not answerable:
        observed: tuple[str, ...] = ()
        answer_value_correct = answer == "INSUFFICIENT_EVIDENCE"
        task_success = answer_value_correct and not used
    else:
        observed = _observed_values(answer, truth)
        answer_type = truth["answer_type"]
        if answer_type == "ORDERED_SEQUENCE":
            answer_value_correct = len(observed) == len(expected) and observed == expected
        elif answer_type == "EXACT_TOKEN":
            answer_value_correct = len(expected) == 1 and observed == expected
        else:
            answer_value_correct = (
                set(observed) == set(expected) and len(observed) == len(set(observed))
            )
        task_success = answer_value_correct and not missing_resources

    provenance_pass = used.issubset(ranked) and used.issubset(visible_ids)
    full_resource_grounding = not missing_resources
    grounding_pass = provenance_pass and full_resource_grounding
    external_coverage = (
        len(required_external & used) / len(required_external) if required_external else None
    )
    candidate_has_full_evidence = bool(required_external) and required_external.issubset(ranked)
    return {
        "task_success": bool(task_success),
        "grounded_task_success": bool(task_success and grounding_pass),
        "grounding_pass": bool(grounding_pass),
        "provenance_pass": bool(provenance_pass),
        "answer_value_correct": bool(answer_value_correct),
        "answer_value_coverage": (
            len(set(observed) & set(expected)) / len(set(expected)) if expected else None
        ),
        "external_evidence_coverage": external_coverage,
        "required_external_count": len(required_external),
        "required_external_used_count": len(required_external & used),
        "candidate_has_full_external_evidence": candidate_has_full_evidence,
        "used_evidence_count": len(used),
        "citation_present": bool(used),
        "output_contract_failure": bool(runtime_arm.get("output_contract_failure")),
        "claim_output_contract_failures": int(runtime_arm.get("claim_output_contract_failures", 0)),
        "unknown_alias_count": len(runtime_arm.get("unknown_aliases", ())),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    observations = [row["arms"][arm] for row in rows]
    metric_names = (
        "task_success", "grounded_task_success", "grounding_pass", "provenance_pass",
        "answer_value_correct", "answer_value_coverage", "external_evidence_coverage",
        "citation_present", "output_contract_failure",
    )
    result: dict[str, Any] = {"count": len(observations)}
    for name in metric_names:
        values = [float(item[name]) for item in observations if item[name] is not None]
        result[name] = _mean(values)
        result[f"{name}_n"] = len(values)
    result["claim_output_contract_failures"] = sum(
        item["claim_output_contract_failures"] for item in observations
    )
    result["unknown_aliases"] = sum(item["unknown_alias_count"] for item in observations)
    return result


def _paired_cluster_bootstrap(
    rows: list[dict[str, Any]], *, metric: str, arm_a: str, arm_b: str,
    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
    only_full_evidence: bool = False,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if not only_full_evidence
        or row["arms"][arm_a]["candidate_has_full_external_evidence"]
        and row["arms"][arm_b]["candidate_has_full_external_evidence"]
    ]
    if not selected:
        return {"count": 0, "subjects": 0, "delta": None, "ci95": None}
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_subject[row["subject_id"]].append(row)
    subject_ids = np.asarray(sorted(by_subject), dtype=object)
    point = float(np.mean([
        float(row["arms"][arm_b][metric]) - float(row["arms"][arm_a][metric])
        for row in selected
    ]))
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled_subjects = rng.choice(subject_ids, size=len(subject_ids), replace=True)
        differences = [
            float(row["arms"][arm_b][metric]) - float(row["arms"][arm_a][metric])
            for subject in sampled_subjects
            for row in by_subject[str(subject)]
        ]
        samples[index] = float(np.mean(differences))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "count": len(selected),
        "subjects": len(subject_ids),
        "delta": point,
        "ci95": [float(low), float(high)],
        "resamples": resamples,
        "seed": seed,
        "cluster_unit": "subject_id; paired arms use identical sampled clusters",
        "eligibility": "all required external evidence in both arms' top-10"
        if only_full_evidence else "all rows in the requested capability slice",
    }


def score_build(
    *, u2f_root: Path = U2F_ROOT, split_manifest_path: Path = SPLIT_MANIFEST,
    corpus_root: Path = CORPUS_ROOT, build_root: Path = BUILD_ROOT,
    report_path: Path = REPORT_JSON, scored_rows_path: Path = SCORED_ROWS,
) -> dict[str, Any]:
    manifest, split_manifest, runtime_by_id, episode_by_id = _verify_execution_freeze(
        u2f_root=u2f_root,
        split_manifest_path=split_manifest_path,
        corpus_root=corpus_root,
        build_root=build_root,
    )
    truth_by_id, truth_access = _load_build_truth_slice(
        u2f_root=u2f_root,
        split_manifest=split_manifest,
        expected_ids=set(runtime_by_id),
    )
    with (corpus_root / "build_runtime_corpus.jsonl").open(encoding="utf-8") as handle:
        visible_docs: dict[str, set[str]] = {}
        for line in handle:
            row = json.loads(line)
            visible_docs[row["episode_id"]] = {
                item["doc_id"] for item in row["documents"]
            }

    scored_rows: list[dict[str, Any]] = []
    for episode_id, runtime_row in runtime_by_id.items():
        episode = episode_by_id[episode_id]
        truth = truth_by_id[episode_id]
        capability_class = truth["capability_requirement_oracle"]["derived_capability_class"]
        arms = {
            arm: _score_arm(
                truth=truth,
                runtime_arm=runtime_row["arms"][arm],
                visible_ids=visible_docs[episode_id],
            )
            for arm in ARM_ORDER
        }
        scored_rows.append({
            "episode_id": episode_id,
            "subject_id": episode.subject_id,
            "capability_class": capability_class,
            "scenario_family": truth.get("structural_evaluator_metadata", {}).get(
                "scenario_family"
            ),
            "available_source_family_count": len(episode.available_source_families),
            "arms": arms,
        })

    slices: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        slices[row["capability_class"]].append(row)
    all_metrics = {
        capability_class: {
            arm: _aggregate(rows, arm)
            for arm in ARM_ORDER
        }
        for capability_class, rows in sorted(slices.items())
    }
    rag_rows = slices.get(PRIMARY_CLASS, [])
    if not rag_rows:
        raise ValueError("frozen BUILD contains no post-freeze RAG evaluation slice")
    primary_delta = _paired_cluster_bootstrap(
        rag_rows, metric="grounded_task_success", arm_a=FLOW_ARMS[0], arm_b=FLOW_ARMS[1]
    )
    task_delta = _paired_cluster_bootstrap(
        rag_rows, metric="task_success", arm_a=FLOW_ARMS[0], arm_b=FLOW_ARMS[1]
    )
    grounding_delta = _paired_cluster_bootstrap(
        rag_rows, metric="grounding_pass", arm_a=FLOW_ARMS[0], arm_b=FLOW_ARMS[1]
    )
    utilization_delta = _paired_cluster_bootstrap(
        rag_rows,
        metric="task_success",
        arm_a=FLOW_ARMS[0],
        arm_b=FLOW_ARMS[1],
        only_full_evidence=True,
    )
    gate = {
        "primary_build_point_delta_at_least_10pp": (
            primary_delta["delta"] is not None and primary_delta["delta"] >= 0.10
        ),
        "primary_build_ci_lower_above_zero": (
            primary_delta["ci95"] is not None and primary_delta["ci95"][0] > 0
        ),
        "grounding_degradation_no_more_than_1pp": (
            grounding_delta["delta"] is not None and grounding_delta["delta"] >= -0.01
        ),
        "utilization_build_point_delta_at_least_10pp": (
            utilization_delta["delta"] is not None and utilization_delta["delta"] >= 0.10
        ),
        "note": "BUILD is developmental only; this is not the frozen DEV gate.",
    }
    rows_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in scored_rows)
    if scored_rows_path.exists() and scored_rows_path.read_bytes() != rows_bytes:
        raise FileExistsError("refusing to overwrite a different BUILD scoring artifact")
    if not scored_rows_path.exists():
        scored_rows_path.parent.mkdir(parents=True, exist_ok=True)
        with scored_rows_path.open("xb") as handle:
            handle.write(rows_bytes)
            handle.flush()
    result = {
        "schema_version": "rag-e6a-build-score-v1",
        "partition": "BUILD",
        "runtime_manifest_sha256": sha256_file(build_root / "build_manifest.json"),
        "reader_output_sha256": manifest["reader_output_sha256"],
        "generation_call_journal_sha256": manifest["generation_call_journal_sha256"],
        "evaluator_truth_opened_after_execution_freeze": True,
        "truth_access": truth_access,
        "episode_count": len(scored_rows),
        "capability_class_counts": {
            key: len(rows) for key, rows in sorted(slices.items())
        },
        "metrics_by_capability_class_and_arm": all_metrics,
        "primary_comparison": {
            "slice": "RAG",
            "candidate": FLOW_ARMS[1],
            "baseline": FLOW_ARMS[0],
            "grounded_task_success_delta": primary_delta,
            "task_success_delta": task_delta,
            "grounding_pass_delta": grounding_delta,
            "utilization_given_full_evidence_task_success_delta": utilization_delta,
        },
        "development_gate": gate,
        "scored_episode_rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
        "future_train_outcomes_decoded": False,
        "reserved_test_ood_materialized": False,
        "reserved_test_ood_opened": False,
        "frozen_dev_truth_opened": False,
    }
    write_immutable_json(report_path, result)
    return result
