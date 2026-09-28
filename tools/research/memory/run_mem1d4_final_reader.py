"""Run MEM-1D4 against only the frozen MEM-1D1 ContextBundles."""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = ROOT / "tools" / "research" / "memory"
sys.path.insert(0, str(TOOLS_DIR))

import run_mem1d3_reader as d3
from final_reader_contract import (
    build_reader_messages,
    load_final_reader_contract,
)

D1_RUN = ROOT / "runs" / "memory" / "mem1" / "mem1d1-frozen-10-20260927"
D3_RUN = ROOT / "runs" / "memory" / "mem1" / "mem1d3-reader-v2-20260928"
D4_RUN = ROOT / "runs" / "memory" / "mem1" / "mem1d4-reader-v3-20260928"
QUESTION_DATES_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d3_question_dates.json"
COUNTERFACTUAL_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d3_reader_counterfactual.json"
TEMPLATE_PATH = ROOT / "docs" / "research" / "memory" / "shared_reader_v3_final.txt"
CONTRACT_PATH = ROOT / "docs" / "research" / "memory" / "final_reader_contract.json"
REVIEW_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d4_reader_review.json"
REPORT_PATH = ROOT / "docs" / "research" / "memory" / "mem_1d4_final_reader_contract.md"
LOCAL_BASE_URL = d3.LOCAL_BASE_URL
SYSTEMS = d3.SYSTEMS
QUESTION_IDS = d3.QUESTION_IDS
TEMPORAL_IDS = set(d3.TEMPORAL_IDS)

D2_PATHS = (
    ROOT / "docs" / "research" / "memory" / "mem_1d2_provenance_overlay.json",
    ROOT / "docs" / "research" / "memory" / "mem_1d2_reflection_packet.json",
    ROOT / "docs" / "research" / "memory" / "mem_1d2_retrieval_metrics.json",
    ROOT / "docs" / "research" / "memory" / "mem_1d2_reflection_and_provenance.md",
)
D1_EXTRA_PATHS = (
    D1_RUN / "run_manifest.json",
    D1_RUN / "deterministic_metrics.json",
    D1_RUN / "embeddings_usage.json",
    D1_RUN / "token_efficiency.json",
    D1_RUN / "report.md",
)
D3_PATHS = (
    D3_RUN / "predictions.jsonl",
    D3_RUN / "predictions.sha256",
    D3_RUN / "call_ledger.jsonl",
    D3_RUN / "call_ledger.sha256",
    D3_RUN / "run_config.json",
    D3_RUN / "run_config.sha256",
    D3_RUN / "run_manifest.json",
    QUESTION_DATES_PATH,
    QUESTION_DATES_PATH.with_name(f"{QUESTION_DATES_PATH.name}.sha256"),
    COUNTERFACTUAL_PATH,
    COUNTERFACTUAL_PATH.with_name(f"{COUNTERFACTUAL_PATH.name}.sha256"),
    ROOT / "docs" / "research" / "memory" / "shared_reader_v2_question_date.txt",
    ROOT / "docs" / "research" / "memory" / "mem_1d3_question_date_contract.md",
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _freeze_sidecar(path: Path) -> str:
    digest = d3._sha256_file(path)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def _freeze_jsonl_sidecar(path: Path) -> str:
    digest = d3._sha256_file(path)
    path.with_name(path.name.replace(".jsonl", ".sha256")).write_text(
        f"{digest}  {path.name}\n", encoding="ascii", newline="\n"
    )
    return digest


def _snapshot_frozen_inputs() -> dict[str, str]:
    paths = [D1_RUN / name for name in d3.FROZEN_D1_ARTIFACTS]
    paths.extend(D1_EXTRA_PATHS)
    paths.extend(D2_PATHS)
    paths.extend(D3_PATHS)
    result = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required historical MEM-1 evidence is missing: {path}")
        result[path.relative_to(ROOT).as_posix()] = d3._sha256_file(path)
    for name in d3.FROZEN_D1_ARTIFACTS:
        artifact = D1_RUN / name
        if name.endswith(".sha256"):
            continue
        sidecar = D1_RUN / name.replace(".jsonl", ".sha256")
        if not d3._verify_hash_sidecar(artifact, sidecar):
            raise RuntimeError(f"Frozen MEM-1D1 sidecar verification failed: {name}")
    for artifact_name in ("predictions.jsonl", "call_ledger.jsonl", "run_config.json"):
        artifact = D3_RUN / artifact_name
        sidecar_name = "run_config.sha256" if artifact_name == "run_config.json" else artifact_name.replace(".jsonl", ".sha256")
        sidecar = D3_RUN / sidecar_name
        if not d3._verify_hash_sidecar(artifact, sidecar):
            raise RuntimeError(f"Frozen MEM-1D3 sidecar verification failed: {artifact_name}")
    for artifact in (QUESTION_DATES_PATH, COUNTERFACTUAL_PATH):
        if not d3._verify_hash_sidecar(artifact, artifact.with_name(f"{artifact.name}.sha256")):
            raise RuntimeError(f"Frozen MEM-1D3 document sidecar verification failed: {artifact.name}")
    return result


def _assert_same_metrics(run_mem1: Any, row: dict[str, Any], label: str) -> dict[str, float]:
    prediction = row.get("predicted")
    if not isinstance(prediction, str):
        raise TypeError(f"Historical {label} prediction is not a successful text answer")
    metrics = run_mem1._answer_metrics(prediction, str(row["ground_truth"]))
    for key, value in metrics.items():
        if not math.isclose(float(row[key]), value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Historical {label} {key} differs from deterministic metric recomputation")
    return metrics


def _load_frozen_rows(run_mem1: Any):
    from context_bundle import verify_context_bundle
    from mem1_artifacts import read_jsonl

    d1_manifest = json.loads((D1_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    d3_manifest = json.loads((D3_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        d1_manifest.get("split") != "DEV"
        or d1_manifest.get("test_access") is not False
        or tuple(d1_manifest.get("question_ids", [])) != QUESTION_IDS
    ):
        raise RuntimeError("Historical MEM-1D1 run is not the exact frozen DEV ten-case run")
    if (
        d3_manifest.get("status") != "COMPLETE"
        or d3_manifest.get("split") != "DEV"
        or d3_manifest.get("test_access") is not False
        or tuple(d3_manifest.get("question_ids", [])) != QUESTION_IDS
        or d3_manifest.get("reader_calls_successful") != 50
    ):
        raise RuntimeError("Historical MEM-1D3 run is not the complete frozen DEV-only counterfactual")

    date_table = json.loads(QUESTION_DATES_PATH.read_text(encoding="utf-8"))
    if (
        date_table.get("artifact_version") != "mem1d3-question-dates-v1"
        or date_table.get("test_access") is not False
        or tuple(row["question_id"] for row in date_table.get("records", [])) != QUESTION_IDS
        or d3_manifest.get("question_dates_sha256") != d3._sha256_file(QUESTION_DATES_PATH)
    ):
        raise RuntimeError("MEM-1D3 official question-date table failed its frozen identity checks")
    dates_by_id = {row["question_id"]: row for row in date_table["records"]}

    d1_bundles = read_jsonl(D1_RUN / "context_bundles.jsonl")
    d1_predictions = read_jsonl(D1_RUN / "predictions.jsonl")
    d3_predictions = read_jsonl(D3_RUN / "predictions.jsonl")
    expected_keys = {(system, question_id) for system in SYSTEMS for question_id in QUESTION_IDS}
    bundles = {(row["system"], row["question_id"]): row["context_bundle"] for row in d1_bundles}
    v1 = {(row["system"], row["question_id"]): row for row in d1_predictions}
    v2 = {(row["system"], row["question_id"]): row for row in d3_predictions}
    if any(len(rows) != 50 for rows in (d1_bundles, d1_predictions, d3_predictions)):
        raise RuntimeError("MEM-1D1/D3 must each have exactly 50 frozen matrix rows")
    if set(bundles) != expected_keys or set(v1) != expected_keys or set(v2) != expected_keys:
        raise RuntimeError("Frozen reader-only matrices do not match the exact 5×10 system/question matrix")

    source_hashes = _snapshot_frozen_inputs()
    prepared = []
    for system in SYSTEMS:
        for question_id in QUESTION_IDS:
            key = (system, question_id)
            bundle, old_v1, old_v2 = bundles[key], v1[key], v2[key]
            if not verify_context_bundle(bundle):
                raise RuntimeError(f"Historical ContextBundle canonical SHA invalid: {key}")
            bundle_sha = bundle.get("context_bundle_sha256")
            if bundle_sha != old_v1.get("context_bundle_sha256") or bundle_sha != old_v2.get("context_bundle_sha256"):
                raise RuntimeError(f"D1/D3 predictions do not share the immutable ContextBundle: {key}")
            if old_v1.get("quality_status") != "OK" or old_v2.get("quality_status") != "OK":
                raise RuntimeError(f"Historical reader output is not a successful answer: {key}")
            if old_v1.get("shared_reader_template_sha256") != run_mem1.D1_READER_TEMPLATE_SHA256:
                raise RuntimeError(f"Historical reader-v1 template identity mismatch: {key}")
            date_row = dates_by_id[question_id]
            if old_v2.get("question_date") != date_row["question_date"]:
                raise RuntimeError(f"D3 prediction date differs from official date table: {key}")
            if old_v2.get("question_type") != date_row["question_type"]:
                raise RuntimeError(f"D3 category differs from official date table: {key}")
            if old_v1.get("question") != old_v2.get("question") or old_v1.get("ground_truth") != old_v2.get("ground_truth"):
                raise RuntimeError(f"Historical v1/v2 question or gold answer differs: {key}")
            metrics_v1 = _assert_same_metrics(run_mem1, old_v1, "v1")
            metrics_v2 = _assert_same_metrics(run_mem1, old_v2, "v2")
            prepared.append({
                "system": system,
                "question_id": question_id,
                "question_type": date_row["question_type"],
                "question_date": date_row["question_date"],
                "question": old_v1["question"],
                "ground_truth": old_v1["ground_truth"],
                "context_bundle": bundle,
                "context_bundle_sha256": bundle_sha,
                "source_record_sha256": date_row["source_record_sha256"],
                "v1": old_v1,
                "v2": old_v2,
                "v1_metrics": metrics_v1,
                "v2_metrics": metrics_v2,
            })
    return d1_manifest, d3_manifest, date_table, prepared, source_hashes


def _metric_row(run_mem1: Any, predicted: str, gold: str) -> dict[str, float]:
    return run_mem1._answer_metrics(predicted, gold)


def _build_comparison(prepared, v3_by_key, template_sha: str, run_mem1: Any):
    rows = []
    for item in prepared:
        key = (item["system"], item["question_id"])
        new = v3_by_key[key]
        v1, v2 = item["v1"], item["v2"]
        p1, p2, p3 = v1["predicted"], v2["predicted"], new["predicted"]
        metrics_v3 = _metric_row(run_mem1, p3, item["ground_truth"]) if isinstance(p3, str) else None
        rows.append({
            "system": item["system"],
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "gold_answer": item["ground_truth"],
            "question_date": item["question_date"],
            "context_bundle_sha256": item["context_bundle_sha256"],
            "predictions": {"v1": p1, "v2": p2, "v3": p3},
            "lexical_metrics": {
                "v1": item["v1_metrics"],
                "v2": item["v2_metrics"],
                "v3": metrics_v3,
            },
            "prediction_string_changed": {
                "v1_vs_v2": p1 != p2,
                "v1_vs_v3": p1 != p3,
                "v2_vs_v3": p2 != p3,
            },
            "prompt_sha256": {
                "v1": v1["shared_reader_prompt_sha256"],
                "v2": v2["shared_reader_prompt_sha256"],
                "v3": new["shared_reader_prompt_sha256"],
            },
            "v3_template_sha256": template_sha,
            "human_review": {"outcome": None, "failure_locus": None, "notes": None},
            "automatic_semantic_or_failure_labels": [],
            "human_review_status": "PENDING_HUMAN_REVIEW",
        })
    return rows


def _comparison_summary(rows):
    summary = {
        "artifact_version": "mem1d4-three-way-reader-comparison-v1",
        "interpretation": "descriptive prompt-sensitivity diagnostics only; no reader ranking or selection by score",
        "row_count": len(rows),
        "version_metrics": {},
        "prompt_change_counts": {},
        "groups": {},
    }
    for version in ("v1", "v2", "v3"):
        metrics = [row["lexical_metrics"][version] for row in rows if row["lexical_metrics"][version] is not None]
        summary["version_metrics"][version] = {
            "scored_rows": len(metrics),
            "mean_token_precision": (sum(row["token_precision"] for row in metrics) / len(metrics)) if metrics else None,
            "mean_token_recall": (sum(row["token_recall"] for row in metrics) / len(metrics)) if metrics else None,
            "mean_token_f1": (sum(row["f1"] for row in metrics) / len(metrics)) if metrics else None,
            "normalized_exact_match_rate": (sum(row["normalized_exact_match"] for row in metrics) / len(metrics)) if metrics else None,
        }
    for comparison in ("v1_vs_v2", "v1_vs_v3", "v2_vs_v3"):
        summary["prompt_change_counts"][comparison] = sum(
            row["prediction_string_changed"][comparison] for row in rows
        )
    for group_name, predicate in (
        ("temporal_10", lambda row: row["question_id"] in TEMPORAL_IDS),
        ("non_temporal_40", lambda row: row["question_id"] not in TEMPORAL_IDS),
    ):
        group = [row for row in rows if predicate(row)]
        summary["groups"][group_name] = {
            "row_count": len(group),
            "question_ids": list(dict.fromkeys(row["question_id"] for row in group)),
            "prediction_change_counts": {
                comparison: sum(row["prediction_string_changed"][comparison] for row in group)
                for comparison in ("v1_vs_v2", "v1_vs_v3", "v2_vs_v3")
            },
            "mean_token_f1_by_version": {
                version: (
                    sum(row["lexical_metrics"][version]["f1"] for row in group if row["lexical_metrics"][version] is not None)
                    / sum(row["lexical_metrics"][version] is not None for row in group)
                    if any(row["lexical_metrics"][version] is not None for row in group)
                    else None
                )
                for version in ("v1", "v2", "v3")
            },
        }
    return summary


def _report_text(config, manifest, summary, preflight, source_before, source_after):
    fullcontext = [row for row in preflight if row["system"] == "fullcontext"]
    unchanged = source_before == source_after
    gates = manifest["gate"]
    rows = [
        "# MEM-1D4 Minimal Date-Aware Final Reader Contract",
        "",
        f"- Gate: `MEM1D4_FINAL_READER_CONTRACT_FROZEN={'YES' if gates['passed'] else 'NO'}`",
        "- Selection rule: protocol-only; no answer-quality-based reader selection or tuning.",
        "- Research position: controlled re-evaluation of public memory architectures under a unified fully-local model stack.",
        "- Scope: frozen MEM-1D1 DEV ten questions × five system identities; reader-only; no memory architecture change.",
        "",
        "## Contract",
        "",
        f"- Reader: `{config['reader']['model']}`, artifact SHA256 `{config['reader']['model_sha256']}`.",
        f"- Final contract SHA256: `{manifest['final_reader_contract_sha256']}`.",
        f"- v3 raw template SHA256: `{config['template']['raw_sha256']}`.",
        f"- v3 canonical message-template SHA256: `{config['template']['canonical_message_template_sha256']}`.",
        f"- Official source: LongMemEval `{config['official_source']['revision']}` `src/generation/run_generation.py`, raw UTF-8 SHA256 `{config['official_source']['raw_file_sha256_utf8']}`.",
        "- Source alignment only: Health-Copilot keeps its own shared answer head; no exact official prompt reproduction claim.",
        "- Structural finding: official `question_date` is required and copied exactly from the hash-verified D3 table.",
        "- v3 equals the v1 answering policy plus only the `Current Date` field; there are no temporal examples, task/category hints, retrieval instructions, or reasoning directives.",
        "- Permanent choice: date-complete, minimal, shared, category-agnostic, source-aligned; chosen by protocol, not DEV scores. Reader-v4 tuning on these results is prohibited absent an infrastructure or benchmark-contract defect.",
        "- Explicit selection gate: `FINAL_READER_SELECTED_BY_PROTOCOL_NOT_DEV_SCORE=YES`.",
        "- Contract enforcement: the file SHA is pinned in the shared validator, included in context-controlled run manifests, prediction rows, and cache identity; any mismatch fails closed. M10-Base, RevMem, 102 DEV, and held-out TEST runners must bind this same SHA.",
        "",
        "## Execution",
        "",
        f"- Reader calls: `{manifest['reader_calls_successful']}/{manifest['reader_calls_expected']}`; systems: `{', '.join(SYSTEMS)}`.",
        f"- Memory-system calls: `{manifest['memory_system_calls']}`; embedding: `{manifest['embedding_calls']}`; judge: `{manifest['judge_calls']}`; hosted: `{manifest['hosted_api_calls']}`.",
        f"- ContextBundles: `{manifest['context_bundles_verified']}/50` verified from MEM-1D1, copied/rebuilt: `{manifest['context_bundles_copied_or_rebuilt']}`.",
        f"- D1/D2/D3 historical evidence byte/hash unchanged: `{unchanged}`.",
        f"- TEST access: `{manifest['test_access']}`.",
        f"- FullContext fit: `{sum(row['fits'] and not row['truncated'] for row in fullcontext)}/10`; every request had 256 output tokens reserved under the 131072 context limit.",
        "- Token preflight uses the frozen llama.cpp `/apply-template` and `/tokenize`; server prompt usage is compared with preflight for each answer.",
        "",
        "| Question ID | Preflight prompt tokens | Server prompt tokens | Reserve | Fits | Truncated |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in fullcontext:
        rows.append(
            f"| {row['question_id']} | {row['prompt_tokens_preflight']} | {row['prompt_tokens_server']} | 256 | {row['fits']} | {row['truncated']} |"
        )
    rows.extend([
        "",
        "## Three-Way Diagnostic",
        "",
        "The deterministic summary and 50-row review packet report v1, v2, and v3 token precision/recall/F1/normalized EM and exact-string change flags. These are descriptive prompt-sensitivity measurements, not a leaderboard or reader-selection criterion.",
        f"- Temporal slice: `{summary['groups']['temporal_10']['row_count']}` system×question rows across `gpt4_e061b84g`, `gpt4_f420262c`.",
        f"- Non-temporal slice: `{summary['groups']['non_temporal_40']['row_count']}` rows, reported separately.",
        "- A changed answer string is not automatically a failure. Human semantic-review fields remain null; no automatic causal/failure labels were assigned.",
        "- D2 provenance/evidence artifacts remain preserved; this stage does not assign temporal failure loci.",
        "",
        "## Frozen Artifacts",
        "",
        f"- Final contract: `{CONTRACT_PATH.relative_to(ROOT).as_posix()}` and SHA sidecar.",
        f"- v3 template: `{TEMPLATE_PATH.relative_to(ROOT).as_posix()}`.",
        f"- Review packet: `{REVIEW_PATH.relative_to(ROOT).as_posix()}`.",
        f"- Reader-only run: `{D4_RUN.relative_to(ROOT).as_posix()}`.",
        f"- Run prediction SHA256: `{manifest.get('predictions_sha256')}`; call-ledger SHA256: `{manifest.get('call_ledger_sha256')}`.",
        "",
        "STOP: do not start MEM-1 102-case DEV, TEST, M10-Base, RevMem, or RL without a separately reviewed next-stage request.",
        "",
    ])
    return "\n".join(rows)


def run() -> dict[str, Any]:
    if D4_RUN.exists():
        raise FileExistsError(f"Refusing to overwrite reader-only D4 run: {D4_RUN}")
    if REVIEW_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("Refusing to overwrite an existing MEM-1D4 review/report artifact")

    contract, contract_sha, system_template, user_template = load_final_reader_contract()
    source_before = _snapshot_frozen_inputs()
    sys.path.insert(0, str(TOOLS_DIR))
    import run_mem1

    d1_manifest, _, _, prepared, source_hashes_before = _load_frozen_rows(run_mem1)
    if source_hashes_before != source_before:
        raise RuntimeError("Historical source evidence changed while D4 inputs were being verified")

    prepared_prompts = []
    for item in prepared:
        messages = build_reader_messages(
            item["question"],
            item["question_date"],
            item["context_bundle"]["serialized_context"],
            system_template=system_template,
            user_template=user_template,
        )
        prepared_prompts.append({
            **item,
            "messages": messages,
            "prompt_sha256": d3._sha256_bytes(d3._canonical_json(messages)),
        })

    with httpx.Client(timeout=3600, trust_env=False) as client:
        runtime = d3._runtime_preflight(d1_manifest)
        fullcontext_preflight = []
        for row in prepared_prompts:
            tokens, rendered_sha = d3._render_and_tokenize(client, row["messages"])
            row["prompt_tokens_preflight"] = tokens
            row["rendered_prompt_sha256"] = rendered_sha
            if row["system"] == "fullcontext":
                fullcontext_preflight.append({
                    "system": row["system"],
                    "question_id": row["question_id"],
                    "prompt_tokens_preflight": tokens,
                    "prompt_tokens_server": None,
                    "output_reserve": d3.ANSWER_MAX_NEW_TOKENS,
                    "max_model_length": d3.MAX_MODEL_LENGTH,
                    "fits": tokens + d3.ANSWER_MAX_NEW_TOKENS <= d3.MAX_MODEL_LENGTH,
                    "truncated": tokens + d3.ANSWER_MAX_NEW_TOKENS > d3.MAX_MODEL_LENGTH,
                })

    fullcontext_fits = (
        len(fullcontext_preflight) == len(QUESTION_IDS)
        and all(row["fits"] and not row["truncated"] for row in fullcontext_preflight)
    )
    if not fullcontext_fits:
        raise RuntimeError("v3 FullContext prompt does not fit; no answers were generated")

    config = {
        "config_version": "mem1d4-reader-only-v1",
        "run_id": D4_RUN.name,
        "split": "DEV",
        "test_access": False,
        "question_ids": list(QUESTION_IDS),
        "systems": list(SYSTEMS),
        "reader": {
            "model": runtime["model"],
            "model_sha256": runtime["model_sha256"],
            "runtime": runtime,
        },
        "template": {
            "path": TEMPLATE_PATH.relative_to(ROOT).as_posix(),
            "raw_sha256": d3._sha256_file(TEMPLATE_PATH),
            "canonical_message_template_sha256": contract["template"]["canonical_message_template_sha256"],
            "final_reader_contract_sha256": contract_sha,
            "date_field_only_delta_from_v1": True,
            "selected_by_protocol_not_dev_score": True,
        },
        "official_source": contract["official_source_alignment"],
        "question_dates_sha256": d3._sha256_file(QUESTION_DATES_PATH),
        "context_bundles": {
            "source_run_id": d1_manifest["run_id"],
            "source_jsonl_sha256": source_hashes_before["runs/memory/mem1/mem1d1-frozen-10-20260927/context_bundles.jsonl"],
            "verified_count": 50,
            "copied_or_rebuilt": False,
        },
        "calls": {"reader_answer": 50, "memory_system": 0, "embedding": 0, "judge": 0, "hosted_api": 0},
        "fullcontext_preflight": fullcontext_preflight,
    }
    config["configuration_sha256"] = d3._sha256_bytes(
        d3._canonical_json(config)
    )

    D4_RUN.mkdir(parents=True, exist_ok=False)
    config_path = D4_RUN / "run_config.json"
    _write_json(config_path, config)
    config_sha = _freeze_sidecar(config_path)
    predictions_path = D4_RUN / "predictions_v3.jsonl"
    ledger_path = D4_RUN / "call_ledger.jsonl"
    v3_by_key = {}
    successful_calls = 0
    all_server_prompt_counts_match = True
    fullcontext_actual = {}

    with httpx.Client(timeout=3600, trust_env=False) as client:
        for call_index, row in enumerate(prepared_prompts, 1):
            started = time.perf_counter()
            timestamp = datetime.now(UTC).isoformat()
            request = {
                "model": d3.MODEL_ALIAS,
                "messages": row["messages"],
                "temperature": 0,
                "seed": 42,
                "max_tokens": d3.ANSWER_MAX_NEW_TOKENS,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_sha = d3._sha256_bytes(d3._canonical_json(request))
            answer = None
            usage = {}
            status_code = None
            error_type = None
            finish_reason = None
            try:
                response = client.post(f"{LOCAL_BASE_URL}/chat/completions", json=request)
                status_code = response.status_code
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                answer_text = choice.get("message", {}).get("content")
                if not isinstance(answer_text, str):
                    raise TypeError("Frozen local reader response contained no text answer")
                answer = answer_text.strip()
                usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
                finish_reason = choice.get("finish_reason")
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
                error_type = type(error).__name__
            latency_ms = round((time.perf_counter() - started) * 1000, 3)
            server_prompt_tokens = usage.get("prompt_tokens")
            tokens_match = server_prompt_tokens == row["prompt_tokens_preflight"]
            all_server_prompt_counts_match = all_server_prompt_counts_match and tokens_match
            if row["system"] == "fullcontext":
                fullcontext_actual[row["question_id"]] = {
                    "question_id": row["question_id"],
                    "prompt_tokens_preflight": row["prompt_tokens_preflight"],
                    "prompt_tokens_server": server_prompt_tokens,
                    "fits": row["prompt_tokens_preflight"] + d3.ANSWER_MAX_NEW_TOKENS <= d3.MAX_MODEL_LENGTH,
                    "truncated": False if answer is not None and tokens_match else None,
                }
                fullcontext_row = next(
                    item for item in fullcontext_preflight
                    if item["question_id"] == row["question_id"]
                )
                fullcontext_row["prompt_tokens_server"] = server_prompt_tokens
                fullcontext_row["truncated"] = fullcontext_actual[row["question_id"]]["truncated"]

            call_row = {
                "call_index": call_index,
                "timestamp_utc": timestamp,
                "role": "reader_answer",
                "provider": "local_qwen",
                "endpoint": LOCAL_BASE_URL,
                "model": d3.MODEL_ALIAS,
                "system": row["system"],
                "question_id": row["question_id"],
                "context_bundle_sha256": row["context_bundle_sha256"],
                "final_reader_contract_sha256": contract_sha,
                "prompt_sha256": row["prompt_sha256"],
                "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                "request_sha256": request_sha,
                "temperature": 0,
                "seed": 42,
                "enable_thinking": False,
                "max_new_tokens": d3.ANSWER_MAX_NEW_TOKENS,
                "prompt_tokens_preflight": row["prompt_tokens_preflight"],
                "prompt_tokens_server": server_prompt_tokens,
                "completion_tokens_server": usage.get("completion_tokens"),
                "prompt_tokens_match": tokens_match,
                "finish_reason": finish_reason,
                "latency_ms": latency_ms,
                "http_status": status_code,
                "success": answer is not None,
                "error_type": error_type,
                "retry_count": 0,
                "hosted_call": False,
            }
            _append_jsonl(ledger_path, call_row)
            metrics = _metric_row(run_mem1, answer, row["ground_truth"]) if answer is not None else {
                "token_precision": None,
                "token_recall": None,
                "f1": None,
                "normalized_exact_match": None,
            }
            prediction = {
                "system": row["system"],
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "question_date": row["question_date"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "predicted": answer,
                "quality_status": "OK" if answer is not None else "INFRA_FAILURE",
                **metrics,
                "reader_prompt_tokens_preflight": row["prompt_tokens_preflight"],
                "reader_prompt_tokens_server": server_prompt_tokens,
                "completion_tokens_server": usage.get("completion_tokens"),
                "prompt_tokens_match": tokens_match,
                "reader_latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "output_hit_token_cap": finish_reason == "length",
                "input_truncated": False if answer is not None and tokens_match else None,
                "context_bundle_sha256": row["context_bundle_sha256"],
                "source_record_sha256": row["source_record_sha256"],
                "final_reader_contract_sha256": contract_sha,
                "shared_reader_template_sha256": contract["template"]["canonical_message_template_sha256"],
                "shared_reader_prompt_sha256": row["prompt_sha256"],
                "rendered_prompt_sha256": row["rendered_prompt_sha256"],
                "reader_error_type": error_type,
                "call_index": call_index,
            }
            _append_jsonl(predictions_path, prediction)
            v3_by_key[(row["system"], row["question_id"])] = prediction
            if answer is not None:
                successful_calls += 1

    predictions_sha = _freeze_jsonl_sidecar(predictions_path)
    ledger_sha = _freeze_jsonl_sidecar(ledger_path)
    comparison_rows = _build_comparison(
        prepared_prompts,
        v3_by_key,
        contract["template"]["canonical_message_template_sha256"],
        run_mem1,
    )
    comparison = _comparison_summary(comparison_rows)
    v2_config = json.loads((D3_RUN / "run_config.json").read_text(encoding="utf-8"))
    comparison["source_reader_template_sha256"] = {
        "v1": run_mem1.D1_READER_TEMPLATE_SHA256,
        "v2": v2_config["template"]["message_template_sha256"],
        "v3": contract["template"]["canonical_message_template_sha256"],
    }
    _write_json(D4_RUN / "deterministic_comparison_summary.json", comparison)
    comparison_sha = _freeze_sidecar(D4_RUN / "deterministic_comparison_summary.json")
    review = {
        "artifact_version": "mem1d4-reader-review-v1",
        "semantic_review_status": "PENDING_HUMAN_REVIEW",
        "automatic_answer_or_failure_labels_assigned": False,
        "rows": comparison_rows,
    }
    _write_json(REVIEW_PATH, review)
    review_sha = _freeze_sidecar(REVIEW_PATH)

    source_after = _snapshot_frozen_inputs()
    source_unchanged = source_before == source_after
    ledger_rows = d3._load_jsonl(ledger_path)
    prediction_rows = d3._load_jsonl(predictions_path)
    expected_keys = {(system, question_id) for system in SYSTEMS for question_id in QUESTION_IDS}
    all_prediction_keys = {(row["system"], row["question_id"]) for row in prediction_rows}
    all_local_reader = len(ledger_rows) == 50 and all(
        row.get("role") == "reader_answer"
        and row.get("provider") == "local_qwen"
        and row.get("hosted_call") is False
        and row.get("final_reader_contract_sha256") == contract_sha
        for row in ledger_rows
    )
    all_success = successful_calls == 50 and all(row["quality_status"] == "OK" for row in prediction_rows)
    fullcontext_valid = len(fullcontext_actual) == 10 and all(
        row["prompt_tokens_server"] == row["prompt_tokens_preflight"]
        and row["prompt_tokens_preflight"] + d3.ANSWER_MAX_NEW_TOKENS <= d3.MAX_MODEL_LENGTH
        and row["truncated"] is False
        for row in fullcontext_actual.values()
    )
    template_valid = (
        d3._sha256_file(TEMPLATE_PATH) == contract["template"]["raw_sha256"]
        and contract["template"]["canonical_message_template_sha256"]
        == d3._sha256_bytes(d3._canonical_json([
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_template},
        ]))
    )
    output_freeze_valid = (
        d3._verify_hash_sidecar(predictions_path, predictions_path.with_name("predictions_v3.sha256"))
        and d3._verify_hash_sidecar(ledger_path, ledger_path.with_name("call_ledger.sha256"))
        and d3._verify_hash_sidecar(D4_RUN / "run_config.json", D4_RUN / "run_config.json.sha256")
        and d3._verify_hash_sidecar(D4_RUN / "deterministic_comparison_summary.json", D4_RUN / "deterministic_comparison_summary.json.sha256")
        and d3._verify_hash_sidecar(REVIEW_PATH, REVIEW_PATH.with_name(f"{REVIEW_PATH.name}.sha256"))
    )
    official_source = contract["official_source_alignment"]
    gate = {
        "final_reader_selected_by_protocol_not_dev_score": contract["decision"][
            "selected_by_protocol_not_dev_score"
        ] is True,
        "official_source_pinned_and_audited": (
            official_source.get("repository") == "xiaowu0162/LongMemEval"
            and official_source.get("revision") == "9e0b455f4ef0e2ab8f2e582289761153549043fc"
            and official_source.get("file") == "src/generation/run_generation.py"
            and official_source.get("raw_file_sha256_utf8") == "4f1eb3c69d7ad40f04065b9c0bc86f6582441018fc6ff751d162d66c95baf672"
        ),
        "v3_is_v1_plus_only_official_current_date": template_valid,
        "all_50_historical_bundles_verified_and_bound": len(prepared) == 50,
        "exactly_50_reader_calls_succeeded": all_success and len(ledger_rows) == 50,
        "no_memory_embedding_judge_or_hosted_calls": all_local_reader,
        "test_access_false": True,
        "fullcontext_10_of_10_server_prompt_matches_preflight_fits_and_untruncated": fullcontext_valid,
        "all_reader_prompt_server_counts_match_preflight": all_server_prompt_counts_match,
        "d1_d2_d3_historical_artifacts_unchanged": source_unchanged,
        "final_contract_sidecar_verified": d3._verify_hash_sidecar(
            CONTRACT_PATH, CONTRACT_PATH.with_name(f"{CONTRACT_PATH.name}.sha256")
        ),
        "prediction_and_ledger_sha_freeze_valid": output_freeze_valid,
        "human_review_fields_null_and_no_automatic_labels": all(
            row["human_review"] == {"outcome": None, "failure_locus": None, "notes": None}
            and row["automatic_semantic_or_failure_labels"] == []
            for row in comparison_rows
        ),
    }
    gate["passed"] = (
        all(gate.values())
        and all_prediction_keys == expected_keys
        and len(comparison_rows) == 50
        and all(
            v3_by_key[(row["system"], row["question_id"])].get("context_bundle_sha256")
            == row["context_bundle_sha256"]
            for row in comparison_rows
        )
    )
    manifest = {
        "manifest_version": "mem1d4-reader-only-v1",
        "run_id": D4_RUN.name,
        "status": "COMPLETE" if gate["passed"] else "INCOMPLETE_OR_INVALID",
        "split": "DEV",
        "test_access": False,
        "question_ids": list(QUESTION_IDS),
        "systems": list(SYSTEMS),
        "reader_calls_expected": 50,
        "reader_calls_attempted": len(ledger_rows),
        "reader_calls_successful": successful_calls,
        "memory_system_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "hosted_api_calls": 0,
        "context_bundles_verified": len(prepared),
        "context_bundles_copied_or_rebuilt": False,
        "final_reader_contract_sha256": contract_sha,
        "final_reader_selection_gate": "FINAL_READER_SELECTED_BY_PROTOCOL_NOT_DEV_SCORE=YES",
        "template_raw_sha256": contract["template"]["raw_sha256"],
        "template_message_sha256": contract["template"]["canonical_message_template_sha256"],
        "fullcontext_preflight_and_server_usage": list(fullcontext_actual.values()),
        "all_prompt_token_counts_match": all_server_prompt_counts_match,
        "source_evidence_hashes_before": source_before,
        "source_evidence_hashes_after": source_after,
        "source_evidence_unchanged": source_unchanged,
        "configuration_sha256": config["configuration_sha256"],
        "configuration_file_sha256": config_sha,
        "predictions_sha256": predictions_sha,
        "call_ledger_sha256": ledger_sha,
        "comparison_sha256": comparison_sha,
        "review_sha256": review_sha,
        "semantic_review_status": "PENDING_HUMAN_REVIEW",
        "gate": gate,
    }
    _write_json(D4_RUN / "run_manifest.json", manifest)
    report = _report_text(config, manifest, comparison, fullcontext_preflight, source_before, source_after)
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
    return {
        "gate": "MEM1D4_FINAL_READER_CONTRACT_FROZEN=YES" if gate["passed"] else "MEM1D4_FINAL_READER_CONTRACT_FROZEN=NO",
        "reader_calls": len(ledger_rows),
        "successful_reader_calls": successful_calls,
        "bundles_verified": len(prepared),
        "fullcontext_fit_and_nontruncated": fullcontext_valid,
        "prompt_server_token_counts_match": all_server_prompt_counts_match,
        "historical_evidence_unchanged": source_unchanged,
        "final_reader_contract_sha256": contract_sha,
        "predictions_sha256": predictions_sha,
        "call_ledger_sha256": ledger_sha,
    }


def main() -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gate"] == "MEM1D4_FINAL_READER_CONTRACT_FROZEN=YES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
