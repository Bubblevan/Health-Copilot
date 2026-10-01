"""Paired frozen-10 diagnostic for deterministic readable memory projection."""

from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.research.memory import final_reader_contract
from tools.research.memory import run_mem3b1_instagram_snapshot_diagnostic_v1 as common


ROOT = common.ROOT
SOURCE_RUN = common.SOURCE_RUN
OUTPUT_RUN = ROOT / "runs/memory/mem3/mem3b1-readable-context-frozen10-diagnostic-v1"
SELECTION_PATH = ROOT / "docs/research/memory/main_smoke_10_manifest.json"
DATASET_MANIFEST_PATH = ROOT / "docs/research/memory/dataset_manifest.json"
QUESTION_IDS = (
    "1cea1afa",
    "1c549ce4",
    "778164c6",
    "fca70973",
    "a82c026e",
    "gpt4_e061b84g",
    "gpt4_f420262c",
    "8550ddae",
    "06878be2",
    "c4ea545c",
)
SELECTION_SHA256 = "5a38ff79d79be6a9db531227d63d6dc22dc619d11d2c701b2b4cc0295f04c911"


def _question_parts(context_row: dict[str, Any]) -> tuple[str, str]:
    message = context_row["reader_messages"][1]["content"]
    match = re.search(r"Current Date: (.+?)\nQuestion: (.+)$", message, re.DOTALL)
    if not match:
        raise RuntimeError(f"frozen reader message has no date/question contract: {context_row['question_id']}")
    return match.group(1), match.group(2)


def _render_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered = []
    for item in items:
        memory = json.loads(item["text"])
        value = memory.get("value")
        if not isinstance(value, dict):
            raise RuntimeError(f"memory item is not a typed proposition: {item.get('memory_id')}")
        proposition = value.get("proposition")
        observed_at = value.get("observed_at")
        authority = value.get("source_authority")
        if (
            not isinstance(proposition, str)
            or proposition != item.get("proposition_text")
            or not isinstance(observed_at, str)
            or authority not in {"user", "assistant", "mixed"}
            or memory.get("memory_id") != item.get("memory_id")
        ):
            raise RuntimeError(f"memory proposition/provenance failed source contract: {item.get('memory_id')}")
        projected = dict(item)
        projected["text"] = (
            "[Memory record]\n"
            f"Observed at: {observed_at}\n"
            f"Source authority: {authority}\n"
            f"Statement: {proposition}"
        )
        projected["reader_projection_format"] = "readable_proposition_v1"
        rendered.append(projected)
    return rendered


def _load_source() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    selection_sha = common._sha_file(SELECTION_PATH)
    if selection_sha != SELECTION_SHA256:
        raise RuntimeError("frozen-10 selection manifest does not match the recorded protocol hash")
    selection = common._read_json(SELECTION_PATH)
    if (
        selection.get("status") != "FROZEN_BEFORE_ANY_10_CASE_RESULTS"
        or selection.get("test_access") is not False
        or tuple(selection.get("question_ids", [])) != QUESTION_IDS
    ):
        raise RuntimeError("frozen-10 selection manifest changed or has test access")
    names = (
        "context_bundles.jsonl",
        "predictions.jsonl",
        "reader_call_ledger.jsonl",
    )
    source_hashes = {name: common._verify_sidecar(SOURCE_RUN / name) for name in names}
    source_hashes["run_manifest.json"] = common._verify_sidecar(SOURCE_RUN / "run_manifest.json")
    source_manifest = common._read_json(SOURCE_RUN / "run_manifest.json")
    if source_manifest.get("stage") != SOURCE_RUN.name or source_manifest.get("status") != "COMPLETE":
        raise RuntimeError("frozen FlatProp source run is incomplete")
    contexts = {
        row["question_id"]: row for row in common._read_jsonl(SOURCE_RUN / "context_bundles.jsonl")
    }
    predictions = {
        row["question_id"]: row for row in common._read_jsonl(SOURCE_RUN / "predictions.jsonl")
    }
    calls = {
        row["question_id"]: row for row in common._read_jsonl(SOURCE_RUN / "reader_call_ledger.jsonl")
    }
    if any(set(table) != set(QUESTION_IDS) for table in (contexts, predictions, calls)):
        raise RuntimeError("frozen context/prediction/call artifacts do not contain the exact ten IDs")
    for question_id in QUESTION_IDS:
        if (
            calls[question_id].get("success") is not True
            or calls[question_id].get("hosted_call") is not False
            or predictions[question_id].get("quality_status") != "OK"
        ):
            raise RuntimeError(f"frozen baseline is not a valid local-reader result: {question_id}")
    source_hashes["selection_manifest.json"] = selection_sha
    return contexts, predictions, source_hashes


def _load_gold() -> dict[str, dict[str, Any]]:
    dataset_manifest = common._read_json(DATASET_MANIFEST_PATH)
    entry = next(row for row in dataset_manifest["datasets"] if row.get("dataset_id") == "longmemeval_s")
    dataset_path = next((path for path in common.DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise RuntimeError("manifest-pinned LongMemEval-S dataset is unavailable")
    if dataset_path.stat().st_size != entry.get("expected_size_bytes"):
        raise RuntimeError("LongMemEval-S size does not match the dataset manifest")
    dataset_sha = common._sha_file(dataset_path)
    if dataset_sha != entry.get("expected_sha256"):
        raise RuntimeError("LongMemEval-S SHA does not match the dataset manifest")
    gold = {}
    for row in common._iter_json_array(dataset_path):
        if row.get("question_id") in QUESTION_IDS:
            gold[row["question_id"]] = {
                "answer": row.get("answer"),
                "question_type": row.get("question_type"),
                "answer_session_ids": row.get("answer_session_ids", []),
            }
    if set(gold) != set(QUESTION_IDS):
        raise RuntimeError("LongMemEval gold join does not match the frozen-10 IDs")
    return gold


def _freeze_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return common._freeze_artifact(path, payload)


def run_once() -> dict[str, Any]:
    if OUTPUT_RUN.exists():
        raise RuntimeError(f"append-never output already exists: {OUTPUT_RUN}")
    contexts, baselines, source_hashes = _load_source()
    gold = _load_gold()
    contract, contract_sha, system_template, user_template = final_reader_contract.load_final_reader_contract()
    plans = []

    with httpx.Client(timeout=180.0, trust_env=False) as client:
        runtime = common._runtime_preflight(client)
        for question_id in QUESTION_IDS:
            source = contexts[question_id]
            if source.get("shared_reader_contract_sha256") != contract_sha:
                raise RuntimeError(f"shared reader contract mismatch for {question_id}")
            original_bundle = source["context_bundle"]
            projected_items = _render_items(original_bundle["items"])
            serialized_context = common._serialize_context(projected_items)
            question_date, question = _question_parts(source)
            messages = final_reader_contract.build_reader_messages(
                question,
                question_date,
                serialized_context,
                system_template=system_template,
                user_template=user_template,
            )
            if (
                messages[0]["content"] != source["reader_messages"][0]["content"]
                or messages[1]["content"].split("Current Date: ", 1)[-1]
                != source["reader_messages"][1]["content"].split("Current Date: ", 1)[-1]
                or [item["memory_id"] for item in projected_items]
                != [item["memory_id"] for item in original_bundle["items"]]
            ):
                raise RuntimeError(f"projection changed reader contract/query/retrieved records: {question_id}")
            projection = {
                "schema_version": 1,
                "question_id": question_id,
                "parent_context_bundle_sha256": source["context_bundle_sha256"],
                "projection_policy": "deterministic readable proposition with observed time and source authority; no record added, removed, or reordered",
                "projection_format": "readable_proposition_v1",
                "items": projected_items,
                "serialized_context": serialized_context,
            }
            projection_bytes = common._canonical(projection)
            projection_sha = common._sha_bytes(projection_bytes)
            reader_context_tokens = common._preflight_token_count(client, serialized_context)
            rendered = client.post(
                "http://127.0.0.1:8081/apply-template",
                json={
                    "messages": messages,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            rendered.raise_for_status()
            rendered_prompt = rendered.json().get("prompt")
            if not isinstance(rendered_prompt, str):
                raise RuntimeError(f"reader template render returned no prompt for {question_id}")
            prompt_tokens = common._preflight_token_count(client, rendered_prompt)
            if prompt_tokens + 256 >= 131072:
                raise RuntimeError(f"frozen reader context budget exceeded: {question_id}")
            request = {
                "model": common.MODEL,
                "messages": messages,
                "temperature": 0,
                "seed": 42,
                "max_tokens": 256,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_bytes = common._canonical(request)
            plans.append(
                {
                    "question_id": question_id,
                    "question_type": gold[question_id]["question_type"],
                    "question": question,
                    "gold": gold[question_id]["answer"],
                    "baseline": baselines[question_id]["predicted"],
                    "baseline_metrics": common._answer_metrics(
                        baselines[question_id]["predicted"], gold[question_id]["answer"]
                    ),
                    "source_context_tokens": original_bundle["context_reader_tokens"],
                    "source_prompt_tokens": source["reader_prompt_tokens_preflight"],
                    "reader_context_tokens": reader_context_tokens,
                    "reader_prompt_tokens": prompt_tokens,
                    "context_bundle_sha256": projection_sha,
                    "projection_bytes": projection_bytes,
                    "projection_sha256": projection_sha,
                    "request_bytes": request_bytes,
                    "request_sha256": common._sha_bytes(request_bytes),
                }
            )

    OUTPUT_RUN.mkdir(parents=True, exist_ok=False)
    request_hashes = {}
    projection_hashes = {}
    for plan in plans:
        qid = plan["question_id"]
        projection_hashes[qid] = common._freeze_artifact(
            OUTPUT_RUN / f"context_projection_{qid}.json", plan["projection_bytes"]
        )
        request_hashes[qid] = common._freeze_artifact(
            OUTPUT_RUN / f"reader_request_{qid}.json", plan["request_bytes"] + b"\n"
        )
    reservation = {
        "run_id": OUTPUT_RUN.name,
        "status": "TEN_LOCAL_POSTS_RESERVED",
        "question_ids": list(QUESTION_IDS),
        "reader_model": common.MODEL,
        "reader_role": "answer model",
        "memory_system": "FlatProp + readable proposition context projection",
        "embedding_model": "NONE (reused frozen retrieval output)",
        "judge_model": "NONE",
        "hosted_api_calls": 0,
        "local_answer_posts_authorized": 10,
        "retries": 0,
        "request_sha256_by_question": {plan["question_id"]: plan["request_sha256"] for plan in plans},
        "request_file_sha256_by_question": request_hashes,
        "context_projection_sha256_by_question": projection_hashes,
        "reader_contract_sha256": contract_sha,
        "source_hashes": source_hashes,
        "runtime_preflight": runtime,
    }
    reservation_sha = _freeze_json(OUTPUT_RUN / "run_reservation.json", reservation)

    rows = []
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        for plan in plans:
            qid = plan["question_id"]
            response_path = OUTPUT_RUN / f"reader_response_{qid}.json"
            call_path = OUTPUT_RUN / f"reader_call_{qid}.json"
            start = time.perf_counter()
            try:
                response = client.post(
                    f"{common.ENDPOINT}/chat/completions",
                    content=plan["request_bytes"],
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                response_bytes = response.content
                response_sha = common._freeze_artifact(response_path, response_bytes)
                payload = json.loads(response_bytes)
                choice = payload["choices"][0]
                predicted = choice.get("message", {}).get("content")
                usage = payload.get("usage", {})
                finish_reason = choice.get("finish_reason")
                prompt_tokens_server = usage.get("prompt_tokens")
                status = (
                    "OK"
                    if isinstance(predicted, str)
                    and finish_reason == "stop"
                    and prompt_tokens_server == plan["reader_prompt_tokens"]
                    else "INFRA_FAILURE"
                )
                predicted = predicted.strip() if isinstance(predicted, str) else None
                metrics = common._answer_metrics(predicted, plan["gold"]) if predicted is not None else None
                error_type = None
                completion_tokens = usage.get("completion_tokens")
            except httpx.HTTPError as exc:
                predicted = None
                metrics = None
                status = "INFRA_FAILURE"
                finish_reason = None
                prompt_tokens_server = None
                completion_tokens = None
                response_sha = None
                error_type = type(exc).__name__
            latency_ms = (time.perf_counter() - start) * 1000
            row = {
                "question_id": qid,
                "question_type": plan["question_type"],
                "question": plan["question"],
                "gold": plan["gold"],
                "baseline_prediction": plan["baseline"],
                "readable_projection_prediction": predicted,
                "baseline_metrics": plan["baseline_metrics"],
                "readable_projection_metrics": metrics,
                "delta_f1": metrics["f1"] - plan["baseline_metrics"]["f1"] if metrics else None,
                "context_bundle_sha256": plan["context_bundle_sha256"],
                "context_reader_tokens_before": plan["source_context_tokens"],
                "context_reader_tokens_after": plan["reader_context_tokens"],
                "reader_prompt_tokens_before": plan["source_prompt_tokens"],
                "reader_prompt_tokens_after": plan["reader_prompt_tokens"],
                "reader_prompt_tokens_server": prompt_tokens_server,
                "prompt_tokens_match": prompt_tokens_server == plan["reader_prompt_tokens"],
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "reader_latency_ms": latency_ms,
                "retrieval_unchanged": True,
                "quality_status": status,
                "infra_error_type": error_type,
                "request_sha256": plan["request_sha256"],
                "response_sha256": response_sha,
            }
            _freeze_json(call_path, {"question_id": qid, "role": "reader_answer", **runtime, **row})
            _freeze_json(OUTPUT_RUN / f"prediction_{qid}.json", row)
            rows.append(row)

    successful = [row for row in rows if row["quality_status"] == "OK"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[row["question_type"]].append(row)

    def means(group: list[dict[str, Any]], key: str, metric: str | None = None) -> float | None:
        values = [row[key][metric] if metric else row[key] for row in group]
        values = [value for value in values if isinstance(value, (int, float))]
        return sum(values) / len(values) if values else None

    summary = {
        "n_reserved": len(plans),
        "n_quality_ok": len(successful),
        "n_infra_failure": len(rows) - len(successful),
        "baseline_mean_f1": means(successful, "baseline_metrics", "f1"),
        "readable_projection_mean_f1": means(successful, "readable_projection_metrics", "f1"),
        "mean_paired_delta_f1": means(successful, "delta_f1"),
        "baseline_mean_normalized_em": means(successful, "baseline_metrics", "normalized_exact_match"),
        "readable_projection_mean_normalized_em": means(successful, "readable_projection_metrics", "normalized_exact_match"),
        "mean_context_reader_tokens_before": means(successful, "context_reader_tokens_before"),
        "mean_context_reader_tokens_after": means(successful, "context_reader_tokens_after"),
        "mean_reader_prompt_tokens_before": means(successful, "reader_prompt_tokens_before"),
        "mean_reader_prompt_tokens_after": means(successful, "reader_prompt_tokens_after"),
        "by_question_type": {
            question_type: {
                "n": len(group),
                "baseline_mean_f1": means(group, "baseline_metrics", "f1"),
                "readable_projection_mean_f1": means(group, "readable_projection_metrics", "f1"),
                "mean_paired_delta_f1": means(group, "delta_f1"),
                "baseline_mean_normalized_em": means(group, "baseline_metrics", "normalized_exact_match"),
                "readable_projection_mean_normalized_em": means(group, "readable_projection_metrics", "normalized_exact_match"),
            }
            for question_type, group in sorted(grouped.items())
        },
        "statistical_claim": "none; ten-case development diagnostic, no confidence interval",
    }
    predictions_payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    predictions_sha = common._freeze_artifact(OUTPUT_RUN / "predictions.jsonl", predictions_payload)
    summary_sha = _freeze_json(OUTPUT_RUN / "diagnostic_summary.json", summary)
    manifest = {
        "run_id": OUTPUT_RUN.name,
        "status": "COMPLETE" if len(successful) == len(plans) else "PARTIAL_INFRA_FAILURE",
        "question_ids": list(QUESTION_IDS),
        "reader_posts": len(successful),
        "hosted_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "retries": 0,
        "source_hashes": source_hashes,
        "reader_contract_sha256": contract_sha,
        "reservation_sha256": reservation_sha,
        "predictions_sha256": predictions_sha,
        "diagnostic_summary_sha256": summary_sha,
        "claim_boundary": "frozen-10 public DEV diagnostic; readable serialization only; no revision admission or stale suppression; no benchmark ranking claim",
    }
    _freeze_json(OUTPUT_RUN / "run_manifest.json", manifest)
    return {"summary": summary, "predictions": rows, "manifest": manifest}


if __name__ == "__main__":
    print(json.dumps(run_once(), ensure_ascii=False, sort_keys=True, indent=2))
