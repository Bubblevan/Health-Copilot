"""Two-case reader-format counterfactual; never substitutes for frozen reader v3."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.research.memory import final_reader_contract
from tools.research.memory import run_mem3b1_instagram_snapshot_diagnostic_v1 as common


SOURCE_RUN = common.SOURCE_RUN
OUTPUT_RUN = ROOT / "runs/memory/mem3/mem3b1-concise-reader-cross-diagnostic-v1"
SYSTEM_PROMPT_PATH = ROOT / "docs/research/memory/reader_answer_contract_v4_diagnostic.txt"
DATASET_MANIFEST_PATH = ROOT / "docs/research/memory/dataset_manifest.json"
CASES = {
    "1cea1afa": {
        "kind": "current_snapshot",
        "structured_run": ROOT / "runs/memory/mem3/mem3b1-instagram-structured-current-diagnostic-v1",
    },
    "c4ea545c": {
        "kind": "change_chain",
        "structured_run": ROOT / "runs/memory/mem3/mem3b1-gym-change-chain-diagnostic-v1",
    },
}


def _sha(path: Path) -> str:
    return common._sha_file(path)


def _freeze_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return common._freeze_artifact(path, payload)


def _load_source() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    for name in ("context_bundles.jsonl", "predictions.jsonl", "reader_call_ledger.jsonl"):
        common._verify_sidecar(SOURCE_RUN / name)
    source_manifest = common._read_json(SOURCE_RUN / "run_manifest.json")
    if source_manifest.get("status") != "COMPLETE":
        raise RuntimeError("frozen baseline run is incomplete")
    contexts = {
        row["question_id"]: row for row in common._read_jsonl(SOURCE_RUN / "context_bundles.jsonl")
    }
    predictions = {
        row["question_id"]: row for row in common._read_jsonl(SOURCE_RUN / "predictions.jsonl")
    }
    source_hashes = {
        name: _sha(SOURCE_RUN / name)
        for name in ("context_bundles.jsonl", "predictions.jsonl", "reader_call_ledger.jsonl")
    }
    source_hashes["run_manifest.json"] = _sha(SOURCE_RUN / "run_manifest.json")
    for qid, config in CASES.items():
        if qid not in contexts or qid not in predictions:
            raise RuntimeError(f"frozen source case is absent: {qid}")
        projection_path = config["structured_run"] / "context_projection.json"
        common._verify_sidecar(projection_path)
        projection = common._read_json(projection_path)
        if (
            projection.get("question_id") != qid
            or projection.get("parent_context_bundle_sha256") != contexts[qid]["context_bundle_sha256"]
        ):
            raise RuntimeError(f"structured projection is not bound to its frozen source bundle: {qid}")
        source_hashes[f"{qid}_structured_projection.json"] = _sha(projection_path)
    return contexts, predictions, source_hashes


def _load_gold() -> dict[str, str]:
    manifest = common._read_json(DATASET_MANIFEST_PATH)
    entry = next(row for row in manifest["datasets"] if row.get("dataset_id") == "longmemeval_s")
    dataset = next((path for path in common.DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset is None or dataset.stat().st_size != entry["expected_size_bytes"]:
        raise RuntimeError("manifest-pinned LongMemEval-S file is unavailable or changed")
    if _sha(dataset) != entry["expected_sha256"]:
        raise RuntimeError("LongMemEval-S hash differs from its frozen manifest")
    result = {}
    for row in common._iter_json_array(dataset):
        if row.get("question_id") in CASES:
            result[row["question_id"]] = row["answer"]
    if set(result) != set(CASES):
        raise RuntimeError("gold labels do not join to the two frozen DEV IDs")
    return result


def _question_parts(context_row: dict[str, Any]) -> tuple[str, str]:
    user = context_row["reader_messages"][1]["content"]
    match = re.search(r"Current Date: (.+?)\nQuestion: (.+)$", user, re.DOTALL)
    if not match:
        raise RuntimeError("frozen date/question reader contract is malformed")
    return match.group(1), match.group(2)


def run_once() -> dict[str, Any]:
    if OUTPUT_RUN.exists():
        raise RuntimeError(f"append-never output already exists: {OUTPUT_RUN}")
    contexts, baselines, source_hashes = _load_source()
    gold = _load_gold()
    _, v3_contract_sha, _, user_template = final_reader_contract.load_final_reader_contract()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    system_prompt_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    plans = []

    with httpx.Client(timeout=180.0, trust_env=False) as client:
        runtime = common._runtime_preflight(client)
        for qid, config in CASES.items():
            source = contexts[qid]
            question_date, question = _question_parts(source)
            raw_items = [dict(item) for item in source["context_bundle"]["items"]]
            structured = common._read_json(config["structured_run"] / "context_projection.json")
            for mode, items in (("raw_json", raw_items), (config["kind"], structured["items"])):
                serialized_context = common._serialize_context(items)
                messages = final_reader_contract.build_reader_messages(
                    question,
                    question_date,
                    serialized_context,
                    system_template=system_prompt,
                    user_template=user_template,
                )
                if messages[1]["content"].split("Current Date: ", 1)[-1] != source["reader_messages"][1]["content"].split("Current Date: ", 1)[-1]:
                    raise RuntimeError(f"counterfactual reader changed the frozen question/date: {qid}")
                projection = {
                    "schema_version": 1,
                    "question_id": qid,
                    "mode": mode,
                    "system_prompt_sha256": system_prompt_sha,
                    "source_context_bundle_sha256": source["context_bundle_sha256"],
                    "items": items,
                    "serialized_context": serialized_context,
                }
                projection_bytes = common._canonical(projection)
                projection_sha = common._sha_bytes(projection_bytes)
                context_tokens = common._preflight_token_count(client, serialized_context)
                rendered = client.post(
                    "http://127.0.0.1:8081/apply-template",
                    json={"messages": messages, "add_generation_prompt": True,
                          "chat_template_kwargs": {"enable_thinking": False}},
                )
                rendered.raise_for_status()
                prompt = rendered.json().get("prompt")
                if not isinstance(prompt, str):
                    raise RuntimeError(f"no rendered prompt for {qid}/{mode}")
                prompt_tokens = common._preflight_token_count(client, prompt)
                if prompt_tokens + 256 >= 131072:
                    raise RuntimeError(f"reader context exceeds pinned maximum for {qid}/{mode}")
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
                plans.append({
                    "question_id": qid,
                    "mode": mode,
                    "question": question,
                    "gold": gold[qid],
                    "v3_baseline_prediction": baselines[qid]["predicted"],
                    "projection_bytes": projection_bytes,
                    "projection_sha256": projection_sha,
                    "request_bytes": request_bytes,
                    "request_sha256": common._sha_bytes(request_bytes),
                    "context_reader_tokens": context_tokens,
                    "reader_prompt_tokens": prompt_tokens,
                    "source_context_reader_tokens": source["context_bundle"]["context_reader_tokens"],
                    "source_reader_prompt_tokens": source["reader_prompt_tokens_preflight"],
                    "reader_messages": messages,
                })

    OUTPUT_RUN.mkdir(parents=True, exist_ok=False)
    projection_hashes = {}
    request_file_hashes = {}
    for plan in plans:
        key = f"{plan['question_id']}:{plan['mode']}"
        safe = f"{plan['question_id']}_{plan['mode']}"
        projection_hashes[key] = common._freeze_artifact(
            OUTPUT_RUN / f"context_{safe}.json", plan["projection_bytes"]
        )
        request_file_hashes[key] = common._freeze_artifact(
            OUTPUT_RUN / f"request_{safe}.json", plan["request_bytes"] + b"\n"
        )
    reservation_sha = _freeze_json(OUTPUT_RUN / "run_reservation.json", {
        "run_id": OUTPUT_RUN.name,
        "status": "FOUR_LOCAL_POSTS_RESERVED",
        "case_ids": list(CASES),
        "modes": ["raw_json", "structured_state_or_change"],
        "reader_model": common.MODEL,
        "reader_role": "answer model",
        "memory_system": "FlatProp; raw versus existing source-grounded state projection",
        "embedding_model": "NONE (reused frozen retrieval bundles)",
        "judge_model": "NONE",
        "reader_contract_v3_sha256_reference_only": v3_contract_sha,
        "reader_system_prompt_sha256": system_prompt_sha,
        "hosted_calls": 0,
        "local_posts_authorized": 4,
        "retries": 0,
        "projection_sha256_by_case_mode": projection_hashes,
        "request_sha256_by_case_mode": {f"{p['question_id']}:{p['mode']}": p["request_sha256"] for p in plans},
        "request_file_sha256_by_case_mode": request_file_hashes,
        "source_hashes": source_hashes,
        "runtime_preflight": runtime,
    })

    rows = []
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        for plan in plans:
            safe = f"{plan['question_id']}_{plan['mode']}"
            start = time.perf_counter()
            response = client.post(
                f"{common.ENDPOINT}/chat/completions",
                content=plan["request_bytes"],
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            response_bytes = response.content
            response_sha = common._freeze_artifact(OUTPUT_RUN / f"response_{safe}.json", response_bytes)
            payload = json.loads(response_bytes)
            choice = payload["choices"][0]
            predicted = choice.get("message", {}).get("content")
            usage = payload.get("usage", {})
            latency_ms = (time.perf_counter() - start) * 1000
            if not isinstance(predicted, str):
                raise RuntimeError(f"reader answer is missing for {safe}")
            predicted = predicted.strip()
            metrics = common._answer_metrics(predicted, plan["gold"])
            row = {
                "question_id": plan["question_id"],
                "mode": plan["mode"],
                "question": plan["question"],
                "gold": plan["gold"],
                "reader_v4_prediction": predicted,
                "reader_v4_metrics": metrics,
                "v3_baseline_prediction_coordinate": plan["v3_baseline_prediction"],
                "context_reader_tokens_v3_raw": plan["source_context_reader_tokens"],
                "context_reader_tokens_v4": plan["context_reader_tokens"],
                "reader_prompt_tokens_v3_raw": plan["source_reader_prompt_tokens"],
                "reader_prompt_tokens_v4": plan["reader_prompt_tokens"],
                "reader_prompt_tokens_server": usage.get("prompt_tokens"),
                "prompt_tokens_match": usage.get("prompt_tokens") == plan["reader_prompt_tokens"],
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens"),
                "reader_latency_ms": latency_ms,
                "quality_status": "OK" if choice.get("finish_reason") == "stop" and usage.get("prompt_tokens") == plan["reader_prompt_tokens"] else "INFRA_FAILURE",
                "retrieval_unchanged": True,
                "request_sha256": plan["request_sha256"],
                "response_sha256": response_sha,
            }
            _freeze_json(OUTPUT_RUN / f"prediction_{safe}.json", row)
            rows.append(row)

    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    summary = {
        "n": len(rows),
        "infra_failures": sum(row["quality_status"] != "OK" for row in rows),
        "mean_f1_by_mode": {
            mode: sum(row["reader_v4_metrics"]["f1"] for row in group) / len(group)
            for mode, group in by_mode.items()
        },
        "normalized_em_by_case_mode": {
            f"{row['question_id']}:{row['mode']}": row["reader_v4_metrics"]["normalized_exact_match"]
            for row in rows
        },
        "mean_reader_context_tokens_by_mode": {
            mode: sum(row["context_reader_tokens_v4"] for row in group) / len(group)
            for mode, group in by_mode.items()
        },
        "mean_reader_prompt_tokens_by_mode": {
            mode: sum(row["reader_prompt_tokens_v4"] for row in group) / len(group)
            for mode, group in by_mode.items()
        },
        "statistical_claim": "none; two-case diagnostic only",
    }
    predictions_sha = common._freeze_artifact(
        OUTPUT_RUN / "predictions.jsonl",
        b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in rows
        ),
    )
    summary_sha = _freeze_json(OUTPUT_RUN / "diagnostic_summary.json", summary)
    manifest_sha = _freeze_json(OUTPUT_RUN / "run_manifest.json", {
        "run_id": OUTPUT_RUN.name,
        "status": "COMPLETE" if all(row["quality_status"] == "OK" for row in rows) else "PARTIAL_INFRA_FAILURE",
        "case_ids": list(CASES),
        "reader_posts": len(rows),
        "hosted_calls": 0,
        "judge_calls": 0,
        "embedding_calls": 0,
        "retries": 0,
        "reader_system_prompt_sha256": system_prompt_sha,
        "prediction_sha256": predictions_sha,
        "summary_sha256": summary_sha,
        "reservation_sha256": reservation_sha,
        "claim_boundary": "diagnostic-only v4 answer formatting counterfactual; frozen reader v3 remains unchanged",
    })
    return {"summary": summary, "predictions": rows, "manifest_sha256": manifest_sha}


if __name__ == "__main__":
    print(json.dumps(run_once(), ensure_ascii=False, sort_keys=True, indent=2))
