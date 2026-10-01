"""Frozen-10 raw/readable context comparison under one diagnostic-only reader prompt."""

from __future__ import annotations

import hashlib
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
from tools.research.memory.run_mem3b1_readable_context_frozen10_v1 import (
    QUESTION_IDS,
    SELECTION_SHA256,
    _load_gold,
    _load_source,
    _question_parts,
    _render_items,
)


OUTPUT_RUN = ROOT / "runs/memory/mem3/mem3b1-v4-readable-projection-frozen10-diagnostic-v1"
READABLE_RUN = ROOT / "runs/memory/mem3/mem3b1-readable-context-frozen10-diagnostic-v1"
SYSTEM_PROMPT_PATH = ROOT / "docs/research/memory/reader_answer_contract_v4_diagnostic.txt"


def _freeze_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return common._freeze_artifact(path, payload)


def _mean(rows: list[dict[str, Any]], path: str, metric: str | None = None) -> float | None:
    values = [row[path][metric] if metric else row[path] for row in rows]
    values = [value for value in values if isinstance(value, (int, float))]
    return sum(values) / len(values) if values else None


def run_once() -> dict[str, Any]:
    if OUTPUT_RUN.exists():
        raise RuntimeError(f"append-never output already exists: {OUTPUT_RUN}")
    contexts, baselines, source_hashes = _load_source()
    gold = _load_gold()
    _, v3_contract_sha, _, user_template = final_reader_contract.load_final_reader_contract()
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    system_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
    source_hashes["selection_sha256"] = SELECTION_SHA256
    source_hashes["reader_v4_prompt_sha256"] = system_sha
    structured_by_qid = {}
    for qid in QUESTION_IDS:
        path = READABLE_RUN / f"context_projection_{qid}.json"
        common._verify_sidecar(path)
        projection = common._read_json(path)
        if (
            projection.get("question_id") != qid
            or projection.get("parent_context_bundle_sha256") != contexts[qid]["context_bundle_sha256"]
        ):
            raise RuntimeError(f"readable projection is detached from frozen source bundle: {qid}")
        structured_by_qid[qid] = projection["items"]
        source_hashes[f"readable_projection_{qid}_sha256"] = common._sha_file(path)

    plans = []
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        runtime = common._runtime_preflight(client)
        for qid in QUESTION_IDS:
            source = contexts[qid]
            question_date, question = _question_parts(source)
            raw_items = [dict(item) for item in source["context_bundle"]["items"]]
            readable_items = structured_by_qid[qid]
            if (
                [item["memory_id"] for item in raw_items]
                != [item["memory_id"] for item in readable_items]
                or len(raw_items) != len(readable_items)
            ):
                raise RuntimeError(f"readable context changed retrieved IDs/order: {qid}")
            for mode, items in (("raw_json", raw_items), ("readable_proposition", readable_items)):
                serialized = common._serialize_context(items)
                messages = final_reader_contract.build_reader_messages(
                    question,
                    question_date,
                    serialized,
                    system_template=system_prompt,
                    user_template=user_template,
                )
                if (
                    messages[0]["content"] != system_prompt
                    or messages[1]["content"].split("Current Date: ", 1)[-1]
                    != source["reader_messages"][1]["content"].split("Current Date: ", 1)[-1]
                ):
                    raise RuntimeError(f"unintended v4 prompt difference for {qid}/{mode}")
                projection = {
                    "schema_version": 1,
                    "question_id": qid,
                    "mode": mode,
                    "system_prompt_sha256": system_sha,
                    "source_context_bundle_sha256": source["context_bundle_sha256"],
                    "items": items,
                    "serialized_context": serialized,
                }
                projection_bytes = common._canonical(projection)
                projection_sha = common._sha_bytes(projection_bytes)
                context_tokens = common._preflight_token_count(client, serialized)
                rendered = client.post(
                    "http://127.0.0.1:8081/apply-template",
                    json={"messages": messages, "add_generation_prompt": True,
                          "chat_template_kwargs": {"enable_thinking": False}},
                )
                rendered.raise_for_status()
                prompt = rendered.json().get("prompt")
                if not isinstance(prompt, str):
                    raise RuntimeError(f"no template-rendered prompt for {qid}/{mode}")
                prompt_tokens = common._preflight_token_count(client, prompt)
                if prompt_tokens + 256 >= 131072:
                    raise RuntimeError(f"prompt exceeds reader context limit: {qid}/{mode}")
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
                    "question_type": gold[qid]["question_type"],
                    "question": question,
                    "gold": gold[qid]["answer"],
                    "baseline_v3": baselines[qid]["predicted"],
                    "mode": mode,
                    "items": items,
                    "projection_bytes": projection_bytes,
                    "projection_sha256": projection_sha,
                    "request_bytes": request_bytes,
                    "request_sha256": common._sha_bytes(request_bytes),
                    "reader_context_tokens": context_tokens,
                    "source_context_tokens_v3": source["context_bundle"]["context_reader_tokens"],
                    "reader_prompt_tokens": prompt_tokens,
                    "source_prompt_tokens_v3": source["reader_prompt_tokens_preflight"],
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
        "status": "TWENTY_LOCAL_POSTS_RESERVED",
        "question_ids": list(QUESTION_IDS),
        "systems": ["FlatProp raw JSON context", "FlatProp readable proposition context"],
        "reader_model": common.MODEL,
        "reader_role": "answer model",
        "memory_system": "FlatProp (same frozen retrieval records; no temporal mutation)",
        "embedding_model": "NONE (reused frozen retrieval bundles)",
        "judge_model": "NONE",
        "reader_contract_v3_sha256_reference_only": v3_contract_sha,
        "diagnostic_reader_system_prompt_sha256": system_sha,
        "hosted_calls": 0,
        "local_posts_authorized": 20,
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
            qid, mode = plan["question_id"], plan["mode"]
            safe = f"{qid}_{mode}"
            start = time.perf_counter()
            try:
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
                finish_reason = choice.get("finish_reason")
                prompt_tokens_server = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                status = (
                    "OK" if isinstance(predicted, str) and finish_reason == "stop"
                    and prompt_tokens_server == plan["reader_prompt_tokens"] else "INFRA_FAILURE"
                )
                predicted = predicted.strip() if isinstance(predicted, str) else None
                metrics = common._answer_metrics(predicted, plan["gold"]) if predicted is not None else None
                error_type = None
            except httpx.HTTPError as exc:
                predicted = None
                metrics = None
                status = "INFRA_FAILURE"
                finish_reason = None
                prompt_tokens_server = None
                completion_tokens = None
                response_sha = None
                error_type = type(exc).__name__
            row = {
                "question_id": qid,
                "question_type": plan["question_type"],
                "mode": mode,
                "question": plan["question"],
                "gold": plan["gold"],
                "reader_v4_prediction": predicted,
                "reader_v4_metrics": metrics,
                "baseline_v3_prediction_coordinate": plan["baseline_v3"],
                "context_reader_tokens_v3_raw": plan["source_context_tokens_v3"],
                "context_reader_tokens_v4": plan["reader_context_tokens"],
                "reader_prompt_tokens_v3_raw": plan["source_prompt_tokens_v3"],
                "reader_prompt_tokens_v4": plan["reader_prompt_tokens"],
                "reader_prompt_tokens_server": prompt_tokens_server,
                "prompt_tokens_match": prompt_tokens_server == plan["reader_prompt_tokens"],
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "reader_latency_ms": (time.perf_counter() - start) * 1000,
                "quality_status": status,
                "infra_error_type": error_type,
                "retrieval_unchanged": True,
                "request_sha256": plan["request_sha256"],
                "response_sha256": response_sha,
            }
            _freeze_json(OUTPUT_RUN / f"prediction_{safe}.json", row)
            rows.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case_mode = {(row["question_id"], row["mode"]): row for row in rows}
    for row in rows:
        if row["quality_status"] == "OK":
            grouped[row["mode"]].append(row)
    paired = []
    for qid in QUESTION_IDS:
        raw = by_case_mode[(qid, "raw_json")]
        readable = by_case_mode[(qid, "readable_proposition")]
        paired.append({
            "question_id": qid,
            "raw_f1": raw["reader_v4_metrics"]["f1"] if raw["reader_v4_metrics"] else None,
            "readable_f1": readable["reader_v4_metrics"]["f1"] if readable["reader_v4_metrics"] else None,
            "delta_f1": (
                readable["reader_v4_metrics"]["f1"] - raw["reader_v4_metrics"]["f1"]
                if raw["reader_v4_metrics"] and readable["reader_v4_metrics"] else None
            ),
        })
    paired_values = [row["delta_f1"] for row in paired if isinstance(row["delta_f1"], (int, float))]
    summary = {
        "n_questions": len(QUESTION_IDS),
        "n_local_reader_posts": len(rows),
        "n_infra_failures": sum(row["quality_status"] != "OK" for row in rows),
        "mean_f1_by_mode": {
            mode: _mean(group, "reader_v4_metrics", "f1") for mode, group in grouped.items()
        },
        "mean_normalized_em_by_mode": {
            mode: _mean(group, "reader_v4_metrics", "normalized_exact_match") for mode, group in grouped.items()
        },
        "mean_paired_delta_f1_readable_minus_raw": sum(paired_values) / len(paired_values) if paired_values else None,
        "mean_context_reader_tokens_by_mode": {
            mode: _mean(group, "context_reader_tokens_v4") for mode, group in grouped.items()
        },
        "mean_reader_prompt_tokens_by_mode": {
            mode: _mean(group, "reader_prompt_tokens_v4") for mode, group in grouped.items()
        },
        "paired_case_deltas": paired,
        "by_question_type": {
            category: {
                mode: {
                    "n": len(group),
                    "mean_f1": _mean(group, "reader_v4_metrics", "f1"),
                    "mean_normalized_em": _mean(group, "reader_v4_metrics", "normalized_exact_match"),
                }
                for mode, group in sorted(
                    {
                        mode: [row for row in grouped.get(mode, []) if row["question_type"] == category]
                        for mode in ("raw_json", "readable_proposition")
                    }.items()
                )
                if group
            }
            for category in sorted({row["question_type"] for row in rows})
        },
        "statistical_claim": "none; frozen-10 DEV diagnostic with a diagnostic-only reader contract",
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
        "question_ids": list(QUESTION_IDS),
        "reader_posts": len(rows),
        "hosted_calls": 0,
        "embedding_calls": 0,
        "judge_calls": 0,
        "retries": 0,
        "reader_system_prompt_sha256": system_sha,
        "predictions_sha256": predictions_sha,
        "diagnostic_summary_sha256": summary_sha,
        "reservation_sha256": reservation_sha,
        "claim_boundary": "frozen-10 public DEV diagnostic; diagnostic-only reader v4; no revision admission; no benchmark ranking claim",
    })
    return {"summary": summary, "manifest_sha256": manifest_sha}


if __name__ == "__main__":
    print(json.dumps(run_once(), ensure_ascii=False, sort_keys=True, indent=2))
