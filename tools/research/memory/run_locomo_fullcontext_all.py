"""Run full LoCoMo with the pinned MemEval FullContext prompt and local Qwen."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from urllib.parse import urlparse

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = ROOT.parent / "external" / "memory"
LOCOMO_ROOT = EXTERNAL / "LoCoMo"
DATASET = LOCOMO_ROOT / "data" / "locomo10.json"
MEMEVAL_ROOT = EXTERNAL / "MemEval"
EXPECTED_LOCOMO_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
EXPECTED_DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
EXPECTED_MEMEVAL_COMMIT = "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4"
ENDPOINT = "http://127.0.0.1:8081/v1"
OUTPUT = ROOT / "runs" / "memory" / "locomo" / "fullcontext-locomo10-qwen-local-parallel2-v1"
MAX_TOKENS = 50
TEMPERATURE = 0.1
SEED = 42
MODEL_CONTEXT_LIMIT = 131072
READER_PARALLELISM = 2
MODEL_SLOT_CONTEXT_LIMIT = MODEL_CONTEXT_LIMIT // READER_PARALLELISM

PROMPT_TEMPLATE = (
    "You are answering questions about a conversation between two people.\n"
    "The conversation history is provided below. Answer based ONLY on information in the conversation.\n\n"
    "Rules:\n"
    "1. Give the SHORTEST answer possible - just the key fact (1-5 words max)\n"
    "2. Use EXACT words from the conversation when possible\n"
    "3. NO full sentences, NO explanations\n"
    "4. For dates, use the format from the conversation\n"
    "5. If the answer is truly not in the conversation, say 'None'\n\n"
    "CONVERSATION:\n{conversation}\n\nNow answer this question: {question}"
)


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _conversation_text(dialogues: list[dict]) -> str:
    lines = []
    current_timestamp = None
    for dialogue in dialogues:
        if dialogue["timestamp"] != current_timestamp:
            current_timestamp = dialogue["timestamp"]
            lines.append(f"\n--- {current_timestamp} ---\n")
        lines.append(f"[{dialogue['speaker']}]: {dialogue['text']}")
    return "\n".join(lines)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3)


def _tokenize_count(text: str) -> int:
    request = urllib.request.Request(
        ENDPOINT.removesuffix("/v1") + "/tokenize",
        data=json.dumps(
            {"content": text, "add_special": True, "parse_special": False}
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return len(json.loads(response.read())["tokens"])


def _run_question(client, model, question_id, conversation, qa_index, question, prompt, compute_f1):
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        answer = (response.choices[0].message.content or "").strip()
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        if prompt_tokens is None or prompt_tokens + MAX_TOKENS > MODEL_SLOT_CONTEXT_LIMIT:
            raise RuntimeError("Reader usage did not verify that the full prompt and answer reserve fit one slot.")
        category = int(question["category"])
        row = {
            "question_id": question_id,
            "sample_id": conversation["sample_id"],
            "source_qa_index": qa_index,
            "category": category,
            "category_name": question["category_name"],
            "predicted": answer,
            "ground_truth": question.get("answer", ""),
            "token_f1": compute_f1(answer, question.get("answer", "")),
            "prompt_tokens": prompt_tokens,
            "truncated": False,
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "latency_ms": latency_ms,
            "finish_reason": response.choices[0].finish_reason,
            "status": "COMPLETED_DIAGNOSTIC",
        }
        ledger = {
            "question_id": question_id,
            "system": "fullcontext",
            "role": "reader_answer",
            "provider": "local_qwen_loopback",
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": row["completion_tokens"],
            "latency_ms": latency_ms,
            "success": True,
            "hosted_calls": 0,
            "truncated": False,
        }
    except Exception as error:  # noqa: BLE001 - keep infrastructure failures distinct
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        row = {
            "question_id": question_id,
            "sample_id": conversation["sample_id"],
            "source_qa_index": qa_index,
            "category": int(question["category"]),
            "category_name": question["category_name"],
            "ground_truth": question.get("answer", ""),
            "status": "INFRA_FAILURE",
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
        }
        ledger = {
            "question_id": question_id,
            "system": "fullcontext",
            "role": "reader_answer",
            "provider": "local_qwen_loopback",
            "success": False,
            "error_type": type(error).__name__,
            "latency_ms": latency_ms,
            "hosted_calls": 0,
        }
    return row, ledger


def main() -> int:
    endpoint_host = urlparse(ENDPOINT).hostname
    if endpoint_host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Reader endpoint must be loopback")
    if _git(LOCOMO_ROOT, "rev-parse", "HEAD") != EXPECTED_LOCOMO_COMMIT:
        raise RuntimeError("LoCoMo checkout does not match the audited revision")
    if _git(MEMEVAL_ROOT, "rev-parse", "HEAD") != EXPECTED_MEMEVAL_COMMIT:
        raise RuntimeError("MemEval checkout does not match the pinned baseline")
    dataset_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise RuntimeError("LoCoMo dataset hash differs from the audited artifact")

    sys.path.insert(0, str(MEMEVAL_ROOT / "src"))
    from agents_memory.evaluation import compute_f1
    from agents_memory.locomo import CATEGORY_NAMES, extract_dialogues

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    conversations = {}
    context_preflight = {}
    for conversation in dataset:
        text = _conversation_text(extract_dialogues(conversation))
        conversations[conversation["sample_id"]] = text
        longest_question_index, longest_question = max(
            enumerate(conversation["qa"]), key=lambda item: len(item[1]["question"])
        )
        raw_prompt_tokens = _tokenize_count(
            PROMPT_TEMPLATE.format(
                conversation=text,
                question=longest_question["question"],
            )
        )
        if raw_prompt_tokens + MAX_TOKENS > MODEL_SLOT_CONTEXT_LIMIT:
            raise RuntimeError(
                f"FullContext prompt does not fit for {conversation['sample_id']}: "
                f"{raw_prompt_tokens}+{MAX_TOKENS}>{MODEL_SLOT_CONTEXT_LIMIT}"
            )
        context_preflight[conversation["sample_id"]] = {
            "representative_question_id": f"{conversation['sample_id']}:qa:{longest_question_index}",
            "raw_prompt_content_tokens": raw_prompt_tokens,
            "answer_budget_tokens": MAX_TOKENS,
            "model_context_limit": MODEL_SLOT_CONTEXT_LIMIT,
            "total_server_context": MODEL_CONTEXT_LIMIT,
            "truncated": False,
        }

    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    client = OpenAI(api_key="local-only-placeholder", base_url=ENDPOINT, timeout=900)
    models = client.models.list().data
    if not models:
        raise RuntimeError("Local reader returned no model id")
    model = models[0].id

    expected_ids = {
        f"{conversation['sample_id']}:qa:{index}"
        for conversation in dataset
        for index, _question in enumerate(conversation["qa"])
    }
    config = {
        "benchmark": "LoCoMo locomo10 full set",
        "dataset_sha256": dataset_hash,
        "locomo_revision": EXPECTED_LOCOMO_COMMIT,
        "memeval_revision": EXPECTED_MEMEVAL_COMMIT,
        "system": "MemEval FullContext local runtime adaptation",
        "reader_provider": "local_qwen_loopback",
        "reader_model": model,
        "endpoint": ENDPOINT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "enable_thinking": False,
        "judge": "NONE",
        "hosted_api": "NONE",
        "context_limit": MODEL_CONTEXT_LIMIT,
        "reader_parallelism": READER_PARALLELISM,
        "per_slot_context_limit": MODEL_SLOT_CONTEXT_LIMIT,
        "context_preflight": context_preflight,
        "question_count": len(expected_ids),
    }
    config_path = OUTPUT / "run_config.json"
    prediction_path = OUTPUT / "predictions.jsonl"
    ledger_path = OUTPUT / "call_ledger.jsonl"
    if OUTPUT.exists():
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError(f"Existing output has no matching frozen config: {OUTPUT}")
    else:
        OUTPUT.mkdir(parents=True)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    existing_rows = []
    if prediction_path.exists():
        existing_rows = [
            json.loads(line)
            for line in prediction_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    latest = {row["question_id"]: row for row in existing_rows}
    completed = {
        question_id
        for question_id, row in latest.items()
        if row.get("status") == "COMPLETED_DIAGNOSTIC"
    }
    if not completed.issubset(expected_ids):
        raise RuntimeError("Existing prediction artifact contains question ids outside LoCoMo")

    failures = sum(row.get("status") == "INFRA_FAILURE" for row in latest.values())
    started_all = time.perf_counter()
    progress_every = 25
    processed_this_run = 0

    def persist(future: Future) -> None:
        nonlocal failures, processed_this_run
        row, ledger = future.result()
        if row["status"] == "INFRA_FAILURE":
            failures += 1
        prediction_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        prediction_file.flush()
        ledger_file.write(json.dumps(ledger, ensure_ascii=False, sort_keys=True) + "\n")
        ledger_file.flush()
        latest[row["question_id"]] = row
        processed_this_run += 1
        if processed_this_run % progress_every == 0:
            valid = [item for item in latest.values() if item.get("status") == "COMPLETED_DIAGNOSTIC"]
            current_f1 = mean(item["token_f1"] for item in valid) if valid else None
            print(
                f"progress={len(completed) + processed_this_run}/{len(expected_ids)} "
                f"infra_failures={failures} running_token_f1="
                f"{current_f1:.4f}" if current_f1 is not None else
                f"progress={len(completed) + processed_this_run}/{len(expected_ids)} infra_failures={failures}",
                flush=True,
            )

    with prediction_path.open("a", encoding="utf-8", newline="\n") as prediction_file, ledger_path.open(
        "a", encoding="utf-8", newline="\n"
    ) as ledger_file, ThreadPoolExecutor(max_workers=READER_PARALLELISM) as pool:
        pending: list[Future] = []
        for conversation in dataset:
            text = conversations[conversation["sample_id"]]
            for qa_index, question in enumerate(conversation["qa"]):
                question_id = f"{conversation['sample_id']}:qa:{qa_index}"
                if question_id in completed:
                    continue
                prompt = PROMPT_TEMPLATE.format(
                    conversation=text,
                    question=question["question"],
                )
                question_with_category = {
                    **question,
                    "category_name": CATEGORY_NAMES.get(int(question["category"]), "unknown"),
                }
                pending.append(
                    pool.submit(
                        _run_question,
                        client,
                        model,
                        question_id,
                        conversation,
                        qa_index,
                        question_with_category,
                        prompt,
                        compute_f1,
                    )
                )
                if len(pending) >= READER_PARALLELISM:
                    persist(pending.pop(0))
        for future in pending:
            persist(future)

    rows = list(latest.values())
    successful = [row for row in rows if row.get("status") == "COMPLETED_DIAGNOSTIC"]
    per_category: dict[str, list[float]] = defaultdict(list)
    per_conversation: dict[str, list[float]] = defaultdict(list)
    for row in successful:
        per_category[row["category_name"]].append(float(row["token_f1"]))
        per_conversation[row["sample_id"]].append(float(row["token_f1"]))
    prediction_bytes = prediction_path.read_bytes()
    prompt_tokens = [int(row["prompt_tokens"]) for row in successful if row.get("prompt_tokens") is not None]
    latencies = [float(row["latency_ms"]) for row in successful if row.get("latency_ms") is not None]
    summary = {
        "run_id": OUTPUT.name,
        "run_dir": str(OUTPUT),
        "status": "COMPLETED_DIAGNOSTIC" if len(successful) == len(expected_ids) else "INCOMPLETE_OR_INFRA_FAILURE",
        "interpretation": "Full 10-conversation LoCoMo local FullContext baseline; no memory architecture ranking claim.",
        "dataset": {
            "repository": "https://github.com/snap-research/locomo",
            "revision": EXPECTED_LOCOMO_COMMIT,
            "file_sha256": dataset_hash,
            "license": "CC BY-NC 4.0; non-commercial use only",
            "conversations": len(dataset),
            "questions_expected": len(expected_ids),
            "questions_scored": len(successful),
        },
        "system": "MemEval FullContext, local runtime adaptation",
        "memeval_revision": EXPECTED_MEMEVAL_COMMIT,
        "reader": {"provider": "local_qwen_loopback", "model": model, "endpoint": ENDPOINT},
        "reader_parallelism": READER_PARALLELISM,
        "embedding": "NONE",
        "judge": "NONE",
        "hosted_calls": 0,
        "infra_failures": len(expected_ids) - len(successful),
        "mean_token_f1": mean(row["token_f1"] for row in successful) if successful else None,
        "per_category_token_f1": {
            name: {"n": len(values), "mean": mean(values)}
            for name, values in sorted(per_category.items())
        },
        "per_conversation_token_f1": {
            name: {"n": len(values), "mean": mean(values)}
            for name, values in sorted(per_conversation.items())
        },
        "prompt_tokens": {
            "mean": mean(prompt_tokens) if prompt_tokens else None,
            "max": max(prompt_tokens) if prompt_tokens else None,
            "model_context_limit": MODEL_CONTEXT_LIMIT,
            "per_slot_context_limit": MODEL_SLOT_CONTEXT_LIMIT,
            "observed_within_limit": max(prompt_tokens) <= MODEL_CONTEXT_LIMIT if prompt_tokens else None,
        },
        "reader_latency_ms": {
            "mean": mean(latencies) if latencies else None,
            "p95": _p95(latencies),
            "total_wall_seconds_this_invocation": round(time.perf_counter() - started_all, 3),
        },
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "prediction_frozen": len(successful) == len(expected_ids),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "COMPLETED_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())
