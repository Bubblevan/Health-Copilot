"""Run a category-balanced five-question LoCoMo FullContext smoke locally."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
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
RUN_ID = f"fullcontext-conv26-five-category-smoke-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
OUTPUT = ROOT / "runs" / "memory" / "locomo" / RUN_ID


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def _select_questions(conversation: dict) -> list[tuple[int, dict]]:
    selected = []
    for category in sorted({int(row["category"]) for row in conversation["qa"]}):
        index, row = next(
            (i, item)
            for i, item in enumerate(conversation["qa"])
            if int(item["category"]) == category
        )
        selected.append((index, row))
    return selected


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite {OUTPUT}")
    endpoint_host = urlparse(ENDPOINT).hostname
    if endpoint_host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Reader endpoint must be loopback")
    if _git(LOCOMO_ROOT, "rev-parse", "HEAD") != EXPECTED_LOCOMO_COMMIT:
        raise RuntimeError("LoCoMo checkout does not match the audited revision")
    if hashlib.sha256(DATASET.read_bytes()).hexdigest() != EXPECTED_DATASET_SHA256:
        raise RuntimeError("LoCoMo dataset hash differs from the audited artifact")
    if _git(MEMEVAL_ROOT, "rev-parse", "HEAD") != EXPECTED_MEMEVAL_COMMIT:
        raise RuntimeError("MemEval checkout does not match the pinned baseline")

    sys.path.insert(0, str(MEMEVAL_ROOT / "src"))
    from agents_memory.evaluation import compute_f1
    from agents_memory.locomo import extract_dialogues

    client = OpenAI(api_key="local-only-placeholder", base_url=ENDPOINT, timeout=900)
    models = client.models.list().data
    if not models:
        raise RuntimeError("Local reader returned no model id")
    model = models[0].id

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    conversation = dataset[0]
    selected = _select_questions(conversation)
    dialogues = extract_dialogues(conversation)
    lines = []
    current_timestamp = None
    for dialogue in dialogues:
        if dialogue["timestamp"] != current_timestamp:
            current_timestamp = dialogue["timestamp"]
            lines.append(f"\n--- {current_timestamp} ---\n")
        lines.append(f"[{dialogue['speaker']}]: {dialogue['text']}")
    conversation_text = "\n".join(lines)
    prompt_template = (
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

    OUTPUT.mkdir(parents=True, exist_ok=False)
    ledger = []
    predictions = []
    for qa_index, question in selected:
        ground_truth = question.get("answer", "")
        prompt = prompt_template.format(
            conversation=conversation_text,
            question=question["question"],
        )
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1,
                seed=42,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            answer = (response.choices[0].message.content or "").strip()
            usage = response.usage
            row = {
                "question_id": f"{conversation['sample_id']}:qa:{qa_index}",
                "source_qa_index": qa_index,
                "category": int(question["category"]),
                "question": question["question"],
                "ground_truth": ground_truth,
                "predicted": answer,
                "token_f1": compute_f1(answer, ground_truth),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "latency_ms": elapsed_ms,
                "finish_reason": response.choices[0].finish_reason,
                "status": "COMPLETED_DIAGNOSTIC",
            }
            ledger.append({
                "question_id": row["question_id"],
                "system": "fullcontext",
                "provider": "local_qwen_loopback",
                "model": model,
                "success": True,
                "latency_ms": elapsed_ms,
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "hosted_calls": 0,
            })
        except Exception as error:  # noqa: BLE001 - preserve infra failure distinctly
            row = {
                "question_id": f"{conversation['sample_id']}:qa:{qa_index}",
                "source_qa_index": qa_index,
                "category": int(question["category"]),
                "question": question["question"],
                "ground_truth": ground_truth,
                "status": "INFRA_FAILURE",
                "error_type": type(error).__name__,
                "error": str(error),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            ledger.append({
                "question_id": row["question_id"],
                "system": "fullcontext",
                "provider": "local_qwen_loopback",
                "success": False,
                "error_type": row["error_type"],
                "hosted_calls": 0,
            })
        predictions.append(row)

    prediction_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in predictions
    ).encode("utf-8")
    ledger_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in ledger
    ).encode("utf-8")
    (OUTPUT / "predictions.jsonl").write_bytes(prediction_bytes)
    (OUTPUT / "call_ledger.jsonl").write_bytes(ledger_bytes)
    valid = [row for row in predictions if row["status"] == "COMPLETED_DIAGNOSTIC"]
    summary = {
        "run_id": RUN_ID,
        "status": "COMPLETED_DIAGNOSTIC" if len(valid) == len(selected) else "INFRA_FAILURE",
        "interpretation": "Five-question LoCoMo smoke only; not a benchmark result or ranking claim.",
        "system": "MemEval FullContext, local runtime adaptation; no memory ingestion or retrieval.",
        "reader": {"provider": "local_qwen_loopback", "model": model, "endpoint": ENDPOINT},
        "generation": {
            "temperature": 0.1,
            "seed": 42,
            "max_tokens": 50,
            "enable_thinking": False,
        },
        "embedding": "NONE",
        "judge": "NONE",
        "hosted_calls": 0,
        "dataset": {
            "repository": "https://github.com/snap-research/locomo",
            "revision": EXPECTED_LOCOMO_COMMIT,
            "file_sha256": EXPECTED_DATASET_SHA256,
            "license": "CC BY-NC 4.0; non-commercial use only",
            "conversation_id": conversation["sample_id"],
            "n_questions": len(selected),
            "question_ids": [row["question_id"] for row in predictions],
        },
        "mean_token_f1": (
            sum(float(row["token_f1"]) for row in valid) / len(valid) if valid else None
        ),
        "per_category": {
            str(category): next(
                (row["token_f1"] for row in valid if row["category"] == category), None
            )
            for category in sorted({int(row["category"]) for _, row in selected})
        },
        "infra_failures": len(selected) - len(valid),
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "prediction_frozen": True,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "COMPLETED_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())
