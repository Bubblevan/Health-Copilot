"""Evaluate the untouched Qwen3-8B checkpoint on PubMedQA's official test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABELS = ("yes", "no", "maybe")
OFFICIAL_TEST_PREFIXES = {
    "yes": ["12377809", "26163474", "19100463", "18537964", "12913878"],
    "no": ["24577079", "24669960", "15502995", "21214884", "24476003"],
    "maybe": ["18284441", "18802997", "17621202", "11411430", "26708803"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(dataset_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(dataset_path)
    records = table.to_pylist()
    dataset_root = dataset_path.parents[1]
    metadata_path = dataset_root / ".hfd" / "repo_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return records, metadata


def official_test_ids(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Reproduce PubMedQA's published seed-0 stratified 50/50 PQA-L split."""
    groups = {label: [] for label in LABELS}
    for row in records:
        label = str(row["final_decision"]).lower()
        if label not in groups:
            raise ValueError(f"Unexpected PubMedQA label: {label}")
        groups[label].append(str(row["pubid"]))

    random.seed(0)
    test_ids: dict[str, list[str]] = {}
    for label, ids in groups.items():
        random.shuffle(ids)
        boundary = math.ceil(len(ids) / 2)
        test_ids[label] = ids[boundary:]

    for label, expected_prefix in OFFICIAL_TEST_PREFIXES.items():
        if test_ids[label][: len(expected_prefix)] != expected_prefix:
            raise ValueError(
                "The local parquet row order no longer reproduces PubMedQA's "
                f"official test split for label={label}."
            )
    return test_ids


def make_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    sections = row["context"].get("contexts") or []
    labels = row["context"].get("labels") or []
    abstract = "\n\n".join(
        f"[{labels[i] if i < len(labels) else 'ABSTRACT'}]\n{text}"
        for i, text in enumerate(sections)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the biomedical research question using only the supplied "
                "PubMed abstract. Output exactly one of: yes, no, maybe. Do not "
                "include an explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {row['question']}\n\nAbstract:\n{abstract}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def parse_label(text: str) -> str | None:
    match = re.search(r"\b(yes|no|maybe)\b", text.strip().lower())
    return match.group(1) if match else None


def calculate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {gold: {pred: 0 for pred in (*LABELS, "invalid")} for gold in LABELS}
    correct = 0
    invalid = 0
    for row in rows:
        gold = row["gold"].lower()
        pred = row["prediction"] or "invalid"
        confusion[gold][pred] += 1
        correct += pred == gold
        invalid += pred == "invalid"

    per_class_f1: dict[str, float] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in (*LABELS, "invalid") if other != label)
        denominator = 2 * tp + fp + fn
        per_class_f1[label] = 2 * tp / denominator if denominator else 0.0

    return {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "macro_f1": sum(per_class_f1.values()) / len(LABELS),
        "per_class_f1": per_class_f1,
        "invalid_outputs": invalid,
        "label_counts": dict(Counter(row["gold"].lower() for row in rows)),
        "confusion_matrix": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/root/gpufree-share/data/eval/PubMedQA/"
            "pqa_labeled/train-00000-of-00001.parquet"
        ),
    )
    parser.add_argument("--model", type=Path, default=Path("/root/gpufree-share/data/Qwen3-8B"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    records, dataset_metadata = load_records(args.dataset)
    test_ids_by_label = official_test_ids(records)
    test_ids = [pubid for label in LABELS for pubid in test_ids_by_label[label]]
    record_by_id = {str(row["pubid"]): row for row in records}
    if len(test_ids) != 500 or len(set(test_ids)) != 500:
        raise ValueError(f"Expected 500 unique official test IDs, got {len(test_ids)}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    selected_rows = [record_by_id[pubid] for pubid in test_ids]
    prompts = [make_prompt(tokenizer, row) for row in selected_rows]
    token_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    if max(token_lengths) + 8 > args.max_model_len:
        raise ValueError(
            f"Longest prompt ({max(token_lengths)} tokens) exceeds max_model_len="
            f"{args.max_model_len} after generation allowance."
        )

    model_metadata_path = args.model / ".hfd" / "repo_metadata.json"
    model_metadata = (
        json.loads(model_metadata_path.read_text()) if model_metadata_path.exists() else {}
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "untouched Qwen3-8B baseline before SFT/RL",
        "model": {
            "repo_id": model_metadata.get("id", "Qwen/Qwen3-8B"),
            "revision": model_metadata.get("sha"),
            "path": str(args.model),
            "dtype": "bfloat16",
        },
        "dataset": {
            "repo_id": dataset_metadata.get("id", "qiaojin/PubMedQA"),
            "revision": dataset_metadata.get("sha"),
            "license": "mit",
            "config": "pqa_labeled",
            "file": str(args.dataset),
            "sha256": sha256_file(args.dataset),
            "source_rows": len(records),
            "split": "official 500-item test set reconstructed from ordered PQA-L using seed 0",
            "split_reference": (
                "https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py"
            ),
            "split_seed": 0,
            "n": len(test_ids),
            "ids_by_label": test_ids_by_label,
        },
        "inference": {
            "engine": "vLLM",
            "temperature": 0.0,
            "max_tokens": 8,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_size": args.batch_size,
            "thinking": False,
            "prompt": "question + all PubMed abstract sections; output one yes/no/maybe label",
        },
        "prompt_tokens": {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8, seed=0)
    predictions_path = args.output_dir / "predictions.jsonl"
    results: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(selected_rows), args.batch_size):
            batch = llm.generate(
                prompts[start : start + args.batch_size], sampling, use_tqdm=False
            )
            for row, result in zip(selected_rows[start : start + args.batch_size], batch):
                raw = result.outputs[0].text
                item = {
                    "pubid": str(row["pubid"]),
                    "question": row["question"],
                    "gold": row["final_decision"].lower(),
                    "prediction": parse_label(raw),
                    "raw_output": raw,
                }
                item["correct"] = item["prediction"] == item["gold"]
                results.append(item)
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
            output.flush()
            print(f"completed {min(start + args.batch_size, len(selected_rows))}/500", flush=True)

    metrics = calculate_metrics(results)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
