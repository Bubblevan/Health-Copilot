"""Controlled, resumable LongMemEval-S runner for the pinned MemEval checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mem1_artifacts import (
    append_jsonl,
    canonical_json,
    find_cached_prediction,
    make_cache_identity,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_hash_sidecar,
    write_hash_sidecar,
    write_run_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
SPLIT_PATH = ROOT / "docs" / "research" / "memory" / "split_manifest.json"
MODEL_PATH = ROOT / "docs" / "research" / "memory" / "model_protocol.json"
DATASET_PATH = ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
PINNED_MEMEVAL_SHA = "807ae6d7d8a5b76f6fe964d5a581d96c036e2ac4"
SYSTEMS = ("fullcontext", "openclaw", "propmem", "mem0", "simplemem")


def _digest_ids(question_ids: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{question_id}\n" for question_id in sorted(question_ids)).encode()
    ).hexdigest()


def resolve_question_ids(
    *,
    split_manifest: dict[str, Any],
    selected_ids: list[str] | None = None,
) -> list[str]:
    dev_ids = set(split_manifest["dev"]["question_ids"])
    ids = list(selected_ids) if selected_ids is not None else sorted(dev_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection must contain unique, non-empty question IDs")
    outside_dev = set(ids) - dev_ids
    if outside_dev:
        raise ValueError(f"Selection contains IDs outside frozen DEV: {sorted(outside_dev)}")
    if split_manifest.get("test_access", False):
        raise ValueError("Split manifest does not certify test_access=false")
    return ids


def select_records(
    raw_items: list[dict[str, Any]], question_ids: list[str], normalize,
) -> list[dict[str, Any]]:
    wanted = set(question_ids)
    records = [normalize(item) for item in raw_items if item.get("question_id") in wanted]
    observed = [record["qa"][0]["question_id"] for record in records]
    if set(observed) != wanted or len(observed) != len(wanted):
        raise ValueError("Dataset does not contain exactly the requested DEV questions")
    return records


def _source_prompt_hash(memeval_src: Path) -> str:
    digest = hashlib.sha256()
    for source in sorted((memeval_src / "agents_memory").rglob("*.py")):
        digest.update(source.relative_to(memeval_src).as_posix().encode())
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def verify_pinned_patch(memeval_root: Path, patch_path: Path, expected_patch_sha: str) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(memeval_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PINNED_MEMEVAL_SHA:
        raise RuntimeError(f"MemEval checkout must be pinned at {PINNED_MEMEVAL_SHA}; found {head}")
    if sha256_file(patch_path) != expected_patch_sha:
        raise RuntimeError("MemEval patch SHA does not match the locked compatibility manifest")
    subprocess.run(
        ["git", "-C", str(memeval_root), "apply", "--reverse", "--check", str(patch_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_system_registry(memeval_root: Path, selected: list[str]):
    source_root = memeval_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    os.environ["HC_MEM1_SYSTEMS"] = ",".join(selected)
    importlib.invalidate_caches()
    registry = importlib.import_module("agents_memory.systems")
    registry = importlib.reload(registry)
    systems = registry.SYSTEMS
    if set(systems) != set(selected):
        raise RuntimeError(f"Patched MemEval did not load requested systems: {sorted(systems)}")
    return systems


def _read_selection(args: argparse.Namespace, dev_manifest: dict[str, Any]) -> list[str]:
    if args.question_id:
        selected = args.question_id
    elif args.selection_manifest:
        selected_doc = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        if selected_doc.get("dataset_sha256") != dev_manifest.get("dataset_sha256"):
            raise ValueError("Selection manifest dataset SHA differs from frozen DEV")
        selected = selected_doc.get("question_ids")
        if not isinstance(selected, list):
            raise ValueError("Selection manifest must contain a question_ids list")
    else:
        selected = None
    return resolve_question_ids(split_manifest=dev_manifest, selected_ids=selected)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "judge"), required=True)
    parser.add_argument("--track", choices=("main", "upstream_parity"), required=True)
    parser.add_argument("--memeval-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--question-id", action="append")
    parser.add_argument("--system", choices=SYSTEMS, action="append")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--split-manifest", type=Path, default=SPLIT_PATH)
    parser.add_argument("--patch", type=Path, default=ROOT / "tools/research/memory/patches/memeval_qwen_main_v1.patch")
    parser.add_argument("--reader-artifact-sha256", default=None)
    return parser.parse_args()


def _load_locked_inputs(args: argparse.Namespace):
    split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    model_protocol = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    if split.get("status") != "FROZEN_ACTIVE_PROTOCOL_LOCKED":
        raise RuntimeError("MEM-1 requires the frozen active MEM-0 split")
    if split.get("dataset_sha256") != sha256_file(args.dataset):
        raise RuntimeError("Local LongMemEval-S bytes do not match the frozen dataset SHA")
    if split.get("dataset_revision") != "98d7416c24c778c2fee6e6f3006e7a073259d48f":
        raise RuntimeError("LongMemEval-S revision differs from the frozen protocol")
    selected_ids = _read_selection(args, split)
    return split, model_protocol, selected_ids


def _provider_manifest(config, selected_systems, system_info, reader_sha):
    uses_embeddings = any(name != "fullcontext" for name in selected_systems)
    return {
        "reader_answer_model": {
            "provider": config.reader_provider,
            "model": config.reader_model,
            "artifact_sha256": reader_sha,
        },
        "memory_system": {
            "systems": selected_systems,
            "display_names": {name: system_info[name].get("architecture", name) for name in selected_systems},
        },
        "embedding_model": {
            "provider": "openai" if uses_embeddings else "none",
            "model": config.embedding_model if uses_embeddings else None,
        },
        "judge_model": {"provider": "openai", "model": config.judge_model},
    }


def _latest_predictions(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest = {}
    for row in rows:
        latest[(row["system"], row["question_id"])] = row
    return latest


def _summarize_predictions(rows: list[dict[str, Any]], systems: list[str], ids: list[str]) -> dict:
    latest = _latest_predictions(rows)
    output = {}
    for system in systems:
        selected = [latest[(system, qid)] for qid in ids if (system, qid) in latest]
        quality = [row for row in selected if row.get("quality_status") == "OK" and row.get("f1") is not None]
        output[system] = {
            "requested": len(ids),
            "completed": len(selected),
            "quality_n": len(quality),
            "infra_failure_n": sum(row.get("quality_status") != "OK" for row in selected),
            "f1_mean": sum(row["f1"] for row in quality) / len(quality) if quality else None,
        }
    return output


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summarize_calls(call_rows: list[dict[str, Any]], systems: list[str]) -> dict:
    output = {}
    roles = ("reader_answer", "memory_ingest", "memory_reasoning", "embedding", "judge")
    for system in systems:
        rows = [row for row in call_rows if row.get("system") == system]
        role_usage = {}
        for role in roles:
            role_rows = [row for row in rows if row.get("role") == role]
            role_usage[role] = {
                "calls": len(role_rows),
                "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in role_rows),
                "completion_tokens": sum(row.get("completion_tokens") or 0 for row in role_rows),
                "failed_calls": sum(row.get("success") is False for row in role_rows),
            }
        answer_latencies = [
            float(row["latency_ms"])
            for row in rows
            if row.get("role") == "reader_answer" and row.get("latency_ms") is not None
        ]
        local_latencies = [
            float(row["latency_ms"])
            for row in rows
            if row.get("provider") == "local_qwen" and row.get("latency_ms") is not None
        ]
        output[system] = {
            "by_role": role_usage,
            "answer_latency_p50_ms": _percentile(answer_latencies, 0.50),
            "answer_latency_p95_ms": _percentile(answer_latencies, 0.95),
            "local_qwen_call_wall_ms": sum(local_latencies),
        }
    return output


def _write_report(
    run_dir: Path,
    run_manifest: dict[str, Any],
    deterministic: dict[str, Any],
    judged: dict[str, Any] | None = None,
) -> None:
    systems = run_manifest["roles"]["memory_system"]["systems"]
    lines = [
        "# MEM-1 Run Report",
        "",
        f"- Run: `{run_manifest['run_id']}`",
        f"- Track: `{run_manifest['track']}`",
        f"- Split: `{run_manifest['split']}` ({len(run_manifest['question_ids'])} frozen DEV questions)",
        f"- Dataset SHA256: `{run_manifest['dataset']['sha256']}`",
        f"- Prediction SHA256: `{deterministic['prediction_sha256']}`",
        "- TEST access: `false`",
        "",
        "| System | Quality N | Infra failures | Token F1 | Judge N | Judge errors | Native accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in systems:
        prediction = deterministic["systems"][system]
        judge_result = (judged or {}).get(system, {})
        f1 = prediction["f1_mean"]
        accuracy = judge_result.get("native_accuracy")
        lines.append(
            "| {system} | {quality} | {infra} | {f1} | {judge_n} | {judge_errors} | {accuracy} |".format(
                system=system,
                quality=prediction["quality_n"],
                infra=prediction["infra_failure_n"],
                f1="-" if f1 is None else f"{f1:.4f}",
                judge_n=judge_result.get("judge_n", "-"),
                judge_errors=judge_result.get("judge_error_n", "-"),
                accuracy="-" if accuracy is None else f"{accuracy:.4f}",
            )
        )
    lines.extend(["", "Reader, memory system, embedding model and judge are recorded as separate roles in `run_manifest.json`.", ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _finalize_generation(
    run_dir: Path,
    manifest: dict[str, Any],
    systems: list[str],
    question_ids: list[str],
    embedding_model: str | None,
) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    sidecar_path = run_dir / "predictions.sha256"
    if sidecar_path.exists() and not verify_hash_sidecar(predictions_path, sidecar_path):
        raise RuntimeError("Frozen prediction hash mismatch; refusing to score")
    rows = read_jsonl(predictions_path)
    latest = _latest_predictions(rows)
    expected = {(name, question_id) for name in systems for question_id in question_ids}
    if set(latest) != expected:
        missing = sorted(expected - set(latest))
        unexpected = sorted(set(latest) - expected)
        raise RuntimeError(f"Generation coverage mismatch; missing={missing}, unexpected={unexpected}")
    if sidecar_path.exists():
        digest = sidecar_path.read_text(encoding="ascii").split()[0]
    else:
        digest = write_hash_sidecar(predictions_path, sidecar_path)
    call_rows = read_jsonl(run_dir / "call_ledger.jsonl")
    efficiency = _summarize_calls(call_rows, systems)
    (run_dir / "token_efficiency.json").write_text(
        json.dumps(efficiency, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    embedding_rows = [row for row in call_rows if row.get("role") == "embedding"]
    (run_dir / "embeddings_usage.json").write_text(
        json.dumps({
            "model": embedding_model,
            "by_system": {
                system_name: {
                    "calls": sum(
                        row.get("system") == system_name for row in embedding_rows
                    ),
                    "input_tokens": sum(
                        row.get("prompt_tokens") or 0 for row in embedding_rows
                        if row.get("system") == system_name
                    ),
                    "failed_calls": sum(
                        row.get("success") is False for row in embedding_rows
                        if row.get("system") == system_name
                    ),
                }
                for system_name in systems
            },
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "prediction_sha256": digest,
        "systems": _summarize_predictions(rows, systems, question_ids),
    }
    (run_dir / "deterministic_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(run_dir, manifest, metrics)
    return metrics


def generate(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ["HC_MEMORY_TRACK"] = args.track
    split, model_protocol, question_ids = _load_locked_inputs(args)
    compatibility = json.loads((ROOT / "docs/research/memory/baseline_compatibility_matrix.json").read_text(encoding="utf-8"))
    patch_sha = compatibility["patch_provenance"]["patch_sha256"]
    verify_pinned_patch(args.memeval_root, args.patch, patch_sha)

    system_names = list(dict.fromkeys(args.system))
    systems = _load_system_registry(args.memeval_root, system_names)
    from agents_memory.benchmarks.longmemeval import _normalize
    from agents_memory.healthcopilot_provider import ProviderConfig, configure_call_ledger
    from agents_memory.locomo import CATEGORY_NAMES

    config = ProviderConfig.from_env()
    reader_sha = args.reader_artifact_sha256 or (
        model_protocol["main_track"]["sha256"]
        if args.track == "main"
        else sha256_bytes(canonical_json({"provider": "openai", "model": config.reader_model}))
    )
    if args.track == "main" and reader_sha != model_protocol["main_track"]["sha256"]:
        raise RuntimeError("Reader artifact SHA differs from frozen Main Track Qwen3-8B")
    raw_items = json.loads(args.dataset.read_text(encoding="utf-8"))
    conversations = select_records(raw_items, question_ids, _normalize)
    source_hash = _source_prompt_hash(args.memeval_root / "src")
    roles = _provider_manifest(config, system_names, systems, reader_sha)
    runner_code_sha = sha256_bytes(canonical_json({
        "runner": sha256_file(__file__),
        "artifact_helpers": sha256_file(Path(__file__).with_name("mem1_artifacts.py")),
    }))
    cache_code_hash = sha256_bytes(canonical_json({
        "memeval_patch_sha256": patch_sha,
        "healthcopilot_runner_sha256": runner_code_sha,
    }))
    config_hashes = {
        name: sha256_bytes(canonical_json({
            "system": name,
            "architecture": systems[name].get("architecture"),
            "infrastructure": systems[name].get("infrastructure"),
            "reader_model": config.reader_model,
            "reader_generation": model_protocol["main_track"].get("generation") if args.track == "main" else "upstream-pinned",
            "embedding_model": config.embedding_model if roles["embedding_model"]["model"] else None,
        }))
        for name in system_names
    }
    roles["memory_system"]["system_config_sha256"] = config_hashes
    roles["memory_system"]["prompt_source_sha256"] = source_hash
    roles["reader_answer_model"]["generation"] = (
        model_protocol["main_track"]["generation"]
        if args.track == "main"
        else "Pinned upstream per-system settings"
    )
    roles["judge_model"]["temperature"] = 0
    roles["judge_model"]["max_tokens"] = 10
    if roles["embedding_model"]["model"]:
        roles["embedding_model"]["dimensions"] = 1536
    run_config_hash = sha256_bytes(canonical_json(config_hashes))
    write_run_manifest(
        args.run_dir / "run_manifest.json",
        run_id=args.run_dir.name,
        track=args.track,
        dataset={
            "id": split["dataset_id"],
            "revision": split["dataset_revision"],
            "sha256": split["dataset_sha256"],
            "dev_ids_sha256": split["dev"]["question_ids_sha256_sorted_lf"],
            "test_access": False,
        },
        split="DEV",
        question_ids=question_ids,
        roles=roles,
        system_config_hash=run_config_hash,
        code_patch_sha256=patch_sha,
        runner_code_sha256=runner_code_sha,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    predictions_path = args.run_dir / "predictions.jsonl"
    sidecar_path = args.run_dir / "predictions.sha256"
    if sidecar_path.exists():
        _finalize_generation(
            args.run_dir,
            json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8")),
            system_names,
            question_ids,
            roles["embedding_model"]["model"],
        )
        return
    configure_call_ledger(args.run_dir / "call_ledger.jsonl")
    source_prompt_hash = {"memory_system_and_prompts": source_hash}
    for system_name in system_names:
        system = systems[system_name]
        for conversation in conversations:
            qa = conversation["qa"][0]
            question_id = qa["question_id"]
            embedding_model = config.embedding_model if roles["embedding_model"]["model"] else None
            cache_identity = make_cache_identity(
                system=system_name,
                question_id=question_id,
                dataset_sha256=split["dataset_sha256"],
                system_config_hash=config_hashes[system_name],
                prompt_hashes=source_prompt_hash,
                reader_artifact_sha256=reader_sha,
                embedding_model=embedding_model,
                code_patch_hash=cache_code_hash,
            )
            cached = find_cached_prediction(predictions_path, cache_identity)
            if cached is not None:
                continue
            system_started = time.perf_counter()
            try:
                with_qa = {**conversation, "qa": [qa]}
                result_rows = system["fn"](
                    with_qa,
                    config.reader_model,
                    False,
                    category_names=CATEGORY_NAMES,
                    judge_fn="longmemeval",
                )
                if len(result_rows) != 1:
                    raise RuntimeError("MEM-1 adapter must return exactly one row per question")
                result = result_rows[0]
            except Exception as error:
                result = {
                    "question_id": question_id,
                    "sample_id": question_id,
                    "question": qa["question"],
                    "ground_truth": qa["answer"],
                    "predicted": None,
                    "category": qa["category"],
                    "category_name": qa["category"],
                    "f1": None,
                    "quality_status": "INFRA_FAILURE",
                    "reader_error_type": type(error).__name__,
                    "reader_messages": None,
                }
            row = {
                **result,
                "system": system_name,
                "question_id": question_id,
                "cache_identity": cache_identity,
                "system_wall_time_ms": round((time.perf_counter() - system_started) * 1000, 3),
            }
            append_jsonl(predictions_path, row)
            if row["quality_status"] != "OK":
                append_jsonl(args.run_dir / "failures.jsonl", {
                    "question_id": question_id,
                    "system": system_name,
                    "failure_type": row.get("reader_error_type", "INFRA_LIBRARY"),
                    "quality_status": "INFRA_FAILURE",
                })

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    _finalize_generation(
        args.run_dir,
        manifest,
        system_names,
        question_ids,
        roles["embedding_model"]["model"],
    )


def judge(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    os.environ["HC_MEMORY_TRACK"] = args.track
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("track") != args.track:
        raise RuntimeError("Judge track must match the frozen run manifest")
    compatibility = json.loads((ROOT / "docs/research/memory/baseline_compatibility_matrix.json").read_text(encoding="utf-8"))
    verify_pinned_patch(
        args.memeval_root,
        args.patch,
        compatibility["patch_provenance"]["patch_sha256"],
    )
    if set(args.system) != set(manifest["roles"]["memory_system"]["systems"]):
        raise RuntimeError("Judge system list must match the frozen run manifest")
    source_root = str(args.memeval_root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from agents_memory.evaluation import evaluate_longmemeval
    from agents_memory.healthcopilot_provider import call_context, configure_call_ledger

    predictions_path = args.run_dir / "predictions.jsonl"
    if not verify_hash_sidecar(predictions_path, args.run_dir / "predictions.sha256"):
        raise RuntimeError("Prediction artifact is not frozen or its hash does not match")
    configure_call_ledger(args.run_dir / "call_ledger.jsonl")
    rows = _latest_predictions(read_jsonl(predictions_path))
    judge_path = args.run_dir / "judge_results.jsonl"
    judged = {
        (row["system"], row["question_id"]): row
        for row in read_jsonl(judge_path)
        if row.get("judge_status") == "OK"
    }
    for system_name in manifest["roles"]["memory_system"]["systems"]:
        for question_id in manifest["question_ids"]:
            key = (system_name, question_id)
            prediction = rows.get(key)
            if prediction is None or prediction.get("quality_status") != "OK" or key in judged:
                continue
            with call_context(system_name, question_id, "judge"):
                result = evaluate_longmemeval(
                    prediction["question"],
                    prediction["ground_truth"],
                    prediction["predicted"],
                    category=str(prediction["category"]),
                    question_id=question_id,
                )
            append_jsonl(judge_path, {"system": system_name, "question_id": question_id, **result})
            if result.get("judge_status") != "OK":
                append_jsonl(args.run_dir / "failures.jsonl", {
                    "question_id": question_id,
                    "system": system_name,
                    "failure_type": "INFRA_JUDGE",
                    "quality_status": "OK",
                    "judge_status": "ERROR",
                })
    judge_rows = read_jsonl(judge_path)
    latest_judge = {
        (row["system"], row["question_id"]): row for row in judge_rows
    }
    output = {}
    for system_name in manifest["roles"]["memory_system"]["systems"]:
        selected = [
            latest_judge[(system_name, question_id)]
            for question_id in manifest["question_ids"]
            if (system_name, question_id) in latest_judge
        ]
        valid = [row["longmemeval_correct"] for row in selected if row.get("judge_status") == "OK"]
        output[system_name] = {
            "judge_n": len(valid),
            "judge_error_n": sum(row.get("judge_status") != "OK" for row in selected),
            "native_accuracy": sum(valid) / len(valid) if valid else None,
        }
    (args.run_dir / "judge_metrics.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metrics_path = args.run_dir / "deterministic_metrics.json"
    deterministic = json.loads(metrics_path.read_text(encoding="utf-8"))
    call_rows = read_jsonl(args.run_dir / "call_ledger.jsonl")
    systems = manifest["roles"]["memory_system"]["systems"]
    (args.run_dir / "token_efficiency.json").write_text(
        json.dumps(_summarize_calls(call_rows, systems), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.run_dir, manifest, deterministic, output)


def main() -> None:
    args = _parse_args()
    if args.stage == "generate" and not args.system:
        raise SystemExit("At least one --system is required for generation")
    if args.stage == "judge" and args.system:
        raise SystemExit("Judge stage reads systems from the frozen run manifest")
    args.system = args.system or []
    if len(args.system) != len(set(args.system)):
        raise SystemExit("Duplicate --system arguments are not allowed")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "generate":
        generate(args)
    else:
        judge(args)


if __name__ == "__main__":
    main()
