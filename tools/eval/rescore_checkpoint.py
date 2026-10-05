"""Reparse frozen raw model outputs with the current deterministic MCQ parser."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/eval"))

from run_common_eval import _load_adapter

from health_ai_copilot.evaluation.runner import _score_dict, summarize_records
from health_ai_copilot.harness.verification import ANSWER_PARSER_REVISION, parse_answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("diagnosisarena", "cmb-common"), required=True)
    parser.add_argument("--prepared-config", type=Path, required=True)
    parser.add_argument("--subset-manifest", type=Path, default=ROOT / "configs/eval/cmb_common_1024.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepared = json.loads(args.prepared_config.read_text(encoding="utf-8"))
    entry = prepared["datasets"][args.dataset]
    adapter_args = SimpleNamespace(
        dataset=args.dataset,
        candidate_view=Path(entry["candidate_view_path"]),
        scorer_view=Path(entry["scorer_view_path"]),
        ids_manifest=Path(entry["ids_manifest_path"]),
        dataset_path=None,
        subset_manifest=args.subset_manifest,
    )
    adapter = _load_adapter(adapter_args)
    cases = {case.case_id: case for case in adapter.cases()}
    checkpoint = args.output / "cases.jsonl"
    summary_path = args.output / "summary.json"
    manifest_path = args.output / "run_manifest.json"
    records: list[dict[str, object]] = []
    with checkpoint.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = str(record["case_id"])
            if case_id not in cases:
                raise ValueError(f"checkpoint case ID is absent from frozen IDs: {case_id}")
            response = record["response"]
            parsed = parse_answer(response.get("answer_text", ""), cases[case_id].answer_schema)
            response["parsed_answer"] = list(parsed) if isinstance(parsed, tuple) else parsed
            record["score"] = _score_dict(adapter.score(
                cases[case_id], SimpleNamespace(parsed_answer=parsed),
            ))
            records.append(record)
    if len(records) != len(cases) or len({str(row["case_id"]) for row in records}) != len(cases):
        raise ValueError(f"refusing to rescore an incomplete or duplicate checkpoint: {len(records)}/{len(cases)}")

    temporary = checkpoint.with_suffix(".jsonl.rescored.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(checkpoint)

    summary = summarize_records(records)
    if summary_path.is_file():
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for field in (
            "run_wall_seconds", "cases_per_second", "provider_requests_per_second",
            "tokens_per_second", "case_concurrency", "provider_calls_including_retries",
            "tokens_including_retries", "provider_calls_per_case_including_retries",
        ):
            if field in old_summary:
                summary[field] = old_summary[field]
    summary.update({
        "dataset_id": adapter.dataset_id,
        "source_revision": adapter.source_revision,
        "dataset_snapshot_sha256": adapter.snapshot_sha256,
        "dataset_selection_sha256": adapter.subset_sha256,
        "case_count": len(cases),
        "checkpoint": checkpoint.name,
        "scoring_revision": ANSWER_PARSER_REVISION,
        "rescored_from_raw_model_outputs": True,
    })
    wall_seconds = float(summary.get("run_wall_seconds") or 0.0)
    if wall_seconds > 0:
        summary.setdefault("cases_per_second", round(len(cases) / wall_seconds, 5))
        if "provider_calls_including_retries" in summary:
            summary["provider_requests_per_second"] = round(
                float(summary["provider_calls_including_retries"]) / wall_seconds, 5,
            )
        if "tokens_including_retries" in summary:
            summary["tokens_per_second"] = round(
                float(summary["tokens_including_retries"]) / wall_seconds, 3,
            )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault(
            "execution_implementation_sha256",
            manifest.get("implementation_sha256", {}),
        )
        parser_path = ROOT / "src/health_ai_copilot/harness/verification.py"
        implementation_files = (
            ROOT / "src/health_ai_copilot/evaluation/datasets.py",
            ROOT / "src/health_ai_copilot/evaluation/runner.py",
            ROOT / "src/health_ai_copilot/harness/runtime.py",
            parser_path,
            ROOT / "src/health_ai_copilot/providers/model.py",
            ROOT / "src/health_ai_copilot/reasoning/single.py",
            ROOT / "src/health_ai_copilot/reasoning/adaptive_mdt.py",
            ROOT / "src/health_ai_copilot/multi_agent/mdagents_style.py",
            ROOT / "tools/eval/run_common_eval.py",
            ROOT / "tools/eval/rescore_checkpoint.py",
            ROOT / "configs/eval/common_eval_v1.json",
            ROOT / "configs/eval/cmb_common_1024.json",
            ROOT / "configs/eval/local_prepared_views_h0.json",
        )
        manifest["postprocessing"] = {
            "rescored_from_raw_model_outputs": True,
            "scoring_revision": ANSWER_PARSER_REVISION,
            "parser_sha256": sha256(parser_path.read_bytes()).hexdigest(),
            "scoring_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "rescored_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest["implementation_sha256"] = {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in implementation_files if path.is_file()
        }
        model_config_path = ROOT / "configs/models/qwen3_8b_base.json"
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        manifest["model"] = model_config
        manifest["model_config_sha256"] = sha256(model_config_path.read_bytes()).hexdigest()
        serving = manifest.get("vllm_runtime_config")
        if not isinstance(serving, dict) or not serving:
            segments = manifest.get("execution_segments", [])
            if segments and isinstance(segments[-1], dict):
                serving = segments[-1].get("vllm_runtime_config")
        if not isinstance(serving, dict) or not serving:
            serving = manifest.get("serving") or model_config.get("serving", {})
        manifest["configured_model_serving"] = model_config.get("serving", {})
        manifest["serving"] = serving
        manifest["scoring_revision"] = ANSWER_PARSER_REVISION
        manifest["decoding"] = {
            "temperature": model_config["serving"]["temperature"],
            "top_p": model_config["serving"].get("top_p", 1.0),
            "do_sample": model_config["serving"].get("do_sample", False),
            "max_output_tokens": model_config["serving"]["max_output_tokens"],
            "chat_template_kwargs": model_config["serving"].get("chat_template_kwargs", {}),
            "seed": model_config["serving"].get("seed"),
            "seed_scope": model_config["serving"].get("seed_scope"),
        }
        manifest["summary"] = summary
        if manifest.get("recovery_composite"):
            manifest["status"] = "TARGETED_TRANSPORT_RECOVERY_SCORED"
            manifest["recovery_composite"]["scoring_status"] = "COMPLETE"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
