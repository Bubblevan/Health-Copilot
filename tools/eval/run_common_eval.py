"""Run one profile or the Qwen3-8B 2x2 RAG/adaptive matrix through Harness."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from health_ai_copilot.evaluation.datasets import (
    CMBAdapter,
    CMBCommon1024Adapter,
    DiagnosisArenaAdapter,
)
from health_ai_copilot.evaluation.factorial import (
    CORE_PROFILE_ALIASES,
    summarize_core_factorial,
)
from health_ai_copilot.evaluation.manifests import file_sha256
from health_ai_copilot.evaluation.runner import run_dataset
from health_ai_copilot.harness.profiles import (
    MemoryMode,
    RetrievalMode,
    profile_from_dict,
)
from health_ai_copilot.harness.runtime import HealthCopilotHarness
from health_ai_copilot.harness.trace import JsonlTraceSink
from health_ai_copilot.providers.model import VllmModelProvider


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("diagnosisarena", "cmb", "cmb-common"), required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile")
    selection.add_argument("--matrix", choices=("core",))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--subset-manifest", type=Path, default=ROOT / "configs/eval/cmb_common_1024.json")
    return parser.parse_args()


def _load_adapter(args: argparse.Namespace):
    configuration = json.loads((ROOT / "configs/eval/common_eval_v1.json").read_text(encoding="utf-8"))
    if args.dataset == "diagnosisarena":
        dataset = configuration["datasets"]["diagnosisarena"]
    elif args.dataset == "cmb-common":
        dataset = configuration["datasets"]["cmb-common"]
    else:
        dataset = configuration["extended_dataset"]
    if dataset.get("readiness", "") != "READY":
        raise SystemExit(f"dataset is not qualified: {dataset.get('readiness', 'BLOCKED')}")
    path = args.dataset_path or ROOT / dataset["snapshot_path"]
    if file_sha256(path) != dataset["snapshot_sha256"]:
        raise SystemExit("dataset snapshot hash does not match its frozen manifest")
    revision = dataset["source_revision"]
    if not revision:
        raise SystemExit("dataset source revision is not pinned")
    if args.dataset == "diagnosisarena":
        return DiagnosisArenaAdapter(path, source_revision=revision)
    if args.dataset == "cmb":
        return CMBAdapter(path, source_revision=revision)
    subset = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    if subset["status"] != "FROZEN":
        raise SystemExit("CMB-COMMON-1024 subset is not frozen")
    return CMBCommon1024Adapter(
        path,
        source_revision=revision,
        case_ids=tuple(subset["case_ids"]),
        selection_seed=subset["selection_seed"],
        expected_subset_sha256=subset["case_ids_sha256"],
    )


async def _run(args: argparse.Namespace) -> None:
    registry = json.loads((ROOT / "configs/eval/profile_registry.json").read_text(encoding="utf-8"))
    aliases = CORE_PROFILE_ALIASES if args.matrix == "core" else (args.profile,)
    profiles = {}
    for alias in aliases:
        try:
            profiles[alias] = profile_from_dict(registry["profiles"][alias])
        except KeyError as exc:
            raise SystemExit(f"unknown profile alias: {alias}") from exc
    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    checkpoint_hash = model_config.get("checkpoint_sha256")
    if model_config.get("status") != "READY" or not checkpoint_hash:
        raise SystemExit("model identity is not frozen; set status READY and checkpoint_sha256")
    required_identity = ("model_id", "base_revision", "tokenizer_revision")
    if any(not model_config.get(item) for item in required_identity):
        raise SystemExit("model/base/tokenizer identity is incomplete")
    if any(model_config.get("model_variant") != profile.model_variant.value for profile in profiles.values()):
        raise SystemExit("model config variant does not match every selected profile")
    if any(profile.model_variant.value != "qwen3_8b_base" for profile in profiles.values()) and not model_config.get("training_manifest_sha256"):
        raise SystemExit("post-trained model requires a frozen training manifest hash")
    if len(checkpoint_hash) != 64 or any(ch not in "0123456789abcdef" for ch in checkpoint_hash.lower()):
        raise SystemExit("checkpoint_sha256 must be a SHA-256 hex digest")
    base_url = os.environ.get("VLLM_BASE_URL")
    served_model = os.environ.get("VLLM_MODEL_NAME")
    if not base_url or not served_model:
        raise SystemExit("set VLLM_BASE_URL and VLLM_MODEL_NAME for the OpenAI-compatible endpoint")
    if any(profile.memory_mode is MemoryMode.READ for profile in profiles.values()):
        raise SystemExit("Memory READ profiles require an explicitly bound MemoryProvider")

    # Validate every requested arm before starting model calls.
    retrieval_providers = {}
    for alias, profile in profiles.items():
        if profile.retrieval_mode is RetrievalMode.STANDARD:
            retrieval_providers[alias] = _load_retrieval_provider()

    # Fail on dataset identity or parser errors before the first full run begins.
    _load_adapter(args).cases()
    for alias, profile in profiles.items():
        provider = VllmModelProvider(
            base_url=base_url,
            model=served_model,
            api_key=os.environ.get("VLLM_API_KEY", "local-vllm"),
            default_temperature=float(model_config["serving"]["temperature"]),
            max_output_tokens=int(model_config["serving"]["max_output_tokens"]),
        )
        output_dir = args.output / alias if args.matrix else args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        traces: dict[str, dict[str, object]] = {}
        trace_sink = JsonlTraceSink(output_dir / "traces.jsonl")

        def collect_trace(trace, *, sink=trace_sink, store=traces) -> None:
            sink(trace)
            store[trace.trace_id] = trace.to_dict()

        harness = HealthCopilotHarness(
            model_providers={profile.model_variant: provider},
            retrieval_provider=retrieval_providers.get(alias),
            trace_sink=collect_trace,
        )
        summary = await run_dataset(
            harness=harness,
            adapter=_load_adapter(args),
            profile=profile,
            model_hash=checkpoint_hash,
            output_dir=output_dir,
            resume=args.resume,
            trace_reader=traces.get,
        )
        print(json.dumps({"profile": alias, "summary": summary}, ensure_ascii=False, indent=2))

    if args.matrix == "core":
        arms = {
            alias: _load_core_arm(args.output / alias, alias)
            for alias in CORE_PROFILE_ALIASES
        }
        correctness = {alias: arm["correctness"] for alias, arm in arms.items()}
        matrix_summary = summarize_core_factorial(correctness)
        single_rag = arms["B1"]["retrieval_evidence_sha256"]
        adaptive_rag = arms["B3"]["retrieval_evidence_sha256"]
        retrieval_case_ids = set(single_rag) | set(adaptive_rag)
        missing_retrieval = sorted(set(correctness["B1"]) - retrieval_case_ids)
        mismatched = sorted(
            case_id for case_id in retrieval_case_ids
            if single_rag.get(case_id) != adaptive_rag.get(case_id)
        )
        matrix_summary["rag_evidence_parity"] = {
            "matched_cases": len(retrieval_case_ids) - len(mismatched),
            "missing_case_ids": missing_retrieval,
            "mismatched_case_ids": mismatched,
            "status": "PASS" if not mismatched and not missing_retrieval
            else "FAIL",
        }
        matrix_summary["dataset"] = args.dataset
        matrix_summary["model_hash"] = checkpoint_hash
        matrix_summary["profile_summaries"] = {
            alias: f"{alias}/summary.json" for alias in CORE_PROFILE_ALIASES
        }
        (args.output / "matrix_summary.json").write_text(
            json.dumps(matrix_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"matrix_summary": matrix_summary}, ensure_ascii=False, indent=2))


def _load_retrieval_provider():
    kb = json.loads((ROOT / "configs/eval/common_medical_kb_v1.json").read_text(encoding="utf-8"))
    if kb.get("status") != "READY" or kb.get("readiness") != "YES":
        raise SystemExit("RAG profile blocked: COMMON_MEDICAL_KB_V1 is not qualified")
    factory_path = kb.get("provider_factory")
    if not factory_path or ":" not in factory_path:
        raise SystemExit("qualified Common KB requires a local retrieval provider factory")
    module_name, factory_name = factory_path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name)
    retrieval_provider = factory(kb)
    if not callable(getattr(retrieval_provider, "retrieve", None)):
        raise SystemExit("configured retrieval factory did not return a RetrievalProvider")
    return retrieval_provider


def _load_core_arm(directory: Path, alias: str) -> dict[str, dict[str, bool] | dict[str, str]]:
    case_path = directory / "cases.jsonl"
    trace_path = directory / "traces.jsonl"
    case_rows = {}
    with case_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("profile_id") != alias:
                raise ValueError(f"profile mismatch in {case_path.name}:{line_no}")
            case_id = str(row["case_id"])
            if case_id in case_rows:
                raise ValueError(f"duplicate case ID in {case_path.name}:{line_no}")
            case_rows[case_id] = row
    traces = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            trace = json.loads(line)
            trace_id = str(trace["trace_id"])
            if trace_id in traces:
                raise ValueError(f"duplicate trace ID in {trace_path.name}:{line_no}")
            traces[trace_id] = trace
    correctness = {}
    retrieval_evidence = {}
    for case_id, row in case_rows.items():
        correctness[case_id] = bool(row["score"]["correct"])
        response = row["response"]
        trace = traces.get(response.get("trace_id"))
        if trace is None:
            continue
        completed = [
            event for event in trace.get("events", ())
            if event.get("kind") == "retrieval_completed"
        ]
        if completed:
            retrieval_evidence[case_id] = str(completed[-1].get("fields", {}).get("evidence_sha256", ""))
    return {"correctness": correctness, "retrieval_evidence_sha256": retrieval_evidence}


if __name__ == "__main__":
    asyncio.run(_run(_arguments()))
