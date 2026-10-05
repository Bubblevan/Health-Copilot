"""Run one profile or the Qwen3-8B 2x2 RAG/adaptive matrix through Harness."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from health_ai_copilot.evaluation.datasets import (
    CMBAdapter,
    CMBCommon1024Adapter,
    DiagnosisArenaAdapter,
    PreparedViewsAdapter,
    SelectedCaseAdapter,
    case_ids_sha256,
)
from health_ai_copilot.evaluation.factorial import (
    CORE_PROFILE_ALIASES,
    summarize_core_factorial,
)
from health_ai_copilot.evaluation.manifests import file_sha256
from health_ai_copilot.evaluation.runner import run_dataset
from health_ai_copilot.harness.budget import BudgetLimits
from health_ai_copilot.harness.profiles import (
    MemoryMode,
    ReasoningMode,
    RetrievalMode,
    profile_from_dict,
)
from health_ai_copilot.harness.runtime import HarnessConfig, HealthCopilotHarness
from health_ai_copilot.harness.trace import JsonlTraceSink
from health_ai_copilot.harness.verification import ANSWER_PARSER_REVISION
from health_ai_copilot.multi_agent.mdagents_style import MDAgentsStyleConfig
from health_ai_copilot.providers.model import VllmModelProvider
from health_ai_copilot.reasoning.adaptive_mdt import AdaptiveMDTReasoner
from health_ai_copilot.reasoning.single import SingleReasoner


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("diagnosisarena", "cmb", "cmb-common"), required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--profile")
    selection.add_argument("--matrix", choices=("core",))
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--vllm-runtime-config",
        type=Path,
        help="JSON manifest of the actual local vLLM engine settings for this run",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--case-ids-file",
        type=Path,
        help="JSON array or {case_ids, case_ids_sha256} to run a frozen subset of the selected dataset",
    )
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--subset-manifest", type=Path, default=ROOT / "configs/eval/cmb_common_1024.json")
    parser.add_argument("--candidate-view", type=Path)
    parser.add_argument("--scorer-view", type=Path)
    parser.add_argument("--ids-manifest", type=Path)
    parser.add_argument("--prepared-config", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--capture-adaptive-diagnostics",
        action="store_true",
        help="persist raw classifier/recruiter replies and validation errors in Adaptive traces",
    )
    args = parser.parse_args()
    if args.prepared_config:
        prepared = json.loads(args.prepared_config.read_text(encoding="utf-8"))
        entry = prepared.get("datasets", {}).get(args.dataset)
        if not isinstance(entry, dict):
            parser.error(f"no prepared views configured for dataset {args.dataset}")
        args.candidate_view = Path(entry["candidate_view_path"])
        args.scorer_view = Path(entry["scorer_view_path"])
        args.ids_manifest = Path(entry["ids_manifest_path"])
    return args


def _load_adapter(args: argparse.Namespace):
    configuration = json.loads((ROOT / "configs/eval/common_eval_v1.json").read_text(encoding="utf-8"))
    if args.dataset == "diagnosisarena":
        dataset = configuration["datasets"]["diagnosisarena"]
    elif args.dataset == "cmb-common":
        dataset = configuration["datasets"]["cmb-common"]
    else:
        dataset = configuration["extended_dataset"]
    prepared_mode = bool(args.candidate_view or args.scorer_view or args.ids_manifest)
    if prepared_mode:
        if args.dataset == "cmb":
            raise SystemExit("prepared-view mode supports DiagnosisArena-915 and CMB-COMMON-1024 only")
        if not all((args.candidate_view, args.scorer_view, args.ids_manifest)):
            raise SystemExit("prepared evaluation requires candidate view, scorer view, and frozen ID manifest")
        if dataset.get("readiness") != "READY_WITH_PREPARED_VIEWS":
            raise SystemExit(f"prepared dataset is not qualified: {dataset.get('readiness', 'BLOCKED')}")
        subset = json.loads(args.subset_manifest.read_text(encoding="utf-8")) if args.dataset == "cmb-common" else {}
        return PreparedViewsAdapter(
            dataset_id="cmb-common-1024" if args.dataset == "cmb-common" else "diagnosisarena",
            candidate_path=args.candidate_view,
            scorer_path=args.scorer_view,
            ids_manifest_path=args.ids_manifest,
            source_revision=dataset["source_revision"],
            expected_count=dataset["expected_cases"],
            expected_candidate_sha256=dataset["candidate_view_sha256"],
            expected_scorer_sha256=dataset["scorer_view_sha256"],
            expected_ids_manifest_sha256=dataset["ids_manifest_sha256"],
            expected_ids_sequence_sha256=dataset["ids_sequence_sha256"],
            expected_subset_sha256=subset.get("case_ids_sha256"),
        )
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
    runtime_config = (
        json.loads(args.vllm_runtime_config.read_text(encoding="utf-8"))
        if args.vllm_runtime_config
        else dict(model_config.get("serving", {}))
    )
    if runtime_config.get("base_url", base_url) != base_url:
        raise SystemExit("vLLM runtime manifest base_url does not match VLLM_BASE_URL")
    if runtime_config.get("served_model", served_model) != served_model:
        raise SystemExit("vLLM runtime manifest served_model does not match VLLM_MODEL_NAME")
    configured_structured = model_config.get("serving", {}).get("structured_outputs_config")
    if (
        configured_structured is not None
        and runtime_config.get("structured_outputs_config") != configured_structured
    ):
        raise SystemExit("vLLM structured_outputs_config differs from the frozen model config")
    if any(profile.memory_mode is MemoryMode.READ for profile in profiles.values()):
        raise SystemExit("Memory READ profiles require an explicitly bound MemoryProvider")

    # Validate every requested arm before starting model calls.
    retrieval_providers = {}
    for alias, profile in profiles.items():
        if profile.retrieval_mode is RetrievalMode.STANDARD:
            retrieval_providers[alias] = _load_retrieval_provider()

    # Fail on dataset identity or parser errors before the first model call.
    _select_cases(_load_adapter(args), args.case_ids_file)
    for alias, profile in profiles.items():
        adapter = _select_cases(_load_adapter(args), args.case_ids_file)
        output_dir = args.output / alias if args.matrix else args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "run_manifest.json"
        current_manifest = _run_manifest(args, alias, profile, model_config, runtime_config, adapter)
        if args.resume and manifest_path.is_file():
            run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field in (
                "git_sha", "model_config_sha256", "implementation_sha256",
                "vllm_runtime_config_sha256",
            ):
                if run_manifest.get(field) != current_manifest.get(field):
                    raise SystemExit(f"resume refused: run manifest {field} changed")
            if run_manifest.get("system_profile") != current_manifest.get("system_profile"):
                raise SystemExit("resume refused: system profile changed")
            old_dataset = run_manifest.get("dataset", {})
            new_dataset = current_manifest.get("dataset", {})
            for field in (
                "source_revision", "combined_dataset_identity_sha256",
                "ids_manifest_sha256", "ids_sequence_sha256",
                "selected_case_ids_sha256", "case_ids_file_sha256",
            ):
                if old_dataset.get(field) != new_dataset.get(field):
                    raise SystemExit(f"resume refused: dataset {field} changed")
        else:
            run_manifest = current_manifest
            run_manifest["started_at_utc"] = datetime.now(UTC).isoformat()
            run_manifest["execution_segments"] = []
        segment = {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "case_concurrency": args.concurrency,
            "served_model": served_model,
            "base_url": base_url,
            "vllm_runtime_config": runtime_config,
        }
        run_manifest.setdefault("execution_segments", []).append(segment)
        run_manifest["status"] = "RUNNING"
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        provider = VllmModelProvider(
            base_url=base_url,
            model=served_model,
            api_key=os.environ.get("VLLM_API_KEY", "local-vllm"),
            default_temperature=float(model_config["serving"]["temperature"]),
            default_top_p=(
                float(model_config["serving"]["top_p"])
                if model_config["serving"].get("top_p") is not None else None
            ),
            max_output_tokens=int(model_config["serving"]["max_output_tokens"]),
            chat_template_kwargs=model_config["serving"].get("chat_template_kwargs"),
        )
        traces: dict[str, dict[str, object]] = {}
        trace_sink = JsonlTraceSink(output_dir / "traces.jsonl")

        def collect_trace(trace, *, sink=trace_sink, store=traces) -> None:
            sink(trace)
            store[trace.trace_id] = trace.to_dict()

        harness = HealthCopilotHarness(
            model_providers={profile.model_variant: provider},
            retrieval_provider=retrieval_providers.get(alias),
            reasoner_factories={
                ReasoningMode.SINGLE: lambda model_provider: SingleReasoner(
                    model_provider,
                    model=served_model,
                    max_output_tokens=int(model_config["serving"]["max_output_tokens"]),
                ),
                ReasoningMode.ADAPTIVE_MDT: lambda model_provider: AdaptiveMDTReasoner(
                    model_provider,
                    model=served_model,
                    config=MDAgentsStyleConfig(
                        model_name=served_model,
                        max_output_tokens=int(model_config["serving"]["max_output_tokens"]),
                        capture_validation_outputs=args.capture_adaptive_diagnostics,
                    ),
                ),
            },
            config=HarnessConfig(budget=BudgetLimits(
                max_provider_calls=int(model_config.get("harness_runtime", {}).get(
                    "max_provider_calls", 32
                )),
                max_tool_calls=int(model_config.get("harness_runtime", {}).get(
                    "max_tool_calls", 16
                )),
                deadline_ms=float(model_config.get("harness_runtime", {}).get(
                    "deadline_ms", 120_000
                )),
            )),
            trace_sink=collect_trace,
        )
        summary = await run_dataset(
            harness=harness,
            adapter=adapter,
            profile=profile,
            model_hash=checkpoint_hash,
            output_dir=output_dir,
            resume=args.resume,
            concurrency=args.concurrency,
            trace_reader=traces.get,
        )
        segment["completed_at_utc"] = datetime.now(UTC).isoformat()
        segment["completed_cases"] = summary["completed_cases_this_invocation"]
        segment["wall_seconds"] = summary["run_wall_seconds"]
        segment["summary"] = summary
        total_wall_seconds = sum(float(item.get("wall_seconds", 0.0)) for item in run_manifest["execution_segments"])
        total_wall_seconds = max(total_wall_seconds, 1e-9)
        summary["run_wall_seconds"] = round(total_wall_seconds, 3)
        summary["completed_cases_this_invocation"] = sum(
            int(item.get("completed_cases", 0)) for item in run_manifest["execution_segments"]
        )
        summary["cases_per_second"] = round(summary["case_count"] / total_wall_seconds, 5)
        summary["provider_requests_per_second"] = round(
            float(summary["provider_calls_per_case"] or 0) * summary["case_count"] / total_wall_seconds,
            5,
        )
        if summary["input_tokens_per_case"] is None or summary["output_tokens_per_case"] is None:
            summary["tokens_per_second"] = None
        else:
            summary["tokens_per_second"] = round(
                (float(summary["input_tokens_per_case"]) + float(summary["output_tokens_per_case"]))
                * summary["case_count"] / total_wall_seconds,
                3,
            )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        run_manifest["status"] = "COMPLETE"
        run_manifest["completed_at_utc"] = datetime.now(UTC).isoformat()
        run_manifest["summary"] = summary
        manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def _run_manifest(args, alias, profile, model_config, runtime_config, adapter) -> dict[str, object]:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "UNKNOWN"
    implementation_files = (
        ROOT / "src/health_ai_copilot/evaluation/datasets.py",
        ROOT / "src/health_ai_copilot/evaluation/runner.py",
        ROOT / "src/health_ai_copilot/harness/runtime.py",
        ROOT / "src/health_ai_copilot/harness/verification.py",
        ROOT / "src/health_ai_copilot/providers/model.py",
        ROOT / "src/health_ai_copilot/reasoning/single.py",
        ROOT / "src/health_ai_copilot/reasoning/adaptive_mdt.py",
        ROOT / "src/health_ai_copilot/multi_agent/mdagents_style.py",
        ROOT / "tools/eval/run_common_eval.py",
        ROOT / "configs/eval/common_eval_v1.json",
        ROOT / "configs/eval/cmb_common_1024.json",
        ROOT / "configs/eval/local_prepared_views_h0.json",
        ROOT / "tools/eval/extract_failed_case_ids.py",
        ROOT / "tools/eval/audit_gold_blind_parse_failures.py",
        ROOT / "tools/eval/build_adaptive_failure_replay_manifest.py",
    )
    input_hashes = {
        "candidate_view_sha256": getattr(adapter, "candidate_sha256", None),
        "scorer_view_sha256": getattr(adapter, "scorer_sha256", None),
        "ids_manifest_sha256": getattr(adapter, "ids_manifest_sha256", None),
        "ids_sequence_sha256": getattr(
            adapter, "frozen_subset_sha256", getattr(adapter, "subset_sha256", None)
        ),
        "combined_dataset_identity_sha256": getattr(adapter, "snapshot_sha256", None),
    }
    return {
        "schema_version": "harness-v1-run-manifest",
        "git_sha": git_sha,
        "dataset": {
            "dataset_id": adapter.dataset_id,
            "source_revision": adapter.source_revision,
            "expected_case_count": len(adapter.cases()),
            "candidate_path": str(getattr(adapter, "candidate_path", args.dataset_path or "")),
            "scorer_path": str(getattr(adapter, "scorer_path", "")),
            "ids_manifest_path": str(getattr(adapter, "ids_manifest_path", args.subset_manifest)),
            "selected_case_ids_sha256": getattr(adapter, "selected_case_ids_sha256", None),
            "case_ids_file_path": str(args.case_ids_file) if args.case_ids_file else None,
            "case_ids_file_sha256": file_sha256(args.case_ids_file) if args.case_ids_file else None,
            **input_hashes,
        },
        "system_profile": {
            "profile_id": alias,
            "model_variant": profile.model_variant.value,
            "retrieval_mode": profile.retrieval_mode.value,
            "memory_mode": profile.memory_mode.value,
            "reasoning_mode": profile.reasoning_mode.value,
        },
        "model": model_config,
        "model_config_sha256": file_sha256(args.model_config),
        "implementation_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in implementation_files if path.is_file()
        },
        "decoding": {
            "temperature": model_config["serving"]["temperature"],
            "top_p": model_config["serving"].get("top_p"),
            "do_sample": model_config["serving"].get("do_sample"),
        "max_output_tokens": model_config["serving"]["max_output_tokens"],
        "chat_template_kwargs": model_config["serving"].get("chat_template_kwargs", {}),
        "structured_outputs_config": model_config["serving"].get("structured_outputs_config", {}),
        "seed": model_config["serving"].get("seed"),
        },
        "scoring_revision": ANSWER_PARSER_REVISION,
        "configured_model_serving": model_config.get("serving", {}),
        "serving": runtime_config,
        "vllm_runtime_config": runtime_config,
        "vllm_runtime_config_sha256": (
            file_sha256(args.vllm_runtime_config) if args.vllm_runtime_config else None
        ),
        "concurrency": args.concurrency,
        "capture_adaptive_diagnostics": args.capture_adaptive_diagnostics,
        "harness_runtime": model_config.get("harness_runtime", {
            "max_provider_calls": 32,
            "max_tool_calls": 16,
            "deadline_ms": 120_000,
        }),
        "retrieval_corpus": None if profile.retrieval_mode.value == "off" else "COMMON_MEDICAL_KB_V1",
    }


def _select_cases(adapter, case_ids_file: Path | None):
    if case_ids_file is None:
        return adapter
    document = json.loads(case_ids_file.read_text(encoding="utf-8"))
    if isinstance(document, list):
        case_ids = document
        declared_hash = None
    elif isinstance(document, dict):
        case_ids = document.get("case_ids")
        declared_hash = document.get("case_ids_sha256")
    else:
        raise SystemExit("case IDs file must be a JSON array or manifest object")
    if not isinstance(case_ids, list) or any(not isinstance(item, str) for item in case_ids):
        raise SystemExit("case IDs file must contain a string 'case_ids' list")
    selected_ids = tuple(case_ids)
    actual_hash = case_ids_sha256(selected_ids) if selected_ids else None
    if declared_hash is not None and declared_hash != actual_hash:
        raise SystemExit("case IDs file hash does not match its case_ids sequence")
    try:
        return SelectedCaseAdapter(adapter, selected_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
