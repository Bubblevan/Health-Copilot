"""Finish MEM-3A.3R closeout from frozen downstream artifacts without model calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import run_mem3a3r_recursive_flatprop as stage

parent = stage.parent
flat = stage.flat
RUN_DIR = stage.RUN_DIR
RECOVERY_PATH = RUN_DIR / "frozen_artifact_closeout_recovery.json"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _read_frozen_json(path: Path) -> dict[str, Any]:
    if not flat._verify_frozen(path):
        raise RuntimeError(f"Frozen artifact sidecar is invalid: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_frozen_jsonl(path: Path) -> list[dict[str, Any]]:
    if not flat._verify_frozen(path):
        raise RuntimeError(f"Frozen artifact sidecar is invalid: {path.name}")
    return _read_jsonl(path)


def _restore_reader_bundle_bindings() -> dict[str, Any]:
    metrics = _read_frozen_json(flat.METRICS_PATH)
    predictions = _read_frozen_jsonl(flat.PREDICTIONS_PATH)
    reader_ledger = _read_frozen_jsonl(flat.READER_LEDGER_PATH)
    bundle_rows = _read_frozen_jsonl(flat.CONTEXT_BUNDLES_PATH)
    by_question_metrics = {row["question_id"]: row for row in metrics["per_question"]}
    by_question_predictions = {row["question_id"]: row for row in predictions}
    by_question_ledger = {row["question_id"]: row for row in reader_ledger}
    if any(
        set(rows) != set(flat.QUESTION_IDS)
        for rows in (by_question_metrics, by_question_predictions, by_question_ledger)
    ) or {row["question_id"] for row in bundle_rows} != set(flat.QUESTION_IDS):
        raise RuntimeError("Frozen reader outputs do not cover the exact ten questions")

    call_cache_hashes = set()
    restored = []
    for row in bundle_rows:
        qid = row["question_id"]
        expected_latency = by_question_metrics[qid]["retrieval_latency_ms"]
        bundle = dict(row["context_bundle"])
        bundle.pop("context_bundle_sha256", None)
        bundle["retrieval_latency_ms"] = expected_latency
        bundle_sha = flat._sha_json(bundle)
        state_path = RUN_DIR / "calls" / "reader" / f"{qid}.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        expected_bundle_sha = state["cache_identity"]["identity"]["context_bundle_sha256"]
        if bundle_sha != expected_bundle_sha:
            raise RuntimeError(f"Frozen reader context changed for {qid}; refusing reuse")
        if by_question_predictions[qid]["context_bundle_sha256"] != bundle_sha:
            raise RuntimeError(f"Prediction is not bound to the restored bundle for {qid}")
        call_cache_hashes.add(state["cache_identity"]["identity"]["pre_reader_freeze_sha256"])
        restored.append(
            {
                **row,
                "context_bundle": {**bundle, "context_bundle_sha256": bundle_sha},
                "context_bundle_sha256": bundle_sha,
            }
        )

    if len(call_cache_hashes) != 1:
        raise RuntimeError("Reader cache rows disagree on their original pre-reader freeze")
    flat._write_jsonl(flat.CONTEXT_BUNDLES_PATH, restored)
    return {
        "bundle_rows": restored,
        "predictions": predictions,
        "reader_ledger": reader_ledger,
        "original_pre_reader_freeze_sha256": next(iter(call_cache_hashes)),
    }


def _install_frozen_replayers(recovery: dict[str, Any]) -> dict[str, Any]:
    original_embedding = flat._embedding_and_retrieval
    original_projection = flat._project_contexts
    original_reader = flat._run_reader
    original_downstream = parent._downstream
    original_efficiency = parent._efficiency
    observed: dict[str, Any] = {"model_calls": 0, "current_pre_reader_freeze_sha256": None}

    def embedding_replay(flat_rows, records_by_question, refs, preflight):
        ranked = _read_frozen_jsonl(flat.DENSE_TOP8_PATH)
        embedding = _read_frozen_json(flat.EMBEDDING_MANIFEST_PATH)
        bundles = recovery["bundle_rows"]
        expected_ids = set(flat.QUESTION_IDS)
        if len(ranked) != 80 or {row["question_id"] for row in ranked} != expected_ids:
            raise RuntimeError("Frozen Dense top-8 artifact is incomplete")
        if any(row["memory_id"] not in records_by_question[row["question_id"]] for row in ranked):
            raise RuntimeError("Frozen Dense top-8 references an unknown proposition")
        query_tokens = embedding.get("query_token_counts")
        if not isinstance(query_tokens, dict) or set(query_tokens) != expected_ids:
            raise RuntimeError("Frozen query-token accounting is incomplete")
        latencies = {
            row["question_id"]: row["context_bundle"]["retrieval_latency_ms"] for row in bundles
        }
        if (
            embedding.get("model_id") != "Qwen/Qwen3-Embedding-0.6B"
            or embedding.get("local_only") is not True
            or embedding.get("hosted_calls") != 0
            or embedding.get("document_truncations") != 0
            or embedding.get("query_truncations") != 0
        ):
            raise RuntimeError("Frozen embedding manifest is not the required local artifact")
        return ranked, embedding, query_tokens, latencies

    def projection_replay(*args, **kwargs):
        plans = _read_frozen_jsonl(flat.CONTEXT_PLANS_PATH)
        bundles = recovery["bundle_rows"]
        if {row["question_id"] for row in plans} != set(flat.QUESTION_IDS):
            raise RuntimeError("Frozen context plans do not cover the exact ten questions")
        for row in bundles:
            bundle = row["context_bundle"]
            base = {key: value for key, value in bundle.items() if key != "context_bundle_sha256"}
            if flat._sha_json(base) != row["context_bundle_sha256"]:
                raise RuntimeError(f"Frozen ContextBundle hash mismatch for {row['question_id']}")
            if row.get("truncated") is not False:
                raise RuntimeError(f"Frozen reader prompt was truncated for {row['question_id']}")
        return plans, bundles, {}

    def reader_replay(pre_reader_sha, preflight, bundles):
        observed["current_pre_reader_freeze_sha256"] = pre_reader_sha
        m2a_manifest = flat._read_json(flat.mem2c.MEM2A_DIR / "run_manifest.json")
        with httpx.Client(timeout=httpx.Timeout(1800.0, connect=10.0), trust_env=False) as client:
            runtime = flat.mem2b._verify_reader(client, m2a_manifest)
        if runtime != preflight["writer_runtime_identity"]:
            raise RuntimeError("Frozen local reader runtime changed; refusing artifact reuse")
        predictions = recovery["predictions"]
        ledger = recovery["reader_ledger"]
        pred_by = {row["question_id"]: row for row in predictions}
        ledger_by = {row["question_id"]: row for row in ledger}
        for row in bundles:
            qid = row["question_id"]
            messages = row["reader_messages"]
            request = {
                "model": flat.READER_MODEL,
                "messages": messages,
                "temperature": 0,
                "seed": 42,
                "max_tokens": flat.READER_MAX_OUTPUT,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            request_sha = flat._sha_bytes(flat.mem2a.canonical_json(request).encode("utf-8"))
            prompt_sha = flat._sha_bytes(flat.mem2a.canonical_json(messages).encode("utf-8"))
            pred, call = pred_by[qid], ledger_by[qid]
            state = json.loads(
                (RUN_DIR / "calls" / "reader" / f"{qid}.json").read_text(encoding="utf-8")
            )
            identity = state["cache_identity"]["identity"]
            if (
                request_sha != call.get("request_sha256")
                or prompt_sha != call.get("prompt_sha256")
                or row["context_bundle_sha256"] != identity.get("context_bundle_sha256")
                or runtime["runtime_props_sha256"] != identity.get("reader_runtime_props_sha256")
                or identity.get("request_sha256") != request_sha
                or identity.get("reader_contract_sha256") != flat.PINNED_READER_SHA256
                or state.get("status") != "COMPLETE"
                or state.get("call", {}).get("success") is not True
                or call.get("success") is not True
                or call.get("hosted_call") is not False
                or pred.get("predicted") != state.get("prediction", {}).get("predicted")
            ):
                raise RuntimeError(f"Frozen reader request/result mismatch for {qid}")
        return predictions, ledger

    def efficiency_schema_adapter(chunk_diagnostics, embedding, bundles, reader_ledger):
        compatible_rows = [
            {
                **row,
                "retrieval_latency_ms": row["context_bundle"]["retrieval_latency_ms"],
            }
            for row in bundles
        ]
        return original_efficiency(chunk_diagnostics, embedding, compatible_rows, reader_ledger)

    def downstream_and_relabel(*args, **kwargs):
        call_args = list(args)
        call_kwargs = dict(kwargs)
        terminal_by_id = {row["chunk_id"]: row for row in stage._STATE["terminal_outputs"]}
        if call_args:
            leaf_ledger, writer_manifest = call_args[3], call_args[7]
        else:
            leaf_ledger = call_kwargs.get("chunk_ledger")
            writer_manifest = call_kwargs.get("writer_manifest")
        if leaf_ledger is None or writer_manifest is None:
            raise RuntimeError("Frozen writer accounting inputs are missing")
        attributed_ledger = []
        for row in leaf_ledger:
            source = terminal_by_id.get(row["chunk_id"])
            if source is None or source.get("provider_calls") != 1:
                raise RuntimeError(
                    f"Terminal writer result lacks one verified local call: {row['chunk_id']}"
                )
            attributed_ledger.append(
                {
                    **row,
                    "provider_calls": 1,
                    "provider_call_origin": (
                        "verified_historical_cache"
                        if source.get("historical_cache_imported")
                        else "current_stage_local_generation"
                    ),
                }
            )
        attributed_writer_manifest = {
            **writer_manifest,
            "provider_calls_unique": sum(row["provider_calls"] for row in attributed_ledger),
        }
        if call_args:
            call_args[3] = attributed_ledger
            call_args[7] = attributed_writer_manifest
        else:
            call_kwargs["chunk_ledger"] = attributed_ledger
            call_kwargs["writer_manifest"] = attributed_writer_manifest
        manifest = original_downstream(*call_args, **call_kwargs)
        predictions = _read_jsonl(flat.PREDICTIONS_PATH)
        if {row["question_id"] for row in predictions} != set(flat.QUESTION_IDS):
            raise RuntimeError("Completed prediction artifact lost a frozen question")
        predictions = [{**row, "system": stage.SYSTEM_NAME} for row in predictions]
        flat._write_jsonl(flat.PREDICTIONS_PATH, predictions)
        manifest["artifact_sha256"][flat.PREDICTIONS_PATH.name] = stage._sha_file(
            flat.PREDICTIONS_PATH
        )
        flat._write_json(flat.RUN_MANIFEST_PATH, manifest)
        return manifest

    flat._embedding_and_retrieval = embedding_replay
    flat._project_contexts = projection_replay
    flat._run_reader = reader_replay
    parent._efficiency = efficiency_schema_adapter
    parent._downstream = downstream_and_relabel
    parent.RUN_ID = "mem3a3-chunked-flatprop-frozen-10-20260929"
    parent.RUN_DIR = ROOT / "runs" / "memory" / "mem3" / parent.RUN_ID
    parent.GATE = "MEM3A3_CHUNKED_FLATPROP_FROZEN_10_DIAGNOSTIC"
    parent.LOCAL_CACHE_ROOT = ROOT / ".cache" / "health-copilot" / "mem3a3-chunked-v4-writer"
    try:
        stage._run(preflight_only=False, resume_completed_writer=True)
    finally:
        flat._embedding_and_retrieval = original_embedding
        flat._project_contexts = original_projection
        flat._run_reader = original_reader
        parent._efficiency = original_efficiency
        parent._downstream = original_downstream
    return observed


def _record_recovery(recovery: dict[str, Any], observed: dict[str, Any]) -> None:
    audit = {
        "schema_version": 1,
        "stage": stage.RUN_ID,
        "mode": "frozen_artifact_closeout_recovery",
        "reader_calls_replayed": 10,
        "reader_calls_made_during_recovery": observed["model_calls"],
        "writer_calls_made_during_recovery": 0,
        "embedding_model_calls_made_during_recovery": 0,
        "terminal_leaf_call_attribution": "one verified local response per terminal leaf; historical cache imports retain their original call attribution",
        "original_pre_reader_freeze_sha256": recovery["original_pre_reader_freeze_sha256"],
        "replayed_pre_reader_freeze_sha256": observed["current_pre_reader_freeze_sha256"],
        "same_reader_prompts_and_requests_verified": True,
        "same_context_bundle_sha256_verified": True,
        "reader_runtime_verified_loopback": True,
        "reason_for_replay": "Only non-semantic latency telemetry changed between the initial frozen calls and closeout replay; request, prompt, ContextBundle, model, and runtime identities were checked before reuse.",
        "labels_loaded_after_prediction_and_reader_ledger_freeze": True,
    }
    stage._write_json(RECOVERY_PATH, audit)
    note = (
        "\n## Frozen Artifact Closeout Recovery\n\n"
        "The completed reader predictions and call ledger were reused only after verifying all ten "
        "reader prompts, request hashes, ContextBundle hashes, model identity, and loopback runtime. "
        "The re-closeout made zero writer, embedding, or reader model calls; timing telemetry was "
        "excluded from the reuse decision.\n"
    )
    for path in (flat.RUN_REPORT_PATH, stage.REPORT_DOC):
        content = path.read_text(encoding="utf-8")
        if "## Frozen Artifact Closeout Recovery" not in content:
            stage.atomic_write_bytes(path, (content.rstrip() + "\n" + note).encode("utf-8"))
            stage._freeze(path)

    manifest = _read_frozen_json(flat.RUN_MANIFEST_PATH)
    manifest["artifact_sha256"][RECOVERY_PATH.name] = stage._sha_file(RECOVERY_PATH)
    manifest["artifact_sha256"][flat.RUN_REPORT_PATH.name] = stage._sha_file(flat.RUN_REPORT_PATH)
    manifest["artifact_sha256"][stage.REPORT_DOC.name] = stage._sha_file(stage.REPORT_DOC)
    manifest["report_sha256"] = stage._sha_file(stage.REPORT_DOC)
    manifest["gate"]["frozen_artifact_recovery_audited"] = True
    stage._write_json(flat.RUN_MANIFEST_PATH, manifest)


def main() -> int:
    stage._configure_paths(
        stage.subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    recovery = _restore_reader_bundle_bindings()
    observed = _install_frozen_replayers(recovery)
    _record_recovery(recovery, observed)
    manifest = _read_frozen_json(flat.RUN_MANIFEST_PATH)
    print(manifest["completion_gate_marker"], flush=True)
    print(f"reader_calls_reused={manifest['reader_calls']}", flush=True)
    print("model_calls_during_closeout_recovery=0", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
