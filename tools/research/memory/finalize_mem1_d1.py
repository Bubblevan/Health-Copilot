"""Validate and package the frozen MEM-1D1 diagnostic for human review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import run_mem1
from context_bundle import verify_context_bundle
from mem1_artifacts import read_jsonl, verify_hash_sidecar

KNOWN_CASE = "1cea1afa"
REPORT_PATH = ROOT / "docs/research/memory/mem_1d1_frozen_10_case_diagnostic.md"
REVIEW_PATH = ROOT / "docs/research/memory/mem_1d1_case_review.json"


def _recall_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    groups = row.get("ranked_session_groups")
    answer_sessions = row.get("answer_session_ids") or []
    if (
        row.get("session_provenance_available") is False
        or not isinstance(groups, list)
        or not answer_sessions
    ):
        return {"recall_at_5": None, "recall_at_10": None, "mrr": None}
    expected = {str(value) for value in answer_sessions}
    reciprocal_rank = None
    retrieved = []
    for rank, group in enumerate(groups, 1):
        matched = expected.intersection(str(value) for value in group)
        retrieved.append({str(value) for value in group})
        if matched and reciprocal_rank is None:
            reciprocal_rank = 1.0 / rank
    recalls = []
    for cutoff in (5, 10):
        found = set().union(*retrieved[:cutoff]) if retrieved[:cutoff] else set()
        recalls.append(len(found & expected) / len(expected))
    return {
        "recall_at_5": recalls[0],
        "recall_at_10": recalls[1],
        "mrr": reciprocal_rank or 0.0,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "token_f1": _mean(rows, "f1"),
        "token_precision": _mean(rows, "token_precision"),
        "token_recall": _mean(rows, "token_recall"),
        "normalized_exact_match": _mean(rows, "normalized_exact_match"),
    }


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}"


def _write_or_verify(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite differing review artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def finalize(run_dir: Path) -> tuple[Path, Path]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection_manifest_sha256") != run_mem1.D1_SELECTION_SHA256:
        raise RuntimeError("Run is not bound to the canonical MEM-1D1 selection manifest")
    systems = list(run_mem1.SYSTEMS)
    question_ids = list(run_mem1.D1_QUESTION_IDS)
    run_mem1._verify_frozen_evidence(run_dir, require_ledgers=True)

    prediction_rows = read_jsonl(run_dir / "predictions.jsonl")
    predictions = run_mem1._latest_predictions(prediction_rows)
    expected = {(system, qid) for system in systems for qid in question_ids}
    if set(predictions) != expected or len(predictions) != 50:
        raise RuntimeError("MEM-1D1 requires exactly 50 terminal prediction rows")
    if any(predictions[key].get("quality_status") != "OK" for key in expected):
        raise RuntimeError("MEM-1D1 predictions contain an infrastructure failure")

    bundles = run_mem1._latest_predictions([
        {
            "system": row.get("system"),
            "question_id": row.get("question_id"),
            "context_bundle": row.get("context_bundle"),
        }
        for row in read_jsonl(run_dir / "context_bundles.jsonl")
    ])
    if set(bundles) != expected:
        raise RuntimeError("MEM-1D1 requires 50 ContextBundles")
    for key, row in bundles.items():
        bundle = row.get("context_bundle")
        prediction = predictions[key]
        if (
            not isinstance(bundle, dict)
            or not verify_context_bundle(bundle)
            or bundle.get("context_bundle_sha256") != prediction.get("context_bundle_sha256")
        ):
            raise RuntimeError(f"ContextBundle integrity/binding failure for {key}")

    call_rows = read_jsonl(run_dir / "call_ledger.jsonl")
    warning_rows = read_jsonl(run_dir / "baseline_warnings.jsonl")
    memory_roles = manifest["roles"]["memory_system"]["systems"]
    context_valid = run_mem1._context_bundle_rows_valid(
        run_dir,
        memory_roles,
        question_ids,
        predictions,
        freeze=False,
    )
    fullcontext_valid = run_mem1._fullcontext_is_valid(predictions, question_ids, call_rows)
    run_mem1._validate_mem1d1_completion(
        manifest,
        systems,
        question_ids,
        predictions,
        call_rows,
        warning_rows,
        fullcontext_valid=fullcontext_valid,
        context_valid=context_valid,
    )

    for artifact, sidecar in (
        ("predictions.jsonl", "predictions.sha256"),
        ("context_bundles.jsonl", "context_bundles.sha256"),
        ("call_ledger.jsonl", "call_ledger.sha256"),
        ("baseline_warnings.jsonl", "baseline_warnings.sha256"),
    ):
        if not verify_hash_sidecar(run_dir / artifact, run_dir / sidecar):
            raise RuntimeError(f"MEM-1D1 frozen evidence hash failed: {artifact}")

    metrics_path = run_dir / "deterministic_metrics.json"
    efficiency_path = run_dir / "token_efficiency.json"
    embeddings_path = run_dir / "embeddings_usage.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
    embeddings = json.loads(embeddings_path.read_text(encoding="utf-8"))
    for system in systems:
        if metrics["systems"][system]["quality_n"] != 10:
            raise RuntimeError(f"Deterministic metrics are incomplete for {system}")

    case_rows = []
    per_system: dict[str, list[dict[str, Any]]] = {system: [] for system in systems}
    for system in systems:
        for question_id in question_ids:
            prediction = predictions[(system, question_id)]
            memory = run_mem1._latest_predictions([
                {
                    "system": system,
                    "question_id": question_id,
                    **_recall_metrics(prediction),
                }
            ])[(system, question_id)]
            row = {
                "system": system,
                "question_id": question_id,
                "category": prediction["category"],
                "case_status": "KNOWN_GATE_CASE" if question_id == KNOWN_CASE else "FRESH_DIAGNOSTIC_CASES",
                "question": prediction["question"],
                "gold": prediction["ground_truth"],
                "prediction": prediction["predicted"],
                "token_metrics": {
                    "precision": prediction["token_precision"],
                    "recall": prediction["token_recall"],
                    "f1": prediction["f1"],
                    "normalized_exact_match": prediction["normalized_exact_match"],
                },
                "context_bundle_sha256": prediction["context_bundle_sha256"],
                "context_reader_tokens": prediction["context_reader_tokens"],
                "context_embedding_tokens": prediction["context_embedding_tokens"],
                "reader_prompt_tokens": prediction["reader_prompt_tokens"],
                "answer_session_recall_at_5": memory["recall_at_5"],
                "answer_session_recall_at_10": memory["recall_at_10"],
                "answer_session_mrr": memory["mrr"],
                "session_provenance_available": prediction["session_provenance_available"],
                "answer_session_ids": prediction.get("answer_session_ids", []),
                "ranked_source_session_groups": prediction.get("ranked_session_groups"),
                "retrieval_latency_ms": prediction["retrieval_latency_ms"],
                "ingestion_latency_ms": prediction["ingestion_latency_ms"],
                "baseline_warning_types": sorted({
                    warning.get("warning_type")
                    for warning in prediction.get("baseline_warnings", [])
                }),
                "failure_attribution_hint": prediction.get("failure_attribution_hint", []),
                "failure_attribution_heuristic": prediction.get("failure_attribution_heuristic") is True,
                "memory_system_diagnostics": prediction.get("memory_system_diagnostics"),
            }
            case_rows.append(row)
            per_system[system].append(prediction)

    all_fresh_summaries = {}
    for system in systems:
        all_rows = per_system[system]
        fresh_rows = [row for row in all_rows if row["question_id"] != KNOWN_CASE]
        all_fresh_summaries[system] = {
            "all_10": _summary(all_rows),
            "fresh_9": _summary(fresh_rows),
        }

    category_counts = {
        category: sum(
            predictions[(system, qid)]["category"] == category
            for system, qid in expected
        ) // len(systems)
        for category in run_mem1.CATEGORIES
    }
    review = {
        "artifact_version": "mem1d1-case-review-v1",
        "run_id": manifest["run_id"],
        "selection_manifest_sha256": manifest["selection_manifest_sha256"],
        "dataset_revision": manifest["dataset"]["revision"],
        "dataset_sha256": manifest["dataset"]["sha256"],
        "test_access": False,
        "systems": systems,
        "question_ids": question_ids,
        "known_case_id": KNOWN_CASE,
        "fresh_case_ids": [qid for qid in question_ids if qid != KNOWN_CASE],
        "category_counts": category_counts,
        "rows": case_rows,
    }
    review_text = json.dumps(review, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    _write_or_verify(REVIEW_PATH, review_text)

    lines = [
        "# MEM-1D1 Frozen 10-Case Diagnostic",
        "",
        f"- Run: `{manifest['run_id']}`",
        "- Gate: `MEM1_FROZEN_10_CASE_DIAGNOSTIC=YES`",
        f"- Dataset: `{manifest['dataset']['id']}` revision `{manifest['dataset']['revision']}`",
        f"- Dataset SHA256: `{manifest['dataset']['sha256']}`",
        f"- Selection SHA256: `{manifest['selection_manifest_sha256']}`",
        f"- Prediction SHA256: `{(run_dir / 'predictions.sha256').read_text(encoding='ascii').split()[0]}`",
        f"- Matrix: `{len(systems)} systems x {len(question_ids)} questions = {len(case_rows)} predictions`",
        "- TEST access: `false`; hosted calls: `0`; judge calls: `0`; native answer head: `not run`",
        "- Case split: `1cea1afa` is KNOWN_GATE_CASE; the other nine are FRESH_DIAGNOSTIC_CASES.",
        "- The manifest contains no abstention cases; no abstention-performance claim is made.",
        "- Token F1 is deterministic lexical overlap, not semantic correctness. This small diagnostic is not a product ranking or benchmark claim.",
        f"- Post-run finalization repair (`run_mem1.py` SHA256 `{run_mem1.sha256_file(run_mem1.__file__)}`) separated ContextBundle content validation from SHA-sidecar freezing. It did not modify the 50 prediction, bundle, call-ledger, or warning-ledger rows and made no model calls; it only validated/froze those artifacts and regenerated deterministic summaries.",
        "",
        "## Frozen Evidence",
        "",
        "| Artifact | SHA256 |",
        "|---|---|",
    ]
    for name in (
        "context_bundles.jsonl",
        "predictions.jsonl",
        "call_ledger.jsonl",
        "baseline_warnings.jsonl",
    ):
        lines.append(
            f"| `{name}` | `{(run_dir / (name.replace('.jsonl', '.sha256'))).read_text(encoding='ascii').split()[0]}` |"
        )
    lines.extend(["", "## All 10 And Fresh 9", "", "| System | Cases | Token F1 | Precision | Recall | Norm. EM |", "|---|---|---:|---:|---:|---:|"])
    for system in systems:
        for label in ("all_10", "fresh_9"):
            summary = all_fresh_summaries[system][label]
            lines.append(
                f"| {system} | {label} (n={summary['n']}) | {_fmt(summary['token_f1'])} | "
                f"{_fmt(summary['token_precision'])} | {_fmt(summary['token_recall'])} | "
                f"{_fmt(summary['normalized_exact_match'])} |"
            )
    lines.extend(["", "## Per-Category", "", "| System | Category | N | Token F1 | Precision | Recall | Norm. EM |", "|---|---|---:|---:|---:|---:|---:|"])
    for system in systems:
        for category in run_mem1.CATEGORIES:
            summary = metrics["systems"][system]["by_category"][category]
            lines.append(
                f"| {system} | {category} | {summary['n']} | {_fmt(summary['token_f1'])} | "
                f"{_fmt(summary['token_precision'])} | {_fmt(summary['token_recall'])} | "
                f"{_fmt(summary['normalized_exact_match'])} |"
            )
    lines.extend(["", "## Memory And Context Diagnostics", "", "| System | Recall@5 | Recall@10 | MRR | Provenance coverage | Reader context tokens | Embedding tokens | Reader prompt tokens | Retrieval ms | Ingestion ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for system in systems:
        diag = metrics["memory_diagnostics"][system]
        available = sum(
            predictions[(system, qid)].get("session_provenance_available") is True
            for qid in question_ids
        )
        lines.append(
            f"| {system} | {_fmt(diag['answer_session_recall_at_5'])} | "
            f"{_fmt(diag['answer_session_recall_at_10'])} | {_fmt(diag['mrr'])} | "
            f"{available}/10 | {_fmt(diag['context_reader_tokens'])} | "
            f"{_fmt(diag['context_embedding_tokens'])} | {_fmt(diag['reader_prompt_tokens'])} | "
            f"{_fmt(diag['retrieval_latency_ms'])} | {_fmt(diag['ingestion_latency_ms'])} |"
        )
    lines.extend([
        "",
        "Category counts (fixed manifest, no rebalancing): "
        + ", ".join(f"{key}={value}" for key, value in category_counts.items()),
        "",
        "FullContext session recall/MRR represent trivial full-history context coverage, not retrieval performance. Null retrieval metrics reflect unavailable source-session provenance, not zero.",
        "",
        "## SimpleMem Fidelity",
        "",
    ])
    simplemem_traces = [
        predictions[("simplemem", qid)].get("memory_system_diagnostics", {})
        for qid in question_ids
    ]
    first_trace = simplemem_traces[0]
    lines.extend([
        f"Official source: `{first_trace.get('source_tag')}` / `{first_trace.get('source_commit')}`.",
        f"Planning enabled for all cases: `{all(trace.get('planning_enabled') is True for trace in simplemem_traces)}`; reflection settings: `{sorted({str(trace.get('reflection_enabled')) for trace in simplemem_traces})}`.",
        "Per-question traces are retained in `predictions.jsonl` and `mem_1d1_case_review.json`; no individual lexical/structured result count is required to be nonzero.",
        "",
        "| Semantic | Keyword | Structured | Merge/deduplicate | Reflection | Native answer head calls all suppressed |",
        "|---:|---:|---:|---:|---:|---|",
        "| " + " | ".join(
            str(sum((trace.get("calls") or {}).get(key, 0) for trace in simplemem_traces))
            for key in ("semantic", "keyword", "structured", "merge_deduplicate")
        ) + " | " + str(sum((trace.get("calls") or {}).get("reflection", 0) for trace in simplemem_traces))
        + f" | {all(trace.get('native_answer_head_invoked') is False for trace in simplemem_traces)} |",
        "",
        "## Local Usage And Warnings",
        "",
        "| System | Reader calls | Qwen local ms | Embedding calls | Embedding tokens | Baseline-internal warnings | Infra failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    warning_summary = metrics["baseline_warnings"]
    for system in systems:
        roles = efficiency[system]["by_role"]
        warning = warning_summary[system]
        lines.append(
            f"| {system} | {roles['reader_answer']['calls']} | "
            f"{_fmt(efficiency[system]['local_reader_wall_ms'])} | "
            f"{roles['embedding']['calls']} | "
            f"{roles['embedding']['prompt_tokens'] or 0} | "
            f"{warning['baseline_internal_warning_count']} | {warning['infra_failure_count']} |"
        )
    lines.extend([
        "",
        f"Embedding accounting is local `{embeddings['model']}` ({embeddings['provider']}); hosted API spend is `$0`.",
        "Baseline-internal recoveries are reported separately from infrastructure failures. No causal failure labels were added; `failure_attribution_hint` remains heuristic.",
        "",
        f"Case-level review artifact: `{REVIEW_PATH.relative_to(ROOT).as_posix()}`.",
        f"Run artifact directory: `{run_dir}`.",
        "",
        "No local Qwen judge, 102-DEV run, TEST, native-answer track, M10-Flat, RevMem, Memora/FAMA, LongMemEval-V2, SFT or RL was run.",
    ])
    report_text = "\n".join(lines) + "\n"
    _write_or_verify(REPORT_PATH, report_text)
    run_report = run_dir / "report.md"
    run_report.write_text(report_text, encoding="utf-8", newline="\n")
    return REPORT_PATH, REVIEW_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report, review = finalize(args.run_dir.resolve())
    print("MEM1_FROZEN_10_CASE_DIAGNOSTIC=YES")
    print(f"report={report}")
    print(f"case_review={review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
