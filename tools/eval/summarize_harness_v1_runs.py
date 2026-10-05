"""Create paired B0/B2 metrics from complete Harness checkpoints."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any


def _wilson(correct: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = correct / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    margin = z * sqrt((p * (1 - p) + z * z / (4 * total)) / total) / scale
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def _read_arm(root: Path, arm: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    directory = root / arm
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    with (directory / "cases.jsonl").open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row["case_id"])
            if case_id in rows:
                raise ValueError(f"duplicate case ID in {arm} checkpoint at line {line_no}")
            rows[case_id] = row
    if len(rows) != summary["case_count"]:
        raise ValueError(f"{arm} has {len(rows)} checkpoint rows, expected {summary['case_count']}")
    return summary, rows


def _answer_key(value: object) -> tuple[str, ...] | str | None:
    if isinstance(value, list):
        return tuple(sorted(str(item) for item in value))
    if isinstance(value, str):
        return value
    return None


def summarize(dataset_root: Path, dataset_id: str) -> dict[str, Any]:
    single_summary, single_rows = _read_arm(dataset_root, "B0")
    adaptive_summary, adaptive_rows = _read_arm(dataset_root, "B2")
    if set(single_rows) != set(adaptive_rows):
        raise ValueError("B0 and B2 case IDs differ")
    single_manifest = json.loads((dataset_root / "B0/run_manifest.json").read_text(encoding="utf-8"))
    adaptive_manifest = json.loads((dataset_root / "B2/run_manifest.json").read_text(encoding="utf-8"))
    for field in ("model_config_sha256",):
        if single_manifest.get(field) != adaptive_manifest.get(field):
            raise ValueError(f"B0/B2 {field} differs")
    if single_manifest["dataset"].get("ids_sequence_sha256") != adaptive_manifest["dataset"].get("ids_sequence_sha256"):
        raise ValueError("B0/B2 frozen ID sequence differs")
    serving_audit = dataset_root.parent.parent / "recovery_inputs/serving-parity-audit.json"

    paired = []
    both_parse = 0
    answer_agreement = 0
    both_correct = single_only_correct = adaptive_only_correct = neither_correct = 0
    per_complexity: dict[str, list[bool]] = {}
    for case_id in sorted(single_rows):
        single = single_rows[case_id]
        adaptive = adaptive_rows[case_id]
        s_correct = bool(single["score"]["correct"])
        a_correct = bool(adaptive["score"]["correct"])
        s_answer = _answer_key(single["response"].get("parsed_answer"))
        a_answer = _answer_key(adaptive["response"].get("parsed_answer"))
        if s_answer is not None and a_answer is not None:
            both_parse += 1
            answer_agreement += s_answer == a_answer
        if s_correct and a_correct:
            both_correct += 1
        elif s_correct:
            single_only_correct += 1
        elif a_correct:
            adaptive_only_correct += 1
        else:
            neither_correct += 1
        complexity = (adaptive.get("trace_summary") or {}).get("mdt_complexity") or "unclassified"
        per_complexity.setdefault(str(complexity), []).append(a_correct)
        paired.append({
            "case_id": case_id,
            "single_correct": s_correct,
            "adaptive_correct": a_correct,
            "single_parsed": s_answer,
            "adaptive_parsed": a_answer,
            "adaptive_complexity": complexity,
        })

    total = len(paired)
    single_correct = int(single_summary["correct"])
    adaptive_correct = int(adaptive_summary["correct"])
    return {
        "schema_version": "harness-v1-paired-b0-b2-v1",
        "dataset_id": dataset_id,
        "case_count": total,
        "model_manifest_sha256": single_manifest["model"].get("model_manifest_sha256"),
        "ids_sequence_sha256": single_manifest["dataset"].get("ids_sequence_sha256"),
        "serving_parity": {
            "status": "INCOMPLETE_B0_RUNTIME_CAPTURE",
            "b0_runtime_config": single_manifest.get("vllm_runtime_config"),
            "b2_runtime_config": adaptive_manifest.get("vllm_runtime_config"),
            "audit_path": str(serving_audit) if serving_audit.is_file() else None,
        },
        "single": {
            "correct": single_correct,
            "accuracy": single_correct / total,
            "wilson_95_ci": _wilson(single_correct, total),
            "parse_success": single_summary["parse_success"],
            "parse_rate": single_summary["parse_rate"],
            "safety_route_abstentions": single_summary.get("safety_route_abstentions", 0),
            "reasoning_failures": single_summary.get("reasoning_failures", 0),
            "model_abstentions": single_summary.get("model_abstentions", 0),
            "answer_format_failures": single_summary.get("answer_format_failures", 0),
            "unparsed_non_safety": single_summary.get("unparsed_non_safety", 0),
            "scoring_revision": single_summary.get("scoring_revision"),
            "provider_calls_per_case": single_summary["provider_calls_per_case"],
            "input_tokens_per_case": single_summary["input_tokens_per_case"],
            "output_tokens_per_case": single_summary["output_tokens_per_case"],
            "total_tokens_per_case": single_summary.get("total_tokens_per_case"),
            "mean_latency_ms": single_summary["mean_latency_ms"],
            "p50_latency_ms": single_summary["p50_latency_ms"],
            "p95_latency_ms": single_summary["p95_latency_ms"],
            "cases_per_second": single_summary.get("cases_per_second"),
            "provider_requests_per_second": single_summary.get("provider_requests_per_second"),
            "tokens_per_second": single_summary.get("tokens_per_second"),
        },
        "adaptive_mdt": {
            "correct": adaptive_correct,
            "accuracy": adaptive_correct / total,
            "wilson_95_ci": _wilson(adaptive_correct, total),
            "parse_success": adaptive_summary["parse_success"],
            "parse_rate": adaptive_summary["parse_rate"],
            "safety_route_abstentions": adaptive_summary.get("safety_route_abstentions", 0),
            "reasoning_failures": adaptive_summary.get("reasoning_failures", 0),
            "model_abstentions": adaptive_summary.get("model_abstentions", 0),
            "answer_format_failures": adaptive_summary.get("answer_format_failures", 0),
            "unparsed_non_safety": adaptive_summary.get("unparsed_non_safety", 0),
            "scoring_revision": adaptive_summary.get("scoring_revision"),
            "provider_calls_per_case": adaptive_summary["provider_calls_per_case"],
            "input_tokens_per_case": adaptive_summary["input_tokens_per_case"],
            "output_tokens_per_case": adaptive_summary["output_tokens_per_case"],
            "total_tokens_per_case": adaptive_summary.get("total_tokens_per_case"),
            "mean_latency_ms": adaptive_summary["mean_latency_ms"],
            "p50_latency_ms": adaptive_summary["p50_latency_ms"],
            "p95_latency_ms": adaptive_summary["p95_latency_ms"],
            "cases_per_second": adaptive_summary.get("cases_per_second"),
            "provider_requests_per_second": adaptive_summary.get("provider_requests_per_second"),
            "tokens_per_second": adaptive_summary.get("tokens_per_second"),
            "mdt_complexity_distribution": adaptive_summary["mdt_complexity_distribution"],
            "accuracy_by_complexity": {
                name: {"n": len(values), "correct": sum(values), "accuracy": mean(values)}
                for name, values in sorted(per_complexity.items())
            },
        },
        "delta_adaptive_minus_single": {
            "correct": adaptive_correct - single_correct,
            "accuracy_percentage_points": round(100 * (adaptive_correct - single_correct) / total, 4),
        },
        "paired_correctness": {
            "both_correct": both_correct,
            "single_only_correct": single_only_correct,
            "adaptive_only_correct": adaptive_only_correct,
            "neither_correct": neither_correct,
        },
        "answer_agreement": {
            "both_parsed": both_parse,
            "matching_parsed_answers": answer_agreement,
            "rate_among_both_parsed": answer_agreement / both_parse if both_parse else None,
            "rate_over_all_cases": answer_agreement / total,
        },
        "case_results": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.dataset_root, args.dataset_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(result, args.output_dir / "report.md")
    print(json.dumps({key: result[key] for key in (
        "dataset_id", "case_count", "single", "adaptive_mdt", "delta_adaptive_minus_single",
        "paired_correctness", "answer_agreement",
    )}, ensure_ascii=False, indent=2))


def _write_report(result: dict[str, Any], path: Path) -> None:
    single = result["single"]
    adaptive = result["adaptive_mdt"]
    delta = result["delta_adaptive_minus_single"]["accuracy_percentage_points"]
    lines = [
        f"# Harness V1 paired result — {result['dataset_id']}",
        "",
        f"Frozen cases: {result['case_count']}  ",
        f"Model manifest SHA256: `{result['model_manifest_sha256']}`  ",
        f"ID sequence SHA256: `{result['ids_sequence_sha256']}`",
        "",
        "| Metric | B0 Single | B2 Adaptive MDT | Delta |",
        "|---|---:|---:|---:|",
        f"| Accuracy | {single['accuracy']:.2%} | {adaptive['accuracy']:.2%} | {delta:+.2f} pp |",
        f"| 95% Wilson CI | {_interval(single['wilson_95_ci'])} | {_interval(adaptive['wilson_95_ci'])} | — |",
        f"| Parse success | {single['parse_rate']:.2%} | {adaptive['parse_rate']:.2%} | — |",
        f"| Safety-route abstentions | {single['safety_route_abstentions']} | {adaptive['safety_route_abstentions']} | — |",
        f"| Reasoning failures | {single['reasoning_failures']} | {adaptive['reasoning_failures']} | — |",
        f"| Explicit model abstentions | {single['model_abstentions']} | {adaptive['model_abstentions']} | — |",
        f"| Answer-format failures | {single['answer_format_failures']} | {adaptive['answer_format_failures']} | — |",
        f"| Provider calls / case | {single['provider_calls_per_case']:.2f} | {adaptive['provider_calls_per_case']:.2f} | — |",
        f"| Input tokens / case | {single['input_tokens_per_case']:.1f} | {adaptive['input_tokens_per_case']:.1f} | — |",
        f"| Output tokens / case | {single['output_tokens_per_case']:.1f} | {adaptive['output_tokens_per_case']:.1f} | — |",
        f"| Total tokens / case | {single['total_tokens_per_case']:.1f} | {adaptive['total_tokens_per_case']:.1f} | — |",
        f"| Mean latency / case | {single['mean_latency_ms']:.1f} ms | {adaptive['mean_latency_ms']:.1f} ms | — |",
        f"| P50 latency / case | {single['p50_latency_ms']:.1f} ms | {adaptive['p50_latency_ms']:.1f} ms | — |",
        f"| P95 latency / case | {single['p95_latency_ms']:.1f} ms | {adaptive['p95_latency_ms']:.1f} ms | — |",
        f"| Throughput | {single['cases_per_second']:.3f} cases/s | {adaptive['cases_per_second']:.3f} cases/s | — |",
        f"| Provider throughput | {single['provider_requests_per_second']:.3f} req/s | {adaptive['provider_requests_per_second']:.3f} req/s | — |",
        f"| Token throughput | {single['tokens_per_second']:.1f} tok/s | {adaptive['tokens_per_second']:.1f} tok/s | — |",
        "",
        f"Parsed answer agreement: {result['answer_agreement']['matching_parsed_answers']}/{result['answer_agreement']['both_parsed']} among cases parsed by both arms.",
        f"Scoring revision: `{single['scoring_revision']}`.",
        "",
        f"Serving parity is qualified as `{result['serving_parity']['status']}`; see `{result['serving_parity']['audit_path'] or 'the run manifests'}`. Treat this paired delta as descriptive, not a controlled causal estimate.",
        "",
        "## Adaptive complexity slices",
        "",
        "| Complexity | N | Correct | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for name, value in adaptive["accuracy_by_complexity"].items():
        lines.append(f"| {name} | {value['n']} | {value['correct']} | {value['accuracy']:.2%} |")
    lines.extend([
        "",
        "Scorer labels were joined by frozen case ID on the evaluator side. Only candidate prompts and answer schema were passed to Harness. Parse failures and Harness safety abstentions remain incorrect in the accuracy denominator.",
        "",
        "Adaptive MDT here is the full complexity-routed strategy; B2−B0 does not isolate the moderator by itself.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _interval(value: list[float] | None) -> str:
    return "—" if value is None else f"{value[0]:.2%}–{value[1]:.2%}"


if __name__ == "__main__":
    main()
