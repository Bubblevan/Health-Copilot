#!/usr/bin/env python3
"""Gold-plane paired analysis for the final B0/B2 no-RAG runs."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = Path(__file__).resolve().parent
B0_CMB = ROOT / "runs/common_eval/h1-final-factorial-20261005/full/cmb-common-1024/B0"
B0_DA = RUN_ROOT / "diagnosisarena-915/B0"
ADAPTIVE_ONLY_PATCH = "src/health_ai_copilot/multi_agent/mdagents_style.py"


def read_arm(path: Path, expected: int) -> tuple[dict, dict[str, dict], dict[str, str | None]]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"{path}: run is not COMPLETE")
    cases: dict[str, dict] = {}
    with (path / "cases.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            case_id = str(row["case_id"])
            if case_id in cases:
                raise RuntimeError(f"{path}: duplicate case id {case_id}")
            cases[case_id] = row
    if len(cases) != expected:
        raise RuntimeError(f"{path}: expected {expected} cases, found {len(cases)}")
    statuses: dict[str, str | None] = {}
    with (path / "traces.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            completed = [
                event.get("fields", {}).get("status")
                for event in trace.get("events", [])
                if event.get("kind") == "request_completed"
            ]
            statuses[str(trace.get("trace_id"))] = str(completed[-1]) if completed else None
    return manifest, cases, statuses


def exact_mcnemar(b0_only: int, b2_only: int) -> float | None:
    discordant = b0_only + b2_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, k) for k in range(min(b0_only, b2_only) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * lower_tail)


def arm_metrics(manifest: dict, cases: dict[str, dict], statuses: dict[str, str | None]) -> dict:
    records = list(cases.values())
    status_counts = Counter(statuses.get(str(row["response"].get("trace_id"))) for row in records)
    parse_ok = sum(bool(row.get("score", {}).get("parse_success")) for row in records)
    correct = sum(bool(row.get("score", {}).get("correct")) for row in records)
    safety_routes = status_counts.get("safety_routed", 0)
    reasoning_failure_flags = Counter(
        flag
        for row in records
        for flag in row.get("response", {}).get("safety_flags", ())
        if str(flag).startswith("reasoning_failure:")
    )
    reasoning_failure_cases = sum(
        any(str(flag).startswith("reasoning_failure:") for flag in row.get("response", {}).get("safety_flags", ()))
        for row in records
    )
    transport_failure_cases = sum(
        any(
            "APIConnectionError" in str(flag) or "APITimeoutError" in str(flag)
            for flag in row.get("response", {}).get("safety_flags", ())
        )
        for row in records
    )
    response = [row.get("response", {}) for row in records]
    tokens = [
        int(item["input_tokens"]) + int(item["output_tokens"])
        for item in response
        if item.get("input_tokens") is not None and item.get("output_tokens") is not None
    ]
    return {
        "profile_id": manifest.get("system_profile", {}).get("profile_id"),
        "accuracy": correct / len(records),
        "correct": correct,
        "case_count": len(records),
        "parse_success_rate": parse_ok / len(records),
        "parse_success": parse_ok,
        "runtime_failures": reasoning_failure_cases,
        "transport_failure_cases": transport_failure_cases,
        "reasoning_failure_flag_counts": dict(reasoning_failure_flags),
        "request_status_counts": dict(status_counts),
        "safety_routes": safety_routes,
        "provider_calls_per_case": mean(float(item.get("provider_calls", 0)) for item in response),
        "tokens_per_case": mean(tokens) if len(tokens) == len(records) else None,
        "mean_latency_ms": mean(float(item.get("latency_ms", 0)) for item in response),
        "retrieval_calls": sum(int((row.get("trace_summary") or {}).get("retrieval_calls", 0)) for row in records),
        "model_hash": manifest.get("model", {}).get("checkpoint_sha256"),
        "scoring_revision": manifest.get("scoring_revision"),
    }


def verify_pair(b0: dict, b2: dict, b0_cases: dict, b2_cases: dict, expected: int) -> list[str]:
    for alias, manifest in (("B0", b0), ("B2", b2)):
        profile = manifest.get("system_profile", {})
        if profile.get("profile_id") != alias or profile.get("retrieval_mode") != "off" or profile.get("memory_mode") != "off":
            raise RuntimeError(f"invalid profile for {alias}: {profile}")
    if set(b0_cases) != set(b2_cases):
        raise RuntimeError("paired case IDs differ")
    if len(b0_cases) != expected:
        raise RuntimeError("paired denominator differs from frozen expected size")
    for field in ("git_sha", "model_config_sha256", "scoring_revision"):
        if b0.get(field) != b2.get(field):
            raise RuntimeError(f"pair identity mismatch: {field}")
    left_hashes = b0.get("implementation_sha256", {})
    right_hashes = b2.get("implementation_sha256", {})
    changed_files = sorted(
        name for name in set(left_hashes) | set(right_hashes)
        if left_hashes.get(name) != right_hashes.get(name)
    )
    if changed_files not in ([], [ADAPTIVE_ONLY_PATCH]):
        raise RuntimeError(f"unexpected implementation hash differences: {changed_files}")
    if changed_files and not (
        b0.get("system_profile", {}).get("reasoning_mode") == "single"
        and b2.get("system_profile", {}).get("reasoning_mode") == "adaptive_mdt"
    ):
        raise RuntimeError("adaptive-only parser patch is allowed only for SINGLE vs ADAPTIVE_MDT")
    if b0.get("model", {}).get("checkpoint_sha256") != b2.get("model", {}).get("checkpoint_sha256"):
        raise RuntimeError("model checkpoint differs")
    for field in ("source_revision", "combined_dataset_identity_sha256", "ids_manifest_sha256", "ids_sequence_sha256", "candidate_view_sha256", "scorer_view_sha256"):
        if b0.get("dataset", {}).get(field) != b2.get("dataset", {}).get(field):
            raise RuntimeError(f"dataset identity mismatch: {field}")
    if b0.get("decoding") != b2.get("decoding"):
        raise RuntimeError("decoding protocol differs")
    if b0.get("model", {}).get("serving") != b2.get("model", {}).get("serving"):
        raise RuntimeError("model serving configuration differs")
    if b0.get("model", {}).get("base_revision") != b2.get("model", {}).get("base_revision"):
        raise RuntimeError("model revision differs")


def analyze(dataset: str, b0_path: Path, b2_path: Path, expected: int) -> dict:
    b0_manifest, b0_cases, b0_status = read_arm(b0_path, expected)
    b2_manifest, b2_cases, b2_status = read_arm(b2_path, expected)
    implementation_hash_differences = verify_pair(
        b0_manifest, b2_manifest, b0_cases, b2_cases, expected,
    )
    b0_metrics = arm_metrics(b0_manifest, b0_cases, b0_status)
    b2_metrics = arm_metrics(b2_manifest, b2_cases, b2_status)
    b0_only = b2_only = agreement = 0
    route_slices: dict[str, list[dict[str, bool]]] = defaultdict(list)
    paired_path = RUN_ROOT / f"{dataset.lower().replace('-', '_')}_paired_cases.jsonl"
    with paired_path.open("w", encoding="utf-8", newline="\n") as output:
        for case_id in sorted(b0_cases):
            left, right = b0_cases[case_id], b2_cases[case_id]
            c0 = bool(left["score"].get("correct"))
            c2 = bool(right["score"].get("correct"))
            b0_only += int(c0 and not c2)
            b2_only += int(c2 and not c0)
            a0 = left.get("response", {}).get("parsed_answer")
            a2 = right.get("response", {}).get("parsed_answer")
            agree = a0 == a2
            agreement += int(agree)
            route = (right.get("trace_summary") or {}).get("mdt_complexity") or "UNCLASSIFIED"
            route_slices[str(route).upper()].append({"b0": c0, "b2": c2})
            output.write(json.dumps({
                "case_id": case_id,
                "b0_correct": c0,
                "b2_correct": c2,
                "b0_parse_success": bool(left["score"].get("parse_success")),
                "b2_parse_success": bool(right["score"].get("parse_success")),
                "b0_answer": a0,
                "b2_answer": a2,
                "answer_agreement": agree,
                "b2_complexity_route": route,
                "b0_request_status": b0_status.get(str(left["response"].get("trace_id"))),
                "b2_request_status": b2_status.get(str(right["response"].get("trace_id"))),
            }, ensure_ascii=False, sort_keys=True) + "\n")
    complexity = {}
    for route, rows in sorted(route_slices.items()):
        complexity[route] = {
            "cases": len(rows),
            "b0_correct": sum(int(row["b0"]) for row in rows),
            "b2_correct": sum(int(row["b2"]) for row in rows),
            "b0_accuracy": mean(int(row["b0"]) for row in rows),
            "b2_accuracy": mean(int(row["b2"]) for row in rows),
        }
    delta_pp = (b2_metrics["accuracy"] - b0_metrics["accuracy"]) * 100
    return {
        "dataset": dataset,
        "b0": b0_metrics,
        "b2": b2_metrics,
        "delta_percentage_points": delta_pp,
        "wrong_to_right_b0_to_b2": b2_only,
        "right_to_wrong_b0_to_b2": b0_only,
        "net_gain_cases": b2_only - b0_only,
        "answer_agreement": agreement / expected,
        "mcnemar_exact_two_sided_p": exact_mcnemar(b0_only, b2_only),
        "complexity_slices_by_b2_route": complexity,
        "implementation_hash_differences": implementation_hash_differences,
        "implementation_hash_note": (
            "The only allowed source difference is the adaptive complexity-output parser; the B0 SINGLE path does not execute it."
            if implementation_hash_differences else None
        ),
        "paired_cases_path": str(paired_path),
        "b0_server_pid": (b0_manifest.get("vllm_runtime_config") or {}).get("vllm_frontend_pid"),
        "b2_server_pid": (b2_manifest.get("vllm_runtime_config") or {}).get("current_server_pid"),
    }


def main() -> int:
    status_path = RUN_ROOT / "final-recovery-status.json"
    state = json.loads(status_path.read_text(encoding="utf-8"))
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"final recovery is not complete: {state.get('status')}")
    final_outputs = state.get("final_outputs", {})
    arms = {
        "CMB-COMMON-1024": (B0_CMB, Path(final_outputs["cmb-common-1024"]), 1024),
        "DiagnosisArena-915": (B0_DA, Path(final_outputs["diagnosisarena-915"]), 915),
    }
    reports=[]
    for dataset, (b0_path,b2_path,expected) in arms.items():
        reports.append(analyze(dataset,b0_path,b2_path,expected))
    combined={"schema_version":"h1-final-ma-pair-analysis-v1","results":reports}
    path=RUN_ROOT/'final_pair_report.json'
    path.write_text(json.dumps(combined,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    lines=["# Final Multi-Agent B0 vs B2 (No RAG)","", "RAG and memory are OFF in both arms. Accuracy denominator includes parse/runtime/safety failures.",""]
    for report in reports:
        lines.extend([
          f"## {report['dataset']}","",
          "| Metric | B0 Single | B2 Adaptive MDT | Delta |","|---|---:|---:|---:|",
          f"| Accuracy | {report['b0']['correct']}/{report['b0']['case_count']} ({report['b0']['accuracy']*100:.2f}%) | {report['b2']['correct']}/{report['b2']['case_count']} ({report['b2']['accuracy']*100:.2f}%) | {report['delta_percentage_points']:+.2f} pp |",
          f"| Parse success | {report['b0']['parse_success']}/{report['b0']['case_count']} ({report['b0']['parse_success_rate']*100:.2f}%) | {report['b2']['parse_success']}/{report['b2']['case_count']} ({report['b2']['parse_success_rate']*100:.2f}%) | — |",
          f"| Runtime failures | {report['b0']['runtime_failures']} | {report['b2']['runtime_failures']} | — |",
          f"| Safety routes | {report['b0']['safety_routes']} | {report['b2']['safety_routes']} | — |",
          f"| Provider calls/case | {report['b0']['provider_calls_per_case']:.3f} | {report['b2']['provider_calls_per_case']:.3f} | — |",
          f"| Tokens/case | {report['b0']['tokens_per_case']:.1f} | {report['b2']['tokens_per_case']:.1f} | — |",
          f"| Mean latency | {report['b0']['mean_latency_ms']/1000:.2f}s | {report['b2']['mean_latency_ms']/1000:.2f}s | — |",
          f"| Wrong→right | — | {report['wrong_to_right_b0_to_b2']} | — |",
          f"| Right→wrong | — | {report['right_to_wrong_b0_to_b2']} | — |",
          f"| McNemar exact p | — | {report['mcnemar_exact_two_sided_p']:.6g} | — |",
          f"| Answer agreement | — | {report['answer_agreement']*100:.2f}% | — |","",
          "### MDT complexity slices","",
          "| B2 route | Cases | B0 accuracy | B2 accuracy |","|---|---:|---:|---:|",
        ])
        for route, data in report['complexity_slices_by_b2_route'].items():
            lines.append(f"| {route} | {data['cases']} | {data['b0_accuracy']*100:.2f}% | {data['b2_accuracy']*100:.2f}% |")
        lines.append("")
    (RUN_ROOT/'final_pair_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
