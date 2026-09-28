"""Run the MEM-3A.2R 16K monolithic FlatProp diagnostic."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import run_mem3a2_minimal_flatprop as base

BASE_COMMIT = "f2c448c20be6321f66e90511e325611c1c37e4fc"
RUN_ID = "mem3a2r-minimal-flatprop-16k-frozen-10-20260929"
RUN_DIR = ROOT / "runs/memory/mem3" / RUN_ID
LOCAL_CACHE_ROOT = ROOT / ".cache/health-copilot/mem3a2r-minimal-flatprop-v3-16k-writer"
MAX_TOKENS = 16384
OUTPUT_RESERVE = 16384
CONTEXT_LIMIT = 131072
PROMPT_SHA256 = "0c0211bac51cf48e8e01663bbee7d9f9db69815afc7d40c1e8d6877f4a0f2376"
CONTRACT_SHA256 = "708da9fce6990c3b410f1bc610009e76dc39a993a786f59bfeac9f5798fd1545"
HISTORICAL_SENTINEL = "7dcf35e1722f16d214f5fe14fd59bc198cd73d0c35ec9858300f0b348e30cb15"
GATE = "MEM3A2R_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC"
GATE_NO = f"{GATE}=NO"
GATE_PENDING = f"{GATE}=PENDING"
GATE_YES = f"{GATE}=YES"
SEMANTIC_SIMILARITY_THRESHOLD = 0.8
WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)

_ORIGINAL_CONFIGURE = base._configure_flat_paths
_ORIGINAL_EXTRACT = base._extract_all
_ORIGINAL_RENDER = base._render_report
_ORIGINAL_EXECUTE = base.execute_or_resume
_ORIGINAL_WRITER_REQUEST = base.writer_v3.writer_request


def _configure_flat_paths() -> None:
    _ORIGINAL_CONFIGURE()
    base.flat.PREFLIGHT_PATH = RUN_DIR / "writer_preflight_v3_16k.json"
    base.flat.RUN_REPORT_PATH = RUN_DIR / "report.md"
    base.flat.COMPARISON_PATH = RUN_DIR / "comparison_mem2d_vs_mem3a2r.json"
    base.flat.CASE_REVIEW_PATH = RUN_DIR / "case_review.json"


def _writer_request_16k(**kwargs: Any) -> dict[str, Any]:
    request = _ORIGINAL_WRITER_REQUEST(**kwargs, max_tokens=MAX_TOKENS)
    if request.get("max_tokens") != MAX_TOKENS:
        raise RuntimeError("MEM-3A.2R writer request did not bind the frozen 16K cap")
    return request


def _execute_16k(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    kwargs["stage_identity"] = "MEM-3A.2R/minimal-flatprop-v3-16k"
    return _ORIGINAL_EXECUTE(**kwargs)


def _write_frozen(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    )
    if path.exists():
        if not base.flat._verify_frozen(path) or path.read_bytes() != payload:
            raise RuntimeError(f"Frozen MEM-3A.2R artifact differs: {path.name}")
        return base.sha256_file(path)
    base.atomic_write_bytes(path, payload)
    return base.flat._freeze(path)


def _runtime_contract(preflight: dict[str, Any]) -> dict[str, Any]:
    runtime = preflight["writer_runtime_identity"]
    if (
        runtime.get("server_build") != "llama.cpp 10068 (571d0d540)"
        or runtime.get("context_tokens") != CONTEXT_LIMIT
        or runtime.get("model_sha256") != base.flat.PINNED_WRITER_MODEL_SHA256
        or runtime.get("loopback_only") is not True
        or runtime.get("server_binary_sha256")
        != "3a8aea5f889c4b4c2ec41c98f4e1ed484bb7a40c4096883acb23d3cfe26b59fb"
    ):
        raise RuntimeError("MEM-3A.2R llama.cpp/Qwen writer runtime is not the frozen local build")
    body = {
        "schema_version": 1,
        "stage": RUN_ID,
        "model_alias": base.flat.READER_MODEL,
        "model_sha256": runtime["model_sha256"],
        "endpoint": runtime["endpoint"],
        "loopback_only": True,
        "llama_cpp_build": runtime["server_build"],
        "llama_server_binary_sha256": runtime["server_binary_sha256"],
        "runtime_props_sha256": runtime["runtime_props_sha256"],
        "temperature": 0,
        "seed": 42,
        "thinking": False,
        "max_tokens": MAX_TOKENS,
        "context_limit": CONTEXT_LIMIT,
        "output_reserve": OUTPUT_RESERVE,
        "transport_unwrap_source_sha256": base.sha256_file(
            Path(base.v2_runtime.__file__).with_name("flat_proposition_writer_v2.py")
        ),
        "packet_validator_source_sha256": base.sha256_file(Path(base.writer_v3.__file__)),
        "writer_request_source_sha256": base.sha256_file(Path(base.writer_v3.__file__)),
        "extractor_prompt_sha256": preflight["extractor_prompt_sha256"],
        "extractor_contract_sha256": preflight["extractor_contract_sha256"],
        "method_id": "flat-proposition-extractor-v3-minimal",
        "hosted_provider": None,
        "hosted_calls": 0,
    }
    body["runtime_contract_sha256"] = base.sha256_bytes(base.canonical_json(body))
    return body


def _execution_order(sessions: dict[str, dict[str, Any]]) -> list[str]:
    if HISTORICAL_SENTINEL not in sessions or len(sessions) != 477:
        raise RuntimeError("Risk-first execution order does not contain the 477-session corpus")
    return [HISTORICAL_SENTINEL] + [key for key in sessions if key != HISTORICAL_SENTINEL]


def _preflight(*, save: bool) -> dict[str, Any]:
    current = base._build_preflight(tokenize=True)
    if (
        current["extractor_prompt_sha256"] != PROMPT_SHA256
        or current["extractor_contract_sha256"] != CONTRACT_SHA256
    ):
        raise RuntimeError("Frozen v3 prompt or contract SHA changed; no writer calls allowed")
    if current.get("output_reserve") != OUTPUT_RESERVE or len(current["requests"]) != 477:
        raise RuntimeError("MEM-3A.2R must preflight 477 requests with a 16K reserve")
    for row in current["requests"]:
        row["max_tokens"] = MAX_TOKENS
        if row["prompt_tokens"] + MAX_TOKENS > CONTEXT_LIMIT or row["truncated"]:
            raise RuntimeError(
                f"MEM-3A.2R prompt plus 16K reserve does not fit: {row['session_identity_sha256']}"
            )
    current["writer_max_tokens"] = MAX_TOKENS
    current["output_reserve"] = OUTPUT_RESERVE
    current["context_limit"] = CONTEXT_LIMIT
    current["max_reserved_prompt_tokens"] = max(
        row["prompt_tokens"] + OUTPUT_RESERVE for row in current["requests"]
    )
    sessions, _, _ = base._source_sessions_with_catalog()
    order = _execution_order(sessions)
    current["execution_order_session_identities"] = order
    current["execution_order_selected_for_failure_efficiency_not_model_quality"] = True
    current["historical_failure_session_identity_sha256"] = HISTORICAL_SENTINEL
    current["requests_are_canonical_order_independent_of_execution_order"] = True
    runtime_contract = _runtime_contract(current)
    runtime_sha = _write_frozen(RUN_DIR / "writer_runtime_contract.json", runtime_contract)
    identity = current["extraction_identity"]["identity"]
    identity["writer_max_tokens"] = MAX_TOKENS
    identity["output_reserve"] = OUTPUT_RESERVE
    identity["runtime_contract_sha256"] = runtime_sha
    identity["execution_order_sha256"] = base.sha256_bytes(
        ("\n".join(order) + "\n").encode("ascii")
    )
    current["extraction_identity"]["identity_sha256"] = base.sha256_bytes(
        base.canonical_json(identity)
    )
    path = RUN_DIR / "writer_preflight_v3_16k.json"
    identity_path = RUN_DIR / "writer_extraction_identity.json"
    if save:
        if path.exists():
            old = base._json(path)
            if not base.flat._verify_frozen(path) or base._preflight_core(
                old
            ) != base._preflight_core(current):
                raise RuntimeError("Existing 16K preflight differs from the live request identity")
            return old
        _write_frozen(path, current)
        _write_frozen(identity_path, current["extraction_identity"])
        threshold_contract = {
            "schema_version": 1,
            "stage": RUN_ID,
            "metric": "lowercase word-token set Jaccard similarity",
            "threshold_inclusive": SEMANTIC_SIMILARITY_THRESHOLD,
            "diagnostic_only": True,
            "applies_after_writer_packets_freeze": True,
            "no_merge_drop_or_retry": True,
        }
        _write_frozen(RUN_DIR / "semantic_noise_threshold.json", threshold_contract)
        protocol = {
            "schema_version": 1,
            "stage": RUN_ID,
            "base_commit_sha": BASE_COMMIT,
            "sole_method_change": "writer max_tokens 4096 -> 16384",
            "prompt_sha256": PROMPT_SHA256,
            "extractor_contract_sha256": CONTRACT_SHA256,
            "writer_runtime_contract_sha256": runtime_sha,
            "preflight_sha256": base.sha256_file(path),
            "preflight_requests": 477,
            "all_prompt_tokens_plus_16384_fit": current["all_prompt_tokens_plus_reserve_fit"],
            "max_reserved_prompt_tokens": current["max_reserved_prompt_tokens"],
            "execution_order_selected_for_failure_efficiency_not_model_quality": True,
            "first_execution_identity": HISTORICAL_SENTINEL,
            "execution_order_sha256": identity["execution_order_sha256"],
            "retries": 0,
            "hosted_calls": 0,
            "semantic_noise_threshold": threshold_contract,
            "label_barrier": "labels remain unloaded through writer, inventory, retrieval, bundles, predictions and reader ledger freeze",
        }
        _write_frozen(RUN_DIR / "protocol_v3_16k.json", protocol)
        return current
    for frozen_path, expected in (
        (path, current),
        (identity_path, current["extraction_identity"]),
        (RUN_DIR / "writer_runtime_contract.json", runtime_contract),
    ):
        if not frozen_path.exists() or not base.flat._verify_frozen(frozen_path):
            raise RuntimeError(f"Required frozen 16K protocol artifact missing: {frozen_path.name}")
        if base._json(frozen_path) != expected:
            raise RuntimeError(
                f"Live 16K protocol differs from frozen artifact: {frozen_path.name}"
            )
    for name in ("semantic_noise_threshold.json", "protocol_v3_16k.json"):
        if not base.flat._verify_frozen(RUN_DIR / name):
            raise RuntimeError(f"Required frozen MEM-3A.2R protocol artifact is invalid: {name}")
    return base._json(path)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        rank = (cursor + 1 + stop) / 2
        for offset in range(cursor, stop):
            ranks[order[offset]] = rank
        cursor = stop
    return ranks


def _correlation(x: list[float], y: list[float]) -> dict[str, float | None]:
    def pearson(a: list[float], b: list[float]) -> float | None:
        if len(a) < 2:
            return None
        ma = sum(a) / len(a)
        mb = sum(b) / len(b)
        da = [value - ma for value in a]
        db = [value - mb for value in b]
        den = math.sqrt(sum(value * value for value in da) * sum(value * value for value in db))
        return sum(left * right for left, right in zip(da, db)) / den if den else None

    return {"pearson": pearson(x, y), "spearman": pearson(_ranks(x), _ranks(y))}


def _distribution(values: list[int | float]) -> dict[str, int | float | None]:
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "total": sum(numeric) if numeric else 0,
        "mean": sum(numeric) / len(numeric) if numeric else None,
        "median": _percentile(numeric, 0.5),
        "p90": _percentile(numeric, 0.9),
        "p95": _percentile(numeric, 0.95),
        "p99": _percentile(numeric, 0.99),
        "maximum": max(numeric) if numeric else None,
    }


def _writer_scale(
    packets: list[dict[str, Any]], ledger: list[dict[str, Any]], sessions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    packet_by_id = {row["session_identity_sha256"]: row for row in packets}
    ledger_by_id = {row["session_identity_sha256"]: row for row in ledger}
    ordered_ids = [row["session_identity_sha256"] for row in packets]
    prop_counts = [len(packet_by_id[key]["propositions"]) for key in ordered_ids]
    completion = [int(ledger_by_id[key]["completion_tokens"]) for key in ordered_ids]
    prompts = [int(ledger_by_id[key]["prompt_tokens"]) for key in ordered_ids]
    spans = [len(sessions[key]["catalog"]) for key in ordered_ids]
    turns = [len(sessions[key]["turns"]) for key in ordered_ids]
    return {
        "schema_version": 1,
        "stage": RUN_ID,
        "session_count": len(ordered_ids),
        "propositions_per_session": _distribution(prop_counts),
        "completion_tokens_per_session": _distribution(completion),
        "prompt_tokens_per_session": _distribution(prompts),
        "propositions_per_source_turn": sum(prop_counts) / max(1, sum(turns)),
        "propositions_per_source_span": sum(prop_counts) / max(1, sum(spans)),
        "zero_proposition_sessions": sum(count == 0 for count in prop_counts),
        "source_turns_per_session": _distribution(turns),
        "source_spans_per_session": _distribution(spans),
        "relationships": {
            "prompt_tokens_vs_completion_tokens": _correlation(prompts, completion),
            "source_span_count_vs_proposition_count": _correlation(spans, prop_counts),
            "source_turn_count_vs_proposition_count": _correlation(turns, prop_counts),
        },
        "historical_4k_partial_run": {
            "successful_sessions": 54,
            "propositions": 567,
            "propositions_per_session": {"mean": 10.5, "median": 9, "p95": 21, "maximum": 27},
            "completion_tokens": {"median": 688, "p95": 1482, "maximum_successful": 2565},
            "descriptive_only_not_reused_as_16k_outcomes": True,
        },
        "diagnostic_only_no_tuning": True,
    }


def _semantic_noise(packets: list[dict[str, Any]]) -> dict[str, Any]:
    threshold = SEMANTIC_SIMILARITY_THRESHOLD
    prop_diagnostics = []
    high_pairs = []
    session_count = 0
    assistant_only = 0
    mixed = 0
    for packet in packets:
        props = packet["propositions"]
        token_sets = []
        for prop in props:
            refs = prop["evidence"]
            turns = {row["source_turn_index"] for row in refs}
            authority = prop["source_authority"]
            assistant_only += authority == "assistant"
            mixed += authority == "mixed"
            tokens = WORD.findall(prop["proposition_text"].lower())
            token_sets.append(set(tokens))
            prop_diagnostics.append(
                {
                    "session_identity_sha256": packet["session_identity_sha256"],
                    "proposition_index": prop["proposition_index"],
                    "source_authority": authority,
                    "evidence_ref_count": len(prop["evidence_refs"]),
                    "source_turn_count": len(turns),
                    "proposition_word_token_count": len(tokens),
                }
            )
        session_pairs = 0
        for left in range(len(props)):
            for right in range(left + 1, len(props)):
                union = token_sets[left] | token_sets[right]
                similarity = (
                    len(token_sets[left] & token_sets[right]) / len(union) if union else 1.0
                )
                if similarity >= threshold:
                    session_pairs += 1
                    high_pairs.append(
                        {
                            "session_identity_sha256": packet["session_identity_sha256"],
                            "left_proposition_index": props[left]["proposition_index"],
                            "right_proposition_index": props[right]["proposition_index"],
                            "jaccard_similarity": similarity,
                        }
                    )
        session_count += session_pairs > 0
    return {
        "schema_version": 1,
        "stage": RUN_ID,
        "threshold": threshold,
        "similarity": "lowercase word-token set Jaccard",
        "proposition_diagnostics": prop_diagnostics,
        "high_similarity_pair_count": len(high_pairs),
        "sessions_with_high_similarity_pairs": session_count,
        "high_similarity_pairs": high_pairs,
        "assistant_only_proposition_count": assistant_only,
        "mixed_authority_proposition_count": mixed,
        "diagnostic_only": True,
        "deduplicated_or_dropped": False,
        "used_to_change_writer_or_retrieval": False,
    }


def _extract_all(preflight: dict[str, Any], sessions: dict[str, dict[str, Any]]):
    order = _execution_order(sessions)
    ordered = {identity: sessions[identity] for identity in order}
    packets, ledger, manifest = _ORIGINAL_EXTRACT(preflight, ordered)
    if len(packets) == 477 and len(ledger) == 477 and all(row.get("success") for row in ledger):
        _write_frozen(
            RUN_DIR / "writer_scale_diagnostics.json", _writer_scale(packets, ledger, sessions)
        )
        _write_frozen(RUN_DIR / "semantic_noise_diagnostics.json", _semantic_noise(packets))
    return packets, ledger, manifest


def _render_report(
    gate: str,
    writer_diag: dict[str, Any],
    embedding: dict[str, Any],
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    report = _ORIGINAL_RENDER(gate, writer_diag, embedding, metrics, comparison, manifest)
    report = report.replace("MEM3A2_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC", GATE)
    report = report.replace("MEM-3A.2 Minimal FlatProp", "MEM-3A.2R 16K Minimal FlatProp")
    scale = base._json(RUN_DIR / "writer_scale_diagnostics.json")
    noise = base._json(RUN_DIR / "semantic_noise_diagnostics.json")
    summary = [
        "",
        "## Writer Capacity and Amplification",
        f"- 16K writer outcomes: {scale['session_count']}; propositions/session mean/median/P90/P95/P99/max: {scale['propositions_per_session']['mean']:.3f} / {scale['propositions_per_session']['median']:.1f} / {scale['propositions_per_session']['p90']:.1f} / {scale['propositions_per_session']['p95']:.1f} / {scale['propositions_per_session']['p99']:.1f} / {scale['propositions_per_session']['maximum']:.0f}.",
        f"- Completion tokens/session median/P90/P95/P99/max: {scale['completion_tokens_per_session']['median']:.1f} / {scale['completion_tokens_per_session']['p90']:.1f} / {scale['completion_tokens_per_session']['p95']:.1f} / {scale['completion_tokens_per_session']['p99']:.1f} / {scale['completion_tokens_per_session']['maximum']:.0f}.",
        f"- Propositions/source turn: {scale['propositions_per_source_turn']:.4f}; propositions/source span: {scale['propositions_per_source_span']:.4f}; zero-proposition sessions: {scale['zero_proposition_sessions']}.",
        f"- Prompt/completion, span/proposition, and turn/proposition Pearson/Spearman: {scale['relationships']}.",
        f"- Preflight reserved prompt maximum (actual tokenizer count + 16,384): {base._json(RUN_DIR / 'writer_preflight_v3_16k.json')['max_reserved_prompt_tokens']} / {CONTEXT_LIMIT} tokens.",
        "- Historical 4K partial: 54 sessions / 567 propositions (mean 10.5, median 9, P95 21, max 27); completion median 688, P95 1,482, max successful 2,565. It was not reused as a 16K result.",
        "",
        "## Semantic Noise Audit",
        f"- Frozen diagnostic threshold: lowercase word-token set Jaccard >= {SEMANTIC_SIMILARITY_THRESHOLD:.1f}; high-similarity pairs: {noise['high_similarity_pair_count']}; sessions containing a pair: {noise['sessions_with_high_similarity_pairs']}.",
        f"- Assistant-only propositions: {noise['assistant_only_proposition_count']}; mixed-authority propositions: {noise['mixed_authority_proposition_count']}. Diagnostic only; no propositions were merged or dropped.",
        "- Risk-first execution order placed the known historical capacity-tail session first for failure efficiency, not model quality; canonical request/preflight order is unchanged.",
        "",
    ]
    return report.rstrip() + "\n" + "\n".join(summary)


def _patch_runner() -> None:
    base.__file__ = str(Path(__file__).resolve())
    base.BASE_COMMIT = BASE_COMMIT
    base.RUN_ID = RUN_ID
    base.RUN_DIR = RUN_DIR
    base.LOCAL_CACHE_ROOT = LOCAL_CACHE_ROOT
    base.OUTPUT_RESERVE = OUTPUT_RESERVE
    base.CONTEXT_LIMIT = CONTEXT_LIMIT
    base.writer_v3.writer_request = _writer_request_16k
    base.execute_or_resume = _execute_16k
    base._configure_flat_paths = _configure_flat_paths
    base._preflight = _preflight
    base._extract_all = _extract_all
    base._render_report = _render_report


def _finalize_success(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["stage"] = RUN_ID
    manifest["base_commit_sha"] = BASE_COMMIT
    manifest["writer_max_tokens"] = MAX_TOKENS
    manifest["writer_output_reserve"] = OUTPUT_RESERVE
    manifest["execution_order_selected_for_failure_efficiency_not_model_quality"] = True
    manifest["completion_gate_marker"] = GATE_YES
    manifest["status"] = "COMPLETE"
    gate = manifest["gate"]
    gate["writer_runtime_contract_16k_frozen"] = base.flat._verify_frozen(
        RUN_DIR / "writer_runtime_contract.json"
    )
    gate["all_477_preflight_reserve_16384_fit"] = (
        len(base._json(RUN_DIR / "writer_preflight_v3_16k.json")["requests"]) == 477
    )
    gate["writer_scale_diagnostics_frozen"] = base.flat._verify_frozen(
        RUN_DIR / "writer_scale_diagnostics.json"
    )
    gate["semantic_noise_threshold_frozen_before_generation"] = base.flat._verify_frozen(
        RUN_DIR / "semantic_noise_threshold.json"
    )
    gate["semantic_noise_diagnostics_frozen_after_packets"] = base.flat._verify_frozen(
        RUN_DIR / "semantic_noise_diagnostics.json"
    )
    gate["first_execution_was_historical_truncation_sentinel"] = (
        base._json(RUN_DIR / "writer_preflight_v3_16k.json")["execution_order_session_identities"][
            0
        ]
        == HISTORICAL_SENTINEL
    )
    gate["request_cap_exactly_16384"] = all(
        row.get("max_tokens") == MAX_TOKENS
        for row in base._json(RUN_DIR / "writer_preflight_v3_16k.json")["requests"]
    )
    gate["no_102_dev_or_test_access"] = True
    extra_paths = [
        RUN_DIR / "writer_runtime_contract.json",
        RUN_DIR / "writer_preflight_v3_16k.json",
        RUN_DIR / "writer_extraction_identity.json",
        RUN_DIR / "semantic_noise_threshold.json",
        RUN_DIR / "protocol_v3_16k.json",
        RUN_DIR / "writer_scale_diagnostics.json",
        RUN_DIR / "semantic_noise_diagnostics.json",
        RUN_DIR / "session_extraction_manifest.json",
        RUN_DIR / "session_extractions.jsonl",
        RUN_DIR / "writer_call_ledger.jsonl",
        RUN_DIR / "flat_propositions.jsonl",
        RUN_DIR / "materialization_ledger.jsonl",
        RUN_DIR / "embedding_manifest.json",
        RUN_DIR / "dense_top8.jsonl",
        RUN_DIR / "context_plans.jsonl",
        RUN_DIR / "context_bundles.jsonl",
        RUN_DIR / "predictions.jsonl",
        RUN_DIR / "reader_call_ledger.jsonl",
        RUN_DIR / "deterministic_metrics.json",
        RUN_DIR / "efficiency.json",
        RUN_DIR / "comparison_mem2d_vs_mem3a2r.json",
        RUN_DIR / "case_review.json",
        RUN_DIR / "report.md",
    ]
    gate["all_compact_artifact_sha_sidecars_valid"] = all(
        path.exists() and base.flat._verify_frozen(path) for path in extra_paths
    )
    passed = all(gate.values())
    manifest["status"] = "COMPLETE" if passed else "FAILED"
    manifest["completion_gate_marker"] = f"{GATE}={'YES' if passed else 'NO'}"
    manifest["artifact_sha256"] = {
        **manifest.get("artifact_sha256", {}),
        **{path.name: base.sha256_file(path) for path in extra_paths},
    }
    base._write_json(RUN_DIR / "run_manifest.json", manifest)
    if not passed:
        raise RuntimeError(GATE_NO)
    print(GATE_YES, flush=True)
    return manifest


def run() -> dict[str, Any]:
    _patch_runner()
    base._configure_flat_paths()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(save=True)
    if (
        len(preflight["requests"]) != 477
        or preflight["truncated_requests"] != 0
        or not preflight["all_prompt_tokens_plus_reserve_fit"]
        or any(row["prompt_tokens"] + MAX_TOKENS > CONTEXT_LIMIT for row in preflight["requests"])
    ):
        raise RuntimeError("MEM-3A.2R full-corpus 16K preflight failed")
    upstream = base._verify_upstream()
    sessions, refs, inventory = base._source_sessions_with_catalog()
    packets, writer_ledger, writer_manifest = _extract_all(preflight, sessions)
    if writer_manifest.get("failure") or len(packets) != 477:
        failure = writer_manifest.get("failure", {})
        if failure.get("code") == "COMPLETION_TRUNCATED":
            failure["code"] = "WRITER_UNBOUNDED_OUTPUT_FAILURE"
            failure["historical_transport_code"] = "COMPLETION_TRUNCATED"
        raise RuntimeError(f"{GATE_NO}; writer stopped before downstream: {failure}")
    result = base._downstream(
        upstream,
        preflight,
        packets,
        writer_ledger,
        sessions,
        refs,
        inventory,
        writer_manifest,
    )
    return _finalize_success(result)


def main() -> int:
    try:
        run()
        return 0
    except Exception as exc:
        _patch_runner()
        base._configure_flat_paths()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        text = str(exc)
        code = (
            "WRITER_UNBOUNDED_OUTPUT_FAILURE"
            if "WRITER_UNBOUNDED_OUTPUT_FAILURE" in text or "COMPLETION_TRUNCATED" in text
            else type(exc).__name__
        )
        failure = {
            "stage": RUN_ID,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": text,
            "failure_classification": code,
            "writer_max_tokens": MAX_TOKENS,
            "completion_gate_marker": GATE_NO,
            "hosted_calls": 0,
            "judge_calls": 0,
            "test_access": False,
            "102_dev_access": False,
        }
        base._write_json(RUN_DIR / "run_failure.json", failure)
        run_manifest_path = RUN_DIR / "run_manifest.json"
        if run_manifest_path.exists():
            run_manifest = base._json(run_manifest_path)
        else:
            run_manifest = {"schema_version": 1}
        run_manifest.update(
            {
                "stage": RUN_ID,
                "status": "FAILED",
                "base_commit_sha": BASE_COMMIT,
                "writer_max_tokens": MAX_TOKENS,
                "completion_gate_marker": GATE_NO,
                "failure_classification": code,
                "failure": text,
                "hosted_calls": 0,
                "judge_calls": 0,
                "labels_loaded": False,
                "test_access": False,
                "102_dev_access": False,
            }
        )
        base._write_json(run_manifest_path, run_manifest)
        base.atomic_write_bytes(
            RUN_DIR / "report.md",
            (
                "# MEM-3A.2R 16K Minimal FlatProp Diagnostic\n\n"
                f"Completion gate: `{GATE_NO}`.\n\n"
                f"Failure classification: `{code}`.\n\n"
                f"Details: `{text}`\n\n"
                "The 4K historical run remains immutable. No downstream stage starts unless all 477 16K writer packets validate.\n"
            ).encode(),
        )
        base.flat._freeze(RUN_DIR / "report.md")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
