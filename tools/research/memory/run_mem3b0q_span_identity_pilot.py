"""Prepare, locally run, and score the frozen MEM-3B0Q-R2 identity pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research.memory import run_mem3b0p_pairwise_admission as b0p_runner
from tools.research.memory import span_grounded_identity as span_identity

SOURCE_RUN = ROOT / "runs/memory/mem3/mem3b0q-factorized-admission-20260930"
RUN_ID = "mem3b0q-span-identity-pilot-20261001"
RUN_DIR = ROOT / "runs" / "memory" / "mem3" / RUN_ID
SOURCE_PATH = SOURCE_RUN / "eligible_records.jsonl"
CONTROL_AUDIT_PATH = SOURCE_RUN / "post_freeze_control_audit.json"
REVIEW_DECISIONS_PATH = SOURCE_RUN / "human_review_decisions_user.json"
PROTOCOL_PATH = ROOT / "docs/research/memory/mem_3b0q_span_identity_pilot_protocol.md"
INPUT_PATH = RUN_DIR / "proposal_inputs.jsonl"
CASE_PATH = RUN_DIR / "diagnostic_cases.json"
RESPONSE_PATH = RUN_DIR / "provider_response.json"
PROPOSALS_PATH = RUN_DIR / "identity_proposals.jsonl"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
RESULT_PATH = RUN_DIR / "diagnostic_result.json"
REPORT_PATH = RUN_DIR / "report.md"
EXPECTED_SOURCE_SHA256 = "1ed014923a7213106e4da384929c6d303ee60a4a607e8b2942231dc03718da7c"
EXPECTED_CONTROL_AUDIT_SHA256 = "0f2bb3c86f4666b12032395f7fbfbb96d06fb1e2c50d54f253de69dbea694a64"
EXPECTED_REVIEW_DECISIONS_SHA256 = "766f602146776e114e6696fa3e404a7954a3ceb3e72c0f3516e51288a8bd5564"
API_BASE = b0p_runner.API_BASE
MODEL_SHA256 = b0p_runner.EXPECTED_MODEL_SHA256
MAX_COMPLETION_TOKENS = 8192
TIMEOUT_SECONDS = 900

RECORD_SELECTORS = {
    "instagram_600": "has reached 600 followers on Instagram",
    "instagram_500": "recently reached 500 followers on Instagram",
    "node_version_a": "has node version v19.7.0",
    "node_version_b": "using node version 19.7.0",
    "wallet_black": "decided to go with a black leather wallet",
    "wallet_neutral": "considering a neutral color like black or brown for their leather wallet",
    "wallet_material": "classic and timeless, with a preference for leather material",
    "trip_japan": "planning a trip to Japan and is seeking travel vlogs",
    "trip_india": "planning a trip to India",
    "reported_value_a": "reported a value of 0 on 18/11/2020",
    "reported_value_b": "reported a value of 0 on 17/02/2021",
    "macrame_1": "planning to try macrame",
    "macrame_2": "interested in trying macrame",
    "macrame_3": "excited about starting their macrame journey",
    "macrame_4": "complete beginner in macrame",
}


class PilotError(RuntimeError):
    """A protocol, input, runtime, or artifact invariant failed."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    if not path.is_file():
        raise PilotError(f"missing_file:{path}")
    return _sha(path.read_bytes())


def _verify_file(path: Path) -> str:
    digest = _sha_file(path)
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise PilotError(f"missing_sidecar:{sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields != [digest, path.name]:
        raise PilotError(f"sidecar_mismatch:{path}")
    return digest


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        import os

        os.fsync(stream.fileno())
    temporary.replace(path)


def _freeze(path: Path, payload: bytes) -> str:
    sidecar = path.with_suffix(".sha256")
    if path.exists() and path.read_bytes() != payload:
        raise PilotError(f"refuse_to_overwrite_frozen_artifact:{path}")
    digest = _sha(payload)
    sidecar_payload = f"{digest}  {path.name}\n".encode()
    if sidecar.exists() and sidecar.read_bytes() != sidecar_payload:
        raise PilotError(f"refuse_to_overwrite_frozen_sidecar:{sidecar}")
    if not path.exists():
        _atomic_write(path, payload)
    if not sidecar.exists():
        _atomic_write(sidecar, sidecar_payload)
    return digest


def _read_jsonl(path: Path, *, frozen: bool = True) -> list[dict[str, Any]]:
    if frozen:
        _verify_file(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_frozen_json(path: Path) -> dict[str, Any]:
    _verify_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PilotError(f"expected_json_object:{path}")
    return value


def _select_unique(rows: list[dict[str, Any]], fragment: str) -> dict[str, Any]:
    matches = [row for row in rows if fragment.casefold() in row["proposition_text"].casefold()]
    if len(matches) != 1:
        raise PilotError(f"selector_not_unique:{fragment}:{len(matches)}")
    return matches[0]


def prepare() -> dict[str, Any]:
    source_sha = _verify_file(SOURCE_PATH)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise PilotError("frozen_b0q_eligible_records_sha_mismatch")
    control_audit = _read_frozen_json(CONTROL_AUDIT_PATH)
    control_audit_sha = _verify_file(CONTROL_AUDIT_PATH)
    review_sha = _verify_file(REVIEW_DECISIONS_PATH)
    if control_audit_sha != EXPECTED_CONTROL_AUDIT_SHA256:
        raise PilotError("frozen_b0q_control_audit_sha_mismatch")
    if review_sha != EXPECTED_REVIEW_DECISIONS_SHA256:
        raise PilotError("frozen_b0q_review_decisions_sha_mismatch")
    source_rows = _read_jsonl(SOURCE_PATH)
    selected = {
        label: _select_unique(source_rows, fragment)
        for label, fragment in RECORD_SELECTORS.items()
    }
    control_pairs = {
        "gym_workout": control_audit["gym_cross_key_control"]["memory_ids"][0],
        "gym_pc": control_audit["gym_cross_key_control"]["memory_ids"][1],
        "purchase_a": control_audit["b0p_false_completed_purchase_control"]["memory_ids"][0],
        "purchase_b": control_audit["b0p_false_completed_purchase_control"]["memory_ids"][1],
    }
    source_by_id = {row["memory_id"]: row for row in source_rows}
    for label, memory_id in control_pairs.items():
        if memory_id not in source_by_id:
            raise PilotError(f"control_record_not_eligible:{label}")
        selected[label] = source_by_id[memory_id]

    if len(selected) != 19 or len({row["memory_id"] for row in selected.values()}) != 19:
        raise PilotError("pilot_selection_must_contain_19_unique_records")

    pair_cases = [
        {"case_id": "instagram_follower_revision_candidate", "memory_ids": ["instagram_500", "instagram_600"], "expected": "SAME_SLOT"},
        {"case_id": "node_version_same_slot", "memory_ids": ["node_version_a", "node_version_b"], "expected": "SAME_SLOT"},
        {"case_id": "wallet_color_same_slot", "memory_ids": ["wallet_black", "wallet_neutral"], "expected": "SAME_SLOT"},
        {"case_id": "wallet_color_vs_material", "memory_ids": ["wallet_black", "wallet_material"], "expected": "NOT_SAME_SLOT"},
        {"case_id": "wallet_neutral_vs_material", "memory_ids": ["wallet_neutral", "wallet_material"], "expected": "NOT_SAME_SLOT"},
        {"case_id": "gym_false_key_collision", "memory_ids": ["gym_workout", "gym_pc"], "expected": "NOT_SAME_SLOT"},
        {"case_id": "completed_purchase_event", "memory_ids": ["purchase_a", "purchase_b"], "expected": "NO_SLOT"},
        {"case_id": "distinct_trip_destinations", "memory_ids": ["trip_japan", "trip_india"], "expected": "NOT_SAME_SLOT"},
        {"case_id": "generic_numeric_attribute", "memory_ids": ["reported_value_a", "reported_value_b"], "expected": "BOTH_UNRESOLVED"},
    ]
    pair_cases.append(
        {
            "case_id": "macrame_interest_not_singleton_revision",
            "memory_ids": [f"macrame_{index}" for index in range(1, 5)],
            "expected": "NO_SLOT",
        }
    )
    id_by_label = {label: row["memory_id"] for label, row in selected.items()}
    diagnostic_cases = {
        "schema_version": 1,
        "stage": RUN_ID,
        "source_sha256": source_sha,
        "b0q_control_audit_sha256": control_audit_sha,
        "b0q_review_decisions_sha256": review_sha,
        "input_count": len(selected),
        "records": [
            {"label": label, "memory_id": id_by_label[label]}
            for label in sorted(selected)
        ],
        "cases": [
            {**case, "memory_ids": [id_by_label[label] for label in case["memory_ids"]]}
            for case in pair_cases
        ],
        "development_only": True,
        "case_labels_may_not_enter_model_request": True,
    }
    proposal_inputs = [
        {
            "memory_id": row["memory_id"],
            "proposition_text": row["proposition_text"],
        }
        for row in sorted(selected.values(), key=lambda item: item["memory_id"])
    ]
    input_sha = _freeze(INPUT_PATH, _jsonl_bytes(proposal_inputs))
    diagnostic_cases["proposal_inputs_sha256"] = input_sha
    case_sha = _freeze(CASE_PATH, _json_bytes(diagnostic_cases))
    result = {
        "stage": RUN_ID,
        "source_sha256": source_sha,
        "proposal_input_sha256": input_sha,
        "diagnostic_case_manifest_sha256": case_sha,
        "record_count": len(proposal_inputs),
        "case_count": len(pair_cases),
        "model_calls": 0,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def _token_count(client: httpx.Client, text: str) -> int:
    response = client.post(
        "http://127.0.0.1:8081/tokenize",
        json={"content": text, "add_special": False, "parse_special": True},
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise PilotError("local_tokenizer_response_invalid")
    return len(tokens)


def generate() -> dict[str, Any]:
    inputs = _read_jsonl(INPUT_PATH)
    if len(inputs) != 19:
        raise PilotError("frozen_proposal_input_count_mismatch")
    if any(path.exists() for path in (RESPONSE_PATH, PROPOSALS_PATH, RUN_MANIFEST_PATH)):
        raise PilotError("generation_artifact_exists_no_second_provider_call")
    with httpx.Client(
        timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10.0), trust_env=False
    ) as client:
        runtime = b0p_runner._verify_runtime(client)
        request = span_identity.build_request(
            inputs, model=runtime["model_id"], max_tokens=MAX_COMPLETION_TOKENS
        )
        request_bytes = _json_bytes(request)
        prompt_tokens = b0p_runner._rendered_prompt_tokens(client, request["messages"])
        schema_text = json.dumps(
            request["response_format"]["json_schema"]["schema"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        schema_tokens = _token_count(client, schema_text)
        if prompt_tokens + schema_tokens + MAX_COMPLETION_TOKENS > runtime["context_tokens"]:
            raise PilotError("prompt_schema_and_completion_reserve_exceed_context")
        started = time.perf_counter()
        response = client.post(f"{API_BASE}/chat/completions", json=request)
        response.raise_for_status()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)

    payload = response.json()
    _freeze(RESPONSE_PATH, _json_bytes(payload))
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise PilotError("local_response_choices_invalid")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if finish_reason != "stop" or not isinstance(content, str):
        raise PilotError(f"local_response_not_terminal:{finish_reason}")
    source_by_id = {
        row["memory_id"]: row for row in _read_jsonl(SOURCE_PATH)
    }
    records = span_identity.validate_response(content.encode("utf-8"), inputs, source_by_id)
    proposal_sha = _freeze(PROPOSALS_PATH, _jsonl_bytes(records))
    manifest = {
        "schema_version": 1,
        "stage": RUN_ID,
        "status": "PROPOSALS_FROZEN_AWAITING_DIAGNOSTIC_JOIN",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": _verify_file(SOURCE_PATH),
        "proposal_inputs_sha256": _verify_file(INPUT_PATH),
        "protocol_sha256": _verify_file(PROTOCOL_PATH),
        "code_sha256": {
            "runner": _sha_file(Path(__file__)),
            "span_identity": _sha_file(Path(span_identity.__file__)),
            "runtime_verifier": _sha_file(Path(b0p_runner.__file__)),
        },
        "runtime": {
            **runtime,
            "generation": {
                "temperature": 0,
                "seed": 42,
                "enable_thinking": False,
                "max_tokens": MAX_COMPLETION_TOKENS,
            },
            "hosted_calls": 0,
            "api_key_required": False,
            "request_endpoint": API_BASE,
            "proxy_environment_used": False,
        },
        "execution": {
            "provider_calls": 1,
            "same_request_retries": 0,
            "batch_subdivision": 0,
            "request_sha256": _sha(request_bytes),
            "assistant_content_sha256": _sha(content.encode("utf-8")),
            "prompt_tokens": prompt_tokens,
            "dynamic_schema_tokens": schema_tokens,
            "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
            "elapsed_ms": elapsed_ms,
            "truncated": False,
        },
        "frozen_artifacts": {
            RESPONSE_PATH.name: _verify_file(RESPONSE_PATH),
            PROPOSALS_PATH.name: proposal_sha,
        },
        "memory_store_mutations": 0,
        "reader_answer_calls": 0,
        "hosted_calls": 0,
        "benchmark_scoring": False,
    }
    manifest_sha = _freeze(RUN_MANIFEST_PATH, _json_bytes(manifest))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "records": len(records),
                "grounded": sum(row["identity_status"] == "GROUNDED" for row in records),
                "unresolved": sum(row["identity_status"] == "UNRESOLVED" for row in records),
                "run_manifest_sha256": manifest_sha,
            },
            sort_keys=True,
        )
    )
    return manifest


def analyze() -> dict[str, Any]:
    manifest = _read_frozen_json(RUN_MANIFEST_PATH)
    if manifest.get("status") != "PROPOSALS_FROZEN_AWAITING_DIAGNOSTIC_JOIN":
        raise PilotError("proposals_not_frozen_before_analysis")
    if manifest.get("code_sha256") != {
        "runner": _sha_file(Path(__file__)),
        "span_identity": _sha_file(Path(span_identity.__file__)),
        "runtime_verifier": _sha_file(Path(b0p_runner.__file__)),
    }:
        raise PilotError("code_changed_after_proposal_freeze")
    proposals = _read_jsonl(PROPOSALS_PATH)
    cases = _read_frozen_json(CASE_PATH)
    proposal_by_id = {row["memory_id"]: row for row in proposals}
    if len(proposal_by_id) != len(proposals):
        raise PilotError("duplicate_proposal_memory_id")
    case_results = []
    for case in cases["cases"]:
        ids = case["memory_ids"]
        rows = [proposal_by_id[mid] for mid in ids]
        keys = [tuple(row["slot_key"]) if row["slot_key"] is not None else None for row in rows]
        expected = case["expected"]
        if expected == "SAME_SLOT":
            passed = all(row["slot_candidate"] for row in rows) and len(set(keys)) == 1
        elif expected == "NOT_SAME_SLOT":
            passed = not (all(row["slot_candidate"] for row in rows) and len(set(keys)) == 1)
        elif expected == "NO_SLOT":
            passed = all(not row["slot_candidate"] for row in rows)
        elif expected == "BOTH_UNRESOLVED":
            passed = all(row["identity_status"] == "UNRESOLVED" for row in rows)
        else:
            raise PilotError(f"unknown_expected_relation:{expected}")
        case_results.append(
            {
                "case_id": case["case_id"],
                "memory_ids": ids,
                "expected": expected,
                "observed_slot_keys": keys,
                "identity_statuses": [row["identity_status"] for row in rows],
                "property_kinds": [row["property_kind"] for row in rows],
                "validation_reasons": [row["validation_reasons"] for row in rows],
                "passed": passed,
            }
        )
    valid_spans = sum(
        row["identity_status"] == "GROUNDED" for row in proposals
    )
    result = {
        "schema_version": 1,
        "stage": RUN_ID,
        "classification": "19_RECORD_DEVELOPMENT_CONTROL_PILOT_NOT_BENCHMARK_EVIDENCE",
        "input_sha256": manifest["proposal_inputs_sha256"],
        "proposal_sha256": _verify_file(PROPOSALS_PATH),
        "model_calls": manifest["execution"]["provider_calls"],
        "hosted_calls": manifest["hosted_calls"],
        "records": len(proposals),
        "grounded_identity_count": valid_spans,
        "unresolved_identity_count": len(proposals) - valid_spans,
        "case_count": len(case_results),
        "case_pass_count": sum(row["passed"] for row in case_results),
        "case_results": case_results,
        "pilot_pass": all(row["passed"] for row in case_results),
        "memory_store_mutations": 0,
        "reader_answer_calls": 0,
        "benchmark_scoring": False,
        "b1_ready": False,
    }
    _freeze(RESULT_PATH, _json_bytes(result))
    report = [
        "# MEM-3B0Q-R2 Span-Grounded Identity Pilot",
        "",
        "Status: development control pilot only; not benchmark evidence or a final mechanism freeze.",
        "",
        f"- Records: {result['records']}; grounded: {result['grounded_identity_count']}; unresolved: {result['unresolved_identity_count']}.",
        f"- Control cases: {result['case_pass_count']}/{result['case_count']} passed.",
        f"- Local Qwen calls: {result['model_calls']}; hosted calls: {result['hosted_calls']}; MemoryStore mutations: 0.",
        "",
        "## Case outcomes",
        "",
        "| Case | Expected | Passed | Slot keys |",
        "|---|---|---:|---|",
    ]
    for row in case_results:
        keys = "; ".join(str(key) if key is not None else "UNRESOLVED" for key in row["observed_slot_keys"])
        report.append(f"| {row['case_id']} | {row['expected']} | {row['passed']} | {keys} |")
    report.extend(
        [
            "",
            f"`MEM3B0Q_SPAN_IDENTITY_PILOT={'YES' if result['pilot_pass'] else 'NO'}`.",
            "`MEM3B1_READY=NO`. This pilot never decides UPDATE/DELETE, current validity, or historical validity. No public LongMemEval or MedMemoryBench claim is made.",
            "",
        ]
    )
    _freeze(REPORT_PATH, "\n".join(report).encode("utf-8"))
    result["marker"] = f"MEM3B0Q_SPAN_IDENTITY_PILOT={'YES' if result['pilot_pass'] else 'NO'}"
    print(json.dumps({"marker": result["marker"], "passed": result["case_pass_count"], "cases": result["case_count"]}, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "generate", "analyze"))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "generate":
        generate()
    else:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
