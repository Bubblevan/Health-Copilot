"""One-case DEV diagnostic for a source-grounded gym frequency revision chain."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.research.memory import final_reader_contract
from tools.research.memory import run_mem3b1_instagram_snapshot_diagnostic_v1 as common
from tools.research.memory.revision_state_materializer_v1 import TemporalProposition, materialize


ROOT = common.ROOT
SOURCE_RUN = common.SOURCE_RUN
OUTPUT_RUN = ROOT / "runs/memory/mem3/mem3b1-gym-change-chain-diagnostic-v1"
QUESTION_ID = "c4ea545c"
OLD_ID = "m10flat3-5f18753132f42768234fb06a27d804fbe9e026fb5d76bac3d152fadd16345273"
CURRENT_ID = "m10flat3-9145d242341cbefbade9c64d1ba786f6f51f87b458f792fe5e8261dc17cadc24"
SLOT = (f"longmemeval:{QUESTION_ID}", "SELF", "GYM_ROUTINE", "SESSIONS_PER_WEEK")


def _freeze_json(name: str, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    return common._freeze_artifact(OUTPUT_RUN / name, payload)


def _load_case() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    names = (
        "context_bundles.jsonl",
        "predictions.jsonl",
        "reader_call_ledger.jsonl",
        "flatprop_inventory.jsonl",
        "materialization_ledger.jsonl",
    )
    hashes = {name: common._verify_sidecar(SOURCE_RUN / name) for name in names}
    manifest = common._read_json(SOURCE_RUN / "run_manifest.json")
    if manifest.get("status") != "COMPLETE" or manifest.get("stage") != SOURCE_RUN.name:
        raise RuntimeError("source FlatProp diagnostic is not complete")
    contexts = common._read_jsonl(SOURCE_RUN / "context_bundles.jsonl")
    predictions = common._read_jsonl(SOURCE_RUN / "predictions.jsonl")
    calls = common._read_jsonl(SOURCE_RUN / "reader_call_ledger.jsonl")
    inventory = common._read_jsonl(SOURCE_RUN / "flatprop_inventory.jsonl")
    ledger = common._read_jsonl(SOURCE_RUN / "materialization_ledger.jsonl")
    context = next(row for row in contexts if row["question_id"] == QUESTION_ID)
    baseline = next(row for row in predictions if row["question_id"] == QUESTION_ID)
    baseline_call = next(row for row in calls if row["question_id"] == QUESTION_ID)
    proposition_rows = [row for row in inventory if row.get("memory_id") in {OLD_ID, CURRENT_ID}]
    ledger_rows = [row for row in ledger if row.get("memory_id") in {OLD_ID, CURRENT_ID}]
    if (
        baseline.get("predicted") != "None"
        or baseline_call.get("success") is not True
        or baseline_call.get("hosted_call") is not False
        or {row["memory_id"] for row in proposition_rows} != {OLD_ID, CURRENT_ID}
        or {row["memory_id"] for row in ledger_rows} != {OLD_ID, CURRENT_ID}
    ):
        raise RuntimeError("frozen gym baseline or source pair identity does not match")
    return context, baseline, {row["memory_id"]: row for row in proposition_rows}, {
        "source_hashes": hashes,
        "ledger_by_id": {row["memory_id"]: row for row in ledger_rows},
        "source_run_manifest_sha256": common._sha_file(SOURCE_RUN / "run_manifest.json"),
    }


def _observation(memory_id: str, row: dict[str, Any], ledger: dict[str, Any]) -> TemporalProposition:
    text = row["proposition_text"]
    if memory_id == OLD_ID:
        value = "Tuesday, Thursday, and Saturday"
        expected_quote = "I go to the gym on Tuesdays, Thursdays, and Saturdays."
    else:
        value = "four times a week"
        expected_quote = "four times a week, actually"
    quotes = [evidence["evidence_quote"] for evidence in row.get("evidence", [])]
    if (
        row.get("source_authority") != "user"
        or ledger.get("source_type") != "session_derived"
        or value not in text
        or not any(expected_quote in quote for quote in quotes)
        or not isinstance(ledger.get("valid_from"), str)
    ):
        raise RuntimeError(f"gym frequency value lacks direct user provenance: {memory_id}")
    return TemporalProposition(
        memory_id=memory_id,
        scope_id=f"longmemeval:{QUESTION_ID}",
        owner_id="SELF",
        object_id="GYM_ROUTINE",
        attribute_id="SESSIONS_PER_WEEK",
        value_text=value,
        observed_at=common._dt(ledger["valid_from"]),
        source_text=text,
        source_session_id=row["source_session_id"],
        source_turn_ids=tuple(
            f"{row['source_session_id']}:{turn}" for turn in row.get("source_turn_indices", [])
        ),
        provenance_hash=common._sha_bytes(text.encode("utf-8")),
        cardinality="SINGLE_VALUE_AT_A_TIME",
        temporal_basis="CURRENT_SNAPSHOT",
    )


def _weekly_frequency(value: str) -> tuple[int, str]:
    weekdays = re.findall(
        r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)s?\b",
        value,
        re.IGNORECASE,
    )
    if weekdays:
        return len({day.casefold().rstrip("s") for day in weekdays}), "derived from listed weekdays"
    numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
    match = re.search(r"\b(one|two|three|four|five|six|seven|\d+) times? (?:a|per) week\b", value, re.I)
    if match:
        token = match.group(1).casefold()
        return (int(token) if token.isdigit() else numbers[token]), "stated by user"
    raise RuntimeError(f"unsupported weekly-frequency surface form: {value!r}")


def _render_pair(
    items: list[dict[str, Any]],
    state: Any,
    proposition_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    chain = state.change_chain(SLOT)
    old = next(row for row in chain if row.source_memory_ids == (OLD_ID,))
    current = next(row for row in chain if row.source_memory_ids == (CURRENT_ID,))
    old_count, old_basis = _weekly_frequency(old.value_text)
    current_count, current_basis = _weekly_frequency(current.value_text)
    if old_count != 3 or current_count != 4 or old.status != "SUPERSEDED" or current.status != "ACTIVE":
        raise RuntimeError("materialized gym revision chain did not retain 3/week then 4/week")
    source_map = {OLD_ID: old, CURRENT_ID: current}
    rendered = []
    for item in items:
        copied = dict(item)
        memory_id = copied.get("memory_id")
        if memory_id in source_map:
            record = source_map[memory_id]
            source = proposition_rows[memory_id]
            count, basis = (old_count, old_basis) if memory_id == OLD_ID else (current_count, current_basis)
            phase = "HISTORICAL STATE" if memory_id == OLD_ID else "CURRENT STATE"
            copied["text"] = "\n".join(
                (
                    f"[{phase}]",
                    "Subject: SELF (the user)",
                    "Object: gym routine",
                    "Attribute: sessions per week",
                    f"Value: {count} sessions/week ({basis})",
                    f"Valid from: {record.valid_from}",
                    f"Source: user-reported session {source['source_session_id']}",
                    f"Source statement: {source['proposition_text']}",
                )
            )
            copied["reader_projection_format"] = "revision_chain_change_v1"
        rendered.append(copied)
    if sum(item.get("reader_projection_format") == "revision_chain_change_v1" for item in rendered) != 2:
        raise RuntimeError("CHANGE projection must expose exactly both revision-chain states")
    return rendered


def _load_gold() -> str:
    manifest = common._read_json(ROOT / "docs/research/memory/dataset_manifest.json")
    entry = next(row for row in manifest["datasets"] if row.get("dataset_id") == "longmemeval_s")
    path = next((candidate for candidate in common.DATASET_PATH_CANDIDATES if candidate.is_file()), None)
    if path is None or path.stat().st_size != entry["expected_size_bytes"]:
        raise RuntimeError("manifest-pinned LongMemEval-S dataset is unavailable or changed")
    if common._sha_file(path) != entry["expected_sha256"]:
        raise RuntimeError("LongMemEval-S dataset SHA does not match its manifest")
    row = next(row for row in common._iter_json_array(path) if row.get("question_id") == QUESTION_ID)
    if row.get("question_type") != "knowledge-update" or row.get("answer") != "Yes":
        raise RuntimeError("frozen DEV gym question/gold identity changed")
    return row["answer"]


def run_once() -> dict[str, Any]:
    if OUTPUT_RUN.exists():
        raise RuntimeError(f"append-never output already exists: {OUTPUT_RUN}")
    context_row, baseline, proposition_rows, lineage = _load_case()
    items = context_row["context_bundle"]["items"]
    by_id = {item["memory_id"]: item for item in items}
    if not {OLD_ID, CURRENT_ID}.issubset(by_id):
        raise RuntimeError("frozen retrieval bundle does not contain both gym revisions")
    observations = [
        _observation(memory_id, proposition_rows[memory_id], lineage["ledger_by_id"][memory_id])
        for memory_id in (OLD_ID, CURRENT_ID)
    ]
    state = materialize(observations)
    historical = state.as_of(SLOT, min(row.observed_at for row in observations))
    current = state.current(SLOT)
    if (
        len(historical) != 1
        or historical[0].source_memory_ids != (OLD_ID,)
        or len(current) != 1
        or current[0].source_memory_ids != (CURRENT_ID,)
        or len(state.change_chain(SLOT)) != 2
    ):
        raise RuntimeError("current/as-of/change materializations disagree")
    projected_items = _render_pair(items, state, proposition_rows)
    serialized_context = common._serialize_context(projected_items)
    original_user = context_row["reader_messages"][1]["content"]
    match = re.search(r"Current Date: (.+?)\nQuestion: (.+)$", original_user, re.DOTALL)
    if not match:
        raise RuntimeError("frozen query/date is malformed")
    contract, contract_sha, system_template, user_template = final_reader_contract.load_final_reader_contract()
    messages = final_reader_contract.build_reader_messages(
        match.group(2), match.group(1), serialized_context,
        system_template=system_template, user_template=user_template,
    )
    if (
        contract_sha != context_row["shared_reader_contract_sha256"]
        or messages[0]["content"] != context_row["reader_messages"][0]["content"]
        or messages[1]["content"].split("Current Date: ", 1)[-1]
        != original_user.split("Current Date: ", 1)[-1]
        or [item["memory_id"] for item in projected_items] != [item["memory_id"] for item in items]
    ):
        raise RuntimeError("CHANGE projection changed the query/contract or retrieval set/order")
    gold = _load_gold()
    projection = {
        "schema_version": 1,
        "question_id": QUESTION_ID,
        "parent_context_bundle_sha256": context_row["context_bundle_sha256"],
        "query_mode": "CHANGE",
        "projection_policy": "preserve both source-grounded states; replace only their JSON wrappers with temporal slot fields",
        "items": projected_items,
        "serialized_context": serialized_context,
    }
    projection_bytes = common._canonical(projection)
    projection_sha = common._sha_bytes(projection_bytes)
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        runtime = common._runtime_preflight(client)
        context_tokens = common._preflight_token_count(client, serialized_context)
        rendered = client.post(
            "http://127.0.0.1:8081/apply-template",
            json={"messages": messages, "add_generation_prompt": True,
                  "chat_template_kwargs": {"enable_thinking": False}},
        )
        rendered.raise_for_status()
        prompt = rendered.json().get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("frozen reader template render failed")
        prompt_tokens = common._preflight_token_count(client, prompt)
        if prompt_tokens + 256 >= 131072:
            raise RuntimeError("frozen Qwen reader context budget exceeded")
    request = {
        "model": common.MODEL,
        "messages": messages,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request_bytes = common._canonical(request)
    request_sha = common._sha_bytes(request_bytes)
    OUTPUT_RUN.mkdir(parents=True, exist_ok=False)
    projection_sha = common._freeze_artifact(OUTPUT_RUN / "context_projection.json", projection_bytes)
    materialized_sha = _freeze_json(
        "materialized_state.json",
        {
            "records": [row.__dict__ for row in state.records],
            "input_sha256": state.input_sha256,
            "output_sha256": state.output_sha256,
            "current_memory_ids": [value for row in current for value in row.source_memory_ids],
            "historical_memory_ids": [value for row in historical for value in row.source_memory_ids],
            "change_chain_memory_ids": [value for row in state.change_chain(SLOT) for value in row.source_memory_ids],
        },
    )
    request_file_sha = common._freeze_artifact(OUTPUT_RUN / "reader_request.json", request_bytes + b"\n")
    reservation_sha = _freeze_json(
        "run_reservation.json",
        {
            "run_id": OUTPUT_RUN.name,
            "status": "ONE_LOCAL_POST_RESERVED",
            "question_id": QUESTION_ID,
            "query_mode": "CHANGE",
            "reader_model": common.MODEL,
            "reader_role": "answer model",
            "memory_system": "FlatProp + source-grounded deterministic revision materializer + CHANGE projection",
            "embedding_model": "NONE (reused frozen retrieval output)",
            "judge_model": "NONE",
            "hosted_calls": 0,
            "local_posts_authorized": 1,
            "request_sha256": request_sha,
            "request_file_sha256": request_file_sha,
            "context_projection_sha256": projection_sha,
            "materialized_state_sha256": materialized_sha,
            "reader_contract_sha256": contract_sha,
            "source_hashes": lineage["source_hashes"],
            "runtime_preflight": runtime,
        },
    )
    with httpx.Client(timeout=180.0, trust_env=False) as client:
        response = client.post(
            f"{common.ENDPOINT}/chat/completions",
            content=request_bytes,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        response_bytes = response.content
    response_sha = common._freeze_artifact(OUTPUT_RUN / "reader_response.json", response_bytes)
    payload = json.loads(response_bytes)
    choice = payload["choices"][0]
    predicted = choice.get("message", {}).get("content")
    usage = payload.get("usage", {})
    if not isinstance(predicted, str):
        raise RuntimeError("local reader returned no answer text")
    predicted = predicted.strip()
    baseline_metrics = common._answer_metrics(baseline["predicted"], gold)
    metrics = common._answer_metrics(predicted, gold)
    status = "OK" if choice.get("finish_reason") == "stop" and usage.get("prompt_tokens") == prompt_tokens else "INFRA_FAILURE"
    result = {
        "question_id": QUESTION_ID,
        "question_type": "knowledge-update",
        "query_mode": "CHANGE",
        "question": match.group(2),
        "gold": gold,
        "baseline_prediction": baseline["predicted"],
        "change_chain_prediction": predicted,
        "baseline_metrics": baseline_metrics,
        "change_chain_metrics": metrics,
        "delta_f1": metrics["f1"] - baseline_metrics["f1"],
        "context_reader_tokens_before": context_row["context_bundle"]["context_reader_tokens"],
        "context_reader_tokens_after": context_tokens,
        "reader_prompt_tokens_before": context_row["reader_prompt_tokens_preflight"],
        "reader_prompt_tokens_after_preflight": prompt_tokens,
        "reader_prompt_tokens_after_server": usage.get("prompt_tokens"),
        "context_items_before": len(items),
        "context_items_after": len(projected_items),
        "historical_memory_id": OLD_ID,
        "current_memory_id": CURRENT_ID,
        "historical_frequency": "3 sessions/week (derived from three user-listed weekdays)",
        "current_frequency": "4 sessions/week (stated by user)",
        "historical_state_preserved_for_change_query": True,
        "retrieval_unchanged": True,
        "reader_contract_sha256": contract_sha,
        "finish_reason": choice.get("finish_reason"),
        "quality_status": status,
        "claim_boundary": "one frozen public LongMemEval DEV case; manually adjudicated slot; deterministic frequency normalization; no aggregate claim",
    }
    result_sha = _freeze_json("diagnostic_result.json", result)
    call = {
        **runtime,
        "role": "reader_answer",
        "question_id": QUESTION_ID,
        "request_sha256": request_sha,
        "response_sha256": response_sha,
        "local_posts": 1,
        "retry_count": 0,
        "prompt_tokens_preflight": prompt_tokens,
        "prompt_tokens_server": usage.get("prompt_tokens"),
        "prompt_tokens_match": usage.get("prompt_tokens") == prompt_tokens,
        "context_reader_tokens": context_tokens,
        "completion_tokens_server": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "quality_status": status,
        "hosted_api_calls": 0,
        "required_api_key": "NONE",
    }
    call_sha = _freeze_json("reader_call.json", call)
    manifest = {
        "run_id": OUTPUT_RUN.name,
        "status": "COMPLETE" if status == "OK" else "INFRA_FAILURE",
        "question_ids": [QUESTION_ID],
        "local_posts": 1,
        "hosted_calls": 0,
        "judge_calls": 0,
        "embedding_calls": 0,
        "retries": 0,
        "source_run_manifest_sha256": lineage["source_run_manifest_sha256"],
        "source_hashes": lineage["source_hashes"],
        "projection_sha256": projection_sha,
        "materialized_state_sha256": materialized_sha,
        "request_sha256": request_sha,
        "request_file_sha256": request_file_sha,
        "response_sha256": response_sha,
        "diagnostic_result_sha256": result_sha,
        "reader_call_sha256": call_sha,
        "reservation_sha256": reservation_sha,
        "prediction_sha_freeze": "PASS" if response_sha and result_sha else "FAIL",
    }
    _freeze_json("run_manifest.json", manifest)
    return result


if __name__ == "__main__":
    print(json.dumps(run_once(), ensure_ascii=False, sort_keys=True, indent=2))
