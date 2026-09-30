"""Read-only verification and identity helpers for E5-B3 evidence replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.rag_e5.counterfactual import ACTION_ORDER, verify_complete_arm

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIVATE_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b1")
DEFAULT_CORPUS_ROOT = Path(
    r"D:\MyLab\Jianli\external\rag_e5\e5a3\corpus\public_health_plus_guideline"
)
DEFAULT_B2_ARTIFACT_ROOT = Path(r"D:\MyLab\Jianli\external\rag_e5\e5b2")
EXPECTED_TASK_SET_SHA256 = "7ac2d95f98efc6fcced665f93977c76c4d219af0cfe7e16f05b1fdd67358979a"
EXPECTED_STATE_SET_SHA256 = "860f5da64a6b94790944a5bd2d50aa4977a098bc51d1b6f822321bf4fbbd1d1c"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path.name}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"expected JSON objects in {path.name}")
    return rows


@dataclass(frozen=True, slots=True)
class FrozenB2ArmContext:
    """Only evidence/action inputs and frozen cost metadata; excludes B2 reader output."""

    case_id: str
    action: str
    b2_run_id: str
    source_arm_file_sha256: str
    source_completion_sha256: str
    question_sha256: str
    state_packet_sha256: str | None
    runtime_capability_context_sha256: str
    retrieval_channels: Mapping[str, Any]
    retrieval_ranking: tuple[dict[str, Any], ...]
    retrieval_ranking_sha256: str
    supplied_chunks: tuple[dict[str, Any], ...]
    supplied_chunk_ids_sha256: str
    supplied_chunks_sha256: str
    source_retrieval_calls: int
    source_retrieval_latency_ms: float
    source_bridge_call_count: int
    source_bridge_latency_ms: float
    source_bridge_input_tokens: int | None
    source_bridge_output_tokens: int | None
    source_bridge_response_sha256: str | None
    source_bridge_fallback: bool
    bridge_artifact: Mapping[str, Any] | None

    def reuse_manifest_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "action": self.action,
            "b2_run_id": self.b2_run_id,
            "source_arm_file_sha256": self.source_arm_file_sha256,
            "source_completion_sha256": self.source_completion_sha256,
            "retrieval_ranking_sha256": self.retrieval_ranking_sha256,
            "retrieval_channels_sha256": canonical_sha256(self.retrieval_channels),
            "supplied_chunk_ids_sha256": self.supplied_chunk_ids_sha256,
            "supplied_chunks_sha256": self.supplied_chunks_sha256,
            "strong_bridge_response_sha256": self.source_bridge_response_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenB2Inputs:
    lock: dict[str, Any]
    execution_manifest: dict[str, Any]
    runtime_cases: tuple[dict[str, Any], ...]
    state_packets_by_ref: Mapping[str, dict[str, Any]]
    arm_contexts: tuple[FrozenB2ArmContext, ...]
    b2_lock_file_sha256: str
    b2_execution_manifest_file_sha256: str
    artifact_set_sha256: str
    reuse_manifest_sha256: str
    reuse_manifest_rows: tuple[dict[str, Any], ...]


def load_and_verify_b2_inputs(
    *,
    private_root: Path = DEFAULT_PRIVATE_ROOT,
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    b2_artifact_root: Path = DEFAULT_B2_ARTIFACT_ROOT,
    b2_lock_path: Path = ROOT / "runs/rag_e5/e5b2_protocol_lock.json",
    b2_execution_manifest_path: Path = ROOT / "runs/rag_e5/e5b2_execution_manifest.json",
) -> FrozenB2Inputs:
    """Verify B2 and project only its immutable task/retrieval evidence into B3."""
    lock = read_json(b2_lock_path)
    lock_body = {key: value for key, value in lock.items() if key != "counterfactual_lock_sha256"}
    lock_sha = lock.get("counterfactual_lock_sha256")
    if not isinstance(lock_sha, str) or canonical_sha256(lock_body) != lock_sha:
        raise ValueError("B2 protocol lock self-hash mismatch")
    if lock.get("status") != "FROZEN_PROTOCOL_NO_OUTCOMES" or lock.get("expected_arms") != 180:
        raise ValueError("B2 inputs are not the frozen 180-arm protocol")
    if lock.get("202608_opened") is not False:
        raise ValueError("B2 lock crosses the excluded 202608 cohort")
    if lock.get("ordered_case_ids_sha256") != EXPECTED_TASK_SET_SHA256:
        raise ValueError("B2 task-set identity differs from the user-frozen SHA")
    if lock.get("state_packet_set_sha256") != EXPECTED_STATE_SET_SHA256:
        raise ValueError("B2 state-set identity differs from the user-frozen SHA")

    for source in lock.get("locked_code", []):
        source_path = ROOT / source["path"]
        if not source_path.is_file() or sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"B2 locked code changed: {source['path']}")

    runtime_path = private_root / "runtime_cases.jsonl"
    if sha256_file(runtime_path) != lock.get("runtime_cases_file_sha256"):
        raise ValueError("frozen B2 runtime case file changed")
    runtime_cases = read_jsonl(runtime_path)
    case_ids = [row.get("case_id") for row in runtime_cases]
    if len(runtime_cases) != 60 or canonical_sha256(case_ids) != lock["ordered_case_ids_sha256"]:
        raise ValueError("runtime cases differ from the B2 task identity/order")
    question_set_sha = canonical_sha256(
        [{"case_id": row["case_id"], "question": row["question"]} for row in runtime_cases]
    )
    if question_set_sha != lock.get("question_set_sha256"):
        raise ValueError("frozen B2 question set changed")
    capability = read_json(private_root / "capability_context.json")
    if canonical_sha256(capability) != lock.get("runtime_capability_context_sha256"):
        raise ValueError("frozen B2 capability context changed")

    state_by_ref: dict[str, dict[str, Any]] = {}
    states_by_user: dict[str, dict[str, Any]] = {}
    for case in runtime_cases:
        state_ref = case.get("state_packet_ref")
        if state_ref is None:
            if case.get("decision_boundary") is not None:
                raise ValueError("state-free B2 case contains a decision boundary")
            continue
        relative = Path(state_ref)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("state packet reference escapes the frozen B2 private root")
        packet = read_json(private_root / relative)
        if packet.get("user_id") != case.get("user_id"):
            raise ValueError("B2 runtime case and state packet user identities differ")
        if packet.get("decision_boundary") != case.get("decision_boundary"):
            raise ValueError("B2 runtime case and state packet cutoff identities differ")
        state_by_ref[state_ref] = packet
        states_by_user[packet["user_id"]] = packet
    packet_identity = [
        {
            "user_id": packet["user_id"],
            "packet_sha256": packet["packet_sha256"],
            "decision_boundary": packet["decision_boundary"]["naive_timestamp"],
        }
        for packet in sorted(states_by_user.values(), key=lambda item: item["user_id"])
    ]
    if canonical_sha256(packet_identity) != EXPECTED_STATE_SET_SHA256:
        raise ValueError("frozen B2 state packet set changed")
    chunks_path = corpus_root / "chunks.jsonl"
    if sha256_file(chunks_path) != lock.get("external_corpus_chunks_sha256"):
        raise ValueError("approved E5 corpus changed after B2 freeze")

    execution = read_json(b2_execution_manifest_path)
    execution_body = {
        key: value for key, value in execution.items() if key != "execution_manifest_sha256"
    }
    if canonical_sha256(execution_body) != execution.get("execution_manifest_sha256"):
        raise ValueError("B2 execution manifest self-hash mismatch")
    if execution.get("status") != "ALL_ARMS_FROZEN_BEFORE_SCORING":
        raise ValueError("B2 execution manifest is not fully frozen")
    if execution.get("counterfactual_lock_sha256") != lock_sha:
        raise ValueError("B2 execution manifest belongs to a different protocol")
    if (
        execution.get("completed_arms") != 180
        or execution.get("expected_arms") != 180
        or execution.get("reader_calls") != 180
        or execution.get("bridge_calls") != 60
        or execution.get("model_calls") != 240
        or execution.get("202608_opened") is not False
    ):
        raise ValueError("B2 execution manifest violates its frozen counts or cohort boundary")
    artifacts = execution.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 180:
        raise ValueError("B2 execution manifest must identify 180 source artifacts")
    artifact_set_sha = canonical_sha256(artifacts)
    if artifact_set_sha != execution.get("artifact_set_sha256"):
        raise ValueError("B2 artifact-set hash mismatch")
    if canonical_sha256([row["run_id"] for row in artifacts]) != execution.get(
        "run_id_set_sha256"
    ):
        raise ValueError("B2 run-ID set hash mismatch")
    if canonical_sha256(case_ids) != EXPECTED_TASK_SET_SHA256:
        raise ValueError("runtime task set changed before B3")

    expected_keys = [(case_id, action) for case_id in sorted(case_ids) for action in ACTION_ORDER]
    observed_keys = [(row.get("case_id"), row.get("action")) for row in artifacts]
    if observed_keys != expected_keys:
        raise ValueError("B2 source artifact order/identity is incomplete or unexpected")
    case_by_id = {row["case_id"]: row for row in runtime_cases}
    contexts: list[FrozenB2ArmContext] = []
    for entry in artifacts:
        case_id, action, run_id = entry["case_id"], entry["action"], entry["run_id"]
        path = b2_artifact_root / "arms" / case_id / action / f"{run_id}.json"
        raw_sha = sha256_file(path)
        if raw_sha != entry.get("file_sha256"):
            raise ValueError(f"B2 source arm file changed: {case_id}/{action}")
        source_arm = verify_complete_arm(
            path, expected_run_id=run_id, expected_lock_sha256=lock_sha
        )
        if (
            source_arm.get("status") != "COMPLETE"
            or source_arm.get("case_id") != case_id
            or source_arm.get("action") != action
            or source_arm.get("completion_sha256") != entry.get("completion_sha256")
        ):
            raise ValueError("B2 source arm completion identity/hash mismatch")
        case = case_by_id[case_id]
        question_sha = hashlib.sha256(case["question"].encode("utf-8")).hexdigest()
        if source_arm.get("question_sha256") != question_sha:
            raise ValueError("B2 arm question identity differs from its frozen runtime case")
        state_ref = case.get("state_packet_ref")
        packet = state_by_ref.get(state_ref) if state_ref else None
        expected_packet_sha = canonical_sha256(packet) if packet is not None else None
        if source_arm.get("state_packet_sha256") != expected_packet_sha:
            raise ValueError("B2 arm state input differs from its frozen runtime state packet")
        if source_arm.get("runtime_capability_context_sha256") != lock[
            "runtime_capability_context_sha256"
        ]:
            raise ValueError("B2 arm capability context differs from the frozen B2 lock")

        retrieval = source_arm.get("retrieval")
        ranking = retrieval.get("ranking", []) if isinstance(retrieval, Mapping) else []
        chunks = source_arm.get("supplied_chunks", [])
        if not isinstance(ranking, list) or not isinstance(chunks, list):
            raise TypeError("B2 ranking and supplied chunks must be lists")
        supplied_ids = [row.get("chunk_id") for row in chunks]
        ranked_top_ids = [row.get("chunk_id") for row in ranking[:5]]
        if not isinstance(retrieval, Mapping):
            raise TypeError("B2 arm is missing its retrieval-context record")
        if retrieval.get("ranking_sha256") != canonical_sha256(ranking):
            raise ValueError("B2 stored retrieval-ranking hash mismatch")
        if retrieval.get("supplied_chunk_ids") != supplied_ids:
            raise ValueError("B2 retrieval/supplied chunk ID lists differ")
        if retrieval.get("supplied_chunk_ids_sha256") != canonical_sha256(supplied_ids):
            raise ValueError("B2 stored supplied-chunk ID hash mismatch")
        if action == "OFF":
            if (
                ranking
                or chunks
                or retrieval.get("channels") != {}
                or source_arm.get("retrieval_calls") != 0
            ):
                raise ValueError("B2 OFF arm unexpectedly contains retrieved evidence")
        elif len(chunks) != 5 or supplied_ids != ranked_top_ids:
            raise ValueError("B2 supplied evidence is not the exact frozen top-five ranking")

        bridge = source_arm.get("bridge")
        bridge_call_count = source_arm.get("bridge_call_count")
        if action == "STRONG":
            if not isinstance(bridge, Mapping) or bridge_call_count != 1:
                raise ValueError("B2 STRONG bridge artifact is missing")
            raw_response = bridge.get("raw_response")
            if not isinstance(raw_response, Mapping):
                raise ValueError("B2 STRONG bridge response is missing")
            response_sha = hashlib.sha256(
                json.dumps(
                    raw_response, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if response_sha != bridge.get("response_sha256"):
                raise ValueError("B2 STRONG bridge response hash mismatch")
            request_sha = hashlib.sha256(
                json.dumps(
                    bridge.get("request"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if request_sha != bridge.get("request_sha256"):
                raise ValueError("B2 STRONG bridge request hash mismatch")
            bridge_response_sha = response_sha
            bridge_input_tokens = raw_response.get("input_tokens")
            bridge_output_tokens = raw_response.get("output_tokens")
            bridge_fallback = bool(bridge.get("fallback_original_query"))
        else:
            if bridge is not None or bridge_call_count != 0:
                raise ValueError("non-STRONG B2 arm unexpectedly contains a bridge")
            bridge_response_sha = None
            bridge_input_tokens = None
            bridge_output_tokens = None
            bridge_fallback = False

        contexts.append(
            FrozenB2ArmContext(
                case_id=case_id,
                action=action,
                b2_run_id=run_id,
                source_arm_file_sha256=raw_sha,
                source_completion_sha256=str(entry["completion_sha256"]),
                question_sha256=question_sha,
                state_packet_sha256=expected_packet_sha,
                runtime_capability_context_sha256=str(
                    source_arm["runtime_capability_context_sha256"]
                ),
                retrieval_channels=dict(retrieval.get("channels", {})),
                retrieval_ranking=tuple(ranking),
                retrieval_ranking_sha256=canonical_sha256(ranking),
                supplied_chunks=tuple(chunks),
                supplied_chunk_ids_sha256=canonical_sha256(supplied_ids),
                supplied_chunks_sha256=canonical_sha256(chunks),
                source_retrieval_calls=int(source_arm.get("retrieval_calls", 0)),
                source_retrieval_latency_ms=float(source_arm.get("retrieval_latency_ms", 0.0)),
                source_bridge_call_count=int(bridge_call_count),
                source_bridge_latency_ms=float(source_arm.get("bridge_latency_ms", 0.0)),
                source_bridge_input_tokens=bridge_input_tokens,
                source_bridge_output_tokens=bridge_output_tokens,
                source_bridge_response_sha256=bridge_response_sha,
                source_bridge_fallback=bridge_fallback,
                bridge_artifact=dict(bridge) if isinstance(bridge, Mapping) else None,
            )
        )
    if len(contexts) != 180:
        raise ValueError("B3 requires all 180 verified B2 source contexts")
    reuse_rows = tuple(row.reuse_manifest_row() for row in contexts)
    return FrozenB2Inputs(
        lock=lock,
        execution_manifest=execution,
        runtime_cases=tuple(runtime_cases),
        state_packets_by_ref=state_by_ref,
        arm_contexts=tuple(contexts),
        b2_lock_file_sha256=sha256_file(b2_lock_path),
        b2_execution_manifest_file_sha256=sha256_file(b2_execution_manifest_path),
        artifact_set_sha256=artifact_set_sha,
        reuse_manifest_sha256=canonical_sha256(reuse_rows),
        reuse_manifest_rows=reuse_rows,
    )


def b3_run_id(case_id: str, action: str, b3_lock_sha256: str) -> str:
    if action not in ACTION_ORDER:
        raise ValueError(f"unsupported E5 action: {action}")
    return canonical_sha256([case_id, action, b3_lock_sha256])


def reader_v2_to_scorer(reader_output: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanical V2 projection; no alias repair or claim rewriting."""
    state_facts = reader_output.get("state_facts", [])
    guidance_facts = reader_output.get("guidance_facts", [])
    citations = sorted(
        {
            citation
            for fact in guidance_facts
            if isinstance(fact, Mapping)
            for citation in fact.get("citations", [])
        }
    )
    return {
        "state_facts": state_facts,
        "guidance_facts": [
            {"statement": fact.get("statement", "")}
            for fact in guidance_facts
            if isinstance(fact, Mapping)
        ],
        "citations": citations,
    }


__all__ = [
    "DEFAULT_B2_ARTIFACT_ROOT",
    "DEFAULT_CORPUS_ROOT",
    "DEFAULT_PRIVATE_ROOT",
    "EXPECTED_STATE_SET_SHA256",
    "EXPECTED_TASK_SET_SHA256",
    "FrozenB2ArmContext",
    "FrozenB2Inputs",
    "b3_run_id",
    "canonical_sha256",
    "load_and_verify_b2_inputs",
    "read_json",
    "read_jsonl",
    "reader_v2_to_scorer",
    "sha256_file",
]
