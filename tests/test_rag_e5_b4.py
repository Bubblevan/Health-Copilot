from __future__ import annotations

import inspect
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.rag_e5 import b4_evaluator, b4_execution
from eval.rag_e5.b4_execution import ACTION_ORDER, make_execution_plan, verify_protocol_lock
from eval.rag_e5.b4_guidance import (
    CONTEXT_SIZE,
    MAX_OUTPUT_TOKENS,
    GuidanceCompletion,
    chat_payload,
    render_guidance_prompt,
)
from eval.rag_e5.b4_materializer import (
    StateClaim,
    classify_runtime_task,
    extract_citations,
    materialize_final_response,
    materialize_state_claims,
    strip_aliases,
)


class FakeTokenizer:
    def __init__(self, count: int = 100) -> None:
        self.count = count
        self.prompts: list[str] = []

    def count_prompt_tokens(self, prompt: str) -> int:
        self.prompts.append(prompt)
        return self.count


class FakeUntrustedGuidanceClient:
    def slots(self) -> list[dict[str, bool]]:
        return [{"is_processing": False}]

    def complete(self, prompt: str) -> GuidanceCompletion:
        return GuidanceCompletion(
            text=(
                '{"action":"OFF","state_claims":[{"field":"body_weight",'
                '"value":"rising"}]} [E999]'
            ),
            finish_reason="stop",
            input_tokens=20,
            output_tokens=12,
            latency_ms=1.0,
            request_payload={"messages": [{"role": "user", "content": prompt}]},
        )


def _runtime_case(case_id: str, question: str, state_ref: str | None) -> dict[str, object]:
    return {
        "case_id": case_id,
        "user_id": case_id.split("-")[0],
        "question": question,
        "state_packet_ref": state_ref,
    }


def _frozen_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, list[dict[str, object]]]]:
    cases: list[dict[str, object]] = []
    packets: dict[str, dict[str, object]] = {}
    sources: dict[str, list[dict[str, object]]] = {}
    contexts = []
    for index in range(20):
        user = f"user{index:02d}"
        state_ref = f"state_packets/{user}.json"
        packets[state_ref] = {
            "packet_sha256": f"packet-{index}",
            "recent_measurement_trends": [{"metric": "body_weight", "trend": "falling"}],
        }
        rows = [
            _runtime_case(
                f"{user}-t0",
                "Describe only the recorded body weight trend. Use one of rising, falling, or stable; do not diagnose, infer a cause, or recommend treatment.",
                state_ref,
            ),
            _runtime_case(
                f"{user}-t1",
                "What general physical activity advice is recommended for adults to support health? Keep it general and separate it from any personal information.",
                None,
            ),
            _runtime_case(
                f"{user}-t2",
                "First, describe only the recorded body weight trend using rising, falling, or stable. Separately, state the general adult dietary-fat advice for reducing the risk of unhealthy weight gain. Do not connect the recorded trend to diet, cause, diagnosis, or treatment, and do not give individualized advice.",
                state_ref,
            ),
        ]
        cases.extend(rows)
        for case in rows:
            for action in ACTION_ORDER:
                evidence = []
                if action != "OFF":
                    evidence = [
                        {
                            "chunk_id": f"{case['case_id']}-{action}-chunk-{rank}",
                            "text": f"Frozen evidence passage {rank} for {action}.",
                            "source_id": "approved-source",
                            "recommendation_id": "rec-1",
                            "section_path": ["section", str(rank)],
                        }
                        for rank in range(1, 6)
                    ]
                source = SimpleNamespace(
                    case_id=case["case_id"],
                    action=action,
                    supplied_chunks=tuple(evidence),
                    source_retrieval_calls=int(action != "OFF"),
                    source_bridge_call_count=int(action == "STRONG"),
                    source_bridge_input_tokens=40 if action == "STRONG" else 0,
                    source_bridge_output_tokens=12 if action == "STRONG" else 0,
                    source_retrieval_latency_ms=1.5 if action != "OFF" else 0.0,
                    source_bridge_latency_ms=2.5 if action == "STRONG" else 0.0,
                    retrieval_ranking_sha256=f"ranking-{case['case_id']}-{action}",
                    supplied_chunks_sha256=f"chunks-{case['case_id']}-{action}",
                    supplied_chunk_ids_sha256=f"ids-{case['case_id']}-{action}",
                    reuse_manifest_row=lambda case_id=case["case_id"], a=action: {
                        "case_id": case_id,
                        "action": a,
                        "source_arm_file_sha256": f"file-{case_id}-{a}",
                    },
                )
                contexts.append(source)
                sources[f"{case['case_id']}:{action}"] = evidence
    frozen = SimpleNamespace(
        runtime_cases=tuple(cases),
        state_packets_by_ref=packets,
        arm_contexts=tuple(contexts),
        lock={"counterfactual_lock_sha256": "frozen-b2-lock"},
    )
    return frozen, sources


def test_state_claim_is_deterministic_and_only_requested_metric() -> None:
    question = "Describe only the recorded body weight trend."
    packet = {
        "recent_measurement_trends": [
            {"metric": "body_weight", "trend": "falling"},
            {"metric": "daily_steps", "trend": "rising"},
        ]
    }
    assert materialize_state_claims(question, packet) == (
        StateClaim("body_weight", "falling"),
    )


def test_state_claim_uses_runtime_packet_only() -> None:
    claim = materialize_state_claims(
        "Describe only the recorded body weight trend.",
        {"recent_measurement_trends": [{"metric": "body_weight", "trend": "stable"}]},
    )[0]
    assert claim.source == "longitudinal_state"
    assert claim.value == "stable"


def test_runtime_task_kind_is_inferred_from_observable_question() -> None:
    assert classify_runtime_task(
        _runtime_case("a", "Describe only the recorded body weight trend.", "state.json")
    ) == "STATE_ONLY"
    assert classify_runtime_task(
        _runtime_case("b", "What general physical activity advice is useful?", None)
    ) == "GUIDANCE_ONLY"
    assert classify_runtime_task(
        _runtime_case("c", "First, describe only the recorded body weight trend. Separately, give advice.", "state.json")
    ) == "STATE_AND_GUIDANCE"


def test_t0_requires_zero_model_calls(tmp_path: Path) -> None:
    frozen, _ = _frozen_fixture(tmp_path)
    plan = make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    state_only = [row for row in plan if row["task_kind"] == "STATE_ONLY"]
    assert len(state_only) == 60
    assert all(row["model_call_count"] == 0 for row in state_only)
    assert all(row["guidance_prompt"] is None for row in state_only)


def test_t2_state_is_not_sent_to_guidance_generator(tmp_path: Path) -> None:
    frozen, _ = _frozen_fixture(tmp_path)
    plan = make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    row = next(item for item in plan if item["task_kind"] == "STATE_AND_GUIDANCE")
    assert row["state_claims"] == [{"field": "body_weight", "value": "falling", "source": "longitudinal_state"}]
    assert row["question"] in row["guidance_prompt"]
    assert "body_weight" not in row["guidance_prompt"]
    assert "longitudinal_state" not in row["guidance_prompt"]
    assert row["model_call_count"] == 1


def test_untrusted_llm_fields_cannot_change_fixed_arm_or_state(
    tmp_path: Path,
) -> None:
    frozen, _ = _frozen_fixture(tmp_path)
    plan = make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    row = next(
        item
        for item in plan
        if item["task_kind"] == "STATE_AND_GUIDANCE" and item["action"] == "STANDARD"
    )
    row["_call_ordinal"] = 1

    arm = b4_execution._complete_arm(
        plan_row=row,
        lock_sha="frozen-lock",
        client=FakeUntrustedGuidanceClient(),
        private_root=tmp_path,
        call_ledger=tmp_path / "call_ledger.jsonl",
    )

    assert arm["action"] == "STANDARD"
    assert arm["state_claims"] == [
        {"field": "body_weight", "value": "falling", "source": "longitudinal_state"}
    ]
    assert arm["resolved_citation_chunk_ids"] == []
    assert arm["invented_evidence_aliases"] == ["E999"]
    assert arm["new_retrieval_calls"] == arm["new_bridge_calls"] == 0


def test_teacher_fields_are_not_part_of_guidance_prompt_interface() -> None:
    params = inspect.signature(render_guidance_prompt).parameters
    assert set(params) == {"question", "passages"}
    assert not {"teacher", "rubric", "gold", "answer", "outcome"}.intersection(params)


def test_evidence_alias_mapping_is_deterministic() -> None:
    passages = [
        {"chunk_id": "c1", "text": "First evidence.", "source_id": "s", "section_path": ["one"]},
        {"chunk_id": "c2", "text": "Second evidence.", "source_id": "s", "section_path": ["two"]},
    ]
    prompt1, alias1 = render_guidance_prompt("Question?", passages)
    prompt2, alias2 = render_guidance_prompt("Question?", passages)
    assert prompt1 == prompt2
    assert alias1 == alias2
    assert [(row["alias"], row["chunk_id"]) for row in alias1] == [("E1", "c1"), ("E2", "c2")]


def test_standard_reuses_exact_b2_top5(tmp_path: Path) -> None:
    frozen, sources = _frozen_fixture(tmp_path)
    before = deepcopy(sources)
    plan = make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    row = next(item for item in plan if item["task_kind"] == "GUIDANCE_ONLY" and item["action"] == "STANDARD")
    assert row["evidence_chunks"] == sources[f"{row['case_id']}:STANDARD"]
    assert len(row["evidence_chunks"]) == 5
    assert sources == before


def test_strong_reuses_exact_b2_top5(tmp_path: Path) -> None:
    frozen, sources = _frozen_fixture(tmp_path)
    plan = make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    row = next(item for item in plan if item["task_kind"] == "STATE_AND_GUIDANCE" and item["action"] == "STRONG")
    assert row["evidence_chunks"] == sources[f"{row['case_id']}:STRONG"]
    assert len(row["evidence_chunks"]) == 5


def test_guidance_parser_accepts_plain_text_without_json_dependency() -> None:
    text = "Adults can aim for 150 minutes weekly. [E2] Not a JSON object."
    aliases = [{"alias": "E2", "chunk_id": "real-chunk", "source_id": "s", "section_path": ["health"]}]
    cited, invented = extract_citations(text, aliases)
    assert cited == ["real-chunk"]
    assert invented == []
    assert strip_aliases(text) == "Adults can aim for 150 minutes weekly.  Not a JSON object."


def test_unknown_citation_alias_is_ignored_and_counted() -> None:
    cited, invented = extract_citations("Claim [E1] [E6] [E99].", [{"alias": "E1", "chunk_id": "c1"}])
    assert cited == ["c1"]
    assert invented == ["E6", "E99"]


def test_final_materializer_resolves_only_known_aliases() -> None:
    final = materialize_final_response(
        task_kind="STATE_AND_GUIDANCE",
        state_claims=[StateClaim("body_weight", "falling")],
        guidance_text="Adults should be active [E1] and check [E7].",
        alias_map=[
            {
                "alias": "E1",
                "chunk_id": "c1",
                "source_id": "who-guideline",
                "section_path": ["Adults", "Activity"],
            }
        ],
    )
    assert final.startswith("Recorded body weight trend: falling.")
    assert "[Source: who-guideline; Adults > Activity; chunk c1]" in final
    assert "E7" not in final


def test_off_has_no_external_aliases(tmp_path: Path) -> None:
    frozen, _ = _frozen_fixture(tmp_path)
    plan = make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    rows = [row for row in plan if row["task_kind"] != "STATE_ONLY" and row["action"] == "OFF"]
    assert len(rows) == 40
    assert all(row["evidence_aliases"] == [] for row in rows)
    assert all("No external evidence is available." in row["guidance_prompt"] for row in rows)


def test_no_new_retrieval_or_bridge_calls(tmp_path: Path) -> None:
    frozen, _ = _frozen_fixture(tmp_path)
    plan = make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    assert len(plan) == 180
    assert all(row["new_retrieval_calls"] == 0 and row["new_bridge_calls"] == 0 for row in plan)


def test_all_actions_use_the_same_8192_generation_ceiling() -> None:
    payload = chat_payload("Question with no JSON schema.")
    assert payload["max_tokens"] == 8192 == MAX_OUTPUT_TOKENS
    assert CONTEXT_SIZE == 32768
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in payload


def test_evaluator_case_scorer_does_not_receive_action_name() -> None:
    assert "action" not in inspect.signature(b4_evaluator.score_materialized_case).parameters


def test_llm_text_cannot_override_deterministic_state_or_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_score_case(**kwargs: object) -> dict[str, float]:
        captured.update(kwargs)
        return {"test_score": 1.0}

    monkeypatch.setattr(b4_evaluator, "score_case", capture_score_case)
    model_text = (
        '{"action":"STRONG","state_claims":[{"field":"body_weight",'
        '"value":"rising"}]} Your weight is rising. [E999]'
    )
    score = b4_evaluator.score_materialized_case(
        teacher={},
        state_claims=[
            {"field": "body_weight", "value": "falling", "source": "longitudinal_state"}
        ],
        guidance_text=model_text,
        cited_chunk_ids=["frozen-b2-chunk"],
        supplied_chunks=[],
    )

    reader_output = captured["reader_output"]
    assert isinstance(reader_output, dict)
    assert reader_output["state_facts"] == [
        {"field": "body_weight", "value": "falling"}
    ]
    assert "Your weight is rising." in reader_output["guidance_facts"][0]["statement"]
    assert reader_output["citations"] == ["frozen-b2-chunk"]
    assert "action" not in inspect.signature(b4_evaluator.score_materialized_case).parameters
    assert score == {"test_score": 1.0}


def test_execution_manifest_must_be_committed_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        b4_evaluator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr="not tracked"),
    )
    with pytest.raises(ValueError, match="must be committed"):
        b4_evaluator._require_committed_manifest(
            b4_evaluator.DEFAULT_EXECUTION_MANIFEST_PATH
        )


def test_scores_are_gated_on_frozen_manifest_before_teacher_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(b4_evaluator, "_read_json", lambda _path: {"protocol_lock_sha256": "lock"})
    monkeypatch.setattr(b4_evaluator, "verify_protocol_lock", lambda _lock: "lock")
    monkeypatch.setattr(b4_evaluator, "verify_frozen_code", lambda _lock: None)
    monkeypatch.setattr(
        b4_evaluator,
        "load_verified_b2",
        lambda _lock: SimpleNamespace(artifact_set_sha256="b2"),
    )
    monkeypatch.setattr(b4_evaluator, "verify_b3_identity", lambda _lock: {})
    monkeypatch.setattr(
        b4_evaluator,
        "_verify_manifest",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("manifest not frozen")),
    )
    monkeypatch.setattr(
        b4_evaluator,
        "read_jsonl",
        lambda _path: pytest.fail("teacher artifact opened before B4 freeze"),
    )
    with pytest.raises(ValueError, match="manifest not frozen"):
        b4_evaluator.score_frozen_run(
            protocol_path=Path("lock.json"),
            manifest_path=Path("manifest.json"),
            private_root=Path("external"),
            report_json_path=Path("report.json"),
            report_markdown_path=Path("report.md"),
        )


def test_b2_b3_source_evidence_objects_are_not_mutated(tmp_path: Path) -> None:
    frozen, sources = _frozen_fixture(tmp_path)
    snapshot = deepcopy(sources)
    make_execution_plan(frozen=frozen, client=FakeTokenizer(), private_root=tmp_path)
    assert sources == snapshot


def test_32768_context_contract_rejects_prompt_overflow(tmp_path: Path) -> None:
    frozen, _ = _frozen_fixture(tmp_path)
    with pytest.raises(ValueError, match="exceeds frozen 32768-context"):
        make_execution_plan(frozen=frozen, client=FakeTokenizer(CONTEXT_SIZE), private_root=tmp_path)


def test_protocol_lock_self_hash_is_checked() -> None:
    body = {"status": "FROZEN_BEFORE_B4_GUIDANCE_CALLS", "202608_opened": False, "source_cohort": "202607_only"}
    lock = {**body, "protocol_lock_sha256": "bad"}
    with pytest.raises(ValueError, match="self-hash"):
        verify_protocol_lock(lock)
