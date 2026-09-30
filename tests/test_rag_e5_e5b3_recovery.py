from __future__ import annotations

from types import SimpleNamespace

from eval.rag_e5 import e5b3_evaluator
from eval.rag_e5.counterfactual import run_id_for
from eval.rag_e5.e5b3_recovery import FrozenB2ArmContext, b3_run_id, reader_v2_to_scorer
from tools.research.rag_e5.run_e5b3_recovery import project_frozen_b2_context


def _context(*, strong: bool = False) -> FrozenB2ArmContext:
    bridge = {"request": {"query": "frozen"}, "raw_response": {"text": "bridge"}}
    return FrozenB2ArmContext(
        case_id="case-a",
        action="STRONG" if strong else "STANDARD",
        b2_run_id="old-run-id",
        source_arm_file_sha256="source-file-sha",
        source_completion_sha256="source-completion-sha",
        question_sha256="question-sha",
        state_packet_sha256="state-sha",
        runtime_capability_context_sha256="capability-sha",
        retrieval_channels={"bm25": ["chunk-a"]},
        retrieval_ranking=({"chunk_id": "chunk-a", "score": 1.0},),
        retrieval_ranking_sha256="ranking-sha",
        supplied_chunks=({"chunk_id": "chunk-a", "text": "approved passage"},),
        supplied_chunk_ids_sha256="chunk-ids-sha",
        supplied_chunks_sha256="chunks-sha",
        source_retrieval_calls=1,
        source_retrieval_latency_ms=12.0,
        source_bridge_call_count=int(strong),
        source_bridge_latency_ms=4.0 if strong else 0.0,
        source_bridge_input_tokens=20 if strong else None,
        source_bridge_output_tokens=10 if strong else None,
        source_bridge_response_sha256="bridge-sha" if strong else None,
        source_bridge_fallback=False,
        bridge_artifact=bridge if strong else None,
    )


def test_b3_run_id_is_new_protocol_namespace() -> None:
    assert b3_run_id("case-a", "OFF", "new-lock") != run_id_for("case-a", "OFF", "old-lock")


def test_reader_v2_projection_preserves_canonical_fields_without_alias_repair() -> None:
    output = {
        "state_facts": [{"field": "body_weight_trend", "value": "increasing"}],
        "guidance_facts": [
            {"statement": "A short supported fact.", "citations": ["chunk-b", "chunk-a"]}
        ],
    }
    projected = reader_v2_to_scorer(output)
    assert projected["state_facts"] == output["state_facts"]
    assert projected["guidance_facts"] == [{"statement": "A short supported fact."}]
    assert projected["citations"] == ["chunk-a", "chunk-b"]
    assert "answer" not in projected


def test_projection_keeps_exact_frozen_evidence_and_strong_bridge_only() -> None:
    standard = project_frozen_b2_context(_context())
    assert standard["passages"] == [{"chunk_id": "chunk-a", "text": "approved passage"}]
    assert standard["retrieval"]["ranking"] == [{"chunk_id": "chunk-a", "score": 1.0}]
    assert standard["bridge"] is None
    assert "reader_parsed" not in standard

    strong = project_frozen_b2_context(_context(strong=True))
    assert strong["bridge"] == {"request": {"query": "frozen"}, "raw_response": {"text": "bridge"}}
    assert strong["source_bridge_call_count"] == 1
    assert strong["source_bridge_input_tokens"] == 20


def test_frozen_context_has_no_b2_reader_output_fields() -> None:
    names = set(FrozenB2ArmContext.__dataclass_fields__)
    assert not any("reader" in name or "answer" in name for name in names)


def test_b3_execution_path_does_not_invoke_retriever_or_bridge() -> None:
    from pathlib import Path

    source = Path("tools/research/rag_e5/run_e5b3_recovery.py").read_text(encoding="utf-8")
    assert "E5B2Retriever" not in source
    assert "render_lamer_prompt" not in source
    assert '"new_retrieval_calls": 0' in source
    assert '"new_bridge_calls": 0' in source


def test_measurement_gate_fails_before_teacher_is_read(tmp_path, monkeypatch) -> None:
    lock = {
        "b3_recovery_lock_sha256": "b3-lock",
        "qualification_report_sha256": "qualification",
        "task_set_sha256": "tasks",
        "state_packet_set_sha256": "states",
        "b2_artifact_set_sha256": "artifacts",
        "b2_reuse_manifest_sha256": "reuse",
        "scorer_version": "frozen-scorer",
        "scorer_module_sha256": "scorer-module",
        "scoring_contract_sha256": "scoring-contract",
        "reader_output_budget": 512,
        "model": {"sha256": "model"},
    }
    manifest = {"execution_manifest_sha256": "manifest"}
    frozen = SimpleNamespace(lock={"counterfactual_lock_sha256": "b2-lock"})
    measurements = {
        "B2": {
            "reader_json_valid_rate": 0.55,
            "finish_reason_length_rate": 0.44,
            "state_contract_match_rate": 0.0,
        },
        "B3": {
            "reader_json_valid_rate": 0.94,
            "reader_json_valid": 169,
            "finish_reason_length": 1,
            "finish_reason_length_rate": 1 / 180,
            "state_contract_match_rate": 0.8,
        },
    }
    monkeypatch.setattr(
        e5b3_evaluator,
        "_load_frozen_run",
        lambda **_: (lock, manifest, frozen, {}, {}),
    )
    monkeypatch.setattr(e5b3_evaluator, "_measurement_rows", lambda **_: (measurements["B2"], measurements["B3"]))
    monkeypatch.setattr(
        e5b3_evaluator,
        "_read_jsonl",
        lambda path: (_ for _ in ()).throw(AssertionError(f"teacher read before gate: {path}")),
    )
    monkeypatch.setattr(
        e5b3_evaluator,
        "sha256_file",
        lambda path: (_ for _ in ()).throw(AssertionError(f"file opened before gate: {path}")),
    )
    report = e5b3_evaluator.score_recovery(
        private_root=tmp_path,
        report_json_path=tmp_path / "report.json",
        report_markdown_path=tmp_path / "report.md",
        private_score_path=tmp_path / "private.json",
    )
    assert report["MEASUREMENT_RECOVERY_GATE"] == "FAIL"
    assert report["teacher_opened"] is False
    assert report["conditional_value_evaluated"] is False
    assert "fixed_action_matrix" not in report
    assert not (tmp_path / "private.json").exists()
