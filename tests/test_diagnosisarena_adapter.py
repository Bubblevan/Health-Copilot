import json

from health_ai_copilot.evaluation.datasets import DiagnosisArenaAdapter


class OneCaseDiagnosisArena(DiagnosisArenaAdapter):
    expected_count = 1


def test_diagnosisarena_maps_gold_only_to_eval_case(tmp_path) -> None:
    path = tmp_path / "diagnosis.jsonl"
    path.write_text(json.dumps({
        "id": "case-1", "Case Information": "Patient details", "Options": {"A": "x", "B": "y"},
        "Right Option": "A",
    }) + "\n", encoding="utf-8")
    case, = OneCaseDiagnosisArena(path, source_revision="rev1").cases()
    assert case.gold == "A"
    assert "x" in case.query
    assert "A" not in case.metadata
