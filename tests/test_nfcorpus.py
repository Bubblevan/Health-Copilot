import json
from pathlib import Path

from health_ai_copilot.eval.nfcorpus import evaluate_nfcorpus, load_nfcorpus


def _write_nfcorpus_fixture(root: Path) -> None:
    (root / "qrels").mkdir()
    (root / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "Salt", "text": "salt and blood pressure"})
        + "\n"
        + json.dumps({"_id": "d2", "title": "Sleep", "text": "sleep and rest"})
        + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "salt blood pressure"}) + "\n", encoding="utf-8"
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\n", encoding="utf-8"
    )


def test_nfcorpus_adapter_and_metrics(tmp_path: Path) -> None:
    _write_nfcorpus_fixture(tmp_path)

    dataset = load_nfcorpus(tmp_path)
    metrics = evaluate_nfcorpus(dataset, top_k=1)

    assert len(dataset.documents) == 2
    assert len(dataset.cases) == 1
    assert metrics["recall_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_1"] == 1.0
