import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import tools.run_e1_2_retrieval as retrieval
from tools.prepare_e1_2_retrieval_inputs import build_retrieval_benchmark, materialize
from tools.run_e1_2_retrieval import build_command


def test_retrieval_materializer_is_split_scoped_and_excludes_labels():
    cases = [
        {
            "case_id": "medqa:a",
            "payload": {
                "question": "Question A?",
                "options": {"A": "one", "B": "two"},
            },
            "gold": {"answer": "secret-label"},
        },
        {
            "case_id": "medqa:b",
            "payload": {"question": "Question B?", "options": {"A": "x", "B": "y"}},
            "gold": {"answer": "secret-label"},
        },
        {
            "case_id": "pubmedqa:historical",
            "payload": {"question": "Exposed?", "options": {"A": "yes", "B": "no"}},
            "gold": {"answer": "secret-label"},
        },
    ]
    manifest = {
        "cases": [
            {"case_id": "medqa:a", "subdataset": "medqa", "split": "DEV"},
            {"case_id": "medqa:b", "subdataset": "medqa", "split": "TEST"},
            {
                "case_id": "pubmedqa:historical",
                "subdataset": "pubmedqa",
                "split": "EXPOSED_HISTORY",
            },
        ],
        "counts": {"medqa": {"DEV": 1}},
    }

    benchmark, counts = build_retrieval_benchmark(cases, manifest, "DEV")
    assert counts == {"medqa": 1}
    assert benchmark["medqa"] == {"a": {"question": "Question A?", "options": {"A": "one", "B": "two"}}}
    serialized = str(benchmark)
    assert "secret-label" not in serialized
    assert "Question B?" not in serialized
    assert "Exposed?" not in serialized


def test_retrieval_command_keeps_indexes_and_cache_on_scratch(tmp_path: Path):
    command = build_command(
        retriever="medcpt",
        split="TEST",
        scratch_root=tmp_path,
        benchmark_path=tmp_path / "inputs" / "benchmark_test.json",
        corpus_dir=Path("D:/corpus"),
        model_root=Path("D:/models"),
    )
    assert str(tmp_path / "index" / "medrag_textbooks_fts5.sqlite3") in command
    assert str(tmp_path / "cache" / "medcpt_textbooks") in command
    assert str(tmp_path / "retrieval" / "test" / "medcpt") in command
    assert command[command.index("--dense-candidate-depth") + 1] == "100"


def test_storage_floor_uses_the_target_volume_capacity(tmp_path: Path, monkeypatch):
    gib = 1024**3
    disk = SimpleNamespace(total=int(803.9 * gib), free=int(190.3 * gib), used=0)
    monkeypatch.setattr(retrieval.shutil, "disk_usage", lambda _path: disk)

    retrieval.ensure_storage(tmp_path, reserve_gib=5.0)


def test_storage_floor_still_enforces_twenty_percent_of_target_volume(tmp_path: Path, monkeypatch):
    gib = 1024**3
    disk = SimpleNamespace(total=int(803.9 * gib), free=int(160.0 * gib), used=0)
    monkeypatch.setattr(retrieval.shutil, "disk_usage", lambda _path: disk)

    try:
        retrieval.ensure_storage(tmp_path, reserve_gib=1.0)
    except OSError as exc:
        assert "160.8 GiB free-space floor" in str(exc)
    else:
        raise AssertionError("storage guard accepted free space below the target volume floor")


def test_materialize_writes_hashed_question_options_only_input_and_sidecar(tmp_path: Path):
    cases_path = tmp_path / "cases.jsonl"
    cases = [
        {
            "case_id": f"{subset}:id",
            "metadata": {"subdataset": subset},
            "payload": {"question": f"Question {subset}?", "options": {"A": "x", "B": "y"}},
            "gold": {"answer": "secret"},
        }
        for subset in ("medqa", "medmcqa", "mmlu")
    ]
    serialized_cases = "".join(json.dumps(case) + "\n" for case in cases).encode()
    cases_path.write_bytes(serialized_cases)
    assignments = [
        {"case_id": case["case_id"], "subdataset": case["metadata"]["subdataset"], "split": "DEV"}
        for case in cases
    ]
    split_manifest_path = tmp_path / "split.json"
    split_manifest = {
        "source_identity": {"normalized_cases_sha256": hashlib.sha256(serialized_cases).hexdigest()},
        "cases": assignments,
        "counts": {name: {"DEV": 1} for name in ("medqa", "medmcqa", "mmlu")},
    }
    split_manifest_path.write_text(json.dumps(split_manifest), encoding="utf-8")
    output_path = tmp_path / "out" / "benchmark_dev.json"

    result = materialize(
        cases_path=cases_path,
        split_manifest_path=split_manifest_path,
        split="DEV",
        output_path=output_path,
    )
    benchmark = json.loads(output_path.read_text(encoding="utf-8"))
    sidecar = json.loads(output_path.with_name(output_path.name + ".manifest.json").read_text(encoding="utf-8"))
    assert result["case_count"] == 3
    assert sidecar["sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert "secret" not in output_path.read_text(encoding="utf-8")
    assert all(set(row) == {"question", "options"} for rows in benchmark.values() for row in rows.values())
