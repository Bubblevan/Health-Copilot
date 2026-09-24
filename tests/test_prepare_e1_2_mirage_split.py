import pytest

from tools.prepare_e1_2_mirage_split import build_split_manifest


def make_case(case_id: str, subdataset: str, question: str) -> dict:
    return {
        "case_id": f"{subdataset}:{case_id}",
        "metadata": {"subdataset": subdataset},
        "payload": {"question": question, "options": {"A": "secret choice"}},
        "gold": {"answer": "secret label"},
    }


def test_split_is_deterministic_and_keeps_duplicate_questions_together():
    cases = [
        make_case("m1", "medqa", "Shared question?"),
        make_case("m2", "medqa", "Unique medqa 1?"),
        make_case("m3", "medqa", "Unique medqa 2?"),
        make_case("c1", "medmcqa", " shared   QUESTION? "),
        make_case("c2", "medmcqa", "Unique medmcqa 1?"),
        make_case("c3", "medmcqa", "Unique medmcqa 2?"),
        make_case("u1", "mmlu", "Unique mmlu 1?"),
        make_case("u2", "mmlu", "Unique mmlu 2?"),
        make_case("p1", "pubmedqa", "Previously exposed question?"),
        make_case("b1", "bioasq", "Previously exposed question 2?"),
    ]

    first = build_split_manifest(cases, raw_sha256="raw", normalized_sha256="normalized")
    second = build_split_manifest(cases, raw_sha256="raw", normalized_sha256="normalized")
    assert first == second
    assignments = {row["case_id"]: row["split"] for row in first["cases"]}
    assert assignments["medqa:m1"] == assignments["medmcqa:c1"]
    assert assignments["pubmedqa:p1"] == "EXPOSED_HISTORY"
    assert assignments["bioasq:b1"] == "EXPOSED_HISTORY"
    for subdataset in ("medqa", "medmcqa", "mmlu"):
        assert first["counts"][subdataset]["DEV"]
        assert first["counts"][subdataset]["TEST"]


def test_split_manifest_never_contains_questions_options_or_gold():
    cases = [
        make_case(f"m{index}", "medqa", f"Unique question {index}?")
        for index in range(1, 6)
    ]
    cases += [
        make_case(f"c{index}", "medmcqa", f"Different question {index}?")
        for index in range(1, 6)
    ]
    cases += [make_case(f"u{index}", "mmlu", f"MMLU question {index}?") for index in range(1, 6)]
    manifest = build_split_manifest(cases, raw_sha256="raw", normalized_sha256="normalized")
    serialized = str(manifest)
    assert "secret choice" not in serialized
    assert "secret label" not in serialized
    assert "Unique question" not in serialized
    assert all(set(row) == {"case_id", "subdataset", "question_sha256", "split"} for row in manifest["cases"])


def test_split_rejects_unknown_subdataset_and_duplicate_ids():
    valid = [make_case("a", "medqa", "q"), make_case("b", "medqa", "q2")]
    valid += [make_case("c1", "medmcqa", "c1"), make_case("c2", "medmcqa", "c2")]
    valid += [make_case("u1", "mmlu", "u1"), make_case("u2", "mmlu", "u2")]
    with_unknown = [*valid, make_case("x", "unknown", "question")]
    with pytest.raises(ValueError, match="unrecognized MIRAGE subdataset"):
        build_split_manifest(with_unknown, raw_sha256="raw", normalized_sha256="normalized")
    with_duplicate_id = [*valid, make_case("a", "medqa", "duplicate")]
    with pytest.raises(ValueError, match="duplicate case_id"):
        build_split_manifest(with_duplicate_id, raw_sha256="raw", normalized_sha256="normalized")
