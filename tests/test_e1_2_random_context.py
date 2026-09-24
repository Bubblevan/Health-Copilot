import sqlite3

from tools.run_e1_2_mirage import evidence_context_characters, random_context_evidence


def test_random_context_sampling_is_deterministic_and_matches_count_and_char_budget():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE chunks (rowid INTEGER PRIMARY KEY, id TEXT, title TEXT, content TEXT)"
    )
    connection.executemany(
        "INSERT INTO chunks (id, title, content) VALUES (?, ?, ?)",
        [(f"random-{index}", f"Title {index}", "z" * 120) for index in range(60)],
    )
    target = [
        {"id": f"target-{index}", "title": f"Target title {index}", "content": "x" * 45}
        for index in range(3)
    ]

    first = random_context_evidence(connection, "case-1", target, seed="fixed-seed")
    second = random_context_evidence(connection, "case-1", target, seed="fixed-seed")

    assert first == second
    evidence, count_matched, budget_matched = first
    assert count_matched is True
    assert budget_matched is True
    assert len(evidence) == len(target)
    assert evidence_context_characters(evidence) == evidence_context_characters(target)
    connection.close()
