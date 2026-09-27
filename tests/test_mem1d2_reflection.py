import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

_module_path = Path(__file__).parents[1] / "tools" / "research" / "memory" / "mem1d2_reflection.py"
_spec = importlib.util.spec_from_file_location("mem1d2_reflection_test", _module_path)
mem1d2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mem1d2)


def _original_formatter(dialogues):
    lines = []
    current_date = None
    for dialogue in dialogues:
        timestamp = dialogue.get("timestamp", "")
        date_part = timestamp.split(" ")[0] if timestamp else ""
        if date_part and date_part != current_date:
            current_date = date_part
            lines.append(f"\n## {current_date}\n")
        speaker = dialogue.get("speaker", "Unknown")
        text = dialogue.get("text", "")
        if timestamp:
            lines.append(f"**{speaker}** ({timestamp}): {text}")
        else:
            lines.append(f"**{speaker}**: {text}")
    return "\n".join(lines)


def test_multiline_turn_map_is_physical_and_markdown_bytes_are_unchanged():
    dialogues = [
        {
            "speaker": "user",
            "text": "first physical line\nsecond physical line",
            "timestamp": "2026-01-01 09:00",
            "session_id": "session-a",
        },
        {
            "speaker": "assistant",
            "text": "third line",
            "timestamp": "2026-01-01 09:00",
            "session_id": "session-b",
        },
    ]

    old_markdown, corrected_markdown, legacy_map, corrected_map = (
        mem1d2.format_with_physical_session_map(dialogues, _original_formatter)
    )

    assert old_markdown == corrected_markdown == _original_formatter(dialogues)
    assert len(legacy_map) < len(corrected_map) == len(corrected_markdown.split("\n"))
    assert corrected_map[:3] == [None, None, None]
    assert corrected_map[3:] == ["session-a", "session-a", "session-b"]
    assert mem1d2.source_sessions_for_chunk(
        SimpleNamespace(start_line=4, end_line=5), corrected_map
    ) == ["session-a"]


def test_selected_dataset_reader_does_not_json_decode_non_target_rows(monkeypatch, tmp_path):
    dataset = tmp_path / "rows.json"
    dataset.write_text(
        json.dumps([
            {"question_id": "dev-a", "question": "needed", "answer": "42"},
            {"question_id": "test-a", "question": "must not decode", "answer": "secret"},
        ]),
        encoding="utf-8",
    )
    calls = []
    original_loads = mem1d2.json.loads

    def tracking_loads(value, *args, **kwargs):
        calls.append(value)
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(mem1d2.json, "loads", tracking_loads)

    selected = mem1d2.load_selected_json_array(dataset, {"dev-a"})

    assert list(selected) == ["dev-a"]
    assert selected["dev-a"]["answer"] == "42"
    assert len(calls) == 1
    assert "test-a" not in calls[0]


def test_provenance_overlay_keys_include_system(monkeypatch):
    monkeypatch.setattr(mem1d2, "sha256_file", lambda _path: "frozen-bundle-hash")
    question_id = "same-question"
    openclaw_text = "OpenClaw chunk"
    propmem_text = "PropMem fallback"

    def item(text, kind, session_id):
        return {
            "text": text,
            "kind": kind,
            "rank": 1,
            "source_session_ids": [f"legacy-{session_id}"],
        }

    bundle_rows = [
        {
            "system": "openclaw",
            "question_id": question_id,
            "context_bundle": {
                "context_bundle_sha256": "openclaw-bundle",
                "items": [item(openclaw_text, "chunk", "openclaw")],
            },
        },
        {
            "system": "propmem",
            "question_id": question_id,
            "context_bundle": {
                "context_bundle_sha256": "propmem-bundle",
                "items": [item(propmem_text, "fallback_chunk", "propmem")],
            },
        },
    ]

    def chunk(text, session):
        return {
            "text": text,
            "text_sha256": mem1d2.sha256_bytes(text.encode("utf-8")),
            "legacy_source_session_ids": [f"legacy-{session}"],
            "corrected_source_session_ids": [f"corrected-{session}"],
            "chunk_index": 1,
        }

    overlay, corrected = mem1d2._overlay_context_items(
        bundle_rows,
        {
            question_id: [chunk(openclaw_text, "openclaw"), chunk(propmem_text, "propmem")],
            "parity_rows": [],
        },
    )

    assert len(overlay["rows"]) == 2
    assert corrected[("openclaw", question_id, 1)] == ["corrected-openclaw"]
    assert corrected[("propmem", question_id, 1)] == ["corrected-propmem"]
