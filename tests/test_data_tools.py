import json
from pathlib import Path

from tools.fetch_m0_data import crawl_sources, parse_html_candidates


def test_html_parser_returns_bounded_review_candidates() -> None:
    body = b"""
    <html><head><title>Demo</title><script><p>ignore me</p></script></head>
    <body><p>navigation noise</p><main><h1>Section</h1><p>First   candidate.</p>
    <p>First candidate.</p><p>Second candidate with more words.</p></main></body></html>
    """

    result = parse_html_candidates(body, max_chars=10, max_candidates=2)

    assert result["page_title"] == "Demo"
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["excerpt"] == "First can…"
    assert "ignore me" not in json.dumps(result)
    assert "navigation noise" not in json.dumps(result)


def test_crawl_writes_review_draft_and_not_knowledge_card(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            [
                {
                    "id": "source-a",
                    "title": "Source A",
                    "url": "https://example.com/a",
                    "publisher": "Example",
                    "jurisdiction": "global",
                    "published_at": None,
                    "tags": ["hypertension"],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tools.fetch_m0_data.fetch_bytes",
        lambda url, **kwargs: (
            b"<html><title>A</title><h1>About</h1><p>Verified candidate.</p></html>",
            {"etag": "demo"},
        ),
    )

    output = tmp_path / "drafts"
    paths = crawl_sources(catalog, output, max_chars=100)

    assert paths == [output / "source-a.json"]
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["review_required"] is True
    assert payload["candidates"][0]["excerpt"] == "Verified candidate."
    assert "content" not in payload
