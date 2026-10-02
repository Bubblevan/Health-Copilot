from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from health_ai_copilot.huiyi.acquire import _canonical_discovered_url, classify_url
from health_ai_copilot.huiyi.chunk import chunk_documents
from health_ai_copilot.huiyi.embed import Qwen3LocalEmbedder
from health_ai_copilot.huiyi.hybrid import reciprocal_rank_fusion
from health_ai_copilot.huiyi.normalize import normalize_raw_corpus
from health_ai_copilot.huiyi.schema import SourceSpec, source_id_for_url


def _write_corpus_inputs(root: Path, pages: list[tuple[str, str, str, str]]) -> None:
    (root / "raw" / "pages").mkdir(parents=True)
    (root / "normalized").mkdir(parents=True)
    catalog: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for url, expected_type, policy, html in pages:
        raw = html.encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        source = SourceSpec(
            source_id=source_id_for_url(url),
            url=url,
            domain="yk.huiyi9e.com",
            source_family="huiyi_official",
            expected_type=expected_type,
            ingestion_policy=policy,
        )
        catalog.append(source.__dict__)
        relative = Path("raw") / "pages" / f"{digest}.html"
        (root / relative).write_bytes(raw)
        manifest.append({
            "source_id": source.source_id,
            "url": url,
            "final_url": url,
            "http_status": 200,
            "fetched_at_utc": "2026-10-02T00:00:00Z",
            "content_type": "text/html",
            "raw_path": relative.as_posix(),
            "raw_sha256": digest,
            "parser_version": "test-fixture-v1",
            "ingestion_policy": policy,
        })
    (root / "source_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    (root / "raw" / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )


def _page(title: str, content: str, *, date: str = "") -> str:
    return f"""<!doctype html><html><head><title>{title}—医院</title></head>
    <body><nav>官网导航应被排除</nav><div class="center_title"><h2 class="main_title">{title}</h2></div>
    <div class="newsdetail_top"><time>{date}</time></div>
    <div class="main_body"><div class="v_news_content">{content}</div></div><footer>版权尾部</footer></body></html>"""


def test_source_id_and_domain_scope_are_deterministic() -> None:
    url = "https://yk.huiyi9e.com/info/1021/1530.htm"
    assert source_id_for_url(url) == source_id_for_url(url)
    assert source_id_for_url(url) != source_id_for_url("https://yk.huiyi9e.com/info/1021/1531.htm")
    assert _canonical_discovered_url("/info/1021/1530.htm", "https://yk.huiyi9e.com/") == url
    assert _canonical_discovered_url("https://example.com/other.htm", "https://yk.huiyi9e.com/") is None
    assert classify_url("https://yk.huiyi9e.com/info/1024/1.htm")[1] == "canonical"
    assert classify_url("https://yk.huiyi9e.com/info/1019/1985.htm")[1] == "raw_only_dynamic"


def test_normalization_keeps_body_and_cjk_punctuation_and_rejects_empty(tmp_path: Path) -> None:
    pages = [
        (
            "https://yk.huiyi9e.com/info/1022/1.htm",
            "department",
            "canonical",
            _page("青光眼专科", "<h3>常见检查</h3><p>检查包括视野检查，必要时进行进一步评估。原文 https://other.example/track</p>"),
        ),
        (
            "https://yk.huiyi9e.com/info/1022/2.htm",
            "department",
            "canonical",
            _page("空内容", "<script>不得入库</script><div></div>"),
        ),
    ]
    _write_corpus_inputs(tmp_path, pages)
    report = normalize_raw_corpus(tmp_path)
    docs = [json.loads(line) for line in (tmp_path / "normalized" / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(docs) == 1
    document = docs[0]
    assert "官网导航" not in document["content"]
    assert "版权尾部" not in document["content"]
    assert "不得入库" not in document["content"]
    assert "检查包括视野检查，必要时进行进一步评估。" in document["content"]
    assert "https://other.example/track" not in document["content"]
    assert document["source_url"] == pages[0][0]
    assert document["freshness_class"] == "STATIC_PROFILE"
    assert document["review_status"] == "AUTO_ACCEPTED_PUBLIC_INFO"
    documents_bytes = (tmp_path / "normalized" / "documents.jsonl").read_bytes()
    normalize_raw_corpus(tmp_path)
    assert (tmp_path / "normalized" / "documents.jsonl").read_bytes() == documents_bytes
    rejected = (tmp_path / "normalized" / "rejected.jsonl").read_text(encoding="utf-8")
    assert "empty_content" in rejected
    assert report["normalization"]["parser_failures"] == 1


def test_dynamic_and_temporary_pages_never_become_documents(tmp_path: Path) -> None:
    pages = [
        (
            "https://yk.huiyi9e.com/info/1025/1.htm",
            "faq",
            "canonical",
            _page("本周专家门诊排班", "<p>本周专家门诊排班见页面。</p>"),
        ),
        (
            "https://yk.huiyi9e.com/info/1019/2.htm",
            "hospital_info",
            "raw_only_dynamic",
            _page("临时通知", "<p>临时通知：活动将于2026年10月10日至2026年10月12日进行。</p>"),
        ),
    ]
    _write_corpus_inputs(tmp_path, pages)
    report = normalize_raw_corpus(tmp_path)
    assert (tmp_path / "normalized" / "documents.jsonl").read_text(encoding="utf-8") == ""
    rejected = [json.loads(line) for line in (tmp_path / "normalized" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["reason"] for row in rejected} == {"raw_only_dynamic"}
    assert report["normalization"]["raw_only_dynamic"] == 2


def test_exact_and_near_duplicate_reports_do_not_merge_clinical_text(tmp_path: Path) -> None:
    base = "。".join(
        f"术后第{day}天应保持眼部清洁，按时使用医生开具的眼药水，并根据恢复情况安排复查"
        for day in range(1, 21)
    ) + "。"
    instructions = [
        ("https://yk.huiyi9e.com/info/1024/10.htm", "patient_education", "canonical", _page("术后护理", f"<p>{base}术后24小时内避免揉眼。</p>")),
        ("https://yk.huiyi9e.com/info/1024/11.htm", "patient_education", "canonical", _page("术后护理", f"<p>{base}术后24小时内避免揉眼。</p>")),
        ("https://yk.huiyi9e.com/info/1024/12.htm", "patient_education", "canonical", _page("术后护理", f"<p>{base}术后48小时内避免揉眼。</p>")),
    ]
    _write_corpus_inputs(tmp_path, instructions)
    report = normalize_raw_corpus(tmp_path)
    documents = [json.loads(line) for line in (tmp_path / "normalized" / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(documents) == 3
    assert len({document["content"] for document in documents}) == 2
    assert report["dedup"]["exact_duplicate_documents"] == 2
    assert report["dedup"]["near_duplicate_flag_count"] >= 1
    assert all(pair["action"] == "flag_only" for pair in report["dedup"]["near_duplicate_pairs"])


def test_chunk_ids_are_stable_and_heading_list_units_remain_atomic(tmp_path: Path) -> None:
    root = tmp_path / "huiyi"
    root.mkdir()
    document = {
        "document_id": "huiyi-doc-test",
        "source_id": "huiyi-official-test",
        "document_type": "perioperative_instruction",
        "department": None,
        "topic": "白内障",
        "title": "白内障术后注意事项",
        "content": "## 用药\n\n- 按医院交代的方法用药。\n- 如果出现异常情况，应联系医院。\n\n## 复查\n\n按照页面说明复查。",
        "review_status": "NEEDS_HUMAN_REVIEW",
        "freshness_class": "STATIC_EDUCATION",
        "source_url": "https://yk.huiyi9e.com/info/1025/10.htm",
    }
    (root / "normalized").mkdir()
    (root / "normalized" / "documents.jsonl").write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    first = chunk_documents(root)
    contents = (root / "chunks" / "chunks.jsonl").read_bytes()
    second = chunk_documents(root)
    assert first["chunks_jsonl_sha256"] == second["chunks_jsonl_sha256"]
    assert contents == (root / "chunks" / "chunks.jsonl").read_bytes()
    rows = [json.loads(line) for line in contents.decode("utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[0]["section_path"] == ["用药"]
    assert rows[1]["section_path"] == ["用药"]
    assert rows[2]["section_path"] == ["复查"]
    assert len({row["chunk_id"] for row in rows}) == len(rows)


def test_embedding_validation_uses_normalized_1024d_vectors() -> None:
    vector = [0.0] * 1024
    vector[0] = 1.003315
    rows = Qwen3LocalEmbedder._validate_vectors([vector], 1)
    assert len(rows) == 1 and len(rows[0]) == 1024
    assert abs(sum(value * value for value in rows[0]) ** 0.5 - 1.0) < 1e-8
    with pytest.raises(ValueError, match="L2-normalized"):
        Qwen3LocalEmbedder._validate_vectors([[0.0] * 1024], 1)


def test_rrf_is_deterministic_and_combines_both_channels() -> None:
    lexical = [{"chunk_id": "a", "source_id": "src-a"}, {"chunk_id": "b", "source_id": "src-b"}]
    dense = [{"chunk_id": "b", "source_id": "src-b"}, {"chunk_id": "c", "source_id": "src-c"}]
    first = reciprocal_rank_fusion(lexical, dense, top_k=3)
    second = reciprocal_rank_fusion(lexical, dense, top_k=3)
    assert [row["chunk_id"] for row in first] == ["b", "a", "c"]
    assert first == second


@pytest.mark.integration
def test_milvus_create_insert_search_filter_drop_when_enabled() -> None:
    import os

    if os.environ.get("HUIYI_MILVUS_INTEGRATION") != "1":
        pytest.skip("Set HUIYI_MILVUS_INTEGRATION=1 with local Milvus 3.0.2 to run integration smoke")
    from health_ai_copilot.huiyi.milvus_store import HuiyiMilvusStore
    store = HuiyiMilvusStore(collection_name="huiyi_test_integration")
    try:
        store.create_collection()
        rows = [
            {
                "chunk_id": f"chunk-{index}",
                "document_id": f"doc-{index}",
                "source_id": f"src-{index}",
                "text": f"测试段落 {index}",
                "title": "测试",
                "document_type": "department" if index < 3 else "faq",
                "department": "青光眼专科",
                "topic": "青光眼",
                "review_status": "AUTO_ACCEPTED_PUBLIC_INFO",
                "freshness_class": "STATIC_PROFILE",
                "source_url": "https://yk.huiyi9e.com/",
                "content_sha256": "0" * 64,
            }
            for index in range(4)
        ]
        vectors = []
        for index in range(4):
            vector = [0.0] * 1024
            vector[index] = 1.0
            vectors.append(vector)
        assert store.insert(rows, vectors) == 4
        result = store.search(vectors[0], top_k=3)
        filtered = store.search(vectors[0], top_k=5, filter_expression='document_type == "department"')
        assert result and len(filtered) == 3
        assert all(row["document_type"] == "department" for row in filtered)
    finally:
        store.client.drop_collection(collection_name="huiyi_test_integration")
        store.close()
