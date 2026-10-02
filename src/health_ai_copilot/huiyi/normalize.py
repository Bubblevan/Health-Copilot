"""Deterministic HTML normalization, review routing, and duplicate reporting."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag

from .schema import (
    CORPUS_SCHEMA_VERSION,
    CanonicalDocument,
    SourceSpec,
    jsonl_bytes,
    sha256_bytes,
    sha256_text,
)

NORMALIZER_VERSION = "huiyi-normalizer-v1"
_DATE = re.compile(r"(?P<year>20\d{2})[-年/.](?P<month>\d{1,2})[-月/.](?P<day>\d{1,2})日?")
_URL_TEXT = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_DYNAMIC_MARKERS = re.compile(
    r"本周|本周一|今日|今天|明日|明天|节假日|国庆期间|春节期间|门诊排班|专家排班|限时|截止报名|报名通道|活动报名|招募|开诊安排|照常接诊|"
    r"活动将于|活动时间|具体时间另行通知|临时通知|暂停接诊|恢复接诊|本月排班|近期排班"
)
_DATE_RANGE = re.compile(
    r"(?:20\d{2}年)?\s*\d{1,2}月\d{1,2}日\s*(?:至|到|—|–|~|～|-)\s*"
    r"(?:20\d{2}年)?\s*\d{1,2}月\d{1,2}日"
    r"|20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\s*(?:至|到|—|–|~|～|-)\s*"
    r"(?:20\d{2}[-/.])?\d{1,2}[-/.]\d{1,2}"
)
_COVID_STALE = re.compile(r"新冠|核酸检测|疫情防控|隔离观察|居家隔离|健康码|行程码")
_PERIOPERATIVE_TITLE = re.compile(r"术前|术后|手术前后|手术前|手术后|复诊须知|复查时间")
_TOPICS = (
    "白内障", "青光眼", "近视", "屈光", "斜视", "弱视", "眼底病", "糖尿病眼病",
    "干眼", "角膜", "眼表", "泪道", "眼外伤", "黄斑", "视网膜", "儿童眼科",
)


def _normalize_text(value: str) -> str:
    # NFC composes equivalent characters without rewriting full-width/CJK
    # punctuation into ASCII forms.
    value = unicodedata.normalize("NFC", value).replace("\u00a0", " ").replace("\u3000", " ")
    value = _URL_TEXT.sub("", value)
    lines = [re.sub(r"[\t \f\v]+", " ", line).strip() for line in value.splitlines()]
    compact: list[str] = []
    for line in lines:
        if not line:
            if compact and compact[-1] != "":
                compact.append("")
            continue
        compact.append(line)
    while compact and not compact[-1]:
        compact.pop()
    return "\n".join(compact)


def _parse_date(value: str) -> str | None:
    match = _DATE.search(value)
    if not match:
        return None
    try:
        return datetime(
            int(match.group("year")), int(match.group("month")), int(match.group("day")), tzinfo=UTC
        ).date().isoformat()
    except ValueError:
        return None


def _main_content(soup: BeautifulSoup) -> Tag | None:
    for selector in (
        ".v_news_content",
        "#vsb_content",
        ".main_body",
        "article",
        ".article-content",
        ".article_content",
        ".content_detail",
        ".detail_content",
    ):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            return node
    return None


def _remove_site_chrome(root: Tag) -> None:
    for node in root.select(
        "script, style, noscript, iframe, nav, footer, header, form, .related_news, "
        ".share, .share_box, .article_share, .newsdetail_operation, .editor, .source, "
        ".top_bar, .top_bar_content, .slibing_bar, .sub_nav_content, .footer_content, "
        ".breadcrumb, .crumb, .recommend, .recommend_news, .copyright"
    ):
        node.decompose()


def _blocks(root: Tag) -> list[str]:
    output: list[str] = []
    selectors = "h1, h2, h3, h4, h5, h6, p, li, blockquote, dt, dd"
    for node in root.select(selectors):
        if node.find_parent(["li", "blockquote"]):
            continue
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if node.name and node.name.startswith("h"):
            level = min(4, int(node.name[1]))
            output.append(f"{'#' * level} {text}")
        elif node.name == "li":
            output.append(f"- {text}")
        else:
            output.append(text)
    if output:
        return output
    fallback = _normalize_text(root.get_text("\n", strip=True))
    return [line for line in fallback.splitlines() if line]


def _page_title(soup: BeautifulSoup, root: Tag, source: SourceSpec) -> str:
    center = soup.select_one(".center_title")
    if isinstance(center, Tag):
        main = center.select_one(".main_title")
        sub = center.select_one(".sub_title")
        main_text = re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip() if main else ""
        sub_text = re.sub(r"\s+", " ", sub.get_text(" ", strip=True)).strip() if sub else ""
        if main_text:
            return f"{main_text} {sub_text}".strip()
    for selector in (".newsdetail_top h1", ".newsdetail_top h2", "h1", "h2", ".center_title"):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            if text and len(text) < 160:
                return text
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for suffix in ("-恩施慧宜眼科医院〔yk.Huiyi9e.com〕", "〔yk.Huiyi9e.com〕"):
        title = title.replace(suffix, "").strip()
    return title or source.title or source.source_id


def _content_date(soup: BeautifulSoup, root: Tag, raw_text: str) -> str | None:
    for selector in (".newsdetail_top", ".newsdetail_date", ".date", "time"):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            value = _parse_date(node.get_text(" ", strip=True))
            if value:
                return value
    return _parse_date(raw_text[:2000])


def _topic(title: str, content: str) -> str | None:
    combined = title + "\n" + content[:1000]
    matches = [topic for topic in _TOPICS if topic in combined]
    return matches[0] if matches else None


def _is_dynamic(text: str) -> bool:
    return bool(_DYNAMIC_MARKERS.search(text) or _DATE_RANGE.search(text))


def _document_type(expected: str, title: str, content: str) -> str:
    if expected in {"patient_education", "faq"} and _PERIOPERATIVE_TITLE.search(title):
        return "perioperative_instruction"
    if expected in {"hospital_info", "department", "doctor_profile", "faq"}:
        return expected
    return "patient_education"


def _freshness(document_type: str, published_at: str | None, fetched_at: str, content: str) -> str:
    if _is_dynamic(content):
        return "DYNAMIC"
    if published_at:
        try:
            fetched_year = int(fetched_at[:4])
            published_year = int(published_at[:4])
            if fetched_year - published_year >= 5:
                return "POTENTIALLY_STALE"
        except (ValueError, TypeError):
            pass
    if document_type in {"hospital_info", "department", "doctor_profile"}:
        return "STATIC_PROFILE"
    return "STATIC_EDUCATION"


def _rejection(source: SourceSpec, raw: dict[str, Any], reason: str, title: str | None = None) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_url": source.url,
        "title": title or source.title,
        "raw_sha256": raw["raw_sha256"],
        "reason": reason,
        "ingestion_policy": "raw_only_dynamic" if reason == "raw_only_dynamic" else source.ingestion_policy,
    }


def _near_duplicate_pairs(documents: list[CanonicalDocument], threshold: float = 0.88) -> list[dict[str, Any]]:
    shingle_sets: dict[str, set[str]] = {}
    for document in documents:
        compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", document.content))
        shingle_sets[document.document_id] = {
            compact[index:index + 5] for index in range(max(1, len(compact) - 4))
        }
    pairs: list[dict[str, Any]] = []
    ordered = sorted(documents, key=lambda item: item.document_id)
    for index, left in enumerate(ordered):
        a = shingle_sets[left.document_id]
        for right in ordered[index + 1:]:
            if left.content_sha256 == right.content_sha256:
                continue
            b = shingle_sets[right.document_id]
            if not a or not b:
                continue
            similarity = len(a & b) / len(a | b)
            if similarity >= threshold:
                pairs.append({
                    "left_document_id": left.document_id,
                    "right_document_id": right.document_id,
                    "similarity": round(similarity, 6),
                    "action": "flag_only",
                })
    return pairs


def normalize_raw_corpus(data_root: Path) -> dict[str, Any]:
    """Materialize source-versioned documents and deterministic quality manifests."""
    catalog_rows = json.loads((data_root / "source_catalog.json").read_text(encoding="utf-8"))
    sources = {source.url: source for source in (SourceSpec.from_dict(row) for row in catalog_rows)}
    raw_manifest = data_root / "raw" / "manifest.jsonl"
    raw_rows = [json.loads(line) for line in raw_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    documents: list[CanonicalDocument] = []
    rejected: list[dict[str, Any]] = []

    for raw in sorted(raw_rows, key=lambda row: (row["source_id"], row["url"])):
        source = sources.get(raw["url"])
        if source is None:
            rejected.append({"source_id": raw["source_id"], "source_url": raw["url"], "reason": "missing_source_catalog_entry"})
            continue
        if raw["ingestion_policy"] == "raw_only_dynamic" or source.ingestion_policy == "raw_only_dynamic":
            rejected.append(_rejection(source, raw, "raw_only_dynamic"))
            continue
        if source.page_role == "discovery_only":
            rejected.append(_rejection(source, raw, "discovery_index_only"))
            continue

        raw_path = data_root / Path(raw["raw_path"])
        raw_bytes = raw_path.read_bytes()
        if sha256_bytes(raw_bytes) != raw["raw_sha256"]:
            raise ValueError(f"raw snapshot SHA mismatch: {raw_path}")
        soup = BeautifulSoup(raw_bytes, "html.parser")
        root = _main_content(soup)
        title = _page_title(soup, root or soup.body or soup, source)
        if root is None:
            rejected.append(_rejection(source, raw, "main_content_not_found", title))
            continue
        _remove_site_chrome(root)
        content = _normalize_text("\n\n".join(_blocks(root)))
        if not content.strip():
            rejected.append(_rejection(source, raw, "empty_content", title))
            continue
        if _COVID_STALE.search(content):
            rejected.append(_rejection(source, raw, "obsolete_covid_measures_need_human_review", title))
            continue
        if _is_dynamic(title + "\n" + content):
            rejected.append(_rejection(source, raw, "raw_only_dynamic", title))
            continue

        document_type = _document_type(source.expected_type, title, content)
        published_at = _content_date(soup, root, raw_bytes.decode("utf-8", errors="replace"))
        freshness = _freshness(document_type, published_at, raw["fetched_at_utc"], content)
        review_status = (
            "AUTO_ACCEPTED_PUBLIC_INFO"
            if document_type in {"hospital_info", "department", "doctor_profile"}
            else "NEEDS_HUMAN_REVIEW"
        )
        if freshness == "POTENTIALLY_STALE" and review_status == "NEEDS_HUMAN_REVIEW":
            review_status = "NEEDS_HUMAN_REVIEW"
        content_hash = sha256_text(content)
        document_id = "huiyi-doc-" + hashlib.sha256(
            f"{source.source_id}\0{content_hash}".encode()
        ).hexdigest()[:24]
        department = title if document_type == "department" else None
        documents.append(CanonicalDocument(
            document_id=document_id,
            source_id=source.source_id,
            source_family=source.source_family,
            source_url=source.url,
            document_type=document_type,
            department=department,
            topic=_topic(title, content),
            title=title,
            content=content,
            published_at=published_at,
            effective_from=None,
            effective_until=None,
            fetched_at=raw["fetched_at_utc"],
            review_status=review_status,
            ingestion_policy="canonical",
            raw_sha256=raw["raw_sha256"],
            content_sha256=content_hash,
            freshness_class=freshness,
        ))

    documents.sort(key=lambda item: (item.document_id, item.source_id))
    rejected.sort(key=lambda row: (row.get("source_id", ""), row.get("reason", "")))
    normalized_dir = data_root / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    documents_bytes = jsonl_bytes([document.to_dict() for document in documents])
    rejected_bytes = jsonl_bytes(rejected)
    (normalized_dir / "documents.jsonl").write_bytes(documents_bytes)
    (normalized_dir / "rejected.jsonl").write_bytes(rejected_bytes)

    exact: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        exact[document.content_sha256].append(document.document_id)
    exact_groups = [
        {"content_sha256": digest, "document_ids": sorted(set(ids)), "action": "preserve_provenance"}
        for digest, ids in sorted(exact.items()) if len(set(ids)) > 1
    ]
    near_pairs = _near_duplicate_pairs(documents)
    dedup_report = {
        "exact_duplicate_groups": exact_groups,
        "exact_duplicate_documents": sum(len(group["document_ids"]) for group in exact_groups),
        "near_duplicate_threshold": 0.88,
        "near_duplicate_pairs": near_pairs,
        "near_duplicate_flag_count": len(near_pairs),
        "policy": "Exact copies retain each source provenance; near duplicates are flagged and never merged.",
    }
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "normalizer_version": NORMALIZER_VERSION,
        "document_count": len(documents),
        "rejected_count": len(rejected),
        "documents_jsonl_sha256": sha256_bytes(documents_bytes),
        "rejected_jsonl_sha256": sha256_bytes(rejected_bytes),
        "raw_manifest_sha256": sha256_bytes(raw_manifest.read_bytes()),
        "source_catalog_sha256": sha256_bytes((data_root / "source_catalog.json").read_bytes()),
    }
    (normalized_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    stats = {
        **manifest,
        "document_types": dict(sorted(_count_by(documents, "document_type").items())),
        "freshness_classes": dict(sorted(_count_by(documents, "freshness_class").items())),
        "review_statuses": dict(sorted(_count_by(documents, "review_status").items())),
        "raw_only_dynamic": sum(row.get("reason") == "raw_only_dynamic" for row in rejected),
        "parser_failures": sum(row.get("reason") in {"main_content_not_found", "empty_content"} for row in rejected),
        "exact_duplicate_count": dedup_report["exact_duplicate_documents"],
        "near_duplicate_flag_count": len(near_pairs),
    }
    return {"normalization": stats, "dedup": dedup_report}


def _count_by(documents: list[CanonicalDocument], field: str) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for document in documents:
        result[str(getattr(document, field))] += 1
    return dict(result)
