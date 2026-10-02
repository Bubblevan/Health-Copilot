"""Polite, bounded acquisition of public HTML from the Huiyi official domain."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .schema import SourceSpec, canonical_url, source_id_for_url

OFFICIAL_DOMAIN = "yk.huiyi9e.com"
USER_AGENT = "Health-Copilot-HuiyiCorpusBot/0.1 (+https://github.com/Bubblevan/Health-Copilot; public corpus acquisition)"
PARSER_VERSION = "huiyi-html-links-v1"
MAX_PAGES_DEFAULT = 240
MAX_DEPTH_DEFAULT = 2
POLITE_DELAY_SECONDS = 0.25
RETRY_DELAYS_SECONDS = (0.5, 1.0)

SEEDS: tuple[tuple[str, str, str, str, str], ...] = (
    ("/", "hospital_info", "raw_only_dynamic", "discovery_only", "医院首页"),
    ("/about.htm", "hospital_info", "canonical", "document", "医院介绍"),
    ("/about/yyjs.htm", "hospital_info", "canonical", "document", "医院简介"),
    ("/Service/line.htm", "hospital_info", "canonical", "document", "交通路线"),
    ("/Service/flowpath.htm", "hospital_info", "canonical", "document", "就医流程"),
    ("/Service/fixedpoint.htm", "hospital_info", "raw_only_dynamic", "document", "医保定点"),
    ("/keshi.htm", "department", "canonical", "discovery_only", "专科目录"),
    ("/zhuanjia.htm", "doctor_profile", "canonical", "discovery_only", "专家目录"),
    ("/jiangtang/aiyan_kepu.htm", "patient_education", "canonical", "discovery_only", "爱眼科普"),
    ("/jiangtang/zixun_dayi.htm", "faq", "canonical", "discovery_only", "资讯答疑"),
    ("/news.htm", "hospital_info", "raw_only_dynamic", "discovery_only", "资讯"),
    ("/news/yaowen.htm", "hospital_info", "raw_only_dynamic", "discovery_only", "慧宜要闻"),
    ("/Service/reserve.htm", "hospital_info", "raw_only_dynamic", "document", "预约挂号"),
    ("/Service/zhuanjiamz.htm", "doctor_profile", "raw_only_dynamic", "document", "专家门诊"),
)


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._label: list[str] = []
        self._in_anchor = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._label = []
            self._in_anchor = True

    def handle_data(self, data: str) -> None:
        if self._in_anchor:
            self._label.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._in_anchor:
            href = (self._href or "").strip()
            label = re.sub(r"\s+", " ", " ".join(self._label)).strip()
            if href:
                self.links.append((href, label))
            self._href = None
            self._label = []
            self._in_anchor = False


def _canonical_discovered_url(href: str, base_url: str) -> str | None:
    absolute = urljoin(base_url, href.strip())
    parts = urlsplit(absolute)
    if parts.scheme.lower() not in {"http", "https"} or (parts.hostname or "").lower() != OFFICIAL_DOMAIN:
        return None
    if parts.username or parts.password or parts.query:
        return None
    path = parts.path or "/"
    if path != "/" and not path.lower().endswith((".htm", ".html")):
        return None
    return urlunsplit(("https", OFFICIAL_DOMAIN, path, "", ""))


def classify_url(url: str, referring_type: str | None = None) -> tuple[str, str, str] | None:
    """Return (type, policy, role) only for explicitly in-scope site sections."""
    path = urlsplit(url).path.lower()
    if path in {"/", "/index.htm", "/index.html", "/news.htm", "/news/yaowen.htm"} or path.startswith(("/news/", "/info/1019/")):
        expected = referring_type or "hospital_info"
        return expected, "raw_only_dynamic", "discovery_only" if path in {"/", "/index.htm", "/index.html", "/news.htm", "/news/yaowen.htm"} else "document"
    if path.startswith(("/service/reserve", "/service/zhuanjiamz", "/service/fixedpoint")):
        return ("doctor_profile" if "zhuanjiamz" in path else "hospital_info", "raw_only_dynamic", "document")
    if path.startswith("/info/1021/"):
        return "doctor_profile", "canonical", "document"
    if path.startswith("/info/1022/"):
        return "department", "canonical", "document"
    if path.startswith("/info/1024/"):
        return "patient_education", "canonical", "document"
    if path.startswith("/info/1025/"):
        return "faq", "canonical", "document"
    if (
        path in {"/keshi.htm", "/zhuanjia.htm", "/jiangtang/aiyan_kepu.htm", "/jiangtang/zixun_dayi.htm"}
        or re.fullmatch(r"/zhuanjia/\d+\.htm", path)
        or re.fullmatch(r"/jiangtang/(?:aiyan_kepu|zixun_dayi)/\d+\.htm", path)
    ):
        expected = "department" if "keshi" in path else "doctor_profile" if "zhuanjia" in path else "patient_education" if "aiyan" in path else "faq"
        return expected, "canonical", "discovery_only"
    if path in {"/about.htm", "/about/yyjs.htm", "/service/line.htm", "/service/flowpath.htm"}:
        return "hospital_info", "canonical", "document"
    if path.startswith("/about/"):
        return "hospital_info", "canonical", "document"
    # A referral from the education / FAQ index provides the source family for
    # legacy CMS article IDs that are otherwise only numeric.
    if path.startswith(("/info/1024", "/info/1025")):
        return referring_type or "patient_education", "canonical", "document"
    return None


def _title_from_html(raw: bytes) -> str | None:
    text = raw.decode("utf-8", errors="replace")
    match = re.search(r"(?is)<title[^>]*>(.*?)</title\s*>", text)
    if not match:
        return None
    title = re.sub(r"<[^>]+>", " ", match.group(1))
    title = re.sub(r"\s+", " ", title).strip()
    suffixes = ("-恩施慧宜眼科医院〔yk.Huiyi9e.com〕", "〔yk.Huiyi9e.com〕")
    for suffix in suffixes:
        title = title.replace(suffix, "").strip()
    return title or None


def _seed_catalog() -> list[SourceSpec]:
    rows: list[SourceSpec] = []
    for path, expected, policy, role, title in SEEDS:
        url = f"https://{OFFICIAL_DOMAIN}{path}"
        rows.append(SourceSpec(
            source_id=source_id_for_url(url),
            url=canonical_url(url),
            domain=OFFICIAL_DOMAIN,
            source_family="huiyi_official",
            expected_type=expected,
            ingestion_policy=policy,
            enabled=True,
            page_role=role,
            title=title,
        ))
    return rows


def crawl_huiyi(
    data_root: Path,
    *,
    max_pages: int = MAX_PAGES_DEFAULT,
    max_depth: int = MAX_DEPTH_DEFAULT,
    delay_seconds: float = POLITE_DELAY_SECONDS,
) -> dict[str, Any]:
    """Fetch bounded same-domain HTML, preserving each response as raw bytes."""
    if max_pages <= 0 or max_depth < 0 or delay_seconds < 0:
        raise ValueError("max_pages must be positive; max_depth and delay must be non-negative")
    raw_pages = data_root / "raw" / "pages"
    raw_pages.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "raw" / "manifest.jsonl"
    catalog_path = data_root / "source_catalog.json"

    catalog: dict[str, SourceSpec] = {item.url: item for item in _seed_catalog()}
    if catalog_path.exists():
        for row in json.loads(catalog_path.read_text(encoding="utf-8")):
            spec = SourceSpec.from_dict(row)
            catalog[spec.url] = spec

    # Queue URLs, preserving the nearest source category for legacy article links.
    pending: list[tuple[str, int, str]] = []
    for spec in sorted(catalog.values(), key=lambda item: item.url):
        if spec.enabled:
            pending.append((spec.url, 0, spec.expected_type))
    scheduled: set[str] = set()
    fetched_rows: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, str]] = {}

    while pending and len(fetched_rows) + len(failures) < max_pages:
        pending.sort(key=lambda item: (
            0 if (classify_url(item[0], item[2]) or ("", "raw_only_dynamic", ""))[1] == "canonical" else 1,
            item[1],
            item[0],
            item[2],
        ))
        url, depth, inherited_type = pending.pop(0)
        normalized = canonical_url(url)
        if normalized in scheduled:
            continue
        scheduled.add(normalized)
        spec = catalog.get(normalized)
        classification = classify_url(normalized, inherited_type)
        if spec is None:
            if classification is None:
                continue
            expected, policy, role = classification
            spec = SourceSpec(
                source_id=source_id_for_url(normalized),
                url=normalized,
                domain=OFFICIAL_DOMAIN,
                source_family="huiyi_official",
                expected_type=expected,
                ingestion_policy=policy,
                page_role=role,
            )
            catalog[normalized] = spec
        if not spec.enabled:
            continue
        if (urlsplit(normalized).hostname or "").lower() != OFFICIAL_DOMAIN:
            raise ValueError(f"refusing to fetch outside official domain: {normalized}")

        response: httpx.Response | None = None
        failure_message: str | None = None
        for attempt in range(len(RETRY_DELAYS_SECONDS) + 1):
            try:
                response = httpx.get(
                    normalized,
                    headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
                    timeout=httpx.Timeout(20.0, connect=10.0),
                    follow_redirects=True,
                )
                final_host = (urlsplit(str(response.url)).hostname or "").lower()
                if final_host != OFFICIAL_DOMAIN:
                    failure_message = f"redirect left allowed domain: {final_host}"
                    response = None
                    break
                if response.status_code >= 500 and attempt < len(RETRY_DELAYS_SECONDS):
                    time.sleep(RETRY_DELAYS_SECONDS[attempt])
                    continue
                break
            except httpx.HTTPError as exc:
                failure_message = type(exc).__name__
                if attempt < len(RETRY_DELAYS_SECONDS):
                    time.sleep(RETRY_DELAYS_SECONDS[attempt])
        if response is None:
            failures[normalized] = {"source_id": spec.source_id, "error": failure_message or "request failed"}
            time.sleep(delay_seconds)
            continue
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if response.status_code != 200 or content_type not in {"text/html", "application/xhtml+xml"}:
            failures[normalized] = {
                "source_id": spec.source_id,
                "error": f"http_status={response.status_code}; content_type={content_type or 'missing'}",
            }
            time.sleep(delay_seconds)
            continue

        raw_bytes = response.content
        digest = hashlib.sha256(raw_bytes).hexdigest()
        raw_relative = Path("raw") / "pages" / f"{digest}.html"
        raw_file = data_root / raw_relative
        if not raw_file.exists():
            raw_file.write_bytes(raw_bytes)
        fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        fetched_rows[normalized] = {
            "source_id": spec.source_id,
            "url": normalized,
            "final_url": canonical_url(str(response.url)),
            "http_status": response.status_code,
            "fetched_at_utc": fetched_at,
            "content_type": content_type,
            "raw_path": raw_relative.as_posix(),
            "raw_sha256": digest,
            "parser_version": PARSER_VERSION,
            "ingestion_policy": spec.ingestion_policy,
        }
        title = _title_from_html(raw_bytes)
        catalog[normalized] = SourceSpec(
            source_id=spec.source_id,
            url=spec.url,
            domain=spec.domain,
            source_family=spec.source_family,
            expected_type=spec.expected_type,
            ingestion_policy=spec.ingestion_policy,
            enabled=spec.enabled,
            page_role=spec.page_role,
            title=spec.title or title,
        )

        if depth < max_depth:
            parser = _LinkParser()
            parser.feed(raw_bytes.decode("utf-8", errors="replace"))
            discovered: dict[str, tuple[str, str]] = {}
            for href, _label in parser.links:
                candidate = _canonical_discovered_url(href, normalized)
                if candidate is None or candidate in scheduled:
                    continue
                candidate_type = classify_url(candidate, spec.expected_type)
                if candidate_type is not None:
                    discovered[candidate] = (candidate_type[0], candidate_type[1])
            for candidate, (candidate_type, _policy) in sorted(discovered.items()):
                pending.append((candidate, depth + 1, candidate_type))
        time.sleep(delay_seconds)

    catalog_rows = [item.__dict__ for item in sorted(catalog.values(), key=lambda item: item.source_id)]
    catalog_path.write_text(
        json.dumps(catalog_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_rows = sorted(fetched_rows.values(), key=lambda item: (item["source_id"], item["url"]))
    manifest_path.write_bytes(b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in manifest_rows
    ))
    report = {
        "scope_domain": OFFICIAL_DOMAIN,
        "discovered_urls": len(catalog),
        "fetched_successfully": len(fetched_rows),
        "failed": len(failures),
        "raw_only_dynamic": sum(row["ingestion_policy"] == "raw_only_dynamic" for row in manifest_rows),
        "max_pages": max_pages,
        "max_depth": max_depth,
        "parser_version": PARSER_VERSION,
        "failures": [{"url": url, **failures[url]} for url in sorted(failures)],
    }
    (data_root / "source_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
