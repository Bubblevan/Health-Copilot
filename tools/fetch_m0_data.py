"""Fetch allow-listed M0 source candidates and two external benchmark artifacts.

This utility deliberately produces review material, not approved KnowledgeCards.
It stores provenance and short text candidates; it does not archive source HTML.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import truststore

truststore.inject_into_ssl()

DEFAULT_USER_AGENT = "Health-Copilot-M0-source-review/0.1"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CANDIDATE_CHARS = 320
DEFAULT_MAX_CANDIDATES = 30


@dataclass(frozen=True)
class RemoteArtifact:
    name: str
    url: str
    filename: str
    description: str
    archive: bool = False


BENCHMARKS = (
    RemoteArtifact(
        name="healthbench",
        url=(
            "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
            "2025-05-07-06-14-12_oss_eval.jsonl"
        ),
        filename="healthbench_oss_eval.jsonl",
        description="OpenAI HealthBench OSS evaluation cases",
    ),
    RemoteArtifact(
        name="mirage",
        url="https://raw.githubusercontent.com/gzxiong/MIRAGE/main/benchmark.json",
        filename="mirage_benchmark.json",
        description="MIRAGE medical RAG benchmark",
    ),
    RemoteArtifact(
        name="nfcorpus",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        filename="nfcorpus.zip",
        description="BEIR NFCorpus biomedical retrieval benchmark",
        archive=True,
    ),
)


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalise_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


class CandidateParser(HTMLParser):
    """Extract headings and paragraph/list candidates without retaining the page."""

    _BLOCK_TAGS: ClassVar[set[str]] = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"}
    _SKIP_TAGS: ClassVar[set[str]] = {"script", "style", "noscript", "svg", "template"}
    _EXCLUDED_TAGS: ClassVar[set[str]] = {"nav", "header", "footer", "aside"}
    _EXCLUDED_MARKERS: ClassVar[set[str]] = {
        "breadcrumb",
        "cookie",
        "footer",
        "header",
        "menu",
        "navigation",
        "sidebar",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page_title = ""
        self._skip_depth = 0
        self._skip_tag_stack: list[str] = []
        self._main_depth = 0
        self._saw_main = False
        self._active_tag: str | None = None
        self._active_parts: list[str] = []
        self._active_in_main = False
        self._section = ""
        self.candidates: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {name.lower(): (value or "").lower() for name, value in attrs}
        marker_text = f"{attrs_map.get('id', '')} {attrs_map.get('class', '')}"
        is_excluded = tag in self._EXCLUDED_TAGS or any(
            marker in marker_text for marker in self._EXCLUDED_MARKERS
        )
        if tag in self._SKIP_TAGS or is_excluded:
            self._skip_depth += 1
            self._skip_tag_stack.append(tag)
            return
        if self._skip_depth:
            return
        if tag == "main":
            self._main_depth += 1
            self._saw_main = True
            return
        if tag == "title":
            self._active_tag = tag
            self._active_parts = []
        elif tag in self._BLOCK_TAGS:
            self._finish_candidate()
            self._active_tag = tag
            self._active_parts = []
            self._active_in_main = self._main_depth > 0

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            if self._skip_tag_stack and tag == self._skip_tag_stack[-1]:
                self._skip_tag_stack.pop()
                self._skip_depth -= 1
            return
        if tag == "main" and self._main_depth:
            self._main_depth -= 1
            return
        if tag == "title" and self._active_tag == "title":
            self.page_title = _normalise_text(" ".join(self._active_parts))
            self._active_tag = None
            self._active_parts = []
        elif tag == self._active_tag and tag in self._BLOCK_TAGS:
            self._finish_candidate()

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and self._active_tag:
            self._active_parts.append(data)

    def _finish_candidate(self) -> None:
        if self._active_tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            text = _normalise_text(" ".join(self._active_parts))
            if text:
                self._section = text
        elif self._active_tag in {"p", "li"}:
            text = _normalise_text(" ".join(self._active_parts))
            if text:
                self.candidates.append(
                    {"section": self._section, "text": text, "in_main": self._active_in_main}
                )
        self._active_tag = None
        self._active_parts = []
        self._active_in_main = False


def parse_html_candidates(
    body: bytes,
    *,
    max_chars: int = DEFAULT_CANDIDATE_CHARS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Return bounded, deduplicated excerpts and no raw page content."""
    parser = CandidateParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    parser.close()
    seen: set[str] = set()
    candidates: list[dict[str, str]] = []
    source_candidates = parser.candidates
    if parser._saw_main:
        source_candidates = [item for item in source_candidates if item["in_main"]]
    for item in source_candidates:
        text = item["text"]
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        if digest in seen:
            continue
        seen.add(digest)
        excerpt = text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"
        candidates.append(
            {"candidate_id": digest, "section": item["section"], "excerpt": excerpt}
        )
        if len(candidates) >= max_candidates:
            break
    return {"page_title": parser.page_title, "candidates": candidates}


def fetch_bytes(
    url: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bytes, dict[str, str]]:
    request = Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "text/html,application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            headers = {
                "etag": response.headers.get("ETag", ""),
                "last_modified": response.headers.get("Last-Modified", ""),
                "content_type": response.headers.get("Content-Type", ""),
            }
            return response.read(), headers
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not fetch {url}: {exc}") from exc


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read source catalog {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise RuntimeError("source catalog must be a non-empty JSON array")
    required = {"id", "title", "url", "publisher", "jurisdiction", "tags"}
    seen: set[str] = set()
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict) or not required.issubset(item):
            raise RuntimeError(f"catalog item {index} is missing one of {sorted(required)}")
        if item["id"] in seen:
            raise RuntimeError(f"duplicate source id in catalog: {item['id']}")
        if not isinstance(item["url"], str) or not item["url"].startswith(("http://", "https://")):
            raise RuntimeError(f"catalog item {item['id']} must use an HTTP(S) URL")
        seen.add(item["id"])
    return sorted(data, key=lambda item: item["id"])


def crawl_sources(
    catalog_path: str | Path,
    output_dir: str | Path,
    *,
    names: Iterable[str] | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_chars: int = DEFAULT_CANDIDATE_CHARS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> list[Path]:
    """Fetch every catalog entry and write one review draft per source."""
    catalog = _load_catalog(Path(catalog_path))
    if names is not None:
        selected = set(names)
        known = {source["id"] for source in catalog}
        unknown = selected - known
        if unknown:
            raise RuntimeError(f"unknown source id(s): {', '.join(sorted(unknown))}")
        catalog = [source for source in catalog if source["id"] in selected]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source in catalog:
        body, headers = fetch_bytes(source["url"], user_agent=user_agent, timeout=timeout)
        parsed = parse_html_candidates(body, max_chars=max_chars, max_candidates=max_candidates)
        payload = {
            "record_type": "knowledge_card_review_draft",
            "review_required": True,
            "source_id": source["id"],
            "title": source["title"],
            "publisher": source["publisher"],
            "jurisdiction": source["jurisdiction"],
            "source_url": source["url"],
            "published_at": source.get("published_at"),
            "collected_at": _now(),
            "http": headers,
            "page_title": parsed["page_title"],
            "candidate_count": len(parsed["candidates"]),
            "candidates": parsed["candidates"],
            "tags": source["tags"],
            "next_step": "Manually verify, summarize into atomic cards, then set reviewed_at/reviewer.",
        }
        target = destination / f"{source['id']}.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(target)
    return written


def _download(url: str, target: Path, *, user_agent: str, timeout: int) -> tuple[int, str]:
    body, _ = fetch_bytes(url, user_agent=user_agent, timeout=timeout)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(body)
    partial.replace(target)
    return len(body), hashlib.sha256(body).hexdigest()


def _extract_zip_safely(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise RuntimeError(f"unsafe path in archive: {member.filename}")
        zip_file.extractall(destination)


def download_benchmarks(
    output_dir: str | Path,
    *,
    names: Iterable[str] | None = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[Path]:
    """Download selected upstream files and write a checksum manifest beside each."""
    selected = set(names) if names else {item.name for item in BENCHMARKS}
    known = {item.name for item in BENCHMARKS}
    unknown = selected - known
    if unknown:
        raise RuntimeError(f"unknown benchmark(s): {', '.join(sorted(unknown))}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for artifact in BENCHMARKS:
        if artifact.name not in selected:
            continue
        target = destination / artifact.filename
        size, sha256 = _download(
            artifact.url, target, user_agent=user_agent, timeout=timeout
        )
        manifest = {
            "name": artifact.name,
            "description": artifact.description,
            "source_url": artifact.url,
            "downloaded_at": _now(),
            "filename": artifact.filename,
            "bytes": size,
            "sha256": sha256,
            "archive": artifact.archive,
            "license_note": "Review the upstream repository/data terms before redistribution.",
        }
        if artifact.archive:
            extracted_to = destination / artifact.name
            _extract_zip_safely(target, extracted_to)
            manifest["extracted_to"] = str(extracted_to)
        manifest_path = target.with_suffix(target.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.extend([target, manifest_path])
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch review-safe M0 data artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crawl = subparsers.add_parser("crawl", help="fetch allow-listed official pages into review drafts")
    crawl.add_argument("--catalog", default="data/source_catalog.json")
    crawl.add_argument("--output", default="artifacts/knowledge_drafts")
    crawl.add_argument("--only", help="comma-separated source IDs; default is the whole catalog")
    crawl.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    crawl.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    crawl.add_argument("--max-chars", type=int, default=DEFAULT_CANDIDATE_CHARS)
    crawl.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)

    benchmarks = subparsers.add_parser("benchmarks", help="download the selected external benchmark files")
    benchmarks.add_argument("--output", default="artifacts/benchmarks")
    benchmarks.add_argument("--only", default="nfcorpus,mirage,healthbench")
    benchmarks.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    benchmarks.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "crawl":
            paths = crawl_sources(
                args.catalog,
                args.output,
                names=(name.strip() for name in args.only.split(",")) if args.only else None,
                user_agent=args.user_agent,
                timeout=args.timeout,
                max_chars=args.max_chars,
                max_candidates=args.max_candidates,
            )
        else:
            names = [name.strip() for name in args.only.split(",") if name.strip()]
            paths = download_benchmarks(
                args.output, names=names, user_agent=args.user_agent, timeout=args.timeout
            )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
