"""Fetch only bounded public Huiyi HTML into content-addressed raw snapshots."""

from __future__ import annotations

import argparse
import json
import sys

from _common import DATA_ROOT, REPO_ROOT, RUN_ROOT, update_summary, write_json

sys.path.insert(0, str(REPO_ROOT / "src"))

from health_ai_copilot.huiyi.acquire import MAX_DEPTH_DEFAULT, MAX_PAGES_DEFAULT, crawl_huiyi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES_DEFAULT)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH_DEFAULT)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    args = parser.parse_args()
    report = crawl_huiyi(DATA_ROOT, max_pages=args.max_pages, max_depth=args.max_depth, delay_seconds=args.delay_seconds)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(RUN_ROOT / "source_report.json", report)
    update_summary()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
