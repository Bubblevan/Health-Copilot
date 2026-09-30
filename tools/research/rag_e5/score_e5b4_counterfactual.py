"""Score only a complete, committed B4 execution manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from eval.rag_e5.b4_evaluator import (DEFAULT_REPORT_JSON,
                                      DEFAULT_REPORT_MARKDOWN,
                                      score_frozen_run)
from eval.rag_e5.b4_execution import (DEFAULT_B4_PRIVATE_ROOT,
                                      DEFAULT_EXECUTION_MANIFEST_PATH,
                                      DEFAULT_PROTOCOL_PATH, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EXECUTION_MANIFEST_PATH)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_B4_PRIVATE_ROOT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MARKDOWN)
    args = parser.parse_args()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(args.manifest.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(args.manifest.relative_to(ROOT))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0 or diff.returncode != 0:
        raise SystemExit("commit the frozen B4 execution manifest before opening teacher labels")
    result = score_frozen_run(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        private_root=args.private_root,
        report_json_path=args.report_json,
        report_markdown_path=args.report_markdown,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
