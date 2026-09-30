"""Execute smoke-only or the frozen teacher-blind E5-B4 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_e5.b4_execution import (DEFAULT_B4_PRIVATE_ROOT,
                                      DEFAULT_EXECUTION_MANIFEST_PATH,
                                      DEFAULT_PROTOCOL_PATH,
                                      run_counterfactual, run_smoke)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EXECUTION_MANIFEST_PATH)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_B4_PRIVATE_ROOT)
    args = parser.parse_args()
    if args.smoke_only:
        result = run_smoke(protocol_path=args.protocol, private_root=args.private_root)
    else:
        result = run_counterfactual(
            protocol_path=args.protocol,
            manifest_path=args.manifest,
            private_root=args.private_root,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
