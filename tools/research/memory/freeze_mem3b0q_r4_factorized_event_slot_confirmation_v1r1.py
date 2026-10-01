"""Create the append-never v1r1 factorized confirmation lock."""

from __future__ import annotations

import os

from tools.research.memory import run_mem3b0q_r4_factorized_event_slot_confirmation_v1r1 as runner


def main() -> int:
    if runner.LOCK_PATH.exists() or runner.LOCK_SHA_PATH.exists():
        raise SystemExit("refusing to overwrite factorized v1r1 lock")
    payload = runner.build_lock_payload()
    data = runner.base._canonical_json(payload) + b"\n"
    digest = runner.base._sha(data)
    runner.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for path, content in (
        (runner.LOCK_PATH, data),
        (runner.LOCK_SHA_PATH, f"{digest}  {runner.LOCK_PATH.name}\n".encode("ascii")),
    ):
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    print(f"FROZEN {runner.LOCK_PATH.name} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
