"""Immutable identities and protocol checks for RAG-E6B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "rag-e6b-reserved-confirmation-v1"
U2F_ROOT_RELATIVE = Path("runs/integration/u2f-owned-v1-55955b2eff38")
RUN_ROOT_RELATIVE = Path("runs/rag_e6b")
PLAN_SHA256 = "db2c7bf4158374a09502a52e8ccb1671c43332aaec9e09913a9a31d0d70f7d1b"
U2F_MANIFEST_SHA256 = "34e6d1a8123ee63eea220c1a1b9b0b9b5f2e3349aeeec05a4504a508d4ca814e"
U2F_DATASET_ROOT_SHA256 = "e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134"
FROZEN_DEV_COMMIT = "f894c99026d823d3ba33834c1a660280ad7ba79e"
FROZEN_DEV_MANIFEST_SHA256 = "da3677baeae7d109b71b8ee619c37514a2c86410639511e4ded518aa11fa514a"
MODEL_SHA256 = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
BGE_SHA256 = "45e1954914e29bd74080e6c1510165274ff5279421c89f76c418878732f64ae7"
LAMER_COMMIT = "11244a4925a39082967a6c9d38ef01f279c316a5"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261002
ARMS = ("VANILLA_STRONG", "RSEL_STRONG")
POOL_SEEDS: dict[str, dict[str, tuple[int, int]]] = {
    "IID_TEST": {"persona": (510000, 510255), "scenario": (4000000, 4000255)},
    "OOD_PATIENT": {"persona": (520000, 520255), "scenario": (4100000, 4100255)},
    "OOD_TASK": {"persona": (530000, 530255), "scenario": (4200000, 4200255)},
    "OOD_TEMPORAL": {"persona": (540000, 540255), "scenario": (4300000, 4300255)},
    "OOD_SOURCE": {"persona": (550000, 550255), "scenario": (4400000, 4400255)},
    "OOD_COMPOSITION": {"persona": (560000, 560511), "scenario": (4500000, 4500511)},
}
OOD_POOLS = tuple(name for name in POOL_SEEDS if name.startswith("OOD_"))
RESERVED_COMPOSITIONS = (
    "deep MEMORY→RAG serial chains",
    "multi-source RAG plus memory revision",
    "three-branch compositional tasks",
    "long-history plus versioned external evidence",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def validate_reserved_plan(plan: dict[str, Any]) -> None:
    if (
        plan.get("schema_version") != "u2f-reserved-pools-v1"
        or plan.get("materialized") is not False
        or plan.get("rows_generated") != 0
    ):
        raise ValueError("reserved plan is not the frozen, unmaterialized U2-F plan")
    pools = plan.get("pools")
    if not isinstance(pools, list) or len(pools) != len(POOL_SEEDS):
        raise ValueError("reserved plan pool count differs from the frozen protocol")
    by_name = {item.get("split_role"): item for item in pools if isinstance(item, dict)}
    if tuple(by_name) != tuple(POOL_SEEDS):
        raise ValueError("reserved pool names/order differ from the frozen protocol")
    for name, expected in POOL_SEEDS.items():
        item = by_name[name]
        actual = {
            "persona": tuple(item.get("persona_seed_range", ())),
            "scenario": tuple(item.get("scenario_seed_range", ())),
        }
        if actual != expected:
            raise ValueError(f"reserved seed ranges changed for {name}")
        if name == "OOD_COMPOSITION":
            if tuple(item.get("reserved_compositions", ())) != RESERVED_COMPOSITIONS:
                raise ValueError("OOD_COMPOSITION requirements changed")
        elif "reserved_compositions" in item:
            raise ValueError(f"unexpected composition override in {name}")


def pool_counts(name: str) -> tuple[int, int, int]:
    """Return (episodes, subjects, sibling pairs) from the frozen seed ranges."""
    seeds = POOL_SEEDS[name]
    subject_count = seeds["persona"][1] - seeds["persona"][0] + 1
    pair_count = seeds["scenario"][1] - seeds["scenario"][0] + 1
    return pair_count * 2, subject_count, pair_count
