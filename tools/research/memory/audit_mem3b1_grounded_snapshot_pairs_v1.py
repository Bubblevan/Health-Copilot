"""Source-only audit of a grounded scalar snapshot and same-key hard negatives."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.research.memory.audit_mem3b1_dev_alternative_cues_v2 import (
    DATASET_PATH_CANDIDATES,
    DATASET_SHA256,
    _sha256,
)
from tools.research.memory.run_mem3b1_dev_revision_pair_diagnostic_v1 import (
    _freeze_artifact,
    _selected_dev_records,
    _source_turns,
)

B0Q = ROOT / "runs/memory/mem3/mem3b0q-factorized-admission-20260930"
SPLIT_PATH = ROOT / "docs/research/memory/split_manifest.json"
OUTPUT = ROOT / "runs/memory/mem3/mem3b1-grounded-snapshot-pairs-audit-v1"
QUESTION_IDS = {"1cea1afa", "a82c026e", "06878be2"}
KEY_TERMS = re.compile(r"[a-z0-9]+")
NUMBER = re.compile(r"(?<![a-z])\d+(?:,\d{3})*(?:\.\d+)?", re.IGNORECASE)
SOURCE_PATTERNS = {
    "500": re.compile(r"I just reached 500 followers last week", re.IGNORECASE),
    "600": re.compile(r"I'm now at 600 followers", re.IGNORECASE),
    "bodyweight": re.compile(r"some bodyweight exercises, maybe the 7-minute workout", re.IGNORECASE),
    "gaming_pc": re.compile(r"building a gaming PC", re.IGNORECASE),
    "trip_india": re.compile(r"trip to India for my uncle's 60th birthday celebration in Pune", re.IGNORECASE),
    "trip_japan": re.compile(r"trip to Japan soon", re.IGNORECASE),
}
EXPECTED_SOURCE = {
    "500": ("1cea1afa", 35),
    "600": ("1cea1afa", 217),
    "bodyweight": ("a82c026e", 39),
    "gaming_pc": ("a82c026e", 155),
    "trip_india": ("06878be2", 81),
    "trip_japan": ("06878be2", 150),
}
EXPECTED_B0Q_ARTIFACTS = {
    "b0r_identity_hints_projection.jsonl": "e378e8b676d3a97ec2816ed06a30394ebb0a2a7e5c222da607460aca94388c6f",
    "candidate_pair_manifest.jsonl": "4cc70617a34e4d4e50a5c9b75ea996cfb30be95b892666cc73f7950ce417def5",
    "eligible_records.jsonl": "1ed014923a7213106e4da384929c6d303ee60a4a607e8b2942231dc03718da7c",
    "pairwise_verdicts.jsonl": "5babc1e6c9b11dcbc9b46ba2cf334a8c0404604e9f3f1fe686b06990df19eb65",
    "revision_semantic_source_projection.jsonl": "7acd97a0516f7fa59d33c369c1e62b120b9ccd0ac8c9def01fff00f5cf635990",
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"expected JSON objects in JSONL: {path}")
    return rows


def _verify_b0q_inputs() -> dict[str, str]:
    manifest = _read_json(B0Q / "run_manifest.json")
    frozen = manifest["frozen_artifacts"]
    hashes: dict[str, str] = {}
    for name, expected in EXPECTED_B0Q_ARTIFACTS.items():
        path = B0Q / name
        digest = _sha_bytes(path.read_bytes())
        if digest != expected or frozen.get(name) != expected:
            raise RuntimeError(f"frozen B0Q artifact changed: {name}")
        hashes[name] = digest
    return hashes


def _key_terms_are_grounded(attribute_key: str, proposition: str) -> bool:
    key_terms = KEY_TERMS.findall(attribute_key.casefold())
    proposition_terms = set(KEY_TERMS.findall(proposition.casefold()))
    singular_terms = {
        token[:-1]
        for token in proposition_terms
        if len(token) > 4 and token.endswith("s")
    }
    proposition_terms.update(singular_terms)
    return bool(key_terms) and all(
        term in proposition_terms
        or (len(term) > 4 and term.endswith("s") and term[:-1] in proposition_terms)
        for term in key_terms
    )


def _pair_lookup(
    records: dict[str, dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    verdict_rows: list[dict[str, Any]],
    *,
    scope_id: str,
    attribute_key: str,
    first_phrase: str,
    second_phrase: str,
) -> dict[str, Any]:
    selected = [
        row for row in records.values()
        if row["scope_id"] == scope_id and row["attribute_key"] == attribute_key
    ]
    first = [row for row in selected if first_phrase.casefold() in row["proposition_text"].casefold()]
    second = [row for row in selected if second_phrase.casefold() in row["proposition_text"].casefold()]
    if len(first) != 1 or len(second) != 1 or first[0]["memory_id"] == second[0]["memory_id"]:
        raise ValueError(f"pair members are not unique: {scope_id} / {attribute_key}")
    left, right = first[0], second[0]
    id_pair = tuple(sorted((left["memory_id"], right["memory_id"])))
    candidates = [
        row for row in candidate_rows
        if tuple(sorted((row["memory_id_a"], row["memory_id_b"]))) == id_pair
    ]
    verdicts = [
        row for row in verdict_rows
        if tuple(sorted((row["memory_id_a"], row["memory_id_b"]))) == id_pair
    ]
    if len(candidates) != 1 or len(verdicts) != 1:
        raise ValueError(f"frozen B0Q pair result is not unique: {id_pair}")
    return {
        "scope_id": scope_id,
        "attribute_key": attribute_key,
        "subject_keys": sorted({left["subject_key"], right["subject_key"]}),
        "memory_ids": [left["memory_id"], right["memory_id"]],
        "propositions": [left["proposition_text"], right["proposition_text"]],
        "attribute_key_lexically_grounded_in_both_propositions": all(
            _key_terms_are_grounded(attribute_key, row["proposition_text"])
            for row in (left, right)
        ),
        "candidate_sources": candidates[0]["candidate_sources"],
        "model_verdict": {
            key: verdicts[0].get(key)
            for key in ("same_state_dimension", "state_cardinality", "value_relation")
        },
        "pair_id": candidates[0]["pair_id"],
    }


def _source_evidence(
    turns_by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name, pattern in SOURCE_PATTERNS.items():
        qid, expected_position = EXPECTED_SOURCE[name]
        matches: list[tuple[dict[str, Any], re.Match[str]]] = []
        for turn in turns_by_id[qid]:
            match = pattern.search(str(turn["text"]))
            if match is not None:
                matches.append((turn, match))
        if len(matches) != 1:
            raise ValueError(f"source evidence match is not unique: {name} ({len(matches)})")
        turn, match = matches[0]
        if turn["source_position"] != expected_position:
            raise ValueError(f"source position drift: {name}")
        source_text = str(turn["text"])
        evidence[name] = {
            "question_id": qid,
            "source_position": turn["source_position"],
            "session_date": turn["session_date"],
            "source_turn_sha256": _sha_bytes(source_text.encode("utf-8")),
            "source_quote": match.group(0),
            "source_role": "user",
            "question_or_gold_used": False,
        }
    return evidence


def build_audit() -> dict[str, Any]:
    split_bytes = SPLIT_PATH.read_bytes()
    split = json.loads(split_bytes.decode("utf-8"))
    dev_ids = set(split["dev"]["question_ids"])
    if not QUESTION_IDS.issubset(dev_ids):
        raise ValueError("a source-audit case is outside frozen DEV")
    dataset_path = next((path for path in DATASET_PATH_CANDIDATES if path.is_file()), None)
    if dataset_path is None:
        raise FileNotFoundError("LongMemEval-S dataset not found")
    dev_records = _selected_dev_records(dataset_path, QUESTION_IDS)
    turns_by_id = {qid: _source_turns(record) for qid, record in dev_records.items()}

    b0q_hashes = _verify_b0q_inputs()
    eligible_rows = _read_jsonl(B0Q / "eligible_records.jsonl")
    candidate_rows = _read_jsonl(B0Q / "candidate_pair_manifest.jsonl")
    verdict_rows = _read_jsonl(B0Q / "pairwise_verdicts.jsonl")
    eligible = {row["memory_id"]: row for row in eligible_rows}
    if len(eligible) != len(eligible_rows):
        raise ValueError("duplicate frozen B0Q eligible memory ID")

    critical = _read_json(B0Q / "critical_case_review.json")
    instagram_ids = critical["instagram_control_memory_ids"]
    instagram_rows = [eligible[mid] for mid in instagram_ids]
    if len(instagram_rows) != 2:
        raise ValueError("expected two frozen Instagram control rows")
    if len({row["attribute_key"] for row in instagram_rows}) != 1:
        raise ValueError("Instagram control attribute key is not identical")
    instagram_pair = _pair_lookup(
        eligible,
        candidate_rows,
        verdict_rows,
        scope_id="longmemeval:1cea1afa",
        attribute_key="instagram_followers",
        first_phrase="600 followers on Instagram",
        second_phrase="500 followers on Instagram",
    )

    workout_ids = critical["gym_control_memory_ids"]
    workout_rows = [eligible[mid] for mid in workout_ids]
    if len(workout_rows) != 2 or len({row["attribute_key"] for row in workout_rows}) != 1:
        raise ValueError("expected frozen same-key gym hard negative")
    workout_pair = _pair_lookup(
        eligible,
        candidate_rows,
        verdict_rows,
        scope_id="longmemeval:a82c026e",
        attribute_key=workout_rows[0]["attribute_key"],
        first_phrase="bodyweight exercises",
        second_phrase="gaming PC",
    )

    trip_pair = _pair_lookup(
        eligible,
        candidate_rows,
        verdict_rows,
        scope_id="longmemeval:06878be2",
        attribute_key="trip_planning",
        first_phrase="Japan",
        second_phrase="India",
    )

    # Audit exact-key pairs without treating lexical support as proof of a revision.
    verdict_by_pair = {
        tuple(sorted((row["memory_id_a"], row["memory_id_b"]))): row
        for row in verdict_rows
    }
    exact_pairs = [row for row in candidate_rows if "EXACT_HINT" in row["candidate_sources"]]
    model_no = 0
    fully_grounded = 0
    grounded_model_no = Counter()
    for pair in exact_pairs:
        left, right = eligible[pair["memory_id_a"]], eligible[pair["memory_id_b"]]
        verdict = verdict_by_pair[tuple(sorted((left["memory_id"], right["memory_id"]))) ]
        if verdict["same_state_dimension"] == "NO":
            model_no += 1
        grounded = all(
            _key_terms_are_grounded(row["attribute_key"], row["proposition_text"])
            for row in (left, right)
        )
        if grounded:
            fully_grounded += 1
            if verdict["same_state_dimension"] == "NO":
                grounded_model_no[left["attribute_key"]] += 1

    source = _source_evidence(turns_by_id)
    target_values = []
    for row in (instagram_rows[1], instagram_rows[0]):
        numbers = NUMBER.findall(row["proposition_text"])
        if len(numbers) != 1:
            raise ValueError("Instagram proposition must contain one numeric snapshot")
        target_values.append(numbers[0].replace(",", ""))
    if target_values != ["500", "600"]:
        raise ValueError(f"Instagram snapshot value drift: {target_values}")
    if not source["500"]["source_position"] < source["600"]["source_position"]:
        raise ValueError("Instagram source observations are not ordered")
    if not instagram_pair["attribute_key_lexically_grounded_in_both_propositions"]:
        raise ValueError("Instagram key is not source-proposition grounded")

    return {
        "schema_version": 1,
        "audit_id": "mem3b1-grounded-snapshot-pairs-audit-v1",
        "diagnostic_only": True,
        "question_or_gold_decoded": False,
        "model_calls": 0,
        "hosted_calls": 0,
        "dataset_sha256": DATASET_SHA256,
        "split_manifest_sha256": _sha_bytes(split_bytes),
        "scope": "three frozen DEV source histories and the pinned B0Q identity/pair artifacts; no TEST rows",
        "frozen_dev_ids": sorted(QUESTION_IDS),
        "b0q_artifact_sha256": b0q_hashes,
        "source_evidence": source,
        "positive_control": {
            "label": "SOURCE_ANCHORED_SCALAR_SNAPSHOT_PAIR",
            "review_status": "manual source review for this diagnostic only",
            "pair": instagram_pair,
            "ordered_scalar_values": ["500", "600"],
            "ordered_source_positions": [source["500"]["source_position"], source["600"]["source_position"]],
            "finding": "Distinct dated observations of the same explicitly named Instagram follower-count metric; B0Q pair verifier rejected the shared dimension despite identical, proposition-grounded typed identity.",
            "not_a_general_accuracy_estimate": True,
        },
        "hard_negatives": [
            {
                "label": "SAME_KEY_BUT_UNGROUNDED_ATTRIBUTE_COLLISION",
                "review_status": "source-only manual hard negative",
                "pair": workout_pair,
                "finding": "The shared workout_preference key is not grounded in the gaming-PC proposition; retain separate facts and do not override the verifier.",
            },
            {
                "label": "SAME_BROAD_EVENT_KEY_DIFFERENT_EVENT_INSTANCES",
                "review_status": "source-only manual hard negative",
                "pair": trip_pair,
                "finding": "Both propositions concern trip planning, but the source describes different destination trips; the broad key is not a same-trip identifier, so preserve coexistence.",
            },
        ],
        "exact_hint_pair_audit": {
            "pair_count": len(exact_pairs),
            "model_same_dimension_no_count": model_no,
            "both_propositions_fully_support_all_literal_attribute_key_terms": fully_grounded,
            "grounded_model_no_by_attribute_key": dict(sorted(grounded_model_no.items())),
            "interpretation": "Exploratory disagreement inventory only; exact-key membership and lexical support do not establish supersession.",
        },
        "method_boundary": [
            "Do not equate exact key equality with a revision: the gym and trip controls block that shortcut.",
            "A narrowly typed scalar snapshot path is a candidate for prospective testing only when the metric key is grounded in both source-backed propositions, each has one distinct scalar value, and independent observation times are ordered.",
            "The current artifact has one clear scalar-snapshot positive and two selected hard negatives; no admission precision or benchmark score is estimated.",
        ],
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite audit output: {OUTPUT}")
    result = build_audit()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _freeze_artifact(OUTPUT / "snapshot_pair_audit.json", payload)
    report = (
        "# Grounded Scalar Snapshot Pair Audit\n\n"
        "This is a source-only DEV diagnostic, not a benchmark result or a revision-admission quality estimate. "
        "It decoded no question/gold fields and made no model or hosted calls.\n\n"
        "## Positive Control\n\n"
        "In frozen DEV case `1cea1afa`, the user reports reaching 500 Instagram followers at source position 35 "
        "(2023-05-27), then reports being at 600 at source position 217 (2023-05-28). The two frozen B0Q records "
        "share scope, subject, and the exact `instagram_followers` key; both propositions explicitly ground the key. "
        "The local pair verifier nevertheless returned `same_state_dimension=NO`, while also returning "
        "`state_cardinality=SINGLE_VALUE_AT_A_TIME` and `value_relation=DIFFERENT`. The pair was not admitted as a slot.\n\n"
        "This is a source-anchored cross-turn scalar snapshot pair, unlike the five earlier `instead of` change mentions. "
        "The earlier 0/5 exact-predecessor finding applies only to those selected cue mentions; it does not imply that "
        "the DEV histories contain no independently anchored updates. The verifier failure is model judgment drift, "
        "not a missing timestamp or missing proposition identity.\n\n"
        "## Hard Negatives\n\n"
        "- The `workout_preference` key collides across a bodyweight-workout statement and a gaming-PC statement. The key "
        "is not grounded in the second proposition, so exact-key equality must not force same-slot admission.\n"
        "- Two `trip_planning` propositions describe separate Japan and India trips. Both share the broad key and lexical "
        "terms, but no common trip instance is established; preserve them as separate events.\n\n"
        "## What This Changes\n\n"
        "A deterministic resolver should not replace pairwise LLM judgment with blind key equality. The next testable "
        "hypothesis is a narrow scalar-snapshot path: require a proposition-grounded metric type, source-backed distinct "
        "scalar values, and independently ordered observation times; retain event-instance qualification for plans and "
        "other multi-instance domains. The pair verifier may propose cardinality/value relation, but cannot veto a "
        "type-certified dimension match. This remains a hypothesis: one positive and two selected hard negatives do not "
        "validate the rule or support a performance claim.\n\n"
        "`MEM3B1_SLOT_ADMISSION_READY=NO` remains unchanged. No 102-case DEV, TEST, MedMemoryBench, or performance "
        "ranking was run.\n"
    )
    _freeze_artifact(OUTPUT / "report.md", report.encode("utf-8"))
    print(json.dumps({
        "output": str(OUTPUT),
        "positive_control": result["positive_control"]["label"],
        "source_order": result["positive_control"]["ordered_source_positions"],
        "model_verdict": result["positive_control"]["pair"]["model_verdict"],
        "hard_negatives": len(result["hard_negatives"]),
        "exact_hint_pairs": result["exact_hint_pair_audit"]["pair_count"],
        "grounded_model_no_by_key": result["exact_hint_pair_audit"]["grounded_model_no_by_attribute_key"],
        "question_or_gold_decoded": result["question_or_gold_decoded"],
        "model_calls": result["model_calls"],
    }, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
