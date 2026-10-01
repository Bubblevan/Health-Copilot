"""Read-only counterfactuals over frozen MEM-3B0Q artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RUN_DIR = Path(__file__).resolve().parents[3] / "runs" / "memory" / "mem3" / (
    "mem3b0q-factorized-admission-20260930"
)
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "runs" / "memory" / "mem3" / (
    "mem3b0q-slot-identity-reflection-20261001"
)

INPUTS = (
    "eligible_records.jsonl",
    "b0r_identity_hints_projection.jsonl",
    "candidate_pair_manifest.jsonl",
    "seed_pairwise_verdicts.jsonl",
    "closure_pairwise_verdicts.jsonl",
    "revision_slot_manifest.json",
    "human_review_decisions_user.json",
    "post_freeze_control_audit.json",
)
KEY_STOPWORDS = frozenset(
    {"user", "state", "preference", "context", "plan", "interest", "current", "reported", "single"}
)


class IntegrityError(RuntimeError):
    """A frozen input or output artifact failed integrity validation."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_sidecar(path: Path) -> str:
    if not path.is_file():
        raise IntegrityError(f"missing_input:{path}")
    digest = sha256_bytes(path.read_bytes())
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise IntegrityError(f"missing_sidecar:{sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[0] != digest or fields[1] != path.name:
        raise IntegrityError(f"sidecar_mismatch:{path}")
    return digest


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _word_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def literal_key_anchor(key: str, proposition: str) -> tuple[bool, list[str]]:
    """Heuristic only: test whether non-generic key tokens occur in the text."""
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", key.casefold())
        if token not in KEY_STOPWORDS
    ]
    proposition_tokens = _word_tokens(proposition)
    return bool(tokens) and all(token in proposition_tokens for token in tokens), tokens


def _components(edges: list[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen: set[str] = set()
    result: list[list[str]] = []
    for vertex in sorted(adjacency):
        if vertex in seen:
            continue
        pending = [vertex]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        seen.update(component)
        if len(component) > 1:
            result.append(sorted(component))
    return result


def analyze(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    hashes = {name: verify_sidecar(run_dir / name) for name in INPUTS}
    records_rows = _load_jsonl(run_dir / "eligible_records.jsonl")
    hints_rows = _load_jsonl(run_dir / "b0r_identity_hints_projection.jsonl")
    pair_rows = _load_jsonl(run_dir / "candidate_pair_manifest.jsonl")
    verdict_rows = _load_jsonl(run_dir / "seed_pairwise_verdicts.jsonl") + _load_jsonl(
        run_dir / "closure_pairwise_verdicts.jsonl"
    )
    slots = json.loads((run_dir / "revision_slot_manifest.json").read_text(encoding="utf-8"))["slots"]
    decisions = json.loads(
        (run_dir / "human_review_decisions_user.json").read_text(encoding="utf-8")
    )["decisions"]
    controls = json.loads((run_dir / "post_freeze_control_audit.json").read_text(encoding="utf-8"))

    records = {row["memory_id"]: row for row in records_rows}
    hints = {row["memory_id"]: row for row in hints_rows}
    verdicts = {
        frozenset((row["memory_id_a"], row["memory_id_b"])): row for row in verdict_rows
    }
    decision_by_slot = {row["revision_slot_id"]: row for row in decisions}

    reviewed_pair_labels: dict[frozenset[str], str] = {}
    group_shape = Counter()
    exact_in_group = Counter()
    pair_by_ids = {
        frozenset((row["memory_id_a"], row["memory_id_b"])): row for row in pair_rows
    }
    for slot in slots:
        label = decision_by_slot[slot["revision_slot_id"]]["decision"]
        members = slot["member_memory_ids"]
        keys = {hints[mid]["attribute_key"] for mid in members}
        group_shape[(label, "same_attribute_key" if len(keys) == 1 else "multiple_attribute_keys")] += 1
        for left, right in itertools.combinations(members, 2):
            pair = frozenset((left, right))
            reviewed_pair_labels[pair] = label
            candidate = pair_by_ids.get(pair)
            if candidate and "EXACT_HINT" in candidate["candidate_sources"]:
                exact_in_group[(label, "EXACT_HINT")] += 1
            else:
                exact_in_group[(label, "NO_EXACT_HINT")] += 1

    exact_rows = [row for row in pair_rows if "EXACT_HINT" in row["candidate_sources"]]
    exact_positive = []
    grounded_singleton_edges: list[tuple[str, str]] = []
    exact_factor_counts = Counter()
    control_results: dict[str, Any] = {}
    for pair in exact_rows:
        left, right = pair["memory_id_a"], pair["memory_id_b"]
        pair_key = frozenset((left, right))
        verdict = verdicts[pair_key]
        same_identity_hint = (
            hints[left]["scope_id"] if "scope_id" in hints[left] else records[left]["scope_id"],
            hints[left]["subject_key"],
            hints[left]["attribute_key"],
        ) == (
            hints[right]["scope_id"] if "scope_id" in hints[right] else records[right]["scope_id"],
            hints[right]["subject_key"],
            hints[right]["attribute_key"],
        )
        anchor_left, _ = literal_key_anchor(
            hints[left]["attribute_key"], records[left]["proposition_text"]
        )
        anchor_right, _ = literal_key_anchor(
            hints[right]["attribute_key"], records[right]["proposition_text"]
        )
        is_positive = (
            verdict["same_state_dimension"] == "YES"
            and verdict["state_cardinality"] == "SINGLE_VALUE_AT_A_TIME"
        )
        exact_factor_counts[
            (
                "positive" if is_positive else "not_positive",
                "same_key" if same_identity_hint else "not_same_key",
                "literal_key_anchors_both" if anchor_left and anchor_right else "anchor_missing",
                verdict["state_cardinality"],
                verdict["same_state_dimension"],
                verdict["value_relation"],
            )
        ] += 1
        if is_positive:
            exact_positive.append(pair_key)
        if (
            same_identity_hint
            and anchor_left
            and anchor_right
            and verdict["state_cardinality"] == "SINGLE_VALUE_AT_A_TIME"
        ):
            grounded_singleton_edges.append((left, right))

    for name, control in (
        ("instagram", controls["instagram_positive_control"]),
        ("gym", controls["gym_cross_key_control"]),
        ("completed_purchase", controls["b0p_false_completed_purchase_control"]),
    ):
        left, right = control["memory_ids"]
        verdict = verdicts[frozenset((left, right))]
        key_equal = (
            hints[left]["subject_key"], hints[left]["attribute_key"]
        ) == (hints[right]["subject_key"], hints[right]["attribute_key"])
        anchor_left, _ = literal_key_anchor(
            hints[left]["attribute_key"], records[left]["proposition_text"]
        )
        anchor_right, _ = literal_key_anchor(
            hints[right]["attribute_key"], records[right]["proposition_text"]
        )
        control_results[name] = {
            "same_subject_attribute_hint": key_equal,
            "literal_attribute_anchor_both": anchor_left and anchor_right,
            "same_state_dimension": verdict["same_state_dimension"],
            "state_cardinality": verdict["state_cardinality"],
            "b0q_admitted": control.get("admitted_to_same_slot"),
            "counterfactual_grounded_exact_singleton_admitted": bool(
                key_equal
                and anchor_left
                and anchor_right
                and verdict["state_cardinality"] == "SINGLE_VALUE_AT_A_TIME"
            ),
            "propositions": [records[left]["proposition_text"], records[right]["proposition_text"]],
        }

    grounded_components = _components(grounded_singleton_edges)
    component_summaries = []
    for component in sorted(grounded_components, key=lambda members: (-len(members), members)):
        keys = sorted({hints[mid]["attribute_key"] for mid in component})
        component_summaries.append(
            {
                "size": len(component),
                "attribute_keys": keys,
                "sample_propositions": [records[mid]["proposition_text"] for mid in component[:3]],
            }
        )

    return {
        "schema_version": 1,
        "stage": "MEM3B0Q_SLOT_IDENTITY_REFLECTION",
        "classification": "READ_ONLY_POST_FREEZE_COUNTERFACTUAL_NOT_A_NEW_ADMISSION_RESULT",
        "analysis_code_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "inputs_sha256": hashes,
        "inputs_unchanged": True,
        "model_calls": 0,
        "hosted_calls": 0,
        "memory_store_mutations": 0,
        "dev_test_or_medmemorybench_opened": False,
        "b0q_review": {
            "reviewed_slots": len(slots),
            "slot_labels": dict(sorted(Counter(row["decision"] for row in decisions).items())),
            "slot_shape_by_attribute_key_cardinality": {
                f"{label}|{shape}": count
                for (label, shape), count in sorted(group_shape.items())
            },
            "intra_slot_pair_source_by_label": {
                f"{label}|{source}": count
                for (label, source), count in sorted(exact_in_group.items())
            },
            "exact_hint_intra_slot_pairs": {
                label: count
                for (label, source), count in sorted(exact_in_group.items())
                if source == "EXACT_HINT"
            },
        },
        "exact_hint_counterfactual": {
            "candidate_pairs": len(exact_rows),
            "pairs_meeting_frozen_b0q_positive_rule": len(exact_positive),
            "frozen_positive_pairs_in_reviewed_slots_by_label": dict(
                sorted(
                    Counter(reviewed_pair_labels.get(pair, "UNREVIEWED") for pair in exact_positive).items()
                )
            ),
            "factor_cross_tab": {
                "|".join(key): count for key, count in sorted(exact_factor_counts.items())
            },
        },
        "literal_key_anchor_heuristic": {
            "definition": "All non-stopword tokens in attribute_key must occur as exact lowercase word tokens in both source propositions; diagnostic heuristic only, no synonym expansion.",
            "candidate_edges": len(grounded_singleton_edges),
            "component_count": len(grounded_components),
            "component_size_histogram": dict(
                sorted(Counter(str(len(members)) for members in grounded_components).items())
            ),
            "components": component_summaries,
            "controls": control_results,
            "reviewed_intra_slot_pairs_admitted": dict(
                sorted(
                    Counter(
                        reviewed_pair_labels.get(frozenset(edge), "UNREVIEWED")
                        for edge in grounded_singleton_edges
                    ).items()
                )
            ),
            "heuristic_warning": "Literal token support is not semantic proof. Its errors do not establish causality and this rule is not recommended as a finalized method.",
        },
        "interpretation": {
            "failure": "Pairwise same-dimension/cardinality is not a stable state-slot identity relation; it collapses propositions about related but distinct attributes, objects, and events.",
            "identity_failure_examples": [
                "workout_preference was attached to both a 7-minute workout and a gaming-PC plan; the key was not grounded in the latter proposition.",
                "reported_value grouped 14 separately dated numeric reports despite the key carrying no measured-attribute identity.",
                "trip_planning grouped two destination-specific plans; the user can hold both plans concurrently.",
            ],
            "revision_failure": "Even correct same-slot identity does not prove a revision: repeated same-value mentions can be corroboration/no-op, while changes need a valid time and update relation.",
            "next_research_direction": "Prototype evidence-span-grounded subject/object/attribute/value proposals, deterministic validation and exact slot keys, plus a separate update-evidence gate. Unresolved identity remains coexisting history; no automatic merge.",
            "readiness": "MEM3B1_READY=NO",
        },
    }


def _write_frozen(
    path: Path, content: bytes, *, allow_code_hash_refresh: bool = False
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            can_refresh_code_hash = False
            if allow_code_hash_refresh and path.name == "reflection_counterfactual.json":
                previous = json.loads(path.read_text(encoding="utf-8"))
                proposed = json.loads(content.decode("utf-8"))
                previous.pop("analysis_code_sha256", None)
                proposed.pop("analysis_code_sha256", None)
                can_refresh_code_hash = previous == proposed
            if not can_refresh_code_hash:
                raise IntegrityError(f"refuse_to_overwrite_frozen_output:{path}")
            path.write_bytes(content)
    else:
        path.write_bytes(content)
    digest = sha256_bytes(content)
    sidecar = path.with_suffix(".sha256")
    sidecar_content = f"{digest}  {path.name}\n".encode()
    sidecar_mismatch = sidecar.exists() and sidecar.read_bytes() != sidecar_content
    if sidecar_mismatch and not (
        allow_code_hash_refresh and path.name == "reflection_counterfactual.json"
    ):
        raise IntegrityError(f"frozen_output_sidecar_mismatch:{sidecar}")
    if not sidecar.exists() or sidecar_mismatch:
        sidecar.write_bytes(sidecar_content)
    return digest


def write_outputs(result: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> None:
    json_content = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_frozen(
        output_dir / "reflection_counterfactual.json",
        json_content,
        allow_code_hash_refresh=True,
    )
    lines = [
        "# MEM-3B0Q Slot-Identity Failure Reflection",
        "",
        "Status: read-only post-freeze counterfactual. This report does not alter B0Q, admit slots, or authorize B1.",
        "",
        "## Frozen review evidence",
        "",
        f"- Reviewed slots: {result['b0q_review']['reviewed_slots']}; labels: `{result['b0q_review']['slot_labels']}`.",
        f"- Slot shape by attribute-key cardinality: `{result['b0q_review']['slot_shape_by_attribute_key_cardinality']}`.",
        f"- Intra-slot pair source by label: `{result['b0q_review']['intra_slot_pair_source_by_label']}`.",
        "- Every false-merge slot spans multiple B0R attribute keys; most true slots also span multiple keys. Exact-key-only therefore trades away substantial recall and is not a sufficient method.",
        "",
        "## Counterfactuals",
        "",
        f"- Exact-hint candidates: {result['exact_hint_counterfactual']['candidate_pairs']}; existing B0Q-positive rule selects {result['exact_hint_counterfactual']['pairs_meeting_frozen_b0q_positive_rule']}.",
        f"- A literal attribute-key token heuristic plus same subject/key and single-value cardinality selects {result['literal_key_anchor_heuristic']['candidate_edges']} edges in components with size histogram `{result['literal_key_anchor_heuristic']['component_size_histogram']}`.",
        f"- Control outcomes under that diagnostic heuristic: `{ {name: row['counterfactual_grounded_exact_singleton_admitted'] for name, row in result['literal_key_anchor_heuristic']['controls'].items()} }`.",
        "- It admits the Instagram positive, blocks the gym and completed-purchase controls, but still joins 14 `reported_value` reports; a token-overlap gate does not establish a well-typed state slot.",
        "- The heuristic is deliberately labelled non-causal and is not recommended for adoption: it also misses reviewed exact-key true/uncertain pairs when their attribute label is paraphrastic.",
        "",
        "## Root cause and next question",
        "",
        "B0Q tests whether two propositions describe one single-valued dimension, but admission needs a stronger relation: same subject, same object/target, same attribute, and actual update evidence. A semantically related pair or shared broad key is not enough. Repeated mentions may be duplicate support, concurrently valid facts, distinct objects, or revisions; temporal order alone cannot decide which.",
        "",
        "Next development target: evidence-span-grounded subject/object/attribute/value proposals with deterministic validation and exact slot keys, followed by a separate update-evidence gate. If a slot cannot be grounded unambiguously, retain coexisting history and do not materialize a revision. This is a design hypothesis, not an evaluated result.",
        "",
        "## Scope and guardrails",
        "",
        "- Frozen B0Q inputs were only read and SHA-verified; no B0Q artifact was changed.",
        "- No model, hosted API, MemoryStore mutation, DEV/TEST question, or MedMemoryBench run was used.",
        "- `MEM3B1_READY=NO`; no materializer, CURRENT/AS_OF/CHANGE, or public scorecard is authorized by this diagnostic.",
        "- Full machine-readable counts and input hashes are in `reflection_counterfactual.json`.",
        "",
    ]
    _write_frozen(output_dir / "report.md", "\n".join(lines).encode("utf-8"))


def main() -> int:
    result = analyze()
    write_outputs(result)
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "classification": result["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
