from __future__ import annotations

import subprocess
from inspect import signature
from pathlib import Path

import pytest

from eval.rag_e6.rsel import rsel_output_row
from eval.rag_e6.scoring import _paired_cluster_bootstrap
from eval.rag_e6b.protocol import (
    BOOTSTRAP_SEED,
    PLAN_SHA256,
    POOL_SEEDS,
    RESERVED_COMPOSITIONS,
    pool_counts,
    read_json,
    validate_reserved_plan,
)
from eval.rag_e6b.runner import _paired_identity_pass
from eval.rag_e6b.scoring import _composition_labels

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_reserved_plan_seed_fidelity_and_expected_counts() -> None:
    path = REPOSITORY_ROOT / "runs/integration/u2f-owned-v1-55955b2eff38/reserved_test_plan.json"
    assert __import__("hashlib").sha256(path.read_bytes()).hexdigest() == PLAN_SHA256
    plan = read_json(path)
    validate_reserved_plan(plan)
    assert pool_counts("IID_TEST") == (512, 256, 256)
    assert pool_counts("OOD_COMPOSITION") == (1024, 512, 512)
    assert tuple(row["split_role"] for row in plan["pools"]) == tuple(POOL_SEEDS)
    assert tuple(plan["pools"][-1]["reserved_compositions"]) == RESERVED_COMPOSITIONS


def test_plan_mutation_fails_closed() -> None:
    path = REPOSITORY_ROOT / "runs/integration/u2f-owned-v1-55955b2eff38/reserved_test_plan.json"
    plan = read_json(path)
    plan["pools"][0]["persona_seed_range"][0] += 1
    with pytest.raises(ValueError, match="seed ranges changed"):
        validate_reserved_plan(plan)


def _paired_row() -> dict:
    vanilla = {
        "ranked_evidence_ids": ["D1"],
        "candidate_union_ids": ["D1", "D2"],
        "channel_ids": [["D1"], ["D2"]],
        "evidence_identity_sha256": "same",
        "evidence_aliases": [{"alias": "[E1]", "document_id": "D1"}],
        "generation_call_ids": ["EP|VANILLA_STRONG|reader"],
        "answer": "SYNVAL-ABCDEF0123",
        "answer_sha256": "answer",
        "cited_aliases": ["[E1]"],
        "used_evidence_ids": ["D1"],
        "output_contract_failure": False,
        "unknown_aliases": [],
    }
    rsel = {
        **vanilla,
        "arm": "RSEL_STRONG",
        "rsel_action": "FALLBACK_NO_MATCH",
        "rsel_ledger": [],
        # The paired-arm identity test must mutate RSEL's call list only.
        "generation_call_ids": list(vanilla["generation_call_ids"]),
    }
    return {"arms": {"VANILLA_STRONG": vanilla, "RSEL_STRONG": rsel}}


def test_paired_evidence_identity_and_no_match_exact_parity() -> None:
    row = _paired_row()
    assert _paired_identity_pass(row)
    row["arms"]["RSEL_STRONG"]["answer"] = "different"
    assert not _paired_identity_pass(row)


def test_rsel_match_can_not_change_evidence_or_add_calls() -> None:
    row = _paired_row()
    row["arms"]["RSEL_STRONG"]["rsel_action"] = "STRUCTURED_RELATION_LEDGER"
    row["arms"]["RSEL_STRONG"]["answer"] = "normalized"
    row["arms"]["RSEL_STRONG"]["answer_sha256"] = "new-answer"
    assert _paired_identity_pass(row)
    row["arms"]["RSEL_STRONG"]["generation_call_ids"].append("extra-call")
    assert not _paired_identity_pass(row)


def test_reserved_runner_never_names_or_opens_truth_artifacts() -> None:
    source = (REPOSITORY_ROOT / "eval/rag_e6b/runner.py").read_text(encoding="utf-8")
    assert "evaluator_truth.jsonl" not in source
    assert "evaluator_only" not in source
    assert "truth_path" not in source


def test_rsel_entrypoint_accepts_no_truth_or_oracle_input() -> None:
    parameters = set(signature(rsel_output_row).parameters)
    assert parameters == {"arm", "question", "evidence", "baseline_row"}
    assert not any("truth" in name or "oracle" in name for name in parameters)


def test_subject_cluster_bootstrap_is_deterministic() -> None:
    rows = []
    for subject_index in range(8):
        for episode_index in range(2):
            base = bool((subject_index + episode_index) % 2)
            rows.append({
                "subject_id": f"S{subject_index}",
                "arms": {
                    "VANILLA_STRONG": {"grounded_task_success": base},
                    "RSEL_STRONG": {"grounded_task_success": True},
                },
            })
    first = _paired_cluster_bootstrap(
        rows,
        metric="grounded_task_success",
        arm_a="VANILLA_STRONG",
        arm_b="RSEL_STRONG",
        resamples=500,
        seed=BOOTSTRAP_SEED,
    )
    second = _paired_cluster_bootstrap(
        rows,
        metric="grounded_task_success",
        arm_a="VANILLA_STRONG",
        arm_b="RSEL_STRONG",
        resamples=500,
        seed=BOOTSTRAP_SEED,
    )
    assert first == second


def test_execution_freeze_commit_gate_rejects_uncommitted_state(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "E6B test"], cwd=tmp_path, check=True)
    path = tmp_path / "runs/rag_e6b/execution/reserved_execution_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze outputs"], cwd=tmp_path, check=True)
    from eval.rag_e6b.scoring import _committed_freeze

    relative = "runs/rag_e6b/execution/reserved_execution_manifest.json"
    assert len(_committed_freeze(tmp_path, relative)) == 40
    path.write_text('{"edited":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not committed and clean"):
        _committed_freeze(tmp_path, relative)


def test_composition_slices_are_post_truth_diagnostics_only() -> None:
    row = {
        "capability_class": "MEMORY+RAG",
        "truth_metadata": {
            "structural_metadata": {
                "dependency_depth": 3,
                "evidence_branch_count": 3,
                "revision_depth": 1,
                "independent_dependency_groups": 3,
                "dependency_width": 3,
                "external_versions": 2,
            },
            "history_regime": "LONG",
        },
    }
    labels = _composition_labels(row)
    assert "deep MEMORY→RAG serial chains" in labels
    assert "multi-source RAG plus memory revision" in labels
    assert "three-branch compositional tasks" in labels
    assert "long-history plus versioned external evidence" in labels
