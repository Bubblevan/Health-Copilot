# U2-F Closeout — Owned Longitudinal Universe

Date: 2026-09-30

Status: **PASS — frozen project-owned synthetic TRAIN/DEV universe**

## Frozen artifact

| Field | Value |
|---|---|
| Dataset ID | `health-copilot-owned-longitudinal-v1` |
| Run ID | `55955b2eff38` |
| Run directory | `runs/integration/u2f-owned-v1-55955b2eff38` |
| Dataset root hash | `e28ea9ef9ecae47d3f27f28c68042066e9af297fe808cafebf1d3c8c80fb2134` |
| Generator commit | `fdaf394be287c8523c394fc626c42cd86e602745` |
| Branch | `integration-u2d-dataset-split-license-20260929` |
| Base commit | `ce68eb7d37463d255a8abd3ad0a1b73cfbbf518e` |
| Manifest status | `PASS` |

The two isolated allowlisted generations were byte-identical. The wrapper compared 35 files including the manifest; an independent SHA-256 pass found zero mismatches across the 34 files listed in `manifest.json`. The complete frozen manifest and per-artifact hashes live in the run directory.

## Publication storage

The GitHub publication snapshot stores `latent_worlds.jsonl` and `natural_language_realizations.jsonl` with Git LFS because each exceeds GitHub's 100 MB Git-blob limit. Their LFS object IDs match the artifact SHA-256 values in the manifest, so the released file contents remain byte-identical to this frozen run. The original raw-blob commit `da89d1ef4e4577c0bf4b98e0a3835f1bc894c328` remains preserved locally; the LFS publication snapshot has a distinct commit SHA.

## Qualification results

- TRAIN: 4,096 episodes / 320 subjects; DEV_IID: 512 / 64; DEV_STRUCTURAL: 512 / 64. Total: 5,120 episodes / 448 subjects.
- The 64 seeded human spot examples passed review: TRAIN 32, DEV_IID 16, DEV_STRUCTURAL 16. No blank query slots, exact duplicate queries, template artifacts, label leakage, impossible grammar, or unintended clinical claims were observed. No examples were manually edited.
- The runtime suite passed all 40,960 deterministic action arms: zero capability/evaluator mismatches, zero failed expected-capability arms, and zero provider calls.
- Source-independence, split, lineage, temporal/revision, duplicate, reserved-plan, and scale gates all passed. The reserved TEST/OOD plan generated zero rows.
- Maximum single-feature accuracy was 0.6531 and maximum lift over majority was 0.0852. Cheap-classifier maxima were 0.8199 on DEV_IID and 0.7892 on DEV_STRUCTURAL; query-only maxima were 0.8135 and 0.7800. All remain below the 0.90 review trigger.
- The final spot-packet audit recorded zero exact duplicate query groups. The full duplicate audit permits duplicates only inside designated matched groups and reports zero unapproved duplicates or cross-split normalized overlap.

The repository `uv` environment passed the regression suite (`643 passed, 2 skipped`), Ruff (`All checks passed`), and `compileall`. One upstream `jieba` deprecation warning was emitted during pytest. No training, fine-tuning, or post-training run was started.

## Data-use boundary

The manifest qualifies partial supervision for `memory_read`, `external_retrieval`, and `answerability`. Architecture and budget remain unresolved; full execution-policy training and post-training are not ready. All rows are synthetic. Evidence-family labels do not qualify external corpora for production use.

## Next-stage routing

Proceed to **U3-R — bind the qualified external Retrieval capability into the owned universe**. The active multi-agent capability record qualifies only the closed `PUBLIC_HEALTH` card catalog; `GUIDELINE` and `LITERATURE` remain unqualified. Existing RAG final evaluation reused a public benchmark, so U3-R should establish integration evidence on the project-owned universe and must not be reported as a fresh untouched-benchmark result.

Memory is not frozen for a combined U3-C stage: MEM-3A remains halted at its extraction gate, while MEM-2D is limited to a small retrieval diagnostic. U3-C's two-capability precondition is therefore unmet. No training stage is routed from U2-F.
