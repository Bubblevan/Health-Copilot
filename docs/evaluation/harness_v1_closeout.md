# H0 closeout status

These flags describe the current checkout only. `NO` means the required qualification evidence is absent; it does not mean the underlying subsystem failed.

```text
HARNESS_V1_SINGLE_ENTRYPOINT=YES
HARNESS_V1_SINGLE_REQUEST_CONTRACT=YES
HARNESS_V1_SINGLE_RESPONSE_CONTRACT=YES
HARNESS_V1_SINGLE_TRACE_SCHEMA=YES
HARNESS_V1_SINGLE_BUDGET_OWNER=YES

RETRIEVAL_PROVIDER_ADAPTER=YES
MEMORY_PROVIDER_ADAPTER=YES
SINGLE_REASONER_ADAPTER=YES
ADAPTIVE_MDT_REASONER_ADAPTER=YES

COMMON_EVAL_DIAGNOSISARENA_915_FROZEN=YES
COMMON_EVAL_CMB_1024_FROZEN=YES
COMMON_EVAL_GOLD_LEAK_TO_RUNTIME=NO
POST_TRAIN_CONTAMINATION_AUDIT_COMPLETE=NO
COMMON_KB_V1_READY=NO

B0_RUN_STARTED=YES
B0_RUN_COMPLETE=YES
B2_RUN_STARTED=YES
B2_TARGETED_RECOVERY_COMPLETE=YES
B2_COMPOSITES_RESCORED_WITH_PARSER_V6=YES
PARSER_V7_OFFLINE_RESCORING=YES
SFT_STARTED_BY_H0=NO
GSPO_STARTED_BY_H0=NO
```

The PT-E0 owner supplied frozen public evaluation IDs and separate candidate/scorer views. Their hashes and source revisions are recorded in `configs/eval/common_eval_v1.json`; the exact ID manifests are preserved under `configs/eval/`. Candidate IDs match both views exactly. The public evaluation sets are frozen; the Common Medical KB and contamination audit remain unqualified.

The Qwen3-8B base identity is frozen. The initial parser-v6 results remain preserved. A follow-up parser-v7 audit rescored stored outputs without model calls: CMB-COMMON-1024 B0 remains 638/1,024 (62.30%), while B2 corrects from 695 to 699/1,024 (68.26%), delta +5.96 pp. DiagnosisArena-915 is unchanged: B0 339/915 (37.05%), B2 280/915 (30.60%), delta −6.45 pp. Parser v6 silently truncated four explicit CMB B2 multi-select sets; three were B0-correct/B2-wrong cases. The final-answer parser does not explain the DiagnosisArena losses: 95 B0-correct cases ended in internal MDT recruitment/complexity validation failures, and 67 contain explicit but incorrect final choices. Details and immutable scoring artifacts are in `runs/common_eval/harness-v1-base-20261005/rescored-parser-v7/`.

The original B2 runs remain `INVALID_INFRASTRUCTURE_FAILURE` and are immutable. Recovery selected only their transport-failed IDs: 674 CMB and 791 DiagnosisArena. Three residual DiagnosisArena transport failures were repaired in a separate three-case run and incorporated through a provenance-recorded overlay; no other cases were rerun. The composite copies contain exactly 1,024 and 915 unique frozen IDs and remain marked `TARGETED_TRANSPORT_RECOVERY_SCORED`; parser-v6 and parser-v7 score-only copies are separate. Serving parity remains incomplete because the historical B0 runtime config was not captured; report the deltas as descriptive rather than causal. B1/B3 remain blocked by `COMMON_MEDICAL_KB_V1`. No SFT or GSPO run was started by H0.

Evaluation machine: NVIDIA L40 (46,068 MiB). The local prepared views and scorer files remain on the PT-E0 data disk; ID manifests, hashes, and results are preserved in this checkout. The isolated loopback port-8001 server successfully completed normal `torch.compile`, CUDA graph capture, and JIT warmup using the shared post-training venv; it was stopped after recovery, and GPU usage returned to zero. Port 8000 was not stopped or reconfigured. CMB full-test scoring remains out of scope; its earlier 352-row partial and the 35 reused rows were left untouched. Common KB qualification and post-training contamination audit remain open.

Final repository validation: `pytest` reports 1,015 passed and 2 skipped; `ruff check src tests tools/eval`, `compileall -q src tests tools/eval`, Health-Copilot import, and `git diff --check` pass. Because this worktree has no project `.venv` or `uv` executable on PATH, temporary pytest/Ruff/tokenizer packages were placed under `/tmp`; the shared post-training venv was not modified for test tooling.
