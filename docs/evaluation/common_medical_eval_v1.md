# Common Medical Evaluation V1

## Evaluation contract

All system arms use `HealthCopilotHarness.execute(profile, HarnessRequest)`. Gold remains in `EvalCase` and the dataset adapter. The runner creates a request from case ID, question, and answer schema only. Deterministic choice scoring counts parse failures as incorrect and does not call an LLM judge.

The alias registry in `configs/eval/profile_registry.json` defines B0-B3, P0-P3, and R0-R3 as orthogonal model/retrieval/memory/reasoning axes. Profiles do not implement alias-specific runtime behavior. Common medical MCQ profiles set Memory OFF. The product API uses the single `product-adaptive-v1` profile; static Single remains only as a paired evaluation control. See [the pruning decision](../architecture/architecture_pruning_decision.md) for the evidence behind this cut.

## Model capability and system capability are separate tracks

Use the same frozen DiagnosisArena-915 and CMB-COMMON-1024 IDs, gold source, parser, and scorer in both tracks, but do not compare their absolute scores as if the runtimes were identical.

| Track | Question | Runtime | Required paired controls |
|---|---|---|---|
| Model capability | Did medical post-training improve Qwen3 itself? | PT-E0 standalone evaluator; no Harness safety, RAG, or MDT | Base / SFT / GSPO share one prompt, thinking mode, token cap, parser, scorer, model-serving runtime, and frozen IDs. |
| System capability | What do RAG and Adaptive MDT add to one model? | Health-Copilot Harness; B0-B3 factorial | All four arms share one base checkpoint, thinking mode, token cap, parser, scorer, vLLM process/config, and frozen IDs. |

PT-E0's `66.70%` CMB base result and the old Harness B0 `62.30%` are from different tracks and are not a model-training delta. The old B0/B2 runs remain development evidence only: Harness thinking was disabled, the output cap was 512, and B0/B2 runtime capture was not fully matched. The final system run must use [`qwen3_8b_base_system_eval.json`](../../configs/models/qwen3_8b_base_system_eval.json), with Qwen thinking explicitly enabled and a 2,048-token provider cap to match the PT-E0 generation protocol. It also freezes a 900-second per-request Harness deadline and xgrammar compact JSON for structured recruitment calls. The run manifest must record the actual vLLM runtime, and all B0-B3 cases must use the same live process/config. Adaptive's per-stage output caps remain method configuration and are recorded with the reasoning implementation hash.

Primary end-to-end accuracy keeps every frozen ID in the denominator; safety routes, parser failures, and runtime failures are not dropped. Report them as separate error categories alongside parse success. Freeze parser v7 across model and system tracks. A parser change requires offline rescoring of every existing arm before any new model run.

## Intended public evaluation sets

| Set | Intended size | Score | H0 state |
|---|---:|---|---|
| DiagnosisArena MCQ | 915 | Exact single-choice accuracy | IDs and candidate/scorer view hashes frozen; evaluation ready |
| CMB-Exam full | 11,200 across 28 subcategories | Exact answer-set accuracy and category macro accuracy | Not scored as a full set; the earlier 352-row partial remains preserved |
| CMB-COMMON-1024 | 1,024 stratified public cases | Exact answer-set accuracy and category macro accuracy | IDs and candidate/scorer view hashes frozen; evaluation ready |

These are `PUBLIC_EXTERNAL_EVAL`, not blind or untouched tests. The frozen IDs were established in PT-E0 before any SFT/GSPO optimizer run. Exact source revisions, ID hashes, candidate-view hashes, and scorer-view hashes are recorded in `configs/eval/common_eval_v1.json` and `configs/eval/local_prepared_views_h0.json`.

## Deterministic subset protocol

Use the supplied canonical CMB ID manifest without re-sampling. It records source revision `935fbc09edf1303d89872b21265ff597f426ac0d`, seed `20261004`, proportional largest-remainder quotas, and deterministic per-category SHA-256 selection. The 1,024-ID sequence hash is `ed5816263ee3f20dec4fd6f76f5b8b6ceb6f89a4b01b35716a2b8afc4ffd4fc4`; the ID-manifest hash is `8bb899d3b1577a606915535749e9455d6e5bc53aed27d7057cf2007918c37397`. The exact manifest is copied to `configs/eval/cmb_common1024_ids.json`. DiagnosisArena uses the frozen 915-ID list at `configs/eval/diagnosisarena915_ids.json`.

Candidate and scorer views are separate. `PreparedViewsAdapter` joins them by ID in evaluator memory; only candidate prompts and answer schemas become `HarnessRequest` fields. Labels are used for `CaseScore` and are never serialized to the model or run checkpoint.

## Runner and checkpointing

```bash
VLLM_BASE_URL=http://127.0.0.1:8001/v1 \
VLLM_MODEL_NAME=qwen3-8b-system-v1 \
uv run python tools/eval/run_common_eval.py \
  --dataset diagnosisarena --profile B0 \
  --model-config configs/models/qwen3_8b_base_system_eval.json \
  --vllm-runtime-config runs/common_eval/<frozen-system-v1-runtime>.json \
  --prepared-config configs/eval/local_prepared_views_h0.json \
  --output runs/common_eval/<run-id>/diagnosisarena/B0 --resume --concurrency 4
```

Start the matching loopback vLLM service before using this example. Use a free port and a runtime-config file that describes the actual server; do not assume or reconfigure a port owned by another task.

Every result is appended and fsynced to `cases.jsonl` as soon as the case finishes. Resume validates dataset/profile/model identity, source hashes, subset identity, and every completed case ID. Independent cases may run concurrently; calls inside a case keep their required order. The record contains response and deterministic score, never gold. Traces are metadata-only JSONL. Summary metrics include exact accuracy, answer parse rate, safety-route abstentions, reasoning failures, model abstentions, answer-format failures, category macro accuracy, provider calls, tokens, mean/P50/P95 latency, throughput, retrieval and MDT metrics where available. Parser v7 recognizes schema-named JSON fields, unambiguous leading answer fields in malformed-rationale JSON, Markdown emphasis, explicit English/Chinese final-choice labels, and repeated labels in explicit multi-select answer lines that include option text. Parser v6 remains frozen for historical comparison; its silent multi-select truncation is documented in the v7 audit below.

The runner rejects missing or mismatched dataset hashes, incomplete model identity, unready Common KB for RAG profiles, and memory-enabled profiles without a bound provider. H1-C has now qualified [`COMMON_MEDICAL_KB_V1`](common_medical_kb_v1_qualification.md): a scope-limited set of 21 CDC and 5 WHO hypertension patient-education cards. Its local factory returns a `RetrievalProvider` only after validating source, model, vector, index, and implementation hashes. B1/B3 are now eligible for the matched factorial, with the corpus coverage limitation reported alongside results.

For the remote Qwen3-8B factorial ablation, the same entry point can run all four B profiles and emit paired deltas and the RAG×Adaptive interaction once Common KB V1 is ready:

```powershell
uv run --project . python tools/eval/run_common_eval.py `
  --dataset cmb-common --matrix core `
  --model-config configs/models/qwen3_8b_base_system_eval.json `
  --output runs/common_eval/<run-id> --resume
```

The matrix maps B0/B1/B2/B3 to Single, Single+RAG, AdaptiveMDT, and AdaptiveMDT+RAG. Each arm has its own checkpoint and trace directory under the output root. `matrix_summary.json` contains descriptive paired accuracy deltas and verifies B1/B3 retrieved-evidence hashes match case by case; per-arm summaries retain call, token, latency, retrieval, and adaptive-routing metrics. Memory stays outside this matrix. All dataset, model, and retrieval gates are checked before the first generation call.

## Common Medical KB gate

`COMMON_KB_V1_READY=YES` for the qualified, narrow CDC/WHO card scope. B1/B3, P1/P3, and R1/R3 can use only the bound provider and index in `configs/eval/common_medical_kb_v1.json`; each run manifest and resume check binds the exact corpus/index identity. The source and coverage limits are documented in [`common_medical_kb_v1_qualification.md`](common_medical_kb_v1_qualification.md). The existence of a READY config is not an accuracy claim.

## H0 execution record

The Base model is `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`, with local model-manifest SHA-256 `e5466c735d57bd3e32d4607a3e372b1862579edfef7864ca514e709a38853e26`. Historical B0 used `temperature=0`, Qwen thinking disabled, and 512 output tokens. Under deterministic parser v6, DiagnosisArena-915 B0 is 339/915 (37.05%) with 915/915 parseable answers and zero answer-format failures. CMB-COMMON-1024 B0 is 638/1,024 (62.30%); 969/1,024 answers produce a choice, the remaining 55 are explicit safety-route abstentions, and answer-format failures are zero. Parser-v2 through v5 artifacts remain separate; parser-v6 copies were rescored from raw model outputs, leaving the original raw B0 runs unchanged. These are historical development results, not the final system factorial baseline.

Serving parity needs a caveat. The B0 run manifests captured the model config but no per-run vLLM runtime config or server process ID. A host vLLM log whose timestamp overlaps B0 records the same Qwen3-8B BF16 weights and vLLM build, but observed `gpu_memory_utilization=0.88`, 16,384 context, automatic KV cache, and prefix caching enabled. The B2 transport-recovery runs used port 8001, 32,768 context, FP8 KV cache, and prefix caching disabled. CMB targeted recovery used the `.46` reservation; DiagnosisArena resumed from `.46`/8 sequences to `.88`/16 sequences after capacity-wait metrics showed unused KV cache. Both arms use temperature 0 and thinking disabled, but exact server parity is unproven and the KV-cache configuration differs. Treat B0/B2 as descriptive same-backbone results, not a controlled causal estimate of reasoning architecture. Details and the source-log hash are in `recovery_inputs/serving-parity-audit.json`; each execution segment is recorded in its run manifest.

The original B2 checkpoints remain marked `INVALID_INFRASTRUCTURE_FAILURE`; their raw rows, traces, and invalidation manifests were not changed. The invalidated CMB source had 671 connection failures; the targeted recovery selected 674 transport failures (671 API connection errors, 2 API timeouts, 1 timeout). The invalidated DiagnosisArena source selected 791 transport failures (788 API connection errors, 2 API timeouts, 1 timeout), leaving its other 124 source rows untouched; these include 37 non-transport reasoning failures. The first DiagnosisArena retry still contained three connection failures (`diagnosisarena:428`, `:602`, `:7`), so those three alone were rerun into a separate repair run and overlaid without editing the original retry checkpoint. All selected IDs now merge into complete cohorts of 1,024 and 915 rows. The full-cohort B2 copies were first rescored from raw outputs with parser v6, then score-only rescored with parser v7; raw model outputs and v6 files remain unchanged.

### Paired B0/B2 results (descriptive)

| Dataset | B0 Single | B2 Adaptive MDT | Delta | B2 parseable | B2 reasoning failures | B2 answer-format failures |
|---|---:|---:|---:|---:|---:|---:|
| CMB-COMMON-1024 | 638/1,024 (62.30%) | 695/1,024 (67.87%) | +5.57 pp | 960/1,024 | 9 | 0 |
| DiagnosisArena-915 | 339/915 (37.05%) | 280/915 (30.60%) | −6.45 pp | 685/915 | 230 | 0 |

The datasets disagree on the Adaptive effect. Under parser v6, CMB has 107 cases correct only under B2 and 50 correct only under B0; parser v7 changes these to 108 and 47. DiagnosisArena has 103 cases correct only under B2 and 162 correct only under B0 under both versions. The DiagnosisArena shortfall is not caused by the final MCQ answer parser: 230 cases ended in internal Adaptive reasoning failures, while the remaining B0-only losses have explicit but incorrect final choices. MDT routes were 279 advanced, 502 intermediate, 124 basic, and 10 failed. CMB mostly routed to basic (890), with 77 intermediate, 1 advanced, and 1 failed. The original paired reports in `runs/common_eval/harness-v1-base-20261005/paired/` preserve v6 scores; corrected v7 paired reports are under `runs/common_eval/harness-v1-base-20261005/rescored-parser-v7/paired/`.

### Parser-v7 score-only follow-up

Parser v6 recorded zero answer-format failures, but that metric missed four silent B2 multi-select truncations where the model’s explicit option set matched the scorer. Parser v7 rescored stored `answer_text` only; no inference or retrieval calls were made. It changes six parsed B2 option sets in CMB, of which four become correct. Three corrected cases had been B0-correct/B2-wrong under v6; the fourth had been wrong in both arms. B0 is unchanged.

| Dataset | B0 parser v7 | B2 parser v7 | Delta | B0-only correct | B2-only correct |
|---|---:|---:|---:|---:|---:|
| CMB-COMMON-1024 | 638/1,024 (62.30%) | 699/1,024 (68.26%) | +5.96 pp | 47 | 108 |
| DiagnosisArena-915 | 339/915 (37.05%) | 280/915 (30.60%) | −6.45 pp | 162 | 103 |

B0 and B2 both have RAG OFF and Memory OFF; manifests and traces confirm zero retrieval calls. In DiagnosisArena, the 230 B2 failures comprise 219 `invalid_team_recruitment` errors on the advanced route, 10 `invalid_complexity_output` errors, and 1 `insufficient_specialists_recruited` error. Among the 162 B0-only correct cases, 95 are Adaptive fail-closed outputs (89 team recruitment, 5 complexity, 1 specialist count); 67 have a parsed but incorrect final answer. The fallback text is a runtime failure response, not a calibrated clinical abstention. See the [parser-v7 audit](../../runs/common_eval/harness-v1-base-20261005/rescored-parser-v7/report.md) and [audit manifest](../../runs/common_eval/harness-v1-base-20261005/rescored-parser-v7/audit_manifest.json) for case-level changes and hashes.


PT-E0 separately reports CMB-COMMON-1024 at 683/1,024 (66.70%), 95% Wilson CI 63.75%–69.52%. It used an empty system prompt, thinking enabled, and up to 2,048 output tokens; it is a reference result, not a paired B0 score under this Harness prompt/decoding contract. B0 and B2 keep the same frozen Harness prompt. Post-training may affect format adherence, which will be measured on its own profile rather than assumed. Parser v7 reports zero answer-format failures and also corrects the identified silent set truncations. The 352-row full-CMB partial remains unscored as a full benchmark; 35 rows are reused within the common subset.

The gold-blind parser-failure audit is stored at [`audit.json`](../../runs/common_eval/harness-v1-base-20261005/parser-audit-v7-gold-blind/audit.json). On parser-v7 outputs, no explicit-choice parser-miss candidates were found; the audit read candidate answer-schema metadata, response text/flags, and trace failure metadata, not scorer gold or correctness. Parser v7 is frozen for the next factorial. Audit schema v2 separates invalid classifier output, invalid recruitment output, provider transport failure, and unspecified runtime failure when trace diagnostics allow it. The tool accepts explicit `--run NAME=CASES_JSONL` entries so the same gold-blind check can cover all four new arms before result freeze.

H1-A located and fixed a Harness provider-copy bug that dropped `json_schema` when the per-request deadline reduced the provider timeout. The fix is covered by a budget-layer regression test. A thinking-enabled one-case B2 smoke then completed advanced recruitment and the full 12-call route. The 219 frozen historical recruitment failures were replayed at the recruiter stage only, using candidate data, the same prompt/schema, and the Harness budget wrapper; no gold or score was read. The compact-schema replay produced 219/219 valid team structures. Findings, exact scope, and artifact paths are in [H1-A diagnostics](h1_adaptive_recruitment_diagnostics.md). These debug artifacts do not replace historical scores or count as a new accuracy baseline.

The H1 debug vLLM service was on loopback port 8001 using BF16 Qwen3-8B, model revision `b968826d9c46dd6066d109eabc6255188de91218`, thinking enabled, temperature 0, top-p 1, and 2,048-token serving cap. Its actual debug runtime is recorded at `runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/vllm-debug-replay-runtime-config.json` (8,192 context, one active sequence, FP8 KV cache, `.40` GPU reservation). This was not the final factorial vLLM profile. Port 8000 remains untouched. H1-C qualified the Common KB. The final four-arm run still requires a newly captured shared vLLM runtime and a 20-case infrastructure smoke under that exact runtime.
