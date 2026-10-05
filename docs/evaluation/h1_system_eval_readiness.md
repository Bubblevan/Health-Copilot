# H1 system evaluation readiness and smoke record

Date: 2026-10-05

## Decision

The H1 infrastructure gate passed. Proceed with the frozen full B0/B1/B2/B3 factorial on `DiagnosisArena-915` and `CMB-COMMON-1024`. The 20-case smokes below are infrastructure checks only; their accuracy is excluded from interpretation and method decisions.

The final run uses the pinned Qwen3-8B BF16 server on loopback port 8001, thinking enabled, greedy decoding, parser v7, and case concurrency 2. The separate Memory service on port 8000 was left running and untouched. Every case is checkpointed immediately, and a resumed run must retain the same code, model, IDs, runtime config, retrieval identity, parser, and profile.

## Frozen factorial protocol

| Arm | Model | Retrieval | Memory | Reasoning |
|---|---|---|---|---|
| B0 | Qwen3-8B Base | OFF | OFF | SINGLE |
| B1 | Qwen3-8B Base | STANDARD | OFF | SINGLE |
| B2 | Qwen3-8B Base | OFF | OFF | ADAPTIVE_MDT |
| B3 | Qwen3-8B Base | STANDARD | OFF | ADAPTIVE_MDT |

All arms share the model endpoint, system-eval prompt/runtime, decoding config, parser v7, frozen case IDs, and scorer. RAG is top-level Harness retrieval; B1 and B3 must receive byte-identical evidence for each case. Adaptive MDT does not retrieve independently.

### Dataset identities

| Dataset | Count | Frozen ID manifest SHA-256 | ID-sequence SHA-256 | Candidate-view SHA-256 | Scorer-view SHA-256 |
|---|---:|---|---|---|---|
| CMB-COMMON-1024 | 1,024 | `8bb899d3b1577a606915535749e9455d6e5bc53aed27d7057cf2007918c37397` | `ed5816263ee3f20dec4fd6f76f5b8b6ceb6f89a4b01b35716a2b8afc4ffd4fc4` | `9b90ef5f072aefdffd7850b284cc2a56f3bef5f7bbc055c0510bb3b6f9615119` | `8bd9402e308b631ec5dac99d4f5681b069a2b92d89cda8249c30952192eefdd2` |
| DiagnosisArena-915 | 915 | `b0203d2259b256dfd600241ab5b6d7972ae4cca8137094e0e79af9a008216c1e` | `b6c4102a3d2bdcf3d464b77f67df0e02a9f04a5036d7a6e45e34aedb606967e7` | `6bff8dab6683b4bf8954f45d1bba27eec6e75d9fba5f0cb5f233fe7f0f9b53dd` | `afb68b68ab95b4aa90e2702858cdc4e7a4167afe56dcd1f324f0a5b7dcff7fd1` |

The full evaluation uses the canonical ID manifests above, not the 20-case smoke lists. Gold remains evaluator-only and is never part of `HarnessRequest`.

### Model and serving identity

- Model: `Qwen/Qwen3-8B`, path `/root/gpufree-share/data/Qwen3-8B`
- Model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Model manifest/checkpoint SHA-256: `e5466c735d57bd3e32d4607a3e372b1862579edfef7864ca514e709a38853e26`
- Model-config SHA-256: `54707a14c869ee58bdac2f01dd3024a8b4df3a08aa6ab13ba8b2feb37b3b624f`
- GPU/runtime: NVIDIA L40; BF16; Python 3.11.17; Torch 2.13.0+cu130; CUDA 13.0; vLLM `0.30.1rc1.dev622+gf03026a54`
- Endpoint: `127.0.0.1:8001`; served name `qwen3-8b-system-v1`; seed `20261004`; temperature `0`; top-p `1`; `do_sample=false`
- Thinking: enabled with `chat_template_kwargs.enable_thinking=true`
- Output cap: 2,048 tokens; max model length 16,384; max sequences 16; max batched tokens 16,384; KV dtype auto; prefix caching off
- Structured output: xgrammar with `disable_any_whitespace=true`
- Actual runtime-config SHA-256: `f7da7d58dc530313e64cc59f51d06fc1c001fdb4f555349ce570a398d47d59dc`
- Actual startup record, losslessly archived: [`vllm-8001-startup.log.gz`](../../runs/common_eval/h1-final-factorial-20261005/vllm-8001-startup.log.gz). Its manifest retains the original uncompressed log hash.

The normal compilation/JIT path completed, including kernel warmup; `--enforce-eager` was not used. During this run the L40 was also serving the Memory workload through port 8000, so throughput is constrained by real shared-GPU contention. Port 8000 and its process were not modified.

## Gate evidence

### H1-A: recruiter execution fix

The old `invalid_team_recruitment` failures were traced to the Harness budget wrapper dropping `json_schema` while rebuilding a model request. The wrapper now preserves the schema; intermediate and advanced recruitment use bounded structured schemas. The 219 frozen historical recruiter failures were replayed at the recruiter stage, without gold or scoring: **219/219 valid team structures, zero provider errors**. This establishes recruiter-stage request correctness only; it is not a full Adaptive rerun or an accuracy result. Details are in [H1-A diagnostics](h1_adaptive_recruitment_diagnostics.md).

### H1-B: parser v7 and gold-blind audits

Parser revision remains `deterministic-mcq-parser-v7`; parser code was not changed. Gold-blind v2 audits were run on all four arms in both 20-case smokes:

- CMB: [`parser-audit-cmb-smoke-all-arms-v2.json`](../../runs/common_eval/h1-final-factorial-20261005/parser-audit-cmb-smoke-all-arms-v2.json)
- DiagnosisArena: [`parser-audit-diagnosisarena-smoke-all-arms-v2.json`](../../runs/common_eval/h1-final-factorial-20261005/parser-audit-diagnosisarena-smoke-all-arms-v2.json)

Both audits report `NO_PARSER_FALSE_NEGATIVE_CANDIDATES_FOUND`, no parser mismatches against the stored parsed answer, and `gold_accessed=false`. Unparsed Adaptive outputs are classified separately as invalid classifier output; one unparsed Single answer in each dataset never commits to a valid option. These remain model/system failures in the full-run denominator. The audit does not auto-correct answers.

### H1-C: Common KB V1

`COMMON_MEDICAL_KB_V1_READY=YES` for a deliberately narrow set of 21 CDC and 5 WHO hypertension patient-education cards. Frozen corpus SHA-256 is `399c528c7784b1a722e20c181d6889f2c4a043d56240bf6e5dd1d4824da5f4e0`; index SHA-256 is `e9ad9ff82073f63867487c118602d5da1e7c4f156dec6ea31d4a3005b8f3834c`. The candidate-view overlap audit found no exact or high five-gram overlaps for either evaluation set. This corpus qualification does not imply general medical coverage or a retrieval quality claim. See [Common KB V1 qualification](common_medical_kb_v1_qualification.md).

### H1-D: actual serving process

The actual process configuration is recorded in [`vllm-runtime-config.json`](../../runs/common_eval/h1-final-factorial-20261005/vllm-runtime-config.json). It matches the model-config decoding fields and was used by all smoke arms. GPU memory utilization was set to 0.50 for the 8001 process while the 8000 Memory service remained untouched.

## Smoke outcomes (infrastructure only)

All eight arm manifests are `COMPLETE`, each with the same 20 selected case IDs within its dataset. The CMB smoke used code SHA `9800e5c679ad5de3bad0e31d70c73a6e6c94ce7b`; the DiagnosisArena smoke used `dfced087414b2254e6c3d5886bc3db3e75308314`. The latter commit only refined the audit tool's failure taxonomy after the CMB smoke; evaluator, Harness, model, retrieval, and reasoning implementation identities are captured in each run manifest.

| Dataset | Arm | Parser success | Gold-blind unparsed classification |
|---|---|---:|---|
| CMB | B0 | 20/20 | none |
| CMB | B1 | 19/20 | 1 model did not commit to a valid option |
| CMB | B2 | 11/20 | 8 invalid Adaptive classifier outputs; 1 model did not commit |
| CMB | B3 | 16/20 | 4 invalid Adaptive classifier outputs |
| DiagnosisArena | B0 | 20/20 | none |
| DiagnosisArena | B1 | 19/20 | 1 model did not commit to a valid option |
| DiagnosisArena | B2 | 7/20 | 13 invalid Adaptive classifier outputs |
| DiagnosisArena | B3 | 9/20 | 11 invalid Adaptive classifier outputs |

The invalid classifier outputs are model/method behavior returned by the preserved complexity-classification prompt and rejected by its output validator. They are not parser false negatives, schema transport loss, or recruiter implementation errors. The implementation fails closed and preserves them as Adaptive failures; this H1 stage does not tune prompts or add a Single fallback.

RAG evidence parity passed for **20/20 cases** in both smoke datasets: B1 and B3 had no missing or mismatched evidence hashes. Per-dataset matrix summaries are retained next to the smoke arms.

The first CMB smoke attempt is retained but explicitly invalidated because B1 failed before its first provider call with a `_RuntimeContext` constructor `TypeError`. The integration was fixed in commit `9800e5c679ad5de3bad0e31d70c73a6e6c94ce7b`; the invalidation record points to the valid `smoke20-r1` replacement. Never resume or combine the first attempt.

Smoke outputs and parser audits do not establish benchmark accuracy and were not used to tune method behavior.

## Full-run execution record

Full runs are launched after this readiness record is committed, on that immutable code SHA. Use `--dataset cmb-common` for the 1,024-case official CMB subset and `--dataset diagnosisarena` for the 915 official cases; omit `--case-ids-file` so the adapters load their canonical manifests. Each dataset uses `--matrix core`, the same frozen model/runtime/prepared-view configs, `--concurrency 2`, and `--capture-adaptive-diagnostics`. Output roots:

```text
runs/common_eval/h1-final-factorial-20261005/full/cmb-common-1024/
runs/common_eval/h1-final-factorial-20261005/full/diagnosisarena-915/
```

Every case is checkpointed to JSONL. If interrupted, resume only with the same code SHA, model/config hashes, dataset identities, retrieval identity, and profile, using the runner's `--resume` option. Keep stdout/stderr in a sibling `*-console.log` file. After both factorials complete, run the same gold-blind parser v2 audit across all eight full arms before reporting any scores or paired deltas; report parse failures, classifier failures, transport/runtime failures, safety abstentions, calls, tokens, latency, and accuracy separately.
