# H1-A Adaptive Recruitment Runtime Diagnostics

## Finding

The 219 historical DiagnosisArena B2 `invalid_team_recruitment` cases were not all malformed recruiter generations. The request trace showed that the orchestrator selected a structured JSON schema, but Harness's `_BudgetedModelProvider` rebuilt `ModelRequest` after reducing the 180-second child timeout to the 120-second request deadline and failed to copy `json_schema`. The provider therefore sent an unconstrained JSON-object request to vLLM. One affected question returned a valid answer object (`answer`/`explanation`) where the orchestrator required a `teams` object.

This was an execution-boundary bug. It must not be counted as evidence about Adaptive MDT's clinical reasoning quality.

## Fixes

- Preserve `json_schema` whenever the Harness budget layer caps a request.
- Send strict JSON Schemas for intermediate specialist recruitment and advanced team recruitment. The schemas bound the team/specialist counts and require the fields that the parser consumes. Prompts and downstream MDT collaboration remain the same.
- Keep raw complexity and recruitment text, hashes, validation errors, and fallback reasons behind the explicit `--capture-adaptive-diagnostics` flag.
- Strip Qwen `<think>...</think>` blocks before parsing the final complexity label.
- Set a 900,000 ms request-level Harness deadline in the new system-evaluation model config. The old 120-second default was too close to the observed full advanced-route smoke latency. This is an evaluation-only budget; product defaults are unchanged.
- Configure vLLM structured output with `disable_any_whitespace=true`. On the first 219-ID recruiter-stage replay, the default grammar allowed greedy decoding to spend the remaining 320-token budget on whitespace after a complete teams array but before the outer JSON object's closing brace. The resulting output was semantically populated but invalid JSON. Compact structured output removes that whitespace path; the installed vLLM config exposes this flag, and the final runtime manifest must record it. See [vLLM StructuredOutputsConfig](https://docs.vllm.ai/en/stable/api/vllm/config/structured_outputs/).

The compact replay used Qwen/Qwen3-8B at local model revision `b968826d9c46dd6066d109eabc6255188de91218` (model manifest SHA-256 `e5466c735d57bd3e32d4607a3e372b1862579edfef7864ca514e709a38853e26`), vLLM `0.30.1rc1.dev622+gf03026a54`, BF16, seed `20261004`, temperature 0, top-p 1, thinking enabled, and the 320-token recruiter cap. Exact model and runtime config snapshots are retained in the compact replay directory; the actual xgrammar startup settings and server log hash are in its manifest.

## Verification

Offline checks passed:

- The budget-recap regression test verifies that structured schemas survive Harness request reconstruction.
- Provider tests verify temperature 0, top-p 1, the thinking template flag, and JSON Schema response format.
- The advanced MDT adapter test verifies the schema crosses the orchestrator/provider boundary.
- The one-case DiagnosisArena B2 smoke ran with thinking enabled and 2,048-token main-call cap. The advanced route made 12 calls; the recruiter returned two parseable teams with six specialists; no recruiter/runtime failure occurred. Its correctness score is not used as an experiment result.
- A separate constrained-recruitment replay of one previously invalid ID under compact JSON returned two teams/six specialists in 159 tokens with `finish_reason=stop`.

The first smoke, before fixing the Harness budget copy, made three calls and reproduced `invalid_team_recruitment`; its recruiter returned an answer/explanation object. The corrected smoke completed the full MDT route. Both traces are retained separately:

- [Pre-fix smoke](../../runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/schema-smoke-case/)
- [Post-fix smoke](../../runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/schema-smoke-case-budgetfix/)

## Frozen 219-ID replay

The selection manifest is [team-recruitment-failures.json](../../runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/team-recruitment-failures.json). It selects IDs only from trace failure codes and never reads gold or correctness.

The replay is intentionally scoped to the failed `multi_team_recruitment` stage: it uses each frozen candidate question, the same advanced recruiter prompt and schema, and the Harness budget wrapper. Classifier and downstream specialist/moderator stages are skipped because they do not diagnose this request-copy bug. The replay stores each raw recruiter response and validation outcome immediately in JSONL and supports resume. It is diagnostic only; it does not score or change the historical B0/B2 artifacts.

The initial default-whitespace replay is [recruitment-replay-20261005](../../runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/recruitment-replay-20261005/). It completed all 219 IDs with 200 valid and 19 incomplete outer JSON objects; all 19 reached the 320-token cap after emitting the team array and whitespace. It preserves the original raw outputs. The compact-JSON one-case artifact is [compact-schema-smoke-diagnosisarena16.json](../../runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/compact-schema-smoke-diagnosisarena16.json). The second full pass at [recruitment-replay-compact-20261005](../../runs/common_eval/harness-v1-base-20261005/adaptive-failure-audit-20261005/recruitment-replay-compact-20261005/) completed **219/219 valid team structures**, zero invalid outputs, and zero provider errors. All finished with `finish_reason=stop`; every response had two teams, with 215 outputs containing six specialists, three containing four, and one containing five. Mean recruiter output was 157.6 tokens and mean latency was 3.48 seconds. This verifies the constrained recruitment request path only; it is not an accuracy result or a full Adaptive rerun. The replay ID sequence SHA-256 is `d3e4ab8452ca88ea6ded5ebd9aa627dcd6485b6e50ef290fc70141f493a70f75`; candidate-view SHA-256 is `6bff8dab6683b4bf8954f45d1bba27eec6e75d9fba5f0cb5f233fe7f0f9b53dd`.

JSONL hashes: default-whitespace replay `d9fad119ad338f9f409350dd1a53004d60c07fbed56e080b62ff2a01e2f7e75d`; compact replay `dd814bd4b74050409b6cdb3059b917c82183eda1430e0c3640cc40bf18675c20`. The compact run's input model-config SHA-256 is `ec312b88fba9c9a6f6f9d2bbecfbc11535536e845f0e76d5812529c8f5c20d30`; its original runtime-manifest input SHA-256 is `f2708d33688473a023c87b2021cd9d7379beafae6c11e21ff89e740bc7c21cc0`. The corrected effective runtime config explicitly records xgrammar plus compact JSON, SHA-256 `54667fee0cefe65ea7ccbadbed6ca0e9a9f4ca5266a89b4986ee6462d9218491`; startup log is archived byte-for-byte as `vllm-debug-replay-compact.log.gz` (gzip, no timestamp); the manifest records the archive SHA-256 and the decompressed original log SHA-256 `daff7ad5f018e3fedb873b7a972d8fdce1fa5282fd2efdb29cefcbf8055f8229`. The default-whitespace startup log is archived alongside it; original uncompressed logs remain in the working directory.

## Evaluation status

Historical parser-v7 B0/B2 scores remain development evidence. Those runs used thinking disabled, a 512-token cap, and incompletely matched vLLM settings. The debug smokes and recruiter replay do not replace them. The parser-v7 gold-blind audit found no explicit-choice parser-miss candidates in the historical B0/B2 checkpoints; parser v7 is frozen for the next factorial, and the same audit must run on all new arms before scores are frozen.

H1-C has qualified `COMMON_MEDICAL_KB_V1` for its documented narrow 21-CDC/5-WHO hypertension patient-education scope. Its corpus/index/provider identities and usage restrictions are in [the qualification record](common_medical_kb_v1_qualification.md). This makes B1/B3 eligible only for that declared corpus; it does not establish broad medical coverage or a retrieval accuracy gain. No final B0/B1/B2/B3 factorial has started yet. H1-D must capture the actual shared vLLM runtime first.

The debug vLLM instances used loopback port 8001 and are stopped. Port 8000 belongs to the separate Memory run and remains running and untouched.
