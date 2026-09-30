# RAG-E5-B4 — Harness-Native Execution Decomposition

## Frozen question

With the same frozen 202607 longitudinal tasks and the same B2 retrieval artifacts, does adding external guideline evidence improve guideline-content quality after longitudinal state is moved out of the language-model output contract and materialized deterministically by the harness?

This is a new B4 protocol, not a B2/B3 correction. B2 and B3 artifacts remain immutable. The only experimental actions are `OFF`, `STANDARD`, and `STRONG`; the latter two consume their exact frozen B2 top-five chunks. B4 performs no new retrieval or bridge generation.

## Responsibility boundary

- The runtime copies only the metric explicitly requested by a state-bearing question from `LongitudinalStatePacket v4` into a deterministic `StateClaim`.
- T0 final responses are materialized directly and make zero model calls.
- For T1 and T2, the model receives the unchanged task question and, depending on action, no passage, the exact B2 STANDARD top five, or the exact B2 STRONG top five.
- The guidance model writes plain text only. It does not serialize state, select canonical state fields, resolve provenance IDs, or produce JSON.
- Evidence aliases are assigned by frozen rank order (`[E1]` through `[E5]`). The runtime resolves only issued aliases back to their exact B2 chunk IDs and source/section provenance. Unknown aliases are ignored and counted.
- The scoring function receives no action label. It scores deterministic state claims, plain-text guidance anchors, supplied chunks, and resolved citations. Teacher labels are opened only after the full 180-arm and 120-call execution manifest has been verified and committed.

## Fixed inference configuration

Qwen3-8B Q4_K_M is pinned to the B2 model SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`. All T1/T2 action arms make one call with temperature 0, seed 0, reasoning disabled through `chat_template_kwargs.enable_thinking=false`, no prompt cache, no retry, and an 8192-token completion ceiling. There is no schema, function call, format-repair, or output-budget ladder.

The experiment contract is 32768 context tokens. The already-running shared loopback server is configured for a 131072-token capacity and is not stopped or reconfigured, to avoid disrupting the other Codex task. Before any generation, B4 uses that server's tokenizer to verify every frozen prompt plus the 8192 completion ceiling and chat-template reserve fit within the 32768 contract. The larger server capacity is recorded as an execution-environment fact, not treated as a larger B4 input budget. The server binary/model hashes, PID, build, GPU offload, Flash Attention, and GPU snapshot are recorded in the protocol lock.

## Execution and scoring order

1. Commit code, tests, prompt, and protocol documentation.
2. Freeze `runs/rag_e5/e5b4_protocol_lock.json` on `codex/rag-e5-b4-decomposed-execution-20260930` from `origin/main@2e2d62ce93a84378a1cad314994a097a9007a6fb`.
3. Use at most three path-only smoke calls (OFF/STANDARD/STRONG); they are isolated from the 120 guidance calls and never scored.
4. Run the sorted 180-arm counterfactual: 60 state-only materializations and 120 single-call guidance generations. Finish all calls and freeze hashes before scoring.
5. Commit/push the execution manifest. Only then may the evaluator open `teacher_cases.jsonl` and score.

The B2 reuse lock, 180-arm artifact-set SHA, state/task/corpus identities, and source ranking/chunk hashes must still match. The only input cohort is `202607`; 202608 remains unopened. Any length termination at 8192, empty response, missing usage, or transport failure is reported; no retries or follow-up budget changes are permitted.

## Metrics and decision gates

Existing deterministic anchor scoring is reused without an LLM judge:

```text
T0 E2E = state score
T1 E2E = 0.75 × guideline-content + 0.25 × grounding
T2 E2E = 0.50 × deterministic state + 0.25 × guideline-content + 0.25 × grounding
```

Guidance quality is the anchor score over T1/T2. Report all fixed-action metrics, per-family metrics, calls/tokens/latency, unknown citation aliases, and the frozen B2 required-source/recommendation @5 coverage snapshot. Also report per-case three-action and OFF|STANDARD two-action oracles, tie-broken oracle counts, action diversity, oracle headroom, STRONG unique wins/collapse, and 10,000 paired bootstrap resamples stratified by user (each user's T0/T1/T2 remain together).

The evaluator reports:

```text
MEASUREMENT_HEALTH
EXTERNAL_RETRIEVAL_CONDITIONAL_VALUE
STANDARD_VALUE
STRONG_INCREMENTAL_VALUE
RECOMMENDED_ACTION_SPACE
POST_TRAINING_DATA_WORTH_BUILDING
```

Positive action value requires a positive fixed-action E2E delta with the paired user-bootstrap 95% CI lower bound above zero. Three-action policy requires the frozen diversity gate and at least 0.05 oracle headroom. The two-action candidate requires meaningful OFF and STANDARD oracle representation while STRONG has fewer than three unique oracle wins and no positive fixed mean increment. B4 does not train a router. If counterfactual data has signal, the next scoped stage is E5-C0 Training-View Transfer; otherwise stop this RAG integration line.
