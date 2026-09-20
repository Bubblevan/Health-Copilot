# M4 — Budgeted and Replayable Harness Runtime

M4 adds a narrow execution-control plane without changing the M0–M3 product contracts, tools, or frozen evaluation gold.

## Boundaries and budgets

All live model adapters call `ProviderExecutor`: generator, Agent, EvidencePolicy, M2 grounding verifier and M3 claim-support verifier. `RunContext` owns a monotonic `RunBudgetState`. Before a provider request or a tool dispatch it checks the deadline, provider-call cap, tool-execution cap, and, when configured, the hard total-token cap. Missing usage under a hard token cap makes later provider calls fail closed. `LiveToolRunner` guards before registry dispatch, so a denied tool never reaches its handler.

## Trace and privacy

`RunTrace` is append-only JSONL. Default `metadata_only` traces reject question, answer, claim, evidence, provider-message and tool-query fields. They keep control-plane identifiers, request fingerprints, budget denials and disposition. `public_eval_content` is an explicit opt-in intended only for reviewed public evaluation artifacts; provider/tool exchanges for replay are stored separately from production metadata traces.

## Replay and failure injection

`RecordingProviderExecutor` and `RecordingToolRunner` capture successful public-evaluation exchanges. `ReplayProviderExecutor` verifies a canonical SHA-256 fingerprint over call kind, model, messages, tools, response format, temperature and output limit, then returns the recorded response. `ReplayToolRunner` verifies tool name and arguments and never holds a live registry. A mismatch fails closed; replay reruns normal control flow rather than reading a saved final response.

The deliberately small offline `FailureInjectionPlan` can fail a specified provider kind or tool call before delegation. It supports counterfactual and fail-closed mechanics tests; it is not a generic hook or plugin framework.

## M4 public diagnostic

`evals/m4_replay.jsonl` is a six-case fixed subset copied from already reviewed `m1_recovery.jsonl` (two recovery paraphrases, two direct-hit controls, two OOD controls). It is a regression diagnostic, not a new benchmark or a generalization estimate. Run `tools/run_m4_replay_eval.py --mode record` with live environment configuration, then run the same command with `--mode replay --run-dir runs/m4/<timestamp>`. The replay command creates no live provider client and executes no live tool.
