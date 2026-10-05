# Health-Copilot Harness V1

## Purpose and status

Harness V1 is the single request, context, execution, budget, trace, and response boundary for product requests and common system evaluations. The Base Qwen3-8B profile and the prepared DiagnosisArena-915/CMB-COMMON-1024 evaluator views are frozen for this local run; Common Medical KB V1 is not qualified.

```text
Product API ─┐
             ├─ HarnessRequest → Safety → Memory/Retrieval → ReasoningStrategy → Verification
Eval runner ─┘                                                │
                           HarnessResponse ← Trace/Budget/Provenance
```

The Harness owns safety routing, memory reads, one top-level retrieval, model identity selection, budgets, provenance checks, answer parsing, and trace/metrics. Reasoners receive only a `ReasoningContext`; they cannot run a second retrieval or memory pipeline.

## Contracts

- `harness/contracts.py`: runtime-visible request and response types. Evaluator gold is not a request field.
- `harness/profiles.py`: independent model, retrieval, memory, and reasoning axes. Common Eval aliases such as B0/B3 are defined only in `configs/eval/profile_registry.json`.
- `providers/model.py`: one async ModelProvider boundary; vLLM uses an OpenAI-compatible endpoint with retries disabled. llama.cpp remains a local legacy adapter.
- `providers/retrieval.py`: `FrozenMedicalRAGProvider` requires a content-addressed ready manifest and hashes the evidence payload so B1/B3 can verify identical retrieved bytes.
- `providers/memory.py`: read-only memory interface with subject and as-of identity.
- `reasoning/single.py` and `reasoning/adaptive_mdt.py`: the two strategies. Adaptive MDT calls the existing MDAgents-style clinical reasoning through a provider bridge; Harness-supplied observations prevent its old harness skills from running a second time.
- `harness/budget.py` and `harness/trace.py`: one provider/tool budget ledger and one metadata-only trace schema.

## Request flow

1. The Harness safety gate sees the current query before memory, retrieval, or provider work.
2. Enabled memory and retrieval providers run once and return typed observed data.
3. Retrieval results carry an SHA-256 over canonical evidence JSON; the hash is recorded in the trace.
4. A selected `ReasoningStrategy` receives query, conversation, patient facts, external evidence, answer schema, and non-sensitive runtime metadata.
5. The Harness checks returned citation IDs against observed memory/evidence and parses the answer deterministically.
6. Provider/tool counts, token usage, latency, and trace ID are returned in the shared response contract.

Provider configuration is dependency injection. A model variant maps to its own configured provider, so the Harness does not inspect training method or checkpoint internals. In Common Eval, all stateless profiles use Memory OFF.

## Product API and compatibility

`POST /medical/answer` can now be built with a `HealthCopilotHarness`; this branch accepts only runtime inputs and returns the shared Harness response plus compatibility aliases. Passing the old `MedicalAgentRuntime` continues to serve its historical response contract for existing clients during migration. The old runtime is not the common evaluation path.

## Readiness gates at H0 closeout

The current run uses separate prepared candidate/scorer views from the PT-E0 data disk for DiagnosisArena-915 and CMB-COMMON-1024. Their source revisions, ordered IDs, and view hashes are frozen in the evaluation manifests. The checkout still has no qualified Common Medical KB V1 manifest or contamination audit. Therefore:

- DiagnosisArena-915 and CMB-COMMON-1024 IDs are **frozen**; scorer labels remain evaluator-only.
- CMB full-test 11,200 is not scored as a full set; the earlier 352-row partial remains historical.
- Common Medical KB is **not ready**; RAG-enabled common profiles are blocked.
- Post-training contamination audit is **not complete**.
- B0 and targeted-recovery B2 are complete for both common sets. Parser-v6 results show Adaptive MDT at +5.57 pp on CMB-COMMON-1024 and −6.45 pp on DiagnosisArena-915; the DiagnosisArena Adaptive arm includes 230 reasoning failures and both arms have zero answer-format failures. This result is descriptive because historical B0 vLLM runtime capture is incomplete. B1/B3, SFT, and GSPO have not started.

The runner refuses a dataset that is not marked ready with matching hashes, refuses an unfrozen checkpoint identity, and blocks RAG profiles unless the Common KB is READY. This run used an NVIDIA L40. The isolated vLLM endpoint on loopback port 8001 completed normal compile and warmup for targeted recovery and was shut down afterward; port 8000 was not stopped or reconfigured. The Harness client uses the frozen Qwen3-8B model identity. See `docs/evaluation/common_medical_eval_v1.md` for the recovered B0/B2 results and serving-parity limits.

## Freeze record

`HARNESS_V1` is an architecture/API checkpoint for code review, not a frozen experimental system profile. Before calling it an evaluation freeze, record code SHA, all prompt hashes, provider/model revision and checkpoint hash, dataset snapshot hashes, Common KB hashes, budget/serving settings, and the contamination report. Do not label missing evidence as a passing flag.
