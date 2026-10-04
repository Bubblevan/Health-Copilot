# Health-Copilot Harness V1

## Purpose and status

Harness V1 is the single request, context, execution, budget, trace, and response boundary for product requests and future common system evaluations. This change introduces the boundary and adapters; it does not claim a production model profile, public dataset snapshot, or Common Medical KB is qualified.

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

The current checkout has local knowledge cards and historical subsystem fixtures, but it has no pinned DiagnosisArena-915 or CMB-11,200 public snapshot, no frozen CMB-COMMON-1024 IDs, no post-training manifest snapshots, and no qualified Common Medical KB V1 manifest. Therefore:

- DiagnosisArena-915 and CMB-COMMON-1024 are **not frozen**.
- Common Medical KB is **not ready**; RAG-enabled common profiles are blocked.
- Post-training contamination audit is **not complete**.
- No B0/B1/B2/B3 benchmark run, SFT, or GSPO run was started by H0.

The runner refuses a dataset that is not marked READY with matching hashes, refuses an unfrozen checkpoint identity, and blocks RAG profiles unless the Common KB is READY. The workstation is RTX 4090; H0 validation is limited to deterministic smoke checks, not the evaluation matrix.

## Freeze record

`HARNESS_V1` is an architecture/API checkpoint for code review, not a frozen experimental system profile. Before calling it an evaluation freeze, record code SHA, all prompt hashes, provider/model revision and checkpoint hash, dataset snapshot hashes, Common KB hashes, budget/serving settings, and the contamination report. Do not label missing evidence as a passing flag.
