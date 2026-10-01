# MEM-3B0Q R4C-05 Candidate Local Smoke Result

## Decision

`R4_CANDIDATE_LOCAL_SMOKE=NO`

The one authorized local inference completed, but the proposal failed the frozen candidate-binding validator. This is a **quality failure**, not an infrastructure failure. The run is a synthetic prompted compliance smoke only; it is not a public benchmark result and does not support a memory-quality, temporal-memory, or stale-memory-reduction claim.

No retry or additional inference was made. This closes the authorized one-case attempt.

## Frozen scope and runtime

- Case: `R4C-05` only.
- Run lock SHA-256: `104b083934104c899f735ecdbaf9cded11b28210b078da446cd28ea3cc3a77ec`.
- Request SHA-256: `6e8ba6d5b563c5df4430bb8f22616909ce0775ae86ae3aef6d8ddfbf121fa653`.
- Local reader: frozen Qwen3-8B Q4_K_M served by pinned llama.cpp `10068 / 571d0d540`; the lock specifies 99 GPU layers, Flash Attention, Q4_0 KV, and 131072 context.
- Endpoint policy: loopback only; the runner performed its locked runtime GET checks and exactly one completion POST.
- Attempts: `1`; retries: `0`; hosted calls: `0`; MemoryStore mutations: `0`.
- HTTP result: `200`; runtime preflight, immediately-before-POST check, and postflight all passed.
- Sampler initialization: `NOT_VERIFIED`; offline grammar preflight does not establish sampler initialization.

The response reports 371 prompt tokens and 216 completion tokens. Its server timing reports 108.733 seconds generation time (about 1.99 tokens/second). This is a single-run operational observation, not a throughput benchmark.

## Observed proposal

The source contains two sentences:

1. The user's personal laptop runs Linux.
2. A tablet uses Windows.

The frozen typed candidates register the owner, the personal laptop, and both operating-system attribute mentions; they intentionally do **not** register `tablet`. The expected output is therefore only the first, independently grounded laptop/Linux atom.

The model returned that correct atom, then emitted a second atom binding `Windows` and the second sentence's operating-system mention to the personal laptop. The prompt explicitly says not to carry an object across sentences or map an unlisted noun to another listed object. The second atom is therefore model output drift and an invalid cross-sentence entity binding, not evidence that the prompt demanded an unstable output shape.

The validator rejected the full proposal with:

```text
JointBindingError:atom_1:value_crosses_typed_anchor
```

In the frozen v2 guard, this diagnostic is raised because the selected laptop-to-second-attribute interval crosses the first typed attribute anchor. The failure is consistent with the bad second binding; it should not be paraphrased as a generic failure to understand the value `Windows`.

## Contract assessment

The per-atom binding rule is appropriately conservative for this control: accepting the second atom would incorrectly attach a tablet's operating system to the laptop. The model was instructed to omit unsupported facts while preserving separately supported atoms, and it did not comply.

There is, however, a separate all-or-nothing behavior: one invalid atom rejects the proposal containing the valid laptop/Linux atom. That is a **contract/harness amplification cost**, not the root cause of the invalid binding. Any future change toward atom-level quarantine or partial acceptance must be designed and qualified offline against the frozen controls, versioned as a new method, and reviewed before new inference. This run does not authorize that follow-up inference.

## Evidence artifacts

Artifacts are preserved under `runs/memory/mem3/mem3b0q-r4-candidate-local-smoke-v1/`:

| Artifact | SHA-256 |
|---|---|
| `reservation.json` | `dc46180deB06407a99f3c257d5a59adb75ce661a10c69b2705c45eaf6720d38c` |
| `request.json` | `6e8ba6d5b563c5df4430bb8f22616909ce0775ae86ae3aef6d8ddfbf121fa653` |
| `response_body.bin` | `6f5fe736323fbda57edaf785d149c5d1a7b13ef9621986d24b1594249e575a8e` |
| `result.json` | `7a378420c1a9f10122f74bb108cfcf2e42d615d5bd859532098077592b2e14d1` |

The runner result records process SHA-256 `351bfc13ad738e1b712b79082654ab7998b161959a37f876ace206eae5c642e8` at each checkpoint. The lock SHA, request SHA, response envelope SHA, token counts, status, runtime check hashes, and `hosted_calls=0` are also recorded in `result.json`.

## Research interpretation

This result is useful as a narrow mechanism diagnostic: the frozen local reader can produce one correct atom and, in the same response, a plausible but prohibited cross-sentence binding; deterministic validation prevented that proposal from mutating memory. It also exposes valid-fact suppression caused by document-level rejection.

It does **not** yet support the central RevMem story. There is no revision materialization, CURRENT/AS_OF/CHANGE evaluation, public LongMemEval or Memora score, or measured stale-memory reduction in this run. The next research iteration should first address the measured partial-acceptance trade-off offline and freeze a new protocol before requesting any broader model run.
