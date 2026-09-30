# RAG-E5-B3 Reader V2 Qualification

## Purpose and boundary

B3 qualifies a claim-first reader contract after the B2 measurement instrument failed. It does not
change the task set, retriever, corpus, or B2 results. Qualification uses 24 independent fixtures
and approved public corpus passages only; it does not open B2 arm outputs, teacher cases, or the
202608 batch. It makes no retrieval or bridge calls.

The reader returns only `state_facts` and `guidance_facts`. State field names and trend values are
schema enums; guidance citations are attached to individual claims. The deterministic
`ReaderStateView` projects allowlisted fields from LongitudinalStatePacket v4 and excludes packet
identity, narrative, provenance internals, teacher labels, future data, and retrieved evidence.

## Frozen qualification protocol

The protocol and its source/data identities are stored in
`runs/rag_e5/e5b3_reader_qualification_lock.json`. The qualification runner verifies the lock,
fixture set, corpus, model, llama.cpp binary/version, and source hashes before making a call.

The fixture strata are:

| Stratum | Cases | Input |
|---|---:|---|
| Q0 | 8 | Synthetic state only; covers body weight/BMI × rising/falling/stable |
| Q1 | 8 | One approved public guideline passage, no state |
| Q2 | 8 | Synthetic state plus five approved public passages |

Candidate output budgets are tried in order: 256, 384, 512. The smallest budget passing all gates
is selected. Each attempted budget makes one call for each of the 24 fixtures; calls are never
retried at the same budget. Any incomplete call ledger prevents a replay.

Qualification requires all of the following at a candidate budget:

- 24/24 valid Reader V2 JSON outputs;
- zero `finish_reason=length` responses;
- exact canonical metric/trend match for all 16 Q0/Q2 fixtures;
- no invented citation and at least one supplied valid citation for each of the 16 Q1/Q2 fixtures;
- nearest-rank p95 completion tokens no greater than 75% of the output cap.

If no candidate passes, the reader qualification stops and B3 measurement recovery is not run.
The generated qualification report is `runs/rag_e5/e5b3_reader_qualification.json`; append-only
call records and server logs are stored under the external `rag_e5/e5b3` artifact directory.

## Runtime identity

The lock pins the Qwen3-8B Q4_K_M model SHA-256, the llama.cpp executable/version, a loopback-only
server, CPU inference (`n-gpu-layers=0`) to match the B2 runtime, 16k context, reasoning disabled,
temperature zero, stateless requests, and prompt caching disabled. The only qualification
variables are the new Reader V2 contract and the predeclared output-budget ladder.

Qualification is an instrument check, not evidence that external retrieval has conditional value.
Only after qualification passes may B3 replay the exact frozen B2 evidence contexts under new run
identities and a separate execution manifest.
