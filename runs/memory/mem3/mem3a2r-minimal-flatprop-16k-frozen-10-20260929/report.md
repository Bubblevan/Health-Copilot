# MEM-3A.2R 16K Minimal FlatProp Diagnostic

Completion gate: MEM3A2R_MINIMAL_FLATPROP_FROZEN_10_DIAGNOSTIC=NO.

## Stop Reason

Pinned base commit: f2c448c20be6321f66e90511e325611c1c37e4fc. The 477-session preflight passed with a 16,384-token output reserve; maximum observed prompt was 19,862 tokens and maximum prompt plus reserve was 36,246 / 131,072. Prompt and extractor contract hashes matched the frozen v3 values. The frozen Qwen3-8B Q4_K_M llama.cpp runtime was llama.cpp 10068 (571d0d540) with the pinned server binary SHA256.

The risk-first historical capacity-tail session was first and passed. The run stopped at unique source session 102 of 477 (source ID 35201d43) on DUPLICATE_EVIDENCE_REF: proposition index 10 repeated evidence reference S0070. The response ended normally with finish_reason stop and 3,350 completion tokens, so this was a structural packet failure, not a 16K capacity failure.

There were 102 distinct new 16K request identities: 101 structurally valid outcomes, one structural failure, zero retries. The failed response envelope remains in the ignored local write-ahead cache; its committed ledger row records SHA256 1789f1ffecdf67f83e726a3c6b44382c39e811747dfdd955f3e66eec11d65d98. Historical 4K outcomes were not reused.

## Not Run

- The remaining 375 writer sessions were not called.
- FlatProp materialization, Dense embeddings/retrieval, ContextPlans, ContextBundles, and reader calls were not started.
- Labels were not loaded; TEST and the 102-question DEV set were not accessed.
- Hosted and judge calls: zero.

The historical MEM-3A.2 NO record remains unchanged. This run confirms the known tail session fits the 16K cap, but the full MEM-3A.2R diagnostic remains NO due to a distinct strict-schema failure. No retry, normalization, semantic repair, or downstream result was applied.
