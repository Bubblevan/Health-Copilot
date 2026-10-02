# RAG-E6B Post-Training Routing Decision

## Decision: do not train from this sprint

The frozen post-training candidate rule was:

```text
capability = RAG
AND RSEL action = FALLBACK_NO_MATCH
AND every required external document is in the shared top-10
AND Vanilla grounded task success = false
```

The one-shot candidate manifest contains **0** rows. No SFT, OPD, or GRPO run was started. This is the correct outcome for the registered selector, not a reason to relax it after seeing TEST/OOD labels.

The observed RSEL gains occur on explicit relation matches. Those are solved by a deterministic Harness-side ledger and are intentionally excluded from training; teaching a model to imitate synthetic `SYNVAL` strings would be the wrong abstraction. On no-match cases, RSEL preserves Vanilla exactly. The zero-candidate result means this evaluation found no row satisfying the frozen “retrieval complete but Vanilla still fails under no-match” condition.

## What is safe to carry forward

Treat RSEL-v1 as a narrow inference-time Harness primitive:

```text
shared retrieved evidence
        ↓
exact requested-key / visible-relation check
        ├─ exact relation found → emit value + source provenance deterministically
        └─ no exact relation   → preserve the paired Vanilla result exactly
```

The contract belongs to the Harness: the model does not choose whether evidence exists, which source identity is trusted, or what provenance to claim. RSEL adds no model call and does not alter retrieval. Keep this deterministic branch separate from any future learned execution policy.

## Future training-data gate

Only a separately frozen future evaluation should produce policy-training examples. A candidate should represent a genuine unresolved execution decision—for example, complete evidence is visible but a natural-language reader fails to compose it, or evidence is insufficient and the policy should abstain/seek more evidence. Keep the input state gold-blind; store the paired trajectories and evaluator outcome separately; never promote evaluator truth, hidden answer strings, or a matched RSEL answer into model-visible context.

Before SFT/OPD, require:

1. Naturalistic or externally grounded tasks beyond the synthetic exact-token grammar.
2. Same query, retrieval result, evidence budget, and reader for candidate and baseline trajectories.
3. A predeclared candidate rule and frozen DEV/holdout split; no TEST-driven threshold changes.
4. Distinct attribution for retrieval miss, evidence-use failure, insufficient evidence, and answer-generation error.
5. Provenance and abstention checks enforced by code, not accepted from model assertions.

Do not start GRPO merely because a candidate manifest exists. First establish clean positive/negative sibling trajectories and a stable reward contract for the **Harness Execution Policy**.

## Current status

```text
RSEL matched relation success: demonstrated on synthetic exact grammar
IID reserved gate: PASS
OOD support rule: PASS on observed synthetic shifts
RAG retrieval improvement: not tested by E6B; retrieval was held identical
Public medical benchmark improvement: not established by E6B
Post-training candidates: 0
SFT / OPD / GRPO: NOT STARTED
```

See [the full reserved result and claim boundary](e6b_reserved_results.md) and [the failure taxonomy](e6b_failure_taxonomy.md).
