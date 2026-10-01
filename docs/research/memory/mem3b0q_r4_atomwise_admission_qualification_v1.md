# MEM-3B0Q R4 Atomwise Admission Qualification

## Result

`ATOMWISE_ADMISSION_OFFLINE_QUALIFICATION=PASS`

The development adapter changes only the failure granularity of proposal validation. It applies the frozen candidate, locality, joint-binding v2, and clause-local v3 checks independently to each atom. A failing atom is quarantined; valid siblings remain eligible. The per-atom validator rules are unchanged. Exact duplicate admitted atoms are deduplicated, retaining the first occurrence.

This adapter is non-mutating. It does not create memory records, resolve revisions, or select CURRENT/AS_OF/CHANGE state.

## Measured Controls

| Diagnostic | Result |
|---|---:|
| Frozen R4 oracle control cases | 20/20 exact expected result |
| Expected gold atoms across those cases | 15/15 retained; 0 quarantined |
| Mixed valid/invalid proposal scenarios | 3 |
| Strict document-level gate: valid atoms retained | 0/3 |
| Atomwise gate: valid atoms retained | 3/3 (100%) |
| Atomwise gate: known invalid atoms quarantined | 3/3 (100%) |

The three mixed scenarios comprise the recorded R4C-05 model response plus two deterministic injected-negative controls (R4C-05 and R4C-06). The injected controls are synthetic corruptions, not model outputs. In the recorded response, the valid laptop/Linux atom is retained and the second atom incorrectly binding a tablet's Windows value to that laptop is quarantined. The two injected scenarios verify the same isolation property against the frozen cross-binding controls.

The prior whole-proposal validator rejected each mixed document and therefore exposed none of its valid atoms downstream. Atomwise admission retains the valid sibling without relaxing the evidence requirements for that atom. The validator reports the isolated proposal as `atom_0`; the adapter separately preserves its original `source_atom_index` (`1` in the recorded response).

## Interpretation Boundary

These are deterministic synthetic mechanism diagnostics, not LongMemEval, Memora, medical-transfer, or unbiased model-generalization results. The 20 gold cases are oracle-constructed plumbing controls, not model predictions. The three-scenario 100% retention/quarantine figures are too small and deliberately adversarial to present as benchmark performance.

What this supports is narrower and useful: **atom-level quarantine prevents one bad proposition from suppressing an independently grounded sibling, while keeping the frozen binding guard intact.** It addresses failure containment; it does not fix the model's cross-sentence binding drift. No prompt or model change, inference, API call, or MemoryStore mutation occurred during this qualification.

## Reproduction

```powershell
python -m tools.research.memory.qualify_mem3b0q_r4_atomwise_admission_v1 `
  --output runs/memory/mem3/mem3b0q-r4-atomwise-admission-qualification-v1/result.json
```

The command refuses to overwrite an existing result. The machine-readable output records the frozen input hashes, per-case counts, quarantine reasons, and explicit limitations in `runs/memory/mem3/mem3b0q-r4-atomwise-admission-qualification-v1/result.json`.

The focused regression suite also injects one invalid atom after two independently valid sibling facts and verifies that both valid facts survive; this additional unit stress test is not added to the three-scenario aggregate above.

Atomwise-only regression suite:

```text
28 passed
```

It covers the recorded model-response replay, both frozen cross-binding controls, all 20 oracle controls including abstentions, duplicate suppression, malformed-envelope fail-closed behavior, and offline accounting.

Combined R4 request/builder/guard/smoke and atomwise suite: `120 passed`. A separate hash-locked five-POST diagnostic over the remaining synthetic controls is defined in `mem3b0q_r4_atomwise_candidate_devset_protocol_v1.md`; those new requests were not part of this offline qualification.

## Next Research Step

Keep this as a development candidate, not a frozen final method. Run the unchanged extractor request on the remaining frozen synthetic candidate controls with atomwise admission enabled, report every case and failure, and then move to the public DEV memory evaluation only after the proposal/materialization contract is stable. Do not tune against public TEST or present these control-pack counts as the final Memory headline.
