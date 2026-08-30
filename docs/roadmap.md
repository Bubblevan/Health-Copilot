# Six-week implementation plan

## Week 1 — Scope and baseline

- Define non-goals and risk policy.
- Add 20 public-source knowledge cards and 40 manually written evaluation questions.
- Build a retrieval-only baseline with source metadata.

## Week 2 — Grounded generation

- Add hybrid retrieval, reranking and citation-constrained answering.
- Track context precision, citation completeness and unsupported claims.

## Week 3 — Harness and adversarial testing

- Complete input, tool and output gates.
- Add emergency, prescription, prompt-injection and stale-source regression tests.

## Week 4 — Human review and observability

- Implement a review-item schema, feedback taxonomy and run traces.
- Obtain feedback only through an authorized, non-patient-data review process.

## Week 5 — Post-training baseline

- Build a licensed, synthetic or publicly permitted SFT dataset.
- Run LoRA SFT; compare against the base model on a frozen evaluation set.

## Week 6 — Preference experiment and project narrative

- Create preference pairs from observed failures.
- Run a small DPO-style experiment, publish model cards and an honest limitations report.

