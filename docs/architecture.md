# Architecture

```text
User question
  -> Input validation / PII minimization
  -> Deterministic safety gateway
       -> urgent care / human review (terminal)
       -> retrieval-eligible question
  -> Query rewrite + hybrid retrieval + reranking
  -> Evidence-aware LLM generation
  -> Citation verifier + output policy
  -> Answer, review queue and observability events
```

## Project structure

| Directory | Responsibility |
| --- | --- |
| `src/` | Runtime code: routing, retrieval, generation, review contracts. |
| `data/knowledge_cards/` | Versioned public-source metadata, not patient data. |
| `evals/` | Golden questions, adversarial prompts and regression reports. |
| `tests/` | Deterministic unit/integration tests. |
| `docs/` | Architecture decisions, data boundary and experiment records. |
| `artifacts/` | Local model outputs and reports; ignored by Git. |

## Harness design

Harness is the outer control system around an LLM, not a fancy prompt. The first version should include:

1. **Input gate**: size limits, PII minimization, injection-pattern logging.
2. **Risk gate**: deterministic emergency/prescription routing before the LLM.
3. **Tool policy**: the generator only reads reviewed knowledge cards; it cannot call arbitrary web URLs.
4. **Output gate**: assertions require citations; disallowed claims are rejected or sent to review.
5. **Human-in-the-loop**: uncertain / stale / high-risk cases become review items rather than model answers.
6. **Traceability**: record prompt version, retrieved source IDs, model version, policy decision and evaluation version.

The initial code implements item 2. Add the other gates one at a time and write regression tests before expanding capability.

