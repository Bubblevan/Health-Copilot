# MEM-3B0Q R4 Candidate Binding Guard v1

Status: `PROPOSAL_ONLY`; offline control tests only. No model request or store mutation is authorized.

## Finding

The candidate-bounded v1 proposal validator confirms that each field points to a source-grounded alias, but it does not constrain the relationship between an attribute alias and a value span. A direct counterexample on frozen control `R4-P15` selected `COLOR` with value `leather`; the v1 validator accepted it because both spans independently occur in the source.

## Guard

This separate, non-mutating wrapper first runs the unchanged v1 validator, then rejects an attribute/value pair if the spans fall into different clauses. Hard boundaries are `;!?` and line breaks, a period followed by whitespace or end-of-input (except a period between digits), and `and` directly before another registered attribute alias. That rule rejects the color/leather cross-pair while preserving color/black, material/leather, the membership value `apples and pears`, and a decimal value such as `3.14`.

The guard is deliberately only a locality heuristic. Abbreviations can still be mistaken for terminal periods. It does not prove semantic binding within a clause, owner-to-object attachment, object identity, cardinality, event semantics, or revision evidence. It must not be treated as a complete extractor, a Memory admission policy, or authorization for B1. The original R4 protocol, freeze manifest, B0P/B0Q artifacts, and scorecard remain unchanged.

## Offline Controls

- `R4-P15 COLOR=leather`: rejected as `attribute_value_cross_clause`.
- `R4-P15 COLOR=black`: accepted by the locality guard.
- `R4-P15 MATERIAL=leather`: accepted by the locality guard.
- `R4-P09 MEMBERSHIP=apples and pears`: accepted; list conjunction is not treated as a new attribute clause.
- `R4-D1 COLOR=3.14`: accepted; the decimal point is not treated as a boundary.

These controls test a bounded guard, not model accuracy or general semantic correctness. No inference, benchmark scoring, hosted request, timestamp/question metadata read, or MemoryStore mutation was performed.
