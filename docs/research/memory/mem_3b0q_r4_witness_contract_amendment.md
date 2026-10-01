# MEM-3B0Q-R4 Witness Contract Amendment

Status: `FROZEN WITH MEM-3B0Q-R4 PROTOCOL v1`; preserves the reviewed v2 design note and narrows only how its source-witness requirement is represented for R4.

## Why

The v2 design note requires source-grounded atom witnesses with Unicode code-point offsets. R4 proposes one atom as a structured set of exact owner, object, attribute, and value spans. A single contiguous atom span is not well-defined for every multi-attribute sentence: two atoms may share the same owner/object evidence while grounding different attribute/value evidence.

## R4 representation

- The unchanged source proposition and its stable `source_id` remain the authoritative provenance record.
- Each atom contains exact `owner_span`, `object_span`, `attribute_span`, and `value_span` strings. Those four field witnesses, grouped in the same atom record, are the atom-level source witness; R4 does not request a separate contiguous `atom_span` string or model-generated offsets.
- Before scoring, the Harness finds each field witness exactly once in the unchanged source string, derives half-open Python Unicode code-point offsets `[start, end)`, persists them beside that field witness, and verifies `source[start:end] == witness` plus byte-identical UTF-8 encoding.
- Repeated or missing witnesses are rejected. The Harness does not trim, normalize, repair, paraphrase, infer offsets from tokens, or expand spans. Canonical identity mapping remains separate from source offsets.
- Atom grouping, atom count, field-to-atom association, and all field contents are checked against the frozen per-source oracle. Provenance coordinates establish where proposed fields came from; they do not establish that the interpretation or relation is correct.

This amendment applies only to the R4 qualification representation. It does not relax exact provenance requirements, authorize MemoryStore mutation, validate general semantic aliases, or authorize B1. The original v2 design document remains unchanged as an audit record.
