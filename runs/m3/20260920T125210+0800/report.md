# M3 standalone capability evaluation

Frozen evidence, proposed query, and reviewed scope version are materialized directly; BM25 is not invoked.

```json
{
  "case_count": 30,
  "capability_policy_accuracy": 1.0,
  "capability_macro_f1": 1.0,
  "sufficient_recall": 1.0,
  "recoverable_recall": 1.0,
  "insufficient_recall": 1.0,
  "matched_topic_exact_set_accuracy": 0.9,
  "out_of_scope_false_recovery_rate": 0.0,
  "in_scope_false_reject_rate": 0.0,
  "policy_error_rate": 0.0,
  "per_class": {
    "sufficient": {
      "case_count": 10,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    },
    "recoverable": {
      "case_count": 10,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    },
    "insufficient": {
      "case_count": 10,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    },
    "conflicting": {
      "case_count": 0,
      "precision": null,
      "recall": null,
      "f1": null
    }
  },
  "confusion_matrix": {
    "sufficient": {
      "sufficient": 10,
      "recoverable": 0,
      "insufficient": 0,
      "conflicting": 0,
      "error": 0
    },
    "recoverable": {
      "sufficient": 0,
      "recoverable": 10,
      "insufficient": 0,
      "conflicting": 0,
      "error": 0
    },
    "insufficient": {
      "sufficient": 0,
      "recoverable": 0,
      "insufficient": 10,
      "conflicting": 0,
      "error": 0
    },
    "conflicting": {
      "sufficient": 0,
      "recoverable": 0,
      "insufficient": 0,
      "conflicting": 0,
      "error": 0
    }
  }
}
```

## False decisions

None.
