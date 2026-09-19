# M2 standalone policy evaluation

```json
{
  "case_count": 24,
  "policy_accuracy": 0.7083333333333334,
  "macro_f1": 0.5238095238095238,
  "sufficient_recall": 0.8333333333333334,
  "recoverable_recall": 0.0,
  "insufficient_recall": 1.0,
  "per_class": {
    "sufficient": {
      "case_count": 6,
      "precision": 0.625,
      "recall": 0.8333333333333334,
      "f1": 0.7142857142857143
    },
    "recoverable": {
      "case_count": 6,
      "precision": null,
      "recall": 0.0,
      "f1": 0.0
    },
    "insufficient": {
      "case_count": 12,
      "precision": 0.75,
      "recall": 1.0,
      "f1": 0.8571428571428571
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
      "sufficient": 5,
      "recoverable": 0,
      "insufficient": 1,
      "conflicting": 0,
      "error": 0
    },
    "recoverable": {
      "sufficient": 3,
      "recoverable": 0,
      "insufficient": 3,
      "conflicting": 0,
      "error": 0
    },
    "insufficient": {
      "sufficient": 0,
      "recoverable": 0,
      "insufficient": 12,
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
