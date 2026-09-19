# M2 standalone policy evaluation

The evaluator materializes frozen evidence_source_ids and never invokes BM25.

## Metrics

```json
{
  "case_count": 24,
  "policy_accuracy": 0.8333333333333334,
  "macro_f1": 0.8222222222222223,
  "sufficient_recall": 1.0,
  "recoverable_recall": 1.0,
  "insufficient_recall": 0.6666666666666666,
  "per_class": {
    "sufficient": {
      "case_count": 8,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    },
    "recoverable": {
      "case_count": 4,
      "precision": 0.5,
      "recall": 1.0,
      "f1": 0.6666666666666666
    },
    "insufficient": {
      "case_count": 12,
      "precision": 1.0,
      "recall": 0.6666666666666666,
      "f1": 0.8
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
      "sufficient": 8,
      "recoverable": 0,
      "insufficient": 0,
      "conflicting": 0,
      "error": 0
    },
    "recoverable": {
      "sufficient": 0,
      "recoverable": 4,
      "insufficient": 0,
      "conflicting": 0,
      "error": 0
    },
    "insufficient": {
      "sufficient": 0,
      "recoverable": 4,
      "insufficient": 8,
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

- `m2p-013`: expected `insufficient`, predicted `recoverable`; reason_codes=`['related_but_incomplete', 'missing_required_evidence']`; proposed_query=`高血压 全麻手术 围术期 风险`; evidence_source_ids=`['cdc-high-blood-pressure-managing-01-monitoring', 'cdc-high-blood-pressure-risk-04-tobacco', 'who-hypertension-03-silent', 'who-hypertension-01-definition', 'who-hypertension-04-risk-factors']`.
- `m2p-014`: expected `insufficient`, predicted `recoverable`; reason_codes=`['related_but_incomplete', 'missing_required_evidence']`; proposed_query=`高血压 乘坐飞机 旅行 安全`; evidence_source_ids=`['nhc-hypertension-day-03-home-monitoring', 'cdc-high-blood-pressure-managing-02-care-plan', 'who-hypertension-01-definition', 'cdc-high-blood-pressure-about-01-us-threshold', 'cdc-high-blood-pressure-managing-01-monitoring']`.
- `m2p-015`: expected `insufficient`, predicted `recoverable`; reason_codes=`['related_but_incomplete', 'missing_required_evidence']`; proposed_query=`高血压 脱发 因果关系`; evidence_source_ids=`['cdc-high-blood-pressure-risk-04-tobacco', 'cdc-high-blood-pressure-risk-05-overweight', 'nhc-hypertension-day-02-weight', 'cdc-high-blood-pressure-measuring-05-white-coat', 'cdc-high-blood-pressure-risk-01-sodium']`.
- `m2p-016`: expected `insufficient`, predicted `recoverable`; reason_codes=`['related_but_incomplete', 'missing_required_evidence']`; proposed_query=`高血压 商业保险 核保 选择`; evidence_source_ids=`['cdc-high-blood-pressure-prevention-01-diet', 'cdc-high-blood-pressure-managing-01-monitoring', 'cdc-high-blood-pressure-risk-04-tobacco', 'who-hypertension-03-silent', 'who-hypertension-01-definition']`.
