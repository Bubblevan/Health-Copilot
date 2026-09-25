# E1.3 Question-only Capability Router — OOF Results

> Evaluation status: **EXPOSED_EXPLORATORY_OOF**. This is not an external test, untouched holdout, or confirmatory evaluation.
> All learned predictions are outer-fold OOF. QA outcomes, tokens, retrieval calls, and latency are counterfactual lookups from existing E1.2 rows; no answer inference was run.

## Frozen inputs and integrity

- Cases: 5,235; outer folds: 5; seed: 20260925; stratification: `subdataset_x_cost_oracle_class`.
- OOF runner commit: `a2e8921ddb0bd3753d9542a61ba52ce4f0666de1`; frozen protocol SHA-256: `c81191fdace9f13c4e937d041fc2a0385e286fbc8ecd69b25941c655958c4e80`.
- Git artifacts omit question text, answer options, gold answers, generated answers, and retrieved evidence; only case identities, labels, actions, and aggregate/counterfactual metrics are recorded.
- Train/evaluation overlap: 0.
- OOF duplicates: 0; missing predictions: 0.
- Disk free GiB start → end: C: 18.36 → 18.31; D: 248.30 → 248.27; E: 194.24 → 194.22.
- Retrieval-benefit prevalence: 5.58% (292/5,235).
- Cost Oracle v2 classes: CLOSED 3,245; BM25 rescue 196; MedCPT rescue 96; unresolved 1,698.

## Stage 1 — retrieval-benefit detection

| Detector | PR-AUC (Average Precision) | AUROC | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 0.0690 | 0.5517 | 6.02% | 27.74% |
| BGE | 0.0624 | 0.5343 | 5.66% | 48.29% |
| TF-IDF direct (diagnostic) | 0.0695 | 0.5486 | 10.19% | 3.77% |

Random-ranking baseline equals positive prevalence: **0.0558** average precision.

## Stage 2 — retriever selection on rescue cases only

| Selector | Cases | Accuracy | Macro F1 | Majority-BM25 accuracy |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF | 292 | 61.99% | 0.5043 | 67.12% |
| BGE | 292 | 55.82% | 0.5233 | 67.12% |

Stage 2 is evaluated only on rescue cases. The majority-action figure is a class-imbalance diagnostic, not the end-task policy baseline.

## End-task quality and measured historical cost

| Policy | QA accuracy | Retrieval rate | Calls | Answer input tokens | p50 proxy ms | p95 proxy ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Closed only | 61.99% | 0.00% | 0 | 1,127,913 | 233.0 | 409.3 |
| Always BM25 | 62.02% | 100.00% | 5,235 | 6,642,084 | 884.7 | 2328.7 |
| Always MedCPT | 62.45% | 100.00% | 5,235 | 6,167,268 | 1267.6 | 1835.0 |
| Random context | 60.52% | 100.00% | 5,235 | 6,667,133 | 1310.8 | 1549.5 |
| Cheap router | 62.23% | 44.15% | 2,311 | 3,640,366 | 404.0 | 2267.3 |
| Jev router | 62.35% | 93.75% | 4,908 | 5,876,822 | 2428.3 | 3044.4 |
| TF-IDF direct | 62.04% | 2.06% | 108 | 1,237,336 | 235.0 | 431.0 |
| TF-IDF hierarchical | 61.80% | 25.69% | 1,345 | 2,493,763 | 270.0 | 1383.4 |
| TF-IDF + always MedCPT | 61.89% | 25.69% | 1,345 | 2,420,647 | 270.0 | 1466.6 |
| BGE hierarchical | 61.87% | 47.62% | 2,493 | 3,626,709 | 417.0 | 1660.2 |
| BGE + always MedCPT | 61.99% | 47.62% | 2,493 | 3,535,311 | 417.0 | 1653.8 |
| Cost oracle v2 | 67.56% | 5.58% | 292 | 1,425,551 | 237.0 | 718.7 |

Latency is a component-summed historical proxy, not production end-to-end latency; learned-classifier inference time is not included.

## Best learned policy: absolute deltas against fixed references

| Reference | Δ accuracy (pp) | Δ retrieval calls | Calls change | Δ input tokens | Tokens change | Δ p95 proxy ms | p95 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cheap | -0.191 | -2,203 | -95.3% | -2,403,030 | -66.0% | -1836.3 | -81.0% |
| Always MedCPT | -0.401 | -5,127 | -97.9% | -4,929,932 | -79.9% | -1404.0 | -76.5% |

## Paired exploratory bootstrap

- Best learned `tfidf_direct` vs Cheap: Δ accuracy -0.191 pp; 95% CI [-0.669, +0.287] pp.
- Best learned vs Always MedCPT: Δ accuracy -0.401 pp; 95% CI [-1.146, +0.344] pp.
- The best learned policy is selected by the pre-frozen highest pooled-OOF-accuracy rule. These paired intervals compare that selected policy and do not adjust for policy-selection optimism; they are exploratory summaries, not confirmatory significance.

## Relative cost and headroom

- Best fixed accuracy: 62.45%; Closed: 61.99%; Cheap: 62.23%; Best learned: 62.04%; Cost Oracle v2: 67.56%.
- Oracle-v2 uplift: +5.578 pp over Closed and +5.119 pp over the best fixed arm; learned gain over Closed: +0.057 pp; closed-to-oracle headroom recovered: 1.0%.
- Learned gain over Cheap: -0.191 pp; relative to Cheap→Oracle headroom: -3.6%.
- Pareto frontier: Closed only, Cost oracle v2, TF-IDF direct.

## Routing failure taxonomy

Counts may overlap: HARMFUL_RETRIEVAL is a subset of WASTED_RETRIEVAL.

| Policy | Missed rescue | Wasted retrieval | Harmful retrieval | Unresolved retrieval | Wrong retriever |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF direct | 281 | 65 | 4 | 32 | 4 |
| TF-IDF hierarchical | 211 | 837 | 67 | 427 | 24 |
| TF-IDF + always MedCPT | 211 | 837 | 62 | 427 | 24 |
| BGE hierarchical | 151 | 1541 | 102 | 811 | 45 |
| BGE + always MedCPT | 151 | 1541 | 97 | 811 | 44 |

## Fold stability

| Policy | Fold | Accuracy | Retrieval rate | Retrieval-benefit PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF direct | 0 | 62.27% | 2.01% | 0.0925 |
| TF-IDF direct | 1 | 61.80% | 1.62% | 0.0735 |
| TF-IDF direct | 2 | 61.99% | 2.10% | 0.0799 |
| TF-IDF direct | 3 | 61.89% | 2.39% | 0.0695 |
| TF-IDF direct | 4 | 62.27% | 2.20% | 0.0549 |
| TF-IDF hierarchical | 0 | 62.08% | 0.00% | 0.0887 |
| TF-IDF hierarchical | 1 | 62.08% | 4.11% | 0.0763 |
| TF-IDF hierarchical | 2 | 60.74% | 90.26% | 0.0775 |
| TF-IDF hierarchical | 3 | 61.89% | 32.19% | 0.0720 |
| TF-IDF hierarchical | 4 | 62.18% | 1.91% | 0.0547 |
| TF-IDF + always MedCPT | 0 | 62.08% | 0.00% | 0.0887 |
| TF-IDF + always MedCPT | 1 | 62.27% | 4.11% | 0.0763 |
| TF-IDF + always MedCPT | 2 | 61.22% | 90.26% | 0.0775 |
| TF-IDF + always MedCPT | 3 | 61.89% | 32.19% | 0.0720 |
| TF-IDF + always MedCPT | 4 | 61.99% | 1.91% | 0.0547 |
| BGE hierarchical | 0 | 62.08% | 0.00% | 0.0652 |
| BGE hierarchical | 1 | 61.41% | 62.37% | 0.0612 |
| BGE hierarchical | 2 | 61.80% | 61.99% | 0.0688 |
| BGE hierarchical | 3 | 61.70% | 100.00% | 0.0794 |
| BGE hierarchical | 4 | 62.37% | 13.75% | 0.0549 |
| BGE + always MedCPT | 0 | 62.08% | 0.00% | 0.0652 |
| BGE + always MedCPT | 1 | 61.51% | 62.37% | 0.0612 |
| BGE + always MedCPT | 2 | 62.46% | 61.99% | 0.0688 |
| BGE + always MedCPT | 3 | 61.60% | 100.00% | 0.0794 |
| BGE + always MedCPT | 4 | 62.27% | 13.75% | 0.0549 |

## Research questions and closeout gates

- **RQ1 question-only retrieval-benefit signal:** YES, weak ranking signal; no useful Pareto policy emerged — positive prevalence=0.0558; TF-IDF PR-AUC=0.0690; BGE PR-AUC=0.0624; best learned OOF accuracy=0.6204 vs closed=0.6199
- **RQ2 semantic BGE vs lexical TF-IDF:** NO / mixed; BGE does not beat TF-IDF on both PR-AUC and hierarchical accuracy — BGE vs TF-IDF Stage-1 PR-AUC 0.0624 vs 0.0690; hierarchical OOF accuracy 0.6187 vs 0.6180. No confirmatory significance test is claimed.
- **RQ3 value of Stage 2 retriever selector:** NO ACCURACY VALUE over always-MedCPT — BGE hierarchical accuracy=0.6187, always-MedCPT=0.6199; rescue-only selector accuracy=0.5582, macro-F1=0.5233, majority-BM25 baseline accuracy=0.6712
- **RQ4 Pareto-domination of Cheap:** NO — frontier=['closed_book', 'cost_oracle_v2', 'tfidf_direct']; best learned=tfidf_direct accuracy=0.6204, retrieval rate=0.0206; cheap accuracy=0.6223, retrieval rate=0.4415
- **RQ5 recovery of oracle headroom:** 1.0% of Closed-to-Oracle-v2 headroom recovered — Closed=0.6199; best fixed=0.6245; Cheap=0.6223; best learned=0.6204; Oracle v2=0.6756; relative-to-Cheap-to-Oracle fraction=-3.6%

| Gate | Result |
| --- | --- |
| `RAG_BASELINES_COMPLETE` | **YES** |
| `OOF_PROTOCOL_VALID` | **YES** |
| `COST_ORACLE_V2_READY` | **YES** |
| `TFIDF_SIGNAL_FOUND` | **YES** |
| `BGE_SIGNAL_FOUND` | **YES** |
| `BGE_BEATS_TFIDF` | **NO** |
| `LEARNED_ROUTER_PARETO_DOMINATES_CHEAP` | **NO** |
| `STRONG_CAPABILITY_ROUTING_SIGNAL` | **NO** |
| `SFT_CANDIDATE` | **NO** |
| `RAG_CLOSEOUT` | **YES** |

Interpretation: keep this as an exposed exploratory routing study. Do not train SFT/GRPO from these OOF outcomes in this sprint. Any future post-training work needs a separately frozen evaluation design and genuinely unseen evaluation data.
