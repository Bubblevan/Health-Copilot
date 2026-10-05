# Parser-v7 offline scoring audit

This is a score-only re-evaluation of the frozen B0/B2 raw model outputs. It made **zero model/provider calls**. Parser-v6 results remain unchanged under `../rescored-parser-v6/`.

## Why parser v7 exists

Parser v6 truncated some explicit multi-select answer lines that included both option letters and option text. For example, `正确选项：B. 心、C. 肾、D. 脾` was scored as only `B`. Parser v7 preserves the enumerated labels in that explicit answer field while leaving compact label lists such as `A、B、C、D` to the existing compact-list parser.

## Paired results

| Dataset | B0 v7 | B2 v7 | Delta | B0-only correct | B2-only correct |
|---|---:|---:|---:|---:|---:|
| cmb-common-1024 | 638/1024 (62.30%) | 699/1024 (68.26%) | +5.96 pp | 47 | 108 |
| diagnosisarena-915 | 339/915 (37.05%) | 280/915 (30.60%) | -6.45 pp | 162 | 103 |

B0 and B2 both use `retrieval_mode=off` and `memory_mode=off`. Their manifests have no retrieval corpus, and all B2 traces show zero retrieval and memory events. The paired differences include no RAG contribution.

## Parser-v6 cases changed in CMB B2

| Case | v6 parsed | v7 parsed | B0 correct in v6? | B2 score changed to correct? |
|---|---|---|---|---|
| cmb:10555 | `['B']` | `['B', 'D']` | no | yes |
| cmb:10699 | `['B']` | `['B', 'C', 'D']` | yes | yes |
| cmb:11188 | `['B']` | `['B', 'C', 'D']` | no | no |
| cmb:3921 | `['B']` | `['B', 'C', 'D', 'E']` | yes | yes |
| cmb:9375 | `['C']` | `['C', 'E']` | yes | yes |
| cmb:9881 | `['C']` | `['C', 'D']` | no | no |

Six CMB B2 parses changed. Four of those raw answer lines exactly match their scorer answer set after recovering every enumerated label. Three corrected cases had been B0-correct/B2-wrong; one had been wrong in both arms. Two additional parsed answer sets changed but remain incorrect against gold. Thus the B0-only/B2-only paired counts move from 50/107 under v6 to 47/108 under v7; CMB B2 moves from 695 to 699 correct, while B0 remains 638. The corrected B2−B0 delta is +61/1,024 = +5.96 pp.

## DiagnosisArena: where B2 loses to B0

Parser v7 leaves DiagnosisArena unchanged: B0 339/915, B2 280/915, delta −6.45 pp. Of 162 cases correct only under B0, 95 end in an Adaptive fallback and 67 have an explicit, parseable but incorrect final answer. The final answer parser agrees with the explicit JSON/fenced-JSON/final-choice field for all 67; no final-answer parsing mismatch was found in this loss set.

Across the full 915-case B2 run, the trace records 230 MDT execution failures: 219 `TypeError:invalid_team_recruitment` on the advanced route, 10 `ValueError:invalid_complexity_output`, and 1 `ValueError:insufficient_specialists_recruited`. Within the 95 B0-correct lost cases, 89 are invalid team recruitment, 5 invalid complexity output, and 1 insufficient specialist recruitment. The fallback text is therefore an orchestrator fail-closed response, not a calibrated clinical abstention.

## Interpretation

The answer parser caused a small confirmed slice of CMB B2 losses, but does not explain the DiagnosisArena regression. DiagnosisArena is primarily affected by internal MDT complexity/team-output validation failures and wrong final choices. Raw classifier/recruiter sub-call text was not persisted—only hashes and failure codes—so its exact malformed payload cannot be reconstructed. This audit did not change prompts, model outputs, or the frozen v6 files.

Hashes and case-level scoring diffs are recorded in `audit_manifest.json`; corrected checkpoints and paired reports are alongside this report.
