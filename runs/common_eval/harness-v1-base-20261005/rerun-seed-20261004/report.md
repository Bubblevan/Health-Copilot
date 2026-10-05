# Fixed-seed B0/B2 repeatability check

## Scope

This is a same-ID 20-case repeatability check for B0 and B2 on each Common Eval dataset. It is **not** a full-dataset rerun or a new accuracy estimate. It uses the frozen parser v7 and preserves the prior full runs.

| Dataset | Sample cases | Sample ID sequence SHA-256 | B0 correct | B2 correct |
|---|---:|---|---:|---:|
| CMB-COMMON-1024 | 20 | `a504be8585da9dffe566e475acf6ba6e4593248ccb7802c603c799dab8f92cbf` | 16/20 | 17/20 |
| DiagnosisArena-915 | 20 | `d6975298f1cc1e66b3875c55b5694d4742d482e04e8d3dccd496533e5498e658` | 9/20 | 8/20 |

The same selected IDs were read from the frozen dataset manifests. CMB's frozen subset uses selection seed `20261004`; the DiagnosisArena IDs were already frozen before this rerun.

## Seed and decoding

The dedicated vLLM instance was started with `--seed 20261004`. Requests use `temperature=0` and greedy decoding. `VllmModelProvider` does not send a per-request seed; the run seed is not an independent random draw for each answer. It controls the frozen subset protocol and initializes the server RNG, while greedy decoding avoids sampling.

The request IDs and model identity were fixed, but the generation setup did not fully match the earlier full run. The existing 8000 service was occupied by the Memory run, so this check used a separate 8001 instance with `max_model_len=8192`, `max_num_seqs=1`, and case concurrency 1. The old run recorded `max_model_len=32768` and case concurrency 4. The largest individual prompt in this sample was 1,726 tokens, well below 8,192, but batch shape and compiled runtime differed. Consequently, this checks same-seed behavior under the available serving setup; it does not establish bit-for-bit reproducibility under identical engine settings.

## Paired comparison to the frozen full-run outputs

| Dataset | Arm | Old correct / 20 | Repeat correct / 20 | Same parsed answer | Same correctness per case | Same raw answer text | Same MDT route |
|---|---|---:|---:|---:|---:|---:|---:|
| CMB-COMMON-1024 | B0 | 16 | 16 | 19/20 | 20/20 | 17/20 | N/A |
| CMB-COMMON-1024 | B2 | 17 | 17 | 20/20 | 20/20 | 4/20 | 19/20 |
| DiagnosisArena-915 | B0 | 9 | 9 | 17/20 | 18/20 | 15/20 | N/A |
| DiagnosisArena-915 | B2 | 4 | 8 | 15/20 | 16/20 | 7/20 | 13/20 |

On the CMB sample, parsed answers and correctness were unchanged for all B2 cases; B0 had one parsed-set change, but that case remained wrong. On DiagnosisArena, B0's aggregate score stayed 9/20, while two individual cases swapped correctness (`diagnosisarena:102` changed correct→wrong; `diagnosisarena:108` changed wrong→correct). B2's route and answers varied more: the repeat recovered four earlier MDT fallbacks (`diagnosisarena:100`, `:101`, `:109`, `:112`), while four different cases (`diagnosisarena:103`, `:104`, `:106`, `:11`) hit advanced-route `TypeError` during this repeat.

## Parser and runtime failures

All 20 B0 CMB, 20 B2 CMB, and 20 B0 DiagnosisArena responses parsed. B2 DiagnosisArena parsed 16/20; the four unparsed rows (`diagnosisarena:103`, `:104`, `:106`, `:11`) are MDT `reasoning_failure:TypeError` fallbacks with no final option to parse. These are execution failures, not evidence of correct answers rejected by the answer parser.

Both B0 and B2 had `retrieval_mode=off` and `memory_mode=off`; the new traces contain no retrieval or memory events.

## Infrastructure attempts

A 32K-context vLLM startup was first rejected because only 0.76 GiB remained for KV cache while 2.25 GiB was required. With context limited to 8K, vLLM compiled and completed JIT warmup without a Ninja error. A B0 full-run attempt at case concurrency 4 then crashed in FlashInfer with `BatchPrefillWithPagedKVCache` CUDA illegal memory access; its checkpoint is preserved but marked `INVALID_INFRASTRUCTURE_FAILURE` and must not be scored. Reducing the instance to one sequence and case concurrency 1 passed all four 20-case sample runs. The 8000 Memory service was not stopped or reconfigured. The retained vLLM console log has trailing whitespace and line endings normalized for repository storage; logged content is unchanged.

## Conclusion

The fixed dataset IDs and seed metadata are stable, but the seed alone did not make every parsed answer, route, or raw completion repeat exactly under a changed engine/batch configuration. This sample found no CMB B2 parser disagreement. DiagnosisArena repeats expose route variability and recurring MDT recruitment failures; they do not justify attributing the full-run B2/B0 gap to parsing. No full Common Eval rerun was performed.
