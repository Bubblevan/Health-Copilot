# MDAgents local reference

## Purpose and scope

This document records the frozen external MDAgents reference reproduction used by the Health-Copilot graft. The reference implementation stays in the separate MDAgents checkout; its source is not vendored into Health-Copilot. The upstream checkout has no root license file, so the product integration independently implements the algorithmic pattern.

- Upstream: mitmedialab/MDAgents
- Pinned commit: 3adbd760ca809b4e7b0c1085d68314b6e7d91e1b
- Local compatibility patch SHA-256: ac86a2fc15162f2771efea56b97316658533ffaeb56d55ab7c3b6a78fdc617a1
- Model: Qwen/Qwen3-8B, revision b968826d9c46dd6066d109eabc6255188de91218
- Local model path: /root/gpufree-share/data/Qwen3-8B
- Dataset: MedQA US test set, 1,273 rows; source jind11/MedQA, revision 27b02f66aac217933c9648a06f82e9f720377925
- Frozen results: /root/gpufree-share/results/mdagents-local/full_medqa/
- Full run manifest: /root/gpufree-share/results/mdagents-local/full_medqa/manifest.json

## Runtime

Inference used the local OpenAI-compatible vLLM endpoint at http://127.0.0.1:8000/v1, served model qwen3-8b-local, BF16, temperature 0, Qwen thinking disabled with chat_template_kwargs.enable_thinking=false, max output 1,024 tokens, and question-level concurrency 8.

The vLLM environment was shared with the adjacent post-training work as requested. It was Python 3.11.17 with vLLM 0.30.1rc1.dev622+gf03026a54, Torch 2.13.0+cu130, and CUDA runtime 13.0. A separate stable vLLM installation could not be completed within the shared storage quota, so the working server used the available nightly build. The service remained loopback-only.

Machine: NVIDIA L40 46,068 MiB, driver 580.126.09, CUDA driver 13.0; two Xeon Platinum 8358 CPUs (128 logical CPUs). uv was 0.12.22.

## Frozen full MedQA result

The two primary arms used the same local model, dataset, output mode, temperature, seed (20261003), and 1,024-token cap. Single forced every question through the basic/single path. Adaptive used the pinned MDAgents complexity classifier, specialist recruitment, collaboration, and moderator.

| Metric | LOCAL_QWEN_SINGLE | LOCAL_QWEN_MDAGENTS |
|---|---:|---:|
| Accuracy | 780/1,273 = 61.27% | 714/1,273 = 56.09% |
| Parse success | 1,103/1,273 = 86.65% | 1,103/1,273 = 86.65% |
| Average provider calls per question | 7.00 | 17.25 |
| Average total tokens per question | 9,857 | 33,681 |
| Mean latency per question | 35.68 s | 161.95 s |
| Throughput | 0.223 cases/s | 0.049 cases/s |

Adaptive complexity distribution and accuracy:

| Route | Cases | Correct | Accuracy |
|---|---:|---:|---:|
| Basic | 847 | 530 | 62.57% |
| Intermediate | 291 | 152 | 52.23% |
| Advanced | 135 | 32 | 23.70% |

Adaptive was 5.18 percentage points below Single on this local Qwen3-8B reproduction and incurred substantially more calls and tokens. Two adaptive rows had runtime errors; they remain in the 1,273-case denominator and were not silently removed. This is a local-backbone result, not a claim that the paper's proprietary-backbone result was reproduced.

## Preserved artifacts

The full single.jsonl, adaptive.jsonl, metrics, paired metrics, and manifest are frozen under /root/gpufree-share/results/mdagents-local/full_medqa/. Do not overwrite them during graft work.
