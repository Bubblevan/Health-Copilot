# MDAgents local reproduction protocol and result

## Reproduction protocol

The local reference followed the pinned upstream algorithm and used a single local Qwen3-8B BF16 model for every semantic call: classifier, recruiter, specialists, summarizer, moderator, and Single baseline. The upstream source was kept outside Health-Copilot because its checkout has no root license file.

Frozen controls:

- Upstream commit: 3adbd760ca809b4e7b0c1085d68314b6e7d91e1b
- Model revision: b968826d9c46dd6066d109eabc6255188de91218
- Dataset source/revision: jind11/MedQA, 27b02f66aac217933c9648a06f82e9f720377925
- Test rows: 1,273; converted test hash: c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb
- Seed: 20261003; temperature: 0; max output: 1,024 tokens
- vLLM endpoint: loopback 127.0.0.1:8000; Qwen thinking disabled for both arms
- Concurrency: 8 independent questions; calls within a question stayed sequential

The MedQA smoke check used the same 20 questions for both arms. It confirmed local-only calls, GPU use, server stability, saved checkpoints, and no runtime errors. The smoke was not used to tune accuracy.

## Full result

| Metric | Single | MDAgents Adaptive | Delta |
|---|---:|---:|---:|
| Accuracy | 61.27% (780/1,273) | 56.09% (714/1,273) | -5.18 pp |
| Parse success | 86.65% | 86.65% | 0.00 pp |
| Calls per question | 7.00 | 17.25 | +10.25 |
| Total tokens per question | 9,857 | 33,681 | +23,823 |
| Mean latency | 35.68 s | 161.95 s | +126.27 s |
| Cases per second | 0.223 | 0.049 | -0.174 |

Adaptive routed 847 questions to basic (62.57% accuracy), 291 to intermediate (52.23%), and 135 to advanced (23.70%). It had two runtime-error rows, retained in the full denominator. Paired answer agreement was 59.31%; Single-only correct was 197, Adaptive-only correct was 131.

The headline result is that this pinned method did not outperform same-model Single in this reproduction. This does not invalidate the paper's results on its stated proprietary backbones. It does mean this local run is a negative transfer result for Qwen3-8B under the frozen protocol; the graft parity check below tests implementation behavior on a small subset and is not another full reproduction.

## Identity and artifacts

The full manifest records hardware, runtime versions, model file hashes, source and converted dataset hashes, result hashes, and run parameters. Frozen outputs remain at:

/root/gpufree-share/results/mdagents-local/full_medqa/
