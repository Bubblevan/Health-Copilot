# Health-Copilot post-training environment

This project keeps the GPU training stack separate from the repository's runtime environment.
The server exposes one NVIDIA L40 with 48 GB VRAM, Driver 580.126.09, and CUDA 13.0. CUDA
execution can be smoke-tested on this host without renting A800s.

## Local paths

- Qwen3-8B source: `/root/gpufree-share/data/Qwen3-8B`
- SFT data: `/root/gpufree-share/data/posttrain/sft`
- RL data: `/root/gpufree-share/data/posttrain/rl`
- Reference repositories: `/root/gpufree-share/reference`

These are external data paths; no model or dataset files are copied into this Git project.

## Environment layout

`uv` is installed at `.tools/uv` on the data disk. Keep managed Python, cache, and the virtual
environment there as well:

```bash
cd training/posttrain
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
export UV_DEFAULT_INDEX="https://pypi.org/simple"
export HF_HOME="/root/gpufree-share/.cache/huggingface"
export HF_DATASETS_CACHE="/root/gpufree-share/.cache/huggingface/datasets"
./.tools/uv python install 3.11
./.tools/uv sync --locked
```

Use `./.tools/uv run --locked python ...` for commands in this environment. Import `unsloth`
before `transformers`, `trl`, or `peft` so its patches are applied before those packages load.
For vLLM/FlashInfer JIT kernel detection, expose the CUDA toolkit and the project environment
on `PATH` (or run through `uv`, which adds the environment's `bin` directory):

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
```

The locked stack is PyTorch 2.13.0+cu130, Transformers 5.17.0, TRL 1.13.0, the pinned Unsloth
commit in `pyproject.toml`, and the pinned CUDA 13 vLLM wheel in `uv.lock`. L40 BF16 CUDA,
GSPO/GDPO config construction, imports, and one local Qwen3 vLLM generation have been smoke
checked. Keep `uv.lock` frozen; change dependencies only as an explicit stack update.

## PT-E0 frozen evaluation and local HB-Pro judge

The Qwen3-8B Base evaluation uses the frozen candidate views and separate scorer views under
`PT_E0_DATA_ROOT`. The Q4 local HealthBench judge is pinned in
`manifests/model/mistral-small-3.1-24b-q4-local-judge.json`; its evaluation prompt, runtime,
scoring formula, and parameter hashes are in `manifests/eval/healthbench_local_judge.json`.

When the L40 has at least 32 GiB free, start the judge server in one terminal:

```bash
bash scripts/start_healthbench_judge_server.sh
```

After all 525 Qwen Base HealthBench predictions and their SHA256 sidecar are frozen, run the
criterion-level local grader and score it separately:

```bash
python scripts/run_healthbench_local_judge.py --base-url http://127.0.0.1:8080
python scripts/score_healthbench_local.py
```

The grader accepts only loopback HTTP, uses no hosted API, and refuses to score if the model,
server binary, prompt, runtime, prediction hashes, or frozen settings differ from their manifests.


For later merged SFT or RL HF checkpoints, reuse the same candidate-side protocol and pass
`--checkpoint-name`, `--model-path`, and `--model-revision` to `run_base_eval.py`. The runner
records checkpoint file hashes and checks that the Qwen chat template and frozen prompt/parser
protocol still match. The scorer takes the checkpoint name from the verified prediction manifest.
