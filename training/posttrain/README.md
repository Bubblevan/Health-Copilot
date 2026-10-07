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
