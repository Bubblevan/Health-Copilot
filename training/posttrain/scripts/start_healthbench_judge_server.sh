#!/usr/bin/env bash
set -euo pipefail

server="/root/gpufree-data/llama.cpp/build/bin/llama-server"
model="/root/gpufree-share/models/Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if [[ "${free_mib}" -lt 32768 ]]; then
  echo "Refusing to start Q4 judge: need at least 32 GiB free VRAM; found ${free_mib} MiB." >&2
  exit 2
fi
exec "${server}" \
  --model "${model}" \
  --alias "Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M" \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --threads 8 \
  --threads-batch 8 \
  --host 127.0.0.1 \
  --port 8080 \
  --no-webui
