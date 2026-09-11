#!/usr/bin/env bash
set -euo pipefail

# GPU 0 and 1 are commonly occupied on this host. Override when needed:
#   GPU_ID=0 ./scripts/start_vllm.sh
GPU_ID="${GPU_ID:-2}"

# FlashInfer 0.6.18 emits --compress-mode=size, which CUDA 12.0's nvcc does
# not support. vLLM's native sampler avoids that JIT compilation path.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec vllm serve Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager
