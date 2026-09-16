#!/usr/bin/env bash
set -euo pipefail

# Every option can be overridden without editing the script, for example:
#   CPU_THREADS=6 GPU_ID=0 PORT=8001 MODEL_ID=Qwen/Qwen3-8B ./scripts/start_vllm.sh
GPU_ID="${GPU_ID:-2}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
CPU_THREADS=6  # Hard cap requested for this server.

# FlashInfer 0.6.18 emits --compress-mode=size, which CUDA 12.0's nvcc does
# not support. vLLM's native sampler avoids that JIT compilation path.
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export RAYON_NUM_THREADS="${CPU_THREADS}"

exec vllm serve "${MODEL_ID}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enforce-eager
