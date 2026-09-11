#!/usr/bin/env bash
set -euo pipefail

# Keep the embedding and reranker models on the same otherwise-free GPU as
# vLLM. start_vllm.sh reserves only 70% so these models have enough VRAM.
GPU_ID="${GPU_ID:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec python -m src.web --host 0.0.0.0 --port 7860 -v
