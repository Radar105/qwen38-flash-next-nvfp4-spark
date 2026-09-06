#!/usr/bin/env bash
set -euo pipefail
cd /home/user/src/vllm-upstream
export PATH=/home/user/venvs/vllm/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/home/user/vllm/hf_cache
export HF_HUB_OFFLINE=1
export MAX_JOBS=1
export NVCC_THREADS=1
export TMPDIR=/home/user/vllm/tmp
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_CACHE_ROOT=/home/user/vllm/cache
export VLLM_QWEN4_PLE_MMAP=1
export VLLM_USE_DEEP_GEMM=0
export XDG_CACHE_HOME=/home/user/vllm/xdg_cache
exec /home/user/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/user/models/nvidia-Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 \
  --port 8092 \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 262144 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.8 \
  --quantization modelopt \
  --kv-cache-memory-bytes 8G \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --prefix-cache-retention-interval 1600 \
  --skip-mm-profiling \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --generation-config vllm \
  --no-enable-flashinfer-autotune \
  --override-generation-config '{"max_new_tokens":131072}'
