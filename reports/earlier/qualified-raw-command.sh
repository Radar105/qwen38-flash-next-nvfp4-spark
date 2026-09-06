#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/user/venvs/vllm/bin:/usr/local/cuda/bin:/home/user/.cargo/bin:/home/user/.local/bin:/usr/bin:/bin
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=1
export NVCC_THREADS=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_QWEN4_PLE_MMAP=1
export VLLM_CACHE_ROOT=/home/user/vllm/cache
export XDG_CACHE_HOME=/home/user/vllm/xdg_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd /home/user/src/vllm-upstream
exec /home/user/venvs/vllm/bin/vllm serve \
  /home/user/models/nvidia-Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --host 127.0.0.1 --port 8099 \
  --tensor-parallel-size 1 --dtype bfloat16 --quantization modelopt \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --enforce-eager --kv-cache-dtype auto --kv-cache-memory-bytes 8G \
  --gpu-memory-utilization 0.80 --no-enable-prefix-caching \
  --load-format safetensors --safetensors-load-strategy lazy \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --skip-mm-profiling --mm-processor-kwargs '{"max_pixels":1048576}' \
  --generation-config vllm --override-generation-config '{"max_new_tokens":131072}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --no-enable-flashinfer-autotune
