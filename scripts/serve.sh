#!/usr/bin/env bash
set -euo pipefail
setup_root="${QWEN_SETUP_ROOT:-$HOME/qwen38-spark}"
mkdir -p "$setup_root/cache" "$setup_root/xdg-cache" "$setup_root/hf-cache" "$setup_root/tmp"
export HF_HOME="$setup_root/hf-cache" TMPDIR="$setup_root/tmp"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$setup_root/venv/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
export NVCC_THREADS=1
export VLLM_PLE_MMAP=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export VLLM_GDN_BA_GEMV_MAX_TOKENS=8
export VLLM_CACHE_ROOT="$setup_root/cache"
export XDG_CACHE_HOME="$setup_root/xdg-cache"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$setup_root/src/vllm"
exec "$setup_root/venv"/bin/vllm serve \
  "${QWEN_MODEL_DIR:-$setup_root/models/nvidia-Qwen3.8-Flash-Next-NVFP4}" \
  --served-model-name nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --host "${QWEN_HOST:-127.0.0.1}" --port "${QWEN_PORT:-8092}" \
  --tensor-parallel-size 1 --dtype bfloat16 --quantization modelopt \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --kv-cache-dtype auto --kv-cache-memory-bytes 8G \
  --moe-backend marlin --engram-config '{"cpu_offload":false}' \
  --gpu-memory-utilization 0.8 --enable-prefix-caching --mamba-cache-mode align --prefix-cache-retention-interval 1600 \
  --load-format safetensors --safetensors-load-strategy lazy \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}' \
  --limit-mm-per-prompt '{}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --skip-mm-profiling --mm-processor-cache-type shm \
  --mm-processor-cache-gb 4 --mm-shm-cache-max-object-size-mb 512 \
  --generation-config auto --override-generation-config '{"max_new_tokens":131072}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}'
