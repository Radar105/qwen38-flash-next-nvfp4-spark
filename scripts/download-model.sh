#!/usr/bin/env bash
set -euo pipefail
setup_root="${QWEN_SETUP_ROOT:-$HOME/qwen38-spark}"
if [[ ! -x "$setup_root/hf-venv/bin/hf" ]]; then
  uv venv --python 3.12 "$setup_root/hf-venv"
  uv pip install --python "$setup_root/hf-venv/bin/python" 'huggingface_hub==1.30.0'
fi
model_dir="${QWEN_MODEL_DIR:-$setup_root/models/nvidia-Qwen3.8-Flash-Next-NVFP4}"
mkdir -p "$model_dir"
exec "$setup_root/hf-venv/bin/hf" download nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --revision fab0aecb760cec45227f6656abcaafa11abca87a \
  --local-dir "$model_dir"
