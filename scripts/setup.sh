#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
setup_root="${QWEN_SETUP_ROOT:-$HOME/qwen38-spark}"
for tool in git uv gcc g++ rustc cargo; do
  command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
test -x "$CUDA_HOME/bin/nvcc" || { echo 'CUDA nvcc was not found' >&2; exit 1; }
test ! -e "$setup_root/src/vllm" && test ! -e "$setup_root/venv" || {
  echo 'Source or venv already exists. Choose a new QWEN_SETUP_ROOT.' >&2; exit 1;
}
mkdir -p "$setup_root/src" "$setup_root/cache" "$setup_root/xdg-cache" "$setup_root/tmp"
git clone --filter=blob:none --no-checkout https://github.com/vllm-project/vllm.git "$setup_root/src/vllm"
git -C "$setup_root/src/vllm" checkout --detach 7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719
git -C "$setup_root/src/vllm" apply --check "$repo_dir/patches/vllm-complete.patch"
git -C "$setup_root/src/vllm" apply "$repo_dir/patches/vllm-complete.patch"
python3.12 "$repo_dir/scripts/verify-baseline.py" --source "$setup_root/src/vllm"
uv venv --python 3.12 "$setup_root/venv"
export PATH="$setup_root/venv/bin:$CUDA_HOME/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda VLLM_MAIN_CUDA_VERSION=13.0
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=12.1a
export VLLM_CACHE_ROOT="$setup_root/cache" XDG_CACHE_HOME="$setup_root/xdg-cache" TMPDIR="$setup_root/tmp"
cd "$setup_root/src/vllm"
uv pip install --python "$setup_root/venv/bin/python" \
  --require-hashes -r "$repo_dir/baseline/requirements.lock"
uv pip install --python "$setup_root/venv/bin/python" --no-build-isolation --no-deps -e .
"$setup_root/venv/bin/python" -c 'import torch,vllm; print("torch",torch.__version__); print("vllm",vllm.__version__)'
