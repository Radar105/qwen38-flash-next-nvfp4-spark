#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export QWEN_SETUP_ROOT="${QWEN_SETUP_ROOT:-$HOME/qwen38-spark}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export QWEN_PORT="${QWEN_PORT:-8092}"
mode="${1:---run}"
case "$mode" in --check|--run) ;; *) echo 'Usage: rebuild.sh [--check|--run]' >&2; exit 2;; esac
for tool in git uv gcc g++ rustc cargo pkg-config python3.12 curl; do
  command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }
done
test "$(uname -m)" = aarch64 || { echo 'This baseline targets Linux aarch64 / GB10.' >&2; exit 1; }
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 13.0' || { echo 'CUDA 13.0 required.' >&2; exit 1; }
python3.12 "$repo_dir/scripts/verify-baseline.py"
echo "Recipe verified. Target: $QWEN_SETUP_ROOT; port: $QWEN_PORT"
if [[ "$mode" == --check ]]; then
  echo 'Read-only check complete. Build, download, launch and inference were not run.'
  exit 0
fi
test ! -e "$QWEN_SETUP_ROOT/src/vllm" && test ! -e "$QWEN_SETUP_ROOT/venv" || {
  echo 'Choose a fresh QWEN_SETUP_ROOT; existing source and environments are preserved.' >&2; exit 1;
}
python3.12 - <<'PY'
import os, socket
from pathlib import Path
port = int(os.environ['QWEN_PORT'])
with socket.socket() as probe:
    probe.bind(('0.0.0.0', port))
available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))) * 1024
if available < 100 * 1024**3:
    raise SystemExit('Need at least 100 GiB available RAM. Stop competing inference workloads before rebuilding.')
PY
mkdir -p "$QWEN_SETUP_ROOT"
exec > >(tee -a "$QWEN_SETUP_ROOT/rebuild.log") 2>&1
bash "$repo_dir/scripts/setup.sh"
model_dir="${QWEN_MODEL_DIR:-$QWEN_SETUP_ROOT/models/nvidia-Qwen3.8-Flash-Next-NVFP4}"
if [[ -n "${QWEN_MODEL_DIR:-}" ]]; then
  # Explicit existing-model mode: validate in place, never download over it.
  python3.12 "$repo_dir/scripts/verify-baseline.py" --model "$model_dir"
else
  bash "$repo_dir/scripts/download-model.sh"
  python3.12 "$repo_dir/scripts/verify-baseline.py" --model "$model_dir"
fi
bash "$repo_dir/scripts/serve.sh" > "$QWEN_SETUP_ROOT/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" > "$QWEN_SETUP_ROOT/server.pid"
# The driver owns only the server it started; it never stops another process.
cleanup() { kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
base_url="http://${QWEN_CHECK_HOST:-127.0.0.1}:$QWEN_PORT"
ready=0
for ((attempt=0; attempt<360; attempt++)); do
  kill -0 "$server_pid" 2>/dev/null || { echo 'Server exited; inspect server.log.' >&2; exit 1; }
  if curl -fsS --max-time 3 "$base_url/health" >/dev/null; then ready=1; break; fi
  sleep 5
done
test "$ready" = 1 || { echo 'Server readiness timed out; inspect server.log.' >&2; exit 1; }
curl -fsS "$base_url/v1/models" > "$QWEN_SETUP_ROOT/models.json"
python3.12 - "$QWEN_SETUP_ROOT/models.json" <<'PY'
import json, sys
models = json.load(open(sys.argv[1]))['data']
assert any(m['id'] == 'nvidia/Qwen3.8-Flash-Next-NVFP4' and m['max_model_len'] == 262144 for m in models)
PY
python3.12 "$repo_dir/scripts/check-prefix.py" --base-url "$base_url" --tokens 20000 --out "$QWEN_SETUP_ROOT/prefix-check"
echo 'PASS: rebuild, model hashes, readiness, exact catalog and cold/warm retrieval. Server remains attached to this driver.'
echo 'Keep this terminal/tmux session open; Ctrl-C stops this server.'
wait "$server_pid"
