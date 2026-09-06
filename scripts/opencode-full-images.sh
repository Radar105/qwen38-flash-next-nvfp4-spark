#!/usr/bin/env bash
set -euo pipefail
# Merge the scoped attachment override without discarding other inline settings.
# This schema was exercised with OpenCode 1.18.29.
export OPENCODE_CONFIG_CONTENT="$(python3 - <<'PY'
import json, os
config = json.loads(os.environ.get('OPENCODE_CONFIG_CONTENT') or '{}')
config.setdefault('attachment', {}).setdefault('image', {}).update({
    'auto_resize': False, 'max_width': 16384, 'max_height': 16384,
    'max_base64_bytes': 67108864,
})
print(json.dumps(config))
PY
)"
exec opencode --model qwen38-flash-next-nvfp4-vllm-local/nvidia/Qwen3.8-Flash-Next-NVFP4 "$@"
