# Download, build, configure and run

This recipe targets Linux aarch64 on one 128GB GB10 system with CUDA 13.0 and
Python 3.12. The measured driver was580.173.02. Use a dedicated environment and
stop other large inference engines before building or loading this model.
Keep enough disk space for the original checkpoint, source build and caches.

The original build used four compilation jobs and took about 54 minutes.
Runtime FlashInfer JIT uses one job because the loaded model leaves little
unified-memory headroom. Do not confuse those two settings.

## 1. Prerequisites

Install Git, a C/C++ toolchain, CMake, Ninja, pkg-config, CUDA13.0, Rust/Cargo,
Python 3.12 and uv using their official distributions. The script checks these
tools and does not install system packages or change the driver.

Sources: [vLLM CUDA installation](https://github.com/vllm-project/vllm/blob/7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719/docs/getting_started/installation/gpu.cuda.inc.md),
[CUDA](https://developer.nvidia.com/cuda-downloads),
[Rust](https://www.rust-lang.org/tools/install), [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Radar105/qwen38-flash-next-nvfp4-spark.git
cd qwen38-flash-next-nvfp4-spark
export QWEN_SETUP_ROOT="$HOME/qwen38-spark"
bash scripts/setup.sh
```

`setup.sh` checks out the exact vLLM base, checks and applies the complete patch,
installs the pinned source requirements, and builds the native extensions. It
refuses to reuse an existing source or venv directory. It does not reset or
overwrite an existing installation.

The source requirements include Torch2.13.0, torchvision0.28.0,
torchaudio2.11.0 and FlashInfer0.6.18. FlashInfer cubin0.6.18 comes from the
official FlashInfer wheel index referenced by vLLM's pinned requirements.
The measured environment is listed in `requirements-observed.txt` for reference.
Do not blindly install that whole inventory into another environment.

A separate fresh-machine build was not repeated for publication. Patch
application and final source hashes were checked in an isolated checkout;
the running deployment passed the functional checks in the report.

## 2. Download the original weights

```bash
export QWEN_SETUP_ROOT="$HOME/qwen38-spark"
bash scripts/download-model.sh
```

Official artifact:
[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fab0aecb760cec45227f6656abcaafa11abca87a).
The script pins revision`fab0aecb760cec45227f6656abcaafa11abca87a` and uses a
separate HF CLI environment. If access requires authentication, use `hf auth
login` locally. Never put a token in a command, config example or commit.

Default directory layout:

```text
~/qwen38-spark/
  src/vllm/                  pinned source plus complete patch
  venv/                      inference environment
  hf-venv/                   download CLI
  models/nvidia-Qwen3.8-Flash-Next-NVFP4/
    config.json
    tokenizer.json
    model.safetensors.index.json
    *.safetensors            11 original checkpoint shards
  cache/                     vLLM kernels/cache
  xdg-cache/                 compiler/library cache
  hf-cache/                  local Hugging Face cache
  tmp/
```

The51,200,245,760-byte PLE tensor is already inside the original checkpoint.
Do not create or download a separate re-quantized PLE table. The mmap reader
locates it through the safetensors index and gathers selected original FP8 rows.

Model terms: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
and the [base Qwen model license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).

## 3. Start the engine directly

```bash
export QWEN_SETUP_ROOT="$HOME/qwen38-spark"
bash scripts/serve.sh
```

The server binds loopback on8092 by default. Override`QWEN_PORT` and`QWEN_HOST`
if needed. Binding it to a network interface changes who can reach it; the
example is intentionally local. The public script uses the same semantic
engine flags as the qualified deployment, with portable directories.

```bash
curl -fsS http://127.0.0.1:8092/health
curl -fsS http://127.0.0.1:8092/v1/models
```

Confirm the exact model ID and`max_model_len:262144`. Context is the total
input plus output budget. The131072 output ceiling does not reserve another
131072 tokens beyond the context limit.

## 4. Check an answer and prefix reuse

```bash
curl -fsS http://127.0.0.1:8092/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/chat-request.json

python3 scripts/check-prefix.py --base-url http://127.0.0.1:8092 --tokens 20000
```

For the longer retrieval check, run when the server is idle:

```bash
python3 scripts/check-prefix.py --base-url http://127.0.0.1:8092 --tokens 250000
```

The script generates a synthetic haystack, embeds three distinct keys, measures
cold and warm server counters, checks both answers, and writes request/result
artifacts into a new output directory. The first250K prefill can take several
minutes. Positive cache counters and correct answers are required for PASS.

## 5. Configure OpenCode

Merge the provider object from`config/opencode-provider.json` into your existing
OpenCode config. Keep unrelated providers and agents. Select:

```text
qwen38-flash-next-nvfp4-vllm-local/nvidia/Qwen3.8-Flash-Next-NVFP4
```

The config advertises262144 context and131072 output, leaves`max_tokens:null`,
keeps thinking enabled, and sets general/header/chunk timeouts to 1200000ms.
Restart or refresh an already-open client if its model label is cached.

OpenCode used a normal OpenAI-compatible route. Our local wrapper/TUI was
separately tested, but those machine-specific managers are not prerequisites
for this portable direct-server setup.

## 6. Troubleshooting

- **No cache hits:** check all three prefix flags, the applied source hashes,
  a sufficiently long unchanged prefix, and the cache metrics. A successful
  answer or enabled flag alone does not prove reuse.
- **Miss after editing a suffix:** retention0 produced a shorter-branch miss
  in this experiment. Keep retention1600 for the tested configuration.
- **Long first prefill:** a new conversation is cold. Prefix caching helps
  repeated prefixes; it does not remove the first pass.
- **SSE timeout:** confirm all three client timeouts survived config merging.
  A short passing stream does not prove a20-minute stream.
- **Memory pressure:** use nvitop and system telemetry on GB10. Keep one large
  engine, one sequence, BF16 KV 8GiB and runtime MAX_JOBS=1 for this recipe.
- **Unexpected behavior after upgrading vLLM:** the cache PRs were unmerged
  when applied. Do not apply this full patch to an arbitrary newer version.
  Rebase deliberately and repeat raw, managed and client checks.

See the full report for preserved failed attempts and the limits of each test.
