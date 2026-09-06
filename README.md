# Qwen3.8 Flash Next NVFP4 on one GB10

<img src="assets/banner-5x2.png" alt="Qwen3.8 Flash Next" width="1000" height="400">

262K context, BF16 KV, MTP2, thinking enabled, and working prefix caching on one
DGX Spark class GB10 system. Original NVIDIA weights, with documented vLLM
patches. This is a tested local configuration, not an unmodified upstream recipe.

## Set up with your agent

Paste this repository link into your coding agent:

```text
https://github.com/Radar105/qwen38-flash-next-nvfp4-spark
```

Ask it to read `AGENTS.md` and `docs/SETUP.md`, inspect your DGX, and adapt the
setup to your machine. It should resolve your CUDA installation, Python
environment, model and cache paths, available memory, ports and existing
services before building or launching. Have it apply the documented patches,
download the pinned weights, then verify model responses and prefix-cache reuse.

## September 6 rebuild baseline

The current baseline includes native 4096×4096 image processing, multi-image
history, xhigh thinking by default, reasoning retention, and a 4 GiB shared-memory
image cache. The [baseline record](docs/BASELINE.md) includes the exact model
revision, all 23 patched source files, a hash-locked dependency set, and checksums
for every original model file. No model or environment copy is required.

After your agent checks the host, one entry point builds, downloads or reuses
verified weights, starts the engine, and checks correct cold/warm retrieval:

```bash
bash scripts/rebuild.sh
```

Use `bash scripts/rebuild.sh --check` for read-only prerequisites and recipe
integrity. A fresh rebuild was not run for this update; the running configuration
and retained vision/reasoning tests were verified.

## vLLM configuration reference

The script below shows the tested settings and the repository's example
installation layout. Its paths must be checked and fitted to your system by
you or your agent before use. The full setup workflow is in [docs/SETUP.md](docs/SETUP.md).

```bash
#!/usr/bin/env bash
set -euo pipefail
setup_root="${QWEN_SETUP_ROOT:-$HOME/qwen38-spark}"
mkdir -p "$setup_root/cache" "$setup_root/xdg-cache" "$setup_root/hf-cache" "$setup_root/tmp"
export HF_HOME="$setup_root/hf-cache" TMPDIR="$setup_root/tmp"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$setup_root/venv/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=1
export NVCC_THREADS=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_QWEN4_PLE_MMAP=1
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
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --enforce-eager --kv-cache-dtype auto --kv-cache-memory-bytes 8G \
  --gpu-memory-utilization 0.8 --enable-prefix-caching --mamba-cache-mode align --prefix-cache-retention-interval 1600 \
  --load-format safetensors --safetensors-load-strategy lazy \
  --limit-mm-per-prompt '{}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --skip-mm-profiling --mm-processor-cache-type shm \
  --mm-processor-cache-gb 4 --mm-shm-cache-max-object-size-mb 512 \
  --generation-config vllm --override-generation-config '{"max_new_tokens":131072}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --no-enable-flashinfer-autotune \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"xhigh"}'
```

The launch uses 262144 context, MTP2, BF16 KV 8GiB and prefix caching with
align/retention1600. The model-card output ceiling is 131072 tokens, bounded
by remaining context.

## Three hours of sustained use

The operator reports three hours of continuous agentic OpenCode work, including
visual tasks and long conversations across multiple compactions. Across 392
active decode log windows in that period, median generation was **22.7 tok/s**,
reaching **35.1 tok/s**. The latest reported prefix-cache hit rate was **94.7%**
and image-cache hit rate **97.5%**. These are production log-window measurements
and cumulative cache rates, not isolated per-request benchmarks.
[Anonymous measurements](baseline/operational-speed.json).

## Measured speeds

With MTP2, the 47,643-token matched test delivered **1,863 tok/s prefill** and
**26.83 tok/s decode**, a **62.2% decode increase** over no MTP. The separate
30K prompt test measured **2,051 tok/s prefill** and **33.84 tok/s decode**.

| MTP2 measurement | Input tokens | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| Matched MTP comparison | 47,643 | 1,863.14 | 26.83 |
| 30K prompt sweep | 29,985 | 2,050.96 | 33.84 |

These September 5 throughput tests used BF16 KV, thinking enabled and prefix
caching off, before the final 262K prefix-cache deployment. Each number is an
individual measured run. The [source data](reports/final/chart_data.json) and
[30K sweep chart](reports/final/09_30K_Decode_Historical.png) are included.

![MTP2 decode and prefill throughput compared with no MTP at 47,643 input tokens](reports/final/02_MTP2_Selection_Historical.png)

## Prefix-cache speeds

The 249,986-token repeated prompt reused 248,000 tokens. Server prefill fell
from **157.90 seconds to 1.45 seconds**. All five 250K retrieval variants passed.
That measures repeated prefill, not a 109x increase in generation speed.

![Prefix-cache measurements](reports/final/03_Prefix_Cache_Cold_vs_Warm.png)

## Start here

- [Prepare a clean headless DGX](docs/HEADLESS.md)
- [Setup and downloads](docs/SETUP.md)
- [Full final report](reports/final/Qwen38_Final_Production_Report.md)
- [PDF atlas](reports/final/Qwen38_Final_Production_Atlas.pdf)
- [Offline HTML report](reports/final/Qwen38_Final_Production_Offline.html)
- [Copyable examples](examples/README.md)
- [Complete source patch](patches/vllm-complete.patch)
- [Patch credits and source links](CREDITS.md)
- [Earlier charts from the same day](reports/earlier/README.txt)

The repository includes PNG charts, editable SVG masters, underlying JSON/CSV,
launch scripts, example requests and results, and an anonymized OpenCode config.
Model weights are downloaded from NVIDIA. They are not stored in this repo.

## Downloads

- [All charts, reports, scripts and examples](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark/releases/download/v2026.09.06-baseline/Qwen38_2026-09-06_Baseline.zip)
- [Offline Git bundle](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark/releases/download/v2026.09.06-baseline/Qwen38_2026-09-06_Repository.bundle)
- [Release files](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark/releases/tag/v2026.09.06-baseline)

The ZIP contains the final and earlier chart sets in separate folders. The Git
bundle contains a clean source snapshot and can be cloned without a network connection:

```bash
git clone Qwen38_2026-09-06_Repository.bundle qwen38-flash-next-nvfp4-spark
```

## Configuration

| Setting | Value |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Model revision | `fab0aecb760cec45227f6656abcaafa11abca87a` |
| vLLM source base | `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719` |
| Measured vLLM version | `0.28.1rc1.dev442+g7fbd44cbe.d20260905` |
| Context | 262144 total tokens |
| Output ceiling | 131072 tokens, bounded by remaining context |
| KV cache | 8 GiB BF16 |
| Speculation | Native MTP, 2 draft tokens |
| Execution | Eager, TP1, one sequence, batch2048 |
| Prefix cache | Enabled, Mamba align, retention interval1600 |
| Thinking | xhigh default, enabled, reasoning retained; low/medium available |
| PLE | Original FP8 table, local mmap reader |
| Vision | Native processor limits, 4096×4096 tested, multi-image history |
| Image cache | Shared memory 4 GiB, maximum object 512 MiB |

The checkpoint is mixed precision. Main routed experts are NVFP4; PLE and MTP
routed experts use their original FP8 formats; other layers include BF16.
The KV dtype is separate from weight quantization.

## What passed

- Operator-confirmed sustained agentic OpenCode use, including long
  conversations across multiple compactions and visual workflows.

- September 6: native 4096×4096 input with 16384 image tokens; PNG/JPEG/WebP,
  four-image API history, five-image OpenCode history, image tools, xhigh and
  retained reasoning. [Details and limits](docs/VISION_REASONING.md).

- 21 raw prefix-cache requests, including 20K/64K/250K cold and warm retrieval,
  changed suffixes, shorter branches, interleaved requests, tools and follow-up.
- Separate raw and managed260,834-token three-needle retrieval.
- 11 managed cache requests, feature smoke tests, actual TUI warm action and
  an actual streaming OpenCode tool roundtrip with positive cache counters.
- 201 core/worker tests and186 cache/scheduler tests. Two PP2 cases require
  more than this single GPU. Source pre-commit passed.
- Complete patch applied to the pinned base in an isolated checkout. All23
  affected source/test files matched the qualified source by SHA-256.

Results are individual measured runs. No extended soak, concurrency above one,
MTP3 result, full131072-token generated answer, or bitwise hidden-state
equivalence is claimed. A fresh-machine build was not repeated for this release.

## Model context

![AA Intelligence Index](reports/final/10_AA_Intelligence_Index.png)

Artificial Analysis lists Qwen3.8-Flash-Next at **45.6** on Intelligence Index
v4.2. The chart keeps one highest-scoring entry per model family across effort
settings and older versions, sorted by score. Distinct model roles remain
separate. Exact selected settings are recorded in the source data.

NVIDIA reports near-parity with FP8 across nine benchmarks. The largest decrease
is0.7 points. See the [official precision results](docs/NVIDIA_PRECISION.md).

## Credits and licenses

Qwen/Alibaba created the base model. NVIDIA produced the ModelOpt NVFP4 artifact.
vLLM, FlashInfer, PyTorch and their contributors provide the inference stack.
Individual patch authors are listed in [CREDITS.md](CREDITS.md). Charts come from measured data.

Code and vLLM-derived patches are provided under Apache-2.0. Model weights retain
the NVIDIA Open Model License and the base model's applicable terms. AA data
belongs to Artificial Analysis. This project does not claim endorsement by
those organizations.
