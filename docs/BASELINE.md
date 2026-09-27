# Rebuild baseline: September 27, 2026

This record freezes the working software recipe without copying the 124 GiB
checkpoint or the installed environment. The original NVIDIA revision and vLLM
base are immutable identifiers. The complete patch contains all 53 modified,
new or removed source/test files, including the PLE mmap reader, the parser
fix and the September 27 additions.

After inspecting and preparing the target GB10 host:

```bash
export QWEN_SETUP_ROOT="$HOME/qwen38-rebuilt"
# Optional: reuse an existing checkpoint in place, without another download.
export QWEN_MODEL_DIR="$HOME/models/nvidia-Qwen3.8-Flash-Next-NVFP4"
bash scripts/rebuild.sh
```

Omit `QWEN_MODEL_DIR` on a new host to download the pinned checkpoint. The
script requires a fresh source/environment destination and refuses a busy
port or less than 100 GiB available RAM. It does not stop competing services.
Plan about 124 GiB for weights plus build, environment and cache space; reusing
existing weights avoids that second allocation. This is an application rebuild,
not an OS, driver, credential or private agent restore.

The driver verifies the recipe, builds, verifies model hashes, starts vLLM,
waits for health, checks the served ID and 262144 context, then runs the 20K
cold/warm retrieval check. Success requires correct answers and positive reuse.
It keeps the server attached; run in tmux for durability. Build and server logs
are in `QWEN_SETUP_ROOT`. It never reports readiness from a successful launch alone.
If `QWEN_HOST` binds a particular non-loopback address, set `QWEN_CHECK_HOST`
to that address for validation. Loopback is the public default.

## What changed since September 15

- vLLM base moved from `9a35c081` (September 11) to `e7900156` (September 27).
  PR #55390 is now in the base. The base also brings #58434 and #58368, which
  fix hybrid MTP with prefix caching: a one-token prompt tail over cached
  state, padded with placeholder drafts, is now verified as a speculative row
  instead of being prefilled into linear-attention state that cannot roll the
  placeholders back; and prompt-tail prefix-cache hits work again under MTP.
- Added to the patch: #55122 (deterministic `persistent_topk` select for the
  sparse-attention indexer; 223 kernel tests pass on GB10), #50021 (bounded
  accepted-token state lookups in GDN speculative decoding), #56724 (drafts
  sampled within each request's top-k/top-p, adapted to keep the padded
  single-row draft logits used on GB10), and #57318 (FlashInfer BF16 matmul
  for the GDN gate projection, enabled by `VLLM_GDN_BA_GEMV_MAX_TOKENS=8`).
- MoE experts run through Marlin (`--moe-backend marlin`): NVFP4 weights,
  BF16 activations, for the main and MTP experts. Previously FlashInfer
  CUTLASS W4A4.
- `--engram-config '{"cpu_offload":false}'` replaces `VLLM_PLE_CPU_OFFLOAD`,
  which upstream removed. `VLLM_USE_DEEP_GEMM=0` is no longer set.
- Batch 4096 (was 2048; measured neutral). MTP stays at 2: MTP3 was measured
  and was slower in every cell on this hardware.
- Tried and not included: #57605 (align-mode lookahead allocation) fails an
  allocator assertion on this base.

## Changes on September 15

- vLLM base moved from `7fbd44cb` (September 5) to `9a35c081` (September 11).
  Two of the five carried fixes (#54713, #55375) are now in the base.
- Execution moved from `--enforce-eager` to CUDA graphs:
  `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_AND_PIECEWISE"}'`
  on Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`).
  The PLE mmap reader was rewritten for this: rows are gathered into a
  module-owned buffer before the captured forward runs. The environment
  variable is now `VLLM_PLE_MMAP=1` (previously `VLLM_QWEN4_PLE_MMAP`).
- Parser: upstream PR #56661 stops the qwen3 streaming parser from treating a
  literal `<tool_call>` in prose or reasoning as a tool-call opener and
  dropping the final answer.
- QSA indexer memory: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` caps the indexer
  logits buffer for long prefill on unified memory (upstream issue #56457).
  Measured on the live host: 250K cold prefill 128.8 s with a 0.22 GiB drop in
  available memory, against 192.5 s and about 25 GiB on September 11 without
  the cap.
- `--generation-config auto` (was `vllm`), `--model-loader-extra-config` with
  four loader threads, and `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
  `--no-enable-flashinfer-autotune` is no longer passed.
- Geometry unchanged: 262144 context, BF16 KV 8 GiB, MTP2, one sequence,
  batch 2048, prefix caching with align mode and retention interval 1600,
  native image processing with the 4 GiB / 512 MiB shared-memory cache.

## Frozen inputs

| Record | Purpose |
|---|---|
| `baseline/baseline.json` | Runtime geometry, identities, environment and template defaults |
| `baseline/model-sha256.json` | Sizes and SHA-256 of every original top-level model file |
| `baseline/requirements.lock` | Resolved build/runtime packages, exact versions and artifact hashes |
| `baseline/constraints.txt` | Observed versions used to constrain resolution |
| `patches/vllm-complete.patch` | Complete delta from the pinned upstream base |
| `patches/patched-source-sha256.json` | Expected patched source bytes |
| `baseline/recipe-sha256.json` | Integrity of the recovery recipe itself |

Model: `nvidia/Qwen3.8-Flash-Next-NVFP4` at
`fab0aecb760cec45227f6656abcaafa11abca87a` (unchanged since September 5; the
model manifest is carried forward).
vLLM base: `e7900156e130c9880eb03b7c1f2df32820e7a2be`.
Observed version: `0.30.1rc1.dev232+g4e56a7287`.
Build date suffixes can differ on reconstruction; source identity is the
base plus the hash-verified patch. Python 3.12, CUDA 13.0 and Linux aarch64
are prerequisites. Build and runtime use `MAX_JOBS=1`, `NVCC_THREADS=1`.

The lock was resolved from the base's `requirements/build/cuda.txt` and
`requirements/cuda.txt` against the live package inventory. All 205 locked
versions equal the running environment; no package needed an override. The
editable vLLM installation is built from the verified source with
`--no-build-isolation --no-deps`; it cannot silently upgrade the locked
dependencies. FlashInfer and its cubin package come from the
[official FlashInfer index](https://flashinfer.ai/whl/), declared in the lock.
System compiler, driver and OS packages are host prerequisites, not bundled
artifacts. Registry availability is still required; this is not an offline backup.

## Verification and limits

Run `bash scripts/rebuild.sh --check` to verify prerequisites and recipe
integrity without changing the runtime. Use `scripts/verify-baseline.py --source
/path/to/source --model /path/to/checkpoint` to verify content without loading
a model. Hashing the checkpoint reads about 124 GiB from disk.

For this publication, the complete patch was applied to a fresh checkout of
the pinned base and all 52 patched files matched the live source by SHA-256;
the portable launch script matched the live process on all 33 flags and its
environment; the lock passed a hash-checked install dry-run; shell and Python
syntax were checked. The live engine passed 20K and 250K cold/warm retrieval,
GSM8K and prompt-NLL checks on September 27
([measurements](../reports/2026-09-27/README.md)). A complete fresh build and
reload on a clean machine has not been executed for this release, so
one-command orchestration is not a claim of a newly measured clean-machine
rebuild. Earlier long-context,
throughput and vision tests keep their dates, their eager-execution
configuration and their limits. Repeat raw checks before adding a wrapper or
client on a reconstructed host.
