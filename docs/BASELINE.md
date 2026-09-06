# Rebuild baseline: September 6, 2026

This record freezes the working software recipe without copying the 124 GiB
checkpoint or the installed environment. The original NVIDIA revision and vLLM
base are immutable identifiers. The complete patch contains all 23 modified or
new source/test files, including the otherwise untracked PLE mmap reader.

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

## Frozen inputs

| Record | Purpose |
|---|---|
| `baseline/baseline.json` | Runtime geometry, identities and template defaults |
| `baseline/model-sha256.json` | Sizes and SHA-256 of every original top-level model file |
| `baseline/requirements.lock` | Resolved build/runtime packages, exact versions and artifact hashes |
| `baseline/constraints.txt` | Observed versions used to constrain resolution |
| `patches/vllm-complete.patch` | Complete delta from the pinned upstream base |
| `patches/patched-source-sha256.json` | Expected patched source bytes |
| `baseline/recipe-sha256.json` | Integrity of the recovery recipe itself |

Model: `nvidia/Qwen3.8-Flash-Next-NVFP4` at
`fab0aecb760cec45227f6656abcaafa11abca87a`.
vLLM base: `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`.
Observed version: `0.28.1rc1.dev442+g7fbd44cbe.d20260905`.
Build date suffixes can differ on reconstruction; source identity is the
base plus the hash-verified patch. Python 3.12, CUDA 13.0 and Linux aarch64
are prerequisites. Build and runtime use `MAX_JOBS=1`, `NVCC_THREADS=1`.

The lock was resolved against the live package inventory. The editable vLLM
installation is built from the verified source with `--no-build-isolation
--no-deps`; it cannot silently upgrade the locked dependencies. The cubin
package is obtained from the [official FlashInfer index](https://flashinfer.ai/whl/flashinfer-cubin/).
System compiler, driver and OS packages are host prerequisites, not bundled
artifacts. Registry availability is still required; this is not an offline backup.

## Verification and limits

Run `bash scripts/rebuild.sh --check` to verify prerequisites and recipe
integrity without changing the runtime. Use `scripts/verify-baseline.py --source
/path/to/source --model /path/to/checkpoint` to verify content without loading
a model. Hashing the checkpoint reads about 124 GiB from disk.

For this publication, source patch reconstruction, live flag parity, dependency
resolution, package-install dry-run, model hashes and shell/Python syntax are
checked. The live endpoint remains in service. A complete fresh build and reload
has not been executed for this release, so one-command orchestration is not a
claim of a newly measured clean-machine rebuild. Existing long-context and
vision tests retain their dates and limits. Repeat raw checks before adding a
wrapper or client on a reconstructed host.
