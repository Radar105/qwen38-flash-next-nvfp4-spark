# Agent guide

This repository documents a qualified single-GB10 Qwen3.8 Flash Next deployment.
Read README.md, docs/SETUP.md, CREDITS.md and the final report before changing it.

1. Preserve original NVIDIA checkpoint revision and the pinned vLLM base.
2. Use the complete patch once. Do not also apply each upstream PR separately.
3. Use QWEN_SETUP_ROOT for portable paths. Do not add personal paths, hostnames,
   credentials, private conversations or machine identity files to this repo.
4. Keep thinking enabled, BF16 KV 8GiB, MTP2, eagerTP1, one sequence,
   prefix caching, align mode and retention1600 unless testing a stated change.
5. Never run a second large engine beside the first. On GB10 use nvitop and
   system telemetry. Keep runtime MAX_JOBS=1.
6. Verify the exact served ID, context, live argv, correct answers and positive
   prefix metrics. Test the raw engine before wrappers or OpenCode.
7. Preserve failed results. A parser failure can differ from a model failure;
   inspect the saved answer before repeating expensive inference.
8. Use synthetic copyable examples. Keep AA scores separate from local
   quantization results. Do not claim MTP3 or a long soak was tested.
9. Keep prose concise, use ordinary technical terms, and avoid em dashes.
10. Update source hashes, documentation and relevant checks after changes.

Setup: `bash scripts/setup.sh`, then `bash scripts/download-model.sh`, then
`bash scripts/serve.sh`. These actions build/download/load substantial assets;
perform them only when requested in the current operating environment.

Validation: `python3 scripts/check-prefix.py --tokens 20000`. The250000-token
option takes longer. Artifacts are written to a new directory, not over old runs.

Publishing this repo does not authorize social media posting or changing
unrelated services. Model weights have their own license and are not included.

## Adapt to the target machine

Treat the README launch script as a configuration reference. Inspect the host
and resolve CUDA, environment, model and cache paths, memory, port ownership
and existing services before executing setup or launch commands. The example
paths are not evidence that the target machine is prepared. Read docs/SETUP.md
and preserve unrelated installations and services.
