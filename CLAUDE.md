# Claude setup guide

Follow [AGENTS.md](AGENTS.md) and [docs/SETUP.md](docs/SETUP.md).

The key files are `scripts/setup.sh`, `scripts/download-model.sh`,
`scripts/serve.sh`, `config/opencode-provider.json`, and
`scripts/check-prefix.py`. `QWEN_SETUP_ROOT` controls the portable installation
directory. Model files stay under its `models/` directory.

Read the final report and patch credits before modifying the engine. Test actual
cache counters and answers, not just flags. Preserve all unrelated user config.
Use synthetic examples and keep private paths, tokens and histories out of Git.


## Adapt to the target machine

Treat the README launch script as a configuration reference. Inspect the host
and resolve CUDA, environment, model and cache paths, memory, port ownership
and existing services before executing setup or launch commands. The example
paths are not evidence that the target machine is prepared. Read docs/SETUP.md
and preserve unrelated installations and services.

## Current recovery baseline (2026-09-06)

Read docs/BASELINE.md and docs/VISION_REASONING.md. Use scripts/rebuild.sh after
host inspection; --check is read-only. Preserve xhigh thinking, preserve_thinking,
native image processing and the 4 GiB / 512 MiB shared-memory image cache.
Keep all private agent canon and credentials outside this public repository.
