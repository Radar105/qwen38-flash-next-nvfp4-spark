# Native vision and retained reasoning: September 6

The current launch preserves the completed image and reasoning qualification:

- `enable_thinking:true`, `preserve_thinking:true`, xhigh default.
- Native image processor settings; the older 1 MP override is removed.
- `--limit-mm-per-prompt '{}'` uses this pinned engine's default modality counts.
- Shared-memory processor cache: 4 GiB, 512 MiB maximum object.
- Startup multimodal profiling skipped; MTP2, BF16 KV 8 GiB, eager TP1,
  262144 context and prefix caching with align/retention1600 retained.

An empty modality dictionary does not mean unlimited images. The model and
engine retain admission and context limits. Video is not qualified by these tests.

## Observed checks

Native 4096×4096 input retained 16384 image tokens and passed four-corner OCR.
PNG, JPEG, WebP, transparent PNG, four simultaneous images, multi-turn image
history, a wide 4096×512 image, natural-image understanding, spatial counting,
table extraction/arithmetic, HTTP image input, repeated-image processor-cache
reuse and image-derived tool execution passed. Invalid image data returned
HTTP 400 and the server remained healthy.

OpenCode 1.18.29 passed two images plus a real tool call, continued four-image
history, then a fifth 4096×4096 image. Serialized requests showed full image
dimensions, xhigh, `preserve_thinking:true`, and assistant reasoning replay.
The anonymous structural evidence is in `baseline/vision-wire-summary.json`.

Server-default rendering retained prior reasoning and selected xhigh; an explicit
`preserve_thinking:false` control removed the historical reasoning marker.
See `baseline/reasoning-verification.json`. Retention means replayed reasoning
stays in the model's input when supplied by the client; it is not persistent
memory across arbitrary disconnected sessions.

## OpenCode launch

Merge `config/opencode-provider.json` into the existing provider configuration.
It sets xhigh and reasoning retention explicitly. Low and medium remain available;
the unsupported high and non-thinking variants remain disabled.

```bash
bash scripts/opencode-full-images.sh
```

This version-matched launcher merges `attachment.image` into existing
`OPENCODE_CONFIG_CONTENT`, disables resizing, and admits up to 16384×16384
dimensions and 64 MiB base64. Those are client admission limits; the model
processor retains its native pixel budget. Other ordinary OpenCode settings
remain available, including a locally selected agent. No private agent files
are shipped here. Recheck this schema when changing OpenCode versions.

The default client launch resized a 4096×4096 test image to 2000×2000. Correct
OCR after resizing did not establish native-resolution behavior; the scoped
launcher supplied the full-resolution proof.

## Sustained OpenCode use

The operator confirms three hours of non-stop agentic work, including long
OpenCode conversations across
multiple compactions, sustained agentic work, and visual use in those workflows.
This operational evidence complements the captured image, reasoning-replay and
completed tool-roundtrip tests. Compaction is client-managed context reduction;
it does not mean the model can exceed its 262144-token request boundary. Production log windows show 22.7 tok/s median generation and 35.1 tok/s
maximum over 392 active decode windows. Latest cumulative prefix/image cache
hit rates were 94.7% / 97.5%; see `baseline/operational-speed.json`. Exact
conversation contents and private agent canon are intentionally not published.

## Retained limitations

One reproducible OCR error changed the synthetic label HERON 9274 to HERON 274
at seed 61. Seed 62 passed. This remains a quality failure, not a cache bug claim.
The API-suite observed peak total memory use was 97.34 GiB, not a bound for all
future workloads. Video, extreme image-count/context combinations, a controlled long soak
and maximum-length generation remain untested. Earlier September 5 reports and
charts preserve the older vision/default-thinking settings as dated history.
