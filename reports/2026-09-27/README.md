# September 27 measurements

Three configurations on the same GB10, one at a time, same model files, same
flags except where named:

| Label | Engine | MTP |
|---|---|---|
| previous build | September 25 tree (upstream `38cc054f` + carried patches) | 3 |
| new build, MTP3 | this release (upstream `e7900156` + complete patch) | 3 |
| **new build, MTP2** | this release, published configuration | **2** |

All runs: one sequence, 262144 context, BF16 KV 8 GiB, Marlin W4A16 MoE,
prefix caching on (each request carries a unique `cache_salt`, so nothing is
reused between runs). Harness: [`qharness.py`](qharness.py), standard library only.

## Speed

1024 generated tokens (`min_tokens` = `max_tokens`, `ignore_eos`), thinking off,
streaming. Decode tok/s excludes the first token, which prefill produces. Two
runs per cell; the table shows the mean. Sampled requests use the model card's
thinking-mode sampling (temperature 1.0, top_p 0.95, top_k 20).

| Decode tok/s | previous, MTP3 | new, MTP3 | **new, MTP2** |
|---|---:|---:|---:|
| 32-token prompt, greedy | 27.96 | 27.49 | **28.53** |
| 32-token prompt, sampled | 24.57 | 24.35 | **26.36** |
| 30,021-token prompt, greedy | 27.09 | 27.45 | **28.25** |
| 30,021-token prompt, sampled | 22.38 | 22.43 | **24.29** |
| 30,021-token prefill tok/s (warm process) | 2,004 to 2,047 | 2,038 to 2,048 | **2,053 to 2,069** |

The first long prompt after a restart prefills at 1,180 to 1,240 tok/s while
kernels and the page cache warm; the table uses the runs after it.

MTP acceptance, new build:

| | MTP3 | MTP2 |
|---|---:|---:|
| Mean acceptance length, greedy | 2.30 to 2.39 | 2.05 to 2.16 |
| Mean acceptance length, sampled | 1.86 to 2.14 | 1.77 to 1.98 |
| Per-draft-token acceptance, sampled | 29% to 38% | 39% to 49% |

The third draft position was accepted about 10% of the time in live logs,
below the cost of the extra draft pass. MTP2 is faster in every cell.

These numbers are not directly comparable with the September 5 figures on the
front page (thinking enabled, shorter outputs, reasoning tokens included).

## Quality

Speculative decoding with the standard rejection sampler does not change the
target distribution, so MTP depth should not move quality. These checks confirm
that nothing in the rebuild did:

| Check | previous, MTP3 | new, MTP3 | new, MTP2 |
|---|---:|---:|---:|
| GSM8K first 100, greedy, thinking off | 97 | 97 | 98 |
| Prompt NLL, prose 4,096 tokens | 0.246 to 0.256 | 0.246 to 0.249 | 0.239 to 0.245 |
| Prompt NLL, prose 32,768 tokens | 0.084 to 0.086 | 0.084 to 0.086 | 0.086 |
| Prompt NLL, prose 131,072 tokens | 0.0452 to 0.0456 | 0.0451 to 0.0455 | not run |
| Prompt NLL, code 4,096 tokens | 0.647 to 0.650 | 0.646 to 0.651 | 0.649 to 0.650 |

Prompt NLL is the mean negative log-likelihood the model assigns to a fixed
text (`prompt_logprobs`), a sampling-free check of the forward pass. Ranges are
repeated identical requests. The one-question GSM8K difference is within
run-to-run variation (see below), not an improvement.

Inputs:

| File | Source | SHA-256 |
|---|---|---|
| `prose.txt` | Project Gutenberg eBook 1342, plain text | `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75` |
| `code.txt` | `vllm/v1/core/sched/scheduler.py` from the September 25 patched tree | `eaaa32726d00cb3b2f5c56c52d582e5cf7dbcad8200a7ff54243481fd23cf8a9` |
| `gsm8k.jsonl` | `openai/grade-school-math` test split | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |

## Known limit: prefill is not bit-reproducible

Scoring the same text twice moves most token log-probabilities very little, but
a few move by up to 4 to 6 nats, and mean NLL at 4K varies by up to about 4%
between identical requests. This occurs on all three configurations, from 1K
to 131K tokens. The deterministic top-k in this release (PR #55122, kernel
tests pass on GB10) did not remove it, so another prefill kernel contributes.
Batch-level averages (NLL, GSM8K) are stable. Greedy output for the same
prompt can differ between runs. Not yet isolated.

## Retrieval and prefix cache (live engine, this release)

`scripts/check-prefix.py`, thinking medium, three audited keys:

| Prompt tokens | Cold prefill s | Warm prefill s | Reused tokens | Answers |
|---:|---:|---:|---:|---|
| 19,969 | 10.13 | 1.27 | 17,600 | pass / pass |
| 249,971 | 129.61 | 1.31 | 248,000 | pass / pass |

Raw results: `results/`, `prefix/`.
