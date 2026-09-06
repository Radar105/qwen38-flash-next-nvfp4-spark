# Qwen3.8 Flash Next : working prefix cache

Qualified and promoted on 2026-09-05. This addendum supersedes the cache-OFF
status in the earlier deployment atlas. Original checkpoint and mixed NVIDIA
quantization retained; 262144 context, 8 GiB BF16 KV, MTP2, eager TP1, one
sequence, batch2048, thinking medium, server output ceiling131072.

## Measured results

| Test | Input | Reused tokens | Cold prefill | Warm prefill | Speedup |
|---|---:|---:|---:|---:|---:|
| 20K | 19,984 | 17,600 | 13.224s | 1.390s | 9.5× |
| 64K | 63,989 | 60,800 | 32.412s | 1.721s | 18.8× |
| 250K | 249,986 | 248,000 | 157.903s | 1.450s | 108.9× |
| 20K managed | 19,984 | 17,600 | 13.493s | 1.398s | 9.7× |

Single sequential requests on the local API. Cold controls use independent
cache_salt namespaces, not a second engine or a cache flush. Output lengths
vary, so the comparison uses server prefill time rather than whole-response
wall time. First cold request includes first-use effects. This is a prefix
prefill improvement; it is not a decode t/s claim or a general agent benchmark.

## Repairs and provenance

All four patches were fetched from the official vllm-project/vllm repository.
They were open/unmerged when retrieved; this remains a locally patched build.

- [PR53798](https://github.com/vllm-project/vllm/pull/53798): bind actual recurrent cache specs and restore state using their block size.
- [PR54076](https://github.com/vllm-project/vllm/pull/54076): materialize each crossed recurrent checkpoint. Local adaptation unwraps UniformTypeKVCacheSpecs; regression added.
- [PR55390](https://github.com/vllm-project/vllm/pull/55390): identify the trailing hybrid MTP group. Its warning alone did not explain in-GPU zero hits.
- [PR54713](https://github.com/vllm-project/vllm/pull/54713): retain the lower checkpoint reachable after speculative lookup drops a block. Context-only adaptation preserves newer local arguments.

Production uses `--enable-prefix-caching --mamba-cache-mode align` and
`--prefix-cache-retention-interval 1600`.
The observed recurrent grid is1600 tokens; the separate QSA scratch ring is8.
Dense checkpoint retention is contained within the fixed8GiB cache pool. It
changes eviction competition, not that allocation. Eviction, earlier prompt
edits, or a changed system/tool prefix can still cause misses; full-context
capacity does not promise multiple complete cached conversations.

## Evidence and limits

201 core/worker tests passed; two PP2 platform cases were unavailable on this
single GPU. Another186 cache/scheduler tests and source pre-commit passed.
Raw qualification covers20K,64K,250K repeats, changed suffixes, return to a
shorter prompt, interleaved requests, automatic tool calls and tool results.
Managed qualification repeats the20K suite. Both paths additionally pass
260834-token retrieval and feature checks. Managed argv parity, actual TUI warm
action and actual streaming OpenCode tool execution are checked separately.

Earlier failed attempts are preserved: three patches still produced0 hits;
four patches with sparse retention0 produced hits but missed a shorter branch.
The final retained-checkpoint configuration passes those checks. Correct
answers are not bitwise hidden-state proof or an exhaustive workload test.
No extended production soak, MTP3 result, or full131072-token generated output
is claimed. The output limit shares the remaining context with the input.

vLLM0.28.1rc1.dev442+g7fbd44cbe.d20260905, source base
7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719 plus the documented prior PLE/MTP
fixes and these cache patches. Original model revision
fab0aecb760cec45227f6656abcaafa11abca87a.

## Captured managed command

All home paths are generalized. Reuse the environment block from the deployment atlas (including
MAX_JOBS=1 and VLLM_QWEN4_PLE_MMAP=1). Do not start a second large engine.

```bash
/home/user/venvs/vllm/bin/python -m vllm.entrypoints.openai.api_server --model /home/user/models/nvidia-Qwen3.8-Flash-Next-NVFP4 --served-model-name nvidia/Qwen3.8-Flash-Next-NVFP4 --host 0.0.0.0 --port 8092 --dtype bfloat16 --kv-cache-dtype auto --max-model-len 262144 --max-num-batched-tokens 2048 --max-num-seqs 1 --gpu-memory-utilization 0.8 --quantization modelopt --kv-cache-memory-bytes 8G --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --limit-mm-per-prompt '{"image":1,"video":0}' --enable-prefix-caching --mamba-cache-mode align --prefix-cache-retention-interval 1600 --skip-mm-profiling --mm-processor-kwargs '{"max_pixels":1048576}' --tensor-parallel-size 1 --enforce-eager --load-format safetensors --safetensors-load-strategy lazy --generation-config vllm --no-enable-flashinfer-autotune --override-generation-config '{"max_new_tokens":131072}'
```
