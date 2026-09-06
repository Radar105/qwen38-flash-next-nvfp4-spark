# Credits and sources

- [Qwen/Alibaba](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): base model and architecture.
- [NVIDIA model artifact](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fab0aecb760cec45227f6656abcaafa11abca87a): original mixed-precision weights.
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer): quantization tooling.
- [vLLM](https://github.com/vllm-project/vllm/tree/7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719): engine and pinned source base.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers): runtime components.
- [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-8-flash-next): Intelligence Index v4.2 data, retrieved 2026-09-06 UTC (September5 MDT).

## Upstream patch authors

| PR | Author | Change | Status when retrieved |
|---|---|---|---|
| [#53798](https://github.com/vllm-project/vllm/pull/53798) | [ptorsten](https://github.com/ptorsten) | [Bugfix] Seed align-mode Mamba state_idx in Mamba blocks | open, local backport |
| [#54076](https://github.com/vllm-project/vllm/pull/54076) | [wickist](https://github.com/wickist) | [Bugfix][V1] Use the Mamba cache group's block size for align-mode chunk splitting | open, local backport |
| [#55390](https://github.com/vllm-project/vllm/pull/55390) | [Navjot10](https://github.com/Navjot10) | [Bugfix] Annotate MTP draft KV cache groups positionally on the hybrid grouping path | open, local backport |
| [#54713](https://github.com/vllm-project/vllm/pull/54713) | [tobymao](https://github.com/tobymao) | [BugFix] Mamba sparse retention keeps a state below each boundary under EAGLE/MTP | open, local backport |
| [#55375](https://github.com/vllm-project/vllm/pull/55375) | [peakcrosser7](https://github.com/peakcrosser7) | [Bugfix][Qwen4Exp] fix state index strides in fused PLE conv | merged |

Exact PR head revisions are in `provenance/upstream-credits.json`. The complete patch includes local context adaptations, UniformTypeKVCacheSpecs unwrapping, tests, the opt-in state-index audit, the existing PLE mmap reader, ModelOpt MTP layer remapping and block-FP8 expert dispatch. Apply the complete patch once; do not stack the individual PRs on top.

Local adaptations are documented separately from upstream contributions.

The banner depicts an illustrative compact computer. It is not a product photograph. Charts were generated with Matplotlib from retained measurements. The earlier charts are preserved as historical results, with anonymous paths for publication.

[Community single-Spark experiments](https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark) informed questions during research; no community implementation is included. The deployed cache fixes come from the official vLLM repository.
