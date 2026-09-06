# NVIDIA reported precision results

NVIDIA reports near-parity with the FP8 baseline across nine benchmarks. NVFP4 is higher on five and lower on four; the largest decrease is0.7 points.

| Benchmark | FP8 | NVFP4 | Difference |
|---|---:|---:|---:|
| GPQA Diamond | 92.0 | 91.5 | -0.5 |
| HLE | 34.7 | 35.4 | +0.7 |
| τ²-Bench Telecom | 90.8 | 90.1 | -0.7 |
| MMMU Pro | 77.1 | 78.3 | +1.2 |
| SciCode | 16.3 | 18.8 | +2.5 |
| AA-LCR | 71.9 | 74.1 | +2.2 |
| IFBench | 80.5 | 81.0 | +0.5 |
| Omniscience | 28.1 | 27.6 | -0.5 |
| Terminal-Bench 2.1 | 83.3 | 82.9 | -0.4 |

NVIDIA used xhigh reasoning, temperature1.0, top_p0.95 and max_new_tokens131072. This supports near-parity on the reported tests.

[Official pinned NVIDIA model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fab0aecb760cec45227f6656abcaafa11abca87a/README.md).
