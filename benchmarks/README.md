# Benchmark evidence

[`local-flint-int4.json`](local-flint-int4.json) records a fresh build and rerun from this standalone repository folder. It contains numerical results and methodology, without model weights, raw local logs, or machine paths.

## Standalone rerun

Apple M3 Ultra, 60 GPU cores, 256 GB unified memory; Zig 0.17.0 ReleaseSafe, Lean 4.35.0; the existing 22.57 GB affine-INT4 Flint export with FP32 native activations and KV cache.

| Engine | Median decode tokens/s |
| --- | ---: |
| nanoattention / Zig FP32 | **120.40** |
| Stock MLX FP32 | 96.84 |
| Stock MLX BF16 | 91.96 |

Native runs: **120.18 / 120.40 / 121.12 tok/s**. Each processes the same 21-token coding prompt and 64 teacher-forced decode inputs, with three warmed repetitions. The runner waits for GPU completion and reads a greedy prediction on the CPU at each decode step. Loading, cache reset/creation, dtype conversion, and warmup are excluded. Native execution uses one submission and 489 dispatches per token. Prefill is serial natively but batched in MLX; it is not an equivalent-operation prefill comparison.

The standalone build passed 42 unit/kernel tests, six generated-binding tests, six proof-verifier tests, and seven independent NumPy synthetic checkpoint cases. All native source and Lean proof files are byte-identical to the pre-extraction version whose full-model, all-vocabulary-logit parity checks are summarized in the JSON. Those parity checks were not rerun for the extraction; the native build, tests, synthetic parity, and performance comparison were rerun in this folder.

## Recursive's published numbers

The [`rec` README](https://github.com/recursive-oss/rec) and [throughput chart](https://github.com/recursive-oss/rec/blob/main/.github/assets/throughput.svg) report Flint on **M4 Pro / 48 GB / 4-bit weights**, from **131.2 tok/s at 1K context** to **22.6 tok/s at 252K**, with five runs, 1,024 output steps and prefill excluded. Those are not matched to this M3 Ultra, short-context, teacher-forced benchmark. This repository does **not** claim to beat `rec`.

The local MLX comparison establishes this measured workload only. It does not establish long-context or server throughput, broad task quality, bitwise equality, or overall MLX feature/performance parity. The runtime still lacks batched prefill, and MLX prefills substantially faster here. Reproduction commands are in the main [README](../README.md).
