# Long-context attention hill climbing

> **Historical, rejected performance candidate:** the chunk-state speed figures
> below describe the earlier reassociated state-merge implementation, not the
> current ordered or MLX-layout paths. They are not accuracy-admitted
> performance results. `--attention-chunk` now uses a score producer followed by
> the original ordered value reduction; its strict 4K sequential-MLX check passed,
> but 8K failed. The separate default-off `--mlx-attention` mode now passes
> bit-exact checks at both contexts and reaches 126.94/119.72/110.75 tok/s at
> 1K/4K/8K. See [current accepted measurements](mlx-layout-long-context.md).

These measurements use **actual 1,024 / 4,096 / 8,192-token code prefixes**, not larger allocation limits with short inputs or fabricated cache positions. Prefixes are deterministic repeated-code token streams produced by `benchmark.code_prompt`; they are performance/numerical workloads, not task-quality evaluations or Recursive's exact benchmark prompt.

## Protocol

- M3 Ultra, 60 GPU cores, 256 GB unified memory; ReleaseSafe; the existing affine-INT4 Flint export with FP32 native activations/accumulators/KV.
- Baseline and coarse tuning: 64 identical teacher-forced decode inputs, three warmed repeats, GPU completion plus synchronous CPU greedy readback per token.
- `--reuse-prefill` computes the **entire real prefix once**, measures that prefill, and copies its actual KV cache and position. Restoration and warmup are outside decode timing. Native prefill throughput therefore has **one sample**, not three independently measured prefills. No cache values or positions are invented.
- The chunk sweep restores the same real **unsplit baseline** prefix for every candidate. This isolates decode scheduling; it does not establish correctness of optimized prefill. Candidate admission requires exact per-position predictions across repeats and against chunk 0, plus more than 3% throughput improvement.
- Separate confirmation uses an optimized real prefill and separate all-vocabulary-logit comparisons after that prefix. Long-context logit checks cover the last prefill row and cached continuation rows, not every earlier prefill position. Existing allclose tolerances remain unchanged.
- Stock MLX executes the same real token IDs, prefilling fresh caches in bounded chunks; its full model never overlaps native inference. Prefill methods differ and are explicitly labeled.

## Unsplit baseline

| Actual prompt tokens | Native decode tok/s | MLX FP32 tok/s | MLX BF16 tok/s |
| --- | ---: | ---: | ---: |
| 1,024 | 50.48 | 83.60 | 78.09 |
| 4,096 | 27.94 | 82.83 | 87.86 |
| 8,192 | 17.40 | 77.36 | 86.02 |

These baseline figures expose the limitation hidden by the earlier 21-token prompt: eight SIMD partitions within only one threadgroup per query head serialize substantial attention work as context grows.

## Implementation

The new staged path splits effective attention windows into independent **128 / 256 / 512 / 1,024-token chunks**, each with eight cooperating SIMD groups. Producers write unnormalized online-softmax states; a separately ordered merge dispatch combines states and normalizes once. The learned sink belongs to exactly chunk 0 / SIMD 0. The active sliding-window start, ring slots, GQA mapping, and dtype/offset semantics are preserved. Effective lengths below 512 retain the existing path; `--attention-chunk 0` selects the unsplit baseline.

Lean-generated constants bind admitted chunk sizes and geometry. Integer certificates cover token/state ownership, guards, sink ownership, index bounds, and no-wrap conditions—not floating-point, Metal memory ordering, or speed. Independent FP64 softmax tests exercise boundaries, ring wrap, and lengths through 8K at unchanged 1e-5 tolerance.

## Coarse shared-prefix sweep

128-token chunks won at each prefix length; all candidates retained exact baseline/repeat per-position predictions. These figures use the same 64-token decode budget, and the speedups compare candidates inside the same shared-prefix process.

| Prompt tokens | Unsplit tok/s | Chunk 128 tok/s | Speedup |
| --- | ---: | ---: | ---: |
| 1,024 | 50.90 | 98.87 | 1.94x |
| 4,096 | 27.96 | 80.02 | 2.86x |
| 8,192 | 17.44 | 62.45 | 3.58x |

## Fresh optimized-prefix confirmation

These are separate **256-token** decode measurements with real optimized prefills and three warmed repeats; do not calculate an apples-to-apples speedup by dividing them by the earlier 64-token baseline.

| Prompt tokens | Native tok/s | Stock MLX FP32 tok/s | Stock MLX BF16 tok/s |
| --- | ---: | ---: | ---: |
| 1,024 | 95.85 | 84.05 | 82.40 |
| 4,096 | 78.70 | 80.10 | 87.49 |
| 8,192 | 66.50 | 76.81 | 83.24 |

## Accuracy admission remains incomplete

Split attention passes independent FP64 kernel checks and all 64 shared-prefix predicted tokens at every context/chunk. Full-model **strict all-logit MLX parity is not established at 4K/8K**, even though all 16 greedy continuation tokens match. Tolerances were not relaxed. With matched sequential MLX prefill, the observed checks are:

| Prompt tokens | All-logit check | Greedy agreement | Maximum absolute logit error |
| --- | --- | ---: | ---: |
| 1,024 | PASS | 16/16 | 0.013748 |
| 4,096 | FAIL | 16/16 | 0.249615 |
| 8,192 | FAIL | 16/16 | 0.461035 |

A 1K diagnostic also found stock MLX chunk-256 versus stock sequential prefill differs by up to 0.2751 in logits (and fails the same allclose criterion). The old unsplit native path already exhibits the chunked-reference mismatch. This does not resolve the 4K/8K native-versus-sequential discrepancies. A RoPE phase-arithmetic experiment did not resolve them and was fully reverted.

**The new path remains experimental and disabled by default (`--attention-chunk 0`).** Select `--attention-chunk 128` explicitly for experiments. No production accuracy, broad task-quality parity, matched `rec` win, or all-green MLX numerical claim is made. Numerical parity investigation remains an open acceptance item.

[Sanitized numerical report](long-context.json) retains individual timings, all candidate rates, pass/fail statuses and source fingerprints. Raw local outputs remain in the ignored `.zig-cache/` folder.
