# LFM3-1.7B on the megakernel (`nano`): seeded QD archive

The first speed measurements of `nano`, the model-agnostic megakernel, on the `LiquidAI/LFM3-1.7B-MLX-4bit` checkpoint. All timings were taken by `tools/nano_qd.py trial` and live in `benchmarks/lfm3-qd/trials.jsonl` (every round of every run) and `archive.json` / `elites.md` (current elites). This note summarises them; it is not the source of truth.

## Protocol

- **Machine**: Apple M3 Ultra, 256 GB, macOS 27.0, shared with other workloads (load average about 12; other locked Flint runs and CPU builds interleaved with these). Every timed process ran under `tools/guarded_run.py` and one inference lock.
- **Binary**: `zig build -Doptimize=ReleaseSafe` (Zig 0.17.0), `zig-out/bin/nano` sha256 `6c354d2e83baf629…` (the pinned reference binary).
- **Workload**: the stock-MLX baseline's (`benchmarks/lfm3-mlx-baseline.md`): BOS then `tools/benchmark.py`'s code continuation ids, cut to exactly 64, 1,024 or 4,096 prompt tokens, then 256 teacher-forced decode tokens. `tools/nano_bench.py tokens` writes it and checks it against the baseline's recorded input.
- **One run** (`nano bench`): load (excluded), one warm-up pass (excluded), then 3 timed repeats from a reset state; the per-run figure is the median. Prefill: the whole prompt in forwards of `batch` rows, head (final norm, 128,000-wide projection, argmax) on the last row only, ending with the GPU argmax read back. Decode: one forward per token with the GPU argmax read back each token. A prediction checksum must agree across repeats.
- **One trial**: three runs of the candidate per length (rounds), interleaved with three runs of the cell's current elite when the cell is occupied; a challenger replaces an elite only if its median beats the elite's by more than max(1%, 2 x the elite's own spread). Dispatches per token are counted by the bridge.
- **Precision**: float32 activations, caches and accumulation, 4-bit weights. The MLX f32 row is the matching precision; MLX bf16 is stock's default.

## Results (medians of rounds, tokens a second)

| candidate | knobs | dispatches / decode token | decode 64 | decode 1,024 | decode 4,096 | prefill 64 | prefill 1,024 | prefill 4,096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reference` | defaults (one dispatch per op, batch 64) | 465 | 178.2 | 120.4 | 63.7 | 509.9 | 488.2 | 433.8 |
| `fused-mlp` | GLU + sandwich-norm fusion | 323 | 190.4 | 127.8 | 64.5 | 627.8 | 598.5 | 516.0 |
| `fused-layer` | all five fusions | 240 | 206.8 | 122.2 | 63.2 | 620.8 | 538.3 | 451.7 |
| `fused-layer-r4` (mutation) | fused-layer + `decode_rows 4` | 240 | 180.9 | 123.6 | 64.4 | 622.2 | 588.6 | 506.3 |
| `wide-batch` | batch 256, prefill tile 4 x 4 | 465 | 168.7 | 119.4 | 62.3 | 620.5 | 624.0 | 542.4 |
| `wide-batch-fused` (mutation) | wide-batch + all five fusions | 240 | 195.9 | 131.3 | 64.2 | 686.0 | 683.4 | 571.8 |
| stock MLX f32 | | | 189.2 | 148.3 | 157.2 | 2,575 | 4,620 | 4,518 |
| stock MLX bf16 | | | 160.1 | 144.8 | 178.3 | 2,487 | 5,007 | 5,396 |

`wide-batch` was first rejected by a driver bug (see `reports/megakernel.md`), so its mutation `wide-batch-fused` opened the wide-batch cells; the re-run of `wide-batch` then contested them. In interleaved rounds it lost all three prefill cells to `wide-batch-fused` (620.5 vs 688.4, 624.0 vs 672.3, 542.4 vs 582.1). It took the three decode cells, which have a different dispatch bin (465 rather than 240). Against the reference executor, batch 256 with a 4 x 4 tile gains 22 to 28% in prefill (separate trials, not interleaved). Its trial record says `agreed: false` because the hypothesis check compares with the contested elite, not with the reference.

`fused-layer-r4` was contested against `fused-layer` in interleaved rounds and lost every cell. Its hypothesis was +3% decode from sharing each lane's prescaled input segment across 4 rows. Measured: -13.1% at 64 tokens, -9.3% at 1,024 and -4.8% at 4,096; prefill was within ±0.3% (prefill uses `prefill_rows`, which it did not change). Halving the SIMD groups per matrix evidently costs more parallelism than the shared loads save at batch 1. The elite's own rounds in that contest were 207.4 to 209.2 (64), 135.4 to 136.5 (1,024) and 67.6 to 67.7 (4,096). The same binary and knobs measured 206.8, 122.2 and 63.2 in its own trial 25 minutes earlier, so expect run-to-run drift of up to about 10% between trials on this shared machine. That is why only interleaved comparisons decide a cell.

## What limits it (for the search)

- **Decode falls with context**: 178 to 64 tok/s from 64 to 4,096 tokens, while MLX stays at about 150 to 180. `mk_attention` gives each (row, head) one threadgroup of 8 SIMD groups that walks every cached position serially, with no split over the sequence. At 4,096 tokens the 6 full-attention layers dominate.
- **Prefill is 4 to 10 times behind MLX**, because the exactness rule makes every prefill row use the decode matvec's expression tree (MLX `qmv_fast` structure: per-lane 16-input segments, `scale * dot + sum * bias`, one `simd_sum`). Token tiles reuse each loaded weight word for up to 4 rows, but this is not a matrix multiply. A `simdgroup_matrix` GEMM would reassociate the sums. Under QD.md that is a different reference and is not admitted unless a separately documented decision changes the reference for prefill and decode together.
- **Dispatch count**: the reference issues 465 dispatches per decode token, each followed by a buffer barrier in one command buffer.

## Variant code status

Every variant compiled into `nano` today holds a cell or is an ancestor of one, except `decode_rows 4` (`mk_linear_q4_r4_*`, measured above, lost) and the 8-row and 1-row tiles (`mk_*_r8_*`, `mk_*_r1_*`, never timed). They are knob values of one templated kernel, not separate code, so nothing is removed.
