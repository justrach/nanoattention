# LFM3-1.7B on the megakernel: the decode search

The decode half of the quality-diversity search (`docs/QD.md`) for LFM3-1.7B, 4-bit, on `nano`. The source of truth for every number here is the archive's trials (`benchmarks/lfm3-qd/trials.jsonl`). Decode stays in the exact tier: every admitted candidate's logits are bit-identical to the pinned reference executor's.

## Protocol

As `benchmarks/lfm3-megakernel.md`: Apple M3 Ultra (shared, load average about 12), every timed process under `tools/guarded_run.py` and the inference lock; workload = the stock-MLX baseline's (BOS + code continuation cut to 64, 1,024 or 4,096 prompt tokens, then 256 teacher-forced decode tokens); one run = load, warm-up, 3 repeats from reset, median; decode reads the GPU argmax back every token (synchronous, as the MLX baseline). A trial interleaves at least 3 rounds of challenger and cell elite; a challenger wins only by more than max(1%, twice the elite's own spread). Screens (`nano_qd.py screen`) are 2-round interleaved comparisons used to choose what to trial; they never take a cell.

## Where it started

| | decode 64 | 1,024 | 4,096 |
| --- | ---: | ---: | ---: |
| `reference` (465 dispatches a token) | 178.2 | 120.4 | 63.7 |
| `fused-mlp` (323) | 190.4 | 127.8 | 64.5 |
| `fused-layer` (240) | 206.8 | 122.2 | 63.2 |
| stock MLX f32 / bf16 (baseline file) | 189.2 / 160.1 | 148.3 / 144.8 | 157.2 / 178.3 |

Decode fell from 178 to 64 tok/s between 64 and 4,096 prompt tokens while stock MLX stayed at 150 to 180.

**Profile** (`fused-layer` at 4,096 tokens, `nano bench --profile-decode 1`, attribution only): `mk_attention` was 11.0 of about 19 ms per token. Each (row, head) gets one threadgroup of 8 SIMD groups that walks every cached position serially: about 3.5 us per step of the online softmax, 512 steps per SIMD group in a full-attention layer, so 1.7 to 2.1 ms per full layer (6 of them) and about 117 us per sliding layer. At 64 tokens attention was still 11 x 55 us. The rest at 64 tokens: 4-bit linears about 2.7 ms (about 440 GB/s), sandwich norms 47 x 9 us, ropes, stores and q/k norms 22 x (6 + 4.6 + 7) us, gated conv 13 x 12.5 us, argmax 40 us.

## Candidates

| candidate | hypothesis (short) | admission | result |
| --- | --- | --- | --- |
| `fused-layer-r4` (builder) | 4 output rows per SIMD group at batch 1 shares each prescaled input segment: +3% | bit-identical | lost every cell: -13.1% (64), -9.3% (1,024), -4.8% (4,096) against `fused-layer` interleaved; halving SIMD groups per matrix costs more parallelism than the shared loads save |
| `wide-batch` (builder) | prefill objective; holds the decode cells of the 465-dispatch bin | bit-identical | 168.7 / 119.4 / 62.3 (its own mechanism cells) |
| `decode_attention = 1` (screen 0) | split single-row attention: a parallel launch computes every position's q.k with `mk_attention`'s lane ownership, fma order and `simd_sum`; a chain reproduces each SIMD group's online-softmax recurrence over the stored scores in the same order | bit-identical on the tiny checkpoints, headless test against `mk_attention` | 4,096: 63.8 -> 148.1 (screen); stepping stone, removed after `= 2` replaced it |
| + `fuse_attn_prep` (screen 0) | q/k per-head RMSNorm, both ropes and the K/V stores of an attention layer in one launch (-5 dispatches a layer) | bit-identical, headless test against the 6 reference dispatches | 148.1 -> 160.2 at 4,096 (196 dispatches) |
| `decode_attention = 2` (screens 1, 2) | the chain's two dependent `precise::exp` per step dominate: compute the prefix max, factor f and exponentials 32 steps at a time (max is exact in any association), same recurrence | bit-identical | 160.8 -> 174.0 at 4,096; 213.3 -> 223.6 at 1,024 |
| chain tuning (screens 3 to 6) | V prefetch, unrolling, pipelined loads | bit-identical | kernel profile 200 -> 120 -> 70 us per full layer at 4,096; screens against `fused-layer`: 223.4 vs 65.0 (4,096), 238.5 vs 128.0 (1,024), 240.0 vs 199.5 (64) |
| `decode_argmax = 1` | two-stage argmax (64 slices per row): the result is an integer, order-independent | equal to `mk_argmax` including ties, signed zeros, -inf rows, non-finite sentinel | part of `split-attn` |
| **`split-attn`** (trial 20261120T160300) | fused-layer + `decode_attention 2` + `fuse_attn_prep` + `decode_argmax 1`: decode at 4,096 from about 65 to above 200, at 1,024 from about 128 to about 230, +15% at 64 | bit-identical on 7 tiny checkpoints x feeds 1/7/64 and `short`/`window`/`long` x feeds 1/64; empirical (the Lean port predates the new kernels) | **239.75 / 238.29 / 224.16**, 197 dispatches a token; new cells `decode|*|<=256|split-attn` |
| x-resident matvec, unrolled matvec (`nano microbench`) | keep the input vector resident across rows; unroll | bit-checked against `mk_linear_q4_r1_t1` | x-resident 2x slower, unrolled +3 to 5% on large shapes only; not wired, removed |

## Re-timed against stock MLX

On the cleaned merged binary, in the same locked runs: `split-attn` 239.7 / 239.5 / 224.2 tok/s at 64 / 1,024 / 4,096 against stock MLX bf16 181.1 / 178.9 / 171.9 and f32 177.2 / 173.3 / 166.0. After the prefill work of this round, `split-attn`'s decode kernels are unchanged (the prefill candidates carry its decode knobs); their decode rates in the final same-session runs with stock MLX are in `benchmarks/lfm3-prefill-search.md`.

## What limits decode now

- One forward a token with a synchronous host argmax read-back, like the MLX baseline. 197 dispatches per token, each followed by a buffer barrier; at 64 tokens the 4-bit matvecs (about 440 GB/s on 0.97 GB of weights) and dispatch overhead dominate.
- At 4,096 tokens the six full-attention layers still cost about 70 us each per token, which is the gap between 239 and 224.
- No pipelining of consecutive tokens (`async_eval`-style) and no speculative decoding: excluded so the headline comparison stays synchronous on both sides.

## Variant code status

Removed, with their measurements above: `decode_attention = 1` (per-step chain), the x-resident and unrolled single-row matvecs, an unwired in_proj + conv kernel. `decode_rows 4` and the 8-row and 1-row tiles of the reference linear remain knob values of one templated kernel (see `benchmarks/lfm3-megakernel.md`).
