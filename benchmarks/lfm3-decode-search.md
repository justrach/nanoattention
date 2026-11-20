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

## Second climb (decode, exact tier)

Same protocol, same archive (`$NANO_WORK/qd/lfm3`), candidates from `split-attn`. Instrumentation added first: `nano bench ... --decode-detail 1` (stderr: per decode token, host encoding, `finish` to completion, command-buffer GPU time, launches, gate recomputations) and `nano d2-probe [--checkpoint DIR]` (`src/mega/decode2_probe.zig`).

### Where a synchronous token goes

`split-attn`, 64-token prompt, 256 teacher-forced tokens, one locked run (numbers vary by 10 to 20% between sessions with the machine's load):

| | us a token |
| --- | ---: |
| wall | 4,146 |
| host encoding (197 launches) | 59 |
| command-buffer GPU time | 3,281 |
| commit to GPU start | ~580 |
| GPU end to `waitUntilCompleted` returning | ~200 |

At 4,096 tokens: wall 4,232, GPU 3,675. So about 0.8 ms a token, a fifth, is neither encoding nor GPU work. `d2-probe`: an empty launch behind a barrier costs ~2.0 to 2.3 us of GPU time (1.6 without the barrier in a serial encoder, 0.22 in a concurrent one without barriers); encoding is ~0.17 us a launch; back-to-back asynchronous command buffers start 0.6 us apart; a host `MTLSharedEvent` signal takes ~620 us to start a waiting buffer. The argmax word is visible to the host about 2 us before `GPUEndTime`.

Tried and dropped, each measured at 64 tokens: spinning on the command buffer's status or on a completed-handler flag instead of `waitUntilCompleted` (no change, end to return still ~180 to 200 us); binding a Metal-allocated copy of the checkpoint instead of the mapping (no change in submission latency or matvec bandwidth; a word-aligned copy matvecs at the same speed as the 2-mod-4 mapping); committing the first steps of a token early (+2%, inside the noise).

### Candidates

| candidate | hypothesis (short) | admission | result |
| --- | --- | --- | --- |
| **`split-attn-gated`** (trial 20261120T202607, family `gated-sync`) | hide the ~0.8 ms of submission and wake latency: the next token's forward is committed while this one runs, held by a one-thread GPU gate the host opens after reading this token and writing the next (`d2_gate`, docs/MEGAKERNEL.md 4a): +20% at 64, +12% at 4,096 | bit-identical, 7 tiny x feeds 1/7/64, `short`/`window`/`long` x 1/64; plus `tools/nano_gate_check.py --long` (feed 1 with batch 64: every token gated; with and without `--spins 1`) | **306.96 / 292.08 / 273.05**, 198 launches a token, against `split-attn` 241.46 / 247.41 / 238.04 in the same trial; three new cells |
| `gated-spins1` (test only) | every gate gives up at once, so stale generations and recomputation run on every case | rejected (eval failed on `tiny/bf16` feed 1): with `batch` 1 a ring had no spare row, and the forward run on a stale token clobbered the oldest row the recomputation reads | fixed by `exec.d2GateSafe` (gating needs ring capacity >= horizon + 2); afterwards `tools/nano_gate_check.py --long --spins 1` (feed 1, batch 64, so every token is gated) is bit-identical on every case, 10 to 13 recomputations per tiny case, and the 64-token bench checksum is unchanged with 13 recomputations |
| **`split-attn-gated-nr`** (trial 20261120T203852) | the 47 sandwich-norm launches a token (~6 us each, three passes each waiting on device memory, r stored and read back) keep y, r and both weights in registers (`d2_norm_regs`, `mk_d2_norm_add_norm`): -75 us a token, +2 to 2.5% | bit-identical as above; kernel test against `mk_norm_add_norm` on signed zeros, subnormals, infinities, NaN, cancellation, ragged widths | 314.02 / **302.89** / **278.47**: replaces at 1,024 (vs 296.82) and 4,096 (vs 273.74); loses at 64 (vs 306.39, margin 7.74) |
| `d2_norm_fuse` (screen) | fold each sandwich norm into the next matvec's launch: every threadgroup recomputes both norms in threadgroup memory, the last one to finish writes r and out (-47 launches a token) | bit-identical on the tiny checkpoints | **failed**: 64 tokens 241.5 -> 203.0 (GPU 3,294 -> 4,286 us a token); ~960 threadgroups each redo a serial two-reduction chain before streaming weights. Removed |
| `d2_conv_regs` (screen) | the decode gated conv with every load up front and the current tap from a register | bit-identical (kernel test) | **failed**: GPU +40 us a token at 64 under gating. Removed |
| `linear_group 64` (screen) | 64-thread threadgroups for the decode matvecs (probe: -1 to -7% on the mid shapes) | not admitted | +0.5 to 1%, inside the noise; not pursued |

Matvec bandwidth (`d2-probe`, real checkpoint bytes, many distinct regions so the system cache does not help, 2 rows a SIMD group, 128 threads): head 128,000 x 2,048 662 GB/s; 7,680 x 2,048 430; 6,144 x 2,048 385 (410 with 64 threads); 2,048 x 7,680 369; 2,048 x 2,048 257 (launch-bound). 4 rows a SIMD group is slower on every shape but 6,144 x 2,048.

### Re-timed against stock MLX (second climb)

Final binary, `tools/nano_bench.py versus` (one locked process per engine, length and round; per process one warm-up run, then 3 repeats, median; 256 teacher-forced decode tokens with the token read back every token), 3 interleaved rounds, decode tokens a second:

| prompt | `split-attn` | `split-attn-gated` | `split-attn-gated-nr` | stock MLX f32 | stock MLX bf16 |
| ---: | --- | --- | --- | --- | --- |
| 64 | 242.6 / 248.0 / 248.3 -> **248.0** | 303.6 / 303.2 / 305.9 -> **303.6** | 308.1 / 309.7 / 304.7 -> **308.1** | 185.8 / 189.0 / 187.2 -> 187.2 | 196.9 / 194.9 / 192.5 -> 194.9 |
| 1,024 | 244.3 / 237.3 / 246.6 -> **244.3** | 291.2 / 295.6 / 295.5 -> **295.5** | 287.1 / 303.2 / 302.0 -> **302.0** | 187.7 / 185.3 / 190.1 -> 187.7 | 193.9 / 195.7 / 195.5 -> 195.5 |
| 4,096 | 239.5 / 238.5 / 239.6 -> **239.5** | 273.8 / 274.1 / 274.6 -> **274.1** | 279.1 / 279.2 / 279.7 -> **279.2** | 185.9 / 157.0 / 184.1 -> 184.1 | 166.9 / 193.8 / 194.4 -> 193.8 |

The gated candidates run the reference's arithmetic with the token read back every token; they differ from `split-attn` only in when the next forward's commands are encoded and committed (docs/MEGAKERNEL.md 4a). Stock MLX's loop here is also synchronous (`.item()` every token, no `async_eval`).

### Against stock MLX's pipelined loop

The tables above time stock MLX with a synchronous read per token. Stock's own generate loop does not work that way: it builds the next step from the lazy argmax and submits it with `mx.async_eval` before reading the current token, which hides the same latency the gate hides. `tools/lfm3_benchmark.py` measures that loop as engines `mlx_f32_async` and `mlx_bf16_async` (free-running greedy, argmax only, the same number of forwards). One locked session on the merged tree (`tools/nano_bench.py versus`, three interleaved rounds, medians, decode tokens a second):

| prompt | presets (gated decode): `exact` / `mlx-prefill` | `nano`, plain synchronous | stock f32 / bf16, pipelined | stock f32 / bf16, synchronous |
| ---: | --- | ---: | --- | --- |
| 64 | 310.0 / 314.1 | 241.5 | 280.0 / 287.1 | 189.3 / 196.0 |
| 1,024 | 301.1 / 302.8 | 249.0 | 275.4 / 285.8 | 187.8 / 196.5 |
| 4,096 | 276.5 / 280.1 | 235.1 | 265.3 / 274.6 | 181.2 / 193.4 |

So the like-for-like margins are: gated against stock's pipelined bf16 loop +8.0% / +5.4% / +0.7% (the last within drift), against pipelined f32 +10.7% / +9.4% / +4.2%; plain synchronous against synchronous bf16 +23% / +27% / +22%. Before this round `nano`'s synchronous decode (about 240) was behind stock's pipelined loop.

### Variant code status (second climb)

Kept: `d2_gate`, `d2_gate_spins` (`mk_d2_gate_packed`, `mk_d2_embed_gated`), `d2_norm_regs` (`mk_d2_norm_add_norm`), `mk_d2_empty` (probe), the bridge's `fm_d2_*` block (`src/metal.m`). Removed with their measurements above: `d2_wait`, `d2_copy_weights`, `d2_split`, the read-modify-write gate, `d2_norm_fuse` (`mk_d2n_*`), `d2_conv_regs`. Not attempted: the pipelined regime (GPU-to-GPU token feed); the gated decode already removes the host latency inside the synchronous protocol, and a pipelined comparison needs a free-running greedy bench mode and an `async_eval` baseline that do not exist yet.
