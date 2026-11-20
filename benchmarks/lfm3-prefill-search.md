# LFM3-1.7B on the megakernel: the prefill search, both tiers

Prompt ingestion (prefill) of LFM3-1.7B, 4-bit, on `nano`: every candidate tried, its hypothesis, admission and timings, in the exact tier and in the `mlx-arith` tier (`docs/QD.md`), with stock MLX in the same locked sessions. The source of truth is `benchmarks/lfm3-qd/trials.jsonl` (every round of every trial) and `archive.json`; screens were single runs and are quoted here only. `docs/MLX-PREFILL.md` says what stock MLX does and what the `mlx-arith` kernels reproduce.

## Protocol

- **Machine**: Apple M3 Ultra (`applegpu_g15d`), 256 GB, macOS 27.0, shared with other workloads (an idle Flint session in the background; other locked runs interleaved). Every timed process ran under `tools/guarded_run.py` and one inference lock.
- **Workload**: the stock-MLX baseline's: BOS then `tools/benchmark.py`'s code continuation ids, cut to exactly 64, 1,024 or 4,096 prompt tokens (`target/nano-bench-tokens.json`, checked against `benchmarks/lfm3-mlx-baseline.json`'s input).
- **One `nano bench` run**: load (excluded), one warm-up pass, 3 timed repeats from a reset state, median. Prefill = all prompt tokens in forwards of `batch` rows, logits and GPU argmax for the last row only, read back. Prefill-only trials and screens pass `--decode-tokens 0`.
- **Trial** (`tools/nano_qd.py trial`): admission, then 3 rounds per length interleaved with the contested cell's elite (and with `--compare` candidates, recorded only); a challenger replaces an elite only by more than max(1%, twice the elite's own spread).
- **Screen** (`nano_qd.py screen`): 2 or 3 interleaved rounds at one length, for choosing what to trial; never takes a cell.
- **Versus** (`tools/nano_bench.py versus`): per length and round, one `nano bench` per candidate then one `tools/lfm3_benchmark.py` process per stock MLX engine (mlx 0.32.4, mlx-vlm 0.8.0; stock `generate`'s prefill: chunks of 2,048 over the first L - 1 tokens, then the last token with logits and a host argmax; warm-up plus 3 runs, median), each process under the lock. This is the head-to-head with stock MLX.
- **Precision**: float32 activations and accumulation, 4-bit weights. `mlx_f32` is stock MLX at the same precision; `mlx_bf16` is stock's default dtype.
- **Admission, exact**: the binary's reference executor reproduces the pin; the candidate's all-vocabulary logits (sha256), top-1 and 64 greedy tokens are bit-identical to the pin on 7 tiny checkpoints (feeds 1, 7, batch) and on `short`, `window`, `long` (feeds 1, 100, 255, batch).
- **Admission, mlx-arith**: kernels bit-identical to the stock MLX operations they port (`tools/nano_mlx_ops.py check`, attached to the trial); end to end, on every tiny checkpoint and on `short` (23 tokens), `window` (700) and `long` (4,096) at feeds 100, 255, 1,024 and the batch: logits and residual streams within the golden's gate (2 x float32 noise floor + the golden producer's own error), top-1 at every stored position, greedy tokens identical; the largest difference from the exact batched path is recorded.

## Final head-to-head (same locked session)

Run `target/pm2/versus-final.json` (2026-11-20, binary `zig-out/bin/nano` sha256 `23249eb4…`, the cleaned binary both trials below admitted): 3 rounds per length, each round one process per engine in this order, every process under the lock, 256 decode tokens after the prompt. Median of rounds (rounds in brackets), tokens a second.

| engine | arithmetic | prefill 64 | prefill 1,024 | prefill 4,096 | decode 64 | decode 1,024 | decode 4,096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill-mm-trim` (exact tier) | bit-identical to the reference | 1,215.8 [1,215.8 / 1,216.2 / 1,210.0] | 1,277.9 [1,276.9 / 1,278.2 / 1,277.9] | 990.8 [991.0 / 990.7 / 990.8] | 240.8 | 237.5 | 219.7 |
| `mlx-prefill-trim` (mlx-arith tier) | stock MLX's batched kernels, goldens' gate | **3,478.3** [3,478.3 / 3,468.1 / 3,519.8] | **4,795.2** [4,795.2 / 4,792.4 / 4,798.6] | **4,749.0** [4,745.3 / 4,752.6 / 4,749.0] | 240.8 | 237.2 | 223.0 |
| stock MLX f32 | float32, as the goldens | 2,523.5 [2,517.8 / 2,542.3 / 2,523.5] | 4,569.3 [4,569.3 / 4,559.5 / 4,570.7] | 4,596.0 [4,593.9 / 4,604.0 / 4,596.0] | 166.3 | 166.2 | 161.0 |
| stock MLX bf16 | stock default | 2,679.1 [2,695.8 / 2,679.1 / 2,595.0] | 5,292.6 [5,292.6 / 5,312.3 / 5,288.4] | 5,412.9 [5,412.9 / 5,414.2 / 5,410.6] | 172.1 | 170.9 | 167.4 |

- `mlx-arith` against stock MLX f32: **+37.8% / +4.9% / +3.3%**; against bf16: +29.8% / -9.4% / -12.3%.
- exact against stock MLX f32: 0.48 / 0.28 / 0.22 of it. `mlx-arith` is 2.9 / 3.8 / 4.8 times the exact tier.
- Decode in both rows is `split-attn`'s (the prefill candidates carry its decode knobs; `split-attn` re-admitted bit for bit on this binary, trial `admitted-untimed`). Its archive cell (239.7 / 239.5 / 224.2) and these rounds agree within the run-to-run drift of this machine (the 4,096 rounds were 224.1 / 219.7 / 218.1 and 223.0 / 211.1 / 223.6). Decode is 1.33 to 1.45 times stock MLX's synchronous decode at every length.
- Before this round (archive at the start): exact `prefill-mm` 1,096.9 / 1,144.2 / 889.5; before the climb `wide-batch-fused` 686 / 683 / 572.

## Every candidate

### Exact tier (bit-identical to the reference executor)

| candidate | hypothesis (short) | admission | prefill tok/s 64 / 1,024 / 4,096 | verdict |
| --- | --- | --- | --- | --- |
| `reference` | the seed: one dispatch per op, batch 64 | pinned | 509.9 / 488.2 / 433.8 | elite (per-op) |
| `fused-mlp` | GLU and sandwich-norm fusion | bit-identical | 627.8 / 598.5 / 516.0 | elite (fused-mlp) |
| `fused-layer` | every fusion | bit-identical | 620.8 / 538.3 / 451.7 | elite (fused-layer) |
| `wide-batch` | batch 256, 4 x 4 row x token tile | bit-identical | 620.5 / 624.0 / 542.4 | lost its prefill cells to `wide-batch-fused` |
| `wide-batch-fused` | + every fusion | bit-identical | 686.0 / 683.4 / 571.8 | elite (wide-batch) |
| `prefill_mm` tiles (screens, batch 64, L 1,024) | exact matrix-matrix linear: `mk_q4_prep` computes each (token, 16-input segment)'s reference xsum and prescaled inputs once; a SIMD group unpacks R rows' codes once and applies them to TT tokens | tiny bit-identical | 4 x 8 half 777, 4 x 4 774, 4 x 16 517, 2 x 8 1,026, 1 x 8 571, 1 x 16 631, 1 x 32 375, 2 x 16 794 (fused-mlp 545 to 588) | 2 x 8 half kept; smaller R wins (register pressure) |
| `prefill_attention` P = 8 (screen) | batched exact attention, K/V loaded once for P (row, head) pairs | headless bit-identical | 996 vs 1,019 at L 1,024 | neutral; removed |
| `mm_stage`, `mm_lazy`, `mm_aligned` | token tile staged in threadgroup memory; codes kept packed; word-aligned weight copy | tiny bit-identical | screened, none beat 2 x 8 half | removed |
| `prefill-mm` (trial 20261120T163201) | split-attn's knobs + batch 1,024 + `prefill_mm` 2 x 8 half: linears were ~92% of prefill at ~1 TMAC/s because every SIMD group re-loaded and re-prescaled its tokens' inputs | bit-identical (7 tiny + short/window/long at 1/7/100/255/1,024) | 1,096.9 / 1,144.2 / 889.5 | elite, then replaced |
| `conv_rows` (screen 4) | the fused gated conv walks 1,024 rows serially per channel; store all rows, then convolve every (row, channel) in parallel, same tap order | tiny feeds 1/7/64 bitwise; kernel test | 1,173 vs 1,136 at 1,024 (+3%), 912 vs 884 at 4,096 | in `prefill-mm-trim` |
| `head_trim` (screen, prefill-mm + conv_rows) | rows without logits stop after the last persistent store (layer 23's conv store): layer 23's conv output, out_proj and FFN, ~3.5% of linear MACs, run for the last row only, as stock MLX's lazy chunk evaluation does | logits sha equal to the pin on short 23, window 700/100, long 1,024 | 1,282 vs 1,241 at 1,024 (+3.3%) | in `prefill-mm-trim` |
| **`prefill-mm-trim`** (trial 20261120T173321) | prefill-mm + conv_rows + head_trim, expect +6% | bit-identical, cleaned binary | **1,229.3 / 1,287.1 / 996.1** vs prefill-mm 1,143.6 / 1,205.3 / 941.2 interleaved | **replaces-elite** in all 3 prefill-mm cells |
| `inexact_mma` (NOT ADMISSIBLE) | naive MLX-style matrix-unit kernel, reassociates every dot product, to measure the cost of exactness | rejected by design | 1,730 at L 1,024, batch 64 | removed (superseded by the mlx-arith tier) |

### mlx-arith tier (bit-identical to stock MLX's kernels; goldens' gate end to end)

| candidate | hypothesis (short) | admission (largest logits difference from the exact batched path) | prefill tok/s 64 / 1,024 / 4,096 | verdict |
| --- | --- | --- | --- | --- |
| `mlx_linear` alone (screen 4, before the codegen fixes) | linears through the port of `affine_qmm_t` (32 x 32 x 32 simdgroup-matrix tiles, each weight dequantized once per 32 tokens) | kernel bit-identical to `mx.quantized_matmul` | 3,152 at 1,024 (batch 2,048), 1,802 at 4,096 (exact attention dominates) | stepping stone |
| `mlx-prefill` (trial 20261120T170304) | + `mlx_attention` (port of steel attention), batch 2,048 (stock's chunk) | PASS everywhere; vs exact up to 1.26e-4 (window) | 2,963.1 / 4,205.4 / 4,229.7 (beside prefill-mm 1,147.5 / 1,199.5 / 940.4) | elite (mlx-prefill), measured before the fixes below |
| codegen of the qmm port (`nano_mlx_ops.py speed`, per op at M 1,024) | it ran 1.13 to 1.24 x MLX's own time on identical tiles: a run-time dtype switch in the scale/bias load, `uint64` group indexing, bounds checks in full tiles | every step re-checked bit-identical | port / MLX: 1.13 -> 1.05 (scale dtype at compile time) -> 1.01-1.04 (loader-style pointers) -> -1% (FULL-tile loop); neutral: drop max-threads attribute, byte code loads, serpentine MMA order | kept: `mk_mlx_qmm_t_{f32,f16,bf16}_g*` |
| sliding-window key origin (fix, not speed) | stock's `RotatingKVCache` gives a later chunk keys from `f - 255`, so its 16-key blocks start there; the port anchored blocks at 0 and differed (2.8M of 4.2M words at T 2,048, KL 4,096) | now bit-identical in stock's layout (17 attention cases) | (changes only forwards starting past 255) | kept |
| attention mask and ring lookup | per-element mask only on blocks with an invisible pair; `key < cap ? key : key % cap` | bit-identical | full attention 2,204 -> 2,170 us (T 2,048), 5,824 -> 5,638 (KL 4,096) | kept |
| screens after these | | | mlx-prefill 3,232 / 4,595 / 4,582 (vs prefill-mm 1,149 / 1,196 / 927) | |
| `mlx_multi` (screens) | unsplit products sharing an input (q/k/v, w1 + w3) in one launch, each tile `mk_mlx_qmm_t`'s: fewer launches and barriers | logits sha unchanged on short/window/long | +1.4% / +1.0% / +0.4% | |
| `mlx-prefill-multi` (trial 20261120T172342) | mlx-prefill + mlx_multi, expect +1% | PASS everywhere; vs exact up to 1.26e-4 | 3,398.3 / 4,657.6 / 4,614.0 (beside mlx-prefill 3,386.6 / 4,614.9 / 4,592.6 and prefill-mm 1,139.1 / 1,200.7 / 938.5) | elite (mlx-multi); its 4,096 rounds ran on a rebuilt binary that differed only by the then-unused `head_trim` knob |
| `head_trim` (screen) | as in the exact tier | | 4,821 vs 4,655 at 1,024 (+3.6%) | |
| **`mlx-prefill-trim`** (trial 20261120T173733) | mlx-prefill-multi + head_trim, expect +3.5% | PASS everywhere on the cleaned binary; vs exact up to 1.26e-4 (window), 2.6e-5 to 4.7e-5 (long); kernels: 245 MLX cases all bit-identical | **3,508.4 / 4,851.8 / 4,750.5** vs mlx-prefill-multi 3,430.2 / 4,683.4 / 4,594.8 interleaved | **replaces-elite** at 1,024 and 4,096 (+3.6%, +3.4%); **loses** at 64: +2.3%, inside the margin (194.9 = twice the elite's own spread of 3,418 to 3,516) |

## What limits prefill now

Profiles (`FLINT_PROFILE=1`, one warm-up and one profiled run, GPU time per run; profiling slows the run, so only the shares are meaningful; `target/pm/prof-final-*.json`):

| | 4-bit linears | attention | everything else |
| --- | ---: | ---: | ---: |
| `mlx-prefill-trim`, 1,024 tokens (214 ms GPU) | 91.7% (`mk_mlx_qmm_multi` 50.2%, `mk_mlx_qmm_t` 41.4%) | 3.1% | 5.2% (silu-mul 2.0, norms 1.3, conv 1.2, attention prep 0.3) |
| `mlx-prefill-trim`, 4,096 tokens | 89.6% | 6.1% | 4.3% |
| `prefill-mm-trim` (exact), 4,096 tokens | 70.6% (`mk_mm_glu` 35.8%, `mk_mm` 34.3%, prep 0.5%) | 28.6% (`mk_attention_gated`, 29 ms a call) | 0.8% |

- **mlx-arith against stock MLX f32**: ahead at every length. The linears now run at MLX's own speed per operation (port / MLX 1.02 to 1.05 from 256 rows, `docs/MLX-PREFILL.md` section 7), and the engine does less around them: fused norms, conv and gate epilogues (stock runs RoPE as ten elementwise kernels a tensor, copies for the conv state and cache updates, transposes), a sliding-window attention that skips masked key blocks (2.5 to 7 times faster than stock's array-mask attention), and one launch for q/k/v and for w1 + w3. At 64 tokens the margin is large (+38%): stock's whole prefill there is 25 ms, mostly fixed cost (building and evaluating two graphs, the 63-token chunk and the last token).
- **Against stock bf16** the tier is 9 to 12% behind at 1,024 and 4,096: stock's bf16 linears stage half-width x and W tiles and its activations are half the bytes. Matching it needs a bf16 activation mode, a different arithmetic that neither tier admits today (the goldens are float32).
- **What is left inside the tier**: (1) the full-attention port is 1.3 to 1.85 times MLX's own `steel_attention` time on the same inputs (same algorithm line by line; the difference is code generation, not found): about 2.5 ms per 1,024 tokens and 11 ms per 4,096; (2) the bridge puts a barrier between every two dispatches where stock MLX lets independent kernels overlap; (3) the 2,048-row forwards keep stock's chunking, so that the second chunk's sliding layers see stock's key layout; one forward of 4,096 rows was 2% faster at 4,096 tokens in screen 5 (2 rounds, before the codegen fixes) but would use a key layout stock MLX never does.
- **Why 4,096 was slower than 1,024 by more than stock MLX's**: with exact attention it was (screen 4: 3,606 at 1,024 against 3,091 at 4,096) because the reference attention gives each (row, head) a threadgroup that walks every visible position serially, which grows with the prompt. With the attention port, and the sliding layers' masked blocks skipped, 4,096 is now 1.0% below 1,024 (stock f32: +0.6%, bf16: +2.3%).
- **Exact tier**: the exact expression tree is MLX's `qmv_fast` per row (lanes own 16-input segments, prescaled inputs, one `simd_sum`), which cannot use the 8 x 8 matrix units without reassociating the sums; its matrix-matrix kernels reuse each unpacked weight for 8 tokens but still issue scalar multiply-adds, at roughly a quarter of the port's rate. And its attention is the reference's (28.6% at 4,096). This is the cost of bit identity with token-at-a-time decoding: 3.8 to 4.8 times at 1,024 to 4,096 tokens.

## Removed variant code

Held no cell and fed no elite (checked against `trials.jsonl` parents); the evidence is the screens above:

- `mm_aligned` (2-byte-shifted copy of the checkpoint buffer for aligned code words), `mm_stage` (`mk_mms_*`, token tile in threadgroup memory), `mm_lazy` (`mk_mml_*`, packed codes unpacked per token): screened, none beat the 2 x 8 half tile.
- `prefill_attention` (`mk_attention[_gated]_p4/p8/p16`): 996 vs 1,019 tok/s at 1,024 (neutral).
- `inexact_mma`, `inexact_mma64` (`mk_mma[64]_q4_inexact`): the NOT ADMISSIBLE gap measurement, 1,730 tok/s at 1,024; superseded by the `mlx-arith` tier, whose kernels are MLX's own.
- The `prefill_mm` tiles other than 2 x 8 half (screens above); `mm_rows` now defaults to 2 so the default knobs validate.
