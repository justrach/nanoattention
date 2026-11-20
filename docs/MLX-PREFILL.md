# What stock MLX does when LFM3 reads a prompt

This answers "what does the MLX version do" for prompt ingestion (prefill) of LFM3-1.7B, 4-bit, with stock MLX 0.32.4 and mlx-vlm 0.8.0 on this repository's machine (Apple M3 Ultra, GPU family `applegpu_g15d`), and says which parts the megakernel's `mlx-arith` tier reproduces bit for bit. Sources read: the installed wheel's kernel headers (`site-packages/mlx/include/mlx/backend/metal/kernels/`), MLX's C++ host code at v0.32.4 (`quantized.cpp`, `scaled_dot_product_attention.cpp`, `normalization.cpp`, `conv.cpp`, `device.cpp`, `fast.cpp`), and `mlx_vlm/models/lfm3/language.py`, `models/lfm2/language.py` (`ShortConv`), `models/cache.py`, `models/base.py`. Everything below is for float32 activations (`model.set_dtype(float32)`, the precision of the goldens and of the megakernel) unless it says bf16.

## 1. One prefill

`mlx_vlm.generate` (and `tools/lfm3_benchmark.py`, which copies it) feeds the prompt of L tokens as:

1. chunks of at most 2,048 tokens (`DEFAULT_PREFILL_STEP_SIZE`) over the first L - 1 tokens; after each chunk, `mx.eval([c.state for c in cache])`;
2. the last token alone, `lm(ids[:, -1:], cache).logits`, then a host argmax.

MLX is lazy: a chunk's call builds a graph and `mx.eval` computes only what the cache states depend on. Consequences:

- **The output head never runs for prompt tokens.** Logits are built but never evaluated for chunk tokens; the final norm and the 2,048 x 128,000 tied-embedding projection run once, for the last token (M = 1).
- **The last layer's tail is skipped too.** LFM3's last layer (23) is a conv layer. Its cache state is the last `L_cache - 1` = 6 rows of `B * x`, which needs only `operator_norm` and `in_proj`. So for chunk tokens MLX never computes layer 23's conv output, `C *`, `out_proj`, post-operator norm, residual add, FFN norm, `w1`, `w3`, `silu`, `w2`, post-FFN norm and add: about 3.5% of the prompt's linear multiply-adds. (The megakernel's exact knob `head_trim` does the same; section 9.)
- The last token runs through the M = 1 kernels: `qmv_fast` for every linear, `sdpa_vector` for attention.

So at L = 1,024 MLX runs one chunk of 1,023 rows and one row; at L = 4,096 chunks of 2,048 and 2,047 rows and one row. At L = 64, one chunk of 63 rows and one row.

**Command encoding.** All kernels of a chunk go into compute encoders created with `MTLDispatchTypeConcurrent`; MLX inserts a buffer barrier only when a kernel reads a buffer an earlier kernel of the encoder wrote (`CommandEncoder::maybeInsertBarrier`), so independent kernels (q/k/v projections, `w1` and `w3`) can overlap. A command buffer is committed every 50 operations (or 50 MB of new buffers) on an Ultra (`'d'`) GPU (`device.cpp`). The graph also contains copies the megakernel does not have: the conv state concatenation, the K/V cache slice updates, contiguous copies of transposed tensors, and RoPE as about ten elementwise kernels per tensor (below).

## 2. The graph of one layer

`DecoderLayer.__call__`: `h = op(operator_norm(x)); x = x + post_operator_norm(h); h = MLP(ffn_norm(x)); x = x + post_ffn_norm(h)` (sandwich norms). 13 conv layers, 6 full-attention and 5 sliding-window (256) layers.

| step | MLX operation | kernel (float32, M rows of the chunk) |
| --- | --- | --- |
| norms (4 per layer, + q/k per head) | `nn.RMSNorm` -> `mx.fast.rms_norm` | `rms_single_row` (section 5) |
| every linear | `nn.QuantizedLinear` -> `mx.quantized_matmul(transpose=True, group 64, bits 4)` | section 3 |
| conv layer | `in_proj` -> split B, C, x -> `B * x` -> concat with the 6 cached rows -> depthwise `conv1d` (7 taps) -> `C * conv` -> `out_proj` | binary multiply, copy, `depthwise_conv_1d`, binary multiply |
| attention layer | q, k, v, gate projections; `q_norm`, `k_norm` per head; RoPE; cache update; `mx.fast.scaled_dot_product_attention`; `out * sigmoid(gate)`; `o_proj` | section 4 |
| MLP | `w2(silu(w1(x)) * w3(x))` | two `qmm`, compiled `silu` (one kernel), binary multiply, `qmm` |
| residuals | `x + h` | binary add |

## 3. The 4-bit linears

`QuantizedMatmul::eval_gpu` (`quantized.cpp`), for `y = x W^T` with x [M, K], W [N, K] packed 4-bit, float32 scales and biases per 64 inputs:

**Choice of kernel.** `vector_limit = get_qmv_batch_limit(K, N)`; on `applegpu_g15d` (gen 15, size `'d'`) that is 32 if K <= 2,048 and N <= 2,048, 18 if both <= 4,096, else 12. If M >= the limit: `qmm_splitk` (W is 2-D, so always this path), which picks `split_k = max(1, 512 / (ceil(N/32) * ceil(M/32)))`, clamps it to K / 64, lowers it until K % (split_k * 64) == 0, and falls back to plain `qmm` (`affine_qmm_t`) when split_k <= 1. Below the limit: `qmv_wide` for M >= 2 (affine, gen >= 15), `qmv_fast` for M = 1 (K is a multiple of 512). No NAX kernels on this GPU (`is_nax_available()` is false), and float32 never takes the NAX path without TF32 anyway.

For LFM3's linears:

| linear | K x N | vector limit | M = 1 | 8 | 31 | 64 | 256 | 1,023-1,024 | 2,047-2,048 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| w1, w3 | 2,048 x 7,680 | 12 | qmv_fast | qmv_wide | split 2 | qmm_t | qmm_t | qmm_t | qmm_t |
| w2 | 7,680 x 2,048 | 12 | qmv_fast | qmv_wide | split 8 | split 4 | qmm_t | qmm_t | qmm_t |
| conv in_proj | 2,048 x 6,144 | 12 | qmv_fast | qmv_wide | split 2 | qmm_t | qmm_t | qmm_t | qmm_t |
| q_proj, o_proj, conv out_proj | 2,048 x 2,048 | 32 | qmv_fast | qmv_wide | qmv_wide | split 4 | qmm_t | qmm_t | qmm_t |
| k_proj, v_proj | 2,048 x 512 | 32 | qmv_fast | qmv_wide | qmv_wide | split 16 | split 4 | qmm_t | qmm_t |
| gate_proj | 2,048 x 16 | 32 | qmv_fast | qmv_wide | qmv_wide | split 32 | split 32 | split 16 | split 8 |
| head | 2,048 x 128,000 | 12 | qmv_fast | (prefill evaluates the head for M = 1 only) | | | | | |

("split s" = `affine_qmm_t_splitk` with s K-partitions plus `steel_gemm_splitk_accum`.)

**Tile and threadgroups** (`qmm_t_impl`, BM = BN = BK = 32, WM = WN = 2): grid (ceil(N/32), ceil(M/32)[, split_k]), 128 threads = 4 SIMD groups per threadgroup; the threadgroup computes a 32-token x 32-output tile. Threadgroup memory: `Xs[32][36]` and `Ws[32][36]` floats (BK + 16 / sizeof(float) padding), 9 KB.

**Loading and dequantization, per 32-wide K block.** `BlockLoader` copies the x tile: thread t fills 8 consecutive inputs `8 (t % 4) .. +7` of token row `t / 4` (zeros for rows past M via `load_safe`). `QuantizedBlockLoader` fills the W tile: thread t reads the 4 code bytes of output row `t / 4`, inputs `8 (t % 4) .. +7`, loads that group's scale s and bias b, and `dequantize<float, 2, 4>` writes `w = s * (byte & 0x0f) + b` and `w' = (s / 16) * (byte & 0xf0) + b` (the high nibble is scaled by s/16 on the unshifted byte: the same value as s * code). Each weight is dequantized once per 32 tokens, into threadgroup memory; rows past N are zeros.

**Accumulation order of one output.** SIMD group g owns the 8 x 8 blocks at tokens `tm + {0, 16}` and outputs `tn + {0, 16}` (tm = 8 (g / 2), tn = 8 (g % 2)). For each K block in ascending order and within it kk = 0, 8, 16, 24: `C = A * B + C` with one `simdgroup_multiply_accumulate` (8 x 8 x 8), A = x[t][k0 + kk .. +7], B = W[o][k0 + kk .. +7]. C starts at 0. So y[t][o] is a chain of K / 8 hardware 8-term multiply-accumulates in K order; the order inside one 8 x 8 x 8 step is the hardware's. With split K, each of the s partitions accumulates its K / s inputs the same way into a float32 partial, and `gemm_splitk_accum` adds them in partition order: `((0 + P0) + P1) + ...`. Results are stored straight from the fragments (each lane holds 2 adjacent elements of each 8 x 8 block), bounds-checked for partial tiles.

This differs from the M = 1 kernel (`qmv_fast`: lanes own 16-input segments, prescaled inputs, one `simd_sum`), which is why stock MLX's batched prefill is not bit-identical to its own token-at-a-time result (about 4e-5 in LFM3's logits), and why the megakernel's exact tier (which keeps the M = 1 tree for every row) cannot use these kernels.

## 4. Attention

`scaled_dot_product_attention` (mlx-vlm `models/base.py`) calls `mx.fast.scaled_dot_product_attention(q, k, v, scale = 128^-0.5, mask)`.

**Mask and keys.** Full-attention layers use `KVCache`: keys are every position from 0; the mask is the string `"causal"` (MLX's `do_causal`, with queries aligned to the last rows). Sliding layers use `RotatingKVCache(max_size = 256)`; for a multi-row chunk `_update_concat` keeps only the last `max_size - 1` = 255 cached keys plus the chunk's own, so a chunk starting at position f sees keys `[max(0, f - 255), f + rows)`; the mask is a boolean array (`create_causal_mask(N, offset, window_size = 256)`) whenever `offset + N > 256`. At L = 4,096 the second chunk's sliding layers see keys 1,793..4,094 (2,302 keys), not 0..4,094.

**Kernel.** Query length <= 8: `sdpa_vector` (the last token). Otherwise, on a non-NAX GPU with head dim 128 and float32: `steel_attention` with BQ = 32 query rows, BK = 16 keys, BD = BV = 128, WM = 4 SIMD groups (8 query rows each), WN = 1; Q staged in threadgroup memory (`Q_smem[32][132]`), K (transposed, `[128][20]`) and V (`[16][132]`) share one buffer; grid (ceil(qL/32), heads); GQA by `head / (16 / 4)`. Per query row and key block kb in order: `S = 0; for dd = 0..15: S = Q[:, 8dd..] K^T[8dd.., :] + S`; `S *= scale * log2(e)`; masked entries := -FLT_MAX (causal: only on blocks near the diagonal; array mask: every block); `m' = max(m, rowmax S)`, `S = exp2(S - m')`, `f = exp2(m - m')`, `l = l f + rowsum S`, `O = O f`, `O = S V + O` (two 8-key halves per 8-wide output slice); finally `O / l`. Row max and row sum over a fragment's 8 columns are `((c0 op c1) op (c2 op c3)) op ((c4 op c5) op (c6 op c7))` (two in-thread elements, then shuffle-xor 1 and 8). With `do_causal` the key loop stops at the diagonal block; with an array mask (the sliding layers) it visits **every** key block of the chunk, which is why stock MLX's sliding attention costs more than its full attention (section 7).

The gate `out * mx.sigmoid(gate_proj(x))[..., None]` is two elementwise kernels after a transpose copy.

## 5. Norms, RoPE, conv, elementwise, embedding

- **RMSNorm** (`rms_single_row`, axis <= 4,096): one threadgroup per row of `ceil(axis / 4)` threads (512 for 2,048; 32 for the per-head 128); each thread sums the squares of its 4 consecutive elements in order, `simd_sum`, then the SIMD partials through threadgroup memory and another `simd_sum`; `inv = metal::precise::rsqrt(acc / axis + eps)`; `out = w * (x * inv)`.
- **RoPE** is not `mx.fast.rope` in this model: `RotaryEmbedding` computes `inv_freq = base ** (-arange(angles) * 2 / dim)` (zero-padded past `partial_rotary_factor`), `angles = positions * inv_freq`, `cos`, `sin`, and `x * cos + concatenate([-b, a]) * sin` with separate elementwise kernels.
- **Short conv**: `B * x` (binary), `concatenate([state, Bx])` (copy), `nn.Conv1d(groups = 2,048, kernel 7, no padding)` -> `depthwise_conv_1d` (grid D x T_out, one thread per output; its kernel source is compiled into the wheel's metallib and was not read here), `C * conv` (binary).
- **Elementwise**: one thread per element (binary/unary kernels). `sigmoid(x) = x < 0 ? y : 1 - y` with `y = 1 / (1 + precise::exp(|x|))`; `silu = x * sigmoid(x)` inside one `mx.compile`d kernel. The megakernel's `mk_sigmoid` / `mk_act` use the same formulas.
- **Embedding**: `QuantizedEmbedding`: gather the rows' codes, scales, biases, then `mx.dequantize` (scale * code + bias).

## 6. bf16 (stock's default dtype)

With the checkpoint's own bf16 activations the same kernels run with T = bfloat16: x and the dequantized W tiles are rounded to bf16 in threadgroup memory (half the bytes), accumulation stays float32, outputs are rounded to bf16 after every operation. That is why stock bf16 prefill is 16 to 18% faster than stock float32 at 1,024 and 4,096 tokens (same-session runs, `benchmarks/lfm3-prefill-search.md`), and why its logits differ from float32 by up to about 0.5. The megakernel computes in float32 throughout; matching bf16 speed would need a bf16 activation mode, which is a different arithmetic and not part of either admission tier.

## 7. Stock MLX's per-operation GPU time

`tools/nano_mlx_ops.py speed` (under the lock, MLX 0.32.4, float32, random 4-bit weights of LFM3's shapes; MLX time per call from a dependent chain of calls minus the chain's dependency cost; "port" is the megakernel's kernel on the same inputs, bit-identical output, `nano mlx-op --repeats 20`). Run of 2026-11-20 on the final binary: `target/pm2/speed-final.json`.

| operation | shape (K x N, or query rows x keys) | rows M | MLX kernel | stock MLX us | port us | port / MLX |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| w1, w3 | 2,048 x 7,680 | 64 | qmm_t | 162 | 226 | 1.40 |
| w1, w3 | 2,048 x 7,680 | 256 | qmm_t | 550 | 563 | 1.02 |
| w1, w3 | 2,048 x 7,680 | 1,024 | qmm_t | 2065 | 2126 | 1.03 |
| w1, w3 | 2,048 x 7,680 | 2,048 | qmm_t | 4082 | 4204 | 1.03 |
| w2 | 7,680 x 2,048 | 64 | qmm_t_splitk 4 + accum | 175 | 276 | 1.58 |
| w2 | 7,680 x 2,048 | 256 | qmm_t | 631 | 715 | 1.13 |
| w2 | 7,680 x 2,048 | 1,024 | qmm_t | 2166 | 2236 | 1.03 |
| w2 | 7,680 x 2,048 | 2,048 | qmm_t | 4191 | 4285 | 1.02 |
| conv in_proj | 2,048 x 6,144 | 64 | qmm_t | 124 | 160 | 1.28 |
| conv in_proj | 2,048 x 6,144 | 256 | qmm_t | 451 | 577 | 1.28 |
| conv in_proj | 2,048 x 6,144 | 1,024 | qmm_t | 1668 | 1698 | 1.02 |
| conv in_proj | 2,048 x 6,144 | 2,048 | qmm_t | 3254 | 3366 | 1.03 |
| q, o, conv out_proj | 2,048 x 2,048 | 64 | qmm_t_splitk 4 + accum | 60 | 97 | 1.61 |
| q, o, conv out_proj | 2,048 x 2,048 | 256 | qmm_t | 172 | 178 | 1.03 |
| q, o, conv out_proj | 2,048 x 2,048 | 1,024 | qmm_t | 570 | 597 | 1.05 |
| q, o, conv out_proj | 2,048 x 2,048 | 2,048 | qmm_t | 1118 | 1144 | 1.02 |
| k, v | 2,048 x 512 | 64 | qmm_t_splitk 16 + accum | 69 | 43 | 0.63 |
| k, v | 2,048 x 512 | 256 | qmm_t_splitk 4 + accum | 92 | 95 | 1.03 |
| k, v | 2,048 x 512 | 1,024 | qmm_t | 176 | 212 | 1.21 |
| k, v | 2,048 x 512 | 2,048 | qmm_t | 319 | 388 | 1.22 |
| gate_proj | 2,048 x 16 | 64 | qmm_t_splitk 32 + accum | 11 | 32 | 2.93 |
| gate_proj | 2,048 x 16 | 256 | qmm_t_splitk 32 + accum | 36 | 37 | 1.02 |
| gate_proj | 2,048 x 16 | 1,024 | qmm_t_splitk 16 + accum | 33 | 44 | 1.34 |
| gate_proj | 2,048 x 16 | 2,048 | qmm_t_splitk 8 + accum | 65 | 58 | 0.89 |
| attention, full, causal | 1,024 x 1,024 | 1,024 | steel_attention | 498 | 919 | 1.85 |
| attention, sliding 256 (array mask) | 1,024 x 1,024 | 1,024 | steel_attention | 1091 | 447 | 0.41 |
| attention, full, causal | 2,048 x 2,048 | 2,048 | steel_attention | 1678 | 2143 | 1.28 |
| attention, full, causal | 2,048 x 4,096 | 2,048 | steel_attention | 4326 | 5673 | 1.31 |
| attention, sliding 256 (array mask) | 2,048 x 2,048 | 2,048 | steel_attention | 3970 | 576 | 0.15 |
| attention, sliding 256 (array mask) | 2,048 x 4,096 | 2,048 | steel_attention | 4318 | 568 | 0.13 |

At 64 rows the per-call times are a few tens of microseconds and the two measurements are not comparable to better than a factor of 1.5 (MLX's figure subtracts its chain's dependency cost; the port's includes one barrier per call). From 256 rows the linear port is within 2 to 5% of MLX's own kernel for the large shapes; it was 13% slower before its scale/bias loads were specialised per dtype at compile time and advanced like MLX's loader (`benchmarks/lfm3-prefill-search.md`). The full-attention port is 1.3 to 1.85 times MLX's time (same algorithm; the difference is code generation, not found); the sliding port is 2.5 to 7 times faster than MLX, because MLX's array mask makes it visit every key block of the chunk.

Other stock operations at 1,024 / 2,048 rows (`target/pm/mlx-op-times.json`, same method, noisier): RMSNorm over 2,048 below the method's resolution at 1,024 rows, 53 us at 2,048; per-head q/k RMSNorm 48 / 80 us; residual add 28 / 54 us; `silu(w1) * w3` 155 / 420 us; attention output times sigmoid gate 39 / 53 us; short conv (`B * x`, conv1d, `C *`) 167 / 321 us; RoPE in mlx-vlm's elementwise form 142 / 314 us.

Whole-model chunks (stock MLX, float32, `nano_mlx_ops.py time --model`, `target/pm/mlx-op-times.json`): 1,024 rows 220 ms (4,650 tok/s), 2,048 rows 444 ms. The linears are about 90% of a chunk in MLX too: at 1,024 rows they sum to about 200 ms of the 220.

Same-session totals (prefill tok/s, `tools/nano_bench.py versus`): see `benchmarks/lfm3-prefill-search.md`.

## 8. What the megakernel reproduces bit for bit (`mlx-arith` tier)

`src/mega/mlx.metal`, selected by knobs (`docs/MEGAKERNEL.md`, section 4):

| stock MLX | port | condition | checked |
| --- | --- | --- | --- |
| `affine_qmm_t` (float32, group 32/64/128) | `mk_mlx_qmm_t_<f32/f16/bf16 scales>_g<group>` | knob `mlx_linear`, M >= MLX's vector limit, split 1 | bit-identical to `mx.quantized_matmul`, 174 cases: LFM3's 6 shapes x random and real-checkpoint weights x 18 batches 32..2,048 including ragged ones, data 2 bytes off a word boundary as in the checkpoint; plus 12 cases with float16 and float32 scale storage |
| `affine_qmm_t_splitk` + `steel_gemm_splitk_accum` | `mk_mlx_qmm_t_splitk_*` + `mk_mlx_splitk_accum` | same, MLX's split > 1 (split chosen by `exec.mlxSplitK` exactly as `qmm_splitk`) | in the same 174 (split 2 to 32) |
| several `affine_qmm_t` launches on one input | `mk_mlx_qmm_multi_*` | knob `mlx_multi`, unsplit products sharing an input (q/k/v, w1 + w3) | each tile is `mk_mlx_qmm_t`'s; equal bit for bit to one launch per matrix (kernel test; logits sha unchanged on the goldens) |
| `steel_attention` float32, D 128, causal or array mask | `mk_mlx_attention`, `mk_mlx_attention_gated` | knob `mlx_attention`, more than 8 query rows | bit-identical to `mx.fast.scaled_dot_product_attention` on 17 cases: full causal (T 9 to 2,048, KL up to 4,096), sliding 256 in stock's `RotatingKVCache` key layout (keys from `max(0, f - 255)`, later chunks included), ring wrap, ragged; gated epilogue equal to `mk_binary(att, sigmoid)` |

Not ported (the `mlx-arith` candidates keep the engine's exact kernels for these): `qmv_wide` / `qmv_fast` (M below the vector limit; decode), `sdpa_vector` (<= 8 query rows; decode), RMSNorm (the engine uses a different reduction tree), RoPE (the engine uses a frequency table, not MLX's ten elementwise kernels), the depthwise conv, the embedding, the elementwise kernels (same formulas, not checked as kernels). The gate multiply is fused into the attention epilogue with `mk_binary`'s formula.

Differences in what is computed, all exact no-ops or choices of grouping:

- The port's sliding-window attention visits only key blocks that hold a visible key for some row of the query block; MLX visits every block. A skipped block contributes `exp2(-FLT_MAX - m) = 0` with factor 1 in MLX, or is cleared by a factor of 0 before the first visible key, so the result is the same up to the sign of an exactly-zero output (none occurred in the checks). The port also applies the per-element mask only to blocks with an invisible (row, key) pair; elsewhere the mask is all-visible and changes nothing.
- Forward size: the `mlx-arith` candidates use forwards of 2,048 rows (stock's chunk). The engine feeds all L prompt tokens in forwards (at L = 1,024 one forward of 1,024 rows; stock: 1,023 + 1), so the rows are grouped into tiles differently from stock at the chunk ends; tile grouping does not change a row's result when MLX does not split K (every chunk-sized linear in LFM3 except `gate_proj`, whose split depends on M: 16 at 1,023-1,024 rows, 8 at 2,047-2,048).
- The engine's whole forward is therefore not bit-identical to stock MLX's prefill (norms, RoPE, conv and the last token differ); the tier's end-to-end gate is the goldens' float32 gate (`docs/QD.md`), which every admitted `mlx-arith` candidate meets.

## 9. The engine's matching schedule choices

- `head_trim` (exact): rows without logits stop after the program's last persistent store, as stock's lazy chunk evaluation does (section 1).
- `mlx_multi`: one launch for independent products, the effect stock MLX gets from its concurrent encoder.
- Batch 2,048: stock's chunk size, so the second chunk's sliding-window key layout is the one stock MLX uses.
