# LFM3-1.7B: confirmed specification

This is the working specification of the model, checked against the stock MLX implementation (mlx-vlm 0.8.0, `mlx_vlm/models/lfm3/` and the `ShortConv` it borrows from `mlx_vlm/models/lfm2/language.py`; mlx 0.32.4), against the safetensors header of the `LiquidAI/LFM3-1.7B-MLX-4bit` checkpoint's `model.safetensors` (sha256 `1be5bf67…3815`), and numerically against an independent NumPy implementation written from this text (`tools/lfm3_reference.py`). The Hugging Face source (`modeling_lfm3.py`, read, never run) is the secondary reference. Where they disagree it says so (section 11). Every statement marked **[numeric]** was settled by running both implementations. The rest was read from source and header and holds in the NumPy model, which matches stock MLX to float32 rounding.

## 1. Shape

| | |
| --- | --- |
| layers | 24 |
| hidden | 2048 |
| vocab (embedding rows, logits) | 128,000 (the tokenizer defines 125,033 ids; rows 125,033 to 127,999 exist but no text maps to them) |
| query heads / KV heads / head dim | 16 / 4 / 128 (GQA: query head `h` reads KV head `h // 4`) |
| MLP intermediate | 7,680 |
| conv width (`conv_L_cache`) | 7, depthwise, causal, no bias |
| sliding window | 256 positions, the query's own included **[numeric]** |
| RMSNorm eps | 1e-5 (`norm_eps`) everywhere, including q/k norm |
| biases | none in any linear or conv layer |
| output head | tied: the (quantized) token embedding |
| parameters | 1,728,801,536 logical (counted from the header) |
| special ids | bos 124894 `<|startoftext|>`, eos 124900 `<|im_end|>`, pad 124893 `<|pad|>` |

`layer_types` (from `config.json`): conv at 0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23; `full_attention` at 2, 6, 10, 14, 18, 22; `sliding_attention` at 4, 8, 12, 16, 20. So there are 13 conv, 6 full and 5 sliding layers. Stock MLX refuses a config without an explicit `layer_types` list. HF derives the same pattern as a default.

## 2. Checkpoint tensors

594 tensors in one file, every name prefixed `language_model.model.` (stock `sanitize` also accepts a `model.` prefix and prepends `language_model.`). The header's `__metadata__` is `{"format": "mlx"}`. Quantization (`config.json`): `{"group_size": 64, "bits": 4, "mode": "affine"}` for every quantized matrix (no per-module overrides).

| name (per layer `layers.N.`) | dtype | shape | count | logical matrix [out, in] |
| --- | --- | --- | --- | --- |
| `embed_tokens.weight` / `.scales` / `.biases` | U32 / BF16 / BF16 | [128000, 256] / [128000, 32] / [128000, 32] | 1 | [128000, 2048] |
| `embedding_norm.weight` | BF16 | [2048] | 1 | |
| `layers.N.operator_norm.weight`, `post_operator_norm.weight`, `ffn_norm.weight`, `post_ffn_norm.weight` | BF16 | [2048] | 24 each | |
| `layers.N.feed_forward.w1` (gate), `w3` (up) `.weight/.scales/.biases` | U32 / BF16 / BF16 | [7680, 256] / [7680, 32] / [7680, 32] | 24 each | [7680, 2048] |
| `layers.N.feed_forward.w2` (down) | U32 / BF16 / BF16 | [2048, 960] / [2048, 120] / [2048, 120] | 24 | [2048, 7680] |
| conv layers: `layers.N.conv.in_proj` | U32 / BF16 / BF16 | [6144, 256] / [6144, 32] / [6144, 32] | 13 | [6144, 2048] |
| conv layers: `layers.N.conv.out_proj` | U32 / BF16 / BF16 | [2048, 256] / [2048, 32] / [2048, 32] | 13 | [2048, 2048] |
| conv layers: `layers.N.conv.conv.weight` | BF16, **not quantized** | [2048, 7, 1] | 13 | |
| attention layers: `layers.N.self_attn.q_proj`, `o_proj` | U32 / BF16 / BF16 | [2048, 256] / [2048, 32] / [2048, 32] | 11 each | [2048, 2048] |
| attention layers: `layers.N.self_attn.k_proj`, `v_proj` | U32 / BF16 / BF16 | [512, 256] / [512, 32] / [512, 32] | 11 each | [512, 2048] |
| attention layers: `layers.N.self_attn.gate_proj` | U32 / BF16 / BF16 | [16, 256] / [16, 32] / [16, 32] | 11 | [16, 2048] |
| attention layers: `layers.N.self_attn.q_norm.weight`, `k_norm.weight` | BF16 | [128] | 11 each | |

154 quantized matrices and 440 floating tensors. There is no `lm_head` tensor. The data section ends at byte 973,008,384 after the 74,994-byte header.

### 2.1 Packed 4-bit affine weights (MLX affine, group 64)

For a logical matrix `W[out, in]`:

- `weight` is U32 `[out, in / 8]`. Element `W[r, c]` is the 4-bit code `q` in word `c // 8` of row `r`, at bits `4 * (c % 8)` through `4 * (c % 8) + 3`. In other words, each row is a little-endian bit stream with element 0 in the lowest bits. Viewed as bytes, byte `b` of a row holds element `2b` in its low nibble and `2b + 1` in its high nibble.
- `scales`, `biases` are BF16 `[out, in / 64]`, one pair per group of 64 consecutive input columns.
- `W[r, c] = scales[r, c // 64] * q + biases[r, c // 64]` with `q` in 0..15 (unsigned; no zero point subtracted. The bias is the group minimum).

**[numeric]** This formula, evaluated in float32 (`float(q) * scale + bias`, BF16 widened exactly), reproduces `mx.dequantize` bit for bit for all 154 matrices (`lfm3_reference.py inspect`: max difference 0). The product is exact in float32 (8-bit mantissa times a 4-bit integer), so the only rounding is the add. Stock MLX's matmul kernels may instead accumulate `scale * Σ x·q + bias * Σ x` per group, which differs at float32 rounding only.

The embedding uses the same format. Embedding row `t` is token `t`'s vector. The output head is `logits[v] = Σ_c W_embed[v, c] * n[c]`, the same dequantized rows (`embed_tokens.as_linear` in MLX).

### 2.2 The conv weight

The tensor is stored channels-last as `[2048, 7, 1]` = `[channel d, tap k, in/groups]` (MLX `Conv1d` layout; the PyTorch checkpoint has `[2048, 1, 7]`, which stock `sanitize` transposes; this export already has the MLX layout). Let `w[d, k] = conv.weight[d, k, 0]`. Tap `k = 6` multiplies the current position and tap `k = 0` the position six back:

```
v[t, d] = Σ_{k=0..6} w[d, k] * u[t + k - 6, d]          (u at negative t is 0)
```

This is cross-correlation as in both frameworks, with no kernel flip.

## 3. Forward

```
h = embed(token)                                   # dequantized embedding row, 2048
for each layer i:
    n = RMSNorm(h, operator_norm)
    if conv:      y = ShortConv(n)
    else:         y = Attention(n, position)
    h = h + RMSNorm(y, post_operator_norm)         # sandwich norm
    f = w2( silu(w1(m)) * w3(m) ),  m = RMSNorm(h, ffn_norm)
    h = h + RMSNorm(f, post_ffn_norm)
logits = W_embed · RMSNorm(h, embedding_norm)      # 128,000
```

`silu(x) = x * sigmoid(x)`. The residual stream `h` is never normalized in place. Both norms of a pair act on the branch output before it is added (`use_sandwich_norm: true`).

### 3.1 RMSNorm

`RMSNorm(x, w) = w * x / sqrt(mean(x^2) + 1e-5)`, over the last axis (2048, or 128 for q/k norm). The weight multiplies directly: there is no `1 + w`. The stored weights are small, for example `operator_norm` of layer 0 lies in [-0.72, 0.62], and `embedding_norm` in [0.013, 4.0]. Stock MLX uses `mx.fast.rms_norm` (float32 accumulation). HF upcasts to float32 explicitly.

### 3.2 Conv layer (gated short convolution)

```
BCx = in_proj(n)                 # 6144
B = BCx[0:2048],  C = BCx[2048:4096],  X = BCx[4096:6144]     # split in this order
u = B * X                        # elementwise
v = depthwise causal conv of u (section 2.2)
y = out_proj(C * v)
```

**Conv state**: per conv layer the last 6 values of `u` (the product `B * X`, before the convolution), 6 × 2048, oldest first, zero at sequence start. For a token batch, position `t` reads `u` from earlier positions of the same batch and then from the carried state. After the batch the state is the last 6 values of `[state, batch u]`. Stock MLX stores exactly this (`ArraysCache`, `cache[0] = concat(state, Bx)[:, -6:]`). HF's `conv_states` hold the same `u` (6 or 7 wide; the width-7 variant also keeps one discarded value). One decode step: `window = [state (6 rows), u_t]`, `v = Σ_k w[:, k] * window[k]`, `state = window[1:]`.

### 3.3 Attention layer

```
q = q_proj(n) as [16, 128];  k = k_proj(n) as [4, 128];  v = v_proj(n) as [4, 128]
q = RMSNorm(q, q_norm) per head;  k = RMSNorm(k, k_norm) per head   # one 128-weight shared by all heads; v is not normed
q, k = RoPE(q, pos), RoPE(k, pos)                                    # after the norm
append k, v to the layer's cache
a[h] = softmax( q[h] · K[h // 4]^T / sqrt(128) ) · V[h // 4]         # over the visible positions
g = sigmoid(gate_proj(n))                                            # [16], from the same normed input n
y = o_proj( concat_h( a[h] * g[h] ) )                                # head-major: element h*128 + j
```

There is no softcap, no sink, no attention bias and no ALiBi, and the scale is `128^-0.5` exactly.

**Visible positions.** Full layers: every position `0..pos`. Sliding layers: positions `j` with `pos - 255 <= j <= pos`, which is **256 positions including the query itself [numeric]**. Stock MLX builds the batched mask as `(i >= j) & (i < j + 256)` (`create_causal_mask(window_size=256)`) and decodes with a `RotatingKVCache(max_size=256)` that holds the newest 256 keys after writing the current one. HF's sliding overlay is `kv_idx > q_idx - 256`, the same rule. The NumPy model was run on the 700-token `window` case with windows of 255, 256 and 257. Against stock MLX float32, the first sliding layer's output (residual index 5) at positions 257 and later differed by 1.7e-2, **3.1e-6** and 2.1e-2 respectively (positions 0 to 254 differ by 3.6e-6 in all three). Logits at positions 257 and later differed by 1.57, **6.1e-4** and 5.69, with top-1 agreement of 439, **443** and 435 out of 443. So a ring of 256 slots per sliding layer, written at `pos % 256` before attending, is exact. Keys are stored after RoPE, so ring order does not matter.

### 3.4 RoPE

Rotate-half layout: for `i` in 0..63 the pair is (element `i`, element `i + 64`) of a head:

```
x'[i]      = x[i] * cos(a_i) - x[i+64] * sin(a_i)
x'[i + 64] = x[i+64] * cos(a_i) + x[i] * sin(a_i)
a_i = pos * inv_freq[i]
inv_freq[i] = theta ^ (-2i / 128)   for i < A,   0 for A <= i < 64,   A = int(partial_rotary_factor * 128) // 2
```

- Full-attention layers: theta = 1,000,000, `partial_rotary_factor` 0.5, so A = 32. Pairs 0..31 (elements 0..31 and 64..95) rotate. **Elements 32..63 and 96..127 are not rotated.** The frequencies keep the full-head progression (`2i / 128`, not `2i / 64`). This is a zero-tail layout, **not** "the first 64 elements rotated": the HF config docstring's wording ("rotate the first `partial_rotary_factor * head_dim` dimensions") does not describe what either implementation's code does.
- Sliding layers: theta = 10,000, factor 1.0, so A = 64: all pairs rotate.
- Conv layers have no RoPE.
- `pos` is the absolute token index from 0 (the KV cache offset; sliding layers use the absolute position too, not the ring slot).

Precision: both implementations form `inv_freq` and the angle in float32 (`angle = float32(pos) * inv_freq`, rounded to float32), then take cos and sin. At position 4,096 a float64 angle differs from the float32 one by up to about 2.3e-4 rad in the fastest pairs. Stock MLX's GPU `pow` is not correctly rounded: it differs from the correctly rounded float32 value by one ulp in 11 of 32 (full) and 29 of 64 (sliding) frequencies. Feeding the NumPy model MLX's exact `inv_freq` and `mx.cos`/`mx.sin` values changed the 700-token logit difference only from 6.1e-4 to 6.4e-4. RoPE rounding is not the dominant float32 error at these lengths.

### 3.5 MLP

`f = w2( silu(w1 · m) * (w3 · m) )`, with `w1` the gate (7680), `w3` the up projection (7680) and `w2` the down projection (2048). No biases.

## 4. Caches and decode state

| layer type | state per layer | size |
| --- | --- | --- |
| conv (13) | last 6 `u = B * X` vectors | 6 × 2048 |
| sliding (5) | ring of the newest 256 post-RoPE K and V (4 heads × 128 each) | 256 × 4 × 128 × 2 |
| full (6) | all K and V | T × 4 × 128 × 2 |

Stock MLX prefill (one batch through the caches), stock MLX one token at a time through the same caches, NumPy whole-sequence-with-masks and NumPy one-token-with-ring-and-shift-register all agree (see `golden/lfm3/MANIFEST.md`). That covers batches that wrap the ring and batches longer than the window.

## 5. Precision: what the reference runs in

- Stock MLX runs in the checkpoint's dtype by default: the 440 floating tensors (scales, biases, norms, conv weight) are BF16 and the activations are BF16. `model.set_dtype(mx.float32)` widens the floating tensors to float32 (exactly) and makes every activation float32. The packed codes are untouched. The goldens come from that float32 run.
- The result depends on the activation dtype. On the 23-token `short` prompt, BF16 versus float32 logits differ by up to 0.35 (RMS 0.042) with top-1 still agreeing at all 23 positions. Exact numbers for every case are in each `meta.json` under `report.mlx_bf16_vs_f32`.
- The NumPy reference computes in float64 from exactly dequantized weights. Stock MLX float32 differs from it by the same amount as the NumPy model run in float32 differs from itself in float64 (the float32 noise floor, recorded per case). An exact float32 engine should expect differences of that size from the goldens, not zero.

## 6. Tokenizer, chat template, stop tokens

- `tokenizer.json` (byte-level BPE, `TokenizersBackend`). Plain `encode` adds no BOS; the chat template writes it.
- Chat template (`chat_template.jinja` in the checkpoint), for one user turn with a generation prompt:

  ```
  <|startoftext|><|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think></think>
  ```

  The ending above is with `enable_thinking=False`, which is what mlx-vlm passes by default and what the goldens use. With the template's own default (`enable_thinking` true) the prompt ends in `<think>` and the model reasons first. A system message becomes `<|im_start|>system\n…<|im_end|>\n`, and tools are inserted into it as `<tools>…</tools>`. Tool calls are rendered as `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` and tool results as `<|im_start|>tool\n<tool_response>\n…\n</tool_response><|im_end|>`.
- Ids: `<|pad|>` 124893, `<|startoftext|>` 124894, `<|endoftext|>` 124895, `<|im_start|>` 124899, `<|im_end|>` 124900, `<think>` 124901, `</think>` 124902, `<tool_call>` 125019, `</tool_call>` 125020. `"\n"` after a role is id 207; `user` 5922, `assistant` 63514.
- **Stop tokens**: `eos_token_id: [124900]` (`<|im_end|>`) in both `config.json` and `generation_config.json`, and stock MLX stops on exactly that. `<|endoftext|>` (124895) is not a stop token in this checkpoint's configs.
- `generation_config.json` asks for sampling (temperature 0.1, top-k 50, repetition penalty 1.1). The goldens and `tools/lfm3_chat.py` are plain greedy argmax, with no penalty.

## 7. Golden format

Written by `tools/lfm3_reference.py goldens` to `$NANO_WORK/golden/lfm3/<case>/` (they are not kept in the repository). `hidden.npy` is `[25, T, 2048]`: index 0 is the embedding output and index `i + 1` the residual stream after layer `i` (before `embedding_norm`). `greedy.json` includes the EOS token when one is emitted. See `golden/lfm3/MANIFEST.md`.

## 8. Tiny synthetic checkpoints

`tools/lfm3_tiny.py --out DIR` writes seven checkpoints with the same tensor names (`language_model.model.` prefix), config keys and MLX affine layout as the real one, each with goldens in the format above (`DIR/<case>/golden/`). Common shape: hidden 64, 4 query heads, 2 KV heads, head dim 16 (so full layers rotate pairs 0..3 of 8), MLP 96, vocab 256, layers `conv, full, conv, sliding, conv, full, sliding`, sliding window 5, a 20-token prompt and 64 greedy tokens (eos id 2).

| case | floats | quantized matrices | conv width |
| --- | --- | --- | --- |
| `f32` | F32 | none | 4 |
| `bf16` | BF16 | none | 4 |
| `q4_g32` | BF16 | 4-bit, group 32 | 4 |
| `q8_g32` | BF16 | 8-bit, group 32 | 4 |
| `q4_g8` | BF16 | 4-bit, group 8 | 4 |
| `q8_g16` | BF16 | 8-bit, group 16 | 4 |
| `q4_g32_conv7` | BF16 | 4-bit, group 32 | 7 (the real width) |

The goldens are NumPy float64 on the values decoded from the written bytes. The case fails unless the sequence and step forms agree to 1e-9; they agree to about 2e-14. With `--check-mlx`, stock mlx-vlm loads every case it supports (groups 32/64/128; groups 8 and 16 are rejected by `mx.quantize`) and matches in float32: logits within 2.1e-5, hidden within 1.5e-5, greedy identical. So the tiny checkpoints are valid stock-MLX LFM3 checkpoints, not just NumPy-shaped ones.

## 9. Things an implementation is likely to get wrong

1. Rotating the first 64 elements of a full-attention head instead of pairs (i, i + 64) for i < 32.
2. Using `2i / 64` instead of `2i / 128` for the full-attention frequencies.
3. A sliding window of 256 *previous* positions (257 including itself). It is 256 including itself.
4. Applying `1 + weight` in RMSNorm.
5. Normalizing V, or applying the gate before attention, or computing the gate from the un-normed residual. The gate comes from the normed input `n`.
6. Splitting `in_proj` as X, B, C or interleaved. The order is B, C, X in contiguous blocks of 2048.
7. Carrying `v` (the conv output) or `B * X * C` in the conv state instead of `u = B * X`.
8. Flipping the conv taps. `w[d, 6]` multiplies the current position.
9. Reading the conv weight as `[7, 2048]`. It is `[2048, 7, 1]`.

## 10. Verified by

`tools/lfm3_reference.py`: `inspect` (header and dequantization), `window-rule`, `compare`, `goldens`, `chunked` (chunked prefill against the `long` golden) and `check` (scores an engine's outputs against a golden case). Numbers are in `golden/lfm3/MANIFEST.md`.

## 11. Where stock MLX and the Hugging Face source differ

None of these changes a float32 single-sequence forward beyond rounding.

1. **RoPE frequencies**: HF `1 / theta^(2i/128)`, MLX `theta^(-2i/128)`. Both are float32 and differ by an ulp in some entries.
2. **RoPE precision in BF16**: HF always rotates in float32 (q and k upcast) but first rounds cos and sin to the activation dtype. MLX rotates in the activation dtype. They are identical in float32.
3. **RMSNorm in BF16**: HF rounds the normalized vector to BF16 and then multiplies by the weight. MLX uses the fused `mx.fast.rms_norm`. They are identical in float32 up to rounding.
4. **Packed documents and padding**: HF resets the conv window at document boundaries (`seq_idx` from restarting `position_ids`) and masks padding inside the conv. Stock MLX has neither, beyond its batch left-padding masks. This is irrelevant for one unpadded sequence.
5. **RoPE types**: HF accepts scaled RoPE types and truncates them to the zero-tail layout. Stock MLX accepts only `default`, which is what this checkpoint uses.
6. **Chat template**: the checkpoint's template `trim`s the system prompt and non-assistant message content. The template in the original `LiquidAI/LFM3-1.7B` repository does not. The goldens use the checkpoint's.
7. **Config defaults**: HF `max_position_embeddings` defaults to 128,000 and MLX to 262,144. The config says 262,144. Neither implementation uses it in the forward.

## 12. Not confirmed

- Behaviour beyond 4,096 tokens: no golden goes further, though nothing in the forward depends on length except the float32 RoPE angle.
- Batched (multi-row, padded) inference: not exercised. Only single sequences are.
- The HF PyTorch implementation itself was never run (no PyTorch, by rule). It is compared by reading only.
