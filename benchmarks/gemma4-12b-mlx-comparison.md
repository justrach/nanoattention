# Gemma 4 12B (`nano` vs `mlx-lm` on Apple Silicon)

Head-to-head throughput benchmark of `nano` megakernel against stock `mlx-lm` on `lmstudio-community/gemma-4-12B-it-MLX-4bit` (4-bit affine quantization, group size 64, 48 decoder layers: 5x sliding window 1024 + 1x full `k_eq_v` attention with `global_head_dim=512`, `layer_scalar`, tied `embed_tokens`, `final_logit_softcapping=30.0`).

## Environment & Methodology

- **Hardware**: Apple M3 Ultra, 256.0 GiB unified memory, macOS 27.0 (`arm64`)
- **Commit**: `f875890` (`zig build -Doptimize=ReleaseSafe`)
- **MLX**: `mlx` 0.32.4, `mlx-lm` 0.32.3.dev17+g736ab0d8c (stock `mlx_lm.models.gemma4`)
- **Prompt Lengths**: 64, 1024, 4096 tokens (`context=4224`), followed by 64 decode tokens, 1 warmup run + 3 measured repeats (medians reported).
- **Workload**: identical Gemma 4 tokenized IDs across all engines (`input_sha256`: `cdd9df220ad8fd84...`). Each engine runs in an isolated process under `tools/guarded_run.py` so weights never overlap in RAM.
- **Dtype Disclosure**: `nano` executes **FP32 activations and FP32 KV caches** (preserving < `1.5e-3` max logit error vs FP32 `mlx-lm` at `L=1280`). Stock `mlx-lm` defaults to **BF16 activations and BF16 KV caches** (`mlx_bf16`); `mlx_f32` converts floating tensors/scales/biases/KV caches to **FP32** to match `nano`'s numerical precision.

## Results

### Prompt Length `L = 64` (64 decode tokens)

| Engine | Precision | Prefill (tok/s) | Prefill (ms) | Decode (tok/s) | Decode (ms/tok) | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nano (`presets/gemma4.json`, greedy) | FP32 act / FP32 KV | 547.9 | 116.8 | 49.67 | 20.13 | 4,871.3 |
| nano (`presets/gemma4.json`, teacher-forced) | FP32 act / FP32 KV | 548.2 | 116.7 | 50.45 | 19.82 | 4,871.4 |
| nano (`presets/exact.json`, greedy) | FP32 act / FP32 KV | 203.4 | 314.6 | 44.03 | 22.71 | 2,836.3 |
| mlx-lm BF16 (`async_eval`) | BF16 act / BF16 KV | 493.4 | 129.7 | 72.30 | 13.83 | 6,500.2 |
| mlx-lm BF16 (sync) | BF16 act / BF16 KV | 491.0 | 130.3 | 60.71 | 16.47 | 6,499.8 |
| mlx-lm FP32 (`async_eval`) | FP32 act / FP32 KV | 444.1 | 144.1 | 66.34 | 15.07 | 6,499.4 |
| mlx-lm FP32 (sync) | FP32 act / FP32 KV | 451.1 | 141.9 | 58.16 | 17.19 | 6,499.1 |

### Prompt Length `L = 1024` (64 decode tokens)

| Engine | Precision | Prefill (tok/s) | Prefill (ms) | Decode (tok/s) | Decode (ms/tok) | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nano (`presets/gemma4.json`, greedy) | FP32 act / FP32 KV | 622.4 | 1,645.3 | 45.03 | 22.21 | 4,871.4 |
| nano (`presets/gemma4.json`, teacher-forced) | FP32 act / FP32 KV | 585.5 | 1,748.8 | 44.81 | 22.32 | 4,871.5 |
| nano (`presets/exact.json`, greedy) | FP32 act / FP32 KV | 224.8 | 4,554.4 | 41.26 | 24.24 | 2,836.3 |
| mlx-lm BF16 (`async_eval`) | BF16 act / BF16 KV | 743.3 | 1,377.7 | 68.32 | 14.64 | 6,500.2 |
| mlx-lm BF16 (sync) | BF16 act / BF16 KV | 725.7 | 1,411.1 | 58.67 | 17.04 | 6,499.8 |
| mlx-lm FP32 (`async_eval`) | FP32 act / FP32 KV | 542.1 | 1,888.9 | 56.23 | 17.78 | 6,499.4 |
| mlx-lm FP32 (sync) | FP32 act / FP32 KV | 566.5 | 1,807.6 | 49.81 | 20.08 | 6,499.1 |

### Prompt Length `L = 4096` (64 decode tokens)

| Engine | Precision | Prefill (tok/s) | Prefill (ms) | Decode (tok/s) | Decode (ms/tok) | Peak RSS (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nano (`presets/gemma4.json`, greedy) | FP32 act / FP32 KV | 647.1 | 6,329.5 | 43.48 | 23.00 | 4,871.3 |
| nano (`presets/gemma4.json`, teacher-forced) | FP32 act / FP32 KV | 612.6 | 6,686.7 | 43.47 | 23.01 | 4,871.7 |
| nano (`presets/exact.json`, greedy) | FP32 act / FP32 KV | 161.7 | 25,326.4 | 38.62 | 25.89 | 2,836.4 |
| mlx-lm BF16 (`async_eval`) | BF16 act / BF16 KV | 652.5 | 6,277.9 | 67.48 | 14.82 | 6,500.2 |
| mlx-lm BF16 (sync) | BF16 act / BF16 KV | 634.2 | 6,459.0 | 57.64 | 17.35 | 6,499.8 |
| mlx-lm FP32 (`async_eval`) | FP32 act / FP32 KV | 509.8 | 8,034.8 | 56.20 | 17.79 | 6,499.4 |
| mlx-lm FP32 (sync) | FP32 act / FP32 KV | 501.9 | 8,161.4 | 49.54 | 20.19 | 6,499.1 |

## Summary Takeaways

| Length | `nano_gemma4` vs `mlx_f32_async` (Prefill / Decode) | `nano_gemma4` vs `mlx_bf16_async` (Prefill / Decode) |
| ---: | :---: | :---: |
| 64 | 1.23x / 0.75x | 1.11x / 0.69x |
| 1024 | 1.15x / 0.80x | 0.84x / 0.66x |
| 4096 | 1.27x / 0.77x | 0.99x / 0.64x |

- **Prefill**: In matching **FP32** precision, `nano` (`presets/gemma4.json`, wide-head MMA prefill kernels `mk_mlx2_attention_d{256,512}` plus `mk_mlx2_*` Q4 GEMMs) beats `mlx-lm` FP32 (`async_eval`) at all three lengths (**+23%** at `L=64`, **+15%** at `L=1024`, **+27%** at `L=4096`). Even against half-precision `mlx-lm` BF16, `nano` (FP32) leads at `L=64` (`547.9` vs `493.4 tok/s`) and matches at `L=4096` (`647.1` vs `652.5 tok/s`). Disabling `mlx_attention2` on `nano` drops `L=4096` prefill to `209.7 tok/s`, showing `3.07x` speedup from the wide-head MMA attention kernels in the sweep.
- **Decode & Memory**: `nano` executes **634 dispatches/token** (with `mk_*_q4_static_{3840,4096,8192,15360}`, `mk_dec_scores_e16_sub*`, and `d2_gate`), delivering `49.67 -> 43.48 tok/s` (`20.1 -> 23.0 ms/tok`) in full **FP32** activations/KV with **`4,871 MiB` peak RSS** (`2,836 MiB` with `presets/exact.json`) vs `~6,500 MiB` RSS and `8,282 MiB` (BF16) / `10,874 MiB` (FP32) Metal peak memory for `mlx-lm`.
- **Precision & Full-Model Parity vs FP32 `mlx-lm`**:
  - `nano_gemma4_greedy` and `nano_exact_greedy` match `mlx_f32` greedy `prediction_checksum` identically across all 3 benchmark lengths (`2113204`, `3902761`, `2746543`), whereas `mlx_bf16` diverges at `L=1024` (`811990`) and `L=4096` (`2812771`).
  - Short real chat prompt (`L=24` + 32 greedy tokens): `max_abs = 3.34e-04` (`gemma4.json`) / `2.94e-04` (`exact.json`), `cos_min = 1.00000000`, `24/24` top-1 match, `32/32` greedy token match.
  - Long prompt (`L=1280 > sliding_window=1024`, positions `0, 511, 1023, 1024, 1279`): `max_abs = 1.48e-03` (`gemma4.json`) / `9.95e-04` (`exact.json`), `cos_min = 1.00000000`, `5/5` top-1 match across the sliding-window boundary.
  - Note: all runs were taken with both `nano` and `mlx-lm` serialized under `tools/guarded_run.py` on a shared machine (~13–17 load avg).

