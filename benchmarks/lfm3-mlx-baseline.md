# LFM3-1.7B stock MLX baseline

The stock MLX speed the native LFM3 engine is compared against, on the `LiquidAI/LFM3-1.7B-MLX-4bit` checkpoint (4-bit affine, group 64, 0.97 GB). The raw data is in `lfm3-mlx-baseline.json`, written by `tools/lfm3_benchmark.py`.

## Protocol

- **Machine**: Apple M3 Ultra, 256 GB, macOS 27.0. Shared with other workloads: load averages were 12 to 16 during the run (Zig builds and other processes were running), with the inference lock held.
- **Software**: mlx 0.32.4, mlx-vlm 0.8.0 (stock `mlx_vlm.models.lfm3`, no patches, no custom kernels), Python 3.12.10, in a dedicated virtual environment.
- **Command**: `python3 tools/guarded_run.py --lock $NANO_LOCK --timeout 3000 -- $NANO_MLX_PYTHON -I tools/lfm3_benchmark.py $LFM3_CHECKPOINT --output benchmarks/lfm3-mlx-baseline.json`
- **Workload**: the prompt is BOS then `benchmark.py`'s `CONTINUATION` code token ids, repeated and cut to exactly 64, 1,024 or 4,096 tokens (raw, no chat template). This is a performance workload, not a quality test. Each prompt is followed by 256 teacher-forced decode tokens (the same `CONTINUATION` ids). Every engine gets identical ids (`input_sha256` in the JSON).
- **Prefill**: as stock `mlx_vlm.generate` does it: chunks of at most 2,048 tokens over the first L - 1 tokens (cache state evaluated, logits never materialized), then the last token with logits and a host argmax. Prefill time is from cache creation's end to that argmax.
- **Decode**: one cached forward per token plus a synchronous host argmax (`.item()`), as in `benchmark.py`. Stock `generate` overlaps steps with `async_eval`, and that pipelining is **not** measured here.
- **Repeats**: per engine and length, one warm-up run (prefill plus 8 decode tokens), then 3 measured runs, each with a fresh cache. The median is reported. Loading, `set_dtype`, cache creation and warm-up are excluded. Greedy checksums were identical across repeats.
- **Engines**: `mlx_bf16` is the checkpoint's own dtype and stock's default. `mlx_f32` is `model.set_dtype(float32)`, the precision of the goldens. Packed weights are 4-bit in both.

## Result (`lfm3-mlx-baseline.json`)

| engine | prompt tokens | prefill tok/s | decode tok/s (256 tokens) |
| --- | --- | --- | --- |
| mlx_bf16 | 64 | 2,487 | 160.1 |
| mlx_bf16 | 1,024 | 5,007 | 144.8 |
| mlx_bf16 | 4,096 | 5,396 | 178.3 |
| mlx_f32 | 64 | 2,575 | 189.2 |
| mlx_f32 | 1,024 | 4,620 | 148.3 |
| mlx_f32 | 4,096 | 4,518 | 157.2 |

A second identical run under the lock straight afterwards (not saved in `benchmarks/`) gave the following:

| engine | prompt tokens | prefill tok/s | decode tok/s |
| --- | --- | --- | --- |
| mlx_bf16 | 64 / 1,024 / 4,096 | 2,256 / 5,013 / 4,711 | 163.1 / 164.0 / 148.0 |
| mlx_f32 | 64 / 1,024 / 4,096 | 2,419 / 4,058 / 4,492 | 162.9 / 159.5 / 159.9 |

## Limits

- **Decode is bound by per-token host overhead here, not by memory bandwidth or attention.** It sits at 145 to 190 tok/s (5 to 7 ms per token) whatever the context length and dtype. Reading the 0.97 GB of weights once would take about 1.2 ms at this machine's bandwidth. Individual runs varied by up to about 20%, and once by 35% (one 107.5 tok/s run). Medians moved by up to 17% between the two passes. Treat differences of under about 20% between these rows as noise. On the loaded machine, that also applies to comparisons with a native run.
- The ordering of bf16 against f32 is not stable between passes. BF16 activations are not faster here, because decode is not compute-bound.
- Prefill at 64 tokens is mostly fixed cost (about 25 ms). At 1,024 and 4,096 tokens it is 4,000 to 5,400 tok/s.
- This is synchronous decode. Stock `mlx_vlm.generate` with `async_eval` pipelining would likely be faster per token; that was not measured.

## Timing the native engine with the same script

`tools/lfm3_benchmark.py CKPT --engines native --native zig-out/bin/nano [--session-arg ARG …]` drives `nano session CKPT …` over the JSONL session protocol. Each run sends `{"reset": true, "tokens": prompt, "max_tokens": 256}` on identical prompt ids and takes the rates from the response's `prefill_seconds` and `decode_seconds` / `decode_forward_tokens`. The session protocol cannot teacher-force, so native decode is free-running greedy (still one forward per token), and the generated token count is recorded. Pass `--engines native,mlx_bf16,mlx_f32` to time both in one guarded run; the native child exits before MLX loads. The script was exercised end to end against a stock-MLX stand-in for the session (`tools/lfm3_mlx_session.py`, with the reference venv first in `PATH`). It has not timed a real native binary, because none existed when this was written.
