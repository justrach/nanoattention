# Raw decode: prepared quantized inputs and producer fusion

**Outcome: no winning configuration; 130 tok/s remains unmet.** The candidates are opt-in experiments and serving defaults are unchanged. Data/configurations/input hashes/actual tested binary hashes are in `raw-prepared-inputs.json`.

## Hypothesis and implementation

INT4/g64 projections repeatedly rescale the same input components and recompute each 16-element segment's quartet-ordered sum. This experiment caches those values once per input vector, without changing packed-dot arithmetic, FP32 reduction, routing, residual order or KV precision.

- `row_tiling_prepared.metal` prepares `[N scaled values][N/16 sums]` and supplies qmatvec, QKV, gate/up and down variants for tiles 1/2/4/8. U8/U32 packed storage and FP32/FP16/BF16 affine metadata are supported. Down uses the full activation stride for cached-sum addressing, even when individual experts have narrower intermediate widths.
- Separate preprocessing added dispatch overhead and was slower. `normalization_prepared.metal` therefore fuses preparation into existing aligned normalization/residual-normalization outputs, using SIMD shuffles and no extra barrier.
- `row_tiling_prepared_fused.metal` prepares down inputs in the gate/up epilogue. Each group owns exactly 16 activation rows, uses one barrier, and leaves padding untouched.
- All paths preserve raw normalization/activation buffers. Host validation restricts the candidate to supported INT4/g64 ordinary execution; speculative batch APIs explicitly reject this mode.

## Protocol

Apple M3 Ultra, 60 GPU cores, 256 GiB unified memory; ReleaseSafe native engine. Seeded random raw IDs (`20260328`), real evaluated prefixes, ordinary full-model single-token teacher-forced decode, FP32 activations/KV, **no speculation**. Every candidate must match all 150,144 vocabulary logits bit-for-bit at every tested position before timing. Prefix restoration, admission and warmup are untimed; candidate order rotates.

## Results

Initial separate-preprocessing smoke test at 640 prefix / 64 decode tokens was exact but slower (91.63–97.10 tok/s versus 108.08 for the comparison configuration). It was not promoted.

A subsequent 640 prefix / 128 decode / three-repeat trial measured:

| Configuration | Median raw tok/s | Dispatches/token |
|---|---:|---:|
| Comparison: two-head reuse, TK2, affine loads, gate/down tile 1 | 120.04 | 489 |
| Prepared inputs + fused normalization | 116.26 | 587 |
| Also fused gate preparation, gate/down tile 1 | 111.58 | 538 |
| Both fusions, gate/down tile 2 | 118.88 | 538 |
| Both fusions, gate tile 4 / down tile 2 | 113.84 | 538 |

The decisive filled-32K trial used 256 decode tokens and five rotated repeats:

| Configuration | Median raw tok/s | Dispatches/token |
|---|---:|---:|
| Ordinary comparison | **85.72** | 503 |
| Prepared inputs + fused normalization | 83.96 | 601 |
| Both fusions, gate/down tile 2 | 82.79 | 552 |

All candidates passed full-vocabulary bitwise admission. Preprocessing arithmetic reuse did not compensate for extra dispatches, additional cached-input reads and changed launch geometry. Fusing producers reduced the extra dispatch count but still did not produce an end-to-end improvement. These data reject this candidate; they do not prove a hardware ceiling or that no other kernel design can reach 130.

Absolute rates differed from previous raw trials; no cause is established. Within-trial comparisons above are the relevant evidence, not cross-trial speedup claims. These are raw teacher-forced execution rates, not coding-agent throughput.

## Verification and limits

- ReleaseSafe build passes.
- Full native/headless suite: **117/117 passed** before the final batch-API guard-only patch.
- The final ordinary-only guard passed its focused test (plus the filtered test-root import). Ordinary GPU kernels are unchanged by that patch; timing records retain their original binary hashes.
- `proofs/Prepared.lean` checks scaled/sum region bounds, disjointness, ownership and padded-route sum addressing. Native builds verify it alongside existing Lean bindings. It does not prove GPU floating-point code generation or performance.
- Device tests cover exact prepared bits, all tiles, mixed affine types, U8/U32 storage, unequal down widths/padding, fused normalization aliases/offsets, fused gate geometry and unsupported-input rejection.
- No independent new MLX comparison or quality evaluation was performed. No speed claim is extrapolated to full 262K context.

No commit or push was made. Xcode's separate Metal Toolchain was installed during the worker's offline compilation check. All owned inference jobs exited and the exclusive inference lock was released.
