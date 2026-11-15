# Decode-only hill climb

## Admission and timing

`tools/tune_decode.py` drives the native `decode_candidates` benchmark protocol.
The first candidate is the unchanged staged tile-2 / original-router baseline.
Head row tiles 1/2/4/8 use the existing certified independent-row kernels. The
packed FP32 router assigns one complete matched dot product to each SIMD group,
packing four rows into a 128-thread group; its per-row accumulation and shuffle
sequence are unchanged. It applies only to MLX-layout FP32 256×2048 routers.

Every candidate receives the same actually computed KV prefix. Before timing,
**every vocabulary bit at every supplied decode position** must match the
baseline. A mismatch rejects that candidate; GPU/inference errors abort the run.
Only admitted candidates are timed, with an untimed eight-token warmup and cache
restore before each repetition. Order rotates between rounds. CPU greedy/finite
checking, command encoding, GPU completion, and readback are included in wall
time. Prefill, admission checks, snapshots and restoration are excluded.

This is teacher-forced single-sequence decode, not free-running generation,
server concurrency, or a claim of broad coding quality. Same-machine fresh
comparisons matter more than historical tok/s recorded under other conditions.

## Why Zig—and what it does not guarantee

Zig owns a small eager scheduler, mapped checkpoint buffers, scratch allocation,
per-layer KV and the Metal launch ABI. This makes resource ownership and kernel
fusion explicit, avoids a Python model runtime in the serving process, and gives
us direct control over specialization. Those choices can improve predictability
and deployment simplicity. They can also lose to MLX's better tuned kernels,
batched matrix operations, graph scheduling or allocator reuse.

The language itself does **not** make GPU arithmetic faster or compress a model.
MLX already runs native Metal kernels on unified memory too. At identical weight
formats and KV precision, weight/cache storage is fundamentally similar. Our
weights remain packed INT4 in mmap-backed buffers, so there is no full expanded
FP32 weight copy. That does not make mapped file size equal resident RAM.

Our KV is currently **FP32**. The same number of BF16 KV elements would occupy
half as many bytes; thus an MLX configuration using BF16 KV can have a smaller
cache than this Zig engine. Lower-precision or quantized KV would be a separate
numerical change, not a free benefit of Zig or part of this hill climb.

The sweep reports measured model-buffer allocations, KV allocations and peak
process RSS. Model-buffer bytes already include KV. RSS includes benchmark-only
KV snapshots and all-logit reference storage, not just serving allocations; it
is not directly comparable to MLX's device allocator statistics. No matched MLX
RSS/device-memory experiment is claimed here. CPU-process RSS also does not
account for all GPU/file-backed weight residency, so the measured RSS must not
be presented as the whole model's unified-memory footprint.

## Measured hill climbs

M3 Ultra, 60 GPU cores, ReleaseSafe, original INT4 checkpoint, FP32 activations
and KV, 256 teacher-forced decode positions; medians of rotated warmed trials:

| Run | Baseline tok/s | Candidate tok/s | Interpretation |
| --- | ---: | ---: | --- |
| 4K, head tiles/packed-router, 5 repeats | 119.71 | 119.29–119.83 | Effectively tied; no default change |
| 4K, parallel expert-down tile 2, 7 repeats | 117.24 | 118.81 | Small 1.34% wall / 1.78% GPU improvement |
| 16K, parallel expert-down tile 2, 7 repeats | 95.86 | 95.94 | No meaningful wall-time win; GPU time ~0.72% lower |
| 16K, parallel expert-down tile 4 | 95.86 | 95.85 | Tied |
| 16K, parallel expert-down tile 8 | 95.86 | 93.88 | Rejected as slower |
| 16K, original fused-down tile 4 / 8 | 95.86 | 91.29 / 83.51 | Rejected as slower |
| 16K, bounded busy-wait completion | 95.65 | 95.70 | No meaningful speedup; CPU cost unacceptable |
| 16K, busy wait + parallel down | 95.65 | 96.19 | Only 0.57% wall gain, not worth the CPU trade-off |
| Final 16K confirmation, parallel down without busy wait | 95.67 | 96.19 | 0.55% wall / 0.81% GPU gain; still too small to promote |

The busy-wait baseline used **0.093 CPU seconds** during a median **2.677 wall
seconds** of decode. Busy wait used **2.711 CPU seconds** during **2.675 wall
seconds**: approximately one CPU core for essentially unchanged speed. That
implementation was removed. Historical busy-wait configs in the evidence are
not supported by the retained executable.

Head/packed-router and down-tiling candidates remain explicit benchmark-only
experiments; **production defaults are unchanged**. A smaller kernel time or a
single higher tok/s sample is not sufficient to promote a configuration. All
admitted modes matched every baseline vocabulary bit across all 256 tested
decode positions, not just greedy choices. This does not establish stock-MLX
agreement at every one of those positions or at untested context lengths.

## Final verification

The retained baseline and tuning harness build in ReleaseSafe; **63 native tests**
and **55 Python tests** pass. Real-prefix snapshot restoration/candidate isolation
and persistent-session NumPy/fresh-native replay pass for F32 and quantized BF16
fixtures, including ring wrap, reset, final-token commitment, accounting and
overflow rejection. The final 4K sequential-FP32-MLX check is **bit-exact** for its
16 sampled all-vocabulary rows and all 16 greedy choices. Existing Lean proofs
and generated bindings verify during the build. No precision/tolerance changes,
commits or pushes were made.

The final confirmation used normal blocking completion. Parallel down consumed
0.095 process CPU seconds versus the baseline's 0.091 during about 2.66–2.68 wall
seconds, unlike the rejected busy-wait mode. It adds 47 dispatches per token
(**503 → 550**, still one submission), explaining why more expert parallelism
can yield a GPU improvement without a substantial end-to-end improvement.

[Raw per-trial timings, memory measurements and fingerprints](decode-hillclimb.json)
preserve both the small gains and the rejected configurations.

## Memory accounting

At the 16,384-token prefix with 16,640 allocated context positions:

- Checkpoint file mappings: **22,568,912,600 bytes** (22.57 GB); not a resident-RAM
  measurement and not an expanded floating-point weight copy.
- Per-layer FP32 KV allocation: **1,027,747,840 bytes** (0.957 GiB).
- All model buffers including KV and candidate temporary storage:
  **1,039,102,596 bytes**. Do not add KV to this again.
- Parallel-down candidate scratch: **57,344 bytes** (56 KiB), allocated only when
  a sweep requests that candidate; ordinary serving does not allocate it.
- The admission reference stores **153,747,456 bytes** of vocabulary bits, and
  prefix restoration has its own KV snapshot. Both are benchmark-only overhead.
- Recorded benchmark CPU-process peak RSS: approximately **2.21 GB**, excluding
  the full weight-residency accounting caveat above.

The earlier real 32K-allocation session measured **1,952,591,872 bytes** (1.82 GiB)
of FP32 KV. Cache growth is determined by sequence length, head geometry,
sliding/full layer counts and element precision—not the host language.

## Reproduce

```sh
zig build -Doptimize=ReleaseSafe
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/tune_decode.py /path/to/flint-int4 \
  --prompt-tokens 16384 --decode-tokens 256 --repeats 7 \
  --candidates '[{"head_row_tile":2},{"head_row_tile":2,"parallel_down":true}]' \
  --output target/decode-confirm.json
```

Omit `--candidates` for the supported 12-configuration kernel sweep. Only use
matching sequential-MLX reference checks for strict numerical admission; this
script does not load MLX or change any reference weights/reductions. Existing
Lean integer contracts/bindings remain checked by the build; they are not a
proof of the new shader, IEEE754 equivalence or GPU performance.
