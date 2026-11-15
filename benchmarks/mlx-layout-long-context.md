# MLX-layout long-context results

## Measured decode throughput

Apple M3 Ultra, 60 GPU cores, 256 GiB unified memory; native ReleaseSafe;
MLX 0.32.4; the same existing Flint affine-INT4/group64 export. Native uses FP32
activations, accumulation and KV caches. These are **same-machine local MLX
comparisons**, not a reproduction of Recursive's published `rec` benchmark.

| Actual prompt tokens | Native tok/s | MLX FP32 tok/s | MLX BF16 tok/s | Native / MLX FP32 |
| --- | ---: | ---: | ---: | ---: |
| 1,024 | **126.94** | 87.14 | 88.81 | **1.46x** |
| 4,096 | **119.72** | 81.86 | 85.73 | **1.46x** |
| 8,192 | **110.75** | 73.14 | 83.66 | **1.51x** |

Each measurement uses 256 identical teacher-forced decode inputs, an eight-token
warmup, three timed repeats, GPU completion and a synchronous CPU greedy readback
per token. Native computes the entire real prefix once, snapshots actual KV and
position, and restores outside decode timing. Native prefill therefore has only
**one sample**, not three. Model loading and warmup are excluded. MLX benchmarks
use stock batched prefill on fresh caches; native prefill remains sequential.
These rates do not establish prefill-performance parity.

All model runs are headless and serialized; native and MLX full models do not
overlap. No cache positions, prefix tokens, weights or logits are fabricated.
Repeated-code prefixes are deterministic numerical/performance fixtures, not
broad quality evaluations or Recursive's exact prompt.

## Separate numerical acceptance

With stock **sequential MLX prefill**, both long-context checks passed the
unchanged `rtol=2e-3, atol=2e-3` criterion and were actually **bit-exact**:

| Prompt tokens | Checked logit rows | Greedy continuation | Maximum absolute error | RMS error |
| --- | ---: | ---: | ---: | ---: |
| 4,096 | 16 | 16/16 | **0.0** | **0.0** |
| 8,192 | 16 | 16/16 | **0.0** | **0.0** |

Every vocabulary entry is compared in each checked row: the last prefill row
and 15 cached decode rows. All earlier prefix tokens execute, but their logits
are not retained or validated. The full 256-token throughput continuation is
not all-logit validated. Bit identity on these fixtures is not a universal model
or task-quality guarantee, and does not establish bit identity against batched
MLX prefill. Earlier batched-versus-sequential reference differences remain a
separate, explicitly unverified scope.

## What fixed the divergence

The checkpoint is hybrid: **14 global NoPE layers and 35 RoPE sliding layers**.
The implementation preserves its `no_rope_layer_indices`; rotary phases are
never applied to the NoPE layers.

The earlier speed candidate changed FP32 association. Small errors accumulated
in cached activations and flipped near-tied MoE expert choices. The admitted
opt-in path instead follows MLX's observable numerical scheduling:

- Adjacent-four RMS normalization, grouped INT4 dot products, stable sigmoid
  and SiLU arithmetic, and the router's accumulation/shuffle tree.
- MLX's RoPE phase construction only in enabled layers.
- A full sliding ring is traversed in **physical cache order**, matching MLX-LM
  `RotatingKVCache`, rather than chronological order. These orders are equivalent
  over real arithmetic but not necessarily bit-equivalent in FP32.
- Expert products are explicitly rounded to FP32 before summation, matching
  MLX's materialized multiply. Otherwise a fused multiply-add introduced the
  first tiny difference. Routed experts are summed before shared/residual adds.
- Below 1,024 effective keys: 32-SIMD vector attention. Otherwise: 128 interleaved
  producer stripes and a separately ordered 32-SIMD merge, adapted from MLX
  v0.32.4's large-device decode path. This fixed policy was measured on
  `applegpu_g15d`; it is not automatic tuning for every Apple GPU.

The measured long-context path uses **one submission and 503 dispatches per
token**, not one monolithic kernel. It remains opt-in; existing defaults and the
historical resident interpreter are not silently switched.

## Verification and Lean scope

- ReleaseSafe application build passed.
- 51 Zig/unit/headless Metal tests passed, including FP64 attention checks through
  8K, physical-ring ordering, matched quantized paths, and router dtypes/offsets.
- 27 CPU protocol tests, 14 generated-binding tests and six proof-verifier tests
  passed.
- Lean build, exact generated-binding checks and axiom audit passed. Only standard
  logical axioms occur; no `sorryAx` or native-decision axiom.

The certificates establish integer coverage, unique ownership, guarded indexing,
geometry and bounds. They **do not prove IEEE754 equivalence**, Metal compiler or
memory-model correctness, liveness or throughput. Numerical acceptance comes
from the separate native-versus-MLX checks above, not from labeling the shaders
formally verified.

## Reproduction

Use a Python environment with the checkpoint's matching MLX/MLX-LM installation.
The native CLI uses `--mlx-attention 1`; Python drivers use `--mlx-attention`.
Neither may be combined with resident execution or `--attention-chunk`.

```sh
zig build -Doptimize=ReleaseSafe
zig build test

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/compare_mlx.py /path/to/flint-int4 \
  --prompt-tokens 8192 --mlx-prefill-chunk-size 1 --decode-tokens 16 \
  --staged-row-tile 2 --mlx-attention

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/benchmark.py /path/to/flint-int4 \
  --prompt-tokens 8192 --decode-tokens 256 --repeats 3 --reuse-prefill \
  --staged-row-tile 2 --mlx-attention --output .zig-cache/matched-8192.json
```

[Machine-readable results](mlx-layout-long-context.json) retain individual
measurements, checked position ranges and source fingerprints without local
checkpoint paths. [Historical rejected experiments](long-context.md) remain
available and must not be confused with these measurements. These measurements
were captured before publication; fingerprints preserve the tested source revisions.
