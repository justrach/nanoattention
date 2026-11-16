# Opt-in BF16 global KV experiment

This is an explicitly selected **storage-precision experiment**, not the default
engine mode. FP32 KV remains available and default. Global layers store K and V
as BF16, round-to-nearest/ties-to-even; sliding layers and all activation,
attention and output-logit arithmetic remain FP32. Checkpoint weights are
unchanged. It is **not stock MLX BF16 inference**.

```sh
flint session /path/to/flint-int4 --context 32768 \
  --mlx-attention 1 --staged-row-tile 2 --kv-bf16 1
```

The owned Graff launcher accepts `--kv-bf16` before its `--` separator. The
adapter forwards the explicit flag and reports `kv_storage` in content-free
request metadata. Unsupported shape/device/resident combinations fail instead
of reinterpreting an FP32 allocation as BF16.

## Initial measured results

At an actual 16,384-token prefix, five warm 256-step teacher-forced repeats:

| Mode | Median decode throughput |
| --- | ---: |
| FP32 KV, accepted two-head global attention | 104.42 tok/s |
| BF16 global KV, default stripe policy | 105.13 tok/s |
| BF16 global KV, 128 stripes | 106.53 tok/s |
| BF16 global KV, 1024 stripes (after scratch fix) | 99.18 tok/s |

These initial precision-mode runs were sequential, not rotated against one
shared cache: their prefixes are independently evaluated with different storage
precision. The small default-policy speed difference is not strong evidence of
a throughput gain. **130 tok/s is not achieved.** The clear benefit so far is
smaller KV storage, not a large decode speedup. A rotated, bitwise-gated B128 trial found scalar
loads at 106.15, vector loads at 106.08, and vector loads plus parallel expert-down
at 106.52 tok/s. All 256 all-vocabulary rows matched the scalar **BF16** reference,
not unrounded FP32 KV. Vectorization alone was not faster and is not enabled.
Affine-load candidates remain experimental and are not represented by this table.

A separate 16K comparison evaluated the same prefix and 16 teacher-forced input
IDs with both native modes, comparing **17 complete vocabulary rows** (final
prefill and every forced continuation input, including the last next-token
prediction). Every prefix token was actually evaluated.

- Top-1 agreement: **17/17**.
- Maximum absolute logit difference: **2.866234**.
- RMS difference: **0.517756**; relative L2: **0.017353**.
- Mean `KL(softmax(FP32 KV) || softmax(BF16 global KV))`: **0.00033635 nats**.
- All **2,552,448 logits per mode** were finite.

The existing five-repeat timing outputs also agree on **all 256 teacher-forced
top-1 choices** between FP32 and default-policy BF16 global KV; predictions were
stable across every repeat in each mode. This broader forced-choice agreement
does not make the logits bit-exact or establish identical free-running output.

This is a numerical comparison, **not bit-exact parity, free-running generation,
stock-MLX-BF16 parity or a model-quality evaluation**. Agreement on these forced
positions cannot guarantee unchanged generated tokens or coding quality.

At that comparison's allocated context of 16,400, analytic K+V allocation is
1,013,985,280 bytes for FP32 versus 543,764,480 bytes for the hybrid mode: a
470,220,800-byte reduction. Sliding KV remains FP32, so total KV is not exactly
halved. These are buffer-size calculations, **not RSS or total engine RAM**.
The B128 timing run at context 16,640 reported 550,645,760 actual allocated KV
bytes; the extra 240 positions explain the different allocation.

## Correctness and failure accounting

The initial native build and 73 tests passed, including BF16 ties/specials,
all 65,536 widening patterns, Q/K normalization and cache stores, sink dtypes and
byte offsets, and every producer-state/merged-output word against FP32 attention
on independently CPU-BF16-rounded input caches. Lean checks the integer rounding
bias, ties-to-even and finite-value no-wrap formula. Neither the proof nor these
kernel tests establish whole-model accuracy against unrounded FP32 KV.

An experimental 1024-stripe override exposed an undersized scratch allocation
and produced a **GPU address fault**. The automatic schedule had allocated only
512 stripes at this context. The override API now validates and grows scratch
before prefix inference, and the forward guard rejects direct unsafe overrides
before encoding. The regression test passed and the real 16K checkpoint rerun
completed all five 256-step timing repeats at 99.18 tok/s without a fault.
All **74 native tests** passed, including both scalar/vector BF16 kernel gates
and scratch-override rejection/growth. The earlier failed trial is retained as
a failure, not a throughput result or a passing run. The default BF16 stripe
policy did not use that unsupported override.

The actual installed Graff also completed the isolated six-test coding fixture
with explicit BF16 global KV: **five requests, one native worker, four cache-hit
follow-ups and zero failures**. All six unchanged tests passed; the temporary
router was removed. Its configured capacity was 32K but only **3,392 tokens**
were cached, so this is not a filled-32K performance check. The continuation was
not asserted identical to the FP32 run. All **110 Python tests** and the Lean
build/axiom audit pass.
