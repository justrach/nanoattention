# Terminal-head elision during sequential prefill

The retained optimization runs the final RMSNorm and vocabulary projection only
for the last token of each new prefix. It still executes every transformer layer,
router, attention operation and KV write at the original absolute position.
Generation, persistent sessions and the native benchmark use `Model.prefill`;
diagnostic `eval` retains full forwards. The resident interpreter keeps its
original generated schedule and falls back to complete forwards.

This is not batched prefill and does not reduce decode attention or KV precision.
The terminal head cannot contribute to later KV entries: the next token starts
from its own embedding, and its input normalization overwrites the scratch norm.
Intermediate vocabulary outputs are no longer computed or checked for finiteness;
per-token router error checks and final-output checks remain.

## Measurement protocol

- Apple M3 Ultra, 60 GPU cores, 256 GiB unified memory; ReleaseSafe, Zig 0.17.
- Existing packed INT4 Flint checkpoint, FP32 activations and KV; no conversion.
- Staged row tile 2, MLX-layout attention; actual 16,384-token raw repeated-code
  prefix, 256 teacher-forced decode forwards, context allocation 16,640.
- Same saved token arrays for every executable. Prefix inference is real; only
  subsequent decode repeats restore a snapshot, outside the timer.
- One timed prefix sample per executable, three warmed decode repetitions.
  Loading, compilation, snapshot restoration and decode warmup are untimed.
- Exclusive inference lock; native models run one at a time, before MLX loads.
- Old and new executable fingerprints and raw timings are in the companion JSON.

A separate one-SIMD-per-row router experiment removed seven idle SIMD groups
while preserving the FP32 tree. It passed bitwise kernel and full-model checks,
but did **not** improve end-to-end GPU decode time. That experiment was removed;
the retained router geometry and all decode kernels are unchanged. Wall-clock
variation alone is not sufficient evidence to keep a kernel change.

## Retained result

| Measurement | Before | Retained |
| --- | ---: | ---: |
| 16,384-token benchmark prefill, wall seconds | 158.133 | 152.407 |
| Same prefill, GPU seconds | 133.579 | 128.403 |
| Persistent-session initial prefill, wall seconds | 151.345 | 145.458 |
| Next 57-token suffix with existing KV, wall seconds | 0.5952 | 0.5767 |

The final-binary benchmark reduces prefill wall latency **3.62%** and GPU latency
**3.87%**. The independent persistent-session check reduces initial-prefix latency
**3.89%**. These are individual timed prefixes, not medians over many initial
prefills. An earlier head-only run measured 145.462 seconds; it is retained in
the evidence rather than selected as the headline result.

Decode is **not** a demonstrated improvement: the three-run median changed from
89.82 to 86.41 tok/s, while median GPU decode time changed from 2.4573 to 2.4392
seconds. The decode kernels are unchanged and wall time varied materially between
trials. Do not present the prefill gain as a token-generation speedup, or compare
these trials directly to older 16K results under different machine conditions.

## Accuracy and KV checks

- All **256 teacher-forced predictions** match the original executable in every
  benchmark repetition.
- At 16K, two persistent turns end at positions **16,400** and **16,473**. All
  **32 generated tokens** match the original; four all-vocabulary boundary rows
  are **bit-exact**, with identical cache positions and allocated/valid KV bytes.
- Stock **sequential FP32 MLX**: the last 4,096-token-prefill row and 15 cached
  decode rows are **bit-exact**; all 16 greedy choices agree. Prefix generation
  uses the optimized prefill API, while diagnostic replay computes full logits.
- Final ReleaseSafe build and **60 native tests** pass; **52 Python tests** pass.
  F32 and quantized BF16 tiny persistent sessions pass independent NumPy and fresh
  native replay, reset, ring wrap, final-token commitment, accounting and overflow.
- The final build checks the existing Lean proofs and generated bindings. No
  resident schedule, proof axiom or floating-point tolerance was changed.

These checks do not prove all prefix logits or every cache element, nor broad
coding quality. The large remaining prefill gap needs a genuinely batched
transformer/MoE path, not additional vocabulary-head elision.

See [raw evidence and executable/source fingerprints](prefill-elision.json).
