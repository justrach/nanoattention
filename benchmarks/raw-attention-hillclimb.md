# Additional bit-exact raw attention hill climb

These trials continue `raw-decode.md`: ordinary native single-token decode, **no speculation**, FP32 activations/KV, seed 20260328, genuinely evaluated 16K/32K random-token prefixes, 256 teacher-forced decode IDs, five rotated repeats, untimed prefix restore/warmup and full-vocabulary admission. Every admitted variant matched all 150,144 logit values bit-for-bit at all 256 positions. Raw data and binary/input hashes: `raw-attention-hillclimb.json`.

**130 tok/s is not achieved. Neither experiment is promoted to serving defaults.**

## Candidates

- `wide_attention`: four disjoint threadgroup transpose planes replace four reuses of one scratch plane. Single/merge kernels keep the original per-component arithmetic and SIMD reduction order but use one threadgroup barrier, trading 16 KiB scratch for fewer barriers.
- `ring_attention`: specialize the single-attention kernel for the complete physical capacity-513/window-513 ring. Before the ring fills, or for other shapes, the host retains the original kernel. The specialization removes runtime modulo and uses a constant loop bound, preserving physical-slot order and all FP32 operations.
- Comparison configuration: two-head reuse, two-token prefetch, affine loads, gate/down row tile 1. The original one-head reference remains candidate zero for full-vocabulary admission.

## Paired measurements

| Trial / configuration | Median raw decode tok/s |
|---|---:|
| 32K wide trial, comparison | 85.49 |
| 32K wide trial, wide transpose | 86.10 |
| 16K ring trial, comparison | 96.48 |
| 16K ring trial, ring specialization | 96.72 |
| 16K ring trial, ring + wide merge | 96.35 |
| 32K ring trial, comparison | 84.95 |
| 32K ring trial, ring specialization | 85.21 |
| 32K ring trial, ring + wide merge | 85.09 |

These sub-percent gains are not compelling evidence for a default change. The earlier raw trials reached approximately 106 tok/s at 16K and 93 tok/s at 32K, while later trials were slower across comparison and reference configurations too. No cause is established; do not compare different trials as paired speedups or present the earlier best result as guaranteed sustained throughput.

## Verification and scope

ReleaseSafe build and proof-binding checks pass. Six targeted headless native tests pass, covering exact single/merge output bits, full-ring specialization across wraps and sink dtypes, unsupported metadata/threadgroup rejection, and existing Metal primitives. The full native suite was not rerun in this phase.

`proofs/Transpose.lean` checks bounds and unique ownership of the four 32x32 scratch planes. `proofs/Prefetch.lean` additionally proves that valid physical ring indices need no modulo. These are integer layout/index proofs, not proofs of Metal synchronization semantics, floating-point compilation or speed. Device tests and full-model admission check those operational results separately.

The new variants remain opt-in via native benchmark candidates `wide_attention` and `ring_attention`. There is no lower-precision KV, batching or speculative decoding hidden in these results. No commit or push was made. All owned inference jobs exited and the exclusive inference lock was released.
