# Third-party notices

## Source-workspace license

This standalone extraction retains the source workspace's Elastic License 2.0
in `LICENSE`, including the Baseten licensor notice. No relicensing is implied.
The upstream components described below retain their respective licenses and
attribution. Model weights are not included and have their own licensing.

## Recursive `rec` resident execution design

The resident Metal decode path in `src/resident.metal` adapts the persistent-worker and atomic arrival/phase barrier approach from [`recursive-oss/rec`](https://github.com/recursive-oss/rec), particularly:

- `engine/src/rec_mlx/kernels/megakernel.py`
- `engine/src/rec_mlx/kernels/device.py`
- `engine/src/rec_mlx/kernels/resident_block.py`

Upstream is licensed under the Apache License, Version 2.0. A copy is included in `licenses/rec-APACHE-2.0.txt`. Attribution remains in the adapted shader.

Our modifications use native Zig ownership and GPU-address descriptors, existing FP32 and affine-quantized matrix primitives, a Lean-generated instruction plan, checked single-token execution, bounded opt-in workers, and explicit failure reporting. This is not the upstream hardware-tuned BF16 implementation, does not add upstream's eight-token sampling packet, and does not imply upstream support for the target GPU.

The independent Lean proof covers the abstract integer schedule/control contract and the generated plan binding. It is not an upstream proof and does not certify Metal's compiler, memory model, numerical kernels, GPU occupancy, or forward progress.

## MLX quantized matrix-vector arithmetic

The masked-nibble activation prescaling and two-word, 16-input lane layout in the INT4/group-64 portions of `src/kernels.metal` and `src/row_tiling.metal` are adapted from MLX's [`mlx/backend/metal/kernels/quantized.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h), including the INT4 `qdot` / `qmv_fast` approach. Multi-row input reuse follows the same design direction; expert selection, native descriptors, model scheduling, FP32 integration, and verification are implemented here.

## MLX-layout attention and numerical scheduling

The optional `src/attention_mlx.metal` path adapts the float32 D128 single-pass
and two-pass vector attention from MLX v0.32.4's
`mlx/backend/metal/kernels/sdpa_vector.h`. It retains the source's adjacent-four
lane layout, interleaved stripes, learned sink ownership, online recurrence and
transpose/reduction association, while using this engine's native descriptors
and ring-cache storage. Its Apple copyright and MIT notice are included in the
shader.

The opt-in normalization, RoPE phase, router GEMV, stable sigmoid and grouped
quantized arithmetic follow the corresponding v0.32.4 `rms_norm.metal`,
`rope.metal`, `gemv.h`, `unary_ops.h` and `quantized.h` implementation strategies.
Native sliding attention also reproduces MLX-LM's full rotating-cache physical
iteration order. Explicit expert-weight rounding preserves a materialization
boundary from MLX's separate multiply and reduction operations; it does not
change model weights or expert selection semantics.

## MLX batched quantized matrix multiply and steel attention (megakernel)

The megakernel's `mlx-arith` prefill kernels in `src/mega/mlx.metal` are ports of
MLX v0.32.4's float32 batched kernels: `affine_qmm_t`, `affine_qmm_t_splitk`,
`QuantizedBlockLoader` and `dequantize` from
`mlx/backend/metal/kernels/quantized.h`, `BlockLoader` and `BlockMMA` from
`steel/gemm/{loader.h, mma.h}`, the split-K accumulation of
`steel/gemm/kernels/steel_gemm_splitk.h`, and the `attention` kernel of
`steel/attn/kernels/steel_attention.h` with `steel/attn/{loader.h, mma.h}`. The
host-side choices in `src/mega/exec.zig` (`mlxVectorLimit`, `mlxSplitK`) follow
`mlx/backend/metal/quantized.cpp` (`get_qmv_batch_limit`, `qmm_splitk`). The
tiles, operand order and accumulation order are MLX's, so that the outputs
equal stock MLX's bit for bit (`tools/nano_mlx_ops.py` checks this against an
installed MLX); buffer addressing, the ring caches, scale-dtype specialisation,
the multi-matrix launch and the key origin for sliding windows are this
engine's. The Apple copyright and MIT notice are included in the shader.

MLX is copyright © 2023 Apple Inc. and licensed under the MIT License; a copy is included in `licenses/mlx-MIT.txt`. This engine does not link to or execute MLX. MLX is used separately only for optional reference comparisons and benchmarks. The integer proofs do not establish IEEE754 equivalence of MLX's or this implementation's reductions; unchanged numerical checks remain required.
