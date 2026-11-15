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

MLX is copyright © 2023 Apple Inc. and licensed under the MIT License; a copy is included in `licenses/mlx-MIT.txt`. This engine does not link to or execute MLX. MLX is used separately only for optional reference comparisons and benchmarks. The integer proofs do not establish IEEE754 equivalence of MLX's or this implementation's reductions; unchanged numerical checks remain required.
