# Third-party notices

## Original contributions and upstream components

Original code and contributions in this repository are licensed under the MIT
License in `LICENSE`. The upstream components described below retain their
respective licenses, copyright notices, and attribution; the top-level license
does not replace those obligations. Model weights are not included and remain
separately licensed.

## Recursive `rec` resident execution design

The resident Metal decode path in `src/resident.metal` adapts the persistent-worker and atomic arrival/phase barrier approach from **Recursive**'s [`recursive-oss/rec`](https://github.com/recursive-oss/rec) (with upstream contributions from **Chris Cummins** ([`@ChrisCummins`](https://github.com/ChrisCummins)), **Audrey Tang** ([`@audreyt`](https://github.com/audreyt)), and **Oliver Yu** ([`@tc3oliver`](https://github.com/tc3oliver))), particularly:

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

## MLX batched quantized matrix multiply and Steel attention (`nano`)

The `nano` prefill kernels in `src/mega/mlx.metal` and `src/mega/mlx2.metal`
adapt MLX v0.32.4's batched quantized matrix multiply (`quantized.h`), the
Steel GEMM loader, MMA and split-K reduction, and the Steel attention kernel,
loader and MMA. The host dispatch and split-K policies in `src/mega/exec.zig`
follow `mlx/backend/metal/quantized.cpp`. Buffer addressing, ring caches,
scale-dtype specialization, multi-matrix dispatch and sliding-window key origin
are adapted for this engine. The shaders retain Apple's copyright and MIT notices.

## mlx-lm Inkling 2 relative attention (`nano`)

`src/mega/inkling_tiles.metal` adapts the batched relative attention kernel of
`mlx-lm`'s Inkling 2 model (`mlx_lm/models/inkling2/kernels.py`, at `0c4c99192230`
of [`jundot/mlx-lm`](https://github.com/jundot/mlx-lm) / [`ml-explore/mlx-lm#2086`](https://github.com/ml-explore/mlx-lm/pull/2086), primarily authored by **Jun Kim** [`@jundot`](https://github.com/jundot) with contributions from **Pedro Cuenca** [`@pcuenca`](https://github.com/pcuenca) and **Lucas Gelfond** [`@lucasgelfond`](https://github.com/lucasgelfond); MIT, copyright © 2026 Apple Inc.). The tile loop is the
source's; operand addressing (arena words, position-major 16-bit K and V
mirrors, ring slots) and run-time head counts, relative extent and window are
this engine's. `mk_relative_attention_prefill` in `src/mega/inkling_ops.metal`
follows the same kernel. The shader retains the copyright and MIT notice.

MLX is copyright © 2023 Apple Inc. and licensed under the MIT License; a copy is included in `licenses/mlx-MIT.txt`. This engine does not link to or execute MLX. MLX is used separately only for optional reference comparisons and benchmarks. The integer proofs do not establish IEEE754 equivalence of MLX's or this implementation's reductions; unchanged numerical checks remain required.

## Qwegn Inkling 2 Mini kernel optimizations (`nano`)

Six Inkling 2 Mini megakernel optimization designs in `src/mega/inkling_fast.metal`,
`src/mega/plan.zig`, `src/mega/exec.zig`, `src/mega/inkling2_loop.zig`, and
`proofs/Inkling.lean` were studied from and adapted after **[`qwegnai/qwegn`](https://github.com/qwegnai/qwegn)**
(authored by **[`@Andrei-cloud`](https://github.com/Andrei-cloud)** / QwegnAI; studied at `v0.36.0`, commit `632425e928c8`, licensed under the Apache License, Version 2.0 — see `licenses/qwegn-APACHE-2.0.txt`):

- `Sources/QWMetal/inkling2/qw_ink2_common.metalh` — constant-cache FP8 E4M3FN and FP4 pair LUT decoding (`expert_fp_lut`)
- `Sources/QWMetal/inkling2/qw_ink2_moe.metal` — single-threadgroup compact MoE tile building / sparse `M=16/32/64` tiling (`expert_compact_tiles`) and BF16 MoE intermediates (`expert_bf16_io`)
- `Sources/QWMetal/inkling2/qw_ink2_sconv_qkvr.metal` — fused sliding short-conv / global QKV + relative prep (`fuse_qkvr_prep`)
- `Sources/QWMetal/inkling2/qw_ink2_attn.metal` — inline single-row relative bias projection inside striped relative attention (`relative_inline_bias`)
- `Sources/QWMetal/inkling2/qw_ink2_ctrl_batch.metal` and `Sources/QWMetal/inkling2/qw_ink2_ctrl_fast.metal` — fused controller message merge + RMSNorm/BF16 cast (`fuse_merge_norm`)
- `Sources/QWMetal/qwlsa_mh_common.metalh`, `Sources/QWMetal/qwlsa_mh.metal`, `Sources/QWMetal/qwlsa_mh_select.metal`, and `Sources/QWCore/QWLSAOperator+Encode.swift` — Full-Slice Multi-Head Q4 Latent Sparse Attention (`lsa`, `lsa_budget`, `lsa_window`, `lsa_coarse_threshold` in `src/mega/inkling_lsa.metal`)

`nano`'s implementations are written natively for `nano`'s Zig/Metal IR, arena word
layout, and Lean-checked integer helpers rather than being direct ports of Qwegn's
Swift/Metal 4 implementation, while preserving full upstream author attribution and
Apache-2.0 notice.

## LoopCD (Loop Contrastive Decoding, Inkling 2 Mini)

The opt-in `--loop-passes` and `--loopcd` (`hidden`, `logits`, `logits-adaptive`)
contrastive loop decoding paths for Inkling 2 Mini in `src/mega/ir.zig`,
`src/mega/inkling2_loop.zig`, `src/mega/inkling_ops.metal`, `src/mega/exec.zig`,
`src/mega/frontend.zig`, `src/mega/main.zig`, and `src/mega/cli.zig` implement the
training-free Loop Contrastive Decoding methods from:

- **Apple LoopCD** (`arXiv:2610.02185`; reference implementation [`tchayintr/apple-loopcd-off-the-shelf`](https://github.com/tchayintr/apple-loopcd-off-the-shelf)):
  LoopCD-Hidden (`h_out = h_strong + w * (h_strong - h_weak)` prior to the decoder coda),
  LoopCD-Logits (`z_out = z_strong + w * (z_strong - z_weak)`), and adaptive confidence-gated
  LoopCD-Logits (`w = omega * (1 - (p_1 - p_2))`).

All LoopCD IR lowering and Metal kernels here are original implementations written for
`nano`.

## 2DSD (Two-Dimensional Speculative Decoding)

The opt-in CPU scheduler in `src/twodsd.zig`, the tiny-model bench in
`src/twodsd_bench.zig`, and the integer contracts in `proofs/TwoDsd.lean`
follow **2DSD: Two-Dimensional Speculative Decoding for Looped Language
Models** (Kim, Chavan, Seo, Hu, Lee, Kang, Mahoney, Keutzer, Hooper and
Gholami, 2026; announced
[https://x.com/minseokim0525/status/2123533353823662556](https://x.com/minseokim0525/status/2123533353823662556)).
Reference implementation: [`SqueezeAILab/2DSD`](https://github.com/SqueezeAILab/2DSD)
(`paper/2DSD.pdf`, `vllm/v1/spec_decode/looped_2d_finalized.py`). The algorithm,
stage names, last-loop KV window constraint, and rejection-sampling residual
are from that work. The Zig scheduler, tiny looped target, CLI fail-closed
path, and Lean contracts are original to this engine. This is not a port of
their vLLM / CUDA runtime, does not include their drafter or shared-KV
checkpoints, and does not implement Metal overlapping decode.
