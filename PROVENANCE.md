# Source provenance and license scope

This repository started as the standalone native Zig/Metal Flint engine, exported
from the verified engine source snapshot `56166c5` without parent history or
checkpoint files. The later Liquid AI port and Thinking Machines Inkling 2 Mini
megakernel path add the `nano` executable and related sources, tools, Lean
contracts, and benchmark evidence without merging earlier workspace history or
bundling model assets.

The repository contains no Rust/CUDA implementation files or Rust/CUDA build
dependencies from the earlier workspace. `build.zig` compiles local Zig entry
points, `src/metal.m`, and `src/image.m`, links the Apple Foundation/Metal/CoreGraphics/ImageIO
frameworks, and checks the local Lean-generated bindings. Native checkpoint loading, model scheduling,
KV ownership, protocol adapters, proofs, tests and tuning tools are kept here.

## Licensing boundaries

- Original implementation and contributions: MIT, as stated in `LICENSE`.
- MLX-derived quantized arithmetic and attention/numerical scheduling: preserve
  Apple copyright/MIT notices in the shaders and `licenses/mlx-MIT.txt`.
- `mlx-lm` Inkling 2 relative attention (`jundot/mlx-lm` / `ml-explore/mlx-lm#2086`,
  Jun Kim `@jundot`, Apple Inc. MIT): preserve the copyright/MIT notice in
  `src/mega/inkling_tiles.metal` and `licenses/mlx-MIT.txt`.
- Recursive `rec` resident scheduling/barrier adaptation: preserve attribution,
  modification notices, and `licenses/rec-APACHE-2.0.txt`.
- Qwegn (`qwegnai/qwegn`, `@Andrei-cloud` / QwegnAI, Apache-2.0): preserve
  attribution and `licenses/qwegn-APACHE-2.0.txt` for the six adapted Inkling 2
  Mini kernel optimization designs (`src/mega/inkling_fast.metal` and
  `THIRD_PARTY_NOTICES.md`).
- Z Lab (`z-lab/dflash`, MIT) and Inco AI (`incoai/Inkling-2-Mini-DFlash2`,
  `incoai/splash`, Apache-2.0): preserve attribution, `licenses/zlab-dflash-MIT.txt`,
  and `licenses/incoai-APACHE-2.0.txt` for the native DFlash 2 block-diffusion
  speculative decoding engine and kernels (`src/mega/dflash2.zig`,
  `src/mega/dflash2.metal`, and `THIRD_PARTY_NOTICES.md`).
- Checkpoints and tokenizers: not included in this source repository. Their
  licenses are separate; the source MIT license does not relicense model files.

`THIRD_PARTY_NOTICES.md` records the retained upstream components and modifications.
No third-party adaptation is represented as exclusively original code. The
integer proofs certify their documented abstract contracts, not authorship,
floating-point equivalence, compiler correctness, or GPU forward progress.
