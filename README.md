# nanoattention

Native **Zig + Metal** inference for [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), with Lean-generated integer execution contracts. Zig owns checkpoint mappings, buffers, model scheduling, and KV caches; Metal kernels do the model computation. The runtime does **not** link MLX, PyTorch, CUDA, or a Python model implementation.

This is a single-sequence, inference-only engine—not a general MLX replacement or a production HTTP server. The executable is named `flint`.

## Requirements

- Apple Silicon macOS and Xcode/Command Line Tools with the macOS SDK.
- **Zig 0.17.0** and **Lean/Lake 4.35.0**. The Lean project uses `Std` only.
- Python 3.9+ for proof-binding checks; NumPy for synthetic numerical verification.
- Optional `transformers`/`tokenizers` for text input, and MLX/MLX-LM for reference comparisons.
- An existing compatible Flint checkpoint and sufficient unified memory. Weights are not included or automatically downloaded.

## Build and verify

Run from the repository root:

```sh
zig build -Doptimize=ReleaseSafe
zig build test -Doptimize=ReleaseSafe
zig build proofs
python3 -B proofs/test_verifier.py
zig test --dep plan -Mroot=proofs/check_bindings.zig -Mplan=src/resident_plan.zig

# Requires NumPy; creates disposable tiny checkpoints.
python3 tools/verify.py zig-out/bin/flint --staged-row-tile 2
```

Native builds automatically build the Lean proofs and reject generated Zig/Metal binding drift. All GPU verification is headless: no windows, activation, or desktop input.

## Run Flint

The native loader supports F32/F16/BF16 safetensors and MLX-style affine INT4/INT8 exports, including U8/U32 packed storage, per-group scales/biases, mixed-width embeddings, stacked experts, and sharded checkpoints. Dequantization happens inside Metal kernels; weights stay in mapped shared memory. NVIDIA NVFP4 is not supported. The published BF16 checkpoint is approximately 80 GB and has not received the same full-model validation as the tested 22.57 GB INT4 export.

```sh
./zig-out/bin/flint inspect /path/to/flint-int4
./zig-out/bin/flint generate /path/to/flint-int4 \
  --tokens-file prompt.json --context 2048 --max-tokens 64 --staged-row-tile 2
```

`prompt.json` is an array of token IDs, not text. Generation is greedy and emits `{"tokens":[...]}`; EOS is included and ends generation. `--context` caps prompt plus generated positions. Use the checkpoint's own tokenizer and chat template.

For direct text input, the optional adapter applies that template and decodes IDs; **all model inference still runs in Zig/Metal**:

```sh
/path/to/python -B tools/chat.py /path/to/flint-int4 \
  "Write a Python binary search function." \
  --context 2048 --max-tokens 64 --staged-row-tile 2
```

`flint eval` emits all vocabulary logits for every supplied token. It is diagnostic output, not an efficient serving protocol. `inspect` validates configuration only, not the complete checkpoint payload.

## Performance and correctness

The optimized staged configuration is selected explicitly with **`--staged-row-tile 2`**; tile 1 remains the default. A validated pre-extraction run of these kernels on an **M3 Ultra, 60 GPU cores, 256 GB unified memory** measured:

| Engine | Median decode tokens/s |
| --- | ---: |
| Zig/Metal FP32 activations, original packed INT4 weights | **121.18** |
| Stock MLX FP32, same local checkpoint/workload | 96.33 |
| Stock MLX BF16, same local checkpoint/workload | 94.19 |

The three native runs measured **121.55 / 120.85 / 121.18 tok/s**. Workload: 21 prompt tokens and 64 teacher-forced decode tokens, three warmed repetitions, fresh/reset caches, synchronous GPU completion and CPU greedy readback. Loading and warmup are excluded. This is not long-context, free-running generation, or server throughput. Native prefill is serial and remains substantially slower than MLX's batched prefill. See [benchmark evidence and methodology](benchmarks/README.md), including the standalone-folder rerun.

**This does not show a win over Recursive's own `rec` benchmark.** Its [published chart](https://github.com/recursive-oss/rec/blob/main/.github/assets/throughput.svg) reports **131.2 tok/s at 1K context** on an M4 Pro with 48 GB, 4-bit weights, five runs and 1,024 output steps; it falls to 22.6 tok/s at 252K context. Our hardware, context, and decode protocol differ. A matched-device/checkpoint/context/protocol comparison is required.

Validation includes **42 unit/kernel tests**, six Lean-binding tests, six verifier-behavior tests, and seven synthetic checkpoint cases against independent NumPy (worst absolute logit error 1.37e-6). Full-checkpoint checks of the pre-extraction kernels against unsteered FP32 MLX compared every vocabulary logit: coding top-1 agreement **36/36 positions**, held-out linked-list **28/28**, and **16/16 greedy tokens** matching for each prompt. Maximum absolute logit errors were **0.0002823 / 0.0002670**, with unchanged allclose tolerances. These checks establish bounded numerical agreement, not bitwise, broad quality, long-context, or default-BF16 parity.

To reproduce optional full-model checks, use a Python environment with the reference packages installed and run engines sequentially:

```sh
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 600 -- \
  /path/to/mlx-python -B tools/compare_mlx.py /path/to/flint-int4 --staged-row-tile 2

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 600 -- \
  /path/to/mlx-python -B tools/benchmark.py /path/to/flint-int4 \
  --decode-tokens 64 --repeats 3 --staged-row-tile 2 --output .zig-cache/benchmark.json
```

The guard refuses a held lock, checks memory headroom, and cleans up only its owned process group. It does not stop other servers. Omit `--lock` only when no other inference stack needs coordination. Native and MLX full models never overlap within these drivers. `FLINT_PROFILE=1` enables diagnostic per-kernel timestamps; profiling perturbs scheduling, so its throughput is **not** a benchmark.

## Execution architecture

- `src/model.zig`, `weights.zig`, `safetensors.zig`: validated model metadata, mmap-backed shared checkpoints, expert descriptors, bounded sliding/full KV caches, and eager scheduling.
- `src/metal.zig`, `metal.m`: a narrow headless Foundation/Metal bridge with cached pipelines, explicit resource residency, dependency barriers, and completion/error handling.
- `kernels.metal`, `row_tiling.metal`: packed INT4/group-64 matvec, descriptor-backed QKV launch fusion, shared gate/up input preparation plus SwiGLU, and fused down/mixture/shared-expert/residual operations.
- `dense_matvec.metal`, `routing.metal`: dtype-specialized loads, eight-SIMD small-output projection reductions, and register-cached sigmoid/top-k with deterministic ties and finite checks.
- `attention_parallel.metal`, `normalization.metal`: eight-SIMD online attention, learned denominator-only sinks, split-half RoPE, parallel RMSNorm, and QK/cache/residual normalization fusion.
- `proofs/`, `src/resident_plan.{zig,metal}`: checked integer contracts and exact generated bindings consumed by the runtime.

The fast staged path uses **one submission and 489 dispatches per token** for Flint. It is not a single monolithic kernel. Flint semantics include attention in the first two dense layers, 47 routed MoE layers, six selected experts plus an ungated shared expert, selection-only router bias, a 513-token sliding window, and RoPE only on sliding layers. Activations, accumulations, and caches remain FP32.

An opt-in resident interpreter (`--resident-workers 1..32`, optional `--resident-row-tile 1|2|4|8`) executes the generated plan in one dispatch. It supports at most 4096 allocated context positions plus checked dimension limits. It is slower than the winning staged path. Failed synchronization poisons the sequence rather than silently falling back. Unsupported higher-worker experiments failed numerical checks despite completion receipts; those results are not admitted.

## What Lean proves—and does not

See [proof documentation](proofs/README.md). Certificates cover execution-plan ordering, instruction/index bounds, guarded row coverage and uniqueness, INT4 reconstruction and ideal integer affine algebra, QKV tile selection, split-dense column ownership, and an abstract arrival-counter transition under explicit distinct-worker premises. Generated constants/helpers bind the selected contracts to Zig and Metal.

**They do not prove whole-shader correctness**, IEEE754 equivalence, Metal compiler or memory-model correctness, occupancy, fairness, liveness, or speed. Those remain assumptions and empirical verification obligations. Standard Lean logical axioms appear in some proofs; there are no added axioms, `sorry`, `admit`, or `native_decide` shortcuts.

## Current limits

No batched prefill, lazy tensor graph, autograd/training, batching, speculative decoding, native tokenizer/template renderer, HTTP server, tool-call parser, or multi-device execution. Generation currently implements greedy selection, not Flint's recommended stochastic sampling. mmap avoids an extra full weight copy but does not guarantee enough physical memory or device buffer capacity. Long-context and broad task-quality parity remain unverified.

## License and attribution

The extraction retains the source workspace's **Elastic License 2.0** in [LICENSE](LICENSE); no relicensing is implied. Adapted Recursive `rec` components retain Apache-2.0 notices, and adapted MLX quantized arithmetic retains MIT notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. Model weights are not bundled and have their own licensing.
