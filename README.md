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

### Use as Codegraff's local model

`tools/codegraff_adapter.py` exposes loopback OpenAI chat/function-tool endpoints
backed by one persistent native `flint session`. It uses the checkpoint's own
local tokenizer/template, preserves exact native KV prefixes across compatible
tool turns, and explicitly reports resets when history changes. All inference
remains Zig/Metal; Python is only the transport/tokenizer/protocol bridge.
See [Codegraff setup and verification](docs/CODEGRAFF.md) and the
[project-router example](examples/codegraff-router.json). SSE is buffered, not
token-live streaming. Full context allocation is not a filled-window benchmark.

## Performance and correctness

The optimized staged configuration is selected explicitly with **`--staged-row-tile 2`**; tile 1 remains the default. A validated pre-extraction run of these kernels on an **M3 Ultra, 60 GPU cores, 256 GB unified memory** measured:

| Engine | Median decode tokens/s |
| --- | ---: |
| Zig/Metal FP32 activations, original packed INT4 weights | **121.18** |
| Stock MLX FP32, same local checkpoint/workload | 96.33 |
| Stock MLX BF16, same local checkpoint/workload | 94.19 |

The three native runs measured **121.55 / 120.85 / 121.18 tok/s**. Workload: 21 prompt tokens and 64 teacher-forced decode tokens, three warmed repetitions, fresh/reset caches, synchronous GPU completion and CPU greedy readback. Loading and warmup are excluded. This is not long-context, free-running generation, or server throughput. Native prefill is serial and remains substantially slower than MLX's batched prefill. See [benchmark evidence and methodology](benchmarks/README.md), including the standalone-folder rerun.

**This does not show a win over Recursive's own `rec` benchmark.** Its [published chart](https://github.com/recursive-oss/rec/blob/main/.github/assets/throughput.svg) reports **131.2 tok/s at 1K context** on an M4 Pro with 48 GB, 4-bit weights, five runs and 1,024 output steps; it falls to 22.6 tok/s at 252K context. Our hardware, context, and decode protocol differ. A matched-device/checkpoint/context/protocol comparison is required.

Validation includes **43 unit/kernel tests**, nine Lean-binding tests, six verifier-behavior tests, and seven synthetic checkpoint cases against independent NumPy (worst absolute logit error 1.37e-6). Full-checkpoint checks of the pre-extraction kernels against unsteered FP32 MLX compared every vocabulary logit: coding top-1 agreement **36/36 positions**, held-out linked-list **28/28**, and **16/16 greedy tokens** matching for each prompt. Maximum absolute logit errors were **0.0002823 / 0.0002670**, with unchanged allclose tolerances. These checks establish bounded numerical agreement, not bitwise, broad quality, long-context, or default-BF16 parity.

To reproduce optional full-model checks, use a Python environment with the reference packages installed and run engines sequentially:

```sh
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 600 -- \
  /path/to/mlx-python -B tools/compare_mlx.py /path/to/flint-int4 --staged-row-tile 2

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 600 -- \
  /path/to/mlx-python -B tools/benchmark.py /path/to/flint-int4 \
  --decode-tokens 64 --repeats 3 --staged-row-tile 2 --output .zig-cache/benchmark.json
```

The guard refuses a held lock, checks memory headroom, and cleans up only its owned process group. It does not stop other servers. Omit `--lock` only when no other inference stack needs coordination. Native and MLX full models never overlap within these drivers. `FLINT_PROFILE=1` enables diagnostic per-kernel timestamps; profiling perturbs scheduling, so its throughput is **not** a benchmark.

## Decode-only hill climbing

`tools/tune_decode.py` compares head/down row tiles, packed router launches and
parallel expert-down scheduling against one real cached prefix. Every vocabulary
bit at every tested decode position must match before a candidate is timed;
round order rotates to reduce machine-state bias. The 4K/16K trials found small
kernel gains but **no meaningful 16K wall-time improvement**, so production
defaults remain unchanged. Losing busy-wait code was removed, not enabled.
See [decode trials, reproduction commands and RAM accounting](benchmarks/decode-hillclimb.md).

Zig provides explicit allocation, ownership and scheduling control—not automatic
GPU speed or lower RAM than MLX. Both use Metal/unified memory. Our KV remains
FP32; equivalent BF16 KV would use half the cache storage. Mapped checkpoint
size, process RSS and device allocations are different measurements.

## Longer-context MLX-layout mode

The default-off `--mlx-attention` path now passes **bit-exact all-vocabulary
comparisons against stock sequential MLX** on the tested 4K/8K prefixes and
16-token continuations. No tolerances, weights or routing decisions were relaxed
or steered. Global attention stays **NoPE**; only the checkpoint's sliding layers
use RoPE.

Same-machine M3 Ultra, ReleaseSafe, 256 teacher-forced decode tokens and three
warmed repeats:

| Actual prompt tokens | Native tok/s | MLX FP32 tok/s | MLX BF16 tok/s |
| --- | ---: | ---: | ---: |
| 1,024 | **126.94** | 87.14 | 88.81 |
| 4,096 | **119.72** | 81.86 | 85.73 |
| 8,192 | **110.75** | 73.14 | 83.66 |

`--prompt-tokens` creates actual exact-length repeated-code prefixes, not larger
allocations with short prompts. `--reuse-prefill` measures one real native
prefill, snapshots its actual KV/position and restores outside decode timing.
Native prefill has **one sample**. Stock MLX performance runs use batched prefill;
strict numerical checks explicitly select sequential prefill. These are decode
measurements, not prefill-performance or broad task-quality parity. The entire
256-token timing continuation is not all-logit validated.

```sh
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/compare_mlx.py /path/to/flint-int4 \
  --prompt-tokens 8192 --mlx-prefill-chunk-size 1 --decode-tokens 16 \
  --staged-row-tile 2 --mlx-attention

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/benchmark.py /path/to/flint-int4 \
  --prompt-tokens 8192 --decode-tokens 256 --repeats 3 --reuse-prefill \
  --staged-row-tile 2 --mlx-attention --output .zig-cache/matched-8192.json
```

The native executable accepts `--mlx-attention 1`; Python drivers accept
`--mlx-attention`. This mode conflicts with resident execution and
`--attention-chunk`. See [current results, fixes and scope](benchmarks/mlx-layout-long-context.md)
and [individual timings/source fingerprints](benchmarks/mlx-layout-long-context.json).
The [earlier state-merge experiments](benchmarks/long-context.md) failed numerical
admission; their speed figures are historical, not current accepted results.
The separate ordered `--attention-chunk` path is still experimental and is not
the mode measured above.

## Persistent multi-turn sessions and KV reuse

`flint session` keeps one loaded model and its actual per-layer KV buffers alive
while reading JSONL requests from stdin. Each request supplies **only new suffix
token IDs**; the native session never replays earlier turns or resets implicitly.
EOF closes the process. This is a local stdio worker, not an HTTP server.

```sh
zig-out/bin/flint session /path/to/flint-int4 \
  --context 32768 --mlx-attention 1 --staged-row-tile 2
```

Example request shape (use the checkpoint's tokenizer/template for real chat):

```json
{"id":1,"tokens":[150002,123,456],"max_tokens":256,"reset":false,"return_logits":false}
```

Responses report generated IDs, `cache_before/after`, `reused_prefix_tokens`, new
prefill/decode counts, commit-inclusive timings, and allocated/valid KV bytes.
Every sampled token—including final EOS or the last length-limited token—is
forwarded into KV before returning. Context capacity reserves that final commit.
A requested `reset` is explicit and validated before mutation. Full-attention
layers retain the sequence; sliding layers keep their bounded 513-token ring.
KV remains FP32, not quantized. Editing earlier history requires an explicit
reset; this is append-only session reuse, not a cross-conversation prefix trie.

Staged generation, sessions, and benchmarks compute the terminal normalization
and vocabulary projection only for the **last new prefix token**. Every prefix
embedding, transformer layer, routing decision, KV write and position increment
still executes. This removes unused output-head work, not context or attention.
Decode still computes full logits for each forwarded token. Diagnostic `eval`
continues to compute every supplied token's logits; resident execution retains
its original certified full-forward plan. This is not batched prefill. See the
[controlled prefill optimization measurements](benchmarks/prefill-elision.md).

`tools/check_sessions.py` checks real persistent tiny checkpoints against both an
independent NumPy implementation and fresh native replay, including ring wrap,
EOS/final-token commitment, reset, byte accounting and overflow rejection.
`tools/coding_multiturn.py` provides an offline three-turn coding exercise with
strict tokenizer/template prefix checks. It refuses to reset, truncate or replay
history to conceal a mismatch. Native and MLX reference invocations are separate;
artifacts live in ignored `target/`. Generated code is syntax-checked only,
**never executed** by this driver.

The large-context matched schedule selects more stripes at 16K/64K. Contexts of
at least 16K currently require the measured M3 Ultra/GQA8 profile; this is not
universal hardware auto-tuning. The configured context ceiling is not a claim
that every length has been benchmarked or quality-validated.

See [16K/32K timings and the executed coding/cache trial](benchmarks/larger-context-sessions.md),
including initial-prefill latency, incomplete responses, and verification limits.

## The megakernel and LFM3 (`nano`)

`zig build` also produces **`nano`**, a second executable built on a model-agnostic core (`src/mega/`). A model is described as a program of generic operations over declared state (activations, append-only and ring KV, short conv windows); a per-architecture front end emits the program from `config.json`, and nothing below the front end knows which model it runs. The first model on it is [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) (24 layers: 13 gated short convolutions, 6 full-attention and 5 sliding-attention), from its official 4-bit MLX export. `nano` has `inspect`, `generate`, `eval`, `bench` and `session`; `nano session` speaks the same JSONL protocol as `flint session`.

Every performance choice is a run-time knob. With no knobs, the engine commands (`session`, `eval`, `bench`) run the plain reference executor (one dispatch per operation), which is what everything else is checked against. Two presets select the tuned paths; `nano run` and `nano chat` ([below](#nano-as-a-command-line-tool)) use the first by default:

```sh
./zig-out/bin/nano session /path/to/lfm3-1.7b-mlx-4bit --context 4608 --knobs src/mega/presets/exact.json
./zig-out/bin/nano session /path/to/lfm3-1.7b-mlx-4bit --context 4608 --knobs src/mega/presets/mlx-prefill.json
```

- `exact.json`: every kernel bit-identical to the reference executor, for decode and for batched prompt ingestion. Decode is latency-hidden (below); the tokens and logits are the reference's.
- `mlx-prefill.json`: decode as above; prompt ingestion through ports of stock MLX's batched quantized matmul and attention. Each ported kernel is bit-identical to the stock MLX operation on the same inputs (`tools/nano_mlx_ops.py check`), so batched ingestion is no longer bit-identical to token-at-a-time decoding, exactly as in MLX.

Same M3 Ultra, same 4-bit checkpoint, each engine in its own process under one lock, interleaved, three rounds, medians; 256 decode tokens after an actual prompt of that length.

Decode, tokens a second:

| Prompt tokens | `exact` | `mlx-prefill` | `nano`, plain synchronous | Stock MLX FP32, pipelined | Stock MLX BF16, pipelined | Stock MLX FP32, synchronous | Stock MLX BF16, synchronous |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | **313** | **313** | 241 | 278 | 286 | 190 | 198 |
| 1,024 | **303** | **303** | 245 | 274 | 275 | 189 | 197 |
| 4,096 | **280** | **279** | 237 | 264 | 275 | 185 | 194 |

A synchronous loop (run one forward, wait, read the token on the host) loses about a fifth of every token to submission and wake latency around the GPU work. Stock MLX's generate loop hides it by submitting the next step before reading the current token (`mx.async_eval`); that is the "pipelined" column, and it is the fair baseline. The presets hide it differently: the next forward is committed early and held by a one-thread GPU gate until the host has read the current token and written the next one, so the host still reads and chooses every token ([how](docs/MEGAKERNEL.md), [measurements](benchmarks/lfm3-decode-search.md)). Against stock's pipelined loop the presets are 10% / 10% / 2% ahead of BF16 and 13% / 11% / 6% ahead of FP32 at 64 / 1,024 / 4,096 tokens in this session; an earlier session gave 8% / 5% / 1% against BF16, so read the 4,096 margin over BF16 as level. Synchronous against synchronous, `nano` is 22 to 24% ahead of stock BF16.

Prompt ingestion, tokens a second:

| Prompt tokens | `exact` | `mlx-prefill` | Stock MLX FP32 | Stock MLX BF16 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1,268 | **3,498** | 2,627 | 2,795 |
| 1,024 | 1,882 | **5,489** | 4,600 | 5,365 |
| 4,096 | 1,745 | **5,477** | 4,661 | 5,461 |

With MLX's own arithmetic, ingestion is ahead of stock MLX in both precisions while staying float32: by 25% at 64 tokens, and by 2% and 0.3% over stock BF16 at 1,024 and 4,096 (the last is level, within drift). Its kernels keep stock MLX's output bits and change only tile shape, staging and scheduling. Exact ingestion, bit-identical to token-at-a-time decoding, is about a third of that at the longer lengths: its 4-input chains run on the GPU's 8 x 8 matrix units and each row's attention runs in parallel ([both tiers](benchmarks/lfm3-prefill-search.md)).

The machine was shared with other workloads; these are one session's medians, not a controlled benchmark. Against goldens produced by stock MLX in float32, the engine's logits are within float32 rounding on 23-, 700- and 4,096-token prompts with identical greedy tokens.

The speed came from a quality-diversity search over exact candidates ([method](docs/QD.md), [archive](benchmarks/lfm3-qd/), [decode](benchmarks/lfm3-decode-search.md), [prefill](benchmarks/lfm3-prefill-search.md)). See also [the megakernel's design and how to add a model](docs/MEGAKERNEL.md), [the model as implemented](docs/LFM3.md), [what stock MLX does on prefill](docs/MLX-PREFILL.md), and [the first measurements](benchmarks/lfm3-megakernel.md). Limits: one sequence, float32 activations, greedy decoding in `nano session` (the command-line tool below also samples), no rewind or prefix cache in `nano session` yet, and Flint itself is not ported to the megakernel.

### `nano` as a command-line tool

`nano` also takes text: a native tokenizer (identical to the `tokenizers` library on a 51,299-string corpus), the model's chat template and the tuned presets built in, no Python.

```sh
export NANO_MODEL=/path/to/lfm3-1.7b-mlx-4bit
./zig-out/bin/nano run "What is the capital of Australia? Answer in one sentence."
./zig-out/bin/nano chat
./zig-out/bin/nano tokenize "Hello world"
```

`run` streams one reply (`--stats`, `--json`, `--system`, `--think`, sampling options, `--prefill mlx` for MLX's faster batched prompt arithmetic); `chat` keeps the model's state between turns. See [docs/CLI.md](docs/CLI.md).

## Execution architecture

- `src/model.zig`, `weights.zig`, `safetensors.zig`: validated model metadata, mmap-backed shared checkpoints, expert descriptors, bounded sliding/full KV caches, and eager scheduling.
- `src/metal.zig`, `metal.m`: a narrow headless Foundation/Metal bridge with cached pipelines, explicit resource residency, dependency barriers, and completion/error handling.
- `kernels.metal`, `row_tiling.metal`: packed INT4/group-64 matvec, descriptor-backed QKV launch fusion, shared gate/up input preparation plus SwiGLU, and fused down/mixture/shared-expert/residual operations.
- `dense_matvec.metal`, `routing.metal`: dtype-specialized loads, eight-SIMD small-output projection reductions, and register-cached sigmoid/top-k with deterministic ties and finite checks.
- `attention_parallel.metal`, `normalization.metal`: eight-SIMD online attention, learned denominator-only sinks, split-half RoPE, parallel RMSNorm, and QK/cache/residual normalization fusion.
- `attention_split.metal`, `attention_ordered.metal`: historical experimental state-merge and ordered score/value paths; no cross-threadgroup spin barrier.
- `attention_mlx.metal`, `normalization_aligned.metal`: opt-in MLX-layout reductions, physical sliding-ring traversal, adjacent-four normalization and matching RoPE phase construction.
- `proofs/`, `src/resident_plan.{zig,metal}`: checked integer contracts and exact generated bindings consumed by the runtime.

The original short-context staged path uses **one submission and 489 dispatches per token** for Flint. The measured MLX-layout long-context mode uses **one submission and 503 dispatches per token**. Neither is a single monolithic kernel. Flint semantics include attention in the first two dense layers, 47 routed MoE layers, six selected experts plus an ungated shared expert, selection-only router bias, a 513-token sliding window, and RoPE only on sliding layers. Activations, accumulations, and caches remain FP32.

An opt-in resident interpreter (`--resident-workers 1..32`, optional `--resident-row-tile 1|2|4|8`) executes the generated plan in one dispatch. It supports at most 4096 allocated context positions plus checked dimension limits. It is slower than the winning staged path. Failed synchronization poisons the sequence rather than silently falling back. Unsupported higher-worker experiments failed numerical checks despite completion receipts; those results are not admitted.

## What Lean proves—and does not

See [proof documentation](proofs/README.md). Certificates cover execution-plan ordering, instruction/index bounds, guarded row coverage and uniqueness, INT4 reconstruction and ideal integer affine algebra, QKV tile selection, split-dense column ownership, split-attention token/state/sink ownership and index bounds, and an abstract arrival-counter transition under explicit distinct-worker premises. Generated constants/helpers bind the selected contracts to Zig and Metal.

**They do not prove whole-shader correctness**, IEEE754 equivalence, Metal compiler or memory-model correctness, occupancy, fairness, liveness, or speed. Those remain assumptions and empirical verification obligations. Standard Lean logical axioms appear in some proofs; there are no added axioms, `sorry`, `admit`, or `native_decide` shortcuts.

## Current limits

No batched prefill, lazy tensor graph, autograd/training, batching, speculative decoding, native tokenizer/template renderer, HTTP server, tool-call parser, or multi-device execution. Generation currently implements greedy selection, not Flint's recommended stochastic sampling. mmap avoids an extra full weight copy but does not guarantee enough physical memory or device buffer capacity. Strict numerical checks cover the tested 4K/8K prefixes and the recorded multi-turn coding stream through 28,623 cached tokens against sequential FP32 MLX. The coding check compares every greedy choice and six all-vocabulary boundary rows, not every logit at every position. The separate 32K timing fixture has no additional full-model all-logit validation claim. Batched-reference equivalence, contexts beyond the measured range and broad task-quality parity remain unverified.

## License and attribution

The extraction retains the source workspace's **Elastic License 2.0** in [LICENSE](LICENSE); no relicensing is implied. Adapted Recursive `rec` components retain Apache-2.0 notices, and adapted MLX quantized arithmetic retains MIT notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. Model weights are not bundled and have their own licensing.
