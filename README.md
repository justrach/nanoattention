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
python3 tools/check_sessions.py
python3 tools/check_prefix_cache.py
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
tool turns, rewinds to the exact common token prefix when earlier history
changes, and reports every append, rewind, reset and disk restore. All inference
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
KV remains FP32, not quantized. Editing earlier history needs an explicit
`truncate` (below) or `reset`; without them the session is append-only.

Staged generation, sessions, and benchmarks compute the terminal normalization
and vocabulary projection only for the **last new prefix token**. Every prefix
embedding, transformer layer, routing decision, KV write and position increment
still executes. This removes unused output-head work, not context or attention.
Decode still computes full logits for each forwarded token. Diagnostic `eval`
continues to compute every supplied token's logits; resident execution retains
its original certified full-forward plan. This is not batched prefill. See the
[controlled prefill optimization measurements](benchmarks/prefill-elision.md).

### Rewind and the persistent prefix cache

Both are opt-in and exact: the state after a rewind or restore is bit-identical
to a fresh sequential run of the same tokens. Without the new flags and fields,
session output is unchanged. The differences from the session protocol
specification they implement are listed below.

- `"truncate": N` rolls the cache back to its first `N` history tokens before
  the request's own tokens (not with `reset`; `N` at most the cached length).
  Full-attention KV needs no copy: lowering the position hides later entries.
  The 513-slot sliding rings do, so the session keeps in-memory checkpoints of
  the rings (73.5 MB each for Flint) at the end of every request's prefill and
  decode and at every `checkpoint_at` position; `--rewind-checkpoints N` (0–64,
  default 4) bounds them. A rewind restores the newest checkpoint at `c <= N`
  and replays recorded history `c..N` through the normal prefill path.
- `"checkpoint_at": [P, ...]` marks absolute positions inside the request's
  own prefill span (strictly increasing). Prefill runs in segments ending at
  each position; the extra work is one output head per segment.
- `--prefix-cache DIR` writes a disk snapshot at each `checkpoint_at` position:
  the token IDs and only the valid KV (positions `0..P` of full layers, the
  valid ring slots, physical order). On `reset: true` the session restores the
  longest snapshot whose stored token IDs equal a prefix of the request, then
  prefills the rest. `--prefix-cache-bytes B` caps the directory (default
  4 GiB); least recently used files (by mtime, refreshed on use) are evicted
  first and a snapshot larger than the cap is not written.
- A snapshot is used only if its format version, checkpoint identity (SHA-256
  of `config.json` plus every shard's name, size, safetensors header and 32
  evenly spaced 4 KiB pages; the 22 GB of weights are not hashed, so an edit
  that misses every sampled page is not detected), and settings key match.
  The settings key covers the Metal device name, a SHA-256 of the running
  `flint` executable (any rebuild invalidates old snapshots), and every engine
  setting that selects kernels or storage (attention mode, KV dtype, row tiles,
  attention chunking and split-attention eligibility, resident mode,
  speculation and the other tuning fields). The configured context is not part
  of the key: full-attention layout does not depend on it, it changes kernel
  choice only through split-attention eligibility (in the key), and ring
  capacity is checked per layer. Writes go to a hidden temporary file renamed
  into place. Truncated files fail a size check before any KV is touched; other corruption
  fails an XXH3 checksum over tokens, layout and state. Files that fail for the
  current key are removed; files for other checkpoints, settings or versions are
  left alone. Temporary files left by a writer that no longer exists are removed
  when a session opens the directory.

Measured on the real checkpoint with a 2,682-token chat-templated system prompt
and tool list ([details](benchmarks/prefix-cache.md)): its snapshot is 227 MB
and restores in 0.22 s, replacing about 23 s of prefill; a new process answered
its first request in 3.3 s instead of 24.7 s (about 2.4 s of the 3.3 s is any new
process's first-forward warmup). Within one session, `truncate` back to the
prefix and a different 37-token user turn prefilled in 0.29 s. Every compared
row's all-vocabulary logits and greedy tokens were bit-identical to a fresh run.

New response fields appear only when used (no new flag or field gives the
same output as before): `truncated_from`, `rewind_checkpoint`, `replayed_tokens`,
`replay_seconds`, `rewind_source` (`none`, `ring_intact`, `memory`, `disk` or
`start`) after a `truncate`; `restored_prefix_tokens`, `restore_seconds`,
`snapshots_written`, `snapshot_bytes`, `snapshot_seconds`, `snapshot_failures`
with `--prefix-cache`; `checkpoints_taken` with `checkpoint_at`; and
`checkpoint_seconds` (ring copies) with any of them. `prefill_seconds` counts
only prefill compute, not restores, replays, ring copies or snapshot writes.

Differences from `PROTOCOL.md`, each the closest correct behaviour:

- A disk restore always leaves at least one request token to prefill, because
  snapshots do not store logits. A snapshot equal to the whole request is not
  used; `tokens` stays nonempty, so to rewind without new input send
  `truncate: N-1` with the `N`th token.
- `prefill_tokens` keeps its meaning (the request's tokens appended to the
  cache); `prefill_tokens - restored_prefix_tokens` of them were computed.
- A rewind needs no checkpoint while no ring slot has been reused (`ring_intact`:
  the cached length is at most the ring capacity), and it may restore a disk
  snapshot of `history[0..c]` when that is longer than the best in-memory
  checkpoint (`disk`). A disk restore is also kept as an in-memory checkpoint.
- Checkpoints from `checkpoint_at` and disk restores are evicted only after
  every automatic checkpoint, so the system-prompt boundary survives many turns.
  `checkpoint_at` positions inside a restored prefix are skipped (the restore
  point itself counts); `checkpoints_taken` reports how many were honoured.
- A failed snapshot write is reported on stderr and in `snapshot_failures`,
  not fatal: the live cache is unaffected. A failed lookup (for example a
  removed directory) is also reported and not fatal: a reset then prefills
  everything, and a rewind replays from position 0, since the lookup may have
  written into the cache. Files are not `fsync`ed; the checksum detects a torn
  file after a crash.
- In-memory checkpoints are host copies of the rings, taken even when no
  `truncate` ever follows (the client cannot know in advance): one ring copy
  per checkpoint, up to `--rewind-checkpoints` × 73.5 MB of host memory for
  Flint. Their time is outside `prefill_seconds` and `decode_seconds`.

`tools/check_prefix_cache.py` checks rewind and the disk cache on tiny F32 and
INT4 checkpoints: every rewound or restored row's prefill and final logits are
bit-identical to a fresh native `eval`, close to the independent NumPy model,
and its greedy tokens match NumPy. It covers truncation to a checkpoint, between
checkpoints, after ring wrap, to 0, before any ring reuse and without
checkpoints; restore into a new process, restore then continue then truncate,
a snapshot written after a truncate and restored as the longest match,
truncation served from disk, a different context, and rejection of snapshots
from a different checkpoint, changed config, changed weight bytes with the same
sizes, or different settings, plus corrupt, truncated and oversized files and a
cache directory removed under a live session. With `--baseline BINARY` (built
from the base commit) it also requires plain session responses to be
byte-identical to that binary's, timing fields aside.

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

No batched prefill, lazy tensor graph, autograd/training, batching, speculative decoding, native tokenizer/template renderer, HTTP server, tool-call parser, or multi-device execution. Generation currently implements greedy selection, not Flint's recommended stochastic sampling. mmap avoids an extra full weight copy but does not guarantee enough physical memory or device buffer capacity. Strict numerical checks cover the tested 4K/8K prefixes and the recorded multi-turn coding stream through 28,623 cached tokens against sequential FP32 MLX. The coding check compares every greedy choice and six all-vocabulary boundary rows, not every logit at every position. The separate 32K timing fixture has no additional full-model all-logit validation claim. Batched-reference equivalence, contexts beyond the measured range and broad task-quality parity remain unverified. The session prefix cache is a local directory for one engine at a time: concurrent processes sharing a directory are safe only to the extent of atomic renames and checksums, eviction is not coordinated between them, and snapshots never carry across machines or rebuilds.

## License and attribution

The extraction retains the source workspace's **Elastic License 2.0** in [LICENSE](LICENSE); no relicensing is implied. Adapted Recursive `rec` components retain Apache-2.0 notices, and adapted MLX quantized arithmetic retains MIT notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`. Model weights are not bundled and have their own licensing.
