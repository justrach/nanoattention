# The megakernel: a model-agnostic execution core

`nano` runs a model described as a **program**: a weight table, a state table and a flat list of generic operations with explicit shapes. A per-architecture **front end** turns `config.json` into that program. Nothing below the front end (planner, executor, kernels, sessions, benchmark) knows which model it runs, so a new architecture built from the existing operations needs no new Metal code. The first and only front end today is LiquidAI's LFM3-1.7B (`src/mega/lfm3.zig`, specification in `docs/LFM3.md`).

"Megakernel" names the design goal (one generic execution core for every model), not a single monolithic GPU kernel: the reference executor issues one dispatch per operation into one command buffer per forward.

| file | role |
| --- | --- |
| `src/mega/ir.zig` | the IR: state kinds, operations, `Program`, `Builder`, structural validation |
| `src/mega/lfm3.zig` | LFM3 front end (`config.json` -> `Program`); the only file that knows LFM3 |
| `src/mega/frontend.zig` | registry: `model_type` -> front end |
| `src/mega/plan.zig` | schedules: the op list grouped into launch steps (reference = one step per op; fusion patterns) |
| `src/mega/exec.zig` | binds a program to a checkpoint and Metal buffers; encodes and runs forwards |
| `src/mega/kernels.metal` | the reference kernels, one generic kernel per operation |
| `src/mega/fused.metal` | fused variants, each behind a knob |
| `src/mega/decode.metal` | single-row variants (split attention, attention prep, two-stage argmax), each behind a knob |
| `src/mega/prefill.metal` | exact multi-row variants (matrix-matrix 4-bit linear, gated conv over rows), each behind a knob |
| `src/mega/decode2.metal` | decode climb 2: the gated pre-encoded decode (gate and gated embedding) and a register-resident sandwich norm, each behind a knob; `mk_d2_empty` for `nano d2-probe` |
| `src/mega/decode2_probe.zig`, `decode2_tests.zig` | `nano d2-probe` (launch, barrier, submission and wake latency, matvec bandwidth); the headless bit tests of `decode2.metal` |
| `src/mega/exact2.metal`, `exact2.zig` | exact multi-row variants on the matrix units (4-bit linear as simdgroup 8 x 8 x 8 steps, batched attention), each behind a knob; `nano exact2-bench` |
| `src/mega/mlx.metal` | the `mlx-arith` tier: ports of stock MLX's batched kernels (`affine_qmm_t`, split K, steel attention), bit-identical to MLX, not to the reference (`docs/MLX-PREFILL.md`) |
| `src/mega/mlxop.zig` | `nano mlx-op`: one MLX-arithmetic operation on given inputs, for `tools/nano_mlx_ops.py` |
| `src/mega/knobs.zig` | the run-time knobs |
| `src/mega/gpu.zig` | the parameter blocks and dispatch over the shared bridge (`src/metal.m`, unchanged) |
| `src/mega/session.zig`, `bench.zig`, `main.zig` | `nano session` (Flint's JSONL protocol), `nano bench`, the CLI |
| `src/mega/kernel_tests.zig` | headless Metal tests of every kernel against plain CPU code |
| `src/mega/candidates/*.json` | the QD seed candidates |
| `tools/nano_verify.py`, `nano_sessions.py`, `nano_bench.py`, `nano_qd.py` | golden checks, session checks, benchmark driver, quality-diversity trials |
| `tools/nano_gate_check.py` | decode climb 2: every admission case at feed 1 with batch 64, so single-row forwards run gated (`d2_gate`), against the pin |
| `tools/nano_mlx_ops.py` | the `mlx-arith` kernels against stock MLX bit for bit (`check`), MLX's and the port's per-operation times (`speed`, `time`) |

## 1. The IR

Everything the GPU touches is 32-bit words. A **state** is a named 2-D array of `width` words per row; its **kind** fixes addressing and lifetime:

| kind | rows allocated | addressed by | lifetime |
| --- | --- | --- | --- |
| `tokens` | batch | batch row | one forward (host-written u32 ids) |
| `scratch` | batch (or a fixed `rows`) | batch row | one forward; undefined between forwards |
| `sequence` | context | absolute position | the session (full-attention K and V) |
| `ring` | `min(horizon + batch, context)` | position mod capacity | the session (sliding K and V, conv windows) |
| `position` | 1 | | host-held count of cached positions; exactly one per program |

A ring's `horizon` is how far back any reader looks before the row being computed (window - 1 for sliding attention, taps - 1 for a causal conv). Capacity `horizon + batch` guarantees that the rows a batch writes never overwrite a row an earlier row of the same batch still has to read, so a batch of T rows sees exactly what T single-row forwards would see. Persistent state is the `sequence` and `ring` buffers plus the position, so a snapshot is those buffers' first `validRows(position)` rows (`ir.State.validRows`) and one integer.

A **view** is a column window `(state, col, width)` of a state's rows. Every operation reads and writes views; row stride is always the state's width. An operation processes `rows` consecutive token rows; row `r` of a forward is absolute position `position + r`.

### Operations

| op | semantics (per row) |
| --- | --- |
| `embed` | `out = table[token]`, dense or affine-quantized; optional bf16 output rounding |
| `rmsnorm` | `out = w * x / sqrt(mean(x^2) + eps)` over consecutive groups of `group` words (whole vector or per head); weight optional; in place allowed |
| `linear` | `out = W x`, `W` [out, in] dense (F32/F16/BF16) or MLX affine-quantized (any bits, any group) |
| `binary` | `out = act_a(a) (+ or *) act_b(b[i / b_every])`; acts identity, silu, sigmoid; `b_every` broadcasts a per-head gate |
| `store` | write the row to its position's slot of a `sequence` or `ring` state |
| `conv` | depthwise causal conv over a ring of past inputs, weight [channels, taps], tap `taps-1` = current position, zeros before position 0 |
| `rope` | rotate-half RoPE on pairs (i, i + rot/2) of each head with a per-pair frequency table; zero frequencies leave pairs unrotated, so partial (zero-tail) rotation is just a table |
| `attention` | grouped-query causal softmax attention over cached K/V of positions `[max(0, p - window + 1), p]` (window 0 = unbounded), fixed scale |
| `argmax` | lowest index of the row maximum; an all-ones sentinel on any non-finite value |

Two program-level fields complete it: `head_start` (ops from here on, the final norm, vocabulary projection and argmax, run only on the rows whose logits are wanted, so prefill skips the 128,000-wide head for all but the last row), and `traces` (views recorded after given ops: the residual stream after the embedding and after each layer, for golden comparison).

`Program.validate` checks every id, that each view lies inside its state, every shape (weight shapes against views, head dims, ring horizons against windows and taps), aliasing that a kernel cannot do in place (`linear`, `attention`), and that there is one position state. The binder then checks every weight's shape and dtype against the checkpoint header (`weights.Store.get`).

### What LFM3 compiles to

465 ops for 24 layers: 1 embed; per conv layer 16 (operator norm, in_proj, `B*X`, store, conv, `C*v`, out_proj, then 9 shared: post norm, add, ffn norm, w1, w3, silu-mul, w2, post norm, add); per attention layer 23 (operator norm, q/k/v/gate projections, q and k per-head norms, two ropes, two stores, attention, sigmoid-gate mul, o_proj, then the same 9); 3 head ops. 57 states: 20 scratch, 12 sequence (6 full layers x K, V), 23 rings (13 conv windows, 5 sliding layers x K, V), tokens, position. `nano inspect CHECKPOINT` prints this.

## 2. The executor

`exec.Engine.init` allocates all `tokens` and `scratch` states in one shared Metal buffer (the arena), one buffer per persistent state, one buffer of RoPE tables, and binds every weight by name and shape to the memory-mapped checkpoint. `forward(tokens, out_first)` writes the ids, encodes one step after another into one command buffer (the shared bridge puts a buffer barrier between consecutive dispatches), runs head steps only for rows `[out_first, T)`, waits, and advances the position. Logits and the GPU argmax of a row are then read straight from the arena. A failed forward poisons the engine until `reset`. `reset` is logical: every slot at a non-negative position is written before it is read and conv reads before position 0 are zeros, so stale rows are never observed.

### Exactness rule

Every reference kernel computes each token row with an expression tree that depends only on that row's data and absolute position, never on the batch size, the row's index in the batch, or how rows are grouped into threads. Reductions have a fixed order: RMSNorm uses a threadgroup size that is a function of the group length alone; the 4-bit linear is MLX's `qmv_fast` structure (lane L owns 16-input segments L, L+32, ...; per segment the inputs are prescaled by powers of two and summed, codes masked in place, and the row accumulates `scale * dot + sum * bias`; one `simd_sum` at the end), and its row and token tiles only choose which SIMD group computes a row; attention is an eight-SIMD online softmax whose visit order is a function of the row's own position range; conv sums taps in ascending order. This is why batched ingestion is bit-identical to one token at a time, and why every variant must keep each row's expression tree (section 4).

## 3. Correctness, as verified

Commands and the numbers they produced are in `reports/megakernel.md`. In short:

- `tools/nano_verify.py` runs `nano eval` on a golden case and compares per-layer residual streams, all-vocabulary logits at the stored positions, top-1 at every stored position and the greedy continuation. The gate is fixed: largest absolute error at most `2 x float32_noise_floor + reference_error` (the golden producer's own distance from float64), top-1 equal everywhere, greedy identical. With several `--feeds` every feed must also be bit-identical to the first. `--f64` additionally compares with the independent NumPy float64 model.
- `tools/nano_sessions.py` checks `nano session` on the tiny checkpoints (against the float64 reference, every turn) and on the real checkpoint (continued session equals a fresh replay bit for bit, batch 1/64/256 bit-identical, EOS, byte accounting).
- `zig build test` runs the IR, front-end, planner, knob and session unit tests and the headless kernel tests.

## 4. Knobs and variants

A **knob** selects among compiled-in variants at run time: `nano ... --knobs FILE.json` and/or `--knob name=value` (flags apply after the file). Unknown names and invalid values fail closed. The defaults are the **reference executor**: one dispatch per op, no fusion. It is always available and is what everything is checked against.

| knob | values | effect |
| --- | --- | --- |
| `batch` | 1..4096 (default 64) | token rows per prefill forward; decode is always 1 row |
| `decode_rows`, `prefill_rows` | 1, 2, 4, 8 | output rows per SIMD group of the 4-bit linear (single-row and multi-row forwards) |
| `decode_tokens`, `prefill_tokens` | 1, 2, 4 | token rows per SIMD group of the 4-bit linear |
| `linear_group` | 32..1024, multiple of 32 | threads per threadgroup of the 4-bit linear |
| `fuse_projections` | bool | up to 4 consecutive 4-bit linears sharing one input in one launch (q/k/v/gate) |
| `fuse_glu` | bool | linear, linear, binary (gate, up, silu-mul) in one launch; intermediates never stored |
| `fuse_norms` | bool | rmsnorm, residual add, rmsnorm (sandwich post-norm + next pre-norm) in one launch |
| `fuse_conv` | bool | mul, store, conv, mul (the gated short conv) in one launch |
| `fuse_gate` | bool | attention with the per-head sigmoid gate as epilogue |
| `decode_attention` | 0, 2 | single-row attention: reference, or split into a parallel scores launch and a block chain over the stored scores (decode.metal) |
| `fuse_attn_prep` | bool | q/k per-head norms, both ropes and the K/V stores of an attention layer in one launch |
| `decode_argmax` | 0, 1 | argmax: one threadgroup per row, or two-stage (64 slices) |
| `prefill_mm` | bool | multi-row 4-bit linears through `mk_q4_prep` plus the exact matrix-matrix kernels (prefill.metal); tile `mm_rows` x `mm_tokens` = 2 x 8, `mm_half`, `mm_group` threads |
| `conv_rows` | bool | multi-row fused gated conv as two launches (store every row, then every (row, channel) in parallel); needs `fuse_conv` |
| `head_trim` | bool | rows without logits stop after the program's last store to a persistent state (`exec.trimmedHead`); traced forwards keep the program's head |
| `mlx_linear` | bool | **mlx-arith tier**: multi-row 4-bit linears at or above MLX's vector limit through the port of MLX's `affine_qmm_t` / split K |
| `mlx_attention` | bool | **mlx-arith tier**: attention of more than 8 rows (head dim 128) through the port of MLX's steel attention |
| `mlx_multi` | bool | with `mlx_linear`: unsplit MLX products sharing an input in one launch |
| `mlx_qmm2` | bool | (c2-prefill-mlx) with `mlx_linear`: the products stock MLX runs unsplit (plain and `mlx_multi`) through `mk_mlx2_qmm` / `mk_mlx2_multi` (mlx2.metal): stock's output bits in 64 x 16 tiles, x read from device memory, double-buffered dequantized weights |
| `mlx_glu2` | bool | (c2-prefill-mlx) with `mlx_linear` and `fuse_glu`: gate, up and the GLU product in one `mk_mlx2_glu` launch, equal bit for bit to the two products then `mk_binary` |
| `mlx_attention2` | bool | (c2-prefill-mlx) with `mlx_attention`: `mk_mlx2_attention[_gated]`, the port's arithmetic rescheduled (heaviest query blocks first, queries in registers, two barriers per key block) |
| `conv_rows2` | bool | (c2-prefill-mlx) with `conv_rows`: `mk_gated_conv_rows2`, the same arithmetic without a 64-bit modulo per tap |
| `d2_gate` | bool | single-row forwards with logits: while token k runs, the forward for k + 1 is encoded and committed behind a one-thread GPU gate that the host opens only after it has read token k and written token k + 1 (section 4a); needs `batch` >= 2 (else ungated) |
| `d2_gate_spins` | 1.. (default 200,000) | the gate's bound in atomic loads; a gate that gives up first costs a synchronous recomputation, never a wrong token |
| `d2_norm_regs` | bool | single-row forwards: the fused sandwich norm keeps y, r and both weights in registers (`mk_d2_norm_add_norm`) instead of storing r and reading it back |
| `x2_linear` | bool | multi-row 4-bit linears through `mk_x2_prep` + `mk_x2mm_q4` (exact2.metal): each quad chain of the reference tree is one simdgroup 8 x 8 x 8 step; projections and GLU run as their separate linears and the reference binary op |
| `x2_half` | bool | with `x2_linear`: the unpacked codes staged as half in threadgroup memory (exact for masked 4-bit codes) |
| `x2_attention` | bool | multi-row attention (head dim up to 256, 1 to 8 query heads per KV head) as scores, softmax scalars and value chains over all rows (exact2.metal) |

None of these changes any row's arithmetic except the `mlx_*` knobs, which replace it by stock MLX's (section 5, "Two tiers"); QD admission (below) checks that bit for bit rather than trusting it. Fusion is a planner pass (`plan.build`) over structural patterns (op kinds, views, activations, weight formats and liveness of intermediates via `plan.deadAfter`), so it applies to any program with the pattern, not to LFM3 by name.

### 4a. Gated pre-encoded decode (`d2_gate`)

Measured with `nano bench --decode-detail 1` (split-attn, 64-token prompt): of 4,146 us a token, 59 us is host encoding, 3,281 us command-buffer GPU time, and about 800 us is neither: about 580 us from `commit` to the GPU starting the buffer and about 200 us from the GPU finishing it to `waitUntilCompleted` returning. Spinning on the status or on a completion handler does not shorten the wake; back-to-back command buffers start 0.6 us apart; a host `MTLSharedEvent` signal takes about 620 us to start a waiting buffer (`nano d2-probe`).

With `d2_gate`, `exec.d2GatedForward` commits the forward for position k + 1 while token k runs, as one command buffer whose first launch is `mk_d2_gate_packed`: one thread spins (bounded) until the gate word carries the forward's generation. The host spins on token k's argmax word (written about 2 us before the GPU end time, after every other launch of the forward), returns it, and when called with token k + 1 writes `generation << 20 | token` into the gate word and encodes k + 2 behind a new gate. `mk_d2_embed_gated` is `mk_embed` with the token read from that word; every other launch is the ordinary one, so the arithmetic is the reference's. The host still reads every token before the next forward starts and chooses the next input, as in the synchronous protocol; only the command encoding and submission are done ahead.

If the gate gives up before the host opens it, the embedding sees a stale generation, writes NaN, the argmax returns the non-finite sentinel and the host recomputes that token synchronously (`d2_redos`). A forward run past the last token (at a stop, or when the host settles before any other forward) writes only rows of positions not yet committed; the next forward at that position rewrites them before reading, provided every wrapping ring keeps a row beyond its horizon plus the current one (`exec.d2GateSafe`: capacity >= horizon + 2, so `batch` >= 2). `Engine.d2Settle` opens a waiting gate and drains; every non-gated forward, `logits` and `deinit` go through it or wait for the current buffer. One extra launch a token (198). `nano_qd.py trial` admits with `batch` = feed, so at feed 1 the engine runs ungated; `tools/nano_gate_check.py` re-runs every admission case at feed 1 with batch 64 (every token gated), and with `--spins 1` forces every gate to give up so the recomputation path runs too.

### Adding a variant

1. Write the kernel (in `fused.metal` or a new `.metal` file appended to `gpu.source`). It must reproduce, for every token row, the reference kernels' expression trees: same lane ownership of reduction inputs, same accumulation order, same formulas; only launches, stores of dead intermediates and redundant loads may go. Reassociating a floating-point sum is a different reference, not a variant, and is not admitted (QD.md).
2. Add a knob value (a new field in `knobs.zig` with a default that keeps the reference, or a new allowed value of an existing knob) and, for a fusion, a pattern in `plan.zig` (`Fusion`, `Step`, a `match*` function) plus its encoder in `exec.encodeFused`. Never change a default to make a variant run.
3. Add a headless test in `kernel_tests.zig` that the variant equals its reference dispatch sequence bit for bit on random data, and a planner test that the steps cover every op exactly once.
4. Run it through `tools/nano_qd.py trial` (section 5).

## 5. The quality-diversity loop (`docs/QD.md`)

**Two tiers.** The exact tier admits a candidate only if its logits are bit-identical to the pinned reference executor's (decode and batched ingestion). The `mlx-arith` tier (`docs/QD.md`) admits prefill candidates whose kernels are bit-identical ports of stock MLX's batched kernels (checked against MLX by `tools/nano_mlx_ops.py check`) and which meet the goldens' float32 gate end to end with identical greedy tokens; they live in their own cells (families `mlx-*`). Decode is exact in both. Exact stays the default; `docs/MLX-PREFILL.md` describes what stock MLX does and what the ports match.

`nano bench CHECKPOINT --prompt-tokens N --decode-tokens 256 --repeats 3 [--bench-tokens F]` reports, as one JSON line, prefill and decode tokens a second (one untimed warm-up run, then the repeats from a reset state; medians; loading excluded; decode is teacher-forced with the GPU argmax read back every token), the per-repeat seconds, forwards, dispatches per prefill forward and per decode token, and a prediction checksum that must be equal across repeats. `tools/nano_bench.py tokens` writes the stock-MLX baseline's exact workload (BOS + `tools/benchmark.py`'s code continuation, and its 256 decode ids) and checks it against `benchmarks/lfm3-mlx-baseline.json`; `nano_bench.py run` runs one guarded process per length.

`tools/nano_qd.py` maintains the archive in `$NANO_WORK/qd/lfm3/` (a copy of the LFM3 search's archive is in `benchmarks/lfm3-qd/`):

- `pin-reference BINARY --why ...`: runs the reference executor (default knobs, one token at a time) of a verified binary on every admission case, checks it against the goldens with `nano_verify`'s gate, and stores the fingerprints (sha256 of the all-vocabulary logits at the stored positions, top-1, greedy) in `reference.json`. A binary whose own reference executor no longer reproduces the pin admits nothing: that catches a mutation that edits the reference kernels. Re-pinning needs `--replace --why` and keeps the old pin in `reference-history.jsonl`.
- `trial BINARY CANDIDATE.json --proposer NAME --why ...`: **admission** (the binary's reference reproduces the pin; then the candidate's fingerprints equal the pin on the 7 tiny checkpoints at feeds 1, 7 and its batch, and on `short` and `window` at feeds 1 and its batch, `--long` adds `long`; an optional `--certify-cmd` such as the Lean checker records `certified` on exit 0, otherwise `empirical`), then **quality** under the inference lock (`nano bench` at 64, 1,024 and 4,096 tokens; the candidate's cell per objective and length is objective x bucket x dispatches-per-token bin x family; an occupied cell is contested by interleaving elite and challenger runs, at least three rounds each, and the challenger wins only if its median beats the elite's by more than max(1%, 2 x the elite's own spread)), then the **record** (`trials.jsonl` appended; `archive.json` rewritten atomically under an exclusive `flock` on `archive.lock`; `elites.md` regenerated). Rejected and losing trials stay in `trials.jsonl`.
- `baseline` records the stock-MLX numbers beside the cells; `parents` prints the occupied cells (where mutations start); `render` regenerates `elites.md`.
- Every `nano eval` and `nano bench` on the real checkpoint goes through `tools/guarded_run.py` and the inference lock (`nano_bench.guarded`, which waits while the lock is held or the guard reports too little memory headroom); tiny-checkpoint evaluations do not. A feed longer than the prompt is clamped to the prompt (one forward), since the engine clamps its batch to the context.
- `trial --admission mlx-arith` admits in the second tier (`admit_mlx`: the binary's reference still reproduces the pin; on every tiny checkpoint and on `short`, `window` and `long` the candidate meets the goldens' gate with identical greedy tokens; the largest difference from the exact batched path is recorded; `target/nano-mlx-ops.json` is attached). `--extra-feeds` adds feeds on the real goldens. `--compare CANDIDATE.json` (repeatable) times other candidates interleaved beside the challenger, recorded, contesting no cell.
- `screen BINARY NAME=knobs.json ...` interleaves knob sets at one length (`--metric prefill|decode`) for a quick, locked comparison; results go to `target/screens.jsonl`.
- `tools/nano_bench.py versus BINARY CKPT NAME=cand.json ... --mlx mlx_f32,mlx_bf16` interleaves candidates with stock MLX (`tools/lfm3_benchmark.py` under `$NANO_MLX_PYTHON`) per length and round, each process under the lock.
- `--objectives decode` (or `prefill`) contests and writes only that objective's cells (each bench run still measures both); `--lengths` restricts the prompt lengths; `--rounds` (default 3) sets the interleaved rounds. For screening an idea cheaply, `tools/nano_bench.py run ... --lengths 1024` is one locked process.
- Certificates: `--certify-cmd "python3 tools/lean_certify.py {candidate}"` (the Lean checker, developed on a separate branch) (exit 0 certified, 1 differs, 2 undecided). The Lean port mirrors a snapshot of these sources and answers undecided (recorded `empirical`) when they differ from it.

A candidate file: `{"name", "knobs", "family" (optional; default per-op / fused-partial / fused-layer from the fusion knobs), "hypothesis", "expected": {"decode": percent, "prefill": percent}}`. The trial records whether the measurement agreed with the hypothesis.

## 6. Adding a model

Write `src/mega/<model>.zig` with `build(gpa, config_bytes, checkpoint, options) !ir.Program`: parse `config.json` (fail closed on every field or feature you do not implement), declare states and weights by checkpoint name and logical shape, emit the ops, mark the head, add traces after the embedding and each layer, and register it by `model_type` in `frontend.zig`. Add a CPU test that the program validates and has the expected states and op counts from a config alone (see `lfm3.zig`). Then produce goldens (a NumPy float64 reference written from the specification, tiny synthetic checkpoints with the same tensor names) and run `nano_verify.py`. If the model needs an operation the IR does not have, add it to `ir.zig` (with validation), a reference kernel to `kernels.metal` (with the exactness rule above), its encoder to `exec.encode`, its reads and writes to `plan.access`, and a headless test against CPU code.

## 7. What Flint would need

Flint (the INT4 checkpoint's `config.json`, `model_type` `flint`): 49 layers, hidden 2048, 32 query and 4 KV heads of 128, layers 0 and 1 dense MLP (6,144) and 47 MoE layers (256 experts of 512, 6 routed per token by a sigmoid router with `norm_topk_prob`, plus an ungated shared expert of 1,024), 14 full-attention layers with **no RoPE** and 35 sliding layers (window 513) with RoPE theta 10,000, learned **attention sinks** (denominator only), RMSNorm eps 1e-6, untied 8-bit output head and 8-bit embedding, 4-bit group-64 elsewhere, 4 EOS ids, 22.6 GB in 5 shards.

Already covered by today's operations: the dense layers, GQA attention, sliding rings, RMSNorm, SwiGLU, the shared expert (linears plus an add), untied head (a different weight), 8-bit embedding and head (`embed` and `mk_linear_generic` handle any bit width, slowly), several EOS ids, multi-shard checkpoints (`weights.Store`). **NoPE global layers need nothing**: the front end simply emits no `rope` op for those layers.

New operations, each with a reference kernel, validation, planner access entry and a headless CPU test:

1. **Router top-k** (`topk`): per row, select k of N scores (sigmoid scores, deterministic lowest-index ties, finite check), optionally add a selection-only bias (Flint's config says `router_expert_bias: false`), renormalise the selected weights (`norm_topk_prob`), write k u32 indices and k float weights to scratch views. A unary activation is also needed (today's `binary` always has two operands): a small `unary` op or an `act` field on `linear`. About 150 lines of Zig and Metal plus tests.
2. **Expert linear** (`expert_linear`): `out[j] = W[idx[j]] x` for the k selected experts of each row, from a stacked quantized weight [E, out, in]; the binder needs 3-D weight shapes and per-expert offsets. For exactness each (row, expert) product must use the same per-row expression tree as `linear`; batched prefill then gathers different experts per row, which the reference can do row by row and a variant can regroup by expert. About 250 lines plus tests; this is where all the performance work for Flint would go.
3. **Mixture** (`mixture`): `out = sum_j w[j] * y[j]` in a fixed j order, plus the shared expert through the existing add. About 60 lines.
4. **Attention sinks**: an optional per-head sink weight on `attention`, added to the softmax denominator only (`exp(sink - m)` merged into the total). The merge order must be fixed and documented; to match the existing Flint path bit for bit it would have to follow `attention_parallel.metal`'s order. About 40 lines.

Honest estimate: the front end (`flint.zig`, about 300 lines) plus these four operations is two to four days of work to correct-against-goldens, given goldens (Flint has MLX references in this repository's tools, but no NumPy float64 model like `lfm3_reference.py`, and one would have to be written). It would be correct long before it is fast: the per-op reference would issue about 49 x 25 = 1,200+ dispatches per decode token, against the hand-fused Flint path's 489, and the expert linears need their own variants (grouping rows by expert in prefill, fusing gate/up across the 6 experts) before the megakernel could compete with `flint`. Bit identity with the existing `flint` binary is not expected: its kernels use different reduction orders (`README.md`, "Execution architecture").

## 8. Limits

- One sequence at a time; no multi-row padded batches.
- Activations, caches and accumulation are float32 (KV bytes are 4 per element). No bf16 activation mode, so outputs differ from stock MLX's default bf16 run by up to 0.5 in logits, as the golden documents for MLX itself; they match MLX float32 at the float32 noise floor.
- Every op is a separate dispatch in the reference schedule (465 per LFM3 decode token); there is no persistent single-dispatch interpreter for the megakernel. With `d2_gate` the gate is a bounded one-thread spin on a host-written word, not threadgroups waiting on each other. The bridge puts a barrier between every two dispatches; stock MLX overlaps independent kernels (its encoder is concurrent), which `mlx_multi` recovers only for products sharing an input.
- Exact prefill is about a third of the `mlx-arith` tier's rate at 1,024 and 4,096 tokens: the exact tree (MLX's `qmv_fast` per row) uses the matrix units only for its 4-input quad chains (`x2_linear`), so half of every 8 x 8 x 8 step is zeros and the quad sums, scale and bias terms, per-lane sums and the pair tree over the 32 lane sums are element-wise work. The `mlx-arith` tier is float32 only; with the restructured kernels (`mlx_qmm2`, `mlx_glu2`, `mlx_attention2`) it is ahead of stock MLX float32 at every length and level with or slightly ahead of stock's bf16 prefill at 1,024 and 4,096 tokens (`benchmarks/lfm3-prefill-search.md`).
- `mlxVectorLimit` encodes MLX's `get_qmv_batch_limit` for Ultra GPUs (architecture size `d`, every generation; this machine is `applegpu_g15d`) only; base, Pro and Max GPUs need their rows of that table before `mlx_linear` reproduces MLX's kernel choice.
- `eval --hidden-out` synchronises the GPU after every traced layer; it is for correctness runs only and is slow fed one token at a time.
- Greedy decoding only; no sampling, repetition penalty or speculative decoding. No rewind and no prefix cache in `nano session` (state is easy to snapshot: the persistent buffers' valid rows and the position).
- The 4-bit fast linear needs `group % 16 == 0`, `in % 16 == 0` and even byte offsets (4-byte for float32 scales or biases); anything else takes `mk_linear_generic` (correct, but about 40 times slower on LFM3). LFM3's safetensors data section starts at byte 75,002, so every tensor sits 2 bytes off a word boundary; `mk_word` reads such a code word as two halves (the same word, so the same arithmetic). An earlier version required 4-byte alignment and silently ran every real-model linear through the generic kernel at 7.9 decode tokens a second. Use `FLINT_PROFILE=1` (the shared bridge's per-dispatch profiler) to see which kernels a program actually runs. Fused projections and GLU additionally need all matrices in one buffer with one group size, else the planner leaves them unfused.
- Contexts beyond 4,096 tokens are not covered by any golden.
- Bit identity of a variant with the reference is established empirically (admission), not proved: the library is compiled with `MTLMathModeSafe` (no reassociation), but whether the Metal compiler contracts a multiply-add into an FMA can differ between a fused kernel and its reference sequence. A Lean certificate covers the expression tree, not the compiler.
