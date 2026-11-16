# Greedy speculative decoding (experimental)

The opt-in path drafts from **actual prompt and committed output tokens only**.
It does not load a second model, peek at a reference continuation, truncate global
attention, or change FP32 arithmetic. A longest-suffix history match proposes up
to 32 input tokens. The first input is already the target's greedy choice.

The native target verifies input rows layer by layer. QKV, output projection,
gate/up, down, and vocabulary projection dispatches process multiple token rows
together. Each token retains the original row dot products, SIMD reductions and
serial MoE mixture order. This is **batched dispatch**, not one kernel for the
whole network and not a matrix-multiply replacement with reassociated sums.
Sliding cache writes and attention run in increasing token order inside each
layer: a full ring cannot be overwritten by a future draft before an earlier
query reads it. Global NoPE layers still attend to the full valid prefix. Their
batched producer shares K/V loads across two query positions and two GQA heads
per SIMD, with a separate causal length for each position. Global KV is written
before this dispatch, but future positions are excluded from each query. Stripe
schedule transitions fall back to the original per-query path.

A proposal is accepted only when it equals the target's exact greedy token.
At the first mismatch, later rows are discarded and the target's correction is
processed normally. Full acceptance permits a target bonus token. EOS and output
budget apply to both accepted and correction/bonus tokens. Every returned token,
including final/EOS, is committed to KV before a session response.

## Cache transactions

The verifier journals only the sliding slots it will overwrite (at most 32
slots per sliding layer), not a complete long-context cache copy. Rejection
restores the rejected suffix's slots byte for byte. Accepted slots remain;
invalid global suffix entries are hidden by the committed position and will be
overwritten before reuse. A pending batch must be explicitly committed or
rolled back before ordinary forward. A failed GPU frame requires reset.

## Opt-in interfaces

Native generation/session: `--speculative-tokens 2|4|8|16|32` (any count 1–32 is
accepted; zero disables). This requires `--mlx-attention 1`, the supported
INT4/g64 projection format and sufficient cache capacity. FP32 KV remains the
default; `--kv-bf16 1` is a separately requested, accuracy-changing mode.

The owned Graff launcher and manual adapter accept `--speculative-tokens N`
(before the launcher's `--` separator). They keep the existing one-worker,
strict-prefix KV reuse and cleanup behavior. Native/adapter metrics expose
`speculative_batches`, `speculative_proposed`, and `speculative_accepted`.
The first already-known target token is included in the proposal/accept counts.

## Verification boundary

The standalone Lean specification proves equality of the accepted draft prefix
with supplied target predictions and bounds the accepted count by output budget.
It also proves integer token/row/lane coverage, ring-slot injectivity and
accepted/rejected slot separation, and committed-position bounds.
It does **not** prove Metal execution, floating-point parity, the history draft's
quality, or speed. The native batching and rejection implementation must pass
GPU and full-model checks separately.

`tools/check_speculative.py` / `flint-speculative-bench` check every-vocabulary target
rows against ordinary forwards, zero/partial rollback, and exact free-running
greedy output before reporting generation speed. Forced reference tokens are
used only in correctness checks, never as drafts for the timed generation.
Timing includes drafting, verification, rejection/rollback, corrections, sync
and committing final output; prefix preparation and snapshot restore are outside
decode timing. The v2 protocol warms each mode, then rotates ordinary and every
draft-size candidate through each timed round from the same saved prefix. The
provided workload repeats tokenized source code and is deliberately favorable
to history drafting; it is not a representative coding-agent score. Acceptance
and throughput depend on the workload.

The final draft-32/Q2/shared-projection-weight profile reached **166.83 tok/s
at actual 16K** and **139.41 tok/s at actual 32K**, over three rotated rounds,
with exact greedy output and final-logit bits. A GPU reduction computes the
lowest-ID argmax and checks every logit's finiteness before CPU acceptance;
complete logits remain available for correctness checks. No FP32 accumulation
order is changed by moving this reduction.

These are deliberately favorable repeated-code completion results, **not general
coding-agent speed**. The real five-turn Graff fixture passed functionally but
reached only **22.83 native decode tok/s** because its history drafts had poor
acceptance. Speculation remains disabled by default; use it only when its draft
acceptance helps. Larger query groups, expert tile 1 and MLP shared weights were
slower and were not promoted. See retained [trials and limitations](../benchmarks/speculative-hillclimb.md).
The engine allocates extra speculative scratch; it is not a RAM reduction.

Experimental tuning flags are `--speculative-query-group 2|4|8`,
`--speculative-expert-tile 0|1|2|4|8`, and `--speculative-weight-reuse 1` in the
native CLI (boolean `--speculative-weight-reuse` in the launcher/adapter).
Defaults remain Q2, inherited expert tile, and original projection kernels.
