# 2DSD (two-dimensional speculative decoding)

Opt-in CPU scheduler matching Kim, Chavan, Seo, Hu, Lee, Kang, Mahoney,
Keutzer, Hooper and Gholami, *2DSD: Two-Dimensional Speculative Decoding for
Looped Language Models* (2026),
[SqueezeAILab/2DSD](https://github.com/SqueezeAILab/2DSD)
(`paper/2DSD.pdf`), announced
[Nov 2026](https://x.com/minseokim0525/status/2123533353823662556).

The paper's looped LMs apply the same block stack `R` times per token. Every
loop adds KV and latency. 2DSD speculates along **tokens** (block size `k`)
and along **depth** (review at loop `d1`, optional promotion/fork at `d2`,
lossless verify at loop `R`), and can share last-loop KV (`shared_last`,
window `W` with `R * k ≤ W`). Greedy output matches ordinary AR; sampling
uses standard rejection sampling with residual renormalization. The paper
reports up to 7.6× geometric-mean greedy throughput over native AR on their
GPU stack — that number is **not** reproduced here.

## Fit in nanoattention / Flint

| Surface | Looped? | What 2DSD needs | Status |
| --- | --- | --- | --- |
| Flint (`flint`) | No. 49 distinct layers, not a weight-tied loop. | Depth speculation and last-loop KV sharing need looping. | `--twodsd 1` fails closed (`TwoDsdRequiresLoopedModel`). Token speculation already exists as `--speculative-tokens` ([SPECULATIVE.md](SPECULATIVE.md)); 2DSD reuses that history drafter on the CPU toy model. |
| LFM3 / LFM2 (`nano`) | No. | Same. | `--twodsd 1` fails closed. |
| Inkling 2 Mini (`inkling2_loop`) | Yes. `loop_start=6`, `loop_end=18`, checkpoint `R=3` (48 visits), per-visit KV. `--loop-passes` already lowers how many times the core is applied. LoopCD is contrastive early-exit, not overlapping 2DSD. | Overlapping decode: several tokens at different loop depths in one step, plus last-loop KV sharing. | Looping exists, but the megakernel still finishes every pass of **one** token before the next. `--twodsd 1` fails closed (`TwoDsdOverlappingDecodeNotWired`). Wiring Metal overlapping decode is not a small, clean change (new KV layout, in-flight tree, rollback, and a different dispatch schedule). |

This repository therefore ships:

1. A paper-faithful **CPU scheduler** (`src/twodsd.zig`) on a tiny exact looped
   target, including stage1 / stage2a / stage2b, last-loop KV accounting, and
   rejection sampling.
2. Opt-in CLI flags that **do not change default generate/session/run/chat**.
3. Lean integer contracts (`proofs/TwoDsd.lean`) for the window, KV formula,
   greedy prefix, and rejection inequality.
4. CPU-only benchmarks vs plain AR on the tiny model. **No Metal speedup is
   claimed.**

## Algorithm (as implemented)

Matches the official `FinalizedTreeExecutor` in
`SqueezeAILab/2DSD` (`looped_2d_finalized.py`):

- **Prefill** the prompt at loop `R`. The first generated token is the target's
  greedy (or sampled) choice; it is not speculated.
- **Draft** a block of up to `k` tokens. `k = 1` is LoopSpec (no token drafter).
  `d1 = R` is the DFlash boundary (no early-loop review). For `k > 1` the CPU
  path accepts `oracle` (exact continuation, tests/bench only),
  `oracle_corrupt` (lossy drafts, still lossless after verify), or `history`
  (`src/speculative.zig` suffix match on committed tokens).
- **Stage 1** (`d1 < R`): review the block at loop `d1`. Greedy keeps a matching
  prefix and freezes the rest; sampling accepts each token with the early-loop
  probability and resamples on the first rejection.
- **Stage 2a**: if loop `d2`'s argmax equals the frozen head, **promote** that
  suffix and review it at `d2`.
- **Stage 2b**: if `d2` disagrees with both the main and frozen heads, **fork**
  an alternative continuation from the residual `max(p_{d2} − p_{d1}, 0)`.
- **Verify** at loop `R`. Greedy accepts the first child equal to the target
  argmax (else a bonus token). Sampling accepts child `x` with
  `min(1, t(x)/g(x))` and renormalizes `max(t − g, 0)` on rejection.
- **KV**: `full` stores history per loop. `shared_last` stores one last-loop
  history plus an in-flight window of `W` per loop; `checkTreeWindow` rejects
  `R * k > W`. Accounting is in `historyKvBytes` / `inklingLoopKvBytes`.

Call counts on the official FakeTreeRunner schedule are reproduced: `k=1,d1=1`
→ 67, `k=4,d1=1` → 19, `k=4,d1=4` → 64.

## Opt-in flags

Default is off (`--twodsd 0`). Tuning flags without `--twodsd 1` are rejected
(`TwoDsdTuningRequiresEnable`). Enabled generate/session/run/chat currently
always errors, before weights load when the config is readable:

| Flag | Default | Notes |
| --- | --- | --- |
| `--twodsd 0\|1` | `0` | Master switch. |
| `--twodsd-k N` | `1` | Block size, 1..16. |
| `--twodsd-stage stage1\|stage2a\|stage2b` | `stage2b` | |
| `--twodsd-d1 N` | `1` | Early review loop. |
| `--twodsd-d2 N` | `2` | Promotion/fork loop; required `d1 < d2 < R` unless stage1. |
| `--twodsd-kv full\|shared_last` | `full` | |
| `--twodsd-window N` | `0` | Required when `shared_last`. |

CPU tests and the tiny-model bench do not go through these flags:

```sh
zig build test-twodsd
zig build twodsd-bench -Doptimize=ReleaseFast
./zig-out/bin/nano-twodsd-bench --tokens 64 --k 4 --stage stage2b --repeats 7
```

## Tests

`src/twodsd.zig` checks, on a tiny looped model (`R=4`, vocab 97):

- Greedy output equals plain AR for all stages and `k ∈ {1,4,8,16}`, including
  a corrupt drafter and the history drafter.
- Official FakeTreeRunner call counts (LoopSpec / 2DSD / DFlash boundary).
- Frozen suffix, promotion, d2 fork, nested promotion rejection, no 64-node cap.
- EOS / partial blocks / executor reuse.
- Rejection sampling `min(1, t/g)` and residual renormalization; `k=1` sampling
  on a Markov loop follows loop-`R`.
- `R * k ≤ W` and last-loop KV byte accounting.
- CLI `Cli.apply` off by default; Flint/LFM/Inkling Metal fail closed.

## What this is not

- Not Metal overlapping decode, not a Flint/Inkling generation speedup, not a
  second-model DFlash drafter, not self-distilled shared-KV weights, and not
  the paper's 7.6× GPU result. See [benchmarks/twodsd.md](../benchmarks/twodsd.md).
