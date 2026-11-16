# Exact greedy speculative hill climb

Hardware: Apple M3 Ultra, 60 GPU cores, 256 GiB unified memory. Native FP32
activations and FP32 KV; no MLX/PyTorch model inference in these runs.

## Final verified profile

The repeated-code completion workload crossed the 130 tok/s threshold at both
actual prefix lengths, using the same final binary and three rotated rounds:

| Actual prefix | Ordinary median | Speculative median | Speedup |
|---|---:|---:|---:|
| 16,384 | 98.53 tok/s | **166.83 tok/s** | 1.69× |
| 32,768 | 90.43 tok/s | **139.41 tok/s** | 1.54× |

Profile: draft 32, Q2, inherited expert row tile 2, shared projection weights,
original MLP kernels, GPU greedy/nonfinite reduction, and FP32 KV. Only active
draft scratch is declared indirectly resident. All three 32K samples exceeded
139 tok/s; all three 16K samples exceeded 157 tok/s. Draft 16 also crossed the
32K threshold (132.70 tok/s median). Ordinary-mode outliers remain in the JSON.

Every candidate passed **32 complete-vocabulary row bit comparisons**, zero and
two partial rollback checks with already-wrapped sliding rings, and exact
256-token free-running greedy/EOS plus final complete-logit bits. Timed drafts
used committed history only, never reference future tokens. Lean integer checks,
**104 native tests and 135 Python tests** passed.

**This is not a general 130 tok/s Graff coding result.** The favorable repeated
code accepted every proposal. The real five-turn Graff fixture instead accepted
192 of 2,660 proposed inputs (including already-known first inputs) and achieved
only **22.83 native decode tok/s** over 310 completion tokens. It nevertheless
passed all six unchanged tests, used one persistent worker, reused KV, removed
the router, and left no owned processes or inference lock behind. It cached
3,369 tokens, not a filled 32K. Keep speculation off for general coding unless
its draft acceptance is beneficial; default remains ordinary decoding. No
full-262K throughput or universal model-quality claim is made.

For copy-heavy continuations, add `--speculative-tokens 32
--speculative-weight-reuse` before the owned launcher's `--` separator. Native
CLI uses `--speculative-weight-reuse 1`. The high-acceptance native check is:

```sh
/path/to/tokenizer-python -B tools/check_speculative.py \
  --checkpoint /path/to/flint-int4 --binary zig-out/bin/flint-speculative-bench \
  --prompt-tokens 32768 --max-tokens 256 --draft-candidates 8,16,32 \
  --query-groups 2 --expert-tiles 0 --weight-reuse --repeats 3 \
  --output /path/to/local-report.json
```

Run GPU work under the project's exclusive resource guard, as in the launcher
documentation. Speculative scratch adds memory; Zig is not inherently a
weight/KV-memory reduction. No new commit or push was made.

## First controlled query-pair results (earlier stage)

Actual computed prefixes, not merely configured capacities. Each timed run
committed 256 free-running greedy tokens. Three rotated rounds included ordinary
and draft sizes 2, 4, and 8, restoring the same physical prefix outside timing.
Warmup, tokenization, loading, prefill and snapshot restore were outside decode
measurements; drafting, all verification/rejected work, sync, correction/bonus
forwards and final-token KV commit were inside them.

| Actual prefix | Ordinary median | Draft 4 median | Draft 8 median | Draft 8 speedup |
|---|---:|---:|---:|---:|
| 16,384 | 103.50 tok/s | 123.12 tok/s | **137.41 tok/s** | 1.33× |
| 32,768 | 88.90 tok/s | 103.58 tok/s | **115.91 tok/s** | 1.30× |

All three draft-8 16K runs exceeded 137 tok/s. The 32K target is **not yet
verified** by these results. An ordinary 16K trial slowed to 90.66 tok/s; it is
retained in the JSON, not discarded. Medians use all three samples per mode.

The workload repeats tokenized binary-search-tree source code, then lets the
model continue freely. It is deliberately favorable to history matching: all
228 proposed inputs were accepted in each draft-8 run, including 29 already-known
first target inputs. It is not an agent benchmark, a diverse coding evaluation,
or evidence that arbitrary workloads run at these rates.

## Rejected 32K query-width / expert-tile candidates

A separate same-prefix, six-mode sweep retained exact token/final-logit parity
and checked full-vocabulary batch rows and two partial commits **for every mode**.
The Q2/default-tile baseline reached 117.48 tok/s. Increasing simultaneous query
positions to Q4 or Q8 was worse (92.91 / 87.46 tok/s), consistent with excessive
register pressure; no profiler diagnosis is claimed. Expert row tile 1 was also
slower (114.39 tok/s even with Q2). None was promoted over Q2/default tile.
The 32K target remains unmet at this stage.

## Shared projection weights and GPU greedy reduction

Sharing packed INT4 projection weights between two tokens improved the 32K
median to **121.68 tok/s**, with exact row/token/final-logit admission. GPU
argmax plus whole-row nonfinite checks then avoided repeated CPU vocabulary
scans, but the next controlled median was still **122.04 tok/s**. Its 256-token
GPU command times were about 2.02 s, versus about 2.09 s wall time: the remaining
bottleneck is predominantly GPU work, not simply CPU readback. The unprofiled
wall-minus-command residual is not a pure CPU measurement.

An independently controlled MLP shared-weight candidate also passed parity but
was slower (120.00 tok/s median) and remains disabled. All these trials and
outliers are retained in the JSON. Larger draft batches are the next experiment;
32K at 130 tok/s is not established by the above measurements.

## Correctness evidence

Before speed admission, native batch rows for eight forced reference inputs
were compared bit-for-bit over the complete vocabulary with ordinary forwards.
Zero-commit and two partial-commit paths exercised 35 already-wrapped sliding
layers. Forced tokens were used only for these untimed correctness checks.
All 256 free-running tokens/EOS and the final complete vocabulary row matched
ordinary native decoding exactly for every timed run and mode. This is measured
input coverage, not a universal floating-point proof or a new full MLX quality
evaluation. Default ordinary/FP32 decoding remains available.

A separate two-request persistent session matched ordinary tokens and prefill/
final logits exactly, reused its prefix, exercised 31 speculative batches and
committed 778 tokens. Its proposals included real rejection (93 accepted of
121 proposed inputs), rather than testing only an all-accepted path.

## Real Graff integration

The actual installed Graff **0.302.74.68** matched GitHub's latest published
release **v0.302.74.68** when checked. The owned launcher used the native
speculative backend on the sandboxed coding fixture: five requests, one worker,
zero failures, all six unchanged fixture tests passed, temporary router removed,
and launcher exit zero. It cached 3,369 tokens with configured capacity 32K—this
was **not** a filled-32K throughput run.

Those Graff turns had 107 batches, 428 proposed inputs and 182 accepted inputs
(including already-known first inputs). Acceptance is much lower than in the
repeated-code speed workload; speculation is not claimed to improve that coding
fixture's speed. It remains opt-in.

The implementation and proof limitations are described in
[the speculative decoder notes](../docs/SPECULATIVE.md). Detailed sanitized
trials and binary/workload hashes are in
[`speculative-hillclimb.json`](speculative-hillclimb.json). Raw local logs and
prompts are excluded. No new commit or push was made for this hill climb.
