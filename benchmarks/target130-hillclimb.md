# Long-context decode hill climb toward 130 tok/s

**130 tok/s is not achieved.** These are actual 16,384- and 32,768-token
prefixes, not short prompts with a large allocated cache. The accepted change
shares each loaded K/V register tile between two query heads in a SIMD group.
All model activations, KV values, accumulation and output logits remain FP32;
checkpoint quantization is unchanged.

## Controlled measurements

Apple M3 Ultra, 60 GPU cores, MLX 0.32.4, Zig 0.17, ReleaseSafe. Each trial uses
one genuinely evaluated repeated-code prefix, a byte snapshot of its KV, and
256 teacher-forced decode steps. Candidate admission compares every vocabulary
word (150,144 logits) at every decode position to the original native path.
Only bit-exact candidates are timed. Five rounds rotate candidate order;
prefix restore, admission and warmup are outside timing. Throughput includes
native encoding, synchronous GPU completion, CPU finite checks and greedy
selection. It is not free-running generation or token-live adapter throughput.

| Actual prefix | Original | Two heads/SIMD | Four heads/SIMD | Two-head speedup |
| --- | ---: | ---: | ---: | ---: |
| 16,384 | 96.10 tok/s | **105.78 tok/s** | 104.16 tok/s | **1.101×** |
| 32,768 | 79.82 tok/s | **91.57 tok/s** | 88.34 tok/s | **1.147×** |

Every candidate above passed all 256 all-vocabulary bitwise checks. These gates
compare candidates to the original **native** path; they are not 256-position
MLX comparisons. Raw repeats and binary/input hashes are in the JSON companion.
The 16K and 32K table rows are independent trials, not measurements from the same
thermal instant.

A fresh 16K check of the promoted automatic path against **unmodified stock
sequential FP32 MLX** compared all vocabulary logits at the final prefill position
and 15 cached decode positions: maximum and RMS error were **0.0**, all 16 top-1
choices matched, and all 16 greedy generated tokens agreed. Both engines actually
evaluated every prefix token; earlier prefill logits were not compared. The full
native build and **68 headless/unit tests** passed; Lean build and axiom audit
passed. These numerical checks establish this workload's parity, not a general
floating-point correctness proof or model-quality benchmark.

The two-head path is automatically selected only for matched M3 Ultra/GQA8
global attention with at least 16,384 resident tokens. Short prefixes, sliding
attention, unsupported shapes and the non-matched path retain their original
kernels. Split stripe counts, per-query lane reductions, sink participation,
online softmax association, scratch layout and merge are unchanged. GQA4 is
available as an experimental benchmark candidate, not selected automatically.

## Rejected candidates

- Fixed-column INT4 loop unrolling: 96.04 → 95.61 tok/s at 16K. Bit-exact,
  but slower; not enabled.
- Constant-stripe/unrolled global attention without head reuse:
  96.21 → 95.65 tok/s at 16K. Bit-exact, but slower; not enabled.
- Packing two/four independent stripes into larger threadgroups: no reliable
  gain over two-head reuse. In a separate six-candidate trial the original was
  95.01, two-head reuse 103.72, two-stripe packing 103.53, four-stripe packing
  102.75, and GQA4/four-stripe packing 99.20 tok/s.
- Combining two-head reuse with parallel expert-down: 103.48 tok/s in that
  same trial versus 103.72 without it. Not promoted.

The later trial's lower absolute speeds are retained rather than substituted
with a best repeat. The accepted result is a consistent relative gain, not a
claim that the highest observed single run is sustained performance.

## Proof and profiling scope

Lean proves integer segment coverage/order/uniqueness, linear resident-cache
modulo elimination, GQA head ownership and grouped-stripe indexing. The normal
build checks the existing generated bindings. These proofs do **not** verify
floating-point arithmetic, the Metal compiler, GPU memory ordering or speed.
Headless device tests compare all producer-state and merged-output bits, test
sink dtypes/offsets and stripe tails, and reject unsupported launch geometry.

A one-token profile started after real 16K prefill recorded 503 valid dispatches,
zero failed/invalid dispatches, using encoder-boundary GPU counters. Global
attention split accounted for 2.455 ms, its merge 0.522 ms, and sliding attention
1.095 ms in that diagnostic. Encoder splitting perturbs execution: those are
**profiling diagnostics, not throughput measurements**. Ordinary timing keeps
one submission per token.

## Further exact-math sweeps and replay

BF16-affine weight-load specialization remained bit-exact but yielded only a
small gain: 104.35 → 105.42 tok/s versus two-head reuse alone in its 16K trial.
A per-stage sweep tested QKV, gate/up, attention projection and expert-down tiles
independently. Tile 1 for gate/up and down was somewhat better; tile 4 for either
was worse. Packing the real checkpoint's BF16 router (rather than the earlier
FP32-only specialization) was correct but essentially tied. These knobs remain
experimental; the default is not selected from a single best repeat.

The SDK-supported indirect-command replay experiment also passed **all 256
all-vocabulary bitwise admission checks**. It records pipeline/buffer/geometry
bindings and updates a bounded parameter table for each token; explicit ICB
barriers preserve dependencies, and ordinary `finish` still waits. Its table
capacity/stride are emitted to C and Zig from Lean, with bounds/alignment proofs.
The proofs do not certify Metal barrier semantics. Headless tests exercise
parameter/geometry/binding changes, command-count changes, clearing before
buffer teardown, and whole-batch abort on a missing kernel.

Replay did **not** improve decode throughput in a five-round 16K trial:

| Same exact-math configuration | Direct encoding | ICB replay |
| --- | ---: | ---: |
| GQA2, BF16 affine loads, gate tile 1, down tile 1 | **106.14 tok/s** | 103.50 tok/s |
| Median GPU time / 256 tokens | **2.18482 s** | 2.27186 s |
| Median process CPU time / 256 tokens | 0.09288 s | **0.08287 s** |

The CPU reduction did not offset slower GPU execution. Replay is available only
as an explicit benchmark experiment and is **not enabled in normal Graff/native
inference**. It remains 503 logical dispatches and one submission per token,
not a single-kernel implementation. Expected negative tests print diagnostics
for rejected profiling and an intentionally missing kernel; they are not passing
inference workloads or production crashes.

BF16 global KV is separately opt-in, reduces memory, and has a non-bit-exact
accuracy comparison in [the BF16 experiment report](bf16-kv.md). Neither that
precision experiment nor replay has reached 130 tok/s.

## Native Graff integration

`tools/graff_native.py` starts an owned loopback adapter with temporary local
routing and environment-only credentials; all inference stays Zig/Metal.
The installed terminal Graff completed a sandboxed six-test coding fixture in
five requests with one native worker, four KV-hit follow-ups and zero failures.
It removed its router and left no owned adapter/native children running.
Configured context was 32K, actual final cache 3,369 tokens: this integration
check is **not** a filled-32K throughput result. See `docs/CODEGRAFF.md`.
