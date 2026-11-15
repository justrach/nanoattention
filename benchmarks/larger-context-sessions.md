# Larger contexts and persistent coding sessions

## Decode measurements at 16K and 32K

Same M3 Ultra (60 GPU cores, 256 GiB unified memory), ReleaseSafe native engine,
existing Flint affine-INT4/group64 export, and MLX 0.32.4. Native activations,
accumulation and KV remain FP32.

| Actual benchmark prefix | Native decode tok/s | MLX FP32 tok/s | MLX BF16 tok/s |
| --- | ---: | ---: | ---: |
| 16,384 | **95.09** | 68.50 | 79.65 |
| 32,768 | **79.47** | 59.24 | 74.88 |

Protocol: 256 identical teacher-forced decode inputs, eight warmup tokens, three
synchronous timed repeats. Native performs a complete real sequential prefill
once, snapshots actual KV/position, and restores outside decode timing. MLX uses
stock batched prefill and fresh caches. These prefixes are the deterministic
repeated-code performance fixture, **not the separate coding conversation**.
These rates are not quality scores or a reproduction of Recursive's benchmark.

Initial long-prefill latency remains a substantial limitation:

| Benchmark prefix | Native sequential prefill | MLX FP32 batched prefill | MLX BF16 batched prefill |
| --- | ---: | ---: | ---: |
| 16,384 | **161.43 s** | 12.23 s | 10.11 s |
| 32,768 | **358.47 s** | 30.31 s | 24.76 s |

Native prefill has one sample; the methods differ deliberately and are labeled.
The fast decode figures must not be presented as end-to-end prefill parity.
Full-model all-logit comparisons for these repeated-code timing workloads are
not claimed; the coding-session reference checks its own recorded stream.

[Individual measurements and fingerprints](larger-context-decode.json).

## What the persistent session changes

`flint session` loads one model and retains the actual per-layer KV buffers across
JSONL requests. New requests contain only new suffix IDs. There is no implicit
reset, history replay, cache-position fabrication, or prefix eviction from the
full-attention layers. Every generated token, including final EOS/length-limited
tokens, is committed before returning. Explicit reset and context overflow are
validated before mutation. EOF shuts the worker down normally; outer guarded-job
interruption also owns the native child.

Flint remains hybrid: 14 global **NoPE** layers retain full-history KV; 35 sliding
**RoPE** layers retain a bounded 513-token ring. KV is FP32, not quantized. Response
memory figures distinguish allocated KV capacity from valid cached history and
do not include model weights or every temporary allocation.

At 16K effective keys the matched large-device attention schedule changes from
128 to 512 interleaved stripes. The generated Lean policy also defines the 64K
transition to 1024; that is **not** a claim of full-model 64K measurement. Large
matched contexts currently require M3 Ultra/GQA8. Independent FP64 kernel checks
cover 16K/32K and all admitted stripe counts; Lean covers integer geometry,
ownership and bounds, not floating-point equivalence or performance.

## Executed three-turn coding exercise

The first user turn included **16,483 actual chat-template tokens** from 82
uniquely structured synthetic source/spec components. Later turns asked for
missing-dependency/cycle handling and then nonmutation plus lexicographically
smallest topological ordering in `plan_order(graph)`. The session allocated a
32,768-token context and retained one model throughout.

The 1,024-token-budget attempt hit the limit inside reasoning on every turn.
Those raw results are preserved, not counted as completed coding answers. A
4,096-token-budget retry produced this observed session:

| Turn | Reused cached tokens | New prompt tokens | Generated tokens | New-prefix processing | Decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 16,483 | 4,096 | 160.95 s | 92.51 |
| 2 | **20,579** | **57** | 4,096 | **0.630 s** | 88.24 |
| 3 | **24,732** | **63** | 3,828 | **0.740 s** | 83.57 |

No prior prompt or response was fed through the model again during these turns.
The driver re-renders/tokenizes history on the CPU to verify exact prefix
identity, but sends only the new suffix to native inference. The final cache
contained **28,623 tokens**, including all reasoning and the final EOS. KV
allocation was **1.82 GiB**; valid cached KV at the end was **1.60 GiB**. These
figures exclude checkpoint weights and other memory.

These are single observed engine-phase timings, not three-repeat conversation
benchmarks. Prefill/decode timings exclude model loading, tokenizer work and
JSON transfer. Responses are returned as whole JSON records, not streamed;
0.630/0.740 seconds are **not client first-byte latency**.

## Cache and numerical results

- All **12,020/12,020 sampled tokens** matched the stock sequential FP32 MLX
  argmax on the identical recorded stream, with one reference cache across turns.
- All vocabulary entries at **six turn boundaries** (post-prefill and final
  committed token per turn) were **bit-exact**, with maximum/RMS error **0.0**.
- A separate native process replayed all **28,623 tokens** from a fresh cache.
  Its final all-vocabulary row was bit-exact with the persistent session.
- Tiny F32 and BF16/affine-quantized checkpoints also passed independent NumPy
  and fresh-native session checks, including explicit reset, ring wrap, byte
  accounting, final-token/EOS commitment and capacity rejection.

This establishes the checked stream/boundaries, not equality of every KV memory
word or every logit at every position. The 32K repeated-code throughput fixture
is separate from this coding transcript and has no additional full-model
all-logit admission claim.

## Coding quality: completion is a separate result

Even with a 4,096-token budget, the **first two responses were length-limited
before a final answer**. The third completed with EOS and returned a syntactically
valid heap-based `plan_order` implementation. Code found inside unfinished
reasoning is no longer counted as a completed answer; a CPU regression test
covers this distinction.

The [inert final example](coding-final-example.txt) is available for inspection.
An attempted network-disabled isolated functional check returned **exit 255
without diagnostics**. Functional execution is therefore **unverified**, not a
pass or an application crash. Generated code was not executed through the host
shell or coding driver; the isolated tool's completion is unknown. These results
are not three completed coding solutions or a broad coding
quality guarantee. MLX matched the same capped generation choices exactly.

[Sanitized coding results](coding-multiturn.json) retain both attempts, cache
counts, timings, numerical checks and source fingerprints. Complete raw
transcripts/logit arrays remain ignored under
`target/coding-multiturn-16k-v1/` and `target/coding-multiturn-16k-v2/`.

## Verification and reproduction

ReleaseSafe build, **60 Zig/unit/headless Metal tests**, **52 CPU protocol tests**,
**15 binding tests** and **six proof-verifier tests** passed. Lean's axiom audit
uses only standard logical axioms. Its proofs concern integer contracts, not
IEEE754 correctness, Metal memory ordering, performance or code quality.

Run these stages sequentially: the persistent native worker must exit before
fresh replay or the MLX model loads. Use a fresh ignored output directory.

```sh
python3 tools/check_sessions.py

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/coding_multiturn.py --native /path/to/flint-int4 \
  --prompt-tokens 16384 --context 32768 --max-tokens 4096 \
  --output target/coding-demo

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/check_session_replay.py /path/to/flint-int4 \
  --output target/coding-demo

python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1200 -- \
  /path/to/mlx-python -B tools/coding_multiturn.py --reference /path/to/flint-int4 \
  --output target/coding-demo
```

The next major throughput gap is **batched prefill**, not whether KV persists.
These measurements were captured before publication; source fingerprints preserve the tested working-tree revisions.
