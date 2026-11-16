# Ordinary raw-token decode at filled 16K / 32K

**130 tok/s is not reached.** This experiment excludes speculative decoding and repeated-code drafts. Serving defaults are unchanged; new prefetch variants remain opt-in tuning candidates.

Hardware: Apple M3 Ultra, 60 GPU cores, 256 GiB unified memory. Native ReleaseSafe build; mapped INT4 Flint checkpoint, FP32 activations and FP32 KV. Full measurements, configurations, hashes, memory accounting and profiler data: `raw-decode.json`.

## Workload and timing

- Seed `20260328`; independent Python `random.Random` streams generate uniformly distributed valid raw token IDs. The decode stream is identical at 16K and 32K. The tokenizer is not loaded.
- The engine actually evaluates all 16,384 / 32,768 prefix tokens. Global attention retains the full prefix; sliding layers retain their normal ring caches. No fabricated cache or position jump.
- Each timed trial performs 256 ordinary single-token forward passes, including the vocabulary head, greedy readback and GPU completion. IDs are teacher-forced random inputs: this measures raw model execution, **not** free-running coding quality or agent throughput.
- Each candidate must match every bit of all 150,144 vocabulary logits at all 256 positions before timing. Prefix snapshots/restoration, admission, warmup, loading and prefill are untimed. Candidate order rotates between repeats.
- Profiler samples run only after throughput timing, from a restored real prefix. Encoder-boundary profiling perturbs execution; its numbers identify bottlenecks, not benchmark throughput.

## Measurements

| Configuration | Filled 16K, tok/s | Filled 32K, tok/s |
|---|---:|---:|
| Original one-head/SIMD reference, first 3-repeat sweep | 95.59 | 79.51 |
| Current two-head/SIMD long-context policy, same sweep | 102.97 | 89.89 |
| Two-head reuse + affine loads + gate/down row tile 1, same sweep | 105.18 | 91.52 |
| New two-head / two-token prefetch candidate, later 5-repeat sweeps | 106.09 | 93.19 |

The last row is **not a paired speedup over the earlier sweep**. Within its own trial it improved over the otherwise identical TK4 candidate by approximately 0.46% at 16K and 1.53% in the final 32K sweep. Another 32K trial measured TK2 at 87.88 versus TK4 at 86.94 tok/s; absolute rates varied. The complete trial history is retained rather than reporting only the fastest result.

One-token prefetch was slower for two-head reuse. Four-head reuse with one-/two-token prefetch passed bitwise admission but reached 92.58 / 92.42 tok/s in the final 32K trial, below two-head TK2 at 93.19. Neither was promoted.

## What changed

`attention_gqa_stream.metal` adds TK1/TK2 producer variants for two-/four-head reuse. Smaller prefetch batches reduce live K/V registers without changing per-head stripe order, FP32 operations, reduction or merge. `gqa_prefetch` selects these candidates in the native tuning harness; default remains TK4.

The driver now supports `--workload random-raw`, `--seed`, and an untimed post-sweep profiler. The first raw profile identified global GQA attention as the largest individual kernel at 32K (about 3.07 ms/token in the perturbed profile), followed by MoE gate/up and down projections. CPU process time was only about 0.09–0.11 seconds per 256-token trial: eliminating Zig/host overhead alone cannot close the gap.

## Correctness scope

- ReleaseSafe build and generated proof-binding checks pass.
- Five targeted headless native tests pass, including TK1/TK2 all-scratch/all-output bit comparisons, unsupported-launch rejection, and existing Metal primitives.
- Seven Python driver tests pass, including deterministic/range/independent raw streams.
- `proofs/Prefetch.lean` proves integer stripe reconstruction and unique chunk ownership for chunk sizes 1/2/4, checked on every native build. It does **not** prove GPU floating-point compilation, speed or model quality.
- All measured candidates passed full-model ordinary-reference bitwise admission at both tested context sizes where run. Four-head variants were timed only at 32K, after synthetic device coverage at both lengths.
- The full native suite was not rerun in this raw phase. Prior MLX parity results remain separate; these runs do not constitute a fresh independent MLX comparison for random inputs.

## Reproduce

Run from `nanoattention`, using the project's Python environment and exclusive inference guard. Replace the local checkpoint and lock placeholders; do not run concurrently with another GPU inference job.

```sh
zig build -Doptimize=ReleaseSafe
python tools/guarded_run.py --lock LOCAL_INFERENCE_LOCK --timeout 2400 -- \
  python tools/tune_decode.py LOCAL_CHECKPOINT \
    --workload random-raw --seed 20260328 \
    --prompt-tokens 32768 --decode-tokens 256 --repeats 5 \
    --candidates '[{"head_row_tile":2},{"head_row_tile":2,"gqa_reuse":2,"affine_loads":true,"gate_row_tile":1,"down_row_tile":1},{"head_row_tile":2,"gqa_reuse":2,"affine_loads":true,"gate_row_tile":1,"down_row_tile":1,"gqa_prefetch":2}]' \
    --output .zig-cache/raw-reproduction.json
```

Use `--prompt-tokens 16384` for 16K. Add `--profile-after-sweep --profile-candidate 1` only for an explicitly untimed kernel profile.
