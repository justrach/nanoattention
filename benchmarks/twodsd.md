# 2DSD CPU benchmark (tiny looped model)

Measured on this Cloud Agent VM, **not** Apple Silicon and **not** Metal.

- Host: Linux x86_64, 4× Intel Xeon (hypervisor), no GPU
- Zig 0.17.0, `-Doptimize=ReleaseFast`
- Target: exact tiny looped CPU model in `src/twodsd.zig` (`R=4`, vocab 97)
- Drafts: oracle (exact continuation) when `k > 1`. Real drafters accept less.
- Output: every run checks greedy tokens against plain AR (`arGreedy`) and
  aborts on mismatch.

```sh
zig build twodsd-bench -Doptimize=ReleaseFast
./zig-out/bin/nano-twodsd-bench --tokens 64 --prompt 8 --k 4 --stage stage2b --repeats 7
```

## Results

Call speedup is **loop-steps vs sequential AR** (`AR calls = R × (tokens−1)`),
not wall-clock vs a Metal decoder and not the paper's GPU 7.6×.

| causal | tokens | k | stage | AR loop-steps | 2DSD mean steps | step speedup | 2DSD wall tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| yes | 64 | 4 | stage2b | 252 | 64.000 | 3.938× | 76 346 |
| yes | 64 | 1 | stage1 | 252 | 252.000 | 1.000× | 252 379 |
| yes | 128 | 16 | stage2b | 508 | 32.000 | 15.875× | 61 863 |
| no | 64 | 4 | stage2b | 252 | 19.000 | 13.263× | 252 810 |

`k=1` / stage1 is LoopSpec with no token drafter: lossless, no step reduction
on this causal toy (every token still walks all `R` loops). Non-causal and
large-`k` oracle drafts collapse many tokens into one tree step; that is the
upper bound the paper's FakeTreeRunner schedule also uses, not a Flint or
Inkling number.

Wall tok/s is the tiny CPU model only. It is **not** Metal, **not** a
checkpoint, and **not** comparable to `nano bench` / `flint-bench`.

## KV accounting (not a runtime cache)

`historyKvBytes` for 12 loop layers, FP32, 2048 elements/position:

| seq | R | policy | bytes | vs `full` |
| ---: | ---: | --- | ---: | ---: |
| 72 (bench `prompt 8 + tokens 64`) | 4 | `full` | 28 311 552 | 1.000 |
| 72 | 4 | `shared_last` W=64 | 32 243 712 | **1.139** (window ≈ seq; sharing does not save) |
| 144 (`prompt 16 + tokens 128`) | 4 | `full` | 56 623 104 | 1.000 |
| 144 | 4 | `shared_last` W=64 | 39 321 600 | 0.694 |
| 4096 (Inkling-shaped, R=3) | 3 | `full` | 1 207 959 552 | 1.000 |
| 4096 | 3 | `shared_last` W=64 | 421 527 552 | 0.349 |

Last-loop sharing helps when committed `seq` is large compared with `W`.
These are byte formulas, not measured Metal allocations. Inkling still stores
per-visit KV in the megakernel.

Raw JSON: [twodsd.json](twodsd.json).
