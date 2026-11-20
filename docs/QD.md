# Searching for speed the quality-diversity way

How the megakernel's kernels (Metal) and schedule (Zig) are tuned for a model: a MAP-Elites archive in place of a single greedy hill-climb. The best candidate is kept in every niche, not one global best, so a change that loses today survives as a stepping stone and the engine can use a different winner in each regime. It was first applied to LFM3-1.7B on `nano`; nothing here is specific to that model. `tools/nano_qd.py` is the driver; the LFM3 search's archive is in `benchmarks/lfm3-qd/`.

## Candidates

A candidate is a complete, named configuration of the engine: a value for every knob (`src/mega/knobs.zig`), where a knob selects among variants that are all compiled in. Candidates are JSON files in `src/mega/candidates/` with a name, a family tag, the knobs, and the hypothesis they test.

- Every variant is selectable at run time (`--knobs FILE.json`, `--knob name=value`). A candidate is never "edit the default and rebuild".
- The plain reference executor (all knobs at their defaults: one dispatch per operation, no fusion) is always available and is never removed: everything is checked against it. Its outputs are pinned (`reference.json`); a binary whose reference no longer reproduces the pin admits nothing.
- A new variant is new code behind a new knob value.

## Evaluation

Two stages. A candidate that fails the first is recorded as rejected and is never timed.

**1. Admission.** In the exact tier, on the tiny synthetic checkpoints (`tools/lfm3_tiny.py`) and on the real checkpoint's golden cases (`tools/lfm3_reference.py goldens`), the candidate's all-vocabulary logits at the stored positions and its greedy tokens are bit-identical to the reference executor's, for single-token decode and for batched ingestion. A candidate that reassociates a floating-point sum is not admitted in this tier, however fast. An external checker (`--certify-cmd`) may show the candidate's arithmetic is the same expression tree as the reference's; the trial then records `certified`, otherwise `empirical`.

**2. Quality.** Under one inference lock, on the real checkpoint: decode tokens a second (256 teacher-forced decode tokens, three warmed repeats, median) and prefill tokens a second, at actual prompt lengths 64, 1,024 and 4,096; loading and warm-up excluded. A challenger replaces an elite only on a repeatable gain: the two are timed interleaved, at least three rounds each, and the challenger's median must beat the elite's by more than the larger of 1% and twice the spread between the elite's own rounds.

Screen cheaply first (`nano_qd.py screen`: one prompt length, one objective, a minute or two), and spend a full trial only on what survives. State the hypothesis before timing; the trial record keeps whether the measurement agreed.

## The archive

A cell is a niche, keyed by where a candidate is used (objective `decode` or `prefill`, and prompt-length bucket `<=256`, `<=1024`, `<=4096`, `>4096`) and by how it works (dispatches per token, binned, and the candidate's family tag), so different approaches do not compete in one cell. Each cell holds its best admitted candidate. Parents for mutation are drawn from occupied cells, not only from the fastest.

- `trials.jsonl`: one line per evaluation, appended, never rewritten. Rejected and losing candidates stay; they are the evidence.
- `archive.json`: the current elite of each cell, rewritten atomically under a lock.
- `elites.md`: a readable table, regenerated from the archive.
- `reference.json`, `reference-history.jsonl`: the pinned reference and every earlier pin with the reason it was replaced.

Variant code that holds no cell and fed no elite is removed at the end of a search, with a line in the benchmark note saying what it was and what it measured.

## The mlx-arith tier

Exact batched ingestion keeps every (token, row) output's arithmetic identical to the single-token kernel's, which rules out a blocked matrix multiply. Stock MLX makes the other choice: its batched quantized matmul (`affine_qmm_t`) and attention (`steel_attention`) accumulate sums in a different order from its own single-token kernels, so its batched prefill is not bit-identical to its token-at-a-time result. `docs/MLX-PREFILL.md` describes what it does.

Prefill therefore has a second tier beside the exact one:

- A kernel in this tier must be bit-identical to the stock MLX operation it ports, on the same inputs: `tools/nano_mlx_ops.py check` compares output bits against MLX on random and real-checkpoint inputs. "MLX-like" does not qualify.
- End to end, a candidate using such kernels must meet the gate the reference executor itself meets against the goldens: logits and residual streams within the float32 noise floor recorded in each golden's `meta.json`, top-1 agreeing at every stored position, greedy tokens identical. Its largest difference from the exact batched path is recorded.
- It is recorded with `admission: "mlx-arith"` and lives in its own cells (family tags prefixed `mlx-`), so it never displaces an exact elite. Decode stays exact in both tiers.

## Running it

The tools read their surroundings from the environment:

| Variable | Meaning | Default |
| --- | --- | --- |
| `NANO_WORK` | Working directory holding `qd/lfm3/` (the archive), `golden/lfm3/` (the goldens) and the lock | `target/nano-work` |
| `NANO_LOCK` | The inference lock file passed to `tools/guarded_run.py` (it must exist) | `$NANO_WORK/inference.lock` |
| `LFM3_CHECKPOINT` | The `LiquidAI/LFM3-1.7B-MLX-4bit` checkpoint directory | `target/lfm3-1.7b-mlx-4bit` |
| `NANO_MLX_PYTHON` | A Python with stock MLX (mlx-vlm 0.7.11 or later), for baselines and the MLX checker | the running interpreter |

```sh
python3 tools/nano_qd.py parents                       # occupied cells to mutate from
python3 tools/nano_qd.py screen  zig-out/bin/nano A.json B.json --lengths 1024
python3 tools/nano_qd.py trial   zig-out/bin/nano src/mega/candidates/NAME.json \
        --proposer WHO --why "the hypothesis" --long
python3 tools/nano_bench.py versus zig-out/bin/nano $LFM3_CHECKPOINT NAME=cand.json --mlx mlx_f32,mlx_bf16
```
