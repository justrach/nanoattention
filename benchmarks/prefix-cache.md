# Session prefix cache and rewind on the real checkpoint

Apple M3 Ultra (60 GPU cores, 256 GiB unified memory), Zig 0.17.0 ReleaseSafe,
the 22.57 GB affine-INT4 Flint export, `flint session --context 8192
--mlx-attention 1 --staged-row-tile 2`. KV is FP32. No speculation.

## Prompt

A coding-agent system prompt plus 20 function-tool definitions, rendered with
the checkpoint's own tokenizer and `chat_template.jinja` (the
`tools/chat.py` path). The system-and-tools part rendered alone is
**2,682 tokens** and is an exact token prefix of both requests:

| Stream | Tokens |
| --- | ---: |
| S (system prompt and tools) | 2,682 |
| S + U1 (first user turn, generation prompt) | 2,720 |
| S + U2 (different user turn) | 2,719 |

Every request generates 32 greedy tokens and returns all-vocabulary logits.

## Results

| Row | Process | Tokens computed | Prefill | Restore | Other |
| --- | --- | ---: | ---: | ---: | --- |
| fresh S+U1, no new flag | new | 2,720 | 23.13 s | | |
| same, base-commit binary | new | 2,720 | 23.23 s | | |
| fresh S+U2, no new flag | new | 2,719 | 23.14 s | | |
| (a) cold S+U1, `checkpoint_at=[2682]` | new | 2,720 | 23.20 s | | snapshot write 0.24 s |
| (c) `truncate: 2682`, then U2 | same as (a) | 37 | 0.29 s | | memory checkpoint, 0 replayed |
| (b) reset S+U2, restored from disk | new | 37 | 2.72 s | 0.22 s | |
| (b) repeated | new | 37 | 2.72 s | 0.22 s | |
| U2 alone (warmup only) | new | 37 | 2.78 s | | |
| (b) after that warmup, same process | warm | 37 | 0.30 s | 0.22 s | |

The snapshot of the 2,682-token prefix is **227,352,688 bytes** (217 MiB):
14 full-attention layers × 2,682 positions × 4 KiB (153.8 MB), 35 full
513-slot rings (73.5 MB), token IDs and layer records. Ring copies for
in-memory checkpoints took 0.01–0.03 s per request.

From process start to the first response, a cold S+U2 request took 24.73 s;
the restored one took 3.34 s. About 2.4 s of that is not the restore: any new
process spends it on its first forward (the 37-token warmup row alone took
2.78 s; after it, the same 37 tokens took 0.30 s). Model load itself was under
2 s with the weights already in the page cache.

## Exactness

Each row's prefill logits (after the last prompt token) and final logits (after
32 generated tokens) were compared bitwise over all 150,144 vocabulary entries,
and its 32 greedy tokens for equality:

| Row | Reference | Prefill logits | Final logits | Greedy tokens |
| --- | --- | --- | --- | --- |
| (a) cold with checkpoint and snapshot | fresh S+U1 | identical | identical | equal |
| (c) rewind, then U2 | fresh S+U2 | identical | identical | equal |
| (b) restored | fresh S+U2 | identical | identical | equal |
| (b) restored, repeated | fresh S+U2 | identical | identical | equal |
| (b) restored after warmup | fresh S+U2 | identical | identical | equal |
| fresh S+U1 | base-commit binary | identical | identical | equal |

## Protocol and limits

One sample per row; no repeats beyond those listed. Each phase (cold and
rewind; the three restoring processes; the three fresh processes) was one run
under the shared inference lock, and native processes never overlapped. The
cold and restore rows are from a second run of those two phases (a third
restoring process was added); the fresh rows are from the first run with the
same binary. The first run's cold/rewind/restore rows were also bit-identical
to the fresh rows (cold prefill 23.63 s, restore 0.22 s, restored prefill
2.72 s). A long-lived engine process outside the lock was resident during all
runs, so some timing noise is possible.

The prompt is synthetic but uses the real template, tokenizer and tool-call
format; it is not a recorded Codegraff session. The rewind measured here is the
favourable case (a pinned memory checkpoint exactly at the target). A rewind
between checkpoints replays the recorded tokens at prefill speed, about 115
tokens/s. The snapshot is tied to the exact `flint` executable: any rebuild
invalidates it.

Reproduce (each phase under the lock, then compare without the model):

```sh
python3 tools/guarded_run.py --lock LOCK --timeout 600 -- \
  /path/to/tokenizer-python -I tools/prefix_cache_benchmark.py CHECKPOINT \
  --output target/prefix-cache-benchmark --phase cold   # then restore, fresh
/path/to/tokenizer-python -I tools/prefix_cache_benchmark.py CHECKPOINT \
  --output target/prefix-cache-benchmark --phase compare
```

`--phase fresh --baseline BINARY` adds the base-commit row. `compare` deletes
the snapshot directory. [Recorded rows and comparisons](prefix-cache.json).
