# Codegraff adapter: native Flint over loopback OpenAI chat

`tools/codegraff_adapter.py` translates Codegraff's OpenAI-compatible requests to
one persistent `flint session` subprocess. **All model inference remains Zig +
Metal.** Python provides HTTP, the checkpoint's local tokenizer/chat template,
and Flint-tool-call/OpenAI-tool-call translation; no MLX/PyTorch model is loaded.

The adapter uses Codegraff's OpenAI-compatible chat contract. Current Codegraff
builds expose `nanoattention` as a built-in local provider; older installed
binaries are detected from their bounded, content-free `--schema` output and
receive a temporary project-local router instead. Neither path changes global
provider/key settings. Model ID is `flint`, provider ID `nanoattention`. The
example contains endpoint metadata, not a credential.

## Owned terminal launcher

`tools/graff_native.py` starts an ephemeral-port loopback adapter, selects
`nanoattention/flint` in the chosen workspace, then runs the installed terminal
Graff. It makes no global provider changes. Run it with the tokenizer Python
environment under the same inference guard:

```sh
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1400 -- \
  /path/to/tokenizer-python -B tools/graff_native.py /path/to/flint-int4 \
  --graff /path/to/installed/graff --workspace /path/to/coding-workspace \
  --context 32768 --max-tokens 4096 --timeout 1200 -- \
  -p 'Your coding task'
```

Omit the arguments after `--` for an interactive terminal session. The launcher
pins the model route and passes an ephemeral API key only in the owned child
environment. Built-in-aware binaries receive the ephemeral loopback URL through
the child environment and do not write router configuration. Older binaries
refuse an existing `.graff/.config.router` and lease a temporary router file.
On exit, failure, timeout or interruption it stops its owned children and removes
only its unchanged router file; user edits or replacements are preserved. No additional inference
lock or process group is created. A configured 32K capacity is not a filled-32K
speed measurement. Normal Graff tool permissions still apply; this launcher is
not itself an OS sandbox. Subagents default to the same local provider/model
with tier selection disabled; explicit additional Graff arguments remain the
user's choice. The adapter still admits only one active model request at a time.

The owned launcher was also exercised with the installed Graff in an OS-sandboxed
synthetic coding workspace: **five requests, one native worker, four KV-hit
follow-ups, zero failures**, and **all six unchanged fixture tests passed**. The
configured capacity was 32,768; only 3,369 tokens were cached. The temporary router
was removed and an OS process check found no launcher adapter/native children
remaining. Its 16 CPU-only cleanup/configuration tests pass. This verifies the
launcher lifecycle and coding integration, not 32K-filled throughput.

### Optional BF16 global KV

Add `--kv-bf16` to the launcher arguments **before** `--` to request the
experimental storage-only mode. The manual adapter accepts the same flag. This
halves global-layer KV storage; sliding KV and activation/attention math remain
FP32. Default behavior is unchanged. It can change logits and generated tokens,
and is not stock MLX BF16 inference. Request metrics expose `kv_storage`.

The BF16 launcher completed the same real sandboxed coding fixture: five
requests, one native worker, four KV-hit follow-ups, zero failures, and all six
unchanged tests passed. It cached 3,392 actual tokens with a 32K allocation and
removed its temporary router. This does not establish filled-32K speed or
unchanged generations. See [numerical and performance measurements](../benchmarks/bf16-kv.md).

### Experimental greedy speculation

The launcher/manual adapter accept `--speculative-tokens N` (range 0–32; zero
keeps ordinary decoding). It uses committed-history drafts and native batched
target verification, with explicit sliding-KV rejection rollback. Every returned
token still comes from exact target greedy acceptance; no draft is exposed
unverified. See [implementation and verification boundaries](SPECULATIVE.md).

For copy-heavy continuations, the measured profile is `--speculative-tokens 32
--speculative-weight-reuse`, before the launcher's `--`. It reached 167/139 tok/s
on filled 16K/32K repeated-code completion. **It is not a general coding speedup:**
the real five-turn Graff fixture had poor draft acceptance and only 22.83 native
decode tok/s with that profile, although all six fixture tests passed and KV
reuse/cleanup worked. Keep the default ordinary mode for general coding unless
measuring your workload shows a benefit.

## Manual adapter setup

From nanoattention, build the native executable, then run the adapter under the
same exclusive inference lock/resource guard used by other local Flint jobs:

```sh
zig build -Doptimize=ReleaseSafe
export NANOATTENTION_API_KEY=local  # development placeholder; loopback only
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 14400 -- \
  /path/to/tokenizer-python -B tools/codegraff_adapter.py /path/to/flint-int4 \
  --context 32768 --max-tokens 4096 --timeout 900 --port 8080
```

Add `--prefix-cache target/prefix-cache` (and optionally `--prefix-cache-bytes`)
to the adapter, or to the launcher before its `--`, to keep exact on-disk
snapshots of the system prompt and tool definitions across adapter restarts and
worker failures. A new worker then restores that prefix instead of prefilling
it: for a 2,682-token system prompt and tool list on the M3 Ultra (context
8192), a 227 MB snapshot restored in 0.22 s in place of about 23 s of prefill,
with bit-identical logits (see [measurements](../benchmarks/prefix-cache.md)).
The first request of any new worker still pays about 2.4 s of first-forward
warmup. A snapshot written at one `--context` is reused at another only if
both select the same attention kernels (split-attention eligibility is in the
settings key). Responses report
`restored_prefix_tokens`, `restore_seconds`, `snapshots_written` and
`snapshot_bytes`. Snapshots are invalidated by any rebuild of `flint` or change
of checkpoint or engine settings. The directory is the user's choice; nothing
is written without the flag.

Only `127.0.0.1` is bound; non-loopback serving is deliberately unsupported.
Use an appropriate private environment token instead of the placeholder on a
shared machine. The server does not log prompts or tool arguments by default.
Stop the owning command with SIGINT/SIGTERM; it closes its native worker.

In the **chosen coding workspace**, place
[`examples/codegraff-router.json`](../examples/codegraff-router.json) at
`.graff/.config.router`. Do not overwrite an existing project router without
choosing how that workspace should route. Nothing installs this file globally.
Then run the installed Codegraff:

```sh
NANOATTENTION_API_KEY=local GRAFF_CONTEXT=32768 \
  graff --model nanoattention/flint
```

`GET /v1/models` advertises the configured context. `POST /v1/chat/completions`
supports text messages, function tools, single-choice responses and SSE. The
native engine is currently **greedy**: unsupported nonzero temperature, top-p
sampling, custom stop strings, logprobs and reasoning-effort controls are
rejected rather than silently claimed to work. The configured server output cap
bounds client `max_tokens`/`max_completion_tokens`; the effective cap is returned
in response metadata. Context reserves every generated token, including EOS.

## KV reuse and tool semantics

- One loaded native model and its real per-layer KV remain alive between calls.
- OpenAI histories contain the full conversation. The adapter restores its own
  previously returned assistant's exact native bytes (reasoning, tool markup,
  terminal EOS) when the corresponding client message still matches.
- It re-renders with the checkpoint's own template and compares **actual token
  IDs**. Only an exact cached prefix permits append-only inference.
- When the new request's IDs diverge from the cached IDs (edited history, a
  retried request, non-round-tripping output), the adapter computes the exact
  common token prefix itself and sends native `truncate` to it (keeping at least
  one new token), instead of a reset. The engine restores its newest checkpoint
  at or below that point and replays its own recorded tokens up to it, exactly.
  Every response reports `cache_action` (`append`, `truncate` or `reset`),
  `truncated_from`, `rewind_checkpoint`, `replayed_tokens`, `replay_seconds`
  and `rewind_source`; `cached_tokens` excludes replayed tokens. A reset is used
  only with no common prefix, after a failure, or when `--prefix-cache` is on
  and the divergence lies inside the stable system/tool prefix (a disk snapshot
  of the new prefix may then be restored instead).
- The stable prefix is the leading system messages plus tool definitions,
  rendered alone with the checkpoint's template. When it is an exact token
  prefix of the request, a request that prefills across it marks it with
  native `checkpoint_at` (reported as `stable_prefix_tokens`/`checkpoint_at`).
- Prefixes are chunked into at most 65,536 IDs per native JSONL request so the
  native 1 MiB frame limit does not preclude a larger configured context. All
  intermediate chunks use zero generated tokens; generation occurs only on the
  last chunk. Total context is checked before the first native mutation.
- Flint XML-like function calls are validated/coerced against the supplied tool
  schemas and converted to standard OpenAI calls. The adapter itself **never
  executes a tool**. Codegraff owns tool permissions and execution.
- Malformed, unknown or missing-required-argument calls fail closed. Tool-looking
  text inside reasoning and length-capped/incomplete thinking cannot execute.
  The parser is not full XML or full JSON Schema; unsupported schema features
  fail closed, and reserved closing delimiters cannot be literal raw values.

This is a **single active sequence**, not concurrent batching or a multi-user
cache service. Concurrent inference returns 429. Switching conversations
rewinds to their common prefix (usually the system/tool prefix) or, with
`--prefix-cache`, may reset and restore a different prefix from disk. Responses include counts/timings/KV allocation metadata;
authenticated `GET /v1/nanoattention/metrics` exposes only bounded content-free
request counters. No prompt, tool arguments or credentials are returned there.

SSE has heartbeat comments during native inference, but generated deltas are
sent **after the complete native response**. It is buffered SSE compatibility,
not token-live generation or a first-token-latency improvement.

## Full-window scope

`--context 262144` configures the checkpoint's full capacity on the currently
supported M3 Ultra/GQA8 long-context profile. FP32 KV allocation is approximately
14.07 GiB, excluding weights and other buffers. A short Codegraff task with that
allocation does **not** exercise 262K-token attention or establish full-window
throughput/quality. Filling that window still requires real prefix inference;
sequential prefill can be very slow and both adapter/client deadlines matter.

## Isolated verification

```sh
python3 -m unittest discover -s tools -p 'test_*adapter.py'
python3 -m unittest discover -s tools -p 'test_flint_tool_protocol.py'
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1800 -- \
  /path/to/tokenizer-python -B tools/check_codegraff_adapter.py /path/to/flint-int4 \
  --graff /path/to/installed/graff --context 262144 \
  --output target/codegraff-adapter-check
```

The driver creates a fresh synthetic Git workspace and HOME, supplies an empty
MCP override, disables telemetry/learning egress, and uses the installed client's
real local-router path. macOS `sandbox-exec` restricts Graff and verification
children to fixture writes and the adapter's loopback port, denying unrelated
user-file reads and Keychain IPC. It verifies routing before inference, requires
real file-tool edits, unchanged tests, observed native KV reuse and passing
fixture tests, then stops its owned adapter. Raw client transcripts stay ignored
under `target/`, not in the repository.

## Executed result

Installed **Codegraff 0.302.74.68** completed the synthetic clamp bug fix using
native Flint through this adapter. The real coding loop made **five completion
requests**, used **one native model worker**, and produced a changed implementation
that passed **all six unchanged fixture tests**. No native/protocol failures were
recorded. All **85 Python tests** pass, including 20 tool-protocol and 10 adapter
contract tests.

| Request | Cached prefix reused | New prefix IDs | Generated IDs | Cache after |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 2,652 | 34 | 2,686 |
| 2 | 2,686 | 190 | 150 | 3,026 |
| 3 | 3,026 | 16 | 18 | 3,060 |
| 4 | 3,060 | 104 | 19 | 3,183 |
| 5 | 3,183 | 97 | 89 | 3,369 |

The four follow-up calls were genuine append-only cache hits, not reset/replay.
The adapter was configured for **262,144 positions** and reported **15,105,929,216
bytes (14.07 GiB)** of KV capacity. Only **3,369 tokens** were actually cached at
the end; valid KV was **266,735,616 bytes**. This verifies full-size allocation
with a short coding session, **not full-window attention, throughput or quality**.

The final adapter and Codegraff processes exited normally; an OS process check
found no matching adapter/native workers afterward and the inference lock was
free. Earlier attempts stopped before inference due to test-driver profile/CLI
preflight mistakes; they are not native model crashes or successful coding runs.
The passing run used the corrected sandbox and CLI configuration. No global
Codegraff provider settings were changed.

[Sanitized result and source fingerprints](../benchmarks/codegraff-adapter.json).
