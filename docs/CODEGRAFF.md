# Codegraff adapter: native Flint over loopback OpenAI chat

`tools/codegraff_adapter.py` translates Codegraff's OpenAI-compatible requests to
one persistent `flint session` subprocess. **All model inference remains Zig +
Metal.** Python provides HTTP, the checkpoint's local tokenizer/chat template,
and Flint-tool-call/OpenAI-tool-call translation; no MLX/PyTorch model is loaded.

The integration uses Codegraff's existing project-local router contract, inspected
at `v0.302.74.68:src/router_config.zig`. No Codegraff source changes or global
provider/key settings are required. Model ID is `flint`, provider ID
`nanoattention`. The example contains endpoint metadata, not a credential.

## Run

From nanoattention, build the native executable, then run the adapter under the
same exclusive inference lock/resource guard used by other local Flint jobs:

```sh
zig build -Doptimize=ReleaseSafe
export NANOATTENTION_API_KEY=local  # development placeholder; loopback only
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 14400 -- \
  /path/to/tokenizer-python -B tools/codegraff_adapter.py /path/to/flint-int4 \
  --context 32768 --max-tokens 4096 --timeout 900 --port 8080
```

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
- Changed history/tool preambles or non-round-tripping output cause an explicit
  native reset and full replay, reported as `cache_reset`; there is no fabricated
  position, silent truncation or claim that a replay was a cache hit.
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
cache service. Concurrent inference returns 429. Switching conversations can
reset the cache. Responses include counts/timings/KV allocation metadata;
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
