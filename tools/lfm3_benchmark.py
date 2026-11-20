#!/usr/bin/env python3
"""LFM3 throughput baseline: stock mlx-vlm on this checkpoint, and later the native engine.

Run under tools/guarded_run.py with the inference lock; this script takes no lock itself.
Mirrors tools/benchmark.py: identical token ids for every engine, a fresh cache each run,
one warm-up run (prefill plus 8 decode tokens) then --repeats measured runs, medians
reported; model loading, cache creation, dtype conversion and warm-up are excluded.

Workload per prompt length L (default 64, 1024, 4096): BOS followed by the stock token ids
of benchmark.py's CONTINUATION code text, repeated and cut to exactly L tokens (raw, no chat
template; a performance workload, not a quality test), then 256 decode tokens.

  prefill: stock mlx-vlm batched prefill as generate() does it: chunks of at most 2048
           (DEFAULT_PREFILL_STEP_SIZE) over the first L - 1 tokens, evaluating only the cache
           state, then the last prompt token with logits and a host argmax.
  decode:  256 teacher-forced tokens (benchmark.py's CONTINUATION ids), one cached forward
           and a synchronous host argmax (`.item()`) per token, like benchmark.py.
  decode, pipelined (engines mlx_bf16_async, mlx_f32_async): stock generate()'s loop.
           Free-running greedy: each step's input is the previous step's lazy argmax, and
           the next step is submitted with mx.async_eval before the current token is read
           on the host, so the host's read-back latency overlaps the next forward. Same
           number of forwards; argmax only (stock also computes log-probabilities).

Engines: mlx_bf16 (the checkpoint's own dtype, stock default) and mlx_f32
(model.set_dtype(float32)); each also as NAME_async (pipelined decode, above). --native BINARY instead times `BINARY session CHECKPOINT` over
its JSONL protocol: per run {"reset": true, "tokens": prompt, "max_tokens": 256}, rates from
the response's prefill_seconds and decode_seconds / decode_forward_tokens. The session
protocol cannot teacher-force, so native decode is free-running greedy (one forward per
token, like the teacher-forced runs); its token count is recorded.

  python3 tools/guarded_run.py --lock $NANO_LOCK --timeout 1800 -- \
      VENV/bin/python -I tools/lfm3_benchmark.py CKPT --output benchmarks/lfm3-mlx-baseline.json
"""
import argparse
import gc
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from benchmark import CONTINUATION  # noqa: E402

STOCK_PREFILL_STEP = 2048  # mlx_vlm.generate.common.DEFAULT_PREFILL_STEP_SIZE
WARMUP_DECODE = 8


def workload(tok, lengths, decode_tokens):
    code = list(tok.encode(CONTINUATION, add_special_tokens=False))
    prompts = {}
    for n in lengths:
        body = (code * (n // len(code) + 1))[:n - 1]
        prompts[n] = [tok.bos_token_id] + body
        assert len(prompts[n]) == n
    copies = 4
    while len(tok.encode(CONTINUATION * copies, add_special_tokens=False)) < decode_tokens:
        copies *= 2
    decode = list(tok.encode(CONTINUATION * copies, add_special_tokens=False))[:decode_tokens]
    return prompts, decode


def mlx_engine(checkpoint, dtype, prompts, decode, repeats, pipelined=False):
    import mlx.core as mx
    from mlx_vlm.utils import load_model

    mx.set_memory_limit(32 * 1024 ** 3)
    mx.set_cache_limit(512 * 1024 ** 2)
    model = load_model(pathlib.Path(checkpoint))
    model.set_dtype(getattr(mx, dtype))
    mx.eval(model.parameters())
    lm = model.language_model
    engine = "mlx_" + {"bfloat16": "bf16", "float32": "f32"}[dtype] + ("_async" if pipelined else "")

    def step(y):
        return mx.argmax(lm(y.reshape(1, 1), cache=cache_ref[0]).logits[0, -1])

    cache_ref = [None]

    def run(prompt, count):
        cache = lm.make_cache()
        start = time.perf_counter()
        ids = mx.array([prompt], dtype=mx.int32)
        done = 0
        while len(prompt) - done > 1:
            n = min(STOCK_PREFILL_STEP, len(prompt) - 1 - done)
            lm(ids[:, done:done + n], cache=cache)
            mx.eval([c.state for c in cache])
            done += n
        logits = lm(ids[:, done:], cache=cache).logits
        first = int(mx.argmax(logits[0, -1]).item())
        prefilled = time.perf_counter()
        checksum = first
        if pipelined:
            cache_ref[0] = cache
            upcoming = step(mx.array(first, dtype=mx.int32))
            mx.async_eval(upcoming)
            for i in range(count):
                current = upcoming
                if i + 1 < count:
                    upcoming = step(current)
                    mx.async_eval(upcoming)
                checksum += int(current.item())
        else:
            for token in decode[:count]:
                logits = lm(mx.array([[token]], dtype=mx.int32), cache=cache).logits
                checksum += int(mx.argmax(logits[0, -1]).item())
        finished = time.perf_counter()
        return {"prefill_seconds": prefilled - start, "decode_seconds": finished - prefilled,
                "decode_tokens": count, "checksum": checksum}

    rows = []
    for n, prompt in prompts.items():
        run(prompt, WARMUP_DECODE)
        for repeat in range(repeats):
            row = {"engine": engine, "prompt_tokens": n, "repeat": repeat, **run(prompt, len(decode))}
            print(json.dumps(row), flush=True)
            rows.append(row)
        mx.clear_cache()
    del model, lm
    gc.collect()
    return rows


def native_engine(binary, checkpoint, extra, prompts, decode_count, repeats):
    """Times the native session; it must have exited before any MLX engine loads."""
    proc = subprocess.Popen([str(binary), "session", str(checkpoint), *extra],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    rows, next_id = [], 1

    def request(prompt, max_tokens):
        nonlocal next_id
        proc.stdin.write(json.dumps({"id": next_id, "tokens": prompt, "max_tokens": max_tokens, "reset": True}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("native session exited with %s" % proc.wait())
        r = json.loads(line)
        if r.get("id") != next_id:
            raise SystemExit("session id mismatch")
        next_id += 1
        return r

    try:
        for n, prompt in prompts.items():
            request(prompt, WARMUP_DECODE)
            for repeat in range(repeats):
                r = request(prompt, decode_count)
                row = {"engine": "native", "prompt_tokens": n, "repeat": repeat,
                       "prefill_seconds": r["prefill_seconds"], "decode_seconds": r["decode_seconds"],
                       "decode_tokens": r.get("decode_forward_tokens", len(r["tokens"])),
                       "generated_tokens": len(r["tokens"]), "eos": r.get("eos"),
                       "checksum": sum(r["tokens"])}
                print(json.dumps(row), flush=True)
                rows.append(row)
    finally:
        proc.stdin.close()
        proc.wait(timeout=60)
    return rows


def summarize(rows, prompts, decode, repeats, notes):
    report = {"machine": {"platform": platform.platform(), "machine": platform.machine(),
                          "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
                                                capture_output=True).stdout.strip(),
                          "memory_bytes": int(subprocess.run(["sysctl", "-n", "hw.memsize"], text=True,
                                                             capture_output=True).stdout.strip() or 0)},
              "versions": versions(), "repeats": repeats, "warmup": "one run per engine and length: prefill + %d decode tokens" % WARMUP_DECODE,
              "decode_tokens": len(decode), "prompt_lengths": list(prompts), **notes, "engines": {}, "runs": rows}
    encoded = json.dumps({"prompts": {str(k): v for k, v in prompts.items()}, "decode": decode},
                         sort_keys=True, separators=(",", ":")).encode()
    report["input_sha256"] = hashlib.sha256(encoded).hexdigest()
    report["input"] = {"prompt_prefix_first_64": prompts[min(prompts)][:64], "decode": decode}
    for engine in sorted({r["engine"] for r in rows}):
        report["engines"][engine] = {}
        for n in prompts:
            sel = [r for r in rows if r["engine"] == engine and r["prompt_tokens"] == n]
            if len(sel) != repeats:
                raise SystemExit("%s at %d: %d runs, expected %d" % (engine, n, len(sel), repeats))
            if engine != "native" and len({r["checksum"] for r in sel}) != 1:
                raise SystemExit("%s at %d: greedy checksum changed between identical repeats" % (engine, n))
            prefill = statistics.median(r["prefill_seconds"] for r in sel)
            decode_s = statistics.median(r["decode_seconds"] for r in sel)
            tokens = sel[0]["decode_tokens"]
            report["engines"][engine][str(n)] = {
                "median_prefill_seconds": prefill, "prefill_tokens_per_second": n / prefill,
                "median_decode_seconds": decode_s, "decode_tokens_per_second": tokens / decode_s,
                "prefill_run_seconds": [r["prefill_seconds"] for r in sel],
                "decode_run_seconds": [r["decode_seconds"] for r in sel],
                "decode_tokens": [r["decode_tokens"] for r in sel],
            }
    return report


def versions():
    import importlib.metadata as md
    v = {"python": platform.python_version(), "macos": platform.mac_ver()[0]}
    for pkg in ("mlx", "mlx-vlm", "transformers", "numpy"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--lengths", default="64,1024,4096")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--engines", default="mlx_bf16,mlx_f32", help="any of mlx_bf16, mlx_f32, mlx_bf16_async, mlx_f32_async, native")
    parser.add_argument("--native", type=pathlib.Path, help="native binary with a `session` command")
    parser.add_argument("--session-arg", action="append", default=[], help="extra `session` argument (repeatable)")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if args.output and checkpoint in args.output.resolve().parents:
        parser.error("--output must not be inside the checkpoint")
    engines = args.engines.split(",")
    if "native" in engines and not args.native:
        parser.error("the native engine needs --native BINARY")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    prompts, decode = workload(tok, [int(x) for x in args.lengths.split(",")], args.decode_tokens)
    del tok
    rows = []
    if "native" in engines:
        rows += native_engine(args.native.resolve(), checkpoint, args.session_arg, prompts, len(decode), args.repeats)
    for engine in engines:
        if engine.startswith("mlx_"):
            base = engine[:-len("_async")] if engine.endswith("_async") else engine
            rows += mlx_engine(checkpoint, {"mlx_bf16": "bfloat16", "mlx_f32": "float32"}[base],
                               prompts, decode, args.repeats, pipelined=engine.endswith("_async"))
    notes = {
        "prompt_protocol": "BOS + benchmark.py CONTINUATION token ids repeated and cut to exactly L tokens; no chat template",
        "prefill_protocol": "stock mlx-vlm generate(): chunks of <= %d over the first L - 1 tokens (cache state "
                            "evaluated), then the last token with logits and host argmax; native: the session's "
                            "prefill_seconds" % STOCK_PREFILL_STEP,
        "decode_protocol": "MLX: teacher-forced, one cached forward and synchronous host argmax per token; "
                           "MLX *_async: free-running greedy, the next step submitted with async_eval before "
                           "the current token is read (stock generate()'s loop); native: free-running greedy "
                           "through the session",
        "excluded": "model load, dtype conversion, cache creation, warm-up",
    }
    report = summarize(rows, prompts, decode, args.repeats, notes)
    text = json.dumps(report, indent=1) + "\n"
    if args.output:
        args.output.write_text(text)
    print(json.dumps(report["engines"], indent=1))


if __name__ == "__main__":
    main()
