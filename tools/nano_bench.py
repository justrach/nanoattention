#!/usr/bin/env python3
"""Megakernel throughput under the repository protocol (docs/QD.md quality stage).

  tokens: write the stock-MLX baseline's workload (BOS + tools/benchmark.py
          CONTINUATION ids, and its 256 teacher-forced decode ids) for `nano bench
          --bench-tokens`, and check it against benchmarks/lfm3-mlx-baseline.json.
          Needs the `tokenizers` package; run it with -I.
  run:    for each prompt length, one guarded `nano bench` process (one warm-up run,
          then --repeats timed runs from a reset state; medians), under the
          inference lock. Loading and warm-up are excluded.

  versus: (prefill-mlx) the engine's candidates and stock MLX interleaved at each
          length: per round, one guarded `nano bench` per candidate, then one guarded
          tools/lfm3_benchmark.py process per MLX engine ($NANO_MLX_PYTHON; the same workload,
          warm-up plus --repeats runs, medians). Prints and writes every round.

  /path/to/tokenizer-python -I tools/nano_bench.py tokens CKPT --out target/nano-bench-tokens.json
  python3 tools/nano_bench.py run zig-out/bin/nano CKPT --bench-tokens target/nano-bench-tokens.json \\
      --knobs cand.json --lengths 64,1024,4096 --output target/bench.json
"""
import os
import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
LOCK = pathlib.Path(os.environ.get("NANO_LOCK", pathlib.Path(os.environ.get("NANO_WORK", HERE.parent / "target/nano-work")) / "inference.lock"))


def cmd_tokens(args):
    sys.path.insert(0, str(HERE))
    from benchmark import CONTINUATION
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.checkpoint / "tokenizer.json"))
    config = json.loads((args.checkpoint / "config.json").read_text())
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    code = encode(CONTINUATION)
    copies = 4
    while len(encode(CONTINUATION * copies)) < 256:
        copies *= 2
    decode = encode(CONTINUATION * copies)[:256]
    out = {"bos": config["bos_token_id"], "code": code, "decode": decode,
           "source": "BOS + tools/benchmark.py CONTINUATION token ids repeated; decode = CONTINUATION repeated, first 256"}
    if args.baseline.exists():
        base = json.loads(args.baseline.read_text())["input"]
        prefix = ([out["bos"]] + (code * 64))[:64]
        out["matches_mlx_baseline"] = prefix == base["prompt_prefix_first_64"] and decode == base["decode"]
        if not out["matches_mlx_baseline"]:
            raise SystemExit("workload differs from the MLX baseline input")
    args.out.write_text(json.dumps(out))
    print("wrote %s: %d code ids, %d decode ids, matches baseline: %s" % (args.out, len(code), len(decode), out.get("matches_mlx_baseline")))


def guarded(argv, timeout, wait=7200):
    """Run argv under tools/guarded_run.py with the shared lock, retrying while it is held."""
    full = [sys.executable, str(HERE / "guarded_run.py"), "--lock", str(LOCK), "--timeout", str(timeout), "--"] + argv
    deadline = time.monotonic() + wait
    while True:
        proc = subprocess.run(full, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout
        # Retry while another agent holds the lock or the guard sees too little memory headroom.
        if any(m in proc.stderr for m in ("BlockingIOError", "Resource temporarily unavailable", "memory headroom")):
            if time.monotonic() > deadline:
                raise SystemExit("inference lock or memory headroom unavailable for too long")
            time.sleep(3)  # short poll: several jobs may share the lock
            continue
        raise SystemExit("guarded run failed (%s):\n%s" % (" ".join(argv), proc.stderr[-3000:]))


def bench_once(binary, checkpoint, length, knobs, bench_tokens, decode=256, repeats=3):
    argv = [str(binary), "bench", str(checkpoint), "--prompt-tokens", str(length), "--decode-tokens", str(decode),
            "--repeats", str(repeats), "--context", str(length + decode)]
    if knobs:
        argv += ["--knobs", str(knobs)]
    if bench_tokens:
        argv += ["--bench-tokens", str(bench_tokens)]
    out = guarded(argv, timeout=900)
    return json.loads(out.strip().splitlines()[-1])


def cmd_run(args):
    rows = []
    for length in [int(x) for x in args.lengths.split(",")]:
        r = bench_once(args.binary, args.checkpoint, length, args.knobs, args.bench_tokens, args.decode_tokens, args.repeats)
        print("L=%5d prefill %.1f tok/s, decode %.2f tok/s, %.1f dispatches/decode token" % (
            length, r["prefill_tokens_per_second"], r["decode_tokens_per_second"], r["dispatches_per_decode_token"]), flush=True)
        rows.append(r)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=1))


MLX_PYTHON = pathlib.Path(os.environ.get("NANO_MLX_PYTHON", sys.executable))


def mlx_once(checkpoint, length, engine, decode, repeats, workdir):
    out = workdir / ("mlx-%s-%d.json" % (engine, length))
    argv = [str(MLX_PYTHON), "-I", str(HERE / "lfm3_benchmark.py"), str(checkpoint), "--lengths", str(length),
            "--engines", engine, "--decode-tokens", str(decode), "--repeats", str(repeats), "--output", str(out)]
    guarded(argv, timeout=900)
    r = json.loads(out.read_text())["engines"][engine][str(length)]
    out.unlink()
    return r


def cmd_versus(args):
    import statistics
    import tempfile
    sets = []
    for spec in args.candidates:
        name, _, path = spec.partition("=")
        cand = json.loads(pathlib.Path(path).read_text())
        sets.append((name, cand.get("knobs", cand)))
    engines = [e for e in args.mlx.split(",") if e]
    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "binary": str(args.binary), "rounds": args.rounds,
              "repeats": args.repeats, "decode_tokens": args.decode_tokens, "lengths": {}}
    with tempfile.TemporaryDirectory(prefix="versus-", dir=HERE.parent / "target") as tmp:
        workdir = pathlib.Path(tmp)
        for length in [int(x) for x in args.lengths.split(",")]:
            runs = {name: [] for name, _ in sets}
            runs.update({e: [] for e in engines})
            for rnd in range(args.rounds):
                for name, knobs in sets:
                    path = workdir / ("knobs-%s.json" % name)
                    path.write_text(json.dumps(knobs))
                    r = bench_once(args.binary, args.checkpoint, length, path, args.bench_tokens, args.decode_tokens, args.repeats)
                    runs[name].append({"prefill": r["prefill_tokens_per_second"], "decode": r["decode_tokens_per_second"]})
                    print("L=%5d round %d %-14s prefill %8.1f decode %7.2f" % (length, rnd, name, runs[name][-1]["prefill"], runs[name][-1]["decode"]), flush=True)
                for e in engines:
                    r = mlx_once(args.checkpoint, length, e, args.decode_tokens, args.repeats, workdir)
                    runs[e].append({"prefill": r["prefill_tokens_per_second"], "decode": r["decode_tokens_per_second"]})
                    print("L=%5d round %d %-14s prefill %8.1f decode %7.2f" % (length, rnd, e, runs[e][-1]["prefill"], runs[e][-1]["decode"]), flush=True)
            report["lengths"][str(length)] = {
                name: {"prefill_rounds": [x["prefill"] for x in v], "decode_rounds": [x["decode"] for x in v],
                       "prefill_median": statistics.median(x["prefill"] for x in v),
                       "decode_median": statistics.median(x["decode"] for x in v)} for name, v in runs.items()}
            for name, v in report["lengths"][str(length)].items():
                print("L=%5d %-14s median prefill %8.1f decode %7.2f" % (length, name, v["prefill_median"], v["decode_median"]), flush=True)
            if args.output:
                args.output.write_text(json.dumps(report, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("checkpoint", type=pathlib.Path)
    t.add_argument("--out", type=pathlib.Path, required=True)
    t.add_argument("--baseline", type=pathlib.Path, default=HERE.parent / "benchmarks/lfm3-mlx-baseline.json")
    r = sub.add_parser("run")
    r.add_argument("binary", type=pathlib.Path)
    r.add_argument("checkpoint", type=pathlib.Path)
    r.add_argument("--knobs", type=pathlib.Path)
    r.add_argument("--bench-tokens", type=pathlib.Path)
    r.add_argument("--lengths", default="64,1024,4096")
    r.add_argument("--decode-tokens", type=int, default=256)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--output", type=pathlib.Path)
    v = sub.add_parser("versus")
    v.add_argument("binary", type=pathlib.Path)
    v.add_argument("checkpoint", type=pathlib.Path)
    v.add_argument("candidates", nargs="+", help="NAME=candidate-or-knobs.json")
    v.add_argument("--bench-tokens", type=pathlib.Path, default=HERE.parent / "target/nano-bench-tokens.json")
    v.add_argument("--mlx", default="mlx_f32,mlx_bf16")
    v.add_argument("--lengths", default="64,1024,4096")
    v.add_argument("--rounds", type=int, default=3)
    v.add_argument("--decode-tokens", type=int, default=256)
    v.add_argument("--repeats", type=int, default=3)
    v.add_argument("--output", type=pathlib.Path)
    args = p.parse_args()
    {"tokens": cmd_tokens, "run": cmd_run, "versus": cmd_versus}[args.command](args)


if __name__ == "__main__":
    main()
