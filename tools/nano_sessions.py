#!/usr/bin/env python3
"""Persistent `nano session` checks on tiny LFM3 checkpoints (cf. tools/check_sessions.py).

For each request: prefill/final logits against the float64 NumPy reference
(tools/lfm3_reference.py, one token at a time through its caches) within the
float32 gate of tools/nano_verify.py, greedy tokens equal to the reference's
argmax, every generated token (EOS included) committed, cache positions, reset,
state byte accounting, and a fresh `nano eval` replay of the whole history whose
final all-vocabulary row is bit-identical to the session's. Then: the same
session at several batch sizes must be bit-identical, EOS must stop and be
committed, and a request that overflows the context must fail before output.

  python3 tools/nano_sessions.py zig-out/bin/nano --tiny target/lfm3-tiny
"""
import os
import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lfm3_reference import Checkpoint, NumpyLFM3  # noqa: E402

CONTEXT = 48


def session(binary, directory, requests, batch, context=CONTEXT, check=True):
    payload = "\n".join(json.dumps(r) for r in requests)  # no final newline: EOF must flush it
    proc = subprocess.run([str(binary), "session", str(directory), "--context", str(context), "--knob", "batch=%d" % batch],
                          input=payload, text=True, capture_output=True, timeout=300)
    if check and proc.returncode != 0:
        raise SystemExit("session failed: " + proc.stderr[-3000:])
    return proc, [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def state_bytes(cfg, batch, context, position):
    hidden, kvw = cfg["hidden_size"], cfg["num_key_value_heads"] * cfg["head_dim"]
    window, taps = cfg["sliding_window"], cfg["conv_L_cache"]
    states = []
    for t in cfg["layer_types"]:
        if t == "conv":
            states.append((min(taps - 1 + batch, context), hidden))
        elif t == "sliding_attention":
            states += [(min(window - 1 + batch, context), kvw)] * 2
        else:
            states += [(context, kvw)] * 2
    allocated = sum(c * w * 4 for c, w in states)
    valid = sum(min(position, c) * w * 4 for c, w in states)
    return allocated, valid


def check_case(binary, directory, workdir):
    cfg = json.loads((directory / "config.json").read_text())
    eos = cfg["eos_token_id"]
    requests = [
        dict(id=1, tokens=[1, 17, 93, 4, 250, 8, 61], max_tokens=5, return_logits=True),
        dict(id=2, tokens=[61, 130], max_tokens=6, return_logits=True),
        dict(id=3, tokens=[7], max_tokens=0, return_logits=True),
        dict(id=4, tokens=[3, 199, 88, 12, 5, 240, 33, 64, 9], max_tokens=4, reset=True, return_logits=True),
        dict(id=5, tokens=[44, 45], max_tokens=7, return_logits=True),
    ]
    _, rows = session(binary, directory, requests, batch=4)
    assert len(rows) == len(requests), rows
    ref = NumpyLFM3(Checkpoint(str(directory))).load()
    f32 = NumpyLFM3(Checkpoint(str(directory)), dtype=np.float32).load()
    st, st32 = ref.new_state(), f32.new_state()
    history, worst, floor = [], 0.0, 0.0

    def forward(token):
        nonlocal floor
        _, lg = ref.step(token, st)
        _, lg32 = f32.step(token, st32)
        floor = max(floor, float(np.abs(lg32.astype(np.float64) - lg).max()))
        return lg

    for request, row in zip(requests, rows):
        if request.get("reset"):
            history, st, st32 = [], ref.new_state(), f32.new_state()
        before = len(history)
        assert row["id"] == request["id"]
        assert row["cache_before"] == row["reused_prefix_tokens"] == before
        assert row["prefill_tokens"] == len(request["tokens"])
        for token in request["tokens"]:
            logits = forward(token)
        worst = max(worst, float(np.abs(np.asarray(row["prefill_logits"], np.float64) - logits).max()))
        generated = row["tokens"]
        assert len(generated) <= request["max_tokens"]
        for token in generated:
            assert token == int(np.argmax(logits)), "greedy token mismatch"
            logits = forward(token)  # every token, the last and EOS included, is committed
        history += request["tokens"] + generated
        assert row["decode_forward_tokens"] == len(generated)
        assert row["cache_after"] == len(history) == st["position"]
        assert row["eos"] == bool(generated and generated[-1] in eos)
        allocated, valid = state_bytes(cfg, 4, CONTEXT, len(history))
        assert row["kv_allocated_bytes"] == allocated, (row["kv_allocated_bytes"], allocated)
        assert row["kv_valid_bytes"] == valid, (row["kv_valid_bytes"], valid)
        worst = max(worst, float(np.abs(np.asarray(row["final_logits"], np.float64) - logits).max()))
        source = workdir / "history.json"
        source.write_text(json.dumps(history))
        out = workdir / "replay.f32"
        fresh = subprocess.run([str(binary), "eval", str(directory), "--context", str(CONTEXT), "--tokens-file", str(source),
                                "--positions", str(len(history) - 1), "--logits-out", str(out)],
                               capture_output=True, text=True, timeout=300)
        assert fresh.returncode == 0, fresh.stderr
        expected = np.fromfile(out, dtype="<f4")
        actual = np.asarray(row["final_logits"], dtype=np.float32)
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), "continued session != fresh replay"
    gate = 2 * floor
    assert worst <= gate, "session logits off the float64 reference: %g > %g" % (worst, gate)

    for batch in (1, 3, 16):
        _, other = session(binary, directory, requests, batch=batch)
        for a, b in zip(rows, other):
            for key in ("tokens", "cache_after", "eos", "kv_valid_bytes"):
                assert a[key] == b[key] or key == "kv_valid_bytes", (batch, key)
            for key in ("prefill_logits", "final_logits"):
                assert np.array_equal(np.asarray(a[key], np.float32).view(np.uint32), np.asarray(b[key], np.float32).view(np.uint32)), (batch, key)

    # EOS: relabel the token greedy emits third as EOS; generation must stop there and commit it.
    third = rows[0]["tokens"][2]
    eos_dir = workdir / "eos"
    shutil.copytree(directory, eos_dir, ignore=shutil.ignore_patterns("golden"))
    cfg2 = dict(cfg, eos_token_id=[third])
    (eos_dir / "config.json").write_text(json.dumps(cfg2))
    _, eos_rows = session(binary, eos_dir, [requests[0], dict(id=9, tokens=[5], max_tokens=2)], batch=4)
    stop = rows[0]["tokens"].index(third) + 1
    assert eos_rows[0]["eos"] and eos_rows[0]["tokens"] == rows[0]["tokens"][:stop], eos_rows[0]["tokens"]
    assert eos_rows[0]["cache_after"] == len(requests[0]["tokens"]) + stop
    assert eos_rows[1]["cache_before"] == eos_rows[0]["cache_after"]
    shutil.rmtree(eos_dir)

    # Capacity reserves every generated token: 7 + 42 > 48 fails before any output.
    bad, out_rows = session(binary, directory, [dict(tokens=requests[0]["tokens"], max_tokens=42)], batch=4, check=False)
    assert bad.returncode != 0 and not out_rows, "overflow must fail before output"
    _, fits = session(binary, directory, [dict(tokens=requests[0]["tokens"], max_tokens=41)], batch=4)
    assert fits[0]["cache_after"] <= CONTEXT
    print("PASS %-14s 5 turns, reference gate %.3g (worst %.3g), fresh replay bitwise, batch 1/3/4/16 bitwise, EOS, reset, bytes, overflow"
          % (directory.name, gate, worst), flush=True)


def check_real(binary, directory, workdir, context):
    """Real checkpoint: no float64 reference (too slow); continued == fresh replay
    bit for bit, and batch sizes 1 / 64 / 256 agree bit for bit."""
    root = pathlib.Path(os.environ.get("NANO_WORK", pathlib.Path(__file__).resolve().parent.parent / "target/nano-work")) / "golden/lfm3"
    golden = json.loads((root / "window/tokens.json").read_text())["tokens"]
    short = json.loads((root / "short/tokens.json").read_text())["tokens"]
    short_greedy = json.loads((root / "short/greedy.json").read_text())["tokens"]
    cfg = json.loads((directory / "config.json").read_text())
    eos = cfg["eos_token_id"]
    assert eos in ([124900], 124900), eos  # the only stop token (docs/LFM3.md section 6)
    requests = [dict(id=1, tokens=golden[:300], max_tokens=8, return_logits=True),
                dict(id=2, tokens=golden[300:340], max_tokens=16, return_logits=True),
                dict(id=3, tokens=[124899], max_tokens=0, return_logits=True),
                dict(id=4, tokens=golden[:30], max_tokens=4, reset=True, return_logits=True),
                # EOS: the short golden's greedy reply ends in 124900 after 9 tokens; 20 are allowed.
                dict(id=5, tokens=short, max_tokens=20, reset=True, return_logits=True)]
    results = {}
    for batch in (1, 64, 256):
        _, rows = session(binary, directory, requests, batch=batch, context=context)
        results[batch] = rows
    history = []
    for request, row in zip(requests, results[1]):
        if request.get("reset"):
            history = []
        assert row["cache_before"] == len(history)
        history += request["tokens"] + row["tokens"]
        assert row["cache_after"] == len(history)
        assert row["prefill_seconds"] > 0 and row["decode_seconds"] >= 0
        for batch in (1, 64, 256):
            r = results[batch][requests.index(request)]
            allocated, valid = state_bytes(cfg, batch, context, len(history))
            assert (r["kv_allocated_bytes"], r["kv_valid_bytes"]) == (allocated, valid), (batch, r["kv_allocated_bytes"], allocated)
        source = workdir / "history.json"
        source.write_text(json.dumps(history))
        out = workdir / "replay.f32"
        fresh = subprocess.run([str(binary), "eval", str(directory), "--context", str(context), "--tokens-file", str(source),
                                "--feed", "64", "--knob", "batch=64",
                                "--positions", str(len(history) - 1), "--logits-out", str(out)], capture_output=True, text=True, timeout=600)
        assert fresh.returncode == 0, fresh.stderr
        expected = np.fromfile(out, dtype="<f4")
        assert np.array_equal(np.asarray(row["final_logits"], np.float32).view(np.uint32), expected.view(np.uint32)), "replay mismatch"
    last = results[1][-1]
    assert last["tokens"] == short_greedy and last["eos"] and short_greedy[-1] == 124900, last["tokens"]
    assert last["cache_after"] == len(short) + len(short_greedy) and last["decode_forward_tokens"] == len(short_greedy)
    for batch in (64, 256):
        for a, b in zip(results[1], results[batch]):
            assert a["tokens"] == b["tokens"] and a["eos"] == b["eos"]
            for key in ("prefill_logits", "final_logits"):
                assert np.array_equal(np.asarray(a[key], np.float32).view(np.uint32), np.asarray(b[key], np.float32).view(np.uint32)), (batch, key)
    print("PASS real checkpoint: 5 turns (300-token prefill crossing the 256 window, reset, EOS stop on 124900 = short golden), fresh replay bitwise, batch 1/64/256 bitwise;"
          " generated %s" % [r["tokens"] for r in results[1]], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("binary", type=pathlib.Path)
    p.add_argument("--tiny", type=pathlib.Path)
    p.add_argument("--checkpoint", type=pathlib.Path, help="also check the real checkpoint (untimed)")
    p.add_argument("--cases", default="f32,q4_g32,q8_g16,q4_g32_conv7")
    args = p.parse_args()
    target = pathlib.Path("target")
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nano-sessions-", dir=target) as tmp:
        if args.tiny:
            for name in args.cases.split(","):
                check_case(args.binary.resolve(), args.tiny / name, pathlib.Path(tmp))
        if args.checkpoint:
            check_real(args.binary.resolve(), args.checkpoint, pathlib.Path(tmp), 1024)
    print("PASS")


if __name__ == "__main__":
    main()
