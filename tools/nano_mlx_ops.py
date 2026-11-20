#!/usr/bin/env python3
"""The megakernel's mlx-arith kernels against the stock MLX operations they port.

docs/QD.md, "The mlx-arith tier": a kernel in the mlx-arith tier must be bit-identical
to the stock MLX operation it ports, on the same inputs, in float32. This tool checks
that, and times the stock operations at LFM3's prefill shapes.

  check: for every case (shape, batch M, weight source), build identical inputs, run
         `nano mlx-op` (the engine's own encoder and kernel on one Metal device, the
         same dispatch path a forward takes) and the stock MLX operation
         (mx.quantized_matmul with float32 activations, scales and biases, as
         model.set_dtype(float32) leaves them), and compare the output words bit for
         bit. Weights are random (codes, bf16-representable scales/biases) or a real
         checkpoint matrix; the engine reads them from a buffer whose data starts 2
         bytes off a word boundary, like the checkpoint's. Batches include sizes that
         are not multiples of the 32-row block and both sides of MLX's split-K rule.
         Writes target/nano-mlx-ops.json (nano_qd.py attaches it to mlx-arith trials).
         Uses the GPU but loads no full model: run it outside timed windows or under
         the lock when timing matters to someone else.
  time:  GPU time of each stock MLX operation LFM3's prefill runs, per call, at
         batches 64..4096 (float32, as mlx_f32), plus whole-model chunk times.
         Loads the checkpoint: run it under the inference lock.

  $NANO_MLX_PYTHON -I tools/nano_mlx_ops.py check zig-out/bin/nano
  python3 tools/guarded_run.py --lock $NANO_LOCK --timeout 1200 -- \\
      $NANO_MLX_PYTHON -I tools/nano_mlx_ops.py time --out target/mlx-op-times.json
"""
import argparse
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CHECKPOINT = pathlib.Path(os.environ.get("LFM3_CHECKPOINT", REPO / "target/lfm3-1.7b-mlx-4bit"))
PREFIX = "language_model.model."
GROUP = 64

# LFM3-1.7B linears as (name, K = in, N = out); layer 0 is conv, layer 2 attention.
LINEARS = [
    ("ffn.w1/w3", 2048, 7680, "layers.0.feed_forward.w1"),
    ("ffn.w2", 7680, 2048, "layers.0.feed_forward.w2"),
    ("conv.in_proj", 2048, 6144, "layers.0.conv.in_proj"),
    ("q/o/out_proj", 2048, 2048, "layers.0.conv.out_proj"),
    ("k/v_proj", 2048, 512, "layers.2.self_attn.k_proj"),
    ("gate_proj", 2048, 16, "layers.2.self_attn.gate_proj"),
    ("head (tied embed)", 2048, 128000, "embed_tokens"),
]


def bf16_round(a):
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    u = ((u.astype(np.uint64) + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)


def random_matrix(rng, n, k):
    words = rng.integers(0, 2 ** 32, size=(n, k // 8), dtype=np.uint64).astype(np.uint32)
    scales = bf16_round(rng.uniform(0.002, 0.05, size=(n, k // GROUP)).astype(np.float32))
    biases = bf16_round(rng.uniform(-0.4, -0.02, size=(n, k // GROUP)).astype(np.float32))
    return words, scales, biases


def real_matrix(base):
    import mlx.core as mx
    t = mx.load(str(CHECKPOINT / "model.safetensors"))
    w = np.array(t[PREFIX + base + ".weight"])
    s = np.array(t[PREFIX + base + ".scales"].astype(mx.float32))
    b = np.array(t[PREFIX + base + ".biases"].astype(mx.float32))
    return w, s, b


def vector_limit(k, n):
    """MLX get_qmv_batch_limit for applegpu_g13d..g15d (exec.mlxVectorLimit)."""
    return 32 if (k <= 2048 and n <= 2048) else (18 if (k <= 4096 and n <= 4096) else 12)


def mlx_qmm(x, words, scales, biases):
    import mlx.core as mx
    y = mx.quantized_matmul(mx.array(x), mx.array(words), mx.array(scales), mx.array(biases),
                            transpose=True, group_size=GROUP, bits=4)
    mx.eval(y)
    return np.array(y)


def engine_qmm(binary, x, words, scales, biases, workdir, variant, repeats=0, sdtype=2):
    """`nano mlx-op linear`: one buffer laid out as 2 pad bytes, codes, scales, biases in
    dtype sdtype (2 bf16, 1 f16, 0 f32 after 2 more pad bytes). MLX's float32 model holds
    the same values as float32, so every dtype must give MLX's bits."""
    m, k = x.shape
    n = words.shape[0]
    blob = bytearray(b"\0\0")
    blob += words.astype("<u4").tobytes()
    if sdtype == 2:
        s_, b_ = [(a.astype(np.float32).view(np.uint32) >> 16).astype("<u2") for a in (scales, biases)]
        back = lambda a: (a.astype(np.uint32) << 16).view(np.float32)
    elif sdtype == 1:
        s_, b_ = scales.astype("<f2"), biases.astype("<f2")
        back = lambda a: a.astype(np.float32)
    else:
        blob += b"\0\0"
        s_, b_ = scales.astype("<f4"), biases.astype("<f4")
        back = lambda a: a
    assert np.array_equal(back(s_), scales) and np.array_equal(back(b_), biases), "scales/biases must be exact in the dtype"
    blob += s_.tobytes() + b_.tobytes()
    (workdir / "w.bin").write_bytes(bytes(blob))
    x.astype("<f4").tofile(workdir / "x.f32")
    out = workdir / "y.f32"
    cmd = [str(binary), "mlx-op", "linear", "--m", str(m), "--k", str(k), "--n", str(n), "--group", str(GROUP),
           "--weights", str(workdir / "w.bin"), "--x", str(workdir / "x.f32"), "--out", str(out), "--variant", variant,
           "--repeats", str(repeats), "--sdtype", str(sdtype)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit("nano mlx-op failed (%s):\n%s" % (" ".join(cmd), proc.stderr[-3000:]))
    y = np.fromfile(out, dtype="<f4").reshape(m, n)
    out.unlink()
    return y, json.loads(proc.stdout.strip().splitlines()[-1])


# (T query rows, KL keys, window, cache capacity): full causal, sliding (array mask), ring wrap.
ATTENTION = [(9, 9, 0, 0), (32, 32, 0, 0), (33, 33, 0, 0), (64, 64, 0, 0), (100, 700, 0, 0), (1023, 1023, 0, 0),
             (1024, 1024, 0, 0), (2047, 4095, 0, 0), (2048, 4096, 0, 0),
             (64, 64, 256, 319), (300, 300, 256, 555), (100, 700, 256, 355), (1024, 1024, 256, 1279),
             (2048, 4096, 256, 2303), (2047, 4095, 256, 2302)]


# Sliding layers: stock mlx_vlm's RotatingKVCache hands a multi-row chunk that starts at
# position f = KL - T only keys [max(0, f - (window - 1)), KL), so MLX's key blocks start
# there, not at position 0 (the second 2,048-token chunk of a 4,096 prompt sees keys
# 1,793..4,095). Every window case is checked in that layout; these add later chunks.
ROTATING = [(1024, 2048, 256, 1279), (64, 4096, 256, 319)]


def rotating_origin(t, kl, window):
    return max(0, kl - t - (window - 1))


def mlx_sdpa(q, k, v, t, kl, window, heads=16, kv_heads=4, origin=0):
    import mlx.core as mx
    Q = mx.array(q.reshape(t, heads, 128)).transpose(1, 0, 2)[None]
    K = mx.array(k[origin:].reshape(kl - origin, kv_heads, 128)).transpose(1, 0, 2)[None]
    V = mx.array(v[origin:].reshape(kl - origin, kv_heads, 128)).transpose(1, 0, 2)[None]
    if window:
        qpos = np.arange(kl - t, kl)[:, None]
        kpos = np.arange(origin, kl)[None]
        mask = mx.array((qpos >= kpos) & (qpos < kpos + window))
    else:
        mask = "causal"
    o = mx.fast.scaled_dot_product_attention(Q, K, V, scale=128 ** -0.5, mask=mask)
    mx.eval(o)
    return np.array(o[0].transpose(1, 0, 2).reshape(t, heads * 128))


def engine_attention(binary, q, k, v, gate, t, kl, window, capacity, workdir, repeats=0, attention2=False):
    for name, arr in (("q", q), ("k", k), ("v", v), ("g", gate)):
        arr.astype("<f4").tofile(workdir / ("%s.f32" % name))
    out = workdir / "o.f32"
    cmd = [str(binary), "mlx-op", "attention", "--t", str(t), "--kl", str(kl), "--window", str(window),
           "--capacity", str(capacity or kl), "--q", str(workdir / "q.f32"), "--k", str(workdir / "k.f32"),
           "--v", str(workdir / "v.f32"), "--gate", str(workdir / "g.f32"), "--out", str(out), "--repeats", str(repeats)]
    if attention2:  # c2-prefill-mlx: mk_mlx2_attention*
        cmd += ["--attention2", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit("nano mlx-op failed (%s):\n%s" % (" ".join(cmd), proc.stderr[-3000:]))
    o = np.fromfile(out, dtype="<f4").reshape(t, -1)
    out.unlink()
    return o, json.loads(proc.stdout.strip().splitlines()[-1])


def check_attention(args, rng, workdir, results, attention2=False):
    ok = True
    for t, kl, window, capacity in ATTENTION + ROTATING:
        layout = "rotating" if window else "full"
        origin = rotating_origin(t, kl, window) if window else 0
        q = rng.standard_normal((t, 16 * 128)).astype(np.float32)
        k = rng.standard_normal((kl, 4 * 128)).astype(np.float32)
        v = rng.standard_normal((kl, 4 * 128)).astype(np.float32)
        gate = rng.standard_normal((t, 16)).astype(np.float32)
        want = mlx_sdpa(q, k, v, t, kl, window, origin=origin)
        got, info = engine_attention(args.binary, q, k, v, gate, t, kl, window, capacity, workdir, attention2=attention2)
        diff = got.view(np.uint32) != want.view(np.uint32)
        row = {"op": "scaled_dot_product_attention", "ported": True, "t": t, "kl": kl, "window": window, "kernel": info["kernels"],
               "key_layout": layout, "key_origin": origin,
               "capacity": capacity or kl, "words": int(diff.size), "differing_words": int(diff.sum()),
               "max_abs": float(np.abs(got.astype(np.float64) - want).max()), "bit_identical": bool(not diff.any()),
               "gate_epilogue_bit_identical": info["gate_epilogue_bit_identical"]}
        ok = ok and row["bit_identical"] and row["gate_epilogue_bit_identical"]
        results.append(row)
        print("sdpa%s T %5d KL %5d window %3d capacity %5d keys from %4d: %s, gated epilogue %s%s" % (
            " (mlx2)" if attention2 else "", t, kl, window, capacity or kl, origin, "bit-identical" if row["bit_identical"] else "DIFFERS",
            "bit-identical" if row["gate_epilogue_bit_identical"] else "DIFFERS",
            "" if row["bit_identical"] else " (%d of %d words, max abs %.3g)" % (row["differing_words"], row["words"], row["max_abs"])), flush=True)
    return ok


def check(args):
    import mlx.core as mx
    rng = np.random.default_rng(args.seed)
    batches = [int(b) for b in args.batches.split(",")]
    results, ok = [], True
    real_cache = {}
    with tempfile.TemporaryDirectory(prefix="mlx-ops-", dir=REPO / "target") as tmp:
        workdir = pathlib.Path(tmp)
        if args.ops in ("all", "attention"):
            ok = check_attention(args, rng, workdir, results) and ok
            # c2-prefill-mlx: the rescheduled attention (mk_mlx2_attention*) on the same cases.
            if "mlx2" in args.variants.split(","):
                ok = check_attention(args, rng, workdir, results, attention2=True) and ok
        for name, k, n, base in (LINEARS if args.ops in ("all", "linear") else []):
            if n > args.max_out:
                continue
            for source in args.sources.split(","):
                if source == "real":
                    if base not in real_cache:
                        real_cache[base] = real_matrix(base)
                    words, scales, biases = real_cache[base]
                else:
                    words, scales, biases = random_matrix(rng, n, k)
                for m in batches:
                    if n * m > args.max_elems:
                        continue
                    x = rng.standard_normal((m, k)).astype(np.float32)
                    if source == "real":
                        x *= 0.5  # activations of a normed residual stream are O(1)
                    if m < vector_limit(k, n):
                        results.append({"op": "quantized_matmul", "linear": name, "k": k, "n": n, "m": m, "source": source,
                                        "ported": False, "note": "below MLX's vector limit %d: stock MLX runs qmv/qmv_wide "
                                        "(not ported; the engine keeps its exact kernel)" % vector_limit(k, n)})
                        print("%-18s %5d->%-6d M %5d %-6s not ported (MLX qmv below vector limit %d)" % (name, k, n, m, source, vector_limit(k, n)))
                        continue
                    want = mlx_qmm(x, words, scales, biases)
                    for variant in args.variants.split(","):
                        got, info = engine_qmm(args.binary, x, words, scales, biases, workdir, variant)
                        diff = got.view(np.uint32) != want.view(np.uint32)
                        row = {"op": "quantized_matmul", "ported": True, "linear": name, "k": k, "n": n, "m": m, "source": source,
                               "variant": variant, "kernel": info.get("kernels"), "split_k": info.get("split_k"),
                               "words": int(diff.size), "differing_words": int(diff.sum()),
                               "max_abs": float(np.abs(got.astype(np.float64) - want).max()),
                               "bit_identical": bool(not diff.any())}
                        ok = ok and row["bit_identical"]
                        results.append(row)
                        print("%-18s %5d->%-6d M %5d %-6s %-8s split_k %-2s %s%s" % (
                            name, k, n, m, source, variant, row["split_k"],
                            "bit-identical" if row["bit_identical"] else "DIFFERS",
                            "" if row["bit_identical"] else " (%d of %d words, max abs %.3g)" % (row["differing_words"], row["words"], row["max_abs"])),
                              flush=True)
        # Scale/bias dtypes other than the checkpoint's bf16 (one kernel each): float16 and
        # float32 storage of the same values must give the same bits.
        for name, k, n, base in (LINEARS if args.ops in ("all", "linear") and args.sources != "real" else []):
            if name not in ("ffn.w2", "q/o/out_proj", "gate_proj"):
                continue
            words, scales, biases = random_matrix(rng, n, k)
            for m in (64, 1000):
                x = rng.standard_normal((m, k)).astype(np.float32)
                want = mlx_qmm(x, words, scales, biases)
                for sdtype, dname in ((1, "f16"), (0, "f32")):
                    got, info = engine_qmm(args.binary, x, words, scales, biases, workdir, "mlx", sdtype=sdtype)
                    diff = got.view(np.uint32) != want.view(np.uint32)
                    row = {"op": "quantized_matmul", "ported": True, "linear": name, "k": k, "n": n, "m": m, "source": "random",
                           "variant": "mlx", "scale_dtype": dname, "kernel": info.get("kernels"), "split_k": info.get("split_k"),
                           "words": int(diff.size), "differing_words": int(diff.sum()), "bit_identical": bool(not diff.any())}
                    ok = ok and row["bit_identical"]
                    results.append(row)
                    print("%-18s %5d->%-6d M %5d random scales %-4s split_k %-2s %s" % (
                        name, k, n, m, dname, row["split_k"], "bit-identical" if row["bit_identical"] else "DIFFERS"), flush=True)
        # c2-prefill-mlx: the fused GLU kernel, and the mlx2 kernels' float16 / float32
        # scale storage, when the mlx2 variant is checked.
        if "mlx2" in args.variants.split(",") and args.ops in ("all", "linear"):
            ok = check_glu(args, rng, workdir, results, batches, real_cache) and ok
            if args.sources != "real":
                for name, k, n, base in LINEARS:
                    if name not in ("ffn.w2", "q/o/out_proj", "k/v_proj"):
                        continue
                    words, scales, biases = random_matrix(rng, n, k)
                    x = rng.standard_normal((1000, k)).astype(np.float32)
                    want = mlx_qmm(x, words, scales, biases)
                    for sdtype, dname in ((1, "f16"), (0, "f32")):
                        got, info = engine_qmm(args.binary, x, words, scales, biases, workdir, "mlx2", sdtype=sdtype)
                        diff = got.view(np.uint32) != want.view(np.uint32)
                        row = {"op": "quantized_matmul", "ported": True, "linear": name, "k": k, "n": n, "m": 1000, "source": "random",
                               "variant": "mlx2", "scale_dtype": dname, "kernel": info.get("kernels"), "split_k": info.get("split_k"),
                               "words": int(diff.size), "differing_words": int(diff.sum()), "bit_identical": bool(not diff.any())}
                        ok = ok and row["bit_identical"]
                        results.append(row)
                        print("%-18s %5d->%-6d M %5d random scales %-4s mlx2 split_k %-2s %s" % (
                            name, k, n, 1000, dname, row["split_k"], "bit-identical" if row["bit_identical"] else "DIFFERS"), flush=True)
    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "binary": str(args.binary), "mlx": mx.__version__,
              "device": mx.device_info().get("architecture") if hasattr(mx, "device_info") else None,
              "all_bit_identical": ok, "cases": results}
    out = args.out or REPO / "target/nano-mlx-ops.json"
    out.write_text(json.dumps(report, indent=1))
    print("ALL BIT-IDENTICAL" if ok else "DIFFERENCES FOUND", "(%d cases) -> %s" % (len(results), out))
    return 0 if ok else 1


# ---- c2-prefill-mlx (prefill, mlx-arith tier): the fused GLU kernel (mk_mlx2_glu) ----

def split_k(m, n, k, group=GROUP):
    """MLX qmm_splitk's partitions (exec.mlxSplitK); 1 = unsplit affine_qmm_t."""
    split = max(1, 512 // (((n + 31) // 32) * ((m + 31) // 32)))
    align = max(group, 32)
    split = min(split, k // align)
    while split > 1 and k % (split * align) != 0:
        split -= 1
    return max(split, 1)


def engine_glu(binary, x, gate, up, workdir, repeats=0):
    """`nano mlx-op glu`: mk_mlx2_glu and the sequence it replaces (port qmm twice, mk_binary).
    Returns the gate and up products (to compare with stock MLX) and the JSON line, whose
    fused_equals_sequence says whether the fused output equals the sequence's bit for bit."""
    m, k = x.shape
    n = gate[0].shape[0]
    blob = bytearray(b"\0\0")
    for words, _, _ in (gate, up):
        blob += words.astype("<u4").tobytes()
    for _, s, b in (gate, up):
        for a in (s, b):
            h = (a.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
            assert np.array_equal((h.astype(np.uint32) << 16).view(np.float32), a), "scales/biases must be bf16"
            blob += h.tobytes()
    (workdir / "glu.bin").write_bytes(bytes(blob))
    x.astype("<f4").tofile(workdir / "x.f32")
    out = workdir / "glu.f32"
    cmd = [str(binary), "mlx-op", "glu", "--m", str(m), "--k", str(k), "--n", str(n), "--group", str(GROUP),
           "--weights", str(workdir / "glu.bin"), "--x", str(workdir / "x.f32"), "--out", str(out), "--repeats", str(repeats)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit("nano mlx-op failed (%s):\n%s" % (" ".join(cmd), proc.stderr[-3000:]))
    res = [np.fromfile(str(out) + suffix, dtype="<f4").reshape(m, n) for suffix in (".gate", ".up")]
    for suffix in ("", ".gate", ".up"):
        pathlib.Path(str(out) + suffix).unlink()
    return res[0], res[1], json.loads(proc.stdout.strip().splitlines()[-1])


def check_glu(args, rng, workdir, results, batches, real_cache):
    """Fused GLU cases: the gate and up products bit-identical to stock MLX's, and the fused
    output bit-identical to the port's sequence (mk_binary on those same products)."""
    ok = True
    k, n = 2048, 7680
    for source in args.sources.split(","):
        if source == "real":
            for base in ("layers.0.feed_forward.w1", "layers.0.feed_forward.w3"):
                if base not in real_cache:
                    real_cache[base] = real_matrix(base)
            gate, up = real_cache["layers.0.feed_forward.w1"], real_cache["layers.0.feed_forward.w3"]
        else:
            gate, up = random_matrix(rng, n, k), random_matrix(rng, n, k)
        for m in batches:
            if n * m > args.max_elems or m < vector_limit(k, n) or split_k(m, n, k) != 1:
                continue
            x = rng.standard_normal((m, k)).astype(np.float32) * (0.5 if source == "real" else 1.0)
            g, u, info = engine_glu(args.binary, x, gate, up, workdir)
            same = [bool(np.array_equal(got.view(np.uint32), mlx_qmm(x, *wsb).view(np.uint32))) for got, wsb in ((g, gate), (u, up))]
            row = {"op": "glu (quantized_matmul x2 + silu product)", "ported": True, "k": k, "n": n, "m": m, "source": source,
                   "variant": "mlx2", "kernel": info["kernels"], "gate_bit_identical": same[0], "up_bit_identical": same[1],
                   "fused_equals_sequence": info["fused_equals_sequence"],
                   "bit_identical": same[0] and same[1] and info["fused_equals_sequence"]}
            ok = ok and row["bit_identical"]
            results.append(row)
            print("%-18s %5d->%-6d M %5d %-6s mlx2     gate %s, up %s, fused GLU %s" % (
                "ffn glu (w1, w3)", k, n, m, source, "bit-identical" if same[0] else "DIFFERS",
                "bit-identical" if same[1] else "DIFFERS", "equals the sequence" if info["fused_equals_sequence"] else "DIFFERS"), flush=True)
    return ok


# ---------------------------------------------------------------- timing

def chain_time(mx, step, x0, reps):
    """Seconds per call of `step`, run `reps` times as a dependent chain in one evaluation.
    MLX encodes independent operations without barriers between them, so each call's
    input is made to depend on the previous output (x0 + 0 * y[0, 0]); the cost of that
    dependency (two tiny elementwise kernels plus one [M, K] add) is measured the same
    way with an identity step and subtracted."""
    def run(fn):
        y = fn(x0)
        mx.eval(y)
        best = []
        for _ in range(3):
            mx.synchronize()
            t0 = time.perf_counter()
            x = x0
            outs = []
            for _ in range(reps):
                y = fn(x)
                outs.append(y)
                x = x0 + mx.reshape(y, (-1,))[0] * 0
            mx.eval(outs)
            mx.synchronize()
            best.append((time.perf_counter() - t0) / reps)
        return min(best)
    return run(step), run(lambda x: x[:1, :1])


def time_ops(args):
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_memory_limit(48 * 1024 ** 3)
    t = mx.load(str(CHECKPOINT / "model.safetensors"))
    batches = [int(b) for b in args.batches.split(",")]
    rows = []

    def record(op, m, sec, dep, note=""):
        rows.append({"op": op, "m": m, "us_per_call": 1e6 * max(sec - dep, 0), "us_raw": 1e6 * sec, "us_dependency": 1e6 * dep, "note": note})
        print("%-34s M %5d %10.1f us  (raw %.1f, dependency %.1f) %s" % (op, m, 1e6 * max(sec - dep, 0), 1e6 * sec, 1e6 * dep, note), flush=True)

    for name, k, n, base in LINEARS:
        w = t[PREFIX + base + ".weight"]
        s = t[PREFIX + base + ".scales"].astype(mx.float32)
        b = t[PREFIX + base + ".biases"].astype(mx.float32)
        mx.eval(w, s, b)
        for m in batches:
            if n * m > 2 ** 28:
                continue
            x = mx.random.normal((m, k)).astype(mx.float32)
            mx.eval(x)
            reps = max(3, min(50, int(2e9 // (m * n * k + 1)) + 3))
            sec, dep = chain_time(mx, lambda v: mx.quantized_matmul(v, w, s, b, transpose=True, group_size=GROUP, bits=4), x, reps)
            record("qmm %s %d->%d" % (name, k, n), m, sec, dep, "flops %.2f TF/s" % (2 * m * n * k / max(sec - dep, 1e-9) / 1e12))
    nw = t[PREFIX + "layers.0.operator_norm.weight"].astype(mx.float32)
    qn = t[PREFIX + "layers.2.self_attn.q_norm.weight"].astype(mx.float32)
    cw = t[PREFIX + "layers.0.conv.conv.weight"].astype(mx.float32)
    for m in batches:
        x = mx.random.normal((m, 2048)).astype(mx.float32)
        x7 = mx.random.normal((m, 7680)).astype(mx.float32)
        mx.eval(x, x7)
        reps = 20
        record("rms_norm 2048", m, *chain_time(mx, lambda v: mx.fast.rms_norm(v, nw, 1e-5), x, reps))
        record("rms_norm per head 16x128", m, *chain_time(mx, lambda v: mx.fast.rms_norm(v.reshape(m, 16, 128), qn, 1e-5).reshape(m, 2048), x, reps))
        record("add 2048", m, *chain_time(mx, lambda v: v + x, x, reps))
        record("silu(a)*b 7680", m, *chain_time(mx, lambda v: nn.silu(v) * x7, x7, reps))
        g = mx.random.normal((m, 16)).astype(mx.float32)
        record("attn out * sigmoid(gate)", m, *chain_time(mx, lambda v: (v.reshape(m, 16, 128) * mx.sigmoid(g)[..., None]).reshape(m, 2048), x, reps))
        x3 = mx.random.normal((m, 6144)).astype(mx.float32)

        def conv_step(v):
            bb, cc, xx = mx.split(v.reshape(1, m, 6144), 3, axis=-1)
            bx = mx.concatenate([mx.zeros((1, 6, 2048)), bb * xx], axis=1)
            return (cc * mx.conv1d(bx, cw, groups=2048)).reshape(m, 2048)
        record("short conv (B*x, conv1d, C*)", m, *chain_time(mx, conv_step, x3, reps))
        inv = 1e6 ** (-mx.arange(64, dtype=mx.float32) * 2 / 128)

        def rope_step(v):
            q = v.reshape(1, m, 16, 128).transpose(0, 2, 1, 3)
            ang = (mx.arange(m)[:, None].astype(mx.float32) * inv)
            ang = mx.concatenate([ang, ang], axis=-1)
            a, b2 = mx.split(q, 2, axis=-1)
            return (q * mx.cos(ang) + mx.concatenate([-b2, a], axis=-1) * mx.sin(ang)).transpose(0, 2, 1, 3).reshape(m, 2048)
        record("rope (stock python form) 16x128", m, *chain_time(mx, rope_step, x, reps))
        for window, label in ((None, "full causal"), (256, "sliding 256 (array mask)")):
            k_ = mx.random.normal((1, 4, m, 128)).astype(mx.float32)
            v_ = mx.random.normal((1, 4, m, 128)).astype(mx.float32)
            mx.eval(k_, v_)
            if window and m > window:
                r = mx.arange(m)
                mask = (r[:, None] >= r[None]) & (r[:, None] < r[None] + window)
            else:
                mask = "causal"

            def sdpa_step(v):
                q = v.reshape(1, m, 16, 128).transpose(0, 2, 1, 3)
                return mx.fast.scaled_dot_product_attention(q, k_, v_, scale=128 ** -0.5, mask=mask).transpose(0, 2, 1, 3).reshape(m, 2048)
            record("sdpa %s (q=k=M)" % label, m, *chain_time(mx, sdpa_step, x, max(3, reps // 4)))
    report = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mlx": mx.__version__, "machine": platform.platform(),
              "device": mx.device_info() if hasattr(mx, "device_info") else None, "rows": rows}
    if args.model:
        report["chunks"] = model_chunks(args.model_batches)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


def model_chunks(batches):
    """Whole-model stock prefill: one chunk of M tokens from an empty cache, evaluating only
    the cache state (as generate() does), float32. Median of 5 after a warm-up."""
    import mlx.core as mx
    from mlx_vlm.utils import load_model
    model = load_model(CHECKPOINT)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    lm = model.language_model
    out = []
    for m in [int(b) for b in batches.split(",")]:
        ids = mx.array([[(i * 7919 + 13) % 120000 for i in range(m)]], dtype=mx.int32)
        times = []
        for rep in range(6):
            cache = lm.make_cache()
            mx.synchronize()
            t0 = time.perf_counter()
            lm(ids, cache=cache)
            mx.eval([c.state for c in cache])
            mx.synchronize()
            if rep:
                times.append(time.perf_counter() - t0)
        sec = statistics.median(times)
        out.append({"m": m, "seconds": sec, "tokens_per_second": m / sec})
        print("whole model, one chunk of %5d tokens from empty: %.2f ms (%.0f tok/s)" % (m, 1e3 * sec, m / sec), flush=True)
    return out


def speed(args):
    """GPU time per call, the engine's port (nano mlx-op --repeats) against stock MLX
    (chain_time), on the same random weights, at each linear shape and batch. The engine's
    output is also checked against MLX's bit for bit. Run under the lock."""
    import mlx.core as mx
    rng = np.random.default_rng(args.seed)
    rows = []
    with tempfile.TemporaryDirectory(prefix="mlx-speed-", dir=REPO / "target") as tmp:
        workdir = pathlib.Path(tmp)
        for name, k, n, base in (LINEARS if args.ops != "attention" else []):
            if n > args.max_out:
                continue
            words, scales, biases = random_matrix(rng, n, k)
            for m in [int(b) for b in args.batches.split(",")]:
                x = rng.standard_normal((m, k)).astype(np.float32)
                want = mlx_qmm(x, words, scales, biases)
                got, info = engine_qmm(args.binary, x, words, scales, biases, workdir, args.variant, args.repeats)
                same = bool(np.array_equal(got.view(np.uint32), want.view(np.uint32)))
                W, S, B, X = mx.array(words), mx.array(scales), mx.array(biases), mx.array(x)
                sec, dep = chain_time(mx, lambda v: mx.quantized_matmul(v, W, S, B, transpose=True, group_size=GROUP, bits=4), X, args.repeats)
                mlx_us = 1e6 * max(sec - dep, 0)
                rows.append({"linear": name, "k": k, "n": n, "m": m, "engine_us": info["us_per_op"], "mlx_us": mlx_us,
                             "split_k": info["split_k"], "bit_identical": same})
                print("%-18s %5d->%-6d M %5d split %-2d engine %8.1f us  MLX %8.1f us  ratio %.2f  %s" % (
                    name, k, n, m, info["split_k"], info["us_per_op"], mlx_us, info["us_per_op"] / max(mlx_us, 1e-9),
                    "bit-identical" if same else "DIFFERS"), flush=True)
        # c2-prefill-mlx: fused GLU against stock's silu(w1 x) * (w3 x) (two qmm, compiled silu, multiply).
        if args.variant == "mlx2" and args.ops != "attention":
            import mlx.nn as nn
            gate, up = random_matrix(rng, 7680, 2048), random_matrix(rng, 7680, 2048)
            G, U = [tuple(mx.array(a) for a in wsb) for wsb in (gate, up)]
            for m in [int(b) for b in args.batches.split(",")]:
                if m < vector_limit(2048, 7680) or split_k(m, 7680, 2048) != 1:
                    continue
                x = rng.standard_normal((m, 2048)).astype(np.float32)
                _, _, info = engine_glu(args.binary, x, gate, up, workdir, args.repeats)
                q = lambda v, wsb: mx.quantized_matmul(v, *wsb, transpose=True, group_size=GROUP, bits=4)
                sec, dep = chain_time(mx, lambda v: nn.silu(q(v, G)) * q(v, U), mx.array(x), args.repeats)
                mlx_us = 1e6 * max(sec - dep, 0)
                rows.append({"linear": "ffn glu (w1, w3, silu product)", "k": 2048, "n": 7680, "m": m, "engine_us": info["us_per_op"],
                             "mlx_us": mlx_us, "split_k": 1, "bit_identical": info["fused_equals_sequence"]})
                print("%-18s %5d->%-6d M %5d split 1  engine %8.1f us  MLX %8.1f us  ratio %.2f  %s" % (
                    "ffn glu", 2048, 7680, m, info["us_per_op"], mlx_us, info["us_per_op"] / max(mlx_us, 1e-9),
                    "fused equals sequence" if info["fused_equals_sequence"] else "DIFFERS"), flush=True)
        for t, kl, window in ((1024, 1024, 0), (1024, 1024, 256), (2048, 2048, 0), (2048, 4096, 0), (2048, 2048, 256), (2048, 4096, 256)):
            if args.ops == "linear":
                break
            q = rng.standard_normal((t, 16 * 128)).astype(np.float32)
            k = rng.standard_normal((kl, 4 * 128)).astype(np.float32)
            v = rng.standard_normal((kl, 4 * 128)).astype(np.float32)
            gate = rng.standard_normal((t, 16)).astype(np.float32)
            got, info = engine_attention(args.binary, q, k, v, gate, t, kl, window, 0, workdir, args.repeats,
                                         attention2=args.variant == "mlx2")
            origin = rotating_origin(t, kl, window) if window else 0
            same = bool(np.array_equal(got.view(np.uint32), mlx_sdpa(q, k, v, t, kl, window, origin=origin).view(np.uint32)))
            K = mx.array(k[origin:].reshape(kl - origin, 4, 128)).transpose(1, 0, 2)[None]
            V = mx.array(v[origin:].reshape(kl - origin, 4, 128)).transpose(1, 0, 2)[None]
            if window:
                qpos, kpos = np.arange(kl - t, kl)[:, None], np.arange(origin, kl)[None]
                mask = mx.array((qpos >= kpos) & (qpos < kpos + window))
            else:
                mask = "causal"
            X = mx.array(q)
            step = lambda x: mx.fast.scaled_dot_product_attention(
                x.reshape(1, t, 16, 128).transpose(0, 2, 1, 3), K, V, scale=128 ** -0.5, mask=mask).transpose(0, 2, 1, 3).reshape(t, 2048)
            sec, dep = chain_time(mx, step, X, max(3, args.repeats // 4))
            mlx_us = 1e6 * max(sec - dep, 0)
            rows.append({"op": "sdpa", "t": t, "kl": kl, "window": window, "engine_us": info["us_per_op"], "mlx_us": mlx_us, "bit_identical": same})
            print("sdpa T %5d KL %5d window %3d          engine %8.1f us  MLX %8.1f us  ratio %.2f  %s" % (
                t, kl, window, info["us_per_op"], mlx_us, info["us_per_op"] / max(mlx_us, 1e-9), "bit-identical" if same else "DIFFERS"), flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("binary", type=pathlib.Path)
    c.add_argument("--batches", default="1,2,7,31,32,33,63,64,65,100,255,256,511,1000,1023,1024,2047,2048")
    c.add_argument("--sources", default="random,real")
    c.add_argument("--variants", default="mlx")
    c.add_argument("--max-out", type=int, default=7680, help="skip linears wider than this (the head)")
    c.add_argument("--max-elems", type=int, default=2 ** 24)
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--ops", default="all", choices=("all", "linear", "attention"))
    c.add_argument("--out", type=pathlib.Path)
    t = sub.add_parser("time")
    t.add_argument("--batches", default="64,256,1024,2048,4096")
    t.add_argument("--model", action="store_true", help="also time whole-model chunks")
    t.add_argument("--model-batches", default="64,256,1024,2048")
    t.add_argument("--out", type=pathlib.Path)
    sp = sub.add_parser("speed")
    sp.add_argument("binary", type=pathlib.Path)
    sp.add_argument("--batches", default="64,1024,2048")
    sp.add_argument("--variant", default="mlx")
    sp.add_argument("--repeats", type=int, default=20)
    sp.add_argument("--max-out", type=int, default=7680)
    sp.add_argument("--seed", type=int, default=11)
    sp.add_argument("--ops", default="all", choices=("all", "linear", "attention"))
    sp.add_argument("--out", type=pathlib.Path)
    args = p.parse_args()
    sys.exit({"check": check, "time": time_ops, "speed": speed}[args.command](args))


if __name__ == "__main__":
    main()
