#!/usr/bin/env python3
"""Compare `nano eval` with LFM3 goldens: per-layer residual streams, logits at the
stored positions, top-1 at every stored position and the greedy continuation.

Gate (fixed before measurement, not tuned): for each case the engine's largest
absolute error, logits and hidden separately, must not exceed
    2 * float32_noise_floor + reference_error
where float32_noise_floor is the error of a float32 NumPy run of the same model
against float64, and reference_error is the golden producer's own distance from
float64 (stock MLX f32 vs NumPy f64 for the real goldens; 0 for the tiny NumPy
float64 goldens, whose floor this script measures with tools/lfm3_reference.py).
Top-1 must agree at every stored position and the greedy tokens must be equal.

With several --feeds, every feed must also be bit-identical to the first one
(all-vocabulary logits at every stored position, residual streams, greedy):
batched ingestion is exact or it fails.

--f64 (real cases) also runs the independent NumPy float64 model
(tools/lfm3_reference.py, whole sequence with masks; about 14 GB of RAM) and
reports the engine's distance from it per layer and at the logits, as a
multiple of the recorded float32 noise floor (the error a float32 NumPy run
makes against the same float64 model). Informational: it does not change the gate.

  python3 tools/nano_verify.py zig-out/bin/nano --tiny target/lfm3-tiny \\
      --golden $NANO_WORK/golden/lfm3 --cases short,window --feeds 1,7,64
"""
import os
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def tiny_floor(checkpoint, golden):
    """float32 NumPy vs the float64 golden: the floor an exact float32 engine sits at."""
    from lfm3_reference import Checkpoint, NumpyLFM3
    tokens = json.loads((golden / "tokens.json").read_text())["tokens"]
    positions = json.loads((golden / "positions.json").read_text())["positions"]
    model = NumpyLFM3(Checkpoint(str(checkpoint)), dtype=np.float32).load()
    hidden, logits = model.sequence(tokens, model.new_state(), logits_at=positions)
    g_logits = np.load(golden / "logits.npy").astype(np.float64)
    g_hidden = np.load(golden / "hidden.npy").astype(np.float64)
    return (float(np.abs(logits.astype(np.float64) - g_logits).max()),
            float(np.abs(hidden.astype(np.float64) - g_hidden).max()))


def run_eval(binary, checkpoint, golden, feed, workdir, extra, want_hidden, runner=None):
    """runner (prefill-mlx): optional callable(cmd) -> stdout, e.g. a guarded run under the lock."""
    tokens = json.loads((golden / "tokens.json").read_text())["tokens"]
    greedy_n = len(json.loads((golden / "greedy.json").read_text())["tokens"])
    logits_path = workdir / ("logits-%d.f32" % feed)
    hidden_path = workdir / ("hidden-%d.f32" % feed)
    cmd = [str(binary), "eval", str(checkpoint), "--tokens-file", str(golden / "tokens.json"),
           "--positions-file", str(golden / "positions.json"), "--logits-out", str(logits_path),
           "--greedy", str(max(greedy_n, 64)), "--feed", str(feed), "--knob", "batch=%d" % max(feed, 1),
           "--context", str(len(tokens) + 64)] + list(extra)
    if want_hidden:
        cmd += ["--hidden-out", str(hidden_path)]
    if runner is not None:
        stdout = runner(cmd)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            raise SystemExit("nano eval failed (%s):\n%s" % (" ".join(cmd), proc.stderr[-4000:]))
        stdout = proc.stdout
    summary = json.loads(stdout.strip().splitlines()[-1])
    vocab = summary["vocab"]
    logits = np.fromfile(logits_path, dtype="<f4").reshape(len(summary["positions"]), vocab)
    hidden = None
    if want_hidden:
        hidden = np.fromfile(hidden_path, dtype="<f4").reshape(summary["traces"], summary["tokens"], summary["trace_width"])
        hidden_path.unlink()
    logits_path.unlink()
    return summary, logits, hidden


def f64_reference(checkpoint, golden, want_hidden):
    from lfm3_reference import Checkpoint, NumpyLFM3
    tokens = json.loads((golden / "tokens.json").read_text())["tokens"]
    positions = json.loads((golden / "positions.json").read_text())["positions"]
    model = NumpyLFM3(Checkpoint(str(checkpoint))).load()
    return model.sequence(tokens, model.new_state(), logits_at=positions, keep_hidden=want_hidden)


def f64_report(logits, hidden, ref, floor):
    r_hidden, r_logits = ref
    d = np.abs(logits.astype(np.float64) - r_logits)
    out = {"logits_max_abs": float(d.max()), "logits_max_rel": float(d.max() / np.abs(r_logits).max()),
           "logits_over_floor": float(d.max() / floor[0])}
    if hidden is not None and r_hidden is not None:
        layers = []
        for i in range(r_hidden.shape[0]):
            e = np.abs(hidden[i].astype(np.float64) - r_hidden[i])
            layers.append({"index": i, "max_abs": float(e.max()), "max_rel": float(e.max() / max(np.abs(r_hidden[i]).max(), 1e-30))})
        out["hidden"] = layers
        out["hidden_max_abs"] = max(l["max_abs"] for l in layers)
        out["hidden_over_floor"] = out["hidden_max_abs"] / floor[1]
    return out


def compare(name, checkpoint, golden, binary, feeds, extra, floor, reference_error, workdir, f64=None):
    g_logits = np.load(golden / "logits.npy")
    g_hidden = np.load(golden / "hidden.npy") if (golden / "hidden.npy").exists() else None
    g_greedy = json.loads((golden / "greedy.json").read_text())["tokens"]
    positions = json.loads((golden / "positions.json").read_text())["positions"]
    results = []
    first = None
    ok = True
    for feed in feeds:
        summary, logits, hidden = run_eval(binary, checkpoint, golden, feed, workdir, extra, g_hidden is not None)
        diff = np.abs(logits.astype(np.float64) - g_logits.astype(np.float64))
        scale = np.abs(g_logits).max()
        row = {"case": name, "feed": feed, "forwards": summary["forwards"],
               "dispatches_per_forward": summary["dispatches"] / summary["forwards"],
               "logits_max_abs": float(diff.max()), "logits_max_rel": float(diff.max() / scale),
               "top1_agree": int(sum(int(a) == int(b) for a, b in zip(summary["top1"], g_logits.argmax(1)))),
               "positions": len(positions), "greedy_equal": summary["greedy"] == g_greedy,
               "greedy_len": len(summary["greedy"]), "golden_greedy_len": len(g_greedy)}
        if g_hidden is not None:
            layers = []
            for i in range(g_hidden.shape[0]):
                d = np.abs(hidden[i].astype(np.float64) - g_hidden[i].astype(np.float64))
                layers.append({"index": i, "max_abs": float(d.max()), "max_rel": float(d.max() / max(np.abs(g_hidden[i]).max(), 1e-30))})
            row["hidden"] = layers
            row["hidden_max_abs"] = max(l["max_abs"] for l in layers)
        gate_logits = 2 * floor[0] + reference_error[0]
        gate_hidden = 2 * floor[1] + reference_error[1]
        row["gate_logits"] = gate_logits
        row["gate_hidden"] = gate_hidden
        passed = (row["logits_max_abs"] <= gate_logits and row["top1_agree"] == len(positions) and row["greedy_equal"]
                  and (g_hidden is None or row["hidden_max_abs"] <= gate_hidden))
        if first is None:
            first = (logits, hidden, summary["greedy"], summary["top1"])
            if f64 is not None:
                row["vs_numpy_f64"] = rep = f64_report(logits, hidden, f64, floor)
                print("%-14s feed %4d vs NumPy f64: logits %.3g (%.2f x floor)%s" % (
                    name, feed, rep["logits_max_abs"], rep["logits_over_floor"],
                    "" if "hidden_max_abs" not in rep else ", hidden %.3g (%.2f x floor)" % (rep["hidden_max_abs"], rep["hidden_over_floor"])), flush=True)
        else:
            same = (np.array_equal(logits.view(np.uint32), first[0].view(np.uint32)) and summary["greedy"] == first[2]
                    and summary["top1"] == first[3]
                    and (hidden is None or np.array_equal(hidden.view(np.uint32), first[1].view(np.uint32))))
            row["bit_identical_to_feed_%d" % feeds[0]] = bool(same)
            passed = passed and same
        row["pass"] = bool(passed)
        ok = ok and passed
        results.append(row)
        extra_s = "" if g_hidden is None else " hidden %.3g (gate %.3g)" % (row["hidden_max_abs"], gate_hidden)
        bit = "" if len(results) == 1 else (" bitwise=%s" % row["bit_identical_to_feed_%d" % feeds[0]])
        print("%-14s feed %4d: logits %.3g (gate %.3g, rel %.3g)%s top1 %d/%d greedy %s (%d)%s %s" % (
            name, feed, row["logits_max_abs"], gate_logits, row["logits_max_rel"], extra_s, row["top1_agree"],
            len(positions), "equal" if row["greedy_equal"] else "DIFFER", row["greedy_len"], bit,
            "PASS" if passed else "FAIL"), flush=True)
    return ok, results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("binary", type=pathlib.Path)
    p.add_argument("--tiny", type=pathlib.Path, help="directory written by tools/lfm3_tiny.py")
    p.add_argument("--golden", type=pathlib.Path, help="real-checkpoint goldens (golden/lfm3)")
    p.add_argument("--checkpoint", type=pathlib.Path, default=pathlib.Path(os.environ.get("LFM3_CHECKPOINT", pathlib.Path(__file__).resolve().parent.parent / "target/lfm3-1.7b-mlx-4bit")))
    p.add_argument("--cases", default="short,window,long")
    p.add_argument("--feeds", default="1")
    p.add_argument("--knobs", type=pathlib.Path, help="JSON knob configuration passed to nano")
    p.add_argument("--knob", action="append", default=[], help="name=value, repeatable")
    p.add_argument("--json", type=pathlib.Path, help="write all results here")
    p.add_argument("--f64", action="store_true", help="also compare real cases with the NumPy float64 model")
    args = p.parse_args()
    feeds = [int(f) for f in args.feeds.split(",")]
    extra = []
    if args.knobs:
        extra += ["--knobs", str(args.knobs)]
    for k in args.knob:
        extra += ["--knob", k]
    all_ok, everything = True, []
    target = pathlib.Path("target")
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nano-verify-", dir=target) as tmp:
        workdir = pathlib.Path(tmp)
        if args.tiny:
            for case in sorted(d for d in args.tiny.iterdir() if (d / "golden").is_dir()):
                floor = tiny_floor(case, case / "golden")
                ok, rows = compare(case.name, case, case / "golden", args.binary.resolve(), feeds, extra, floor, (0.0, 0.0), workdir)
                for r in rows:
                    r["float32_noise_floor"] = floor
                all_ok &= ok
                everything += rows
        if args.golden:
            for name in args.cases.split(","):
                g = args.golden / name
                meta = json.loads((g / "meta.json").read_text())
                floor = (meta["float32_noise_floor_logits"], meta["float32_noise_floor_hidden"])
                ref = (meta["max_abs_diff_mlx_f32_vs_numpy_logits"], meta["max_abs_diff_mlx_f32_vs_numpy_hidden"])
                f64 = f64_reference(args.checkpoint, g, (g / "hidden.npy").exists()) if args.f64 else None
                ok, rows = compare(name, args.checkpoint, g, args.binary.resolve(), feeds, extra, floor, ref, workdir, f64)
                all_ok &= ok
                everything += rows
    if args.json:
        args.json.write_text(json.dumps(everything, indent=1))
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
