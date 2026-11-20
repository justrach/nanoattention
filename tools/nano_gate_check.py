#!/usr/bin/env python3
"""decode climb 2: admission add-on for gated candidates (knob d2_gate).

`nano_qd.py trial` admits with --knob batch=FEED, and at feed 1 a batch of 1 leaves
the rings no spare row, so the engine decodes ungated there (exec.d2GateSafe).
This runs every admission case at feed 1 with batch 64, so every single-row
forward with logits goes through the gate, and compares the fingerprints
(all-vocabulary logits at the stored positions, top-1, greedy) with the pinned
reference. --spins 1 forces every gate to give up at once, so the stale-generation
recomputation runs too (the tiny cases print how many tokens were recomputed).
Real-checkpoint runs take the inference lock (nano_bench.guarded).

  python3 tools/nano_gate_check.py zig-out/bin/nano src/mega/candidates/split-attn-gated-nr.json --long [--spins 1]
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nano_qd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary", type=pathlib.Path)
    ap.add_argument("candidate", type=pathlib.Path, help="candidate or knobs JSON")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--spins", type=int, help="override d2_gate_spins")
    ap.add_argument("--tiny", type=pathlib.Path, default=nano_qd.REPO / "target/lfm3-tiny")
    args = ap.parse_args()
    k = json.loads(args.candidate.read_text())
    k = dict(k.get("knobs", k))
    k["batch"] = 64
    if args.spins is not None:
        k["d2_gate_spins"] = args.spins
    pin = nano_qd.load_pin()
    ok = True
    with tempfile.TemporaryDirectory(prefix="gate-check-", dir=nano_qd.REPO / "target") as tmp:
        tmp = pathlib.Path(tmp)
        knobs = tmp / "knobs.json"
        knobs.write_text(json.dumps(k))
        for name, ckpt, golden, _ in nano_qd.cases(args.tiny, args.long):
            tokens = json.loads((golden / "tokens.json").read_text())["tokens"]
            out = tmp / "logits.f32"
            cmd = [str(args.binary.resolve()), "eval", str(ckpt), "--tokens-file", str(golden / "tokens.json"),
                   "--positions-file", str(golden / "positions.json"), "--logits-out", str(out), "--greedy", "64",
                   "--feed", "1", "--context", str(len(tokens) + 64), "--knobs", str(knobs)]
            note = ""
            if name.startswith("tiny/"):
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode:
                    raise SystemExit("%s: %s" % (name, proc.stderr[-2000:]))
                stdout = proc.stdout
                note = " ".join(l for l in proc.stderr.splitlines() if "recomputed" in l)
            else:
                stdout = nano_qd.nano_bench.guarded(cmd, timeout=1800)
            summary = json.loads(stdout.strip().splitlines()[-1])
            got = {"logits_sha256": hashlib.sha256(np.fromfile(out, dtype="<u4").tobytes()).hexdigest(),
                   "greedy": summary["greedy"], "top1": summary["top1"]}
            same = nano_qd.same_fingerprint(got, pin["cases"][name])
            ok = ok and same
            print("%-20s feed 1, batch 64: %s %s" % (name, "bit-identical" if same else "DIFFERS", note), flush=True)
    print("ALL BIT-IDENTICAL" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
