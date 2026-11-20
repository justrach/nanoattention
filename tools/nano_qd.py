#!/usr/bin/env python3
"""Quality-diversity trials for the megakernel (docs/QD.md).

A candidate is a JSON file:
  {"name": "glu-norms-r4", "knobs": {...nano knobs...}, "family": "fused-mlp" (optional),
   "hypothesis": "what it should gain and why", "expected": {"decode": 5.0, "prefill": 0.0}}

The reference is pinned. `pin-reference` runs the reference executor (default
knobs: one dispatch per op, no fusion; fed one token at a time) of a verified
binary on every admission case, checks it against the goldens (logits within
2 x float32 noise floor + the golden producer's own error, top-1 and greedy
equal), and stores its fingerprints (sha256 of the all-vocabulary logits at the
stored positions, top-1, greedy) in reference.json beside the archive. Replacing
a pin needs --replace and --why (a documented decision); the old pin is kept in
reference-history.jsonl.

`trial` runs, in order:
  1. Admission (exactness). The binary's own reference executor must reproduce the
     pin (else the binary changed the reference arithmetic and nothing on it is
     admitted). Then for every tiny checkpoint (feeds 1, 7 and the candidate's
     batch) and the real goldens `short` and `window` (feeds 1 and the candidate's
     batch; `--long` adds `long`): all-vocabulary logits at the stored positions,
     top-1 and the greedy tokens must be bit-identical to the pin. A certificate
     command (`--certify-cmd`, e.g. the Lean checker; `{candidate}` is replaced by
     the candidate path) exiting 0 records `certified`, otherwise `empirical`. A
     failure is recorded as rejected, never timed.
  2. Quality (speed), under the inference lock: `nano bench` at prompt lengths 64,
     1,024 and 4,096 (256 teacher-forced decode tokens, warm-up plus three repeats,
     medians). For each objective (decode, prefill) and length the candidate's cell is
     (objective, length bucket, dispatches-per-token bin, family). An empty cell takes
     the candidate. An occupied one is contested by interleaving elite and challenger
     (A, B, A, B, ..., --rounds each); the challenger wins only if its median beats the
     elite's by more than max(1%, 2 x the elite's own spread between rounds).
  3. Record: one line appended to trials.jsonl; archive.json rewritten atomically
     (temporary file, rename) under an exclusive flock on archive.lock; elites.md
     regenerated from the archive.

  python3 tools/nano_qd.py trial zig-out/bin/nano cand.json --proposer megakernel --why "..."
  python3 tools/nano_qd.py pin-reference zig-out/bin/nano --why "first verified reference"
  python3 tools/nano_qd.py baseline     # record benchmarks/lfm3-mlx-baseline.json in the archive
  python3 tools/nano_qd.py render       # regenerate elites.md
  python3 tools/nano_qd.py parents      # occupied cells as JSON: where mutations start
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
WORK = pathlib.Path(os.environ.get("NANO_WORK", REPO / "target/nano-work"))
ARCHIVE = WORK / "qd/lfm3"
CHECKPOINT = pathlib.Path(os.environ.get("LFM3_CHECKPOINT", REPO / "target/lfm3-1.7b-mlx-4bit"))
GOLDEN = WORK / "golden/lfm3"
LENGTHS = (64, 1024, 4096)
sys.path.insert(0, str(HERE))
import nano_bench  # noqa: E402
from nano_verify import tiny_floor  # noqa: E402

PIN = ARCHIVE / "reference.json"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bucket(length):
    for limit in (256, 1024, 4096):
        if length <= limit:
            return "<=%d" % limit
    return ">4096"


def dispatch_bin(per_token):
    for limit in (64, 128, 256):
        if per_token <= limit:
            return "<=%d" % limit
    return ">256"


FUSIONS = ("fuse_projections", "fuse_glu", "fuse_norms", "fuse_conv", "fuse_gate")


def family(candidate):
    if candidate.get("family"):
        return candidate["family"]
    on = [k for k in FUSIONS if candidate["knobs"].get(k)]
    if not on:
        return "per-op"
    return "fused-layer" if len(on) == len(FUSIONS) else "fused-partial"


# ---- admission ------------------------------------------------------------------------

def eval_case(binary, checkpoint, golden, feed, knobs_path, workdir, keep_logits=False):
    """One `nano eval` of a golden case. Real-checkpoint runs take the inference lock
    (docs/QD.md: during the search an unlocked GPU job corrupts whoever is being timed)."""
    tokens = json.loads((golden / "tokens.json").read_text())["tokens"]
    # A feed longer than the prompt is the whole prompt in one forward; the engine
    # clamps its batch to the context, so ask for what it can take.
    feed = min(feed, len(tokens))
    out = workdir / "logits.f32"
    cmd = [str(binary), "eval", str(checkpoint), "--tokens-file", str(golden / "tokens.json"),
           "--positions-file", str(golden / "positions.json"), "--logits-out", str(out), "--greedy", "64",
           "--feed", str(feed), "--context", str(len(tokens) + 64)]
    if knobs_path:
        cmd += ["--knobs", str(knobs_path)]
    cmd += ["--knob", "batch=%d" % feed]
    if pathlib.Path(checkpoint).resolve() == CHECKPOINT.resolve():
        try:
            stdout = nano_bench.guarded(cmd, timeout=1800)
        except SystemExit as e:
            return None, str(e)[-2000:]
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            return None, proc.stderr[-2000:]
        stdout = proc.stdout
    summary = json.loads(stdout.strip().splitlines()[-1])
    logits = np.fromfile(out, dtype="<u4")
    out.unlink()
    result = {"logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(), "greedy": summary["greedy"],
              "top1": summary["top1"], "dispatches_per_forward": summary["dispatches"] / summary["forwards"],
              "feed": feed}
    if keep_logits:
        result["logits"] = logits.view("<f4").reshape(len(summary["positions"]), summary["vocab"])
    return result, None


FINGERPRINT = ("logits_sha256", "greedy", "top1")


def same_fingerprint(a, b):
    return all(a[k] == b[k] for k in FINGERPRINT)


def cases(tiny, long_case):
    """(name, checkpoint, golden, feeds) for every admission case."""
    out = []
    if tiny and tiny.exists():
        for d in sorted(x for x in tiny.iterdir() if (x / "golden").is_dir()):
            out.append(("tiny/" + d.name, d, d / "golden", (1, 7)))
    for name in ("short", "window") + (("long",) if long_case else ()):
        if (GOLDEN / name).is_dir():
            out.append((name, CHECKPOINT, GOLDEN / name, (1,)))
    return out


def reference(binary, case, workdir, cache_dir):
    name, checkpoint, golden, _ = case
    path = cache_dir / (name.replace("/", "_") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    result, err = eval_case(binary, checkpoint, golden, 1, None, workdir)
    if result is None:
        raise SystemExit("reference executor failed on %s: %s" % (name, err))
    path.write_text(json.dumps(result))
    return result


def load_pin():
    if not PIN.exists():
        raise SystemExit("no pinned reference (%s): run `nano_qd.py pin-reference` with a verified binary first" % PIN)
    return json.loads(PIN.read_text())


def admit(binary, candidate_path, candidate, tiny, long_case, certify_cmd, extra_feeds=()):
    binary_sha = sha256(binary)
    pin = load_pin()
    cache_dir = REPO / "target/qd-reference" / binary_sha[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch = int(candidate["knobs"].get("batch", 64))
    checks, ok, reason = [], True, None
    knobs_file = REPO / "target" / ("qd-knobs-%s.json" % candidate["name"])
    knobs_file.write_text(json.dumps(candidate["knobs"]))
    with tempfile.TemporaryDirectory(prefix="qd-admit-", dir=REPO / "target") as tmp:
        workdir = pathlib.Path(tmp)
        for case in cases(tiny, long_case):
            pinned = pin["cases"].get(case[0])
            if pinned is None:
                raise SystemExit("case %s is not in the pinned reference; re-pin with it" % case[0])
            ref = reference(binary, case, workdir, cache_dir)
            if not same_fingerprint(ref, pinned):
                ok, reason = False, "this binary's reference executor differs from the pinned reference on %s" % case[0]
                checks.append({"case": case[0], "feed": 1, "reference_matches_pin": False})
                print("  admission %-22s: REFERENCE CHANGED (binary's reference executor != pin)" % case[0], flush=True)
                break
            # extra_feeds: more batch boundaries on the real goldens.
            feeds = case[3] + (batch,) + (tuple(extra_feeds) if not case[0].startswith("tiny/") else ())
            for feed in sorted(set(feeds)):
                got, err = eval_case(binary, case[1], case[2], feed, knobs_file, workdir)
                same = got is not None and same_fingerprint(got, pinned)
                checks.append({"case": case[0], "feed": feed, "bit_identical": bool(same), "error": err,
                               "dispatches_per_forward": got and got["dispatches_per_forward"]})
                print("  admission %-22s feed %4d: %s" % (case[0], feed, "bit-identical" if same else "DIFFERS"), flush=True)
                ok = ok and same
                if not same:
                    reason = ("%s feed %d: eval failed, not compared" % (case[0], feed) if got is None
                              else "%s feed %d not bit-identical to the reference" % (case[0], feed))
                    break
            if not ok:
                break
    mode, certificate = "empirical", None
    if ok and certify_cmd:
        proc = subprocess.run(shlex.split(certify_cmd.replace("{candidate}", str(candidate_path))), capture_output=True, text=True, timeout=3600)
        certificate = {"command": certify_cmd, "returncode": proc.returncode, "output": (proc.stdout + proc.stderr)[-2000:]}
        if proc.returncode == 0:
            mode = "certified"
    return {"admitted": ok, "mode": mode, "certificate": certificate, "checks": checks, "binary_sha256": binary_sha,
            "reason": reason, "reference": "pinned reference %s (default knobs, one dispatch per op, feed 1)" % pin["binary_sha256"][:16]}


def golden_gate(case, got):
    """The reference executor against the golden: nano_verify's gate on logits, top-1, greedy."""
    name, checkpoint, golden, _ = case
    g_logits = np.load(golden / "logits.npy").astype(np.float64)
    g_greedy = json.loads((golden / "greedy.json").read_text())["tokens"]
    if name.startswith("tiny/"):
        floor, ref_err = tiny_floor(checkpoint, golden)[0], 0.0
    else:
        meta = json.loads((golden / "meta.json").read_text())
        floor, ref_err = meta["float32_noise_floor_logits"], meta["max_abs_diff_mlx_f32_vs_numpy_logits"]
    err = float(np.abs(got["logits"].astype(np.float64) - g_logits).max())
    gate = 2 * floor + ref_err
    top1 = int((np.asarray(got["top1"]) == g_logits.argmax(1)).sum())
    passed = err <= gate and top1 == len(g_logits) and got["greedy"] == g_greedy
    return {"logits_max_abs": err, "gate": gate, "top1": top1, "positions": len(g_logits),
            "greedy_equal": got["greedy"] == g_greedy, "pass": bool(passed)}


def pin_reference(args):
    binary = args.binary.resolve()
    if PIN.exists() and not args.replace:
        raise SystemExit("%s exists; replacing the reference needs --replace and --why" % PIN)
    if args.replace and not args.why:
        raise SystemExit("--replace needs --why (the documented decision)")
    pin = {"version": 1, "time": now(), "binary_sha256": sha256(binary), "worktree": str(REPO), "why": args.why,
           "reference": "default knobs (one dispatch per op, no fusion), fed one token at a time", "cases": {}}
    with tempfile.TemporaryDirectory(prefix="qd-pin-", dir=REPO / "target") as tmp:
        workdir = pathlib.Path(tmp)
        for case in cases(args.tiny, args.long):
            got, err = eval_case(binary, case[1], case[2], 1, None, workdir, keep_logits=True)
            if got is None:
                raise SystemExit("reference executor failed on %s: %s" % (case[0], err))
            gate = golden_gate(case, got)
            print("  pin %-22s logits %.3g (gate %.3g) top1 %d/%d greedy %s: %s" % (
                case[0], gate["logits_max_abs"], gate["gate"], gate["top1"], gate["positions"],
                "equal" if gate["greedy_equal"] else "DIFFER", "PASS" if gate["pass"] else "FAIL"), flush=True)
            if not gate["pass"]:
                raise SystemExit("reference executor fails the golden gate on %s: not pinned" % case[0])
            got.pop("logits")
            pin["cases"][case[0]] = dict(got, golden_check=gate)
    with Archive(ARCHIVE, readonly=True):
        if PIN.exists():
            with open(ARCHIVE / "reference-history.jsonl", "a") as f:
                f.write(json.dumps(json.loads(PIN.read_text())) + "\n")
        fd, tmp = tempfile.mkstemp(dir=ARCHIVE, prefix=".reference-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(pin, f, indent=1)
        os.chmod(tmp, 0o644)
        os.rename(tmp, PIN)
    print("pinned %d cases from %s" % (len(pin["cases"]), pin["binary_sha256"][:16]))
    return 0


def parents(args):
    with Archive(ARCHIVE, readonly=True) as a:
        cells = a.data["cells"]
    out = [{"cell": k, "candidate": c["candidate"], "tokens_per_second": c["tokens_per_second"], "trial": c["trial"]}
           for k, c in sorted(cells.items())]
    print(json.dumps(out, indent=1))
    return 0


# ---- archive --------------------------------------------------------------------------

class Archive:
    def __init__(self, root, readonly=False):
        self.root, self.readonly = root, readonly
        root.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(root / "archive.lock", "a+")

    def __enter__(self):
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        path = self.root / "archive.json"
        self.data = json.loads(path.read_text()) if path.exists() else {"version": 1, "cells": {}, "baselines": {}}
        return self

    def __exit__(self, *exc):
        if exc[0] is None and not self.readonly:
            fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".archive-", suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(tmp, 0o644)
            os.rename(tmp, self.root / "archive.json")
            render(self.root, self.data)
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        return False

    def append_trial(self, record):
        with open(self.root / "trials.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")


def render(root, data):
    lines = ["# LFM3 megakernel elites", "",
             "Generated by `tools/nano_qd.py` from `archive.json`; do not edit. One row per occupied cell "
             "(objective, prompt bucket, dispatches per token, family). Rates are medians of interleaved rounds "
             "(each round: one `nano bench` process, warm-up plus three repeats).", "",
             "| objective | bucket | dispatches/token | family | candidate | tokens/s | rounds | prompt | admission | trial |",
             "| --- | --- | --- | --- | --- | ---: | --- | ---: | --- | --- |"]
    for key in sorted(data["cells"]):
        c = data["cells"][key]
        lines.append("| %s | %s | %s | %s | `%s` | %.2f | %s | %d | %s | %s |" % (
            c["objective"], c["bucket"], c["dispatch_bin"], c["family"], c["candidate"]["name"], c["tokens_per_second"],
            " / ".join("%.1f" % r for r in c["rounds"]), c["prompt_tokens"], c["admission"], c["trial"]))
    if data.get("baselines"):
        lines += ["", "## Stock-MLX baseline (not candidates)", "", "| engine | prompt | prefill tok/s | decode tok/s |", "| --- | ---: | ---: | ---: |"]
        for engine, rows in sorted(data["baselines"].items()):
            for length, r in sorted(rows.items(), key=lambda kv: int(kv[0])):
                lines.append("| %s | %s | %.1f | %.2f |" % (engine, length, r["prefill_tokens_per_second"], r["decode_tokens_per_second"]))
    (root / "elites.md").write_text("\n".join(lines) + "\n")


# ---- quality --------------------------------------------------------------------------

def bench(binary, knobs, length, bench_tokens, workdir, tag, decode=256):
    path = workdir / ("knobs-%s.json" % tag)
    path.write_text(json.dumps(knobs))
    return nano_bench.bench_once(binary, CHECKPOINT, length, path, bench_tokens, decode=decode)


def rates(r):
    prefill_dispatch = r["dispatches_per_prefill_forward"] * r["prefill_forwards"] / r["prompt_tokens"]
    return {"decode": (r["decode_tokens_per_second"], r["dispatches_per_decode_token"]),
            "prefill": (r["prefill_tokens_per_second"], prefill_dispatch)}


def quality(binary, candidate, archive_root, rounds, bench_tokens, lengths, objectives=("decode", "prefill"), compare=()):
    fam = family(candidate)
    # Prefill-only trials skip the 256 decode tokens: decode cells are not contested then.
    decode = 256 if "decode" in objectives else 0
    with Archive(archive_root, readonly=True) as a:
        snapshot = json.loads(json.dumps(a.data))
    timings, contests = {}, []
    with tempfile.TemporaryDirectory(prefix="qd-bench-", dir=REPO / "target") as tmp:
        workdir = pathlib.Path(tmp)
        for length in lengths:
            first = bench(binary, candidate["knobs"], length, bench_tokens, workdir, "c", decode)
            r0 = rates(first)
            keys = {obj: "|".join((obj, bucket(length), dispatch_bin(r0[obj][1]), fam)) for obj in objectives}
            elites = {}
            for obj, key in keys.items():
                cell = snapshot["cells"].get(key)
                if cell and cell["candidate"]["name"] != candidate["name"]:
                    elites.setdefault(cell["candidate"]["name"], cell)
            # prefill-mlx: --compare candidates are timed interleaved too (side by side,
            # e.g. the exact elite beside an mlx-arith candidate) but contest no cell.
            for cmp in compare:
                elites.setdefault(cmp["name"], {"candidate": cmp, "compare_only": True})
            series = {"__candidate__": [first]}
            for name in elites:
                series[name] = []
            for rnd in range(rounds):
                for name, cell in elites.items():
                    try:
                        series[name].append(bench(binary, cell["candidate"]["knobs"], length, bench_tokens, workdir, "e", decode))
                    except SystemExit as e:
                        series[name] = None
                        print("  elite %s not runnable on this binary: %s" % (name, str(e)[:200]))
                        elites[name] = None
                if rnd > 0:
                    series["__candidate__"].append(bench(binary, candidate["knobs"], length, bench_tokens, workdir, "c", decode))
            if len(series["__candidate__"]) < rounds:
                series["__candidate__"].append(bench(binary, candidate["knobs"], length, bench_tokens, workdir, "c", decode))
            timings[str(length)] = {"candidate": series["__candidate__"], "elites": {k: v for k, v in series.items() if k != "__candidate__"}}
            for obj, key in keys.items():
                mine = [rates(r)[obj][0] for r in series["__candidate__"]]
                cell = snapshot["cells"].get(key)
                contest = {"cell": key, "objective": obj, "prompt_tokens": length, "candidate_rounds": mine,
                           "candidate_median": statistics.median(mine), "dispatches_per_token": r0[obj][1]}
                if not cell:
                    contest["verdict"] = "new-cell"
                elif cell["candidate"]["name"] == candidate["name"]:
                    contest["verdict"] = "already-elite"
                elif elites.get(cell["candidate"]["name"]) is None:
                    contest["verdict"] = "elite-unrunnable"
                else:
                    theirs = [rates(r)[obj][0] for r in series[cell["candidate"]["name"]]]
                    em = statistics.median(theirs)
                    margin = max(0.01 * em, 2 * (max(theirs) - min(theirs)))
                    contest.update(elite=cell["candidate"]["name"], elite_rounds=theirs, elite_median=em, margin=margin,
                                   gain_percent=100 * (contest["candidate_median"] - em) / em)
                    contest["verdict"] = "replaces-elite" if contest["candidate_median"] > em + margin else "loses"
                print("  %-34s %s: candidate %.2f%s -> %s" % (key, obj, contest["candidate_median"],
                      "" if "elite_median" not in contest else " vs elite %.2f (margin %.2f)" % (contest["elite_median"], contest["margin"]),
                      contest["verdict"]), flush=True)
                contests.append(contest)
                for cmp in compare:
                    theirs = series.get(cmp["name"])
                    if not theirs:
                        continue
                    theirs = [rates(r)[obj][0] for r in theirs]
                    em = statistics.median(theirs)
                    contest.setdefault("compared", {})[cmp["name"]] = {
                        "rounds": theirs, "median": em, "gain_percent": 100 * (contest["candidate_median"] - em) / em}
                    print("  %-34s %s: candidate %.2f vs %s %.2f (%+.1f%%, comparison only)" % (
                        key, obj, contest["candidate_median"], cmp["name"], em, 100 * (contest["candidate_median"] - em) / em), flush=True)
    return timings, contests, snapshot


def trial(args):
    candidate_path = args.candidate.resolve()
    candidate = json.loads(candidate_path.read_text())
    for field in ("name", "knobs"):
        if field not in candidate:
            raise SystemExit("candidate needs %r" % field)
    binary = args.binary.resolve()
    record = {"trial": "%s-%s" % (datetime.datetime.now().strftime("%Y%m%dT%H%M%S"), candidate["name"]), "time": now(),
              "candidate": {"name": candidate["name"], "knobs": candidate["knobs"], "family": family(candidate)},
              "parent": args.parent or candidate.get("parent"), "proposer": args.proposer, "why": args.why or candidate.get("why"),
              "hypothesis": candidate.get("hypothesis"), "expected": candidate.get("expected"),
              "worktree": str(REPO), "binary_sha256": sha256(binary)}
    print("trial %s (%s)" % (record["trial"], record["candidate"]["family"]), flush=True)
    extra = [int(x) for x in args.extra_feeds.split(",") if x]
    if args.admission == "mlx-arith":
        if not family(candidate).startswith("mlx-"):
            raise SystemExit("an mlx-arith candidate needs a family tag starting with 'mlx-' (docs/QD.md: its own cells)")
        admission = admit_mlx(binary, candidate, args.tiny, extra)
    else:
        admission = admit(binary, candidate_path, candidate, args.tiny, args.long, args.certify_cmd, extra)
    record["admission"] = admission
    if not admission["admitted"]:
        record["verdict"] = "rejected"
        with Archive(ARCHIVE) as a:
            a.append_trial(record)
        print("REJECTED: %s" % admission["reason"])
        return 1
    if args.admit_only:
        record["verdict"] = "admitted-untimed"
        with Archive(ARCHIVE) as a:
            a.append_trial(record)
        print("ADMITTED (%s), untimed" % admission["mode"])
        return 0
    lengths = [int(x) for x in args.lengths.split(",")]
    objectives = tuple(args.objectives.split(","))
    if not objectives or any(o not in ("decode", "prefill") for o in objectives):
        raise SystemExit("--objectives takes decode, prefill or decode,prefill")
    compare = [json.loads(pathlib.Path(c).read_text()) for c in args.compare]
    record["compared_with"] = [c["name"] for c in compare]
    timings, contests, snapshot = quality(binary, candidate, ARCHIVE, args.rounds, args.bench_tokens, lengths, objectives, compare)
    record["timings"] = timings
    record["contests"] = contests
    expected = candidate.get("expected") or {}
    agreement = {}
    for obj, pct in expected.items():
        gains = [c["gain_percent"] for c in contests if c["objective"] == obj and "gain_percent" in c]
        if gains:
            measured = statistics.median(gains)
            agreement[obj] = {"expected_percent": pct, "measured_percent": measured,
                              "agreed": (measured > 0) == (pct > 0) and (pct <= 0 or measured >= pct / 2)}
    record["hypothesis_check"] = agreement
    applied = []
    with Archive(ARCHIVE) as a:
        for c in contests:
            if c["verdict"] not in ("new-cell", "replaces-elite"):
                continue
            current = a.data["cells"].get(c["cell"])
            before = snapshot["cells"].get(c["cell"])
            if (current or {}).get("trial") != (before or {}).get("trial"):
                c["verdict"] += "-not-applied-cell-changed"
                continue
            obj, bkt, dbin, fam = c["cell"].split("|")
            a.data["cells"][c["cell"]] = {
                "objective": obj, "bucket": bkt, "dispatch_bin": dbin, "family": fam,
                "candidate": {"name": candidate["name"], "knobs": candidate["knobs"]},
                "tokens_per_second": c["candidate_median"], "rounds": c["candidate_rounds"], "prompt_tokens": c["prompt_tokens"],
                "dispatches_per_token": c["dispatches_per_token"], "admission": admission["mode"], "trial": record["trial"],
                "binary_sha256": record["binary_sha256"], "worktree": str(REPO), "updated": now(), "parent": record["parent"]}
            applied.append(c["cell"])
        record["verdict"] = "elite" if applied else "admitted-lost"
        record["cells_taken"] = applied
        a.append_trial(record)
    print("%s: %d cell(s) taken" % (record["verdict"], len(applied)))
    return 0


def baseline(args):
    data = json.loads(args.baseline.read_text())
    with Archive(ARCHIVE) as a:
        for engine, rows in data["engines"].items():
            a.data["baselines"][engine] = {
                str(k): {"prefill_tokens_per_second": v["prefill_tokens_per_second"], "decode_tokens_per_second": v["decode_tokens_per_second"]}
                for k, v in rows.items() if isinstance(v, dict)}
        a.data["baseline_source"] = {"file": str(args.baseline), "machine": data.get("machine"), "versions": data.get("versions"),
                                     "decode_protocol": data.get("decode_protocol"), "prefill_protocol": data.get("prefill_protocol")}
    print("recorded %s" % ", ".join(sorted(data["engines"])))
    return 0


def render_cmd(args):
    with Archive(ARCHIVE):
        pass  # rewriting the archive regenerates elites.md
    return 0


def screen(args):
    """Cheap screening (docs/QD.md), by decode or prefill rate.
    Interleave the named knob sets (A, B, A, B, ...) at one prompt length under the lock;
    print each one's medians and the gain of each over the first. Appended to
    target/screens.jsonl, and with --record to trials.jsonl as a 'screen' line (never an elite)."""
    sets = []
    for spec in args.knobs:
        name, _, path = spec.partition("=")
        cand = json.loads(pathlib.Path(path).read_text())
        sets.append((name, cand.get("knobs", cand)))
    decode = args.decode_tokens if args.decode_tokens is not None else (256 if args.metric == "decode" else 0)
    key = "%s_tokens_per_second" % args.metric
    runs = {name: [] for name, _ in sets}
    with tempfile.TemporaryDirectory(prefix="qd-screen-", dir=REPO / "target") as tmp:
        workdir = pathlib.Path(tmp)
        for rnd in range(args.rounds):
            for name, knobs in sets:
                path = workdir / ("knobs-%s.json" % name)
                path.write_text(json.dumps(knobs))
                r = nano_bench.bench_once(args.binary.resolve(), CHECKPOINT, args.length, path, args.bench_tokens,
                                          decode=decode, repeats=args.repeats)
                runs[name].append(r)
                print("  round %d %-20s decode %.2f prefill %.1f (%.0f dispatches/prefill forward)" % (
                    rnd, name, r["decode_tokens_per_second"], r["prefill_tokens_per_second"], r["dispatches_per_prefill_forward"]), flush=True)
    base = statistics.median(r[key] for r in runs[sets[0][0]])
    summary = {"time": now(), "length": args.length, "rounds": args.rounds, "metric": args.metric, "decode_tokens": decode,
               "binary_sha256": sha256(args.binary.resolve()), "note": args.note, "sets": {}}
    for name, knobs in sets:
        vals = [r[key] for r in runs[name]]
        med = statistics.median(vals)
        summary["sets"][name] = {"knobs": knobs, "decode": [r["decode_tokens_per_second"] for r in runs[name]],
                                 "prefill": [r["prefill_tokens_per_second"] for r in runs[name]], "median": med,
                                 "gain_percent": 100 * (med - base) / base,
                                 "dispatches_per_prefill_forward": runs[name][0]["dispatches_per_prefill_forward"]}
        print("%-20s %s median %.2f (%+.1f%%) rounds %s" % (name, args.metric, med, 100 * (med - base) / base, " / ".join("%.1f" % x for x in vals)))
    with open(REPO / "target/screens.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    if args.record:
        with Archive(ARCHIVE) as a:
            a.append_trial({"trial": "%s-screen" % datetime.datetime.now().strftime("%Y%m%dT%H%M%S"), "time": now(),
                            "verdict": "screen", "proposer": args.proposer, "why": args.note, "worktree": str(REPO), "screen": summary})
    return 0


def admit_mlx(binary, candidate, tiny, extra_feeds=()):
    """Admission in the mlx-arith tier (docs/QD.md, "The mlx-arith tier").
    The binary's reference executor must still reproduce the pin. Then, on every tiny
    checkpoint (feeds 7 and the batch) and on short, window and long (the batch, plus
    extra feeds), the candidate must meet the goldens' gate the reference itself meets:
    logits and residual streams within 2 x float32 noise floor + the golden producer's
    own error, top-1 at every stored position, greedy identical. The largest difference
    from the exact batched path (reference executor, feed 64, itself checked against the
    pin) is recorded. The per-kernel gate, bit identity with stock MLX's operation, is
    tools/nano_mlx_ops.py; its last result (target/nano-mlx-ops.json) is attached."""
    from nano_verify import run_eval
    binary_sha = sha256(binary)
    pin = load_pin()
    cache_dir = REPO / "target/qd-reference" / binary_sha[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch = int(candidate["knobs"].get("batch", 64))
    knobs_file = REPO / "target" / ("qd-knobs-%s.json" % candidate["name"])
    knobs_file.write_text(json.dumps(candidate["knobs"]))
    guarded = lambda cmd: nano_bench.guarded(cmd, timeout=3600)
    checks, ok, reason = [], True, None
    with tempfile.TemporaryDirectory(prefix="qd-admit-", dir=REPO / "target") as tmp:
        workdir = pathlib.Path(tmp)
        for case in cases(tiny, True):
            name, checkpoint, golden, _ = case
            real = not name.startswith("tiny/")
            runner = guarded if real else None
            pinned = pin["cases"].get(name)
            if pinned is None:
                raise SystemExit("case %s is not in the pinned reference; re-pin with it" % name)
            if not same_fingerprint(reference(binary, case, workdir, cache_dir), pinned):
                ok, reason = False, "this binary's reference executor differs from the pinned reference on %s" % name
                break
            g_logits = np.load(golden / "logits.npy").astype(np.float64)
            g_hidden = np.load(golden / "hidden.npy") if (golden / "hidden.npy").exists() else None
            g_greedy = json.loads((golden / "greedy.json").read_text())["tokens"]
            if real:
                meta = json.loads((golden / "meta.json").read_text())
                floor = (meta["float32_noise_floor_logits"], meta["float32_noise_floor_hidden"])
                ref_err = (meta["max_abs_diff_mlx_f32_vs_numpy_logits"], meta["max_abs_diff_mlx_f32_vs_numpy_hidden"])
            else:
                floor, ref_err = tiny_floor(checkpoint, golden), (0.0, 0.0)
            gate = (2 * floor[0] + ref_err[0], 2 * floor[1] + ref_err[1])
            n_tokens = len(json.loads((golden / "tokens.json").read_text())["tokens"])
            exact = run_eval(binary, checkpoint, golden, min(64, n_tokens), workdir, [], g_hidden is not None, runner)
            exact_fp = hashlib.sha256(exact[1].astype("<f4").view("<u4").tobytes()).hexdigest()
            if exact_fp != pinned["logits_sha256"]:
                ok, reason = False, "exact batched path differs from the pin on %s (binary broken)" % name
                break
            feeds = ((7,) if not real else ()) + (batch,) + (tuple(extra_feeds) if real else ())
            for feed in sorted(set(min(f, n_tokens) for f in feeds)):
                summary, logits, hidden = run_eval(binary, checkpoint, golden, feed, workdir, ["--knobs", str(knobs_file)],
                                                   g_hidden is not None, runner)
                row = {"case": name, "feed": feed, "dispatches_per_forward": summary["dispatches"] / summary["forwards"],
                       "logits_max_abs": float(np.abs(logits.astype(np.float64) - g_logits).max()), "gate_logits": gate[0],
                       "top1_agree": int((np.asarray(summary["top1"]) == g_logits.argmax(1)).sum()), "positions": len(g_logits),
                       "greedy_equal": summary["greedy"] == g_greedy,
                       "vs_exact_logits_max_abs": float(np.abs(logits.astype(np.float64) - exact[1].astype(np.float64)).max())}
                passed = row["logits_max_abs"] <= gate[0] and row["top1_agree"] == len(g_logits) and row["greedy_equal"]
                if g_hidden is not None:
                    row["hidden_max_abs"] = float(np.abs(hidden.astype(np.float64) - g_hidden.astype(np.float64)).max())
                    row["gate_hidden"] = gate[1]
                    row["vs_exact_hidden_max_abs"] = float(np.abs(hidden.astype(np.float64) - exact[2].astype(np.float64)).max())
                    passed = passed and row["hidden_max_abs"] <= gate[1]
                row["pass"] = bool(passed)
                checks.append(row)
                print("  mlx-arith %-22s feed %4d: logits %.3g (gate %.3g)%s top1 %d/%d greedy %s; vs exact %.3g: %s" % (
                    name, feed, row["logits_max_abs"], gate[0],
                    "" if g_hidden is None else " hidden %.3g (gate %.3g)" % (row["hidden_max_abs"], gate[1]),
                    row["top1_agree"], row["positions"], "equal" if row["greedy_equal"] else "DIFFER",
                    row["vs_exact_logits_max_abs"], "PASS" if passed else "FAIL"), flush=True)
                if not passed:
                    ok, reason = False, "%s feed %d fails the goldens' gate" % (name, feed)
                    break
            if not ok:
                break
    ops = REPO / "target/nano-mlx-ops.json"
    return {"admitted": ok, "mode": "mlx-arith", "certificate": None, "checks": checks, "binary_sha256": binary_sha,
            "reason": reason, "mlx_ops": json.loads(ops.read_text()) if ops.exists() else None,
            "reference": "goldens' gate (docs/QD.md mlx-arith tier); exact path = pinned reference %s" % pin["binary_sha256"][:16]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trial")
    t.add_argument("binary", type=pathlib.Path)
    t.add_argument("candidate", type=pathlib.Path)
    t.add_argument("--parent")
    t.add_argument("--proposer", required=True)
    t.add_argument("--why")
    t.add_argument("--tiny", type=pathlib.Path, default=REPO / "target/lfm3-tiny")
    t.add_argument("--long", action="store_true", help="also admit on the 4,096-token golden (slow)")
    t.add_argument("--certify-cmd", help="certificate checker; {candidate} is the candidate path")
    t.add_argument("--admit-only", action="store_true")
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    t.add_argument("--objectives", default="decode,prefill",
                   help="cells to contest; a bench run measures both, only these objectives' cells are compared and written")
    t.add_argument("--bench-tokens", type=pathlib.Path, default=REPO / "target/nano-bench-tokens.json")
    # prefill search and mlx-arith tier
    t.add_argument("--extra-feeds", default="", help="extra feeds for the real goldens, e.g. 100,255")
    t.add_argument("--compare", action="append", default=[],
                   help="candidate file timed interleaved beside this one (recorded, contests no cell); repeatable")
    t.add_argument("--admission", default="exact", choices=("exact", "mlx-arith"),
                   help="exact: bit-identical to the pinned reference; mlx-arith: the goldens' gate (docs/QD.md second tier)")
    s = sub.add_parser("screen")
    s.add_argument("binary", type=pathlib.Path)
    s.add_argument("knobs", nargs="+", help="NAME=candidate-or-knobs.json; the first is the baseline")
    s.add_argument("--length", type=int, default=1024)
    s.add_argument("--metric", default="prefill", choices=("prefill", "decode"))
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--decode-tokens", type=int, help="default 256 for --metric decode, 0 for prefill")
    s.add_argument("--bench-tokens", type=pathlib.Path, default=REPO / "target/nano-bench-tokens.json")
    s.add_argument("--note", default="")
    s.add_argument("--record", action="store_true")
    s.add_argument("--proposer", default="prefill-mlx")
    b = sub.add_parser("baseline")
    b.add_argument("--baseline", type=pathlib.Path, default=REPO / "benchmarks/lfm3-mlx-baseline.json")
    sub.add_parser("render")
    sub.add_parser("parents")
    q = sub.add_parser("pin-reference")
    q.add_argument("binary", type=pathlib.Path)
    q.add_argument("--tiny", type=pathlib.Path, default=REPO / "target/lfm3-tiny")
    q.add_argument("--long", action="store_true", help="also pin the 4,096-token golden (slow)")
    q.add_argument("--replace", action="store_true")
    q.add_argument("--why")
    args = p.parse_args()
    sys.exit({"trial": trial, "baseline": baseline, "render": render_cmd, "parents": parents,
              "pin-reference": pin_reference, "screen": screen}[args.command](args))


if __name__ == "__main__":
    main()
