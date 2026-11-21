#!/usr/bin/env python3
"""Compare native nano and pinned MLX on the original Inkling checkpoint.

Teacher-forced decode or free-running greedy generation, with the next MLX
step enqueued before the previous argmax read-back. One warmup and one timed run per process; alternate process
order each round. The output is throughput, not a model-quality gate.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def mlx_worker(checkpoint, prompt_count, decode_count, batch, mode):
    import mlx.core as mx
    from mlx_lm.utils import load_model

    model, _ = load_model(checkpoint, model_config={"vision_config": None})
    # src/mega/bench.zig's deterministic default workload, unchanged.
    state = 0x9E3779B97F4A7C15
    tokens = []
    for _ in range(prompt_count + decode_count):
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        tokens.append((state >> 33) % 100_000)
    steps = [mx.array([[t]]) for t in tokens[prompt_count:]]
    predictions = [0] * (decode_count + 1)
    warmup_predictions = None
    for _ in range(2):
        cache = model.make_cache()
        t0 = time.perf_counter()
        for i in range(0, prompt_count, batch):
            ids = mx.array([tokens[i:min(i + batch, prompt_count)]])
            pred = mx.argmax(model(ids, cache=cache)[0, -1])
            mx.eval(pred)
        predictions[0] = int(pred.item())
        checksum = predictions[0]
        t1 = time.perf_counter()
        prev = None
        for j, ids in enumerate(steps):
            if mode == "greedy":
                ids = pred.astype(mx.int32).reshape(1, 1)
            pred = mx.argmax(model(ids, cache=cache)[0, -1])
            mx.async_eval(pred)
            if prev is not None:
                predictions[j] = int(prev.item())
                checksum += predictions[j]
            prev = pred
        predictions[-1] = int(prev.item())
        checksum += predictions[-1]
        t2 = time.perf_counter()
        if warmup_predictions is None:
            warmup_predictions = predictions.copy()
        elif predictions != warmup_predictions:
            raise RuntimeError("MLX predictions changed between warmup and timed run")
    return {"prefill_seconds": t1 - t0, "decode_seconds": t2 - t1, "checksum": checksum,
            "predictions": predictions, "decode_mode": mode}


def main(argv):
    if argv[:1] == ["mlx-worker"]:
        checkpoint = Path(argv[1])
        print(json.dumps(mlx_worker(checkpoint, *map(int, argv[2:5]), argv[5])))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mlx-python", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--prompt", type=int, default=64)
    parser.add_argument("--decode", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--decode-mode", choices=("teacher-forced", "greedy"), default="teacher-forced")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--knob", action="append", default=[])
    parser.add_argument("--reference-binary", type=Path)
    args = parser.parse_args(argv)
    if min(args.prompt, args.decode, args.batch, args.rounds, args.timeout) < 1:
        parser.error("lengths, batch, rounds and timeout must be positive")
    if any(k.startswith("batch=") for k in args.knob):
        parser.error("use --batch to keep both engines on the same batch size")

    checkpoint = str(args.checkpoint.resolve())
    common = ["bench", checkpoint, "--context", str(max(args.batch, args.prompt + args.decode)),
              "--prompt-tokens", str(args.prompt), "--decode-tokens", str(args.decode),
              "--repeats", "1", "--knob", "batch=" + str(args.batch)]
    if args.decode_mode == "greedy":
        common.extend(("--decode-mode", "greedy"))
    commands = {
        "mlx": [str(args.mlx_python.absolute()), str(Path(__file__).resolve()), "mlx-worker", checkpoint,
                str(args.prompt), str(args.decode), str(args.batch), args.decode_mode],
        "native": [str(args.binary.resolve()), *common, *(part for knob in args.knob for part in ("--knob", knob))],
    }
    if args.reference_binary is not None:
        commands["native_reference"] = [str(args.reference_binary.resolve()), *common]
    guard = [sys.executable, str(Path(__file__).with_name("guarded_run.py")), "--lock", str(args.lock),
             "--timeout", str(args.timeout), "--"]
    samples = {name: [] for name in commands}
    names = list(commands)
    for i in range(args.rounds):
        for name in names[i % len(names):] + names[:i % len(names)]:
            value = json.loads(subprocess.check_output(guard + commands[name], text=True))
            if value.get("decode_mode", "teacher-forced") != args.decode_mode:
                raise ValueError(f"{name} did not use the requested decode mode")
            sample = value if name == "mlx" else {
                "prefill_seconds": value["prefill_seconds"][0],
                "decode_seconds": value["decode_seconds"][0],
                "checksum": value["prediction_checksum"],
                "predictions": value.get("predictions"),
            }
            picks = sample["predictions"]
            if picks is None and name != "native_reference":
                raise ValueError(f"{name} did not report predictions; rebuild or use --reference-binary")
            if picks is not None and (len(picks) != args.decode + 1 or sum(picks) != sample["checksum"]):
                raise ValueError(f"{name} returned invalid predictions")
            samples[name].append(sample)
    result = dict(prompt=args.prompt, decode=args.decode, batch=args.batch, rounds=args.rounds,
                  decode_mode=args.decode_mode, knobs=args.knob)
    for name, runs in samples.items():
        result[name] = dict(samples=runs,
                            prefill_tps=args.prompt / statistics.median(r["prefill_seconds"] for r in runs),
                            decode_tps=args.decode / statistics.median(r["decode_seconds"] for r in runs))
    result["native_mismatches"] = [
        [
            {"step": step, "native": a, "mlx": b}
            for step, (a, b) in enumerate(zip(native["predictions"], mlx["predictions"]))
            if a != b
        ]
        for native, mlx in zip(samples["native"], samples["mlx"])
    ]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
