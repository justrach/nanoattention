#!/usr/bin/env python3
"""Sequential, prediction-gated staged-row tuning; run inside guarded_run.py.

Coarse results never establish the requested throughput objective. Confirmed native timings
still require separate compare_mlx.py logit checks and a contemporary reference
benchmark. No defaults are edited, no checkpoint is written, no model overlaps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/flint-bench"))
    parser.add_argument("--coarse-tokens", type=int, choices=range(1, 65), default=32)
    parser.add_argument("--repeats", type=int, choices=range(1, 11), default=2)
    parser.add_argument("--target-tokens-per-second", type=benchmark.positive_u32, default=100)
    parser.add_argument("--output", type=Path, default=Path(".zig-cache/staged-tune.json"))
    args = parser.parse_args()
    if os.environ.get("FLINT_PROFILE") == "1":
        parser.error("unset FLINT_PROFILE: profiling throughput is not a benchmark")
    checkpoint, binary, output = (p.resolve() for p in (args.checkpoint, args.binary, args.output))
    if checkpoint in output.parents or output == checkpoint:
        parser.error("output must not be in checkpoint directory")
    if not binary.is_file() or not checkpoint.is_dir():
        parser.error("checkpoint and compiled benchmark must exist")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                              trust_remote_code=False, fix_mistral_regex=False)
    template = checkpoint / "chat_template.jinja"
    if template.is_file():
        tokenizer.chat_template = template.read_text()
    prompt, decode = benchmark.workload(tokenizer, 64)
    del tokenizer
    root = Path(__file__).resolve().parents[1]
    report = {"status": "running", "precision": "native FP32 activations", "target_tokens_per_second": args.target_tokens_per_second,
              "input": {"prompt": prompt, "decode": decode}, "candidates": {},
              "benchmark_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (root / "src").iterdir() if p.suffix in (".metal", ".m", ".zig")},
              "note": "Native timing/prediction admission is not complete MLX logit validation."}
    report["input_sha256"] = hashlib.sha256(json.dumps(report["input"], sort_keys=True,
                                                     separators=(",", ":")).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    def evaluate(tile, phase, count, repeats, expected=None):
        candidate = report["candidates"].setdefault(str(tile), {"staged_row_tile": tile})
        result = candidate[phase] = {"status": "running"}
        save()
        try:
            rows = benchmark.native_runs(binary, checkpoint, prompt, decode[:count], repeats,
                                         emit_predictions=True, staged_row_tile=tile)
            result["runs"] = rows
            # Staged execution intentionally retains multiple dispatches.
            expected_predictions = expected
            for row in rows:
                actual = row.get("predictions")
                if (not isinstance(actual, list) or len(actual) != count or
                        any(type(t) is not int or not 0 <= t <= 0xFFFFFFFF for t in actual)):
                    raise ValueError("missing per-position predictions")
                if expected_predictions is not None and actual != expected_predictions:
                    raise ValueError("position-wise predictions differ from seed/repeat")
                expected_predictions = actual
            seconds = statistics.median(row["decode_seconds"] for row in rows)
            gpu = statistics.median(row["decode_gpu_seconds"] for row in rows)
            result.update(status="valid", predictions=expected_predictions, median_decode_seconds=seconds,
                          decode_tokens_per_second=count / seconds, median_gpu_seconds=gpu)
        except Exception as exc:
            result.update(status="rejected", error="{}: {}".format(type(exc).__name__, exc))
        save()
        print("tile={} phase={} status={} tok/s={}".format(tile, phase, result["status"],
              result.get("decode_tokens_per_second")), flush=True)
        return result

    save()
    try:
        seed = evaluate(1, "coarse", args.coarse_tokens, args.repeats)
        if seed["status"] != "valid":
            raise RuntimeError("seed failed")
        for tile in (2, 4, 8):
            evaluate(tile, "coarse", args.coarse_tokens, args.repeats, seed["predictions"])
        ranked = sorted((int(t) for t, c in report["candidates"].items() if c["coarse"]["status"] == "valid"),
                        key=lambda t: report["candidates"][str(t)]["coarse"]["median_decode_seconds"])
        full_seed = evaluate(1, "full", 64, 3)
        if full_seed["status"] != "valid" or full_seed["predictions"][:args.coarse_tokens] != seed["predictions"]:
            raise RuntimeError("full seed failed or changed coarse prefix")
        for tile in ranked[:3]:
            if tile != 1:
                evaluate(tile, "full", 64, 3, full_seed["predictions"])
        full = [(int(t), c["full"]) for t, c in report["candidates"].items()
                if c.get("full", {}).get("status") == "valid"]
        winner, timing = min(full, key=lambda item: item[1]["median_decode_seconds"])
        report.update(status="native_confirmation_complete", winner={"staged_row_tile": winner,
                      "decode_tokens_per_second": timing["decode_tokens_per_second"],
                      "native_target_measured": timing["decode_tokens_per_second"] >= args.target_tokens_per_second,
                      "mlx_logit_validation": "not performed by this script"})
        save()
    except BaseException as exc:
        report.update(status="failed", error="{}: {}".format(type(exc).__name__, exc))
        save()
        raise
    print(json.dumps({"output": str(output), "winner": report["winner"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
