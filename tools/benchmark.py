#!/usr/bin/env python3
"""Benchmark local Flint Zig/Metal against stock MLX, without downloading weights.

Run under tools/guarded_run.py; this script deliberately does not acquire another
runtime's lock or import any external repository's guard. The native child exits
before MLX loads a full model. Neither engine writes to the checkpoint directory.

From zig/: python3 tools/guarded_run.py --timeout 600 -- \
    python3 tools/benchmark.py CHECKPOINT --output result.json
Build flint-bench in ReleaseFast/ReleaseSafe first. Individual measurements are
printed as NDJSON, followed by the summary JSON also saved with --output.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

# Set these before importing Transformers, huggingface_hub, or mlx_lm. Only local
# tokenizer/model files are allowed, including when a required file is missing.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = (
    "Write a Python binary search tree with insert, search, and delete methods. "
    "Include detailed comments."
)
CONTINUATION = """class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
            return
        current = self.root
        while True:
            if value < current.value:
                current = current.left
            else:
                current = current.right
"""
ENGINES = ("zig_fp32", "mlx_fp32", "mlx_bf16")


def positive_u32(value):
    number = int(value)
    if not 0 < number <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("must be in 1..4294967295")
    return number


def resident_worker_count(value):
    number = int(value)
    if not 1 <= number <= 32:
        raise argparse.ArgumentTypeError("must be in 1..32 (omit for staged execution)")
    return number


def workload(tokenizer, decode_tokens):
    """Keep the original coding workload, including every repeated input token."""
    prompt = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=True,
        add_generation_prompt=True, return_dict=False,
    ))
    # Four copies reproduce the original benchmark; extend rather than silently
    # truncating when a caller requests a longer teacher-forced continuation.
    copies = 4
    while True:
        continuation = tokenizer.encode(CONTINUATION * copies, add_special_tokens=False)
        if len(continuation) >= decode_tokens:
            break
        if not continuation:
            raise ValueError("tokenizer produced an empty continuation")
        copies *= 2
    decode = list(continuation[:decode_tokens])
    if not prompt:
        raise ValueError("tokenizer produced an empty prompt")
    for token in prompt + decode:
        if not isinstance(token, int) or not 0 <= token <= 0xFFFFFFFF:
            raise ValueError("tokenizer returned an invalid u32 token ID")
    return prompt, decode


def validate_rows(rows, engine, repeats, resident_workers=0):
    if len(rows) != repeats:
        raise ValueError("{} returned {} runs, expected {}".format(engine, len(rows), repeats))
    for repeat, row in enumerate(rows):
        if row.get("engine") != engine or row.get("repeat") != repeat:
            raise ValueError("unexpected benchmark engine or repeat index")
        for field in ("prefill_seconds", "decode_seconds"):
            seconds = row.get(field)
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("{} has an invalid {}".format(engine, field))
        for field in ("prefill_gpu_seconds", "decode_gpu_seconds"):
            if field in row:
                seconds = row[field]
                if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("{} has an invalid {}".format(engine, field))
        if not isinstance(row.get("checksum"), int) or row["checksum"] < 0:
            raise ValueError("{} has an invalid checksum".format(engine))
        if engine == "zig_fp32":
            for field in ("submissions_per_token", "dispatches_per_token"):
                if field not in row and not resident_workers:
                    continue
                count = row.get(field)
                if type(count) is not int or count <= 0:
                    raise ValueError("{} has a missing or invalid {}".format(engine, field))
                if resident_workers and count != 1:
                    raise ValueError("resident execution requires {}=1, got {}".format(field, count))
    if len({row["checksum"] for row in rows}) != 1:
        raise ValueError("{} greedy checksum changed between identical repeats".format(engine))


def native_runs(binary, checkpoint, prompt, decode, repeats, resident_workers=0,
                resident_row_tile=1, emit_predictions=False, staged_row_tile=1):
    if resident_row_tile not in (1, 2, 4, 8) or (resident_row_tile != 1 and not resident_workers):
        raise ValueError("resident row tile must be 1/2/4/8; nondefault tiles require resident workers")
    if staged_row_tile not in (1, 2, 4, 8) or (staged_row_tile != 1 and resident_workers):
        raise ValueError("staged row tile must be 1/2/4/8; nondefault tiles require staged execution")
    with tempfile.TemporaryDirectory(prefix="flint-bench-") as temporary:
        input_path = Path(temporary) / "input.json"
        input_path.write_text(json.dumps({
            "checkpoint": str(checkpoint), "prompt": prompt,
            "decode": decode, "repeats": repeats, "resident_workers": resident_workers,
            "resident_row_tile": resident_row_tile, "emit_predictions": emit_predictions,
            "staged_row_tile": staged_row_tile,
        }), encoding="utf-8")
        # Inherit stderr for native diagnostics, keeping stdout machine-readable.
        # The outer guarded_run.py owns the timeout and resource limits for both
        # engines; waiting here guarantees that full models never overlap.
        result = subprocess.run(
            [str(binary), str(input_path)], text=True,
            stdout=subprocess.PIPE, check=True,
        )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    validate_rows(rows, "zig_fp32", repeats, resident_workers)
    return rows


def mlx_runs(checkpoint, prompt, decode, repeats):
    # No custom kernels, monkeypatches, compilation flags, or checkpoint conversion.
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_memory_limit(32 * 1024 ** 3)
    mx.set_cache_limit(512 * 1024 ** 2)
    model, _ = load(str(checkpoint), tokenizer_config={
        "fix_mistral_regex": False, "local_files_only": True, "trust_remote_code": False,
    })
    rows = []
    for engine, dtype in (("mlx_fp32", mx.float32), ("mlx_bf16", mx.bfloat16)):
        # Cast floating parameters/scales, not the packed integer quantized weights.
        model.set_dtype(dtype)
        mx.eval(model.parameters())

        def run(count):
            cache = make_prompt_cache(model)
            start = time.perf_counter()
            logits = model(mx.array([prompt], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            prefilled = time.perf_counter()
            checksum = 0
            for token in decode[:count]:
                logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
                # item() is the per-token GPU->CPU synchronization, matching the
                # native forward+greedy path rather than timing lazy graph builds.
                checksum += int(mx.argmax(logits[0, -1]).item())
            finished = time.perf_counter()
            return {
                "engine": engine, "prefill_seconds": prefilled - start,
                "decode_seconds": finished - prefilled, "checksum": checksum,
            }

        run(min(8, len(decode)))
        selected = []
        for repeat in range(repeats):
            row = run(len(decode))
            row["repeat"] = repeat
            selected.append(row)
            print(json.dumps(row), flush=True)
        validate_rows(selected, engine, repeats)
        rows.extend(selected)
        gc.collect()
    return rows


def summarize(rows, prompt, decode, repeats, resident_workers=0, resident_row_tile=1,
              staged_row_tile=1):
    report = {
        "native_execution": {
            "mode": "resident_single_dispatch" if resident_workers else "staged",
            "resident_workers": resident_workers,
            "resident_row_tile": resident_row_tile,
            "staged_row_tile": staged_row_tile,
        },
        "prompt_tokens": len(prompt), "decode_tokens": len(decode), "repeats": repeats,
        "warmup_decode_tokens": min(8, len(decode)),
        "workload": (
            "Identical teacher-forced code token IDs; fresh/reset KV cache each run; "
            "one cached forward plus synchronous greedy CPU readback per decode token. "
            "Model loading, reset/cache creation, dtype conversion, and warmup excluded."
        ),
        "prefill_note": (
            "Zig calls forward once per prompt token; stock MLX prefills the prompt as "
            "one batch. Prefill rates compare engine capabilities, not identical dispatches."
        ),
        "precision_note": (
            "zig_fp32 and mlx_fp32 use FP32 floating parameters/activations; mlx_bf16 "
            "uses BF16. Native execution mode changes scheduling, not this math baseline. "
            "Packed checkpoint quantized weights remain packed in all engines. "
            "Checksums are diagnostics, not proof of logit or token-by-token parity."
        ),
        # Saving actual input IDs makes comparisons reproducible even across
        # tokenizer releases, without recording machine-specific checkpoint paths.
        "input": {"prompt": prompt, "decode": decode},
        "runs": rows,
        "engines": {},
        "ratios": {},
    }
    encoded = json.dumps(report["input"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["input_sha256"] = hashlib.sha256(encoded).hexdigest()
    for engine in ENGINES:
        selected = [row for row in rows if row["engine"] == engine]
        validate_rows(selected, engine, repeats, resident_workers)
        prefill = statistics.median(row["prefill_seconds"] for row in selected)
        decode_seconds = statistics.median(row["decode_seconds"] for row in selected)
        report["engines"][engine] = {
            "median_prefill_seconds": prefill,
            "median_decode_seconds": decode_seconds,
            "prefill_tokens_per_second": len(prompt) / prefill,
            "decode_tokens_per_second": len(decode) / decode_seconds,
            "prefill_run_seconds": [row["prefill_seconds"] for row in selected],
            "decode_run_seconds": [row["decode_seconds"] for row in selected],
            "checksums": [row["checksum"] for row in selected],
        }
        if engine == "zig_fp32":
            # Preserve observed per-run counters; do not infer counts from the
            # requested mode or relabel scheduling as a different math engine.
            for field in ("submissions_per_token", "dispatches_per_token",
                          "prefill_gpu_seconds", "decode_gpu_seconds"):
                if any(field in row for row in selected):
                    report["engines"][engine][field] = [row.get(field) for row in selected]
    for engine in ENGINES[1:]:
        for phase in ("prefill", "decode"):
            field = phase + "_tokens_per_second"
            zig_rate = report["engines"]["zig_fp32"][field]
            mlx_rate = report["engines"][engine][field]
            report["ratios"]["zig_fp32_{}_speedup_over_{}".format(phase, engine)] = zig_rate / mlx_rate
            report["ratios"]["{}_{}_speedup_over_zig_fp32".format(engine, phase)] = mlx_rate / zig_rate
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="existing local Flint checkpoint directory")
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/flint-bench"))
    parser.add_argument("--decode-tokens", type=positive_u32, default=64)
    parser.add_argument("--repeats", type=positive_u32, default=3)
    parser.add_argument("--resident-workers", type=resident_worker_count, default=0,
                        help="opt into resident single-dispatch execution with 1..32 workers; default: staged")
    parser.add_argument("--resident-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per resident workgroup; nondefault requires --resident-workers")
    parser.add_argument("--output", type=Path, help="optional complete JSON report (outside checkpoint)")
    parser.add_argument("--staged-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per staged workgroup; nondefault conflicts with --resident-workers")
    args = parser.parse_args()
    if args.resident_row_tile != 1 and not args.resident_workers:
        parser.error("--resident-row-tile requires --resident-workers")
    if args.staged_row_tile != 1 and args.resident_workers:
        parser.error("nondefault --staged-row-tile conflicts with --resident-workers")
    checkpoint = args.checkpoint.resolve()
    binary = args.binary.resolve()
    if not checkpoint.is_dir():
        parser.error("checkpoint must be an existing local directory")
    if not binary.is_file():
        parser.error("build flint-bench first or supply --binary")
    if args.output:
        output = args.output.resolve()
        if output == checkpoint or checkpoint in output.parents:
            parser.error("--output must not write inside the checkpoint directory")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        fix_mistral_regex=False,
    )
    template = checkpoint / "chat_template.jinja"
    if template.is_file():
        tokenizer.chat_template = template.read_text(encoding="utf-8")
    if not tokenizer.chat_template:
        parser.error("checkpoint has no local chat template")
    prompt, decode = workload(tokenizer, args.decode_tokens)
    rows = native_runs(binary, checkpoint, prompt, decode, args.repeats,
                       args.resident_workers, args.resident_row_tile,
                       staged_row_tile=args.staged_row_tile)
    for row in rows:
        print(json.dumps(row), flush=True)
    del tokenizer
    gc.collect()
    rows.extend(mlx_runs(checkpoint, prompt, decode, args.repeats))
    report = summarize(rows, prompt, decode, args.repeats,
                       args.resident_workers, args.resident_row_tile,
                       staged_row_tile=args.staged_row_tile)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
