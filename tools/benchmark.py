#!/usr/bin/env python3
"""Benchmark local Flint Zig/Metal against stock MLX, without downloading weights.

Run under tools/guarded_run.py; this script deliberately does not acquire another
runtime's lock or import any external repository's guard. The native child exits
before MLX loads a full model. Neither engine writes to the checkpoint directory.

From zig/: python3 tools/guarded_run.py --timeout 600 -- \
    python3 tools/benchmark.py CHECKPOINT --output result.json
Build flint-bench in ReleaseFast/ReleaseSafe first. Individual measurements are
printed as NDJSON, followed by the summary JSON also saved with --output.

--prompt-tokens selects an exact-length raw code prefix made by repeating stock
code token IDs. This is a context-length performance workload, not a quality
evaluation or the original short chat prompt. All prefix tokens are really
prefilled; MLX uses bounded 512-token chunks only in this opt-in protocol.
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
LONG_PREFILL_CHUNK_SIZE = 512
ATTENTION_CHUNKS = (0, 128, 256, 512, 1024)
MLX_ATTENTION_IMPLEMENTATION = "mlx_layout_adaptive_stripes"
LONG_PROMPT_NOTE = (
    "Exact-length raw code prefix: repeat tokenizer.encode(CONTINUATION, "
    "add_special_tokens=False) token IDs and truncate to the requested length; "
    "no chat template, padding, or synthetic cache positions. Context-length "
    "performance workload, not quality evaluation or the original short chat prompt."
)


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


def code_prompt(tokenizer, prompt_tokens):
    """Build a real raw code prefix by deterministic repetition of stock token IDs."""
    if type(prompt_tokens) is not int or not 0 < prompt_tokens <= 0xFFFFFFFF:
        raise ValueError("prompt_tokens must be in 1..4294967295")
    stock = list(tokenizer.encode(CONTINUATION, add_special_tokens=False))
    if not stock:
        raise ValueError("tokenizer produced an empty code prefix")
    if any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in stock):
        raise ValueError("tokenizer returned an invalid u32 token ID")
    copies, remainder = divmod(prompt_tokens, len(stock))
    return stock * copies + stock[:remainder]


def workload(tokenizer, decode_tokens, prompt_tokens=None):
    """Preserve the original short workload unless an exact raw prefix is requested."""
    if prompt_tokens is None:
        prompt = list(tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        ))
    else:
        prompt = code_prompt(tokenizer, prompt_tokens)
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
                resident_row_tile=1, emit_predictions=False, staged_row_tile=1,
                reuse_prefill=False, attention_chunk=None, mlx_attention=False):
    if resident_row_tile not in (1, 2, 4, 8) or (resident_row_tile != 1 and not resident_workers):
        raise ValueError("resident row tile must be 1/2/4/8; nondefault tiles require resident workers")
    if staged_row_tile not in (1, 2, 4, 8) or (staged_row_tile != 1 and resident_workers):
        raise ValueError("staged row tile must be 1/2/4/8; nondefault tiles require staged execution")
    if mlx_attention and (resident_workers or attention_chunk is not None):
        raise ValueError("mlx attention conflicts with resident/split attention")
    if attention_chunk is not None:
        if type(attention_chunk) is not int or attention_chunk not in ATTENTION_CHUNKS:
            raise ValueError("attention chunk must be 0/128/256/512/1024")
        if resident_workers:
            raise ValueError("attention chunk requires staged execution")
    with tempfile.TemporaryDirectory(prefix="flint-bench-") as temporary:
        input_path = Path(temporary) / "input.json"
        input_path.write_text(json.dumps({
            "checkpoint": str(checkpoint), "prompt": prompt,
            "decode": decode, "repeats": repeats, "resident_workers": resident_workers,
            "resident_row_tile": resident_row_tile, "emit_predictions": emit_predictions,
            "staged_row_tile": staged_row_tile,
            **({"reuse_prefill": True} if reuse_prefill else {}),
            **({"mlx_attention": True} if mlx_attention else {}),
            **({"attention_chunk": attention_chunk} if attention_chunk is not None else {}),
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
    if mlx_attention:
        validate_mlx_attention(rows)
    if reuse_prefill:
        validate_reused_prefill(rows)
    return rows


def validate_mlx_attention(rows):
    if not rows or any(row.get("attention_implementation") != MLX_ATTENTION_IMPLEMENTATION
                       for row in rows):
        raise ValueError("native rows must label MLX-layout attention consistently")


def validate_reused_prefill(rows):
    if not rows or any(row.get("prefill_reused") is not True or row.get("prefill_samples") != 1
                       for row in rows):
        raise ValueError("native rows must label reused real prefill and its single sample")
    if len({row["prefill_seconds"] for row in rows}) != 1:
        raise ValueError("reused native prefill must report the same single measurement")


def mlx_prefill(model, mx, prompt, cache, chunk_size=None):
    """Actually evaluate every prefix token; keep full logits only in legacy mode."""
    if chunk_size is None:
        logits = model(mx.array([prompt], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        return logits
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("prefill chunk size must be positive")
    if not prompt:
        raise ValueError("cannot prefill an empty prompt")
    for start in range(0, len(prompt), chunk_size):
        logits = model(mx.array([prompt[start:start + chunk_size]], dtype=mx.int32), cache=cache)
        # Bound vocabulary-sized output storage as well as lazy graph growth.
        # Every chunk updates the same real KV cache at its actual positions.
        logits = logits[:, -1:, :]
        mx.eval(logits)
    return logits


def mlx_runs(checkpoint, prompt, decode, repeats, prefill_chunk_size=None):
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
            logits = mlx_prefill(model, mx, prompt, cache, prefill_chunk_size)
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
              staged_row_tile=1, prompt_tokens=None, reuse_prefill=False, mlx_attention=False):
    if mlx_attention and resident_workers:
        raise ValueError("mlx attention conflicts with resident attention")
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
    if mlx_attention:
        report["precision_note"] = (
            "FP32 activations, accumulation, and checkpoint weights are unchanged; "
            "MLX-layout mode changes floating-point reduction association. "
            "Checksums are diagnostics, not proof of logit or token-by-token parity."
        )
        report["native_execution"]["attention_implementation"] = MLX_ATTENTION_IMPLEMENTATION
    if prompt_tokens is not None:
        if prompt_tokens != len(prompt):
            raise ValueError("requested prompt length does not match actual input")
        report["prompt_protocol"] = LONG_PROMPT_NOTE
        report["context_length"] = len(prompt) + len(decode)
        report["mlx_prefill_chunk_size"] = LONG_PREFILL_CHUNK_SIZE
        report["prefill_note"] = (
            "Zig calls forward once per actual prompt token; stock MLX prefills all "
            "actual prompt tokens in chunks of at most 512, synchronizing every chunk "
            "and retaining only its final logit row. Prefill rates compare engine "
            "capabilities, not identical dispatches. No padded or fabricated KV cache."
        )
    if reuse_prefill:
        report["workload"] = (
            "Identical teacher-forced code token IDs; native restores the actual "
            "measured prefill KV cache and position before warmup/repeats; MLX creates "
            "and actually prefills a fresh KV cache each run. One cached forward plus "
            "synchronous greedy CPU readback per decode token. Model loading, cache "
            "snapshot/restore, dtype conversion, and warmup excluded from decode timing."
        )
        report["native_execution"]["reuse_prefill"] = True
        report["native_execution"]["prefill_samples"] = 1
        report["prefill_note"] += (
            " Native performs one measured real prefill, snapshots its actual KV cache "
            "and position, and restores that prefix outside decode timing for warmup/"
            "repeats. Its prefill rate has one sample, not repeated measurements. "
            "MLX still performs fresh actual prefill before every warmup/repeat."
        )
    encoded = json.dumps(report["input"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["input_sha256"] = hashlib.sha256(encoded).hexdigest()
    for engine in ENGINES:
        selected = [row for row in rows if row["engine"] == engine]
        validate_rows(selected, engine, repeats, resident_workers)
        if engine == "zig_fp32" and mlx_attention:
            validate_mlx_attention(selected)
        if engine == "zig_fp32" and reuse_prefill:
            validate_reused_prefill(selected)
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
            if mlx_attention:
                report["engines"][engine]["attention_implementation"] = MLX_ATTENTION_IMPLEMENTATION
            chunks = [row["attention_chunk"] for row in selected if "attention_chunk" in row]
            if chunks:
                if (len(chunks) != len(selected) or any(type(c) is not int or c not in ATTENTION_CHUNKS for c in chunks)
                        or len(set(chunks)) != 1):
                    raise ValueError("native runs must report one consistent attention chunk")
                report["native_execution"]["attention_chunk"] = chunks[0]
                report["engines"][engine]["attention_chunk"] = chunks
            if reuse_prefill:
                report["engines"][engine]["prefill_run_seconds"] = [selected[0]["prefill_seconds"]]
                report["engines"][engine]["prefill_samples"] = 1
                report["engines"][engine]["prefill_reused"] = True
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
    parser.add_argument("--prompt-tokens", type=positive_u32,
                        help="exact raw repeated-code prefix length; not the short chat/quality workload")
    parser.add_argument("--repeats", type=positive_u32, default=3)
    parser.add_argument("--reuse-prefill", action="store_true",
                        help="native only: measure one real prefill, then restore its actual KV prefix before repeats")
    parser.add_argument("--resident-workers", type=resident_worker_count, default=0,
                        help="opt into resident single-dispatch execution with 1..32 workers; default: staged")
    parser.add_argument("--resident-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per resident workgroup; nondefault requires --resident-workers")
    parser.add_argument("--output", type=Path, help="optional complete JSON report (outside checkpoint)")
    parser.add_argument("--staged-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per staged workgroup; nondefault conflicts with --resident-workers")
    parser.add_argument("--attention-chunk", type=int, choices=ATTENTION_CHUNKS,
                        help="staged attention chunk; 0 disables splitting; omitted uses native default")
    parser.add_argument("--mlx-attention", action="store_true", help="experimental MLX-layout native attention")
    args = parser.parse_args()
    if args.mlx_attention and (args.resident_workers or args.attention_chunk is not None):
        parser.error("--mlx-attention conflicts with resident/split attention")
    if args.attention_chunk is not None and args.resident_workers:
        parser.error("--attention-chunk requires staged execution")
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
    if args.prompt_tokens is None and not tokenizer.chat_template:
        parser.error("checkpoint has no local chat template")
    prompt, decode = workload(tokenizer, args.decode_tokens, args.prompt_tokens)
    rows = native_runs(binary, checkpoint, prompt, decode, args.repeats,
                       args.resident_workers, args.resident_row_tile,
                       staged_row_tile=args.staged_row_tile, reuse_prefill=args.reuse_prefill,
                       attention_chunk=args.attention_chunk, mlx_attention=args.mlx_attention)
    for row in rows:
        print(json.dumps(row), flush=True)
    del tokenizer
    gc.collect()
    rows.extend(mlx_runs(checkpoint, prompt, decode, args.repeats,
                         prefill_chunk_size=LONG_PREFILL_CHUNK_SIZE if args.prompt_tokens is not None else None))
    report = summarize(rows, prompt, decode, args.repeats,
                       args.resident_workers, args.resident_row_tile,
                       staged_row_tile=args.staged_row_tile, prompt_tokens=args.prompt_tokens,
                       reuse_prefill=args.reuse_prefill, mlx_attention=args.mlx_attention)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
