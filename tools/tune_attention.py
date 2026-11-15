#!/usr/bin/env python3
"""Prediction-gated attention tuning on ONE shared, genuinely computed code prefix.

Must run under an external resource/timeout guard, from nanoattention/:
    python3 tools/guarded_run.py --timeout 900 -- \\
        python3 tools/tune_attention.py CHECKPOINT --prompt-tokens 4096

One native process prefills the actual benchmark.workload prefix with chunk 0,
then snapshots its real KV cache and position. Every candidate warms up at most
8 decode tokens and restores that prefix outside each decode timer. This only
tunes cached decode from shared real baseline prefill, NOT full optimized prefill
or MLX logit verification. No MLX model is imported, no defaults are changed, and
no checkpoint files are written. The external guard owns resource limits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

import benchmark

PREFILL_NOTE = (
    "Shared real baseline prefill: chunk 0 actually evaluates every code-prefix "
    "token once; its actual KV cache and position are restored outside decode "
    "timing for every candidate/repeat. Prefill has one shared sample, not one "
    "sample per chunk/repeat. This is not full optimized prefill or MLX logit verification."
)


def summarize_candidates(rows, chunks, repeats, decode_tokens):
    """Gate each candidate on complete repeats and baseline per-position parity."""
    grouped = {chunk: [] for chunk in chunks}
    for row in rows:
        chunk = row.get("attention_chunk")
        if type(chunk) is not int or chunk not in grouped:
            raise ValueError("native returned an unexpected or missing attention_chunk")
        grouped[chunk].append(row)
    benchmark.validate_reused_prefill(rows)
    # All candidates must refer to the same single GPU-prefill sample, too.
    gpu_prefill = [row["prefill_gpu_seconds"] for row in rows if "prefill_gpu_seconds" in row]
    if gpu_prefill and (len(gpu_prefill) != len(rows) or len(set(gpu_prefill)) != 1):
        raise ValueError("shared baseline GPU prefill measurement changed")
    candidates, baseline_predictions = {}, None
    for chunk in chunks:
        selected = grouped[chunk]
        candidate = candidates[str(chunk)] = {"attention_chunk": chunk, "status": "rejected"}
        try:
            benchmark.validate_rows(selected, "zig_fp32", repeats)
            for row in selected:
                if row.get("error"):
                    raise ValueError("native row reports an error")
                actual = row.get("predictions")
                if (not isinstance(actual, list) or len(actual) != decode_tokens or
                        any(type(t) is not int or not 0 <= t <= 0xFFFFFFFF for t in actual)):
                    raise ValueError("missing or invalid per-position predictions")
                if sum(actual) != row["checksum"]:
                    raise ValueError("prediction checksum does not match per-position predictions")
                if baseline_predictions is None:
                    # chunks is normalized to put the unsplit baseline first.
                    baseline_predictions = actual
                if actual != baseline_predictions:
                    raise ValueError("per-position predictions differ from chunk 0 baseline/repeat")
            seconds = statistics.median(row["decode_seconds"] for row in selected)
            candidate.update(status="valid", median_decode_seconds=seconds,
                             decode_tokens_per_second=decode_tokens / seconds,
                             decode_run_seconds=[row["decode_seconds"] for row in selected],
                             prefill_reused=True, prefill_samples=1)
            # Preserve all observed counters, including future native counters,
            # in the top-level runs rather than inferring them from the mode.
            if all("decode_gpu_seconds" in row for row in selected):
                candidate["median_decode_gpu_seconds"] = statistics.median(
                    row["decode_gpu_seconds"] for row in selected)
        except ValueError as exc:
            candidate["error"] = str(exc)
            if chunk == 0:
                raise ValueError("chunk 0 baseline failed validation: {}".format(exc)) from exc
    baseline = candidates["0"]["median_decode_seconds"]
    ranked = sorted((c for c in candidates.values() if c["status"] == "valid"),
                    key=lambda c: c["median_decode_seconds"])
    for candidate in ranked:
        candidate["decode_speedup_over_baseline"] = baseline / candidate["median_decode_seconds"]
        candidate["throughput_improvement_fraction"] = candidate["decode_speedup_over_baseline"] - 1
    # A timing win of <=3% is not admitted; the baseline itself is not a new win.
    admitted = [c for c in ranked if c["attention_chunk"] != 0 and
                c["decode_speedup_over_baseline"] > 1.03]
    return {"candidates": candidates, "timing_rank": [c["attention_chunk"] for c in ranked],
            "best_admitted": dict(admitted[0]) if admitted else None,
            "admission_rule": "Exact baseline/repeat per-position predictions and >3% decode throughput improvement over chunk 0"}


def stream_native(binary, payload, on_row):
    """One owned child; outer guarded_run.py bounds time, memory, and descendants."""
    with tempfile.TemporaryDirectory(prefix="flint-attention-tune-") as temporary:
        input_path = Path(temporary) / "input.json"
        input_path.write_text(json.dumps(payload), encoding="utf-8")
        child = subprocess.Popen([str(binary), str(input_path)], text=True,
                                 stdout=subprocess.PIPE, bufsize=1)
        try:
            for line in child.stdout:
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("native output must contain JSON objects")
                    on_row(row)
            code = child.wait()
            if code:
                raise subprocess.CalledProcessError(code, [str(binary), "INPUT_JSON"])
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            child.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/flint-bench"))
    parser.add_argument("--prompt-tokens", type=benchmark.positive_u32, required=True,
                        help="required actual raw code-prefix length, not synthetic cache positions")
    parser.add_argument("--decode-tokens", type=benchmark.positive_u32, default=64)
    parser.add_argument("--repeats", type=benchmark.positive_u32, default=3)
    parser.add_argument("--chunks", type=int, nargs="+", choices=benchmark.ATTENTION_CHUNKS,
                        default=list(benchmark.ATTENTION_CHUNKS),
                        help="chunk 0 baseline is always inserted first; duplicates are removed")
    parser.add_argument("--staged-row-tile", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--output", type=Path, default=Path(".zig-cache/attention-tune.json"))
    args = parser.parse_args()
    if os.environ.get("FLINT_PROFILE") == "1":
        parser.error("unset FLINT_PROFILE: profiling throughput is not a benchmark")
    checkpoint, binary, output = (p.resolve() for p in (args.checkpoint, args.binary, args.output))
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    for path in (output, temporary_output.resolve()):
        if path == checkpoint or checkpoint in path.parents:
            parser.error("--output and its temporary file must not write inside the checkpoint directory")
    if not checkpoint.is_dir() or not binary.is_file():
        parser.error("checkpoint and compiled benchmark must exist")
    chunks = list(dict.fromkeys([0, *args.chunks]))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                              trust_remote_code=False, fix_mistral_regex=False)
    prompt, decode = benchmark.workload(tokenizer, args.decode_tokens, args.prompt_tokens)
    del tokenizer
    payload = {"checkpoint": str(checkpoint), "prompt": prompt, "decode": decode,
               "repeats": args.repeats, "resident_workers": 0, "resident_row_tile": 1,
               "staged_row_tile": args.staged_row_tile, "attention_candidates": chunks,
               "reuse_prefill": True, "emit_predictions": True}
    report = {"status": "running", "prefill_note": PREFILL_NOTE,
              "prompt_protocol": benchmark.LONG_PROMPT_NOTE,
              "prompt_tokens": len(prompt), "decode_tokens": len(decode),
              "context_length": len(prompt) + len(decode), "repeats": args.repeats,
              "warmup_decode_tokens_per_candidate": min(8, len(decode)),
              "native_execution": {"mode": "staged", "staged_row_tile": args.staged_row_tile,
                                   "reuse_prefill": True, "prefill_samples": 1,
                                   "baseline_attention_chunk": 0},
              "attention_candidates": chunks, "input": {"prompt": prompt, "decode": decode},
              "runs": [], "best_admitted": None,
              "mlx_logit_verification": "not performed", "full_optimized_prefill": "not measured"}
    encoded = json.dumps(report["input"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["input_sha256"] = hashlib.sha256(encoded).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary_output.replace(output)

    def on_row(row):
        # Reject non-finite/invalid JSON before it can poison the durable report,
        # so a malformed native line still leaves a writable failure report.
        json.dumps(row, allow_nan=False)
        chunk = row.get("attention_chunk")
        if type(chunk) is not int or chunk not in chunks:
            raise ValueError("unexpected or missing native attention_chunk")
        selected = [r for r in report["runs"] if r["attention_chunk"] == chunk]
        if len(selected) >= args.repeats:
            raise ValueError("native returned too many repeats for attention chunk")
        report["runs"].append(row)
        save()
        print(json.dumps({"status": "running", "prefill_note": PREFILL_NOTE,
                          "completed_runs": len(report["runs"]),
                          "expected_runs": len(chunks) * args.repeats, "run": row}, allow_nan=False), flush=True)

    save()
    try:
        stream_native(binary, payload, on_row)
        report.update(summarize_candidates(report["runs"], chunks, args.repeats, len(decode)))
        report["shared_baseline_prefill_seconds"] = report["runs"][0]["prefill_seconds"]
        report["status"] = "native_decode_tuning_complete"
        save()
    except BaseException as exc:
        report.update(status="failed", best_admitted=None,
                      error="{}: {}".format(type(exc).__name__, exc))
        save()
        raise
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
