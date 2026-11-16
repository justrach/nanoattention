#!/usr/bin/env python3
"""Measure hybrid global-BF16 KV drift against the same native FP32-KV engine.

Run under an EXTERNAL resource/timeout guard holding the shared inference lock:
  python3 nanoattention/tools/guarded_run.py --lock EXISTING_LOCK --timeout 3600 -- \
    python3 nanoattention/tools/compare_bf16_kv.py CHECKPOINT --output result.json

This driver does not acquire a second lock, build anything, download a model, or
modify checkpoint files. Both native eval children run sequentially. Every input
token is evaluated; only final-prefill and teacher-forced continuation logits are
emitted. This is numerical comparison, NOT quality or free-running evaluation.
The candidate stores only global KV in BF16 (round-to-nearest); sliding KV and
activation/attention math remain FP32. It is not stock MLX BF16 inference.

The metrics module imports only NumPy and the standard library. Transformers and
the existing benchmark workload are imported lazily, only by the CLI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

MAX_CONTEXT = 262144
DEFAULT_GLOBAL_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48)


def compare_logits(reference, candidate, *, expected_shape=None):
    """Return strict-JSON metrics; malformed/nonfinite arrays have status failed.

    KL is D_KL(softmax(native FP32 KV) || softmax(hybrid BF16 KV)) in nats,
    calculated from shifted log-softmax in float64. No accuracy threshold is
    imposed. Relative L2 is undefined (null) for a nonzero error / zero baseline.
    """
    report = {"status": "failed", "issues": []}
    arrays = []
    for name, values in (("fp32", reference), ("bf16", candidate)):
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            report[ name ] = {"shape": None, "count": None, "finite_count": None}
            report["issues"].append(name + "_invalid_array")
            arrays.append(None)
            continue
        finite = int(np.count_nonzero(np.isfinite(array)))
        report[name] = {"shape": list(array.shape), "count": int(array.size),
                        "finite_count": finite, "nonfinite_count": int(array.size) - finite}
        if array.ndim != 2 or not array.size or array.shape[-1] < 2:
            report["issues"].append(name + "_invalid_shape")
        if finite != array.size:
            report["issues"].append(name + "_nonfinite_logits")
        if expected_shape is not None and array.shape != tuple(expected_shape):
            report["issues"].append(name + "_unexpected_shape")
        arrays.append(array)
    if all(array is not None for array in arrays) and arrays[0].shape != arrays[1].shape:
        report["issues"].append("shape_mismatch")
    if report["issues"]:
        return report
    ref, cand = arrays
    delta = cand - ref
    # Native logits are FP32. Float64 keeps the squared-error and softmax
    # calculations well inside range even for extreme finite FP32 values.
    ref_norm = float(np.linalg.norm(ref))
    error_norm = float(np.linalg.norm(delta))
    relative_l2 = error_norm / ref_norm if ref_norm else (0.0 if error_norm == 0 else None)
    rows = []
    for row, (left, right) in enumerate(zip(ref, cand)):
        left_top, right_top = int(np.argmax(left)), int(np.argmax(right))
        left_pair, right_pair = np.partition(left, -2)[-2:], np.partition(right, -2)[-2:]
        left_shift, right_shift = left - np.max(left), right - np.max(right)
        left_log = left_shift - np.log(np.sum(np.exp(left_shift), dtype=np.float64))
        right_log = right_shift - np.log(np.sum(np.exp(right_shift), dtype=np.float64))
        kl = float(np.sum(np.exp(left_log) * (left_log - right_log), dtype=np.float64))
        # KL is nonnegative analytically; remove only floating roundoff below 0.
        kl = max(0.0, kl)
        rows.append({"row": row, "fp32_top1": left_top, "bf16_top1": right_top,
                     "top1_agrees": left_top == right_top,
                     "fp32_top1_margin": float(left_pair[-1] - left_pair[-2]),
                     "bf16_top1_margin": float(right_pair[-1] - right_pair[-2]),
                     "softmax_kl_nats": kl})
    report.update(status="measured", max_absolute_logit_error=float(np.max(np.abs(delta))),
                  rms_logit_error=float(np.sqrt(np.mean(delta * delta, dtype=np.float64))),
                  relative_l2_logit_error=relative_l2,
                  relative_l2_note="null means nonzero error with zero FP32 reference norm",
                  top1_agreement_count=sum(row["top1_agrees"] for row in rows),
                  top1_agreement_fraction=sum(row["top1_agrees"] for row in rows) / len(rows),
                  mean_softmax_kl_nats=float(np.mean([row["softmax_kl_nats"] for row in rows])),
                  kl_direction="FP32 KV || hybrid BF16 KV; float64 log-softmax; negative roundoff clamped to zero",
                  rows=rows)
    return report


def analytic_kv_allocation(config, context):
    """Mirror native per-layer cache capacities, not process RSS or live usage."""
    defaults = {"num_hidden_layers": 49, "num_key_value_heads": 4, "head_dim": 128,
                "sliding_window": 513, "max_position_embeddings": MAX_CONTEXT,
                "vocab_size": 150144}
    dims = {key: config.get(key, default) for key, default in defaults.items()}
    if any(type(value) is not int or value <= 0 for value in dims.values()):
        raise ValueError("invalid configuration dimensions")
    if type(context) is not int or not 0 < context <= min(MAX_CONTEXT, dims["max_position_embeddings"]):
        raise ValueError("invalid context length")
    default_types = ["full_attention" if i in DEFAULT_GLOBAL_LAYERS else "sliding_attention"
                     for i in range(49)]
    types = config.get("layer_types", default_types)
    if (not isinstance(types, list) or len(types) != dims["num_hidden_layers"]
            or any(kind not in ("full_attention", "sliding_attention") for kind in types)):
        raise ValueError("invalid layer types")
    sliding = types.count("sliding_attention")
    if sliding and config.get("use_sliding_window", True) is not True:
        raise ValueError("sliding layers require sliding-window support")
    global_layers = len(types) - sliding
    # Include both K and V; dense-MLP layers still allocate attention caches.
    components = 2 * dims["num_key_value_heads"] * dims["head_dim"]
    global_elements = components * context * global_layers
    sliding_elements = components * min(context, dims["sliding_window"]) * sliding
    fp32 = global_elements * 4 + sliding_elements * 4
    hybrid = global_elements * 2 + sliding_elements * 4
    return {"scope": "analytic allocation of K+V buffers only; not RSS, measured memory, or total engine memory",
            "dimensions": dims, "global_layers": global_layers, "sliding_layers": sliding,
            "global_capacity_per_layer": context,
            "sliding_capacity_per_layer": min(context, dims["sliding_window"]),
            "fp32": {"global_bytes": global_elements * 4, "sliding_bytes": sliding_elements * 4,
                     "total_bytes": fp32},
            "hybrid_bf16": {"global_bytes": global_elements * 2, "sliding_bytes": sliding_elements * 4,
                            "total_bytes": hybrid},
            "saved_bytes": fp32 - hybrid}


def eval_command(binary, checkpoint, tokens_file, context, logits_from, *, bf16=False):
    command = [str(binary), "eval", str(checkpoint), "--tokens-file", str(tokens_file),
               "--mlx-attention", "1", "--staged-row-tile", "2",
               "--context", str(context), "--logits-from", str(logits_from)]
    if bf16:
        command += ["--kv-bf16", "1"]
    return command


def parse_logits(stdout):
    """Keep every emitted vocabulary row; reject non-array or malformed output."""
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, list) or any(type(x) not in (int, float) for x in row):
            raise ValueError("native output must contain numeric logit arrays")
        rows.append(row)
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path,
                        default=Path(__file__).resolve().parents[1] / "zig-out/bin/flint")
    parser.add_argument("--prompt-tokens", type=positive_int, default=16384)
    parser.add_argument("--decode-tokens", type=positive_int, default=16)
    parser.add_argument("--native-timeout", type=positive_int,
                        help="seconds per child; default scales with full evaluated length (minimum 180)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    full_len = args.prompt_tokens + args.decode_tokens
    if full_len > MAX_CONTEXT:
        parser.error("prompt plus decode exceeds native maximum context 262144")
    checkpoint, binary, output = args.checkpoint.resolve(), args.binary.resolve(), args.output.resolve()
    if output == checkpoint or checkpoint in output.parents:
        parser.error("output must be outside the read-only checkpoint directory")
    if output == binary:
        parser.error("output must not overwrite the native binary")
    # Set offline mode before importing Transformers or benchmark helpers.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    timeout = args.native_timeout or max(180, (full_len + 15) // 16 + 60)
    report = {"status": "failed", "protocol": "native FP32 KV versus hybrid global BF16 KV",
              "scope": "numerical comparison only; no quality, free-run, or accuracy-pass claim",
              "precision": "global KV BF16 round-to-nearest; sliding KV FP32; same FP32 math and checkpoint weights",
              "binary": {"name": binary.name},
              "prompt_tokens": args.prompt_tokens, "teacher_forced_decode_tokens": args.decode_tokens,
              "evaluated_input_tokens": full_len, "native_timeout_seconds": timeout,
              "execution": {"order": ["fp32", "bf16"], "mlx_attention": True, "staged_row_tile": 2},
              "positions": {"start": args.prompt_tokens - 1, "end_inclusive": full_len - 1,
                            "count": args.decode_tokens + 1,
                            "note": "zero-based input positions; final prefill plus every forced decode input, including the final next-token prediction"},
              "prompt_protocol": "benchmark.workload exact repeated code-token prefix; all prefix tokens really evaluated; no fabricated cache"}
    phase = "local_setup"
    try:
        report["binary"]["sha256"] = file_sha256(binary)
        config = json.loads((checkpoint / "config.json").read_text())
        report["kv_allocation"] = analytic_kv_allocation(config, full_len)
        phase = "offline_tokenizer"
        from transformers import AutoTokenizer
        from benchmark import workload
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                                  trust_remote_code=False, fix_mistral_regex=False)
        prompt, decode = workload(tokenizer, args.decode_tokens, args.prompt_tokens)
        if len(prompt) != args.prompt_tokens or len(decode) != args.decode_tokens:
            raise ValueError("workload token count mismatch")
        payload = json.dumps(prompt + decode, separators=(",", ":"))
        report["input_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        arrays = []
        with tempfile.TemporaryDirectory(prefix="flint-kv-comparison-") as temp:
            tokens_file = Path(temp) / "input.json"
            tokens_file.write_text(payload)
            for bf16 in (False, True):
                phase = "bf16_eval" if bf16 else "fp32_eval"
                result = subprocess.run(eval_command(binary, checkpoint, tokens_file, full_len,
                                                     len(prompt) - 1, bf16=bf16),
                                        text=True, capture_output=True, check=True, timeout=timeout)
                arrays.append(parse_logits(result.stdout))
        phase = "metrics"
        report["logits"] = compare_logits(*arrays, expected_shape=(args.decode_tokens + 1,
                                                                 report["kv_allocation"]["dimensions"]["vocab_size"]))
        report["status"] = report["logits"]["status"]
    except subprocess.TimeoutExpired:
        report["failure"] = {"phase": phase, "reason": "native_timeout"}
    except subprocess.CalledProcessError as error:
        report["failure"] = {"phase": phase, "reason": "native_nonzero_exit", "returncode": error.returncode}
    except Exception:
        # Never copy exception text, native stderr, raw logits/prompts, or local
        # checkpoint paths into a shareable report (including failed runs).
        report["failure"] = {"phase": phase, "reason": "invalid_local_input_or_output"}
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.write_text(serialized)
    print(serialized, end="")
    return 0 if report["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
