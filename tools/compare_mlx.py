#!/usr/bin/env python3
"""Optional local full-checkpoint check against an unsteered MLX FP32 baseline.

Run under guarded_run.py. Native inference finishes before MLX loads the model;
no checkpoints are modified, converted, or downloaded. Requires mlx-lm locally.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def resident_worker_count(value):
    number = int(value)
    if not 1 <= number <= 32:
        raise argparse.ArgumentTypeError("must be in 1..32 (omit for staged execution)")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "zig-out/bin/flint")
    parser.add_argument("--prompt", default="What is 2+2? Reply briefly.")
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--resident-workers", type=resident_worker_count, default=0,
                        help="opt into resident single-dispatch execution with 1..32 workers; default: staged")
    parser.add_argument("--resident-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per resident workgroup; nondefault requires --resident-workers")
    parser.add_argument("--staged-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per staged workgroup; nondefault conflicts with --resident-workers")
    args = parser.parse_args()
    if args.resident_row_tile != 1 and not args.resident_workers:
        parser.error("--resident-row-tile requires --resident-workers")
    if args.staged_row_tile != 1 and args.resident_workers:
        parser.error("nondefault --staged-row-tile conflicts with --resident-workers")
    resident_flags = (["--resident-workers", str(args.resident_workers),
                       "--resident-row-tile", str(args.resident_row_tile)]
                      if args.resident_workers else [])
    resident_flags += ["--staged-row-tile", str(args.staged_row_tile)]
    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, trust_remote_code=False, fix_mistral_regex=False)
    tokenizer.chat_template = (args.checkpoint / "chat_template.jinja").read_text()
    tokens = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)
    with tempfile.TemporaryDirectory(prefix="flint-mlx-comparison-") as temp:
        prompt = Path(temp) / "prompt.json"
        prompt.write_text(json.dumps(tokens))
        context = max(32, len(tokens) + args.decode_tokens)
        generated_result = subprocess.run([
            str(args.binary.resolve()), "generate", str(args.checkpoint.resolve()),
            "--tokens-file", str(prompt), "--context", str(context),
            "--max-tokens", str(args.decode_tokens), *resident_flags,
        ], text=True, stdout=subprocess.PIPE, check=True, timeout=180)
        generated = json.loads(generated_result.stdout)["tokens"]
        # Replay the native continuation so every cached decode position has
        # a logit reference, without keeping both full engines resident.
        prompt.write_text(json.dumps(tokens + generated[:-1]))
        result = subprocess.run([
            str(args.binary.resolve()), "eval", str(args.checkpoint.resolve()),
            "--tokens-file", str(prompt), "--context", str(context), *resident_flags,
        ], text=True, stdout=subprocess.PIPE, check=True, timeout=180)
        native = np.array([json.loads(line) for line in result.stdout.splitlines()], dtype=np.float32)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    mx.set_memory_limit(32 * 1024 ** 3)
    mx.set_cache_limit(512 * 1024 ** 2)
    model, _ = load(str(args.checkpoint), tokenizer_config={"fix_mistral_regex": False})
    # Match the native FP32 activation/accumulation baseline. Quantized integer
    # weights retain their exact packed values; floating scales/norms are widened.
    model.set_dtype(mx.float32)
    cache = make_prompt_cache(model)
    logits = model(mx.array([tokens], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    parts = [np.array(logits[0], dtype=np.float32)]
    mlx_generated = [int(mx.argmax(logits[0, -1]).item())] if generated else []
    for token in generated[:-1]:
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        parts.append(np.array(logits[0], dtype=np.float32))
        mlx_generated.append(int(mx.argmax(logits[0, -1]).item()))
    reference = np.concatenate(parts, axis=0)
    delta = native - reference
    report = {
        "native_execution": {
            "mode": "resident_single_dispatch" if args.resident_workers else "staged",
            "resident_workers": args.resident_workers,
            "resident_row_tile": args.resident_row_tile,
            "staged_row_tile": args.staged_row_tile,
        },
        "precision_note": "Execution mode changes scheduling, not the native FP32 math baseline.",
        "prompt_tokens": len(tokens),
        "generated_tokens": len(generated),
        "evaluated_positions": len(reference),
        "greedy_decode_agreement": sum(a == b for a, b in zip(generated, mlx_generated)),
        "max_absolute_logit_error": float(np.max(np.abs(delta))),
        "rms_logit_error": float(np.sqrt(np.mean(delta * delta))),
        "top1_agreement": int(np.sum(np.argmax(native, axis=-1) == np.argmax(reference, axis=-1))),
        "last_native_top1": int(native[-1].argmax()),
        "last_mlx_top1": int(reference[-1].argmax()),
    }
    print(json.dumps(report, indent=2), flush=True)
    np.testing.assert_allclose(native, reference, rtol=2e-3, atol=2e-3)
    assert generated == mlx_generated, (generated, mlx_generated)
    assert report["last_native_top1"] == report["last_mlx_top1"]
    print("PASS: native quantized Flint prefill, cached decode logits, and greedy tokens agree with unsteered FP32 MLX")


if __name__ == "__main__":
    main()
