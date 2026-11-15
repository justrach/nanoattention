#!/usr/bin/env python3
"""Optional local full-checkpoint check against an unsteered MLX FP32 baseline.

Run under guarded_run.py. Native inference finishes before MLX loads the model;
no checkpoints are modified, converted, or downloaded. Requires mlx-lm locally.

--prompt-tokens uses the benchmark's exact-length raw repeated-code prefix, not
--prompt or a quality/chat evaluation. All prefix tokens are really evaluated;
only the last prefill row and cached decode rows are compared, and stock MLX
prefills in 256-token chunks to bound vocabulary-sized logit output storage.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from benchmark import ATTENTION_CHUNKS, LONG_PROMPT_NOTE, code_prompt, mlx_prefill, positive_u32

LONG_PREFILL_CHUNK_SIZE = 256

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
    parser.add_argument("--decode-tokens", type=positive_u32, default=16)
    parser.add_argument("--native-timeout", type=positive_u32,
                        help="seconds per native child; default scales with actual prefix length, minimum 180")
    parser.add_argument("--prompt-tokens", type=positive_u32,
                        help="exact raw repeated-code prefix length; overrides --prompt, not quality evaluation")
    parser.add_argument("--mlx-prefill-chunk-size", type=positive_u32,
                        help="long-context reference prefill batch size (default 256); 1 matches native sequential prefill")
    parser.add_argument("--resident-workers", type=resident_worker_count, default=0,
                        help="opt into resident single-dispatch execution with 1..32 workers; default: staged")
    parser.add_argument("--resident-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per resident workgroup; nondefault requires --resident-workers")
    parser.add_argument("--staged-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per staged workgroup; nondefault conflicts with --resident-workers")
    parser.add_argument("--attention-chunk", type=int, choices=ATTENTION_CHUNKS,
                        help="staged attention chunk; 0 disables splitting; omitted uses native default")
    parser.add_argument("--mlx-attention", action="store_true", help="experimental MLX-layout native attention")
    args = parser.parse_args()
    if args.mlx_attention and (args.resident_workers or args.attention_chunk is not None):
        parser.error("--mlx-attention conflicts with resident/split attention")
    if args.mlx_prefill_chunk_size is not None and args.prompt_tokens is None:
        parser.error("--mlx-prefill-chunk-size requires --prompt-tokens")
    mlx_chunk_size = args.mlx_prefill_chunk_size or LONG_PREFILL_CHUNK_SIZE
    if args.attention_chunk is not None and args.resident_workers:
        parser.error("--attention-chunk requires staged execution")
    if args.resident_row_tile != 1 and not args.resident_workers:
        parser.error("--resident-row-tile requires --resident-workers")
    if args.staged_row_tile != 1 and args.resident_workers:
        parser.error("nondefault --staged-row-tile conflicts with --resident-workers")
    resident_flags = (["--resident-workers", str(args.resident_workers),
                       "--resident-row-tile", str(args.resident_row_tile)]
                      if args.resident_workers else [])
    resident_flags += ["--staged-row-tile", str(args.staged_row_tile)]
    if args.attention_chunk is not None:
        resident_flags += ["--attention-chunk", str(args.attention_chunk)]
    if args.mlx_attention:
        resident_flags += ["--mlx-attention", "1"]
    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, trust_remote_code=False, fix_mistral_regex=False)
    if args.prompt_tokens is None:
        tokenizer.chat_template = (args.checkpoint / "chat_template.jinja").read_text()
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=True, add_generation_prompt=True, return_dict=False)
    else:
        tokens = code_prompt(tokenizer, args.prompt_tokens)
    logits_from = len(tokens) - 1 if args.prompt_tokens is not None else 0
    native_timeout = args.native_timeout or max(180, (len(tokens) + 15) // 16 + 60)
    with tempfile.TemporaryDirectory(prefix="flint-mlx-comparison-") as temp:
        prompt = Path(temp) / "prompt.json"
        prompt.write_text(json.dumps(tokens))
        context = max(32, len(tokens) + args.decode_tokens)
        generated_result = subprocess.run([
            str(args.binary.resolve()), "generate", str(args.checkpoint.resolve()),
            "--tokens-file", str(prompt), "--context", str(context),
            "--max-tokens", str(args.decode_tokens), *resident_flags,
        ], text=True, stdout=subprocess.PIPE, check=True, timeout=native_timeout)
        generated = json.loads(generated_result.stdout)["tokens"]
        # Replay the native continuation so every cached decode position has
        # a logit reference, without keeping both full engines resident.
        prompt.write_text(json.dumps(tokens + generated[:-1]))
        result = subprocess.run([
            str(args.binary.resolve()), "eval", str(args.checkpoint.resolve()),
            "--tokens-file", str(prompt), "--context", str(context), *resident_flags,
            *(["--logits-from", str(logits_from)] if args.prompt_tokens is not None else []),
        ], text=True, stdout=subprocess.PIPE, check=True, timeout=native_timeout)
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
    logits = mlx_prefill(model, mx, tokens, cache,
                         mlx_chunk_size if args.prompt_tokens is not None else None)
    parts = [np.array(logits[0], dtype=np.float32)]
    mlx_generated = [int(mx.argmax(logits[0, -1]).item())] if generated else []
    for token in generated[:-1]:
        logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        parts.append(np.array(logits[0], dtype=np.float32))
        mlx_generated.append(int(mx.argmax(logits[0, -1]).item()))
    reference = np.concatenate(parts, axis=0)
    expected_positions = len(tokens) + max(0, len(generated) - 1) - logits_from
    if native.shape != reference.shape or len(reference) != expected_positions:
        raise ValueError("native/MLX logit shapes or validated position count do not match")
    delta = native - reference
    report = {
        "native_execution": {
            "mode": "resident_single_dispatch" if args.resident_workers else "staged",
            "resident_workers": args.resident_workers,
            "resident_row_tile": args.resident_row_tile,
            "staged_row_tile": args.staged_row_tile,
        },
        "precision_note": ("FP32 activations, accumulation, and checkpoint weights are unchanged; MLX-layout mode changes floating-point reduction association."
                           if args.mlx_attention else "Execution mode changes scheduling, not the native FP32 math baseline."),
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
    if args.mlx_attention:
        report["native_execution"]["attention_implementation"] = "mlx_layout_128_stripes"
    if args.attention_chunk is not None:
        report["native_execution"]["attention_chunk"] = args.attention_chunk
    if args.prompt_tokens is not None:
        report.update({
            "prompt_protocol": LONG_PROMPT_NOTE,
            "context_length": context,
            "validated_position_range": {
                "start": logits_from, "end_inclusive": logits_from + len(reference) - 1,
                "indexing": "zero-based absolute input token position",
            },
            "mlx_prefill_chunk_size": mlx_chunk_size,
            "native_timeout_seconds": native_timeout,
            "reference_prefill_mode": "sequential" if mlx_chunk_size == 1 else "batched_chunks",
            "prefill_note": (
                "Native and MLX actually prefill every prefix token; native eval emits "
                "only --logits-from onward, and MLX retains only the final prefill row "
                "plus cached decode rows. Earlier prefill logits are not validated."
            ),
        })
    print(json.dumps(report, indent=2), flush=True)
    np.testing.assert_allclose(native, reference, rtol=2e-3, atol=2e-3)
    assert generated == mlx_generated, (generated, mlx_generated)
    assert report["last_native_top1"] == report["last_mlx_top1"]
    compared = "last-prefill and cached decode logits" if args.prompt_tokens is not None else "prefill, cached decode logits"
    print("PASS: native quantized Flint {}, and greedy tokens agree with unsteered FP32 MLX".format(compared))


if __name__ == "__main__":
    main()
