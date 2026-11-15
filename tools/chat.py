#!/usr/bin/env python3
"""Local tokenizer/chat-template adapter; inference itself is native Zig + Metal.

Requires transformers/tokenizers in the selected Python environment. Does not
load an MLX/PyTorch model, download files, start a server, or change checkpoints.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--binary", type=Path, default=Path(__file__).resolve().parents[1] / "zig-out/bin/flint")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--resident-workers", type=int, choices=range(1, 33), help="opt into the Lean-plan single-kernel path (1..32 workers)")
    parser.add_argument("--resident-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per resident workgroup; nondefault requires --resident-workers")
    parser.add_argument("--staged-row-tile", type=int, choices=(1, 2, 4, 8), default=1,
                        help="rows per staged workgroup; nondefault conflicts with --resident-workers")
    args = parser.parse_args()
    if args.resident_row_tile != 1 and args.resident_workers is None:
        parser.error("--resident-row-tile requires --resident-workers")
    if args.staged_row_tile != 1 and args.resident_workers:
        parser.error("nondefault --staged-row-tile conflicts with --resident-workers")
    resident_flags = ([] if args.resident_workers is None else
                      ["--resident-workers", str(args.resident_workers),
                       "--resident-row-tile", str(args.resident_row_tile)])
    resident_flags += ["--staged-row-tile", str(args.staged_row_tile)]
    print("execution={} workers={} resident_row_tile={} staged_row_tile={}".format(
        "resident_single_dispatch" if args.resident_workers else "staged",
        args.resident_workers or 0, args.resident_row_tile, args.staged_row_tile), file=sys.stderr)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, trust_remote_code=False, fix_mistral_regex=False)
    template = args.checkpoint / "chat_template.jinja"
    if not template.is_file():
        raise SystemExit("The checkpoint's chat_template.jinja is required")
    tokenizer.chat_template = template.read_text()
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    with tempfile.TemporaryDirectory(prefix="flint-zig-prompt-") as temp:
        prompt = Path(temp) / "prompt.json"
        prompt.write_text(json.dumps(tokens))
        result = subprocess.run([
            str(args.binary.resolve()), "generate", str(args.checkpoint.resolve()),
            "--tokens-file", str(prompt), "--max-tokens", str(args.max_tokens),
            "--context", str(args.context), *resident_flags,
        ], text=True, stdout=subprocess.PIPE, timeout=args.timeout, check=True)
    generated = json.loads(result.stdout)["tokens"]
    print(tokenizer.decode(generated, skip_special_tokens=False))


if __name__ == "__main__":
    main()
