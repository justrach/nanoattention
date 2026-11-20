#!/usr/bin/env python3
"""LFM3 text front end: chat template, native JSONL session, greedy reply; optional stock MLX.

Renders the checkpoint's own chat_template.jinja (thinking off unless --thinking, as
mlx-vlm does), sends {"id","tokens","max_tokens","reset"} lines to one persistent
`nano session CHECKPOINT ...` process, and decodes the generated ids. Generation is
whatever the session does (greedy). --mlx also runs the same prompt tokens through stock
mlx-vlm greedy after the native process has exited (never both models at once), and
reports whether the two token streams match. --no-native runs stock MLX alone.

--interactive keeps one session across turns and sends only the new suffix tokens when the
re-rendered conversation extends the cached history token for token; when it does not (the
template drops earlier turns' empty <think></think>, for example), it says so and resets.

Needs transformers/tokenizers; --mlx needs mlx and mlx-vlm (the reference venv).
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EOS = 124900


class Session:
    """One native `session` child; requests and responses are single JSON lines."""

    def __init__(self, binary, checkpoint, extra, timeout):
        command = [str(binary), "session", str(checkpoint), *extra]
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.timeout, self.next_id = timeout, 1

    def request(self, tokens, max_tokens, reset):
        req = {"id": self.next_id, "tokens": tokens, "max_tokens": max_tokens, "reset": reset}
        self.next_id += 1
        start = time.perf_counter()
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit("native session exited (status %s)" % self.proc.wait(timeout=self.timeout))
        response = json.loads(line)
        if response.get("id") != req["id"]:
            raise SystemExit("session answered id %r for request %r" % (response.get("id"), req["id"]))
        response["wall_seconds"] = time.perf_counter() - start
        return response

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def load_tokenizer(checkpoint):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    template = pathlib.Path(checkpoint) / "chat_template.jinja"
    if not template.is_file():
        raise SystemExit("the checkpoint's chat_template.jinja is required")
    tok.chat_template = template.read_text()
    return tok


def render(tok, messages, thinking):
    return list(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_dict=False, enable_thinking=thinking))


def reply_text(tok, ids):
    body = ids[:-1] if ids and ids[-1] == EOS else ids
    return tok.decode(body, skip_special_tokens=False)


def mlx_greedy(checkpoint, tokens, max_tokens, dtype):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from lfm3_reference import StockMLX
    stock = StockMLX(checkpoint, dtype)
    cache = stock.cache()
    start = time.perf_counter()
    _, logits = stock.run(tokens, cache, [len(tokens) - 1], keep_hidden=False)
    prefilled = time.perf_counter()
    out = stock.greedy(logits[0], cache, [EOS], max_tokens)
    return out, prefilled - start, time.perf_counter() - prefilled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("prompt", nargs="?", help="user message (omit with --interactive)")
    parser.add_argument("--system", help="optional system message")
    parser.add_argument("--binary", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[1] / "zig-out/bin/nano")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--thinking", action="store_true", help="render with enable_thinking=True")
    parser.add_argument("--session-arg", action="append", default=[],
                        help="extra argument for `nano session` (repeatable), e.g. --session-arg=--context --session-arg=8192")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--mlx", action="store_true", help="also run stock MLX on the same tokens, afterwards")
    parser.add_argument("--no-native", action="store_true", help="stock MLX only")
    parser.add_argument("--mlx-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--interactive", action="store_true", help="multi-turn on one session; empty line quits")
    parser.add_argument("--json", action="store_true", help="print a JSON record instead of the reply text")
    args = parser.parse_args()
    if not args.interactive and not args.prompt:
        parser.error("a prompt is required unless --interactive")
    if args.interactive and (args.no_native or args.mlx):
        parser.error("--interactive drives the native session only")
    tok = load_tokenizer(args.checkpoint)
    messages = [{"role": "system", "content": args.system}] if args.system else []

    if args.interactive:
        session = Session(args.binary.resolve(), args.checkpoint.resolve(), args.session_arg, args.timeout)
        history = []
        try:
            while True:
                try:
                    text = input("> ")
                except EOFError:
                    break
                if not text.strip():
                    break
                messages.append({"role": "user", "content": text})
                tokens = render(tok, messages, args.thinking)
                reset = tokens[:len(history)] != history or not history
                if reset and history:
                    print("[history is not a token prefix of the new render; resetting]", file=sys.stderr)
                suffix = tokens if reset else tokens[len(history):]
                r = session.request(suffix, args.max_tokens, reset)
                history = tokens + r["tokens"]
                reply = reply_text(tok, r["tokens"])
                messages.append({"role": "assistant", "content": reply})
                print(reply)
                print("[%d new tokens, %d generated, prefill %.3fs, decode %.3fs]" % (
                    len(suffix), len(r["tokens"]), r["prefill_seconds"], r["decode_seconds"]), file=sys.stderr)
        finally:
            session.close()
        return

    messages.append({"role": "user", "content": args.prompt})
    tokens = render(tok, messages, args.thinking)
    record = {"prompt_tokens": len(tokens)}
    if not args.no_native:
        session = Session(args.binary.resolve(), args.checkpoint.resolve(), args.session_arg, args.timeout)
        try:
            r = session.request(tokens, args.max_tokens, True)
        finally:
            session.close()
        record["native"] = {"tokens": r["tokens"], "text": reply_text(tok, r["tokens"]),
                            **{k: r[k] for k in ("prefill_seconds", "decode_seconds", "eos") if k in r}}
    if args.mlx or args.no_native:
        ids, prefill, decode = mlx_greedy(args.checkpoint, tokens, args.max_tokens, args.mlx_dtype)
        record["mlx"] = {"dtype": args.mlx_dtype, "tokens": ids, "text": reply_text(tok, ids),
                         "prefill_seconds": prefill, "decode_seconds": decode}
    if "native" in record and "mlx" in record:
        a, b = record["native"]["tokens"], record["mlx"]["tokens"]
        common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        record["tokens_identical"] = a == b
        record["common_prefix_tokens"] = common
    if args.json:
        print(json.dumps(record, indent=1))
        return
    for name in ("native", "mlx"):
        if name in record:
            if len(record) > 2:
                print("--- %s%s" % (name, " (%s)" % args.mlx_dtype if name == "mlx" else ""))
            print(record[name]["text"])
    if "tokens_identical" in record:
        print("--- native and stock MLX tokens %s (common prefix %d)" % (
            "identical" if record["tokens_identical"] else "DIFFER", record["common_prefix_tokens"]))


if __name__ == "__main__":
    main()
