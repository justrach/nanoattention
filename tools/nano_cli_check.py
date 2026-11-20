#!/usr/bin/env python3
"""End-to-end checks of `nano run` and `nano chat` on a real checkpoint (docs/CLI.md).

  golden CHECKPOINT CASE_DIR   `nano run --json` on the case's prompt text must reproduce the
                               case's greedy tokens (greedy.json) and their text
  chat CHECKPOINT              a three-turn `nano chat --json` fed from a pipe must give the same
                               reply tokens as one `nano session` fed the same ids per turn
                               (session with the reference knobs and with the chat's preset)

Each nano process runs alone, one after another; run this script under the inference lock:
  python3 tools/guarded_run.py --lock LOCK --timeout 280 -- python3 tools/nano_cli_check.py chat CKPT
Standard library only. Exit 0 when every comparison is identical.
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

TURNS = [
    "What is the capital of Australia? Answer in one sentence.",
    "And what is its population, roughly?",
    "Name one famous building there.",
]


def nano(binary, *args, stdin=None):
    proc = subprocess.run([str(binary), *args], input=stdin, capture_output=True, text=True, timeout=240)
    if proc.returncode != 0:
        sys.exit("nano %s failed (%d): %s" % (args[0], proc.returncode, proc.stderr[-2000:]))
    return proc


def cmd_golden(args):
    case = pathlib.Path(args.case)
    meta = json.loads((case / "meta.json").read_text())
    prompt = meta["description"].split('"')[1]
    want = json.loads((case / "greedy.json").read_text())["tokens"]
    prompt_ids = json.loads((case / "tokens.json").read_text())["tokens"]
    out = json.loads(nano(args.binary, "run", str(args.checkpoint), prompt, "--json", "--max-tokens", str(len(want))).stdout)
    ids = nano(args.binary, "tokenize", str(args.checkpoint), prompt, "--chat").stdout
    want_text = nano(args.binary, "detokenize", str(args.checkpoint), *map(str, want)).stdout
    ok = out["reply_ids"] == want and json.loads(ids) == prompt_ids and out["text"] == want_text
    print("golden %s: prompt %d tokens (ids %s the golden's), reply %s -> %s; text %r (%s); stop %s"
          % (case.name, out["prompt_tokens"], "equal to" if json.loads(ids) == prompt_ids else "DIFFERENT FROM",
             out["reply_ids"], "identical to greedy.json" if out["reply_ids"] == want else "DIFFERS from %s" % want,
             out["text"], "equal" if out["text"] == want_text else "DIFFERS", out["stop"]))
    return 0 if ok else 1


def session_replies(binary, checkpoint, fed, max_tokens, extra):
    lines = "".join(json.dumps({"id": i + 1, "tokens": ids, "max_tokens": max_tokens}) + "\n" for i, ids in enumerate(fed))
    out = nano(binary, "session", str(checkpoint), *extra, stdin=lines).stdout
    return [json.loads(line)["tokens"] for line in out.splitlines() if line.strip()]


def cmd_chat(args):
    stdin = "".join(t + "\n" for t in TURNS)
    out = nano(args.binary, "chat", str(args.checkpoint), "--json", "--max-tokens", str(args.max_tokens), stdin=stdin).stdout
    turns = [json.loads(line) for line in out.splitlines() if line.strip()]
    if len(turns) != len(TURNS):
        sys.exit("chat answered %d turns for %d" % (len(turns), len(TURNS)))
    fed = [t["prompt_ids"] for t in turns]
    replies = [t["reply_ids"] for t in turns]
    status = 0
    for name, extra in (("reference knobs", []), ("exact preset", ["--knobs", str(ROOT / "src/mega/presets/exact.json")])):
        got = session_replies(args.binary, args.checkpoint, fed, args.max_tokens, extra)
        same = got == replies
        status |= 0 if same else 1
        print("chat vs session (%s): %d turns, fed %s tokens, replies %s tokens: %s"
              % (name, len(turns), [len(f) for f in fed], [len(r) for r in replies], "identical" if same else "DIFFERENT"))
    for i, t in enumerate(turns):
        print("  turn %d (%s): %r" % (i + 1, t["stop"], t["text"][:200]))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("golden", "chat"))
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("case", nargs="?", help="golden case directory (golden)")
    parser.add_argument("--binary", type=pathlib.Path, default=ROOT / "zig-out/bin/nano")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.command == "golden" and not args.case:
        parser.error("golden needs CASE_DIR")
    sys.exit(cmd_golden(args) if args.command == "golden" else cmd_chat(args))


if __name__ == "__main__":
    main()
