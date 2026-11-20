#!/usr/bin/env python3
"""Stand-in for `nano session CKPT`: the session JSONL protocol served by stock MLX float32 greedy.

For exercising tools/lfm3_chat.py and tools/lfm3_benchmark.py without the native engine. Needs
mlx and mlx-vlm on the python3 found in PATH (the reference venv). Timings are stock MLX, not native.
"""
import json, sys, time
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from lfm3_reference import StockMLX
assert sys.argv[1] == "session"
stock = StockMLX(sys.argv[2], "float32")
cache, position = stock.cache(), 0
for line in sys.stdin:
    req = json.loads(line)
    if req.get("reset"):
        cache, position = stock.cache(), 0
    before = position
    t0 = time.perf_counter()
    _, lg = stock.run(req["tokens"], cache, [len(req["tokens"]) - 1], keep_hidden=False)
    t1 = time.perf_counter()
    out = stock.greedy(lg[0], cache, [124900], req.get("max_tokens", 128))
    # commit the final token too, as the native session does
    stock.run([out[-1]], cache, [], keep_hidden=False)
    t2 = time.perf_counter()
    position = before + len(req["tokens"]) + len(out)
    print(json.dumps({"id": req.get("id", 0), "tokens": out, "eos": out[-1] == 124900, "cache_before": before,
                      "cache_after": position, "prefill_tokens": len(req["tokens"]), "decode_forward_tokens": len(out),
                      "reused_prefix_tokens": before, "prefill_seconds": t1 - t0, "decode_seconds": t2 - t1}), flush=True)
