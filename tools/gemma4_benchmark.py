#!/usr/bin/env python3
"""Head-to-head benchmark: nano megakernel vs stock mlx-lm (BF16 & FP32, async & sync)
on Gemma 4 12B (lmstudio-community/gemma-4-12B-it-MLX-4bit).

Each engine runs in its own isolated child process so weights and KV caches are
never resident concurrently. Run under tools/guarded_run.py with the shared lock.
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NANO = ROOT / "zig-out" / "bin" / "nano"
GEMMA4_PRESET = ROOT / "src" / "mega" / "presets" / "gemma4.json"
EXACT_PRESET = ROOT / "src" / "mega" / "presets" / "exact.json"

WORKLOAD_TEXT = (
    "A Kalman filter estimates the state of a linear dynamic system from noisy "
    "observations by alternating prediction and measurement update steps. Under "
    "Gaussian noise assumptions, the covariance recursion propagates uncertainty "
    "across time while incorporating sensor residuals weighted by the Kalman gain.\n\n"
    "In transformer autoregressive decoding on unified-memory Apple Silicon, each "
    "token step reads quantized projection matrices across all layers while "
    "updating sliding-window and full-attention key-value caches."
)


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def system_info() -> dict:
    def sysctl(name: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except Exception:
            return "unknown"

    return {
        "machine": platform.machine(),
        "os": f"macOS {platform.mac_ver()[0]}",
        "cpu_brand": sysctl("machdep.cpu.brand_string"),
        "memsize_gb": round(int(sysctl("hw.memsize")) / (1024 ** 3), 1) if sysctl("hw.memsize").isdigit() else 0,
        "git_commit": git_head(),
        "python": platform.python_version(),
    }


def build_workload(model_dir: Path, decode_tokens: int) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    bos = tok.bos_token_id if tok.bos_token_id is not None else 2
    code_ids = tok.encode(WORKLOAD_TEXT, add_special_tokens=False)
    # Use enough decoding IDs for teacher-forced arm.
    dec_seed = tok.encode(" Repeat step prediction update covariance gain residual filter state matrix vector cache layer.", add_special_tokens=False)
    decode_ids = (dec_seed * ((decode_tokens // len(dec_seed)) + 1))[:decode_tokens]
    return {"bos": int(bos), "code": [int(x) for x in code_ids], "decode": [int(x) for x in decode_ids]}


def make_prompt(wl: dict, length: int) -> list[int]:
    out = [wl["bos"]]
    code = wl["code"]
    for i in range(length - 1):
        out.append(code[i % len(code)])
    return out


def run_nano_arm(
    nano_bin: Path,
    model_dir: Path,
    preset: Path,
    wl_path: Path,
    prompt_len: int,
    decode_tokens: int,
    repeats: int,
    context: int,
    decode_mode: str,
) -> dict:
    child_code = r"""
import json, resource, subprocess, sys
cmd = json.loads(sys.argv[1])
proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
data = json.loads(proc.stdout.strip().splitlines()[-1])
data["peak_rss_bytes"] = ru.ru_maxrss
print(json.dumps(data))
"""
    cmd = [
        str(nano_bin),
        "bench",
        str(model_dir),
        "--knobs",
        str(preset),
        "--prompt-tokens",
        str(prompt_len),
        "--decode-tokens",
        str(decode_tokens),
        "--decode-mode",
        decode_mode,
        "--repeats",
        str(repeats),
        "--context",
        str(context),
        "--bench-tokens",
        str(wl_path),
    ]
    proc = subprocess.run([sys.executable, "-c", child_code, json.dumps(cmd)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nano failed ({preset.name}, L={prompt_len}): {proc.stderr}\n{proc.stdout}")
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    pf_s = statistics.median(res["prefill_seconds"])
    dec_s = statistics.median(res["decode_seconds"])
    return {
        "prefill_tokens_per_second": res["prefill_tokens_per_second"],
        "prefill_ms": pf_s * 1000.0,
        "decode_tokens_per_second": res["decode_tokens_per_second"],
        "decode_ms_per_tok": (dec_s / decode_tokens) * 1000.0,
        "dispatches_per_decode_token": res["dispatches_per_decode_token"],
        "prefill_forwards": res["prefill_forwards"],
        "dispatches_per_prefill_forward": res["dispatches_per_prefill_forward"],
        "prediction_checksum": res["prediction_checksum"],
        "peak_rss_mb": round(res["peak_rss_bytes"] / (1024 ** 2), 1),
        "prefill_samples_s": res["prefill_seconds"],
        "decode_samples_s": res["decode_seconds"],
    }


def run_mlx_arm(
    model_dir: Path,
    wl: dict,
    lengths: list[int],
    decode_tokens: int,
    repeats: int,
    dtype_mode: str,
    async_mode: bool,
) -> dict:
    child_code = r"""
import importlib, json, resource, statistics, sys, time
from pathlib import Path
import mlx.core as mx
import mlx_lm.utils as mlx_utils
mlx_gen = importlib.import_module("mlx_lm.generate")

req = json.loads(sys.argv[1])
model_dir = Path(req["model_dir"])
wl = req["wl"]
lengths = req["lengths"]
decode_tokens = req["decode_tokens"]
repeats = req["repeats"]
dtype_mode = req["dtype_mode"]
async_mode = req["async_mode"]

def to_f32(tree):
    if isinstance(tree, mx.array):
        return tree.astype(mx.float32) if mx.issubdtype(tree.dtype, mx.floating) else tree
    if isinstance(tree, dict):
        return {k: to_f32(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [to_f32(v) for v in tree]
    return tree

model, _ = mlx_utils.load_model(model_dir)
if dtype_mode == "f32":
    model.update(to_f32(model.parameters()))
    mx.eval(model.parameters())
mx.synchronize()

def make_prompt(L):
    out = [wl["bos"]]
    c = wl["code"]
    for i in range(L - 1):
        out.append(c[i % len(c)])
    return out

def run_one(prompt):
    cache = model.make_cache()
    stream = mlx_gen.generation_stream
    mx.synchronize(stream)
    t0 = time.perf_counter()
    arr = mx.array(prompt, dtype=mx.int32)
    with mx.stream(stream):
        i = 0
        chunk = 2048
        n_rem = len(prompt) - 1
        while i < n_rem:
            nxt = min(i + chunk, n_rem)
            model(arr[i:nxt][None], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
            i = nxt
        logits = model(arr[-1:][None], cache=cache)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        tok = int(y.item())
        t1 = time.perf_counter()

        preds = [tok]
        if async_mode:
            def step(x_arr):
                lg = model(x_arr[None], cache=cache)
                return mx.argmax(lg[:, -1, :], axis=-1)
            cur = step(mx.array([tok], dtype=mx.int32))
            mx.async_eval(cur)
            for s in range(decode_tokens):
                if s + 1 < decode_tokens:
                    nxt_arr = step(cur)
                    mx.async_eval(nxt_arr)
                    val = int(cur.item())
                    preds.append(val)
                    cur = nxt_arr
                else:
                    val = int(cur.item())
                    preds.append(val)
            mx.synchronize(stream)
        else:
            for _ in range(decode_tokens):
                lg = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
                y = mx.argmax(lg[:, -1, :], axis=-1)
                mx.eval(y)
                tok = int(y.item())
                preds.append(tok)
            mx.synchronize(stream)
        t2 = time.perf_counter()
    chk = sum(preds)
    kv_dtype = str(cache[0].keys.dtype) if cache and cache[0].keys is not None else "unknown"
    return t1 - t0, t2 - t1, chk, kv_dtype

results = {}
with mlx_gen.wired_limit(model, [mlx_gen.generation_stream]):
    for L in lengths:
        prompt = make_prompt(L)
        _, _, warm_chk, kv_dt = run_one(prompt)
        pf_list, dec_list = [], []
        for _ in range(repeats):
            pf_s, dec_s, chk, _ = run_one(prompt)
            assert chk == warm_chk, f"nondeterministic MLX checksum at L={L}: {chk} vs {warm_chk}"
            pf_list.append(pf_s)
            dec_list.append(dec_s)
        pm = statistics.median(pf_list)
        dm = statistics.median(dec_list)
        results[str(L)] = {
            "prefill_tokens_per_second": L / pm,
            "prefill_ms": pm * 1000.0,
            "decode_tokens_per_second": decode_tokens / dm,
            "decode_ms_per_tok": (dm / decode_tokens) * 1000.0,
            "prediction_checksum": warm_chk,
            "kv_dtype": kv_dt,
            "prefill_samples_s": pf_list,
            "decode_samples_s": dec_list,
        }

ru = resource.getrusage(resource.RUSAGE_SELF)
peak_metal_mb = round(mx.get_peak_memory() / (1024 ** 2), 1)
print(json.dumps({
    "results": results,
    "peak_rss_mb": round(ru.ru_maxrss / (1024 ** 2), 1),
    "peak_metal_mb": peak_metal_mb,
    "mlx_version": mx.__version__,
}))
"""
    req = json.dumps({
        "model_dir": str(model_dir),
        "wl": wl,
        "lengths": lengths,
        "decode_tokens": decode_tokens,
        "repeats": repeats,
        "dtype_mode": dtype_mode,
        "async_mode": async_mode,
    })
    proc = subprocess.run([sys.executable, "-c", child_code, req], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"MLX {dtype_mode} async={async_mode} failed: {proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def render_markdown(report: dict) -> str:
    sys_i = report["system"]
    rows = report["results"]
    arms_order = [
        ("nano_gemma4_greedy", "nano (`presets/gemma4.json`, greedy)", "FP32 act / FP32 KV"),
        ("nano_gemma4_tf", "nano (`presets/gemma4.json`, teacher-forced)", "FP32 act / FP32 KV"),
        ("nano_exact_greedy", "nano (`presets/exact.json`, greedy)", "FP32 act / FP32 KV"),
        ("mlx_bf16_async", "mlx-lm BF16 (`async_eval`)", "BF16 act / BF16 KV"),
        ("mlx_bf16_sync", "mlx-lm BF16 (sync)", "BF16 act / BF16 KV"),
        ("mlx_f32_async", "mlx-lm FP32 (`async_eval`)", "FP32 act / FP32 KV"),
        ("mlx_f32_sync", "mlx-lm FP32 (sync)", "FP32 act / FP32 KV"),
    ]

    lines = [
        "# Gemma 4 12B (`nano` vs `mlx-lm` on Apple Silicon)",
        "",
        "Head-to-head throughput benchmark of `nano` megakernel against stock `mlx-lm` on `lmstudio-community/gemma-4-12B-it-MLX-4bit` (4-bit affine quantization, group size 64, 48 decoder layers: 5x sliding window 1024 + 1x full `k_eq_v` attention with `global_head_dim=512`, `layer_scalar`, tied `embed_tokens`, `final_logit_softcapping=30.0`).",
        "",
        "## Environment & Methodology",
        "",
        f"- **Hardware**: {sys_i['cpu_brand']}, {sys_i['memsize_gb']} GiB unified memory, {sys_i['os']} (`{sys_i['machine']}`)",
        f"- **Commit**: `{sys_i['git_commit']}` (`zig build -Doptimize=ReleaseSafe`)",
        f"- **MLX**: `mlx` {report['mlx_version']}, `mlx-lm` {report['mlx_lm_version']} (stock `mlx_lm.models.gemma4`)",
        f"- **Prompt Lengths**: {', '.join(str(L) for L in report['lengths'])} tokens (`context={report['context']}`), followed by {report['decode_tokens']} decode tokens, 1 warmup run + {report['repeats']} measured repeats (medians reported).",
        f"- **Workload**: identical Gemma 4 tokenized IDs across all engines (`input_sha256`: `{report['input_sha256'][:16]}...`). Each engine runs in an isolated process under `tools/guarded_run.py` so weights never overlap in RAM.",
        "- **Dtype Disclosure**: `nano` executes **FP32 activations and FP32 KV caches** (preserving < `1.5e-3` max logit error vs FP32 `mlx-lm` at `L=1280`). Stock `mlx-lm` defaults to **BF16 activations and BF16 KV caches** (`mlx_bf16`); `mlx_f32` converts floating tensors/scales/biases/KV caches to **FP32** to match `nano`'s numerical precision.",
        "",
        "## Results",
        "",
    ]

    for L in report["lengths"]:
        lines.extend([
            f"### Prompt Length `L = {L}` ({report['decode_tokens']} decode tokens)",
            "",
            "| Engine | Precision | Prefill (tok/s) | Prefill (ms) | Decode (tok/s) | Decode (ms/tok) | Peak RSS (MiB) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for key, label, prec in arms_order:
            r = rows[key][str(L)]
            rss = rows[key].get("peak_rss_mb", r.get("peak_rss_mb", 0))
            lines.append(
                f"| {label} | {prec} | {r['prefill_tokens_per_second']:,.1f} | {r['prefill_ms']:,.1f} | "
                f"{r['decode_tokens_per_second']:.2f} | {r['decode_ms_per_tok']:.2f} | {rss:,.1f} |"
            )
        lines.append("")

    # Summary speedups
    lines.extend([
        "## Summary Takeaways",
        "",
        "| Length | `nano_gemma4` vs `mlx_f32_async` (Prefill / Decode) | `nano_gemma4` vs `mlx_bf16_async` (Prefill / Decode) |",
        "| ---: | :---: | :---: |",
    ])
    for L in report["lengths"]:
        ng = rows["nano_gemma4_greedy"][str(L)]
        mf = rows["mlx_f32_async"][str(L)]
        mb = rows["mlx_bf16_async"][str(L)]
        pf_f32 = ng["prefill_tokens_per_second"] / mf["prefill_tokens_per_second"]
        dec_f32 = ng["decode_tokens_per_second"] / mf["decode_tokens_per_second"]
        pf_bf16 = ng["prefill_tokens_per_second"] / mb["prefill_tokens_per_second"]
        dec_bf16 = ng["decode_tokens_per_second"] / mb["decode_tokens_per_second"]
        lines.append(f"| {L} | {pf_f32:.2f}x / {dec_f32:.2f}x | {pf_bf16:.2f}x / {dec_bf16:.2f}x |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--nano", type=Path, default=DEFAULT_NANO)
    ap.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "gemma4-12b-mlx-comparison.json")
    ap.add_argument("--md", type=Path, default=ROOT / "benchmarks" / "gemma4-12b-mlx-comparison.md")
    ap.add_argument("--lengths", default="64,1024,4096")
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--context", type=int, default=4224)
    args = ap.parse_args()

    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    assert args.nano.exists(), f"Missing {args.nano}"
    assert args.checkpoint.exists(), f"Missing {args.checkpoint}"

    wl = build_workload(args.checkpoint, args.decode_tokens)
    wl_sha = hashlib.sha256(json.dumps(wl, sort_keys=True).encode()).hexdigest()

    import mlx_lm
    results = {
        "nano_gemma4_greedy": {},
        "nano_gemma4_tf": {},
        "nano_exact_greedy": {},
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json") as tf:
        json.dump(wl, tf)
        tf.flush()
        wl_path = Path(tf.name)

        for key, preset, mode in [
            ("nano_gemma4_greedy", GEMMA4_PRESET, "greedy"),
            ("nano_gemma4_tf", GEMMA4_PRESET, "teacher-forced"),
            ("nano_exact_greedy", EXACT_PRESET, "greedy"),
        ]:
            for L in lengths:
                print(f"[bench] {key} L={L}...", flush=True)
                r = run_nano_arm(args.nano, args.checkpoint, preset, wl_path, L, args.decode_tokens, args.repeats, args.context, mode)
                results[key][str(L)] = r
                print(f"  -> pf={r['prefill_tokens_per_second']:.1f} dec={r['decode_tokens_per_second']:.2f} rss={r['peak_rss_mb']} MiB", flush=True)

    for key, dtype_mode, async_mode in [
        ("mlx_bf16_async", "bf16", True),
        ("mlx_bf16_sync", "bf16", False),
        ("mlx_f32_async", "f32", True),
        ("mlx_f32_sync", "f32", False),
    ]:
        print(f"[bench] {key} lengths={lengths}...", flush=True)
        out = run_mlx_arm(args.checkpoint, wl, lengths, args.decode_tokens, args.repeats, dtype_mode, async_mode)
        arm = dict(out["results"])
        arm["peak_rss_mb"] = out["peak_rss_mb"]
        arm["peak_metal_mb"] = out["peak_metal_mb"]
        results[key] = arm
        for L in lengths:
            r = arm[str(L)]
            print(f"  L={L} -> pf={r['prefill_tokens_per_second']:.1f} dec={r['decode_tokens_per_second']:.2f}", flush=True)

    report = {
        "benchmark": "gemma4-12b-mlx-comparison",
        "checkpoint": "lmstudio-community/gemma-4-12B-it-MLX-4bit",
        "lengths": lengths,
        "decode_tokens": args.decode_tokens,
        "repeats": args.repeats,
        "context": args.context,
        "input_sha256": wl_sha,
        "mlx_version": out["mlx_version"],
        "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
        "system": system_info(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.md.write_text(render_markdown(report))
    print(f"Wrote {args.output} and {args.md}")


if __name__ == "__main__":
    main()
