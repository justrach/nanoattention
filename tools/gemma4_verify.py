#!/usr/bin/env python3
"""Synthetic Gemma 4 parity suite: NumPy FP64 (seq==step), nano (exact, feed=1/4,
presets/gemma4, d2_gate), MLX FP32 (`mlx_lm`), and multi-turn `nano session` + reset."""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx_lm.models.base as mlx_base
import mlx_lm.utils as mlx_utils
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
NANO_BIN = ROOT / "zig-out" / "bin" / "nano"
EXACT_PRESET = ROOT / "src" / "mega" / "presets" / "exact.json"
GEMMA4_PRESET = ROOT / "src" / "mega" / "presets" / "gemma4.json"


def bf16_round_f32(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    u = x.view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def pack_affine_q4(rng: np.random.Generator, out_dim: int, in_dim: int, group: int = 64, scale_std: float = 0.03):
    """Create MLX affine 4-bit quantized tensor parts (weight u32, bf16 scales, bf16 biases) + exact f64 dense matrix."""
    assert in_dim % group == 0
    assert group % 8 == 0
    groups = in_dim // group
    w_dense_raw = rng.normal(0.0, scale_std, size=(out_dim, in_dim)).astype(np.float32)
    wq, sc, bs = mx.quantize(mx.array(w_dense_raw), group_size=group, bits=4, mode="affine")
    sc_bf16 = sc.astype(mx.bfloat16)
    bs_bf16 = bs.astype(mx.bfloat16)
    mx.eval(wq, sc_bf16, bs_bf16)

    w_u32 = np.array(wq, copy=True)
    # Use BF16-rounded scales/biases so NumPy matches Metal mk_load(..., BF16) exactly.
    sc_f32 = np.array(sc_bf16.astype(mx.float32), copy=True)
    bs_f32 = np.array(bs_bf16.astype(mx.float32), copy=True)
    deq = mx.dequantize(wq, sc_bf16.astype(mx.float32), bs_bf16.astype(mx.float32), group_size=group, bits=4, mode="affine")
    mx.eval(deq)
    dense_f64 = np.array(deq, dtype=np.float64)
    return {
        "weight": wq,
        "scales": sc_bf16,
        "biases": bs_bf16,
        "dense": dense_f64,
        "u32": w_u32,
        "sc_f32": sc_f32,
        "bs_f32": bs_f32,
        "groups": groups,
    }


def build_synthetic_checkpoint(ckpt_dir: Path, seed: int = 42):
    rng = np.random.default_rng(seed)
    cfg = {
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified",
        "vocab_size": 256,
        "eos_token_id": [1, 106, 50],
        "tie_word_embeddings": True,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 64,
            "global_head_dim": 512,
            "attention_k_eq_v": True,
            "attention_bias": False,
            "sliding_window": 6,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "final_logit_softcapping": 30.0,
            "max_position_embeddings": 256,
            "vocab_size": 256,
            "eos_token_id": 1,
            "tie_word_embeddings": True,
            "enable_moe_block": False,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "use_double_wide_mlp": False,
            "rope_traditional": False,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "proportional",
                },
            },
        },
    }
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    prefix = "language_model.model."
    tensors = {}
    ref = {"cfg": cfg, "layers": []}

    def add_norm(key: str, dim: int, val_base: float = 1.0) -> np.ndarray:
        vals = (val_base + rng.normal(0.0, 0.15, size=(dim,))).astype(np.float32)
        arr = mx.array(vals, dtype=mx.bfloat16)
        mx.eval(arr)
        tensors[key] = arr
        return np.array(arr.astype(mx.float32), dtype=np.float64)

    def add_q4(base_key: str, out_dim: int, in_dim: int, std: float = 0.04) -> np.ndarray:
        q = pack_affine_q4(rng, out_dim, in_dim, group=64, scale_std=std)
        tensors[f"{base_key}.weight"] = q["weight"]
        tensors[f"{base_key}.scales"] = q["scales"]
        tensors[f"{base_key}.biases"] = q["biases"]
        return q["dense"]

    tcfg = cfg["text_config"]
    h = tcfg["hidden_size"]
    inter = tcfg["intermediate_size"]
    heads = tcfg["num_attention_heads"]
    ref["embed"] = add_q4(f"{prefix}embed_tokens", tcfg["vocab_size"], h, std=0.05)

    for i, lt in enumerate(tcfg["layer_types"]):
        sliding = lt == "sliding_attention"
        dim = tcfg["head_dim"] if sliding else tcfg["global_head_dim"]
        kvh = tcfg["num_key_value_heads"] if sliding else tcfg["num_global_key_value_heads"]
        q_dim = heads * dim
        kv_dim = kvh * dim
        lp = f"{prefix}layers.{i}."
        layer_ref = {
            "sliding": sliding,
            "dim": dim,
            "kv_heads": kvh,
            "k_eq_v": (not sliding) and tcfg["attention_k_eq_v"],
            "input_layernorm": add_norm(f"{lp}input_layernorm.weight", h),
            "q_proj": add_q4(f"{lp}self_attn.q_proj", q_dim, h),
            "k_proj": add_q4(f"{lp}self_attn.k_proj", kv_dim, h),
            "v_proj": None if ((not sliding) and tcfg["attention_k_eq_v"]) else add_q4(f"{lp}self_attn.v_proj", kv_dim, h),
            "q_norm": add_norm(f"{lp}self_attn.q_norm.weight", dim),
            "k_norm": add_norm(f"{lp}self_attn.k_norm.weight", dim),
            "o_proj": add_q4(f"{lp}self_attn.o_proj", h, q_dim),
            "post_attention_layernorm": add_norm(f"{lp}post_attention_layernorm.weight", h),
            "pre_feedforward_layernorm": add_norm(f"{lp}pre_feedforward_layernorm.weight", h),
            "gate_proj": add_q4(f"{lp}mlp.gate_proj", inter, h),
            "up_proj": add_q4(f"{lp}mlp.up_proj", inter, h),
            "down_proj": add_q4(f"{lp}mlp.down_proj", h, inter),
            "post_feedforward_layernorm": add_norm(f"{lp}post_feedforward_layernorm.weight", h),
        }
        scalar_val = np.array([0.65 + 0.12 * i], dtype=np.float32)
        scalar_arr = mx.array(scalar_val, dtype=mx.bfloat16)
        mx.eval(scalar_arr)
        tensors[f"{lp}layer_scalar"] = scalar_arr
        layer_ref["layer_scalar"] = float(np.array(scalar_arr.astype(mx.float32), dtype=np.float64)[0])
        ref["layers"].append(layer_ref)

    ref["norm"] = add_norm(f"{prefix}norm.weight", h)
    mx.save_safetensors(str(ckpt_dir / "model.safetensors"), tensors)
    return ref


def rmsnorm_f64(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    y = x * inv
    return y if w is None else y * w


def gelu_tanh_f64(x: np.ndarray) -> np.ndarray:
    k = math.sqrt(2.0 / math.pi)
    u = k * (x + 0.044715 * (x ** 3))
    return 0.5 * x * (1.0 + np.tanh(u))


def make_rope_freqs_f64(dim: int, theta: float, partial: float) -> np.ndarray:
    half = dim // 2
    rot_pairs = int(math.floor(partial * dim / 2.0))
    freqs = np.zeros((half,), dtype=np.float64)
    for j in range(rot_pairs):
        freqs[j] = 1.0 / (theta ** ((2.0 * j) / dim))
    return freqs


def apply_rope_f64(x: np.ndarray, freqs: np.ndarray, pos: int) -> np.ndarray:
    # x: [heads, dim]
    half = x.shape[-1] // 2
    angles = pos * freqs
    c = np.cos(angles)[None, :]
    s = np.sin(angles)[None, :]
    a = x[:, :half]
    b = x[:, half:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def forward_numpy(ref: dict, tokens: list[int], kv_cache=None):
    tcfg = ref["cfg"]["text_config"]
    h = tcfg["hidden_size"]
    heads = tcfg["num_attention_heads"]
    eps = tcfg["rms_norm_eps"]
    win = tcfg["sliding_window"]
    cap = tcfg["final_logit_softcapping"]
    f_sliding = make_rope_freqs_f64(tcfg["head_dim"], 10000.0, 1.0)
    f_full = make_rope_freqs_f64(tcfg["global_head_dim"], 1000000.0, 0.25)

    if kv_cache is None:
        kv_cache = [([], []) for _ in ref["layers"]]
    base_pos = len(kv_cache[0][0])

    logits_all = []
    for step_idx, tok in enumerate(tokens):
        pos = base_pos + step_idx
        x = ref["embed"][tok].copy() * math.sqrt(h)
        for li, layer in enumerate(ref["layers"]):
            dim = layer["dim"]
            kvh = layer["kv_heads"]
            freqs = f_sliding if layer["sliding"] else f_full
            xn = rmsnorm_f64(x, layer["input_layernorm"], eps)
            q = (layer["q_proj"] @ xn).reshape(heads, dim)
            k = (layer["k_proj"] @ xn).reshape(kvh, dim)
            v = k.copy() if layer["k_eq_v"] else (layer["v_proj"] @ xn).reshape(kvh, dim)

            v = rmsnorm_f64(v, None, eps)
            q = apply_rope_f64(rmsnorm_f64(q, layer["q_norm"], eps), freqs, pos)
            k = apply_rope_f64(rmsnorm_f64(k, layer["k_norm"], eps), freqs, pos)

            k_hist, v_hist = kv_cache[li]
            k_hist.append(k)
            v_hist.append(v)

            start = max(0, pos + 1 - win) if layer["sliding"] else 0
            k_win = np.stack(k_hist[start : pos + 1], axis=0)  # [S, kvh, dim]
            v_win = np.stack(v_hist[start : pos + 1], axis=0)
            g = heads // kvh
            att = np.zeros((heads, dim), dtype=np.float64)
            for head_i in range(heads):
                kvh_i = head_i // g
                scores = k_win[:, kvh_i, :] @ q[head_i]
                scores -= np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                att[head_i] = probs @ v_win[:, kvh_i, :]

            o = layer["o_proj"] @ att.reshape(-1)
            x = x + rmsnorm_f64(o, layer["post_attention_layernorm"], eps)

            ff_in = rmsnorm_f64(x, layer["pre_feedforward_layernorm"], eps)
            gate = gelu_tanh_f64(layer["gate_proj"] @ ff_in)
            up = layer["up_proj"] @ ff_in
            mlp_out = layer["down_proj"] @ (gate * up)
            x = x + rmsnorm_f64(mlp_out, layer["post_feedforward_layernorm"], eps)
            x = x * layer["layer_scalar"]

        hn = rmsnorm_f64(x, ref["norm"], eps)
        logits = ref["embed"] @ hn
        logits = np.tanh(logits / cap) * cap
        logits_all.append(logits)
    return np.stack(logits_all, axis=0), kv_cache


def run_nano(args: list[str], stdin_text: str | None = None):
    proc = subprocess.run(
        [str(NANO_BIN), *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def nano_eval(ckpt_dir: Path, prompt: list[int], feed: int, knobs_path: Path | None = None, extra_knobs: list[str] = ()):
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        args = [
            "eval",
            str(ckpt_dir),
            "--tokens",
            ",".join(str(t) for t in prompt),
            "--feed",
            str(feed),
            "--context",
            "64",
            "--logits-out",
            f.name,
        ]
        if knobs_path is not None:
            args.extend(["--knobs", str(knobs_path)])
        for kv in extra_knobs:
            args.extend(["--knob", kv])
        out = run_nano(args)
        summary = json.loads(out.strip().splitlines()[-1])
        logits = np.fromfile(f.name, dtype=np.float32).reshape(len(prompt), summary["vocab"])
        return summary, logits


def mlx_fp32_forward(ckpt_dir: Path, prompt: list[int], gen_tokens: int = 4):
    model, _ = mlx_utils.load_model(ckpt_dir)
    # Cast floating parameters/scales/biases to float32 so MLX reference executes in FP32.
    def to_f32(tree):
        if isinstance(tree, mx.array):
            return tree.astype(mx.float32) if mx.issubdtype(tree.dtype, mx.floating) else tree
        if isinstance(tree, dict):
            return {k: to_f32(v) for k, v in tree.items()}
        if isinstance(tree, list):
            return [to_f32(v) for v in tree]
        return tree

    model.update(to_f32(model.parameters()))
    mx.eval(model.parameters())

    cache = model.make_cache()
    logits = model(mx.array([prompt], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    prefill_logits = np.array(logits[0], dtype=np.float32)

    cur = int(np.argmax(prefill_logits[-1]))
    greedy = [cur]
    for _ in range(gen_tokens - 1):
        step_logits = model(mx.array([[cur]], dtype=mx.int32), cache=cache)
        mx.eval(step_logits)
        cur = int(np.argmax(np.array(step_logits[0, -1], dtype=np.float32)))
        greedy.append(cur)
    return prefill_logits, greedy


def main():
    assert NANO_BIN.exists(), f"Missing {NANO_BIN}; build first"
    prompt = [5, 19, 42, 77, 13, 199, 250, 3, 88, 127, 64, 15]

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        ref = build_synthetic_checkpoint(ckpt_dir)

        # 1. NumPy FP64 sequence == step
        np_seq, _ = forward_numpy(ref, prompt)
        kv = None
        step_rows = []
        for tok in prompt:
            l, kv = forward_numpy(ref, [tok], kv_cache=kv)
            step_rows.append(l[0])
        np_step = np.stack(step_rows, axis=0)
        np_diff = float(np.max(np.abs(np_seq - np_step)))
        assert np_diff < 1e-12, f"NumPy seq vs step mismatch: {np_diff}"
        print(f"[PASS] NumPy FP64 seq == step (max_abs={np_diff:.2e})")

        # 2. nano eval: reference feed=1 vs feed=4 bitwise
        sum_ref1, log_ref1 = nano_eval(ckpt_dir, prompt, feed=1)
        sum_ref4, log_ref4 = nano_eval(ckpt_dir, prompt, feed=4)
        assert np.array_equal(log_ref1.view(np.uint32), log_ref4.view(np.uint32)), "reference feed=1 vs feed=4 not bitwise!"
        np_err = float(np.max(np.abs(log_ref1.astype(np.float64) - np_seq)))
        assert np_err < 1e-3, f"nano vs NumPy max_abs too high: {np_err}"
        assert sum_ref1["top1"] == np.argmax(np_seq, axis=-1).tolist()
        print(f"[PASS] nano ref feed=1 == feed=4 bitwise; vs NumPy f64 max_abs={np_err:.2e}, top1 12/12")

        # 3. presets/exact.json (and d2_gate) vs reference bitwise
        sum_ex, log_ex = nano_eval(ckpt_dir, prompt, feed=4, knobs_path=EXACT_PRESET, extra_knobs=["batch=64"])
        assert np.array_equal(log_ref1.view(np.uint32), log_ex.view(np.uint32)), "exact preset differs bitwise from ref!"
        print("[PASS] nano presets/exact.json == reference bitwise")

        # 4. presets/gemma4.json (exercising batched prefill including wide MMA attn when feed=12 > 8)
        sum_g4, log_g4 = nano_eval(ckpt_dir, prompt, feed=12, knobs_path=GEMMA4_PRESET, extra_knobs=["batch=64"])
        g4_err = float(np.max(np.abs(log_g4 - log_ref1)))
        assert g4_err < 2e-3, f"presets/gemma4.json max_abs={g4_err}"
        assert sum_g4["top1"] == sum_ref1["top1"]
        print(f"[PASS] nano presets/gemma4.json (feed=12) max_abs vs ref={g4_err:.2e}, top1 12/12")

        # 5. generate: d2_gate=false vs d2_gate=true bitwise identical token stream
        gen_off = json.loads(run_nano([
            "generate", str(ckpt_dir),
            "--tokens", ",".join(str(t) for t in prompt),
            "--max-tokens", "6",
            "--context", "64",
            "--knobs", str(GEMMA4_PRESET),
            "--knob", "batch=64",
            "--knob", "d2_gate=false",
        ]).strip().splitlines()[-1])
        gen_on = json.loads(run_nano([
            "generate", str(ckpt_dir),
            "--tokens", ",".join(str(t) for t in prompt),
            "--max-tokens", "6",
            "--context", "64",
            "--knobs", str(GEMMA4_PRESET),
            "--knob", "batch=64",
            "--knob", "d2_gate=true",
        ]).strip().splitlines()[-1])
        assert gen_off["tokens"] == gen_on["tokens"], f"d2_gate mismatch: {gen_off} vs {gen_on}"
        print(f"[PASS] nano generate d2_gate=true == d2_gate=false ({gen_on['tokens']})")

        # 6. mlx_lm FP32 parity against nano eval & generate
        mlx_logits, mlx_greedy = mlx_fp32_forward(ckpt_dir, prompt, gen_tokens=len(gen_off["tokens"]))
        mlx_err = float(np.max(np.abs(log_ref1 - mlx_logits)))
        mlx_top1 = np.argmax(mlx_logits, axis=-1).tolist()
        assert mlx_top1 == sum_ref1["top1"], f"MLX vs nano top1 mismatch: {mlx_top1} vs {sum_ref1['top1']}"
        assert mlx_err < 2e-3, f"MLX vs nano max_abs={mlx_err}"
        assert mlx_greedy == gen_off["tokens"], f"MLX greedy {mlx_greedy} != nano {gen_off['tokens']}"
        print(f"[PASS] mlx_lm FP32 parity: max_abs={mlx_err:.2e}, top1 12/12, greedy={mlx_greedy}")

        # 7. nano session multi-turn (prefix cache hit / turn 2 continuation) and reset
        # Note: turn 1 prefill+decode commits 1 generated token (prompt[:6] + gen0), OR use max_tokens=0?
        # Look: with max_tokens=1, session commits prompt[:7] without EOS if not eos -> wait: test max_tokens=0!
        # Wait: let's test prompt[:6] with max_tokens=1, then continue, vs manual continuation AND reset!
        reqs = [
            {"id": 1, "tokens": prompt[:7], "max_tokens": 0, "return_logits": True, "reset": True},
            {"id": 2, "tokens": prompt[7:], "max_tokens": len(gen_off["tokens"]), "return_logits": True, "reset": False},
            {"id": 3, "tokens": prompt, "max_tokens": len(gen_off["tokens"]), "return_logits": True, "reset": True},
            {"id": 4, "tokens": prompt[:7], "max_tokens": 0, "return_logits": True, "reset": True},
        ]
        sess_out = run_nano(
            ["session", str(ckpt_dir), "--context", "64", "--knobs", str(GEMMA4_PRESET), "--knob", "batch=64"],
            stdin_text="\n".join(json.dumps(r) for r in reqs) + "\n",
        )
        resps = [json.loads(line) for line in sess_out.strip().splitlines()]
        assert len(resps) == 4
        r1, r2, r3, r4 = resps
        assert r1["cache_after"] == 7 and r2["cache_before"] == 7
        assert r2["tokens"] == r3["tokens"] == gen_off["tokens"]
        r2_err = float(np.max(np.abs(np.array(r2["prefill_logits"], dtype=np.float32) - log_g4[-1])))
        assert r2_err < 1e-4, f"session turn2 logits vs eval={r2_err}"
        assert np.array_equal(
            np.array(r1["prefill_logits"], dtype=np.float32),
            np.array(r4["prefill_logits"], dtype=np.float32),
        ), "session reset turn1 != turn4!"
        print(f"[PASS] nano session turn1+turn2+reset parity (turn2 err={r2_err:.2e}, tokens={r2['tokens']})")


if __name__ == "__main__":
    main()
