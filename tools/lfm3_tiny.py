#!/usr/bin/env python3
"""Tiny synthetic LFM3 checkpoints with NumPy goldens, for checking an engine in seconds.

NumPy only (any Python with NumPy). Writes, per case, DIR/<case>/ with config.json and
model.safetensors (the real checkpoint's tensor names, `language_model.model.` prefix,
config keys and MLX affine layout) and DIR/<case>/golden/ in the golden format of
docs/LFM3.md (tokens.json, positions.json, logits.npy, hidden.npy, greedy.json, meta.json).

The goldens come from tools/lfm3_reference.py's NumPy model, run in float64 on the values
decoded back from the written bytes (never from the unrounded random matrices). Its two
forms, whole sequence with masks and one token at a time with a ring and conv shift
register, must agree to 1e-9 or the case fails. --check-mlx additionally loads each
checkpoint whose group size stock MLX supports into stock mlx-vlm (float32) and compares.

  python3 tools/lfm3_tiny.py --out target/lfm3-tiny [--check-mlx]
"""
import argparse
import json
import pathlib
import shutil
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lfm3_reference import Checkpoint, NumpyLFM3, layer_report, logit_report  # noqa: E402

LAYER_TYPES = ["conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "sliding_attention"]
PROMPT = [1, 17, 93, 4, 250, 8, 61, 61, 130, 7, 222, 45, 3, 199, 88, 12, 5, 240, 33, 64]
GREEDY_TOKENS = 64


def encode_float(values, dtype):
    """Bytes to write and the exact float32 values they hold."""
    if dtype == "F32":
        v = values.astype("<f4")
        return v, v.tobytes()
    if dtype == "F16":
        v = values.astype("<f2")
        return v.astype(np.float32), v.tobytes()
    if dtype == "BF16":
        bits = values.astype(np.float32).view(np.uint32)
        bf = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
        return (bf.astype(np.uint32) << 16).view(np.float32), bf.tobytes()
    raise ValueError(dtype)


def quantize(values, bits, group_size, scale_dtype):
    """MLX affine: per group of `group_size` input columns, code = round((x - min) / scale), w = scale * code + min.

    Codes are packed least significant first into a little-endian stream, stored as U32 words.
    """
    rows, cols = values.shape
    assert cols % group_size == 0 and (cols * bits) % 32 == 0
    g = values.reshape(rows, cols // group_size, group_size)
    lo, hi = g.min(-1), g.max(-1)
    scale = np.maximum((hi - lo) / ((1 << bits) - 1), np.float32(1e-6))
    scale_v, scale_b = encode_float(scale, scale_dtype)
    bias_v, bias_b = encode_float(lo, scale_dtype)
    codes = np.clip(np.rint((g - bias_v[..., None]) / scale_v[..., None]), 0, (1 << bits) - 1).astype(np.uint8)
    codes = codes.reshape(rows, cols)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4) if bits == 4 else codes
    words = np.ascontiguousarray(packed).view("<u4")
    return {
        "weight": ("U32", list(words.shape), words.tobytes()),
        "scales": (scale_dtype, list(scale.shape), scale_b),
        "biases": (scale_dtype, list(lo.shape), bias_b),
    }


def cases():
    return [
        dict(name="f32", dtype="F32"),
        dict(name="bf16", dtype="BF16"),
        dict(name="q4_g32", dtype="BF16", bits=4, group_size=32),
        dict(name="q8_g32", dtype="BF16", bits=8, group_size=32),
        dict(name="q4_g8", dtype="BF16", bits=4, group_size=8),
        dict(name="q8_g16", dtype="BF16", bits=8, group_size=16),
        dict(name="q4_g32_conv7", dtype="BF16", bits=4, group_size=32, conv_k=7, seed=3002),
    ]


def build(directory, dtype, bits=None, group_size=None, conv_k=4, seed=3001, **_):
    hidden, heads, kv_heads, head_dim, inter, vocab = 64, 4, 2, 16, 96, 256
    cfg = {
        "architectures": ["Lfm3ForCausalLM"], "model_type": "lfm3",
        "bos_token_id": 1, "eos_token_id": [2], "pad_token_id": 0,
        "conv_L_cache": conv_k, "conv_bias": False, "head_dim": head_dim, "hidden_size": hidden,
        "intermediate_size": inter, "layer_types": LAYER_TYPES, "max_position_embeddings": 4096,
        "norm_eps": 1e-05, "num_attention_heads": heads, "num_hidden_layers": len(LAYER_TYPES),
        "num_key_value_heads": kv_heads,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
        },
        "sliding_window": 5, "tie_word_embeddings": True, "use_sandwich_norm": True,
        "vocab_size": vocab, "vision_config": {},
        "dtype": {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}[dtype],
    }
    if bits:
        cfg["quantization"] = {"group_size": group_size, "bits": bits, "mode": "affine"}
        cfg["quantization_config"] = dict(cfg["quantization"])
    rng = np.random.default_rng(seed)
    raw = {}

    def tensor(name, shape, kind="matrix"):
        if kind == "norm":
            raw[name] = rng.normal(0.6, 0.15, shape).astype(np.float32)
        elif kind == "conv":
            raw[name] = rng.normal(0, 0.4, shape).astype(np.float32)
        else:
            raw[name] = rng.normal(0, 1 / np.sqrt(shape[-1]), shape).astype(np.float32)

    tensor("embed_tokens.weight", (vocab, hidden))
    raw["embed_tokens.weight"] *= np.float32(np.sqrt(hidden) * 0.5)
    tensor("embedding_norm.weight", (hidden,), "norm")
    for i, t in enumerate(LAYER_TYPES):
        p = "layers.%d." % i
        for n in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"):
            tensor(p + n + ".weight", (hidden,), "norm")
        tensor(p + "feed_forward.w1.weight", (inter, hidden))
        tensor(p + "feed_forward.w3.weight", (inter, hidden))
        tensor(p + "feed_forward.w2.weight", (hidden, inter))
        if t == "conv":
            tensor(p + "conv.in_proj.weight", (3 * hidden, hidden))
            tensor(p + "conv.out_proj.weight", (hidden, hidden))
            tensor(p + "conv.conv.weight", (hidden, conv_k, 1), "conv")  # MLX channels-last layout
        else:
            tensor(p + "self_attn.q_proj.weight", (heads * head_dim, hidden))
            tensor(p + "self_attn.k_proj.weight", (kv_heads * head_dim, hidden))
            tensor(p + "self_attn.v_proj.weight", (kv_heads * head_dim, hidden))
            tensor(p + "self_attn.o_proj.weight", (hidden, heads * head_dim))
            tensor(p + "self_attn.gate_proj.weight", (heads, hidden))
            tensor(p + "self_attn.q_norm.weight", (head_dim,), "norm")
            tensor(p + "self_attn.k_norm.weight", (head_dim,), "norm")

    tensors = {}
    for name, values in raw.items():
        quantized = bits and values.ndim == 2 and not name.endswith("conv.conv.weight")
        if quantized:
            base = name[:-len(".weight")]
            for suffix, entry in quantize(values, bits, group_size, dtype).items():
                tensors[base + "." + suffix] = entry
        else:
            _, payload = encode_float(values, dtype)
            tensors[name] = (dtype, list(values.shape), payload)

    header, data = {"__metadata__": {"format": "mlx"}}, bytearray()
    for name in sorted(tensors):
        tdtype, shape, payload = tensors[name]
        header["language_model.model." + name] = {"dtype": tdtype, "shape": shape,
                                                  "data_offsets": [len(data), len(data) + len(payload)]}
        data.extend(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(data))
    (directory / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    return cfg


def golden(directory, case):
    """NumPy float64 goldens from the bytes just written; both NumPy forms must agree."""
    ckpt = Checkpoint(directory)
    model = NumpyLFM3(ckpt, np.float64).load()
    tokens = PROMPT
    positions = list(range(len(tokens)))
    hidden, logits = model.sequence(tokens, model.new_state(), positions)
    st = model.new_state()
    steps = [model.step(t, st) for t in tokens]
    step_hidden = np.stack([h for h, _ in steps], axis=1)
    step_logits = np.stack([lg for _, lg in steps])
    form_diff = max(float(np.max(np.abs(step_hidden - hidden))), float(np.max(np.abs(step_logits - logits))))
    if not form_diff < 1e-9:
        raise SystemExit("%s: NumPy sequence and step forms disagree by %g" % (case["name"], form_diff))
    greedy = model.greedy(step_logits[-1], st, GREEDY_TOKENS)
    out = directory / "golden"
    out.mkdir(exist_ok=True)
    (out / "tokens.json").write_text(json.dumps({"tokens": tokens}) + "\n")
    (out / "positions.json").write_text(json.dumps({"positions": positions}) + "\n")
    np.save(out / "logits.npy", logits.astype(np.float32))
    np.save(out / "hidden.npy", hidden.astype(np.float32))
    (out / "greedy.json").write_text(json.dumps({"tokens": greedy}) + "\n")
    meta = {
        "case": case["name"],
        "produced_by": "tools/lfm3_tiny.py: NumPy float64 reference (tools/lfm3_reference.py NumpyLFM3), "
                       "weights decoded from the written safetensors bytes",
        "checkpoint": {k: v for k, v in case.items() if k != "name"},
        "tokens": len(tokens),
        "sliding_window": model.cfg.window, "conv_L_cache": model.cfg.conv_k,
        "covers": "prompt of %d tokens wraps the %d-slot sliding ring and the %d-wide conv state several times; "
                  "greedy continues %d more tokens" % (len(tokens), model.cfg.window, model.cfg.conv_k, GREEDY_TOKENS),
        "numpy_sequence_vs_step_max_abs_diff": form_diff,
        "greedy": "plain argmax after the last prompt token; EOS (2) included when emitted; at most 64",
    }
    return meta, hidden, logits, greedy, out


def check_mlx(directory, hidden, logits, greedy):
    """Stock mlx-vlm on the tiny checkpoint in float32: same tensors, same config keys, same answers."""
    import lfm3_reference as R
    stock = R.StockMLX(directory, "float32")
    cache = stock.cache()
    hid, lg = stock.run(PROMPT, cache, range(len(PROMPT)))
    g = stock.greedy(lg[-1], cache, [2], GREEDY_TOKENS)
    return {"hidden_max_abs_diff": max(r["max_abs_diff"] for r in layer_report(hid, hidden)),
            "logits": logit_report(lg, logits), "greedy_equal": g == greedy}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--check-mlx", action="store_true", help="also compare stock mlx-vlm (group sizes 32/64/128)")
    args = parser.parse_args()
    failures = 0
    for case in cases():
        directory = args.out / case["name"]
        if directory.exists():
            shutil.rmtree(directory)
        build(directory, **case)
        meta, hidden, logits, greedy, out = golden(directory, case)
        line = "%-14s sequence/step agree to %.1e; greedy %d tokens" % (case["name"], meta["numpy_sequence_vs_step_max_abs_diff"], len(greedy))
        if args.check_mlx and case.get("group_size", 32) in (32, 64, 128):
            r = check_mlx(directory, hidden, logits, greedy)
            meta["stock_mlx_f32_vs_numpy"] = r
            ok = r["logits"]["max_abs_diff"] < 1e-4 and r["hidden_max_abs_diff"] < 1e-4 and r["greedy_equal"]
            failures += not ok
            line += "; stock MLX f32: logits %.1e hidden %.1e greedy %s%s" % (
                r["logits"]["max_abs_diff"], r["hidden_max_abs_diff"], "equal" if r["greedy_equal"] else "DIFFERENT",
                "" if ok else "  FAIL")
        (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(line, flush=True)
    if failures:
        raise SystemExit("%d case(s) disagree with stock MLX" % failures)
    print("PASS: %d tiny LFM3 checkpoints with goldens under %s" % (len(cases()), args.out))


if __name__ == "__main__":
    main()
