#!/usr/bin/env python3
"""Headless end-to-end verification on synthetic Flint checkpoints, not real model quality.

NumPy is a test-only dependency. Generates local disposable floating and affine
INT4/INT8 MLX shards, then compares every token's Metal logits against an
independent CPU implementation using the checkpoint's decoded values.
"""
import argparse
import json
import pathlib
import struct
import subprocess
import tempfile

import numpy as np


def encode_float(values, dtype):
    """Return both the bytes written to disk and their exact float32 values."""
    if dtype == "F32":
        rounded = values.astype("<f4")
        return rounded, rounded.tobytes()
    if dtype == "F16":
        half = values.astype("<f2")
        return half.astype(np.float32), half.tobytes()
    if dtype != "BF16":
        raise ValueError("unsupported floating dtype: " + dtype)
    bits = values.astype(np.float32).view(np.uint32)
    bf = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype("<u2")
    return (bf.astype(np.uint32) << 16).view(np.float32), bf.tobytes()


def quantize_matrix(values, bits, group_size, dtype, packed_dtype):
    """MLX affine groups along input columns; least significant code comes first.

    The oracle is decoded from the packed bytes and rounded scale/bias payloads,
    never from the original floating matrix. U32 is only a little-endian view of
    the same U8 byte stream, not a different quantization scheme.
    """
    rows, cols = values.shape
    assert bits in (4, 8) and cols % group_size == 0
    assert packed_dtype in ("U8", "U32")
    groups = values.reshape(rows, cols // group_size, group_size)
    minimum, maximum = groups.min(axis=-1), groups.max(axis=-1)
    scale = np.maximum((maximum - minimum) / ((1 << bits) - 1), np.float32(1e-6))
    codes = np.clip(np.rint((groups - minimum[..., None]) / scale[..., None]), 0, (1 << bits) - 1).astype(np.uint8).reshape(rows, cols)
    if bits == 4:
        assert cols % 2 == 0
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    else:
        packed = codes.copy()
    if packed_dtype == "U32":
        assert packed.shape[-1] % 4 == 0
        packed = packed.view("<u4")
    scale_values, scale_bytes = encode_float(scale, dtype)
    bias_values, bias_bytes = encode_float(minimum, dtype)

    # Decode independently with scalar byte offsets, also checking code order.
    payload = packed.tobytes()
    unpacked = np.empty((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            bit_offset = (row * cols + col) * bits
            code = (payload[bit_offset // 8] >> (bit_offset % 8)) & ((1 << bits) - 1)
            assert code == codes[row, col]
            group = col // group_size
            unpacked[row, col] = np.float32(code) * scale_values[row, group] + bias_values[row, group]
    tensors = {
        "weight": (packed_dtype, list(packed.shape), payload),
        "scales": (dtype, list(scale.shape), scale_bytes),
        "biases": (dtype, list(minimum.shape), bias_bytes),
    }
    return unpacked, tensors


def checkpoint(directory, dtype, sharded, biased, *, bits=None, embedding_bits=None,
               group_size=2, packed_dtype="U8", indexed=True):
    # Quantized dimensions make every input row divisible by eight, including
    # expert down projections, so both INT4/U32 and INT8/U32 are representable.
    hidden, dense, moe, shared = (16, 24, 8, 16) if bits else (8, 12, 6, 10)
    cfg = dict(
        hidden_size=hidden, head_dim=4, num_attention_heads=4, num_key_value_heads=2,
        num_hidden_layers=3, intermediate_size=dense, moe_intermediate_size=moe,
        shared_expert_intermediate_size=shared, num_experts=4, num_experts_per_tok=2,
        vocab_size=16, bos_token_id=0, pad_token_id=15, eos_token_id=[15],
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        mlp_only_layers=[0], no_rope_layer_indices=[0, 2], sliding_window=3,
        max_position_embeddings=32, rms_norm_eps=1e-6, rope_theta=10000.0,
        torch_dtype={"F32": "float32", "F16": "float16", "BF16": "bfloat16"}[dtype],
    )
    if bits:
        # No per-module bit overrides or mlx_export metadata: the loader must
        # infer the embedding's different bit width from bytes per logical row.
        cfg["quantization"] = dict(bits=bits, group_size=group_size)
    rng = np.random.default_rng(4701)
    raw = {}

    def matrix(name, shape, norm=False):
        raw[name] = (rng.normal(1 if norm else 0, .1 if norm else .2, shape)).astype(np.float32)

    matrix("model.embed_tokens.weight", (cfg["vocab_size"], hidden))
    matrix("model.norm.weight", (hidden,), True)
    matrix("lm_head.weight", (cfg["vocab_size"], hidden))
    for layer in range(cfg["num_hidden_layers"]):
        p = "model.layers.%d." % layer
        for norm in ("input_layernorm", "post_attention_layernorm"):
            matrix(p + norm + ".weight", (hidden,), True)
        for proj, shape in (("q", (16, hidden)), ("k", (8, hidden)), ("v", (8, hidden)), ("o", (hidden, 16))):
            matrix(p + "self_attn." + proj + "_proj.weight", shape)
        for norm in ("q", "k"):
            matrix(p + "self_attn." + norm + "_norm.weight", (cfg["head_dim"],), True)
        matrix(p + "self_attn.softmax_offset", (cfg["num_attention_heads"],))
        mlps = [("mlp", dense)] if layer in cfg["mlp_only_layers"] else [
            ("mlp.experts.%d" % e, moe) for e in range(cfg["num_experts"])
        ] + [("mlp.shared_expert", shared)]
        for prefix, size in mlps:
            for proj, shape in (("gate", (size, hidden)), ("up", (size, hidden)), ("down", (hidden, size))):
                matrix(p + prefix + "." + proj + "_proj.weight", shape)
        if layer not in cfg["mlp_only_layers"]:
            matrix(p + "mlp.gate.weight", (cfg["num_experts"], hidden))
            if biased:
                matrix(p + "mlp.gate.expert_bias", (cfg["num_experts"],))

    decoded, tensors = {}, {}
    for name, values in raw.items():
        if bits and values.ndim == 2 and not name.endswith("mlp.gate.weight"):
            matrix_bits = (embedding_bits or bits) if name == "model.embed_tokens.weight" else bits
            decoded[name], quantized = quantize_matrix(values, matrix_bits, group_size, dtype, packed_dtype)
            prefix = name.removesuffix("weight")
            for suffix, tensor in quantized.items():
                tensors[prefix + suffix] = tensor
        else:
            decoded[name], payload = encode_float(values, dtype)
            tensors[name] = (dtype, list(values.shape), payload)

    if bits:
        # Real MLX exports stack routed experts. Keep the oracle's weights
        # unstacked, so its routing and matrix arithmetic need no loader logic.
        for layer in range(cfg["num_hidden_layers"]):
            if layer in cfg["mlp_only_layers"]:
                continue
            p = "model.layers.%d.mlp." % layer
            for proj in ("gate", "up", "down"):
                for suffix in ("weight", "scales", "biases"):
                    experts = [tensors.pop(p + "experts.%d.%s_proj.%s" % (e, proj, suffix)) for e in range(cfg["num_experts"])]
                    tensor_dtype, shape, _ = experts[0]
                    assert all(t[0] == tensor_dtype and t[1] == shape for t in experts)
                    tensors[p + "switch_mlp.%s_proj.%s" % (proj, suffix)] = (
                        tensor_dtype, [cfg["num_experts"], *shape], b"".join(t[2] for t in experts))

    if not sharded:
        filenames = ["model.safetensors"]
    elif indexed:
        filenames = ["part-1.safetensors", "part-2.safetensors"]
    else:
        filenames = ["model-00000.safetensors", "model-00001.safetensors"]
    weight_map = {}
    for shard, filename in enumerate(filenames):
        header, data = {"__metadata__": {"format": "pt"}}, bytearray()
        for i, (name, (tensor_dtype, shape, payload)) in enumerate(tensors.items()):
            if i % len(filenames) != shard:
                continue
            header[name] = dict(dtype=tensor_dtype, shape=shape, data_offsets=[len(data), len(data) + len(payload)])
            data.extend(payload)
            weight_map[name] = filename
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        (directory / filename).write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (directory / "config.json").write_text(json.dumps(cfg))
    if sharded and indexed:
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return cfg, decoded


class Reference:
    def __init__(self, cfg, weights):
        self.cfg, self.w = cfg, weights
        self.keys, self.values = ([[] for _ in range(cfg["num_hidden_layers"])] for _ in range(2))
        self.position = 0

    def norm(self, x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.cfg["rms_norm_eps"]) * weight

    def rope(self, x):
        half = x.shape[-1] // 2
        angles = self.position * np.power(np.float32(self.cfg["rope_theta"]), -2 * np.arange(half, dtype=np.float32) / x.shape[-1])
        a, b = x[..., :half], x[..., half:]
        return np.concatenate((a * np.cos(angles) - b * np.sin(angles), b * np.cos(angles) + a * np.sin(angles)), axis=-1)

    def mlp(self, prefix, x):
        gate = self.w[prefix + ".gate_proj.weight"] @ x
        up = self.w[prefix + ".up_proj.weight"] @ x
        return self.w[prefix + ".down_proj.weight"] @ (gate / (1 + np.exp(-gate)) * up)

    def forward(self, token):
        x = self.w["model.embed_tokens.weight"][token].copy()
        heads, kv_heads, head_dim = (self.cfg[k] for k in ("num_attention_heads", "num_key_value_heads", "head_dim"))
        heads_per_kv = heads // kv_heads
        for layer in range(self.cfg["num_hidden_layers"]):
            p = "model.layers.%d." % layer
            n = self.norm(x, self.w[p + "input_layernorm.weight"])
            q = (self.w[p + "self_attn.q_proj.weight"] @ n).reshape(heads, head_dim)
            k = (self.w[p + "self_attn.k_proj.weight"] @ n).reshape(kv_heads, head_dim)
            v = (self.w[p + "self_attn.v_proj.weight"] @ n).reshape(kv_heads, head_dim)
            q = self.norm(q, self.w[p + "self_attn.q_norm.weight"])
            k = self.norm(k, self.w[p + "self_attn.k_norm.weight"])
            if layer not in self.cfg["no_rope_layer_indices"]:
                q, k = self.rope(q), self.rope(k)
            self.keys[layer].append(k)
            self.values[layer].append(v)
            window = self.cfg["sliding_window"] if self.cfg["layer_types"][layer] == "sliding_attention" else self.position + 1
            keys, values = np.array(self.keys[layer][-window:]), np.array(self.values[layer][-window:])
            out = np.zeros_like(q)
            for head in range(heads):
                scores = keys[:, head // heads_per_kv] @ q[head] / np.sqrt(np.float32(head_dim))
                sink = self.w[p + "self_attn.softmax_offset"][head]
                maximum = max(float(scores.max()), float(sink))
                probs = np.exp(scores - maximum)
                out[head] = (probs @ values[:, head // heads_per_kv]) / (probs.sum() + np.exp(sink - maximum))
            x += self.w[p + "self_attn.o_proj.weight"] @ out.ravel()
            n = self.norm(x, self.w[p + "post_attention_layernorm.weight"])
            if layer in self.cfg["mlp_only_layers"]:
                x += self.mlp(p + "mlp", n)
            else:
                scores = 1 / (1 + np.exp(-(self.w[p + "mlp.gate.weight"] @ n)))
                choice = scores + self.w.get(p + "mlp.gate.expert_bias", 0)
                ids = np.argsort(-choice, kind="stable")[:self.cfg["num_experts_per_tok"]]
                probs = scores[ids] / (scores[ids].sum() + 1e-20)
                for expert, prob in zip(ids, probs):
                    x += prob * self.mlp(p + "mlp.experts.%d" % expert, n)
                x += self.mlp(p + "mlp.shared_expert", n)
        self.position += 1
        return self.w["lm_head.weight"] @ self.norm(x, self.w["model.norm.weight"])


def resident_worker_count(value):
    number = int(value)
    if not 1 <= number <= 32:
        raise argparse.ArgumentTypeError("must be in 1..32 (omit for staged execution)")
    return number


def run(binary, command, directory, tokens, *extra, resident_workers=0, resident_row_tile=1,
        staged_row_tile=1):
    if resident_row_tile not in (1, 2, 4, 8) or (resident_row_tile != 1 and not resident_workers):
        raise ValueError("nondefault resident row tile requires resident workers")
    if staged_row_tile not in (1, 2, 4, 8) or (staged_row_tile != 1 and resident_workers):
        raise ValueError("staged row tile must be 1/2/4/8; nondefault tiles require staged execution")
    resident_flags = (["--resident-workers", str(resident_workers),
                       "--resident-row-tile", str(resident_row_tile)]
                      if resident_workers else [])
    return subprocess.run([str(binary), command, str(directory), "--tokens", ",".join(map(str, tokens)), "--context", "16", *extra, *resident_flags, "--staged-row-tile", str(staged_row_tile)], text=True, capture_output=True, timeout=90)


def cases():
    """Floating regressions plus byte/word packing and mixed-width MLX exports."""
    return [
        dict(dtype="F32", sharded=False, biased=False),
        dict(dtype="F16", sharded=True, biased=True),
        dict(dtype="BF16", sharded=True, biased=False),
        dict(dtype="F16", sharded=False, biased=True, bits=4, embedding_bits=8, group_size=2, packed_dtype="U8"),
        dict(dtype="BF16", sharded=True, biased=False, bits=4, embedding_bits=8, group_size=4, packed_dtype="U32", indexed=False),
        dict(dtype="BF16", sharded=True, biased=True, bits=8, group_size=4, packed_dtype="U8"),
        dict(dtype="F16", sharded=True, biased=False, bits=4, group_size=2, packed_dtype="U32"),
    ]


def case_name(case):
    layout = "indexed shards" if case["sharded"] and case.get("indexed", True) else "no-index shards" if case["sharded"] else "single"
    label = "%s %s bias=%s" % (case["dtype"], layout, case["biased"])
    if case.get("bits"):
        label += " INT%d/embed%d %s group=%d stacked-experts" % (
            case["bits"], case.get("embedding_bits", case["bits"]), case["packed_dtype"], case["group_size"])
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
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
    binary = args.binary.resolve()
    worst = 0.0
    for case in cases():
        with tempfile.TemporaryDirectory(prefix="flint-metal-test-") as temp:
            directory = pathlib.Path(temp)
            cfg, weights = checkpoint(directory, **case)
            tokens = [0, 3, 7, 4, 9, 1, 8]  # two ring-buffer wraps, full layers retain all seven tokens
            ref = Reference(cfg, weights)
            expected = np.array([ref.forward(t) for t in tokens])
            result = run(binary, "eval", directory, tokens, resident_workers=args.resident_workers, resident_row_tile=args.resident_row_tile, staged_row_tile=args.staged_row_tile)
            if result.returncode:
                raise RuntimeError(case_name(case) + ": " + result.stderr)
            actual = np.array([json.loads(line) for line in result.stdout.splitlines()])
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5, err_msg=case_name(case))
            error = float(np.max(np.abs(actual - expected)))
            worst = max(worst, error)
            generated, logits = [], expected[-1]
            for i in range(4):
                token = int(np.argmax(logits))
                generated.append(token)
                if token in cfg["eos_token_id"]:
                    break
                if i < 3:
                    logits = ref.forward(token)
            result = run(binary, "generate", directory, tokens, "--max-tokens", "4", resident_workers=args.resident_workers, resident_row_tile=args.resident_row_tile, staged_row_tile=args.staged_row_tile)
            if result.returncode:
                raise RuntimeError(case_name(case) + ": " + result.stderr)
            assert json.loads(result.stdout)["tokens"] == generated, case_name(case)
            for bad, extra, error_name in (([16], (), "InvalidToken"), (tokens, ("--context", "2"), "ContextExhausted")):
                result = run(binary, "eval", directory, bad, *extra, resident_workers=args.resident_workers, resident_row_tile=args.resident_row_tile, staged_row_tile=args.staged_row_tile)
                assert result.returncode != 0 and error_name in result.stderr, result.stderr
            print("%s: all 7 token logits and greedy decode agree (max abs %.3g)" % (case_name(case), error))
    mode = ("resident single-dispatch (%d workers, row tile %d)" %
            (args.resident_workers, args.resident_row_tile)) if args.resident_workers else "staged (row tile %d)" % args.staged_row_tile
    print("PASS: real headless Metal vs independent NumPy Flint; worst absolute error %.3g; execution=%s" % (worst, mode))


if __name__ == "__main__":
    main()
