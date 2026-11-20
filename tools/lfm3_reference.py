#!/usr/bin/env python3
"""LFM3 reference outputs: stock MLX teacher-forced extraction, an independent NumPy model, goldens.

Run with the reference venv (mlx, mlx-vlm >= 0.7.11, numpy, transformers, tokenizers;
no PyTorch). The NumPy half imports nothing else, so tools/lfm3_tiny.py can reuse it
with any NumPy. Never imports or executes code shipped with a checkpoint.

  python -I tools/lfm3_reference.py inspect CKPT          # header table, dequant vs mx.dequantize
  python -I tools/lfm3_reference.py compare CKPT --prompt "Hi"   # MLX vs NumPy, per layer
  python -I tools/lfm3_reference.py window-rule CKPT      # settle the sliding-window rule
  python -I tools/lfm3_reference.py goldens CKPT --out DIR --cases short,window,long
  python -I tools/lfm3_reference.py check GOLDEN_CASE_DIR CANDIDATE_DIR   # score an engine's outputs

NumPy computes in float64 by default (--numpy-dtype float32 is available): weights are
dequantized exactly from the packed codes and the BF16 scales/biases, so its difference
from stock MLX float32 is MLX's float32 rounding. The rotary angle is formed in float32
exactly as both reference implementations do (see docs/LFM3.md).
"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import struct
import sys
import time

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PREFIXES = ("language_model.model.", "model.")
GREEDY_TOKENS = 64
WINDOW_POSITIONS = (0, 1, 5, 6, 7, 254, 255, 256, 257, 258, 511, 512, 513)


# ---------------------------------------------------------------- checkpoint

def decode_float(raw, dtype):
    """Exact float32 values of an F32/F16/BF16 payload."""
    if dtype == "F32":
        return raw.view("<f4").astype(np.float32)
    if dtype == "F16":
        return raw.view("<f2").astype(np.float32)
    if dtype == "BF16":
        return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
    raise ValueError("not a floating dtype: " + dtype)


def unpack_codes(packed_bytes, rows, cols, bits):
    """MLX affine packing: a little-endian bit stream per row, element 0 in the lowest bits."""
    b = packed_bytes.reshape(rows, -1)
    if bits == 8:
        codes = b
    elif bits == 4:
        codes = np.stack((b & 0x0F, b >> 4), axis=-1).reshape(rows, -1)
    elif bits == 2:
        codes = np.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(rows, -1)
    else:
        raise ValueError("unsupported bit width %d" % bits)
    assert codes.shape == (rows, cols), (codes.shape, rows, cols)
    return codes


class Checkpoint:
    """Reads safetensors headers itself and maps payloads with np.memmap (no safetensors import)."""

    def __init__(self, directory):
        self.directory = pathlib.Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        index = self.directory / "model.safetensors.index.json"
        if index.is_file():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.directory.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError("no safetensors in %s" % self.directory)
        self.entries, self.prefix = {}, None
        for filename in files:
            path = self.directory / filename
            with open(path, "rb") as f:
                (length,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(length))
            data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + length)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                prefix = next((p for p in PREFIXES if name.startswith(p)), None)
                if prefix is None:
                    raise ValueError("unexpected tensor name " + name)
                if self.prefix not in (None, prefix):
                    raise ValueError("mixed tensor prefixes")
                self.prefix = prefix
                start, end = meta["data_offsets"]
                self.entries[name[len(prefix):]] = (meta["dtype"], tuple(meta["shape"]), data[start:end], filename)
        q = self.config.get("quantization") or self.config.get("quantization_config")
        self.group_size = q["group_size"] if q else None
        if q and q.get("mode", "affine") != "affine":
            raise ValueError("only MLX affine quantization is supported")

    def header_table(self):
        return [(n, d, s) for n, (d, s, _, _) in sorted(self.entries.items())]

    def quantized(self, base):
        return base + ".scales" in self.entries

    def floats(self, name):
        dtype, shape, raw, _ = self.entries[name]
        return decode_float(np.asarray(raw), dtype).reshape(shape)

    def quant_parts(self, base):
        wdtype, wshape, raw, _ = self.entries[base + ".weight"]
        scales, biases = self.floats(base + ".scales"), self.floats(base + ".biases")
        rows, groups = scales.shape
        cols = groups * self.group_size
        word_bits = {"U32": 32, "U8": 8}[wdtype]
        bits = wshape[-1] * word_bits // cols
        assert wshape[-1] * word_bits == bits * cols, (base, wshape, cols)
        codes = unpack_codes(np.asarray(raw), rows, cols, bits)
        return codes, scales, biases, bits

    def matrix(self, base, dtype=np.float64, rows=None):
        """Dequantized [out, in] matrix: w = scale[g] * code + bias[g], g = column // group_size.

        In float64 this is exact; in float32 the product is exact and the add rounds once.
        """
        if not self.quantized(base):
            w = self.floats(base + ".weight").astype(dtype)
            return w if rows is None else w[rows]
        codes, scales, biases, _ = self.quant_parts(base)
        if rows is not None:
            codes, scales, biases = codes[rows], scales[rows], biases[rows]
        gs = self.group_size
        shape = codes.shape
        g = codes.reshape(shape[0], -1, gs).astype(dtype)
        w = g * scales.astype(dtype)[..., None] + biases.astype(dtype)[..., None]
        return w.reshape(shape)

    def conv_weight(self, layer, dtype=np.float64):
        """[hidden, K] with w[d, k] multiplying u[t + k - (K - 1)]; MLX stores [hidden, K, 1]."""
        w = self.floats("layers.%d.conv.conv.weight" % layer)
        if w.ndim == 3 and w.shape[2] == 1:
            w = w[:, :, 0]
        elif w.ndim == 3 and w.shape[1] == 1:  # PyTorch layout [hidden, 1, K]
            w = w[:, 0, :]
        return w.astype(dtype)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- NumPy model

class Config:
    def __init__(self, raw):
        self.raw = raw
        self.hidden = raw["hidden_size"]
        self.layers = raw["num_hidden_layers"]
        self.heads = raw["num_attention_heads"]
        self.kv_heads = raw["num_key_value_heads"]
        self.head_dim = raw.get("head_dim") or self.hidden // self.heads
        self.vocab = raw["vocab_size"]
        self.eps = raw["norm_eps"]
        self.conv_k = raw["conv_L_cache"]
        self.window = raw.get("sliding_window")
        self.layer_types = raw["layer_types"]
        self.rope = raw["rope_parameters"]
        eos = raw.get("eos_token_id", [])
        self.eos = eos if isinstance(eos, list) else [eos]
        if not raw.get("tie_word_embeddings", True):
            raise ValueError("only the tied output head is implemented")
        if not raw.get("use_sandwich_norm", True) or raw.get("conv_bias", False):
            raise ValueError("this reference implements sandwich norm and bias-free conv only")


def rope_inv_freq(theta, partial, head_dim):
    """float32, as stock MLX: theta ** (-(2 i) / head_dim) for i < A, zero for the unrotated tail."""
    angles = int(head_dim * partial) // 2
    i = np.arange(angles, dtype=np.float32)
    inv = np.power(np.float32(theta), (-i * np.float32(2)) / np.float32(head_dim)).astype(np.float32)
    return np.concatenate([inv, np.zeros(head_dim // 2 - angles, np.float32)])


class NumpyLFM3:
    """Independent forward written from docs/LFM3.md, in float64 (default) or float32.

    One State serves both forms: `sequence` (a token batch with explicit causal and window
    masks over absolute positions, reading earlier state) and `step` (one token, the sliding
    layers' ring of `window` slots and the conv shift register, no masks at all).
    """

    def __init__(self, ckpt, dtype=np.float64, window=None):
        self.ckpt, self.dtype = ckpt, dtype
        self.cfg = Config(ckpt.config)
        self.window = window or self.cfg.window  # positions a query sees, itself included
        self.w = {}
        self.inv_freq = {}
        for t in ("full_attention", "sliding_attention"):
            if t in self.cfg.rope:
                p = self.cfg.rope[t]
                if p.get("rope_type", "default") != "default":
                    raise ValueError("only default RoPE")
                self.inv_freq[t] = rope_inv_freq(p["rope_theta"], p.get("partial_rotary_factor", 1.0), self.cfg.head_dim)

    def load(self):
        c, d = self.cfg, self.dtype
        get = lambda base: self.ckpt.matrix(base, d)
        norm = lambda name: self.ckpt.floats(name + ".weight").astype(d)
        self.w["embed"] = get("embed_tokens")
        self.w["embedding_norm"] = norm("embedding_norm")
        for i, t in enumerate(c.layer_types):
            p = "layers.%d." % i
            for n in ("operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"):
                self.w[p + n] = norm(p + n)
            for n in ("w1", "w2", "w3"):
                self.w[p + n] = get(p + "feed_forward." + n)
            if t == "conv":
                self.w[p + "in_proj"] = get(p + "conv.in_proj")
                self.w[p + "out_proj"] = get(p + "conv.out_proj")
                self.w[p + "conv"] = self.ckpt.conv_weight(i, d)
            else:
                for n in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
                    self.w[p + n] = get(p + "self_attn." + n)
                for n in ("q_norm", "k_norm"):
                    self.w[p + n] = norm(p + "self_attn." + n)
        return self

    # -- helpers
    def rms(self, x, weight):
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.cfg.eps) * weight

    def rope(self, x, positions, layer_type):
        """x [T, heads, D]; rotate-half pairs (i, i + D/2); angle = float32(pos) * inv_freq in float32."""
        angle = (np.asarray(positions, np.float32)[:, None] * self.inv_freq[layer_type][None, :]).astype(np.float32)
        cos = np.cos(angle.astype(self.dtype))[:, None, :]
        sin = np.sin(angle.astype(self.dtype))[:, None, :]
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def mlp(self, p, x):
        g = x @ self.w[p + "w1"].T
        return (g * self.sigmoid(g) * (x @ self.w[p + "w3"].T)) @ self.w[p + "w2"].T

    def qkv(self, p, x, positions, layer_type):
        c = self.cfg
        T = x.shape[0]
        q = self.rms((x @ self.w[p + "q_proj"].T).reshape(T, c.heads, c.head_dim), self.w[p + "q_norm"])
        k = self.rms((x @ self.w[p + "k_proj"].T).reshape(T, c.kv_heads, c.head_dim), self.w[p + "k_norm"])
        v = (x @ self.w[p + "v_proj"].T).reshape(T, c.kv_heads, c.head_dim)
        return self.rope(q, positions, layer_type), self.rope(k, positions, layer_type), v

    def gate_out(self, p, x, attn):
        g = self.sigmoid(x @ self.w[p + "gate_proj"].T)  # [T, heads]
        return (attn * g[:, :, None]).reshape(x.shape[0], -1) @ self.w[p + "o_proj"].T

    def new_state(self):
        c = self.cfg
        st = {"position": 0, "conv": {}, "keys": {}, "values": {}, "ring_k": {}, "ring_v": {}}
        for i, t in enumerate(c.layer_types):
            if t == "conv":
                st["conv"][i] = np.zeros((c.conv_k - 1, c.hidden), self.dtype)
            elif t == "full_attention":
                st["keys"][i], st["values"][i] = [], []
            else:
                st["ring_k"][i] = np.zeros((self.window, c.kv_heads, c.head_dim), self.dtype)
                st["ring_v"][i] = np.zeros((self.window, c.kv_heads, c.head_dim), self.dtype)
        return st

    def ring_history(self, st, i):
        """Cached sliding keys/values in temporal order, with their absolute positions."""
        n = st["position"]
        pos = np.arange(max(0, n - self.window), n)
        slots = pos % self.window
        return st["ring_k"][i][slots], st["ring_v"][i][slots], pos

    def logits(self, h):
        return self.rms(h, self.w["embedding_norm"]) @ self.w["embed"].T

    # -- form 1: a token batch with masks
    def sequence(self, tokens, st, logits_at=(), keep_hidden=True):
        """Advance `st` by len(tokens). Returns (hidden [L+1, T, H] or None, logits [len(logits_at), V])."""
        c = self.cfg
        T = len(tokens)
        p0 = st["position"]
        positions = np.arange(p0, p0 + T)
        x = self.w["embed"][np.asarray(tokens)].copy()
        hidden = [x.copy()] if keep_hidden else None
        for i, t in enumerate(c.layer_types):
            p = "layers.%d." % i
            n = self.rms(x, self.w[p + "operator_norm"])
            if t == "conv":
                bcx = n @ self.w[p + "in_proj"].T
                B, C, X = bcx[:, :c.hidden], bcx[:, c.hidden:2 * c.hidden], bcx[:, 2 * c.hidden:]
                u = np.concatenate([st["conv"][i], B * X], axis=0)  # K-1 carried values, then this batch
                wk = self.w[p + "conv"]
                v = sum(wk[:, k] * u[k:k + T] for k in range(c.conv_k))
                st["conv"][i] = u[T:]
                y = (C * v) @ self.w[p + "out_proj"].T
            else:
                q, k, v = self.qkv(p, n, positions, t)
                if t == "full_attention":
                    st["keys"][i].extend(k)
                    st["values"][i].extend(v)
                    K, V = np.array(st["keys"][i]), np.array(st["values"][i])
                    kpos = np.arange(len(K))
                    allowed = kpos[None, :] <= positions[:, None]
                else:
                    hk, hv, hpos = self.ring_history(st, i)
                    K, V = np.concatenate([hk, k]), np.concatenate([hv, v])
                    kpos = np.concatenate([hpos, positions])
                    d = positions[:, None] - kpos[None, :]
                    allowed = (d >= 0) & (d < self.window)
                    slots = positions % self.window
                    st["ring_k"][i][slots[-self.window:]] = k[-self.window:]
                    st["ring_v"][i][slots[-self.window:]] = v[-self.window:]
                rep = c.heads // c.kv_heads
                out = np.empty_like(q)
                scale = 1 / np.sqrt(self.dtype(c.head_dim))
                for h in range(c.heads):
                    s = (q[:, h] @ K[:, h // rep].T) * scale
                    s = np.where(allowed, s, -np.inf)
                    e = np.exp(s - s.max(axis=1, keepdims=True))
                    out[:, h] = (e / e.sum(axis=1, keepdims=True)) @ V[:, h // rep]
                y = self.gate_out(p, n, out)
            x = x + self.rms(y, self.w[p + "post_operator_norm"])
            x = x + self.rms(self.mlp(p, self.rms(x, self.w[p + "ffn_norm"])), self.w[p + "post_ffn_norm"])
            if keep_hidden:
                hidden.append(x.copy())
        st["position"] = p0 + T
        idx = np.asarray(logits_at, dtype=np.int64)
        lg = self.logits(x[idx]) if len(idx) else np.zeros((0, c.vocab), self.dtype)
        return (np.stack(hidden) if keep_hidden else None), lg

    # -- form 2: one token, caches only
    def step(self, token, st):
        """Advance `st` by one token. Returns (hidden [L+1, H], logits [V])."""
        c = self.cfg
        pos = st["position"]
        x = self.w["embed"][token].copy()
        hidden = [x.copy()]
        for i, t in enumerate(c.layer_types):
            p = "layers.%d." % i
            n = self.rms(x, self.w[p + "operator_norm"])
            if t == "conv":
                bcx = self.w[p + "in_proj"] @ n
                B, C, X = bcx[:c.hidden], bcx[c.hidden:2 * c.hidden], bcx[2 * c.hidden:]
                window = np.vstack([st["conv"][i], (B * X)[None]])  # oldest first, newest last
                v = np.einsum("dk,kd->d", self.w[p + "conv"], window)
                st["conv"][i] = window[1:]
                y = self.w[p + "out_proj"] @ (C * v)
            else:
                q, k, v = (a[0] for a in self.qkv(p, n[None], [pos], t))
                if t == "full_attention":
                    st["keys"][i].append(k)
                    st["values"][i].append(v)
                    K, V = np.array(st["keys"][i]), np.array(st["values"][i])
                else:
                    slot = pos % self.window
                    st["ring_k"][i][slot], st["ring_v"][i][slot] = k, v
                    filled = min(pos + 1, self.window)
                    K, V = st["ring_k"][i][:filled], st["ring_v"][i][:filled]
                rep = c.heads // c.kv_heads
                out = np.empty_like(q)
                for h in range(c.heads):
                    s = K[:, h // rep] @ q[h] / np.sqrt(self.dtype(c.head_dim))
                    e = np.exp(s - s.max())
                    out[h] = (e / e.sum()) @ V[:, h // rep]
                y = self.gate_out(p, n[None], out[None])[0]
            x = x + self.rms(y, self.w[p + "post_operator_norm"])
            x = x + self.rms(self.mlp(p, self.rms(x, self.w[p + "ffn_norm"])[None])[0], self.w[p + "post_ffn_norm"])
            hidden.append(x.copy())
        st["position"] = pos + 1
        return np.stack(hidden), self.logits(x)

    def greedy(self, logits, st, count=GREEDY_TOKENS):
        """Plain argmax continuation; every emitted token (EOS included) is forwarded except the last."""
        out = []
        for n in range(count):
            token = int(np.argmax(logits))
            out.append(token)
            if token in self.cfg.eos or n == count - 1:
                break
            _, logits = self.step(token, st)
        return out


# ---------------------------------------------------------------- stock MLX

class StockMLX:
    """Stock mlx-vlm LFM3, driven layer by layer with the stock modules and mask helpers.

    The loop below is Lfm3Model.__call__ verbatim plus recording; `check_loop` asserts that it
    reproduces the stock model's own logits bit for bit.
    """

    def __init__(self, checkpoint, dtype="float32"):
        import mlx.core as mx
        from mlx_vlm.utils import load_model
        from mlx_vlm.models.lfm3 import language as lang
        self.mx, self.lang = mx, lang
        self.model = load_model(pathlib.Path(checkpoint))
        self.lm = self.model.language_model
        self.dtype_name = dtype
        self.model.set_dtype(getattr(mx, dtype))
        mx.eval(self.model.parameters())

    def cache(self):
        return self.lm.make_cache()

    def run(self, tokens, cache, logits_at=(), keep_hidden=True):
        mx, lang, inner = self.mx, self.lang, self.lm.model
        h = inner.embed_tokens(mx.array([list(tokens)], dtype=mx.int32))
        mx.eval(h)
        hidden = [np.array(h[0].astype(mx.float32))] if keep_hidden else None
        for layer, c in zip(inner.layers, cache):
            if layer.is_attention_layer:
                window = inner.config.sliding_window if layer.layer_type == "sliding_attention" else None
                mask = lang.create_attention_mask(h, c, window_size=window)
            else:
                mask = lang.create_ssm_mask(h, c)
            h = layer(h, mask, c, None)
            mx.eval(h)
            if keep_hidden:
                hidden.append(np.array(h[0].astype(mx.float32)))
        normed = inner.embedding_norm(h)
        idx = list(logits_at)
        logits = np.zeros((0, self.lm.config.vocab_size), np.float32)
        if idx:
            lg = inner.embed_tokens.as_linear(normed[:, mx.array(idx)])
            mx.eval(lg)
            logits = np.array(lg[0].astype(mx.float32))
        return (np.stack(hidden) if keep_hidden else None), logits

    def full_logits(self, tokens, cache):
        out = self.model(self.mx.array([list(tokens)], dtype=self.mx.int32), cache=cache).logits
        self.mx.eval(out)
        return np.array(out[0].astype(self.mx.float32))

    def check_loop(self, tokens):
        """Stock model call vs this loop on the same tokens: must be identical."""
        a = self.full_logits(tokens, self.cache())
        _, b = self.run(tokens, self.cache(), logits_at=range(len(tokens)), keep_hidden=False)
        return float(np.max(np.abs(a - b)))

    def greedy(self, logits, cache, eos, count=GREEDY_TOKENS):
        mx = self.mx
        out = []
        for n in range(count):
            token = int(np.argmax(logits))
            out.append(token)
            if token in eos or n == count - 1:
                break
            lg = self.model(mx.array([[token]], dtype=mx.int32), cache=cache).logits
            mx.eval(lg)
            logits = np.array(lg[0, -1].astype(mx.float32))
        return out


def versions():
    import importlib.metadata as md
    v = {"python": platform.python_version(), "numpy": np.__version__, "machine": platform.machine(),
         "macos": platform.mac_ver()[0]}
    for pkg in ("mlx", "mlx-vlm", "transformers", "tokenizers"):
        try:
            v[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return v


# ---------------------------------------------------------------- prompts

def tokenizer(checkpoint):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    tok.chat_template = (pathlib.Path(checkpoint) / "chat_template.jinja").read_text()
    return tok


def chat_tokens(tok, text, enable_thinking=False):
    """The checkpoint's own template; thinking off, as mlx-vlm renders it by default."""
    return list(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                        add_generation_prompt=True, return_dict=False,
                                        enable_thinking=enable_thinking))


SHORT_PROMPT = "What is the capital of Australia? Answer in one sentence."


def case_definition(name, checkpoint, tok):
    """Token ids, stored positions and a human description for each golden case."""
    if name == "short":
        tokens = chat_tokens(tok, SHORT_PROMPT)
        return tokens, list(range(len(tokens))), {
            "prompt": "chat template (enable_thinking=False), one user message: " + json.dumps(SHORT_PROMPT)}
    if name == "window":
        words = (pathlib.Path(checkpoint) / "LICENSE").read_text().split()
        target = 700
        lo, hi = 1, len(words)
        while lo < hi:  # largest word prefix whose chat-templated prompt fits in `target` tokens
            mid = (lo + hi + 1) // 2
            text = " ".join(words[:mid]) + "\n\nSummarize the license text above in two sentences."
            if len(chat_tokens(tok, text)) <= target:
                lo = mid
            else:
                hi = mid - 1
        text = " ".join(words[:lo]) + "\n\nSummarize the license text above in two sentences."
        tokens = chat_tokens(tok, text)
        positions = sorted(set(WINDOW_POSITIONS) | {len(tokens) - 1})
        return tokens, positions, {
            "prompt": ("chat template (enable_thinking=False), one user message: the first %d "
                       "whitespace-separated words of the checkpoint's LICENSE joined by single spaces, "
                       "then \"\\n\\nSummarize the license text above in two sentences.\"" % lo)}
    if name == "long":
        import textwrap
        source = pathlib.Path(textwrap.__file__)
        body = tok.encode(source.read_text(), add_special_tokens=False)
        body = (body * (4096 // len(body) + 1))[:4095]
        tokens = [tok.bos_token_id] + body
        return tokens, list(range(4092, 4096)), {
            "prompt": ("raw, no chat template: BOS then the tokens of Python %s's textwrap.py "
                       "(sha256 %s), repeated as needed and cut to 4,096 tokens in total"
                       % (platform.python_version(), file_sha256(source)))}
    raise ValueError(name)


# ---------------------------------------------------------------- comparisons

def layer_report(a, b):
    """Max |a - b| per hidden index and the max |b| it is relative to."""
    return [{"index": i, "max_abs_diff": float(np.max(np.abs(a[i] - b[i]))),
             "max_abs_value": float(np.max(np.abs(b[i])))} for i in range(a.shape[0])]


def logit_report(a, b):
    d = np.abs(a - b)
    return {"max_abs_diff": float(d.max()), "rms_diff": float(np.sqrt(np.mean((a - b) ** 2))),
            "max_abs_logit": float(np.max(np.abs(b))),
            "top1_agree": int(np.sum(a.argmax(-1) == b.argmax(-1))), "rows": int(a.shape[0])}


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- commands

def cmd_inspect(args):
    import mlx.core as mx
    ckpt = Checkpoint(args.checkpoint)
    rows = ckpt.header_table()
    print("tensors: %d, prefix %r, group_size %s" % (len(rows), ckpt.prefix, ckpt.group_size))
    worst, count = 0.0, 0
    for name, dtype, shape in rows:
        if not name.endswith(".scales"):
            continue
        base = name[:-len(".scales")]
        ours = ckpt.matrix(base, np.float32)
        _, _, _, bits = ckpt.quant_parts(base)
        full = ckpt.prefix + base
        from mlx.core import load as mxload  # stock reader, used only for this cross-check
        if count == 0:
            stock_all = mxload(str(ckpt.directory / ckpt.entries[base + ".weight"][3]))
        theirs = mx.dequantize(stock_all[full + ".weight"], stock_all[full + ".scales"].astype(mx.float32),
                               stock_all[full + ".biases"].astype(mx.float32), group_size=ckpt.group_size, bits=bits)
        diff = float(np.max(np.abs(np.array(theirs) - ours)))
        worst = max(worst, diff)
        count += 1
    print("dequantized %d quantized matrices; max |mx.dequantize(f32) - ours(f32)| = %g" % (count, worst))
    if args.table:
        for name, dtype, shape in rows:
            print(name, dtype, list(shape))


def cmd_compare(args):
    """Stock MLX (float32 and bfloat16) vs NumPy on one token sequence, per layer."""
    tok = tokenizer(args.checkpoint)
    tokens = [int(t) for t in args.tokens.split(",")] if args.tokens else chat_tokens(tok, args.prompt)
    ckpt = Checkpoint(args.checkpoint)
    report = run_case(args.checkpoint, ckpt, tokens, list(range(len(tokens))), args, step_from=0)
    report.pop("arrays")
    print(json.dumps(report, indent=1))


def cmd_window_rule(args):
    """NumPy with windows 255/256/257 (positions seen, itself included) vs stock MLX float32."""
    tok = tokenizer(args.checkpoint)
    tokens, _, _ = case_definition("window", args.checkpoint, tok)
    positions = list(range(len(tokens)))
    stock = StockMLX(args.checkpoint, "float32")
    hid_mlx, lg_mlx = stock.run(tokens, stock.cache(), positions)
    del stock
    ckpt = Checkpoint(args.checkpoint)
    model = NumpyLFM3(ckpt, np.float64).load()
    sliding = [i + 1 for i, t in enumerate(model.cfg.layer_types) if t == "sliding_attention"]
    result = {"tokens": len(tokens)}
    for w in (255, 256, 257):
        model.window = w
        hid, lg = model.sequence(tokens, model.new_state(), positions)
        first = sliding[0]
        early = float(np.max(np.abs(hid[first][:255] - hid_mlx[first][:255])))
        late = float(np.max(np.abs(hid[first][257:] - hid_mlx[first][257:])))
        result["window_%d" % w] = {
            "first_sliding_layer_output_index": first,
            "max_abs_diff_positions_0_254": early,
            "max_abs_diff_positions_257_on": late,
            "logits_positions_257_on": logit_report(lg[257:], lg_mlx[257:]),
        }
        log(w, early, late)
    print(json.dumps(result, indent=1))


def run_case(checkpoint, ckpt, tokens, positions, args, step_from=0, keep_hidden=True, greedy=True):
    """Everything for one token sequence; returns a report and the arrays to store."""
    eos = Config(ckpt.config).eos
    T = len(tokens)
    rep = {"tokens": T, "positions": len(positions)}
    t0 = time.time()

    # Stock MLX float32: batched prefill with the stock caches, recorded per layer, then greedy.
    stock = StockMLX(checkpoint, "float32")
    if T <= 1024:
        rep["mlx_loop_vs_stock_call_max_abs"] = stock.check_loop(tokens)
    cache = stock.cache()
    hid_mlx, lg_all = stock.run(tokens, cache, sorted(set(positions) | {T - 1}), keep_hidden)
    order = sorted(set(positions) | {T - 1})
    lg_mlx = lg_all[[order.index(p) for p in positions]]
    greedy_mlx = stock.greedy(lg_all[-1], cache, eos) if greedy else None
    log("mlx float32 prefill + greedy: %.1fs" % (time.time() - t0))

    # Stock MLX float32 one token at a time through the same stock caches.
    cache = stock.cache()
    rows = {}
    for n, token in enumerate(tokens):
        _, lg = stock.run([token], cache, [0] if n in positions else [], keep_hidden=False)
        if n in positions:
            rows[n] = lg[0]
    rep["mlx_f32_tokenwise_vs_batched"] = logit_report(np.array([rows[p] for p in positions]), lg_mlx)
    log("mlx float32 token-at-a-time: %.1fs" % (time.time() - t0))

    # Stock MLX at the checkpoint's own dtype (bfloat16), batched.
    stock_bf = StockMLX(checkpoint, "bfloat16")
    hid_bf, lg_bf = stock_bf.run(tokens, stock_bf.cache(), positions, keep_hidden=False)
    rep["mlx_bf16_vs_f32"] = logit_report(lg_bf, lg_mlx)
    del stock, stock_bf
    log("mlx bfloat16: %.1fs" % (time.time() - t0))

    # NumPy: whole sequence, then (from step_from) one token at a time with caches.
    dtype = np.dtype(args.numpy_dtype).type
    model = NumpyLFM3(ckpt, dtype).load()
    log("numpy weights: %.1fs" % (time.time() - t0))
    hid_np, lg_np = model.sequence(tokens, model.new_state(), positions, keep_hidden)
    log("numpy sequence: %.1fs" % (time.time() - t0))
    st = model.new_state()
    if step_from:
        model.sequence(tokens[:step_from], st, (), keep_hidden=False)
    step_hidden, step_logits = {}, {}
    for n in range(step_from, T):
        h, lg = model.step(tokens[n], st)
        if keep_hidden:
            step_hidden[n] = h
        if n in positions:
            step_logits[n] = lg
    log("numpy step: %.1fs" % (time.time() - t0))
    stepped = [p for p in positions if p >= step_from]
    rep["numpy_dtype"] = args.numpy_dtype
    rep["numpy_step_from"] = step_from
    rep["numpy_step_vs_sequence"] = {
        "logits": logit_report(np.array([step_logits[p] for p in stepped]),
                               lg_np[[positions.index(p) for p in stepped]]),
    }
    if keep_hidden:
        hs = np.stack([step_hidden[n] for n in range(step_from, T)], axis=1)
        rep["numpy_step_vs_sequence"]["hidden_max_abs_diff"] = float(np.max(np.abs(hs - hid_np[:, step_from:])))
    rep["mlx_f32_vs_numpy"] = {"logits": logit_report(lg_mlx, lg_np)}
    if keep_hidden:
        rep["mlx_f32_vs_numpy"]["hidden"] = layer_report(hid_mlx, hid_np)
        rep["mlx_f32_vs_numpy"]["hidden_rms_diff_last_layer"] = float(np.sqrt(np.mean((hid_mlx[-1] - hid_np[-1]) ** 2)))
    if args.numpy_dtype == "float64":
        # The float32 noise floor: the same NumPy model in float32 against itself in float64.
        m32 = NumpyLFM3(ckpt, np.float32).load()
        hid_32, lg_32 = m32.sequence(tokens, m32.new_state(), positions, keep_hidden)
        del m32
        rep["numpy_f32_vs_numpy_f64"] = {"logits": logit_report(lg_32, lg_np)}
        if keep_hidden:
            rep["numpy_f32_vs_numpy_f64"]["hidden_max_abs_diff_per_layer"] = [
                float(np.max(np.abs(hid_32[i] - hid_np[i]))) for i in range(hid_np.shape[0])]
            rep["numpy_f32_vs_numpy_f64"]["hidden_rms_diff_last_layer"] = float(
                np.sqrt(np.mean((hid_32[-1] - hid_np[-1]) ** 2)))
        log("numpy float32 floor: %.1fs" % (time.time() - t0))
    if greedy:
        greedy_np = model.greedy(step_logits[T - 1] if T - 1 in step_logits else lg_np[-1], st)
        rep["greedy_mlx_f32"] = greedy_mlx
        rep["greedy_numpy"] = greedy_np
        agree = 0
        for a, b in zip(greedy_mlx, greedy_np):
            if a != b:
                break
            agree += 1
        rep["greedy_common_prefix"] = agree
    log("numpy greedy: %.1fs" % (time.time() - t0))
    rep["arrays"] = {"hidden": hid_mlx, "logits": lg_mlx, "greedy": greedy_mlx}
    return rep


def cmd_goldens(args):
    out = pathlib.Path(args.out)
    tok = tokenizer(args.checkpoint)
    ckpt = Checkpoint(args.checkpoint)
    weights_file = pathlib.Path(args.checkpoint) / "model.safetensors"
    sha = file_sha256(weights_file) if weights_file.is_file() else None
    for name in args.cases.split(","):
        tokens, positions, desc = case_definition(name, args.checkpoint, tok)
        log("case %s: %d tokens, %d positions" % (name, len(tokens), len(positions)))
        long = name == "long"
        # Long: hidden states are compared but not stored; NumPy steps only the last 4 tokens
        # (after a batched NumPy prefill) and the greedy continuation.
        rep = run_case(args.checkpoint, ckpt, tokens, positions, args,
                       step_from=len(tokens) - 4 if long else 0)
        arrays = rep.pop("arrays")
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "tokens.json").write_text(json.dumps({"tokens": tokens}) + "\n")
        (d / "positions.json").write_text(json.dumps({"positions": positions}) + "\n")
        np.save(d / "logits.npy", arrays["logits"].astype(np.float32))
        if not long:
            np.save(d / "hidden.npy", arrays["hidden"].astype(np.float32))
        elif (d / "hidden.npy").exists():
            (d / "hidden.npy").unlink()
        (d / "greedy.json").write_text(json.dumps({"tokens": arrays["greedy"]}) + "\n")
        hidden_diffs = rep["mlx_f32_vs_numpy"].get("hidden")
        floor = rep.get("numpy_f32_vs_numpy_f64", {})
        meta = {
            "case": name,
            "description": desc["prompt"],
            "produced_by": ("stock mlx-vlm LFM3 (mlx_vlm.models.lfm3), model.set_dtype(float32): "
                            "float32 activations, scales, biases, norms and conv weights; packed 4-bit codes "
                            "unchanged. Batched prefill through the stock caches (make_cache), driven layer by "
                            "layer with the stock modules to record the residual stream; greedy decode through "
                            "the same caches."),
            "checkpoint": str(pathlib.Path(args.checkpoint).name),
            "checkpoint_sha256": sha,
            "versions": versions(),
            "tokens": len(tokens),
            "logit_positions": positions,
            "hidden": None if long else "[25, %d, 2048]: embedding output, then the residual stream after each layer" % len(tokens),
            "greedy": "plain argmax from the logits after the last prompt token; EOS (124900) is included "
                      "when emitted and ends the list; at most 64 tokens",
            "max_abs_diff_mlx_f32_vs_numpy_logits": rep["mlx_f32_vs_numpy"]["logits"]["max_abs_diff"],
            "max_abs_diff_mlx_f32_vs_numpy_hidden": (max(r["max_abs_diff"] for r in hidden_diffs)
                                                     if hidden_diffs else None),
            "float32_noise_floor_logits": floor.get("logits", {}).get("max_abs_diff"),
            "float32_noise_floor_hidden": (max(floor["hidden_max_abs_diff_per_layer"])
                                           if "hidden_max_abs_diff_per_layer" in floor else None),
            "noise_floor_note": ("float32_noise_floor_* is the same NumPy model run in float32 against itself in "
                                 "float64: the error an exact float32 implementation of this model makes. "
                                 "Stock MLX float32 sits at that floor."),
            "report": rep,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        log("wrote %s" % d)
        print(json.dumps({k: meta[k] for k in ("case", "tokens", "max_abs_diff_mlx_f32_vs_numpy_logits",
                                               "max_abs_diff_mlx_f32_vs_numpy_hidden")}), flush=True)


def cmd_chunked(args):
    """Chunked prefill against the stored `long` golden: stock MLX in several chunk sizes, NumPy in uneven chunks."""
    gold_dir = pathlib.Path(args.golden)
    tokens = json.loads((gold_dir / "tokens.json").read_text())["tokens"]
    positions = json.loads((gold_dir / "positions.json").read_text())["positions"]
    gold, greedy = np.load(gold_dir / "logits.npy"), json.loads((gold_dir / "greedy.json").read_text())["tokens"]
    tail = len(tokens) - positions[0]
    assert positions == list(range(positions[0], len(tokens))), "needs a contiguous tail of stored positions"
    eos = Config(Checkpoint(args.checkpoint).config).eos
    stock = StockMLX(args.checkpoint, "float32")
    for chunk in (2048, 1000, 256):
        cache, done = stock.cache(), 0
        while len(tokens) - done > tail:
            n = min(chunk, len(tokens) - tail - done)
            stock.run(tokens[done:done + n], cache, [], keep_hidden=False)
            done += n
        _, lg = stock.run(tokens[done:], cache, range(tail), keep_hidden=False)
        same = stock.greedy(lg[-1], cache, eos) == greedy
        print("stock MLX f32, chunks of %d: logits vs golden %.2e, top-1 %d/%d, greedy %s"
              % (chunk, np.abs(lg - gold).max(), (lg.argmax(1) == gold.argmax(1)).sum(), tail, "equal" if same else "DIFFERENT"))
    del stock
    model = NumpyLFM3(Checkpoint(args.checkpoint), np.float64).load()
    st, done, sizes = model.new_state(), 0, (300, 1000, 2048)
    for n in sizes:
        model.sequence(tokens[done:done + n], st, (), keep_hidden=False)
        done += n
    model.sequence(tokens[done:len(tokens) - tail], st, (), keep_hidden=False)
    _, lg = model.sequence(tokens[len(tokens) - tail:], st, range(tail), keep_hidden=False)
    print("NumPy f64, chunks %s then the rest: logits vs golden %.2e, top-1 %d/%d"
          % ("/".join(map(str, sizes)), np.abs(lg - gold).max(), (lg.argmax(1) == gold.argmax(1)).sum(), tail))


def cmd_check(args):
    """Score a candidate (an engine's logits.npy, optional hidden.npy and greedy.json) against one golden case.

    NumPy only. Fails on top-1 or greedy disagreement or differences past the given bounds; the
    defaults are ten times the float32 noise floor recorded in MANIFEST.md, not a fitted tolerance.
    """
    gold, cand = pathlib.Path(args.golden), pathlib.Path(args.candidate)
    result, failures = {}, []
    g, c = np.load(gold / "logits.npy"), np.load(cand / "logits.npy").astype(np.float32)
    if g.shape != c.shape:
        raise SystemExit("logits shape %s, golden %s" % (c.shape, g.shape))
    result["logits"] = logit_report(c, g)
    if result["logits"]["max_abs_diff"] > args.logits_atol:
        failures.append("logits")
    if result["logits"]["top1_agree"] != g.shape[0]:
        failures.append("top1")
    if (gold / "hidden.npy").exists() and (cand / "hidden.npy").exists():
        gh, ch = np.load(gold / "hidden.npy", mmap_mode="r"), np.load(cand / "hidden.npy", mmap_mode="r")
        if gh.shape != ch.shape:
            raise SystemExit("hidden shape %s, golden %s" % (ch.shape, gh.shape))
        per = [float(np.max(np.abs(np.asarray(ch[i], np.float32) - gh[i]))) for i in range(gh.shape[0])]
        result["hidden_max_abs_diff_per_index"] = per
        bad = [i for i, d in enumerate(per) if d > args.hidden_atol]
        if bad:
            failures.append("hidden from index %d (layer %d output)" % (bad[0], bad[0] - 1))
    if (cand / "greedy.json").exists():
        a = json.loads((gold / "greedy.json").read_text())["tokens"]
        b = json.loads((cand / "greedy.json").read_text())["tokens"]
        common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        result["greedy"] = {"golden": len(a), "candidate": len(b), "common_prefix": common}
        if a != b:
            failures.append("greedy")
    result["bounds"] = {"logits_atol": args.logits_atol, "hidden_atol": args.hidden_atol}
    print(json.dumps(result, indent=1))
    if failures:
        raise SystemExit("FAIL %s: %s" % (gold.name, ", ".join(failures)))
    print("PASS %s" % gold.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inspect", help="header table; our dequantization vs mx.dequantize")
    p.add_argument("checkpoint")
    p.add_argument("--table", action="store_true")
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("compare", help="stock MLX vs NumPy on a prompt or token ids")
    p.add_argument("checkpoint")
    p.add_argument("--prompt", default=SHORT_PROMPT)
    p.add_argument("--tokens", help="comma-separated ids (overrides --prompt)")
    p.add_argument("--numpy-dtype", choices=("float64", "float32"), default="float64")
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser("window-rule", help="settle how many positions a sliding query sees")
    p.add_argument("checkpoint")
    p.set_defaults(func=cmd_window_rule)
    p = sub.add_parser("goldens", help="write golden cases")
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--cases", default="short,window,long")
    p.add_argument("--numpy-dtype", choices=("float64", "float32"), default="float64")
    p.set_defaults(func=cmd_goldens)
    p = sub.add_parser("chunked", help="chunked prefill vs the stored long golden")
    p.add_argument("checkpoint")
    p.add_argument("--golden", default=str(pathlib.Path(os.environ.get("NANO_WORK", pathlib.Path(__file__).resolve().parent.parent / "target/nano-work")) / "golden/lfm3/long"))
    p.set_defaults(func=cmd_chunked)
    p = sub.add_parser("check", help="score an engine's outputs against one golden case (NumPy only)")
    p.add_argument("golden", help="golden case directory, e.g. golden/lfm3/short")
    p.add_argument("candidate", help="directory with logits.npy and optionally hidden.npy, greedy.json")
    p.add_argument("--logits-atol", type=float, default=1e-3)
    p.add_argument("--hidden-atol", type=float, default=1e-3)
    p.set_defaults(func=cmd_check)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
