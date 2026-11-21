#!/usr/bin/env python3
"""Compare two Inkling runs saved in nano eval's summary/raw-output format.

Each summary is the single JSON object printed by `nano eval`; logits and,
optionally, hidden traces are the corresponding little-endian float32 files.
This script compares outputs only: generate the reference with the pinned
Inkling-capable implementation and use identical token IDs and positions.

Example:
  python3 tools/nano_inkling_verify.py \
    --reference-summary ref.json --candidate-summary native.json \
    --reference-logits ref.f32 --candidate-logits native.f32
"""
import argparse
import json
import math
import pathlib
import struct
import sys

CHUNK_FLOATS = 16384


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return value


def load_summary(path):
    try:
        value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("%s: cannot read nano eval JSON summary: %s" % (path, exc))
    if not isinstance(value, dict):
        raise ValueError("%s: summary must be a JSON object" % path)
    for key in ("tokens", "positions", "vocab", "top1", "greedy"):
        if key not in value:
            raise ValueError("%s: summary is missing %r" % (path, key))

    tokens = _integer(value["tokens"], "tokens", 1)
    vocab = _integer(value["vocab"], "vocab", 1)
    positions = value["positions"]
    if not isinstance(positions, list):
        raise ValueError("positions must be a JSON array")
    previous = -1
    for position in positions:
        position = _integer(position, "position")
        if position >= tokens or position <= previous:
            raise ValueError("positions must be increasing and less than tokens")
        previous = position

    for key in ("top1", "greedy"):
        ids = value[key]
        if not isinstance(ids, list):
            raise ValueError("%s must be a JSON array" % key)
        if key == "top1" and len(ids) != len(positions):
            raise ValueError("top1 length must match positions")
        for token in ids:
            if _integer(token, key + " token") >= vocab:
                raise ValueError("%s token is outside the vocabulary" % key)

    if ("traces" in value) != ("trace_width" in value):
        raise ValueError("traces and trace_width must be provided together")
    if "traces" in value:
        _integer(value["traces"], "traces")
        _integer(value["trace_width"], "trace_width")
    return value


def compare_metadata(reference, candidate):
    keys = ("tokens", "positions", "vocab", "top1", "greedy")
    for key in keys:
        if reference[key] != candidate[key]:
            raise ValueError("%s mismatch: reference=%r candidate=%r" %
                             (key, reference[key], candidate[key]))
    for key in ("traces", "trace_width"):
        if key in reference or key in candidate:
            if reference.get(key) != candidate.get(key):
                raise ValueError("%s mismatch: reference=%r candidate=%r" %
                                 (key, reference.get(key), candidate.get(key)))


def compare_f32(reference_path, candidate_path, count, atol, rtol, label):
    expected_bytes = count * 4
    ref_size = pathlib.Path(reference_path).stat().st_size
    got_size = pathlib.Path(candidate_path).stat().st_size
    if ref_size != expected_bytes or got_size != expected_bytes:
        raise ValueError("%s byte length mismatch: expected %d, reference %d, candidate %d" %
                         (label, expected_bytes, ref_size, got_size))
    max_abs = 0.0
    max_rel = 0.0
    first_bad = None
    index = 0
    with open(reference_path, "rb") as ref, open(candidate_path, "rb") as got:
        while index < count:
            n = min(CHUNK_FLOATS, count - index)
            ref_data = ref.read(n * 4)
            got_data = got.read(n * 4)
            if len(ref_data) != n * 4 or len(got_data) != n * 4:
                raise ValueError("%s ended before its declared float count" % label)
            for offset, (a,) in enumerate(struct.iter_unpack("<f", ref_data)):
                b = struct.unpack_from("<f", got_data, offset * 4)[0]
                if not math.isfinite(a) or not math.isfinite(b):
                    if first_bad is None:
                        first_bad = index + offset
                    continue
                error = abs(a - b)
                max_abs = max(max_abs, error)
                max_rel = max(max_rel, error / max(abs(a), 1e-30))
                if error > atol + rtol * abs(a) and first_bad is None:
                    first_bad = index + offset
            index += n
    if first_bad is not None:
        raise ValueError("%s differs at flat index %d (max abs %.7g, max rel %.7g; atol %.7g, rtol %.7g)" %
                         (label, first_bad, max_abs, max_rel, atol, rtol))
    print("%s: %d float32 values agree (max abs %.7g, max rel %.7g)" %
          (label, count, max_abs, max_rel))


def reference_main(argv):
    parser = argparse.ArgumentParser(description="Run the pinned Inkling MLX reference (token IDs only)")
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--tokens-file", required=True, type=pathlib.Path)
    parser.add_argument("--logits-out", required=True, type=pathlib.Path)
    parser.add_argument("--hidden-out", type=pathlib.Path)
    parser.add_argument("--feed", type=int, default=1)
    parser.add_argument("--greedy", type=int, default=16)
    args = parser.parse_args(argv)
    if args.feed < 1 or args.greedy < 0:
        parser.error("feed must be positive and greedy nonnegative")

    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.inkling2.layers import DecoderLayer
    from mlx_lm.utils import load_model

    prompt = json.loads(args.tokens_file.read_text(encoding="utf-8"))
    if isinstance(prompt, dict):
        prompt = prompt["tokens"]
    if not isinstance(prompt, list) or not prompt:
        parser.error("token file must contain a nonempty list")
    raw_config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    if raw_config["model_type"] != "inkling2_loop":
        parser.error("the reference requires the original Inkling 2 Mini checkpoint")
    dims = raw_config["text_config"]
    vocab = dims["unpadded_vocab_size"]
    if any(_integer(t, "token") >= vocab for t in prompt):
        parser.error("token ID is outside the usable vocabulary")
    eos = raw_config["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)

    # Do not remap or requantize experts. MLX's pinned loader binds the original
    # two formats; vision parameters are not needed for token-ID-only inference.
    model, _ = load_model(args.checkpoint, model_config={"vision_config": None})
    cache = model.make_cache()
    text = model.model.language_model
    hidden = None
    logits_chunks = []
    current = None
    original = DecoderLayer.__call__

    def capture(layer, *a, **kw):
        value = original(layer, *a, **kw)
        if current is not None:
            current.append(value[0])
        return value

    if args.hidden_out:
        DecoderLayer.__call__ = capture
    try:
        for i in range(0, len(prompt), args.feed):
            ids = mx.array([prompt[i:i + args.feed]])
            embed = text.embed(ids)
            current = [embed] if args.hidden_out else None
            result = model(ids, cache=cache, input_embeddings=embed)
            logits_chunks.append(np.asarray(result[0].astype(mx.float32)))
            if current is not None:
                if len(current) != len(cache) + 1:
                    raise ValueError("reference trace count differs from the unrolled cache count")
                if hidden is None:
                    hidden = [[] for _ in current]
                for layer, value in zip(hidden, current):
                    layer.append(np.asarray(value[0].astype(mx.float32)))
        current = None
        logits = np.concatenate(logits_chunks, axis=0)
        row = logits[-1]
        greedy = []
        for i in range(args.greedy):
            token = int(row.argmax())
            greedy.append(token)
            if token in eos or i + 1 == args.greedy:
                break
            row = np.asarray(model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32))
    finally:
        DecoderLayer.__call__ = original
    logits.astype("<f4", copy=False).tofile(args.logits_out)
    if hidden is not None:
        np.stack([np.concatenate(layer, axis=0) for layer in hidden]).astype("<f4", copy=False).tofile(args.hidden_out)
    print(json.dumps(dict(tokens=len(prompt), positions=list(range(len(prompt))), vocab=vocab,
                          top1=logits.argmax(axis=1).tolist(), greedy=greedy,
                          traces=len(cache) + 1, trace_width=dims["hidden_size"])))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-summary", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-summary", required=True, type=pathlib.Path)
    parser.add_argument("--reference-logits", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-logits", required=True, type=pathlib.Path)
    parser.add_argument("--reference-hidden", type=pathlib.Path)
    parser.add_argument("--candidate-hidden", type=pathlib.Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be finite and nonnegative")
    try:
        reference = load_summary(args.reference_summary)
        candidate = load_summary(args.candidate_summary)
        compare_metadata(reference, candidate)
        if reference["tokens"] < 0 or reference["vocab"] <= 0:
            raise ValueError("invalid token or vocabulary count in summary")
        positions = reference["positions"]
        if not isinstance(positions, list):
            raise ValueError("positions must be a JSON array")
        compare_f32(args.reference_logits, args.candidate_logits,
                    len(positions) * reference["vocab"], args.atol, args.rtol, "logits")
        if bool(args.reference_hidden) != bool(args.candidate_hidden):
            raise ValueError("provide both hidden files or neither")
        if args.reference_hidden:
            if "traces" not in reference or "trace_width" not in reference:
                raise ValueError("summary lacks trace dimensions for hidden comparison")
            compare_f32(args.reference_hidden, args.candidate_hidden,
                        reference["traces"] * reference["tokens"] * reference["trace_width"],
                        args.atol, args.rtol, "hidden traces")
        print("greedy token IDs agree exactly: %r" % reference["greedy"])
    except (OSError, ValueError) as exc:
        print("FAIL: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(reference_main(sys.argv[2:]) if sys.argv[1:2] == ["reference"] else main())
