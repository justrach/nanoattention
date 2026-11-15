#!/usr/bin/env python3
"""Check a completed coding session against a fresh native full-history replay.

Run only after the persistent session has exited, before loading MLX. This is a
cache-correctness diagnostic, not a throughput benchmark or code-quality test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from coding_multiturn import ProtocolError, save_json, validate_transcript


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1] / 'zig-out/bin/flint')
    p.add_argument('--timeout', type=int, default=1200)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1] / 'target'
    if not args.output.resolve().is_relative_to(root.resolve()):
        p.error('--output must be inside the ignored target directory')
    transcript = json.loads((args.output / 'transcript.json').read_text())
    with np.load(args.output / 'boundaries.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    turns = validate_transcript(transcript, arrays, np)
    if transcript['native_execution'] != dict(mlx_attention=1, staged_row_tile=2):
        raise ProtocolError('unsupported native execution mode for replay')
    tokens = transcript['conversation_token_ids']
    with tempfile.TemporaryDirectory(prefix='flint-session-replay-') as temp:
        source = Path(temp) / 'tokens.json'
        source.write_text(json.dumps(tokens))
        completed = subprocess.run([
            str(args.binary.resolve()), 'eval', str(args.checkpoint.resolve()),
            '--tokens-file', str(source), '--context', str(transcript['context']),
            '--logits-from', str(len(tokens)-1), '--mlx-attention', '1', '--staged-row-tile', '2',
        ], check=True, text=True, stdout=subprocess.PIPE, timeout=args.timeout)
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise ProtocolError('fresh replay must return one final vocabulary row')
    fresh = np.asarray(json.loads(lines[0]), dtype=np.float32)
    persistent = arrays['turn_{}_final_logits'.format(len(turns))]
    if fresh.shape != persistent.shape or not np.isfinite(fresh).all():
        raise ProtocolError('fresh replay shape/finiteness mismatch')
    bit_exact = bool(np.array_equal(fresh.view(np.uint32), persistent.view(np.uint32)))
    report = dict(status='pass' if bit_exact else 'fail', bit_exact=bit_exact,
                  replayed_tokens=len(tokens), checked_position=len(tokens)-1,
                  vocabulary_size=len(fresh),
                  max_absolute_error=float(np.max(np.abs(fresh.astype(np.float64)-persistent))),
                  transcript_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                  scope='one final all-vocabulary row after a fresh full-history native replay; not every cached KV element')
    save_json(args.output / 'fresh-native-replay.json', report)
    print(json.dumps(report, indent=2))
    if not bit_exact:
        raise ProtocolError('persistent and fresh native logits differ')


if __name__ == '__main__':
    main()
