#!/usr/bin/env python3
"""Real headless persistent-session checks on disposable tiny checkpoints.

Checks incremental/native-fresh bit identity, independent NumPy logits, final
sample/EOS cache commitment, explicit reset, physical sliding wrap and KV bytes.
No model downloads, server, generated-code execution or desktop interaction.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import verify


def check(binary, directory, case):
    cfg, weights = verify.checkpoint(directory, **case)
    requests = [
        dict(id=1, tokens=[1, 2, 3, 4], max_tokens=3, return_logits=True),
        dict(id=2, tokens=[5, 6], max_tokens=4, return_logits=True),
        dict(id=3, tokens=[2], max_tokens=0, return_logits=True),
        dict(id=4, tokens=[4, 8, 3], max_tokens=3, reset=True, return_logits=True),
    ]
    # No final newline: EOF must still process the final complete JSON record.
    payload = '\n'.join(json.dumps(r) for r in requests)
    result = subprocess.run([str(binary), 'session', str(directory), '--context', '32',
                             '--staged-row-tile', '2'], input=payload, text=True,
                            capture_output=True, check=True, timeout=90)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == len(requests)
    oracle = verify.Reference(cfg, weights)
    history = []
    element_bytes = cfg['num_key_value_heads'] * cfg['head_dim'] * 4 * 2
    capacities = [cfg['sliding_window'] if t == 'sliding_attention' else 32
                  for t in cfg['layer_types']]
    for request, row in zip(requests, rows):
        if request.get('reset'):
            history = []
            oracle = verify.Reference(cfg, weights)
        before = len(history)
        assert row['id'] == request['id']
        assert row['cache_before'] == row['reused_prefix_tokens'] == before
        assert row['prefill_tokens'] == len(request['tokens'])
        for token in request['tokens']:
            logits = oracle.forward(token)
        np.testing.assert_allclose(row['prefill_logits'], logits, rtol=2e-4, atol=2e-5)
        generated = row['tokens']
        assert len(generated) <= request['max_tokens']
        for token in generated:
            assert token == int(np.argmax(logits)), 'greedy token mismatch'
            logits = oracle.forward(token)  # Final token, including EOS, is cached.
        history += request['tokens'] + generated
        assert row['decode_forward_tokens'] == len(generated)
        assert row['cache_after'] == len(history) == oracle.position
        assert row['eos'] == bool(generated and generated[-1] in cfg['eos_token_id'])
        assert row['kv_allocated_bytes'] == sum(capacities) * element_bytes
        assert row['kv_valid_bytes'] == sum(min(len(history), c) for c in capacities) * element_bytes
        np.testing.assert_allclose(row['final_logits'], logits, rtol=2e-4, atol=2e-5)
        # Worker has exited before a fresh native model runs. Compare the exact
        # same token history, same allocation, and one final all-vocabulary row.
        source = directory / 'session-replay.json'
        source.write_text(json.dumps(history))
        fresh = subprocess.run([str(binary), 'eval', str(directory), '--context', '32',
                                '--tokens-file', str(source), '--logits-from', str(len(history)-1),
                                '--staged-row-tile', '2'], check=True, text=True,
                               capture_output=True, timeout=90)
        expected = np.asarray(json.loads(fresh.stdout), dtype=np.float32)
        actual = np.asarray(row['final_logits'], dtype=np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    # Reserve the final generated token too; max=29 with 4 prompt tokens exceeds
    # context 32, even though the old standalone max-1 reservation would fit.
    bad = subprocess.run([str(binary), 'session', str(directory), '--context', '32'],
                         input=json.dumps(dict(tokens=[1, 2, 3, 4], max_tokens=29))+'\n',
                         text=True, capture_output=True, timeout=90)
    assert bad.returncode != 0 and not bad.stdout.strip(), 'overflow must fail before output'
    print('PASS: {} persistent turns, NumPy/fresh parity, reset, final-token commitment, KV bytes and overflow'.format(case['dtype']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=Path('zig-out/bin/flint'))
    args = p.parse_args()
    binary = args.binary.resolve()
    for case in (dict(dtype='F32', sharded=False, biased=False),
                 dict(dtype='BF16', sharded=True, biased=True, bits=4,
                      embedding_bits=8, group_size=4, packed_dtype='U32')):
        with tempfile.TemporaryDirectory(prefix='flint-session-check-') as temp:
            check(binary, Path(temp), case)


if __name__ == '__main__':
    main()
