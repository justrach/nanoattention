#!/usr/bin/env python3
"""Check session rewind (`truncate`) and the on-disk prefix cache on tiny checkpoints.

Every engine-side comparison is exact: the all-vocabulary prefill and final
logits of a rewound or restored session must be bit-identical to a fresh
sequential native `eval` of the same token stream. Each row is also compared
with the independent NumPy model (the same tolerance as check_sessions.py) and
its greedy tokens with NumPy argmax. Covers truncate to a checkpoint, between
checkpoints, after the sliding ring wrapped, to 0, before any ring reuse, with
no checkpoints, restore from disk into a new process, restore then continue
then truncate, a snapshot written after a truncate and restored as the longest
match, truncate served from disk, a different context, and rejection of
snapshots made with a different checkpoint, a changed config, changed weight
bytes with identical sizes, different settings, or corrupt/truncated files, an
over-cap snapshot, and a cache directory removed under a live session.
With --baseline, plain session responses (no new flag or field) must equal the
base binary's byte for byte apart from timing. Snapshots live in a disposable
directory under the ignored target/.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import verify

CONTEXT = 32
ROOT = Path(__file__).resolve().parents[1]


class Worker:
    def __init__(self, binary, directory, *flags, context=CONTEXT, tile=2):
        self.process = subprocess.Popen(
            [str(binary), 'session', str(directory), '--context', str(context),
             '--staged-row-tile', str(tile), *map(str, flags)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError('session ended: ' + self.process.stderr.read())
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=90)
        stderr = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        assert code == 0, stderr
        return stderr


class Checker:
    def __init__(self, binary, directory, cfg, weights, context=CONTEXT, tile=2):
        self.binary, self.directory = binary, directory
        self.cfg, self.weights = cfg, weights
        self.context, self.tile = context, tile
        self.history = []
        self.fresh_runs = 0

    def fresh(self, tokens):
        source = self.directory / 'fresh.json'
        source.write_text(json.dumps(tokens))
        result = subprocess.run([str(self.binary), 'eval', str(self.directory), '--context', str(self.context),
                                 '--tokens-file', str(source), '--logits-from', str(len(tokens) - 1),
                                 '--staged-row-tile', str(self.tile)],
                                check=True, text=True, capture_output=True, timeout=90)
        self.fresh_runs += 1
        return np.asarray(json.loads(result.stdout), dtype=np.float32)

    def oracle(self, tokens):
        reference = verify.Reference(self.cfg, self.weights)
        for token in tokens:
            logits = reference.forward(token)
        return reference, logits

    def exact(self, actual, tokens, label):
        actual = np.asarray(actual, dtype=np.float32)
        expected = self.fresh(tokens)
        assert actual.shape == expected.shape, label
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), \
            '{}: logits differ from a fresh sequential run of {} tokens'.format(label, len(tokens))
        _, logits = self.oracle(tokens)
        np.testing.assert_allclose(actual, logits, rtol=2e-4, atol=2e-5, err_msg=label)

    def apply(self, request, row, label):
        if request.get('reset'):
            self.history = []
        if 'truncate' in request:
            assert row['truncated_from'] == len(self.history), label
            self.history = self.history[:request['truncate']]
        before = len(self.history)
        assert row['id'] == request['id'], label
        assert row['cache_before'] == row['reused_prefix_tokens'] == before, label
        assert row['prefill_tokens'] == len(request['tokens']), label
        prompt = self.history + request['tokens']
        self.exact(row['prefill_logits'], prompt, label + ' prefill')
        reference, logits = self.oracle(prompt)
        for token in row['tokens']:
            assert token == int(np.argmax(logits)), label + ': greedy token mismatch'
            logits = reference.forward(token)
        assert len(row['tokens']) <= request['max_tokens'], label
        self.history = prompt + row['tokens']
        assert row['cache_after'] == len(self.history), label
        self.exact(row['final_logits'], self.history, label + ' final')

    def run(self, worker, label, **request):
        request.setdefault('return_logits', True)
        row = worker.request(request)
        self.apply(request, row, label)
        return row


def expect(row, label, **fields):
    for key, value in fields.items():
        assert row.get(key) == value, '{}: {}={!r}, expected {!r}'.format(label, key, row.get(key), value)


def snapshot_bytes(cfg, tokens, context=CONTEXT):
    stride = cfg['num_key_value_heads'] * cfg['head_dim'] * 4 * 2
    state = sum(min(cfg['sliding_window'] if t == 'sliding_attention' else context, tokens) * stride
                for t in cfg['layer_types'])
    return 104 + 4 * tokens + 32 * len(cfg['layer_types']) + state


def rewind_cases(binary, directory, cfg, weights):
    check = Checker(binary, directory, cfg, weights)
    worker = Worker(binary, directory)
    # 7 prefix tokens wrap the 3-slot ring twice; checkpoint at 4 is pinned.
    row = check.run(worker, 'seed', id=1, tokens=[1, 2, 3, 4, 5, 6, 7], max_tokens=3, reset=True, checkpoint_at=[4])
    expect(row, 'seed', checkpoints_taken=1)
    assert 'rewind_source' not in row and 'restored_prefix_tokens' not in row
    row = check.run(worker, 'truncate to a checkpoint', id=2, truncate=4, tokens=[9, 2], max_tokens=2)
    expect(row, 'truncate to a checkpoint', rewind_source='memory', rewind_checkpoint=4, replayed_tokens=0)
    # Checkpoints now: 4 (pinned), 6 (end of prefill), 6 + decoded (end of decode).
    end = len(check.history)
    assert end > 7, 'decode must leave room between checkpoints'
    row = check.run(worker, 'truncate between checkpoints', id=3, truncate=end - 1, tokens=[3], max_tokens=2)
    expect(row, 'truncate between checkpoints', rewind_source='memory', rewind_checkpoint=6, replayed_tokens=end - 7)
    row = check.run(worker, 'no-op truncate', id=4, truncate=len(check.history), tokens=[8], max_tokens=1)
    expect(row, 'no-op truncate', rewind_source='none', replayed_tokens=0)
    row = check.run(worker, 'truncate to 0', id=5, truncate=0, tokens=[3, 1, 4, 1, 5], max_tokens=2)
    expect(row, 'truncate to 0', rewind_source='start', rewind_checkpoint=0, replayed_tokens=0)
    row = check.run(worker, 'append after truncate', id=6, tokens=[9, 2, 6], max_tokens=0)
    assert row['tokens'] == []
    worker.close()
    # Before any ring slot is reused, lowering the position alone is exact.
    worker = Worker(binary, directory)
    check.run(worker, 'short seed', id=1, tokens=[5, 6], max_tokens=1, reset=True)
    row = check.run(worker, 'ring intact', id=2, truncate=1, tokens=[7, 1], max_tokens=2)
    expect(row, 'ring intact', rewind_source='ring_intact', rewind_checkpoint=1, replayed_tokens=0)
    worker.close()
    # No in-memory checkpoints: the same rewind replays recorded history from 0.
    worker = Worker(binary, directory, '--rewind-checkpoints', 0)
    check.run(worker, 'seed without checkpoints', id=1, tokens=[1, 2, 3, 4, 5, 6, 7], max_tokens=3, reset=True, checkpoint_at=[4])
    row = check.run(worker, 'truncate without checkpoints', id=2, truncate=5, tokens=[9, 2], max_tokens=2)
    expect(row, 'truncate without checkpoints', rewind_source='start', rewind_checkpoint=0, replayed_tokens=5)
    worker.close()
    for bad in (dict(id=1, tokens=[1], truncate=1),
                dict(id=1, tokens=[1], reset=True, truncate=0),
                dict(id=1, tokens=[1, 2], reset=True, checkpoint_at=[3]),
                dict(id=1, tokens=[1, 2], reset=True, checkpoint_at=[2, 1]),
                dict(id=1, tokens=[1, 2], reset=True, checkpoint_at=[0])):
        result = subprocess.run([str(binary), 'session', str(directory), '--context', str(CONTEXT)],
                                input=json.dumps(bad) + '\n', text=True, capture_output=True, timeout=90)
        assert result.returncode != 0 and not result.stdout.strip(), bad
    return check.fresh_runs


def cache_cases(binary, directory, other, cfg, weights, cache):
    flags = ('--prefix-cache', cache)
    prefix = [1, 2, 3, 4, 5, 6]
    check = Checker(binary, directory, cfg, weights)
    worker = Worker(binary, directory, *flags)
    row = check.run(worker, 'cold', id=1, tokens=prefix + [7, 8], max_tokens=3, reset=True, checkpoint_at=[6])
    expect(row, 'cold', restored_prefix_tokens=0, snapshots_written=1, snapshot_failures=0,
           snapshot_bytes=snapshot_bytes(cfg, 6), checkpoints_taken=1)
    worker.close()
    files = sorted(cache.glob('*.kvsnap'))
    assert len(files) == 1 and files[0].stat().st_size == snapshot_bytes(cfg, 6)
    assert not [p for p in cache.iterdir() if p.suffix != '.kvsnap'], 'temporary files left behind'
    snapshot = files[0].read_bytes()

    # A new process restores the prefix and prefills only the remainder.
    worker = Worker(binary, directory, *flags)
    row = check.run(worker, 'restore', id=1, tokens=prefix + [9], max_tokens=3, reset=True, checkpoint_at=[6])
    expect(row, 'restore', restored_prefix_tokens=6, snapshots_written=0, checkpoints_taken=1)
    check.run(worker, 'continue after restore', id=2, tokens=[4, 4], max_tokens=2)
    row = check.run(worker, 'truncate to restore point', id=3, truncate=6, tokens=[2, 7], max_tokens=2)
    expect(row, 'truncate to restore point', rewind_source='memory', rewind_checkpoint=6, replayed_tokens=0)
    # A rewound session marks and writes a longer prefix of its new history.
    row = check.run(worker, 'checkpoint after truncate', id=4, truncate=7, tokens=[3, 5], max_tokens=1, checkpoint_at=[9])
    expect(row, 'checkpoint after truncate', snapshots_written=1, checkpoints_taken=1,
           snapshot_bytes=snapshot_bytes(cfg, 9))
    longer = check.history[:9]
    # A shorter request than any snapshot restores nothing and stays exact.
    row = check.run(worker, 'short reset', id=5, tokens=prefix, max_tokens=1, reset=True)
    expect(row, 'short reset', restored_prefix_tokens=0)
    worker.close()
    worker = Worker(binary, directory, *flags)
    row = check.run(worker, 'restore longest', id=1, tokens=longer + [1], max_tokens=2, reset=True)
    expect(row, 'restore longest', restored_prefix_tokens=9)
    worker.close()
    for path in cache.glob('*.kvsnap'):
        if path.read_bytes() != snapshot:
            path.unlink()
    assert sorted(cache.glob('*.kvsnap')) == files

    # Without memory checkpoints, a truncate above the snapshot is served from disk.
    worker = Worker(binary, directory, *flags, '--rewind-checkpoints', 0)
    check.run(worker, 'restore without checkpoints', id=1, tokens=prefix + [9, 3], max_tokens=3, reset=True)
    row = check.run(worker, 'truncate from disk', id=2, truncate=7, tokens=[5], max_tokens=2)
    expect(row, 'truncate from disk', rewind_source='disk', rewind_checkpoint=6, replayed_tokens=1)
    worker.close()

    # Smaller context: same ring capacity and stored layout, so the snapshot applies.
    wide = Checker(binary, directory, cfg, weights, context=16)
    worker = Worker(binary, directory, *flags, context=16)
    row = wide.run(worker, 'other context', id=1, tokens=prefix + [2], max_tokens=1, reset=True)
    expect(row, 'other context', restored_prefix_tokens=6)
    worker.close()

    # Rejections: different settings, a different checkpoint, a changed config.
    tiled = Checker(binary, directory, cfg, weights, tile=1)
    worker = Worker(binary, directory, *flags, tile=1)
    row = tiled.run(worker, 'different settings', id=1, tokens=prefix + [9], max_tokens=1, reset=True)
    expect(row, 'different settings', restored_prefix_tokens=0)
    worker.close()
    for path, label in ((other, 'different checkpoint'),):
        other_cfg, other_weights = verify.checkpoint(path, dtype='F32', sharded=False, biased=True)
        foreign = Checker(binary, path, other_cfg, other_weights)
        worker = Worker(binary, path, *flags)
        row = foreign.run(worker, label, id=1, tokens=prefix + [9], max_tokens=1, reset=True)
        expect(row, label, restored_prefix_tokens=0)
        worker.close()
    edited = directory.parent / (directory.name + '-config')
    shutil.copytree(directory, edited)
    (edited / 'config.json').write_text((directory / 'config.json').read_text() + '\n')
    foreign = Checker(binary, edited, cfg, weights)
    worker = Worker(binary, edited, *flags)
    row = foreign.run(worker, 'changed config', id=1, tokens=prefix + [9], max_tokens=1, reset=True)
    expect(row, 'changed config', restored_prefix_tokens=0)
    worker.close()
    shutil.rmtree(edited)
    # Same config, shard names and sizes; one weight byte differs (sampled identity).
    requantized = directory.parent / (directory.name + '-weights')
    shutil.copytree(directory, requantized)
    shard = sorted(requantized.glob('*.safetensors'))[0]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(bytes(data))
    worker = Worker(binary, requantized, *flags)
    row = worker.request(dict(id=1, tokens=prefix + [9], max_tokens=0, reset=True))
    expect(row, 'changed weights', restored_prefix_tokens=0)
    worker.close()
    shutil.rmtree(requantized)
    assert files[0].read_bytes() == snapshot, 'foreign sessions must not touch the snapshot'

    # Corrupt (same size) and truncated files are detected, ignored and removed.
    for label, damaged in (('corrupt', snapshot[:-1] + bytes([snapshot[-1] ^ 1])), ('truncated', snapshot[:-1])):
        files[0].write_bytes(damaged)
        worker = Worker(binary, directory, *flags)
        row = check.run(worker, label, id=1, tokens=prefix + [9], max_tokens=1, reset=True)
        expect(row, label, restored_prefix_tokens=0)
        worker.close()
        assert not files[0].exists(), label + ' snapshot must be removed'
    # A cap below one snapshot writes nothing.
    worker = Worker(binary, directory, '--prefix-cache', cache, '--prefix-cache-bytes', snapshot_bytes(cfg, 6) - 1)
    row = check.run(worker, 'over cap', id=1, tokens=prefix + [9], max_tokens=1, reset=True, checkpoint_at=[6])
    expect(row, 'over cap', snapshots_written=0, snapshot_failures=0)
    worker.close()
    assert not list(cache.iterdir())
    # A cache directory that disappears is reported, not fatal, and stays exact.
    worker = Worker(binary, directory, *flags, '--rewind-checkpoints', 0)
    check.run(worker, 'seed before removal', id=1, tokens=prefix + [9, 3], max_tokens=2, reset=True)
    shutil.rmtree(cache)
    row = check.run(worker, 'truncate without directory', id=2, truncate=7, tokens=[5], max_tokens=1)
    expect(row, 'truncate without directory', rewind_source='start', rewind_checkpoint=0, replayed_tokens=7)
    row = check.run(worker, 'reset without directory', id=3, tokens=prefix + [1], max_tokens=1, reset=True, checkpoint_at=[6])
    expect(row, 'reset without directory', restored_prefix_tokens=0, snapshots_written=0, snapshot_failures=1)
    stderr = worker.close()
    assert 'Prefix cache lookup failed' in stderr and 'not written' in stderr, stderr
    return check.fresh_runs + wide.fresh_runs + tiled.fresh_runs


def unchanged_cases(binary, baseline, directory):
    """Without the new flags or fields, every response line equals the baseline
    binary's byte for byte, timing fields aside."""
    requests = [dict(id=1, tokens=[1, 2, 3, 4, 5, 6, 7], max_tokens=3, reset=True, return_logits=True),
                dict(id=2, tokens=[9, 2], max_tokens=2, return_logits=True),
                dict(id=3, tokens=[4], max_tokens=0),
                dict(id=4, tokens=[3, 1, 4], max_tokens=4, reset=True, return_logits=True),
                dict(id=5, tokens=[5, 9, 2, 6, 5, 3], max_tokens=5)]
    stdin = ''.join(json.dumps(r) + '\n' for r in requests)
    outputs = []
    for executable in (binary, baseline):
        result = subprocess.run([str(executable), 'session', str(directory), '--context', str(CONTEXT),
                                 '--staged-row-tile', '2'], input=stdin, text=True, capture_output=True,
                                check=True, timeout=90)
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        for line in lines:
            for key in ('prefill_seconds', 'decode_seconds'):
                assert type(line.pop(key)) in (int, float)
        outputs.append([json.dumps(line) for line in lines])
    assert len(outputs[0]) == len(requests) and outputs[0] == outputs[1], 'session output changed without new fields'
    return len(requests)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, default=ROOT / 'zig-out/bin/flint')
    p.add_argument('--baseline', type=Path, help='optional binary built from the unmodified base commit')
    args = p.parse_args()
    binary = args.binary.resolve()
    (ROOT / 'target').mkdir(exist_ok=True)
    for label, case in (('F32', dict(dtype='F32', sharded=False, biased=False)),
                        ('INT4 (BF16 scales, 8-bit embeddings)',
                         dict(dtype='BF16', sharded=True, biased=True, bits=4,
                              embedding_bits=8, group_size=4, packed_dtype='U32'))):
        with tempfile.TemporaryDirectory(prefix='flint-prefix-cache-check-', dir=ROOT / 'target') as temp:
            directory, other, cache = Path(temp) / 'model', Path(temp) / 'other', Path(temp) / 'cache'
            directory.mkdir()
            other.mkdir()
            cfg, weights = verify.checkpoint(directory, **case)
            runs = rewind_cases(binary, directory, cfg, weights)
            runs += cache_cases(binary, directory, other, cfg, weights, cache)
            print('PASS: {} rewind and prefix cache; {} bit-exact fresh-native comparisons, '
                  'NumPy parity and greedy tokens'.format(label, runs))
            if args.baseline:
                count = unchanged_cases(binary, args.baseline.resolve(), directory)
                print('PASS: {} {} plain session responses identical to the baseline binary'.format(label, count))


if __name__ == '__main__':
    main()
