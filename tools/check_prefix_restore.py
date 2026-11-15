#!/usr/bin/env python3
"""Check real-prefix KV restoration and eval output slicing on tiny checkpoints."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('zig-out/bin/flint'))
    parser.add_argument('--benchmark-binary', type=Path, default=Path('zig-out/bin/flint-bench'))
    args = parser.parse_args()
    binary, bench = args.binary.resolve(), args.benchmark_binary.resolve()
    prompt = [1, 2, 3, 4, 5, 1, 2, 6, 7]
    decode = [1, 5, 8, 9, 3, 4]
    for case in (dict(dtype='F32', sharded=False, biased=False),
                 dict(dtype='BF16', sharded=True, biased=True, bits=4, group_size=4, packed_dtype='U32')):
        with tempfile.TemporaryDirectory(prefix='flint-prefix-check-') as temporary:
            directory = Path(temporary)
            verify.checkpoint(directory, **case)
            source = directory / 'benchmark.json'
            base = dict(checkpoint=str(directory), prompt=prompt, decode=decode, repeats=3,
                        emit_predictions=True, staged_row_tile=2, attention_chunk=0)
            results = []
            for reuse in (False, True):
                source.write_text(json.dumps(dict(base, reuse_prefill=reuse)))
                proc = subprocess.run([str(bench), str(source)], check=True, text=True,
                                      stdout=subprocess.PIPE, timeout=90)
                rows = [json.loads(line) for line in proc.stdout.splitlines()]
                assert len(rows) == 3
                for row in rows:
                    assert row['prefill_reused'] == reuse
                    assert row['prefill_samples'] == (1 if reuse else 3)
                if reuse:
                    assert len({r['prefill_seconds'] for r in rows}) == 1
                    assert len({r['prefill_gpu_seconds'] for r in rows}) == 1
                results.extend(rows)
            expected = results[0]['predictions']
            assert all(row['predictions'] == expected for row in results)
            source.write_text(json.dumps(dict(base, reuse_prefill=True,
                                              attention_candidates=[0, 128, 256, 512, 1024])))
            proc = subprocess.run([str(bench), str(source)], check=True, text=True,
                                  stdout=subprocess.PIPE, timeout=90)
            rows = [json.loads(line) for line in proc.stdout.splitlines()]
            assert len(rows) == 15
            assert [row['attention_chunk'] for row in rows] == [c for c in (0, 128, 256, 512, 1024) for _ in range(3)]
            assert all(row['predictions'] == expected for row in rows)
            tokens = prompt + decode
            common = [str(binary), 'eval', str(directory), '--tokens', ','.join(map(str, tokens)),
                      '--context', str(len(tokens)), '--staged-row-tile', '2']
            all_rows = subprocess.run(common, check=True, text=True, stdout=subprocess.PIPE, timeout=90)
            selected = subprocess.run(common + ['--logits-from', str(len(prompt) - 1)], check=True,
                                      text=True, stdout=subprocess.PIPE, timeout=90)
            assert all_rows.stdout.splitlines()[len(prompt) - 1:] == selected.stdout.splitlines()
            print('PASS: {} real-prefix restore, candidate isolation and logit slicing'.format(case['dtype']))


if __name__ == '__main__':
    main()
