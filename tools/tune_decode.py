#!/usr/bin/env python3
"""Parity-gated decode-only hill climb, run under tools/guarded_run.py.

One genuine prefix is snapshotted. Every candidate must match every baseline
vocabulary bit at every teacher-forced decode position before timing. Candidate
order rotates between rounds; restore, admission, and warmup are untimed.
This is not free-running generation or a prefill benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import random
import statistics
import subprocess
import tempfile

from benchmark import workload, positive_u32

DEFAULT_CANDIDATES = [dict(head_row_tile=tile, packed_router=packed)
                      for packed, tiles in ((False, (2, 1, 4, 8)), (True, (2, 4, 8)))
                      for tile in tiles]
DEFAULT_CANDIDATES += [dict(head_row_tile=2, parallel_down=True, down_row_tile=tile)
                       for tile in (2, 4, 8)]
DEFAULT_CANDIDATES += [dict(head_row_tile=2, down_row_tile=tile) for tile in (4, 8)]


def raw_workload(vocab_size, prompt_tokens, decode_tokens, seed):
    """Independent streams keep decode IDs fixed across prefix lengths."""
    if not 0 < vocab_size <= 0xffffffff or prompt_tokens < 1 or decode_tokens < 1:
        raise ValueError('raw workload requires a valid vocabulary and nonempty sequences')
    prompt_rng = random.Random(seed)
    decode_rng = random.Random(seed ^ 0x9E3779B9)
    return ([prompt_rng.randrange(vocab_size) for _ in range(prompt_tokens)],
            [decode_rng.randrange(vocab_size) for _ in range(decode_tokens)])


def profile_metrics(stderr):
    """Keep only the content-free native profiler object, never arbitrary logs."""
    for line in stderr.splitlines():
        if not line.startswith('{'):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if (isinstance(value, dict) and value.get('type') == 'flint_kernel_profile'
                and value.get('benchmark') is False and isinstance(value.get('kernels'), list)):
            return value
    return None


def summarize(rows, candidates, repeats, decode_tokens):
    admissions = {row['candidate']: row for row in rows if row.get('phase') == 'validation'}
    if set(admissions) != set(range(len(candidates))):
        raise ValueError('missing candidate admission results')
    baseline = [row for row in rows if row.get('phase') == 'timing' and row['candidate'] == 0]
    if len(baseline) != repeats:
        raise ValueError('baseline repeat count mismatch')
    expected_predictions = baseline[0]['predictions']
    if len(expected_predictions) != decode_tokens:
        raise ValueError('decode prediction count mismatch')
    results = []
    for index, config in enumerate(candidates):
        admission = admissions[index]
        selected = [row for row in rows if row.get('phase') == 'timing' and row['candidate'] == index]
        if admission['logits_bit_exact'] is not True:
            if selected:
                raise ValueError('rejected candidate was timed')
            results.append(dict(candidate=index, config=config, admitted=False))
            continue
        if len(selected) != repeats or sorted(row['repeat'] for row in selected) != list(range(repeats)):
            raise ValueError('candidate repeat count mismatch')
        if any(row['predictions'] != expected_predictions for row in selected):
            raise ValueError('candidate predictions changed')
        wall = [row['decode_seconds'] for row in selected]
        gpu = [row['decode_gpu_seconds'] for row in selected]
        if any(not 0 < seconds < float('inf') for seconds in wall + gpu):
            raise ValueError('invalid decode timing')
        results.append(dict(candidate=index, config=config, admitted=True,
                            bit_exact_positions=admission['positions'], vocabulary=admission['vocabulary'],
                            decode_seconds=wall, decode_gpu_seconds=gpu,
                            median_decode_seconds=statistics.median(wall),
                            median_gpu_seconds=statistics.median(gpu),
                            decode_tokens_per_second=decode_tokens/statistics.median(wall)))
        if all('decode_cpu_seconds' in row for row in selected):
            cpu = [row['decode_cpu_seconds'] for row in selected]
            results[-1].update(decode_cpu_seconds=cpu, median_cpu_seconds=statistics.median(cpu))
    first = results[0]
    for result in results:
        if result['admitted']:
            result['wall_speedup_over_baseline'] = first['median_decode_seconds']/result['median_decode_seconds']
            result['gpu_speedup_over_baseline'] = first['median_gpu_seconds']/result['median_gpu_seconds']
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('zig-out/bin/flint-bench'))
    parser.add_argument('--prompt-tokens', type=positive_u32, default=16384)
    parser.add_argument('--decode-tokens', type=positive_u32, default=256)
    parser.add_argument('--repeats', type=positive_u32, default=5)
    parser.add_argument('--workload', choices=('code', 'random-raw'), default='code')
    parser.add_argument('--seed', type=int, default=20260328)
    parser.add_argument('--profile-after-sweep', action='store_true',
                        help='profile up to 8 ordinary tokens after all timing, never timed as throughput')
    parser.add_argument('--profile-candidate', type=int, default=1)
    parser.add_argument('--candidates', type=json.loads, default=DEFAULT_CANDIDATES,
                        help='JSON array; first entry must be head tile 2, packed_router false')
    parser.add_argument('--prepare-replay', action='store_true', help='compile ICB-capable pipelines for both direct/replay candidates')
    parser.add_argument('--kv-bf16', action='store_true', help='same BF16-global-KV reference for every candidate, not FP32 accuracy admission')
    parser.add_argument('--bf16-blocks', type=int, choices=(0, 128, 512, 1024), default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.bf16_blocks and not args.kv_bf16:
        parser.error('--bf16-blocks requires --kv-bf16')
    if args.profile_after_sweep and args.prepare_replay:
        parser.error('--profile-after-sweep conflicts with --prepare-replay')
    if args.profile_after_sweep and not 0 <= args.profile_candidate < len(args.candidates):
        parser.error('--profile-candidate must select an existing candidate')
    if args.workload == 'random-raw':
        vocab_size = json.loads((args.checkpoint/'config.json').read_text())['vocab_size']
        prompt, decode = raw_workload(vocab_size, args.prompt_tokens, args.decode_tokens, args.seed)
        recipe = 'uniform_raw_ids_independent_python_random_streams_v1'
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True,
                                                  trust_remote_code=False, fix_mistral_regex=False)
        prompt, decode = workload(tokenizer, args.decode_tokens, args.prompt_tokens)
        recipe = 'repeat_local_tokenizer_code_ids_then_truncate_no_special_tokens_v1'
    config = dict(checkpoint=str(args.checkpoint.resolve()), prompt=prompt, decode=decode,
                  repeats=args.repeats, staged_row_tile=2, reuse_prefill=True,
                  mlx_attention=True, kv_bf16=args.kv_bf16, bf16_blocks=args.bf16_blocks,
                  prepare_replay=args.prepare_replay,
                  profile_after_sweep=args.profile_after_sweep,
                  profile_candidate=args.profile_candidate,
                  decode_candidates=args.candidates)
    with tempfile.TemporaryDirectory(prefix='flint-decode-sweep-') as temp:
        path = Path(temp)/'input.json'
        path.write_text(json.dumps(config))
        # time -l's max RSS is a process measurement, not Metal private-memory
        # accounting, and includes this benchmark's KV snapshot/logit reference.
        process = subprocess.run(['/usr/bin/time', '-l', str(args.binary.resolve()), str(path)],
                                 text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in process.stdout.splitlines()]
    results = summarize(rows, args.candidates, args.repeats, args.decode_tokens)
    peak = re.search(r'(\d+)\s+maximum resident set size', process.stderr)
    sample = next(row for row in rows if row.get('phase') == 'timing')
    memory = {key: sample[key] for key in ('checkpoint_mapped_bytes', 'model_buffer_bytes',
                                          'kv_allocated_bytes', 'reference_logit_bytes')}
    memory['benchmark_peak_rss_bytes'] = int(peak.group(1)) if peak else None
    memory['scope'] = ('Model buffers include KV; do not add KV twice. Mapped checkpoint bytes '
                       'are file size, not resident RAM. Peak process RSS includes benchmark-only '
                       'KV snapshots and logit references, and is not an MLX memory comparison.')
    report = dict(protocol='real prefix; all-logit bitwise admission within the selected KV storage mode; rotated candidates; warm decode only',
                  workload=args.workload, prompt_recipe=recipe,
                  seed=args.seed if args.workload == 'random-raw' else None,
                  speculation=False, decode_mode='ordinary_single_token_teacher_forced_full_model',
                  profiler=profile_metrics(process.stderr) if args.profile_after_sweep else None,
                  kv_storage=sample.get('kv_storage', 'fp32'), bf16_blocks=args.bf16_blocks,
                  pipelines_icb_capable=args.prepare_replay,
                  prompt_tokens=len(prompt), decode_tokens=len(decode), repeats=args.repeats,
                  input_sha256=hashlib.sha256(json.dumps(dict(prompt=prompt, decode=decode), sort_keys=True).encode()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  memory=memory, candidates=results, raw_rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(memory=memory, candidates=results), indent=2))


if __name__ == '__main__':
    main()
