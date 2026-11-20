#!/usr/bin/env python3
"""Measure the session prefix cache and rewind on a real Flint checkpoint.

Renders a coding-agent system prompt and tool list (about 2,700 tokens) plus two
different user turns with the checkpoint's own tokenizer and chat template. Each
phase is one short run under the inference lock; native processes run one after
another, never concurrently:

  cold      --prefix-cache: reset S+U1 with checkpoint_at=[len(S)] (writes the
            disk snapshot), then truncate to len(S) and send U2 (rewind)
  restore   two new processes, each reset S+U2, restoring S from disk; then a
            third that first prefills U2 alone (warmup), then reset S+U2
  fresh     no new flag: S+U1 in one new process and S+U2 in another; with
            --baseline, S+U1 again with the unmodified base binary
  compare   no model: the prefill and final all-vocabulary logits (return_logits)
            and greedy tokens of cold/rewind/restored rows must be bit-identical
            to the fresh row with the same token stream; then the snapshot
            directory is deleted

  python3 tools/guarded_run.py --lock LOCK --timeout 600 -- \\
    /path/to/tokenizer-python -I tools/prefix_cache_benchmark.py CHECKPOINT \\
    --output target/prefix-cache-benchmark --phase cold
"""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _key, _value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                     'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false'}.items():
    os.environ[_key] = _value

SYSTEM = """You are a coding agent working inside a user's repository through function calls.
Work in small, verifiable steps. Read the relevant files before editing them, and never
guess at file contents, APIs or command output. Prefer the smallest change that fixes
the problem, keep the existing style, and do not reformat unrelated code.

Rules:
1. Use the provided tools for every file read, search, edit and command. Do not claim
   to have run a command unless a tool result shows its output.
2. Before editing, state in one sentence what you will change and why.
3. After editing, run the narrowest test that exercises the change. If it fails, read
   the failure carefully and fix the cause; do not weaken or skip the test.
4. Never delete files, rewrite history, or touch paths outside the workspace unless the
   user explicitly asks. Ask before any destructive or irreversible action.
5. Keep secrets out of output: do not print environment variables, tokens or keys.
6. When a task is ambiguous, make the most reasonable assumption, say what it was, and
   continue. Stop and ask only when continuing could cause damage.
7. Report results plainly: what changed, how it was verified, and anything left undone.
   Do not describe a partial result as complete.

Editing conventions: edits replace an exact, unique span of the current file. Include
enough surrounding context to make the span unique. Preserve indentation exactly.
For new files, write the complete content. Use relative paths from the workspace root.

Command conventions: commands run in a POSIX shell in the workspace root with a time
limit. Long-running servers must not be started. Network access may be unavailable;
if a command needs the network and fails, report it instead of retrying in a loop.

Testing conventions: discover the project's test runner from its configuration files
(for example pyproject.toml, package.json, Cargo.toml, build.zig or a Makefile) before
running tests. Run the specific test file or test name first, then the wider suite if
the change could affect other modules.
"""


def tool(name, description, properties, required):
    return dict(type='function', function=dict(name=name, description=description, parameters=dict(
        type='object', properties=properties, required=required, additionalProperties=False)))


def text(description):
    return dict(type='string', description=description)


def integer(description):
    return dict(type='integer', description=description)


TOOLS = [
    tool('read_file', 'Read a text file from the workspace. Returns numbered lines. Use offset and limit for large files instead of reading everything.',
         dict(path=text('Workspace-relative path of the file to read.'), offset=integer('First line to return, starting at 1.'),
              limit=integer('Maximum number of lines to return.')), ['path']),
    tool('write_file', 'Create a new file or replace an existing file with the given complete content. Parent directories are created as needed.',
         dict(path=text('Workspace-relative path of the file to write.'), content=text('The complete new file content.')), ['path', 'content']),
    tool('edit_file', 'Replace one exact, unique occurrence of old_text with new_text in a file. Fails if old_text is missing or occurs more than once.',
         dict(path=text('Workspace-relative path of the file to edit.'), old_text=text('Exact text to replace, including whitespace.'),
              new_text=text('Replacement text.')), ['path', 'old_text', 'new_text']),
    tool('list_directory', 'List the entries of a directory, marking subdirectories with a trailing slash. Hidden entries are included.',
         dict(path=text('Workspace-relative directory path; use . for the root.'), depth=integer('How many levels to descend, 1 to 3.')), ['path']),
    tool('glob', 'Find files whose workspace-relative paths match a glob pattern such as src/**/*.py. Results are sorted by modification time.',
         dict(pattern=text('Glob pattern relative to the workspace root.'), limit=integer('Maximum number of paths to return.')), ['pattern']),
    tool('grep', 'Search file contents with a regular expression. Returns matching lines with paths and line numbers.',
         dict(pattern=text('Regular expression to search for.'), path=text('File or directory to search; defaults to the workspace root.'),
              include=text('Optional glob restricting which files are searched.'), context=integer('Lines of context around each match.'),
              case_sensitive=dict(type='boolean', description='Whether matching is case sensitive.')), ['pattern']),
    tool('run_command', 'Run a shell command in the workspace root and return its exit status, standard output and standard error.',
         dict(command=text('The shell command line to run.'), timeout_seconds=integer('Time limit in seconds, at most 600.'),
              description=text('One short sentence saying what the command does.')), ['command', 'description']),
    tool('run_tests', 'Run the project tests, or a selected subset, using the detected test runner. Returns a summary and the failure output.',
         dict(selector=text('Optional test file, test name or keyword expression.'), verbose=dict(type='boolean', description='Include passing test names.')), []),
    tool('git_status', 'Show the working tree status: modified, added, deleted and untracked files relative to HEAD.', {}, []),
    tool('git_diff', 'Show the unified diff of uncommitted changes, optionally for one path or against a given revision.',
         dict(path=text('Optional path to restrict the diff.'), revision=text('Optional revision to diff against; defaults to HEAD.'),
              staged=dict(type='boolean', description='Show staged changes only.')), []),
    tool('git_log', 'Show recent commits with hashes, authors, dates and subjects.',
         dict(path=text('Optional path to restrict the log.'), limit=integer('Number of commits to show.')), []),
    tool('move_path', 'Move or rename a file or directory inside the workspace. Fails if the destination exists.',
         dict(source=text('Existing workspace-relative path.'), destination=text('New workspace-relative path.')), ['source', 'destination']),
    tool('delete_path', 'Delete a file or an empty directory inside the workspace. Requires explicit user approval first.',
         dict(path=text('Workspace-relative path to delete.'), reason=text('Why the deletion is needed.')), ['path', 'reason']),
    tool('search_symbols', 'Find definitions of functions, classes, methods or constants by name across the workspace using the language index.',
         dict(name=text('Symbol name or prefix.'), kind=dict(type='string', enum=['function', 'class', 'method', 'constant', 'any'], description='Kind of symbol.'),
              limit=integer('Maximum number of results.')), ['name']),
    tool('find_references', 'List the places where a symbol defined at a given file and line is referenced.',
         dict(path=text('File containing the definition.'), line=integer('Line of the definition, starting at 1.'),
              symbol=text('Name of the symbol on that line.')), ['path', 'line', 'symbol']),
    tool('read_many_files', 'Read several small files at once. Each result is numbered like read_file. Prefer read_file for large files.',
         dict(paths=dict(type='array', items=text('Workspace-relative path.'), description='Paths to read, at most 8.')), ['paths']),
    tool('apply_patch', 'Apply a unified diff to the workspace. The patch must apply cleanly; fuzzy matching is not attempted.',
         dict(patch=text('A unified diff with paths relative to the workspace root.')), ['patch']),
    tool('ask_user', 'Ask the user a short clarifying question and wait for the answer. Use only when continuing could cause damage.',
         dict(question=text('The question, in one or two sentences.'), options=dict(type='array', items=text('A possible answer.'), description='Optional suggested answers.')), ['question']),
    tool('todo', 'Record or update the plan for the current task as a short checklist that is shown to the user.',
         dict(items=dict(type='array', description='The full checklist, in order.', items=dict(type='object', properties=dict(
             text=text('The step.'), done=dict(type='boolean', description='Whether the step is finished.')), required=['text', 'done']))), ['items']),
    tool('web_fetch', 'Fetch a public documentation page and return its text. Unavailable when the network is disabled.',
         dict(url=text('An https URL.'), max_characters=integer('Truncate the returned text to this many characters.')), ['url']),
]

USER_1 = ("The clamp() helper in src/numeric.py returns the lower bound when the value equals the upper bound. "
          "Find the bug, fix it, and run the tests for that module.")
USER_2 = ("Add a --verbose flag to the command-line entry point in src/cli.py that prints each processed file name, "
          "and add a test for it next to the existing CLI tests.")


class Session:
    def __init__(self, command, log):
        self.command, self.log = command, log
        started = time.monotonic()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
        self.started = started

    def request(self, payload):
        start = time.monotonic()
        self.process.stdin.write(json.dumps(payload) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('native session ended: ' + self.process.stderr.read())
        row = json.loads(line)
        row['wall_seconds'] = time.monotonic() - start
        # Includes process start and model load for a process's first request.
        row['since_spawn_seconds'] = time.monotonic() - self.started
        return row

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=120)
        stderr = self.process.stderr.read()
        with open(self.log, 'a') as f:
            f.write('$ ' + ' '.join(self.command) + '\n' + stderr)
        if code:
            raise RuntimeError('native session failed: ' + stderr)
        return stderr


def bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


SUMMARY = ('cache_before', 'cache_after', 'prefill_tokens', 'decode_forward_tokens', 'reused_prefix_tokens',
           'prefill_seconds', 'decode_seconds', 'wall_seconds', 'since_spawn_seconds', 'kv_valid_bytes',
           'truncated_from', 'rewind_checkpoint', 'replayed_tokens', 'replay_seconds', 'rewind_source',
           'restored_prefix_tokens', 'restore_seconds', 'snapshots_written', 'snapshot_bytes', 'snapshot_seconds',
           'snapshot_failures', 'checkpoints_taken', 'checkpoint_seconds', 'tokens')


def save(output, phase, rows, meta):
    """Rows without logits as JSON; logits exactly, as float32 arrays."""
    arrays = {}
    for name, row in rows.items():
        for key in ('prefill_logits', 'final_logits'):
            arrays[name + '.' + key] = np.asarray(row.pop(key), dtype=np.float32)
    np.savez(output / (phase + '-logits.npz'), **arrays)
    (output / (phase + '.json')).write_text(json.dumps(dict(meta, rows=rows), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--binary', type=Path, default=ROOT / 'zig-out/bin/flint')
    p.add_argument('--baseline', type=Path, help='binary built from the unmodified base commit (fresh phase)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('render', 'cold', 'restore', 'fresh', 'compare'), required=True)
    p.add_argument('--context', type=int, default=8192)
    p.add_argument('--max-tokens', type=int, default=32)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'target').resolve()):
        p.error('--output must be inside the ignored target directory')
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'prefix-cache'
    log = output / (args.phase + '-stderr.log')
    if args.phase == 'compare':
        return compare(output, cache)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True,
                                              trust_remote_code=False, fix_mistral_regex=False)
    tokenizer.chat_template = (args.checkpoint / 'chat_template.jinja').read_text()

    def render(messages, generation):
        options = dict(tools=TOOLS, add_generation_prompt=generation)
        text = tokenizer.apply_chat_template(messages, tokenize=False, **options)
        ids = list(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options))
        if list(tokenizer.encode(text, add_special_tokens=False)) != ids:
            raise SystemExit('template tokenization disagrees with its text')
        return ids

    system = [dict(role='system', content=SYSTEM)]
    stable = render(system, False)
    first = render(system + [dict(role='user', content=USER_1)], True)
    second = render(system + [dict(role='user', content=USER_2)], True)
    for ids in (first, second):
        if ids[:len(stable)] != stable:
            raise SystemExit('stable prefix is not an exact token prefix')
    s = len(stable)
    meta = dict(machine=dict(model=platform.machine(), platform=platform.platform()),
                checkpoint=args.checkpoint.name, context=args.context, max_tokens=args.max_tokens,
                flags='--context {} --mlx-attention 1 --staged-row-tile 2'.format(args.context),
                stable_prefix_tokens=s, first_prompt_tokens=len(first), second_prompt_tokens=len(second))
    if args.phase == 'render':
        print(json.dumps(meta))
        return
    common = ['--context', str(args.context), '--mlx-attention', '1', '--staged-row-tile', '2']

    def native(binary):
        return [str(binary.resolve()), 'session', str(args.checkpoint.resolve())] + common

    budget = args.max_tokens
    rows = {}
    started = time.monotonic()
    if args.phase == 'cold':
        if cache.exists():
            shutil.rmtree(cache)
        cold = Session(native(args.binary) + ['--prefix-cache', str(cache)], log)
        rows['cold'] = cold.request(dict(id=1, tokens=first, max_tokens=budget, reset=True, checkpoint_at=[s], return_logits=True))
        rows['rewind'] = cold.request(dict(id=2, truncate=s, tokens=second[s:], max_tokens=budget, return_logits=True))
        cold.close()
        snapshots = sorted(cache.glob('*.kvsnap'))
        meta['snapshot_files'] = [dict(name=f.name, bytes=f.stat().st_size) for f in snapshots]
    elif args.phase == 'restore':
        if len(list(cache.glob('*.kvsnap'))) != 1:
            raise SystemExit('run the cold phase first: expected exactly one snapshot')
        for name in ('restored', 'restored_repeat'):
            restored = Session(native(args.binary) + ['--prefix-cache', str(cache)], log)
            rows[name] = restored.request(dict(id=1, tokens=second, max_tokens=budget, reset=True, return_logits=True))
            restored.close()
        # Separates the restore from a new process's first-forward warmup: the
        # user turn alone (no snapshot is a prefix of it), then the restore.
        restored = Session(native(args.binary) + ['--prefix-cache', str(cache)], log)
        rows['warmup'] = restored.request(dict(id=1, tokens=second[s:], max_tokens=0, reset=True, return_logits=True))
        rows['restored_warm'] = restored.request(dict(id=2, tokens=second, max_tokens=budget, reset=True, return_logits=True))
        restored.close()
    else:
        runs = [('fresh_first', args.binary, first), ('fresh_second', args.binary, second)]
        if args.baseline:
            runs.append(('baseline_first', args.baseline, first))
        for name, binary, ids in runs:
            fresh = Session(native(binary), log)
            rows[name] = fresh.request(dict(id=1, tokens=ids, max_tokens=budget, reset=True, return_logits=True))
            fresh.close()
    meta['phase_seconds'] = time.monotonic() - started
    save(output, args.phase, rows, meta)
    print(json.dumps(dict(meta, rows={k: {f: v[f] for f in SUMMARY if f in v} for k, v in rows.items()}), indent=2))


def compare(output, cache):
    rows, logits, meta = {}, {}, {}
    for phase in ('cold', 'restore', 'fresh'):
        data = json.loads((output / (phase + '.json')).read_text())
        rows.update(data.pop('rows'))
        meta[phase] = data
        with np.load(output / (phase + '-logits.npz')) as arrays:
            logits.update({k: arrays[k] for k in arrays.files})
    s = meta['cold']['stable_prefix_tokens']
    for phase in ('restore', 'fresh'):
        for key in ('stable_prefix_tokens', 'first_prompt_tokens', 'second_prompt_tokens', 'context', 'max_tokens', 'flags'):
            if meta[phase][key] != meta['cold'][key]:
                raise SystemExit('phases were run with different prompts or settings: ' + key)
    pairs = [('cold', 'fresh_first'), ('rewind', 'fresh_second'), ('restored', 'fresh_second'),
             ('restored_repeat', 'fresh_second'), ('restored_warm', 'fresh_second')]
    if 'baseline_first' in rows:
        pairs.append(('fresh_first', 'baseline_first'))
    comparisons = {}
    for name, reference in pairs:
        comparisons[name] = dict(
            reference=reference,
            prefill_logits_bit_exact=bool(np.array_equal(bits(logits[name + '.prefill_logits']),
                                                         bits(logits[reference + '.prefill_logits']))),
            final_logits_bit_exact=bool(np.array_equal(bits(logits[name + '.final_logits']),
                                                       bits(logits[reference + '.final_logits']))),
            greedy_tokens_equal=rows[name]['tokens'] == rows[reference]['tokens'],
            vocabulary=int(logits[name + '.final_logits'].size))
    files = meta['cold'].get('snapshot_files', [])
    checks = dict(
        cold_snapshot=rows['cold'].get('snapshots_written') == 1 and len(files) == 1 and
        files[0]['bytes'] == rows['cold'].get('snapshot_bytes'),
        rewind_from_memory=rows['rewind'].get('rewind_source') == 'memory' and
        rows['rewind'].get('rewind_checkpoint') == s and rows['rewind'].get('replayed_tokens') == 0,
        restored_prefix=all(rows[n].get('restored_prefix_tokens') == s for n in ('restored', 'restored_repeat', 'restored_warm'))
        and rows['warmup'].get('restored_prefix_tokens') == 0)
    result = dict(meta['cold'], snapshot_file_bytes=files[0]['bytes'] if files else 0,
                  phase_seconds={p: meta[p]['phase_seconds'] for p in meta},
                  rows={k: {f: v[f] for f in SUMMARY if f in v} for k, v in rows.items()},
                  comparisons=comparisons, checks=checks,
                  passed=all(checks.values()) and all(all(v for k, v in c.items() if k not in ('reference', 'vocabulary'))
                                                      for c in comparisons.values()))
    result.pop('snapshot_files', None)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if cache.exists():
        shutil.rmtree(cache)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit('prefix cache benchmark failed its exactness gate')


if __name__ == '__main__':
    main()
