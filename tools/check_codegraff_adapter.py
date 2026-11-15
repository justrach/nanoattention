#!/usr/bin/env python3
"""Run installed Codegraff against native Flint in a disposable coding fixture.

Run under guarded_run.py. No global provider settings or credentials change.
macOS sandbox-exec confines Graff and its tests to fixture writes, denies user
file reads outside the fixture/binary, and permits network only to our adapter.
Full context allocation is not a full-window prefill or quality benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from urllib.request import Request, urlopen


FIXTURE = '''def clamp(value, lower, upper):
    if lower > upper:
        raise ValueError("invalid interval")
    return min(value, upper)
'''
TESTS = '''import unittest
from clamp import clamp

class ClampTests(unittest.TestCase):
    def test_below(self): self.assertEqual(clamp(-4, 0, 10), 0)
    def test_inside(self): self.assertEqual(clamp(4, 0, 10), 4)
    def test_above(self): self.assertEqual(clamp(14, 0, 10), 10)
    def test_negative(self): self.assertEqual(clamp(-9, -5, -1), -5)
    def test_equal(self): self.assertEqual(clamp(0, 3, 3), 3)
    def test_invalid(self):
        with self.assertRaises(ValueError): clamp(0, 5, 1)
'''


def receive_line(process, timeout):
    result = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: result.put(process.stdout.readline()), daemon=True).start()
    line = result.get(timeout=timeout)
    if not line:
        raise RuntimeError('adapter exited before readiness')
    return json.loads(line)


def fetch(url):
    request = Request(url, headers={'Authorization': 'Bearer local'})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def stop_owned(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def sandbox_profile(output, graff, port):
    quote = lambda value: json.dumps(str(value))
    # Root startup and tool children share these OS restrictions. System
    # libraries remain readable; unrelated /Users data and Keychain IPC do not.
    return '''(version 1)
(allow default)
(deny network*)
(allow network-outbound (remote ip "localhost:{port}"))
(deny mach-lookup (global-name "com.apple.securityd"))
(deny file-read* (require-all (subpath "/Users")
  (require-not (subpath {output}))
  (require-not (literal {graff}))))
(deny file-write* (require-all (subpath "/")
  (require-not (subpath {output}))
  (require-not (literal "/dev/null"))))
'''.format(port=port, output=quote(output), graff=quote(graff))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--graff', type=Path, required=True)
    p.add_argument('--binary', type=Path, default=Path('zig-out/bin/flint'))
    p.add_argument('--context', type=int, default=32768)
    p.add_argument('--max-tokens', type=int, default=4096)
    p.add_argument('--timeout', type=float, default=1200)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output, graff = args.output.resolve(), args.graff.resolve()
    if not output.is_relative_to((root/'target').resolve()):
        p.error('--output must be inside ignored target/')
    if not graff.is_file() or not args.binary.is_file() or not shutil.which('sandbox-exec'):
        p.error('installed Graff, native flint, and sandbox-exec are required')
    output.mkdir(parents=True, exist_ok=False)
    workspace, home = output/'workspace', output/'home'
    for path in (workspace, home, output/'tmp', workspace/'.graff'):
        path.mkdir(parents=True, exist_ok=True)
    (workspace/'clamp.py').write_text(FIXTURE)
    (workspace/'test_clamp.py').write_text(TESTS)
    tests_hash = hashlib.sha256(TESTS.encode()).hexdigest()
    subprocess.run(['/usr/bin/git', 'init', '--quiet', str(workspace)], check=True)
    mcp = output/'empty-mcp.json'
    mcp.write_text('{"mcpServers":{}}\n')
    env = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', HOME=str(home), TMPDIR=str(output/'tmp'),
               LANG='en_US.UTF-8', NANOATTENTION_API_KEY='local', GRAFF_CONTEXT=str(args.context),
               GRAFF_NO_TELEMETRY='1', GRAFF_LEARNING_PRIVACY='local', GRAFF_FLEET='off',
               GRAFF_MCP_CONFIG=str(mcp))
    baseline = subprocess.run(['/usr/bin/python3', '-m', 'unittest', '-q'], cwd=workspace,
                              env=env, capture_output=True, text=True, timeout=30)
    if baseline.returncode == 0:
        raise RuntimeError('fixture must fail before Codegraff edits')
    (output/'baseline-tests.txt').write_text(baseline.stdout+baseline.stderr)
    server, agent = None, None
    server_log = (output/'adapter-stderr.txt').open('w')
    report = dict(status='unverified', configured_context=args.context,
                  scope='synthetic coding fixture with native KV; not a filled full-window benchmark')
    try:
        server = subprocess.Popen([sys.executable, '-B', str(root/'tools/codegraff_adapter.py'),
                                   str(args.checkpoint.resolve()), '--binary', str(args.binary.resolve()),
                                   '--port', '0', '--context', str(args.context),
                                   '--max-tokens', str(args.max_tokens), '--timeout', str(args.timeout)],
                                  env=dict(os.environ, NANOATTENTION_API_KEY='local'),
                                  stdout=subprocess.PIPE, stderr=server_log, text=True)
        ready = receive_line(server, 90)
        if ready.get('event') != 'ready' or ready.get('host') != '127.0.0.1':
            raise RuntimeError('unexpected adapter readiness')
        base = 'http://127.0.0.1:{}'.format(ready['port'])
        profile = output/'fixture.sb'
        profile.write_text(sandbox_profile(output, graff, ready['port']))
        prefix = ['/usr/bin/sandbox-exec', '-f', str(profile)]
        router = dict(id='nanoattention', name='nanoattention native Flint', base_url=base+'/v1',
                      env_key='NANOATTENTION_API_KEY', default_model='flint', takes_effort=False)
        (workspace/'.graff/.config.router').write_text(json.dumps(router, indent=2)+'\n')
        route = subprocess.run(prefix+[str(graff), 'route', 'nanoattention/flint'], cwd=workspace,
                               env=env, capture_output=True, text=True, timeout=60)
        (output/'route.txt').write_text(route.stdout+route.stderr)
        if (route.returncode != 0 or 'nanoattention/flint → nanoattention/flint' not in route.stdout
                or 'billing: local' not in route.stdout):
            raise RuntimeError('Codegraff route did not select the local nanoattention provider')
        system = ('You are a coding agent in a disposable fixture. Use only read_file, edit_file or '
                  'write_file to fix the requested file. Do not run commands, delegate, use network, '
                  'or touch other files. Do not ask questions. The external test driver will run '
                  'the tests. Keep reasoning brief and finish with one sentence.')
        task = ('Read clamp.py and test_clamp.py, then fix clamp.py so all provided tests pass. '
                'Do not modify tests. Use the file tools to apply the fix, not just show code.')
        with (output/'graff-stdout.jsonl').open('w') as stdout, (output/'graff-stderr.txt').open('w') as stderr:
            agent = subprocess.Popen(prefix+[str(graff), '-p', task, '--model', 'nanoattention/flint',
                                      '--system-prompt', system, '--no-resume', '--no-autocommit', '--yolo',
                                      '--no-telemetry', '--learning-privacy', 'local', '--no-subagent-tier',
                                      '--subagent-provider', 'nanoattention', '--subagent-model', 'flint',
                                      '--max-model-calls', '6', '--max-run-tool-calls', '8',
                                      '--max-tool-calls', '8', '--output-format', 'stream-json',
                                      '--context-limit', 'skill_catalog_bytes=256',
                                      '--context-limit', 'agents_md_bytes=256',
                                      '--context-limit', 'mcp_schema_bytes=256'], cwd=workspace, env=env,
                                     stdout=stdout, stderr=stderr)
            agent.wait(timeout=args.timeout)
        metrics = fetch(base+'/v1/nanoattention/metrics')
        (output/'adapter-metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        if agent.returncode != 0 or metrics['failures'] or metrics['worker_starts'] != 1:
            raise RuntimeError('Codegraff/native run did not finish cleanly')
        if sum(row['tool_call_count'] for row in metrics['requests']) < 2:
            raise RuntimeError('real Codegraff file-tool cycle was not observed')
        if not any(row['reused_prefix_tokens'] > 0 for row in metrics['requests'][1:]):
            raise RuntimeError('native KV reuse was not observed')
        if hashlib.sha256((workspace/'test_clamp.py').read_bytes()).hexdigest() != tests_hash:
            raise RuntimeError('Codegraff modified the tests')
        final = subprocess.run(prefix+['/usr/bin/python3', '-m', 'unittest', '-v'], cwd=workspace,
                               env=env, capture_output=True, text=True, timeout=30)
        (output/'final-tests.txt').write_text(final.stdout+final.stderr)
        if final.returncode != 0 or (workspace/'clamp.py').read_text() == FIXTURE:
            raise RuntimeError('Codegraff did not produce a passing real fixture change')
        report.update(status='pass', codegraff_exit=agent.returncode, fixture_tests=6,
                      adapter_requests=len(metrics['requests']), native_worker_starts=metrics['worker_starts'],
                      metrics=metrics, network_scope='OS sandbox: loopback adapter only',
                      filesystem_scope='OS sandbox: fixture writes; unrelated user reads denied')
    finally:
        stop_owned(agent)
        stop_owned(server)
        server_log.close()
        report['adapter_exit'] = server.returncode if server else None
        (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
