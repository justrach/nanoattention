#!/usr/bin/env python3
"""Launch terminal Graff on an owned loopback Flint adapter, without global edits.

Run under tools/guarded_run.py --lock <existing-inference-lock> --timeout <budget>
-- <python> tools/graff_native.py <checkpoint> --graff <installed-graff>
--workspace <workspace> [options] -- [Graff arguments]. The outer guard owns the
lock and process group; this launcher never acquires a lock or creates a group.
--context allocates KV capacity, not a guarantee of filled-window throughput.
"""
import argparse
import json
import math
import os
from pathlib import Path
import queue
import secrets
import signal
import stat
import subprocess
import sys
import threading


class LaunchError(Exception):
    pass


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def supports_builtin_flint(graff):
    """Detect the built-in provider via the public, content-free schema."""
    probe_env = dict(os.environ)
    probe_env.pop('NANOATTENTION_API_KEY', None)
    probe_env.pop('GRAFF_NANOATTENTION_URL', None)
    try:
        result = subprocess.run([str(graff), '--schema'], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                env=probe_env, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0 or len(result.stdout) > 4 * 1024 * 1024:
        return False
    try:
        schema = json.loads(result.stdout)
    except (ValueError, TypeError):
        return False
    providers = schema.get('providers') if isinstance(schema, dict) else None
    return (schema.get('harness') == 'simple-harness'
            and isinstance(providers, list)
            and any(isinstance(item, dict) and item.get('id') == 'nanoattention'
                    and item.get('kind') == 'openai'
                    and item.get('default_model') == 'flint' for item in providers))


class RouterLease:
    """Exclusive local metadata file; never replace existing or user-edited data."""
    def __init__(self, workspace):
        self.directory = workspace / '.graff'
        self.path = self.directory / '.config.router'
        self.identity = None
        self.content = None

    def check(self):
        if self.directory.is_symlink():
            raise LaunchError('workspace .graff must not be a symlink')
        if os.path.lexists(self.path):
            raise LaunchError('workspace already has .graff/.config.router; refusing to overwrite')

    def acquire(self, port):
        self.check()
        self.directory.mkdir(exist_ok=True)
        self.content = (json.dumps(dict(
            id='nanoattention', name='nanoattention native Flint',
            base_url='http://127.0.0.1:{}/v1'.format(port),
            env_key='NANOATTENTION_API_KEY', default_model='flint',
            takes_effort=False), indent=2) + '\n').encode()
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise LaunchError('workspace router appeared during startup; refusing to overwrite') from None
        try:
            info = os.fstat(fd)
            self.identity = (info.st_dev, info.st_ino)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(self.content)
        except BaseException:
            self.release()
            raise

    def release(self):
        if self.identity is None:
            return
        try:
            info = self.path.lstat()
            if (stat.S_ISREG(info.st_mode)
                    and (info.st_dev, info.st_ino) == self.identity
                    and self.path.read_bytes() == self.content):
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.identity = None


def receive_ready(process, timeout, context):
    """One blocking readiness event, not health-check or process polling."""
    result = queue.Queue(maxsize=1)

    def read():
        try:
            result.put(process.stdout.readline(65537))
        except Exception:
            result.put(None)

    threading.Thread(target=read, daemon=True).start()
    try:
        line = result.get(timeout=timeout)
    except queue.Empty:
        raise LaunchError('adapter readiness timed out') from None
    try:
        ready = json.loads(line) if line and len(line) <= 65536 else None
    except (ValueError, TypeError):
        ready = None
    if (not isinstance(ready, dict) or ready.get('event') != 'ready'
            or ready.get('host') != '127.0.0.1' or ready.get('model') != 'flint'
            or type(ready.get('port')) is not int or not 0 < ready['port'] < 65536
            or ready.get('context_length') != context):
        raise LaunchError('adapter exited or returned invalid readiness/context metadata')
    return ready['port']


def stop_owned(process):
    if process is None:
        return
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    finally:
        if process.stdout is not None:
            process.stdout.close()


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index('--') if '--' in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--graff', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--binary', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'zig-out/bin/flint')
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--kv-bf16', action='store_true', help='experimental BF16 global KV; FP32 remains default')
    parser.add_argument('--speculative-tokens', type=int, default=0, choices=range(33),
                        help='opt-in greedy history drafting and batched target verification (0 disables)')
    parser.add_argument('--speculative-query-group', type=int, default=2, choices=(2, 4, 8))
    parser.add_argument('--speculative-expert-tile', type=int, default=0, choices=(0, 1, 2, 4, 8))
    parser.add_argument('--speculative-weight-reuse', action='store_true')
    parser.add_argument('--prefix-cache', type=Path,
                        help='opt-in directory of exact on-disk prefix snapshots, passed to the adapter')
    parser.add_argument('--prefix-cache-bytes', type=int, help='prefix snapshot size cap (default 4 GiB)')
    parser.add_argument('--timeout', type=float, default=1200,
                        help='native request and Graff session time budget in seconds')
    args = parser.parse_args(argv[:split])
    args.graff_args = argv[split + 1:]
    if not args.speculative_tokens and (args.speculative_query_group != 2 or args.speculative_expert_tile or args.speculative_weight_reuse):
        parser.error('speculative tuning requires --speculative-tokens')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    if args.prefix_cache_bytes is not None and (args.prefix_cache is None or args.prefix_cache_bytes <= 0):
        parser.error('--prefix-cache-bytes requires --prefix-cache and a positive size')
    if args.prefix_cache is not None:
        args.prefix_cache = args.prefix_cache.resolve()
    if not 0 < args.max_tokens < args.context:
        parser.error('require 0 < --max-tokens < --context')
    if any(value == '--model' or value.startswith('--model=')
           or value.startswith('-m') and not value.startswith('--')
           for value in args.graff_args):
        parser.error('Graff --model/-m is pinned to nanoattention/flint')
    for name in ('checkpoint', 'graff', 'workspace', 'binary'):
        setattr(args, name, getattr(args, name).resolve())
    if not args.workspace.is_dir() or not args.checkpoint.is_dir():
        parser.error('workspace and checkpoint must be existing directories')
    for path in (args.graff, args.binary):
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error('--graff and --binary must be executable files')
    try:
        maximum = json.loads((args.checkpoint / 'config.json').read_text())['max_position_embeddings']
        if type(maximum) is not int or args.context > maximum:
            raise ValueError()
    except (OSError, ValueError, KeyError, TypeError):
        parser.error('checkpoint must declare a valid max_position_embeddings >= --context')
    return args


def run(args):
    built_in = supports_builtin_flint(args.graff)
    lease = RouterLease(args.workspace)
    if not built_in:
        lease.check()
    server = agent = None
    # Preserve the user's normal interactive auth environment. Only this child
    # tree receives the ephemeral adapter credential; it is never serialized.
    env = dict(os.environ, NANOATTENTION_API_KEY=secrets.token_urlsafe(32),
               GRAFF_CONTEXT=str(args.context))
    previous = {}

    def interrupted(signum, frame):
        raise Interrupted(signum)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, interrupted)
        server = subprocess.Popen([
            sys.executable, '-B', str(Path(__file__).with_name('codegraff_adapter.py')),
            str(args.checkpoint), '--binary', str(args.binary), '--port', '0',
            '--context', str(args.context), '--max-tokens', str(args.max_tokens),
            '--timeout', str(args.timeout)] + (['--kv-bf16'] if getattr(args, 'kv_bf16', False) else []) + (['--speculative-tokens', str(args.speculative_tokens),
                '--speculative-query-group', str(getattr(args, 'speculative_query_group', 2)),
                '--speculative-expert-tile', str(getattr(args, 'speculative_expert_tile', 0))]
                if getattr(args, 'speculative_tokens', 0) else []) +
                (['--speculative-weight-reuse'] if getattr(args, 'speculative_weight_reuse', False) else []) +
                (['--prefix-cache', str(args.prefix_cache)] if getattr(args, 'prefix_cache', None) is not None else []) +
                (['--prefix-cache-bytes', str(args.prefix_cache_bytes)] if getattr(args, 'prefix_cache_bytes', None) is not None else []),
            env=env, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, start_new_session=False)
        port = receive_ready(server, min(90, args.timeout), args.context)
        if built_in:
            env['GRAFF_NANOATTENTION_URL'] = 'http://127.0.0.1:{}/v1/chat/completions'.format(port)
        else:
            lease.acquire(port)
        agent = subprocess.Popen([str(args.graff), '--model', 'nanoattention/flint',
                                  '--no-subagent-tier', '--subagent-provider', 'nanoattention',
                                  '--subagent-model', 'flint'] + args.graff_args, cwd=args.workspace, env=env,
                                 start_new_session=False)
        return agent.wait(timeout=args.timeout)
    finally:
        # A repeated terminal/guard signal must not interrupt owned teardown.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            try:
                stop_owned(agent)
            finally:
                try:
                    stop_owned(server)
                finally:
                    lease.release()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main(argv=None):
    try:
        return run(parse_args(argv))
    except Interrupted as error:
        return 128 + error.signum
    except KeyboardInterrupt:
        return 130
    except LaunchError as error:
        # LaunchError messages are static launcher text, never adapter output.
        print('graff_native: ' + str(error), file=sys.stderr)
        return 1
    except (OSError, subprocess.TimeoutExpired):
        # Do not echo child command lines, environment, or potentially sensitive
        # adapter diagnostics from third-party exceptions.
        print('graff_native: local process failed or exceeded its time budget; '
              'no global configuration changed', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
