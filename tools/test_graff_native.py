"""CPU-only launcher/file-lease tests; no subprocesses, sockets or model loads.

python3 -m unittest discover -s nanoattention/tools -p test_graff_native.py -v
"""
import argparse
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import graff_native as launcher


def ready(context=32768, **changes):
    metadata = dict(event='ready', host='127.0.0.1', port=19234,
                    model='flint', context_length=context)
    metadata.update(changes)
    return json.dumps(metadata) + '\n'


class FakeProcess:
    def __init__(self, output=None, failure=None, code=0, stubborn=False):
        self.stdout = io.StringIO(output) if output is not None else None
        self.returncode = None
        self.failure = failure
        self.code = code
        self.stubborn = stubborn
        self.terminated = self.killed = False
        self.waits = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.returncode = -signal.SIGTERM

    def kill(self):
        self.killed = True
        self.returncode = -signal.SIGKILL

    def wait(self, timeout):
        self.waits += 1
        if self.returncode is not None:
            return self.returncode
        if self.failure is not None:
            failure, self.failure = self.failure, None
            raise failure
        if self.stubborn:
            raise subprocess.TimeoutExpired('fake', timeout)
        self.returncode = self.code
        return self.code


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.args = argparse.Namespace(
            workspace=self.workspace, checkpoint=self.workspace / 'checkpoint',
            graff=Path('/fake/graff'), binary=Path('/fake/flint'),
            context=32768, max_tokens=4096, timeout=20, kv_bf16=False,
            graff_args=['-p', 'synthetic task'])
        self.router = self.workspace / '.graff/.config.router'

    def test_ready_runs_pinned_child_and_never_persists_secret(self):
        server, agent = FakeProcess(ready()), FakeProcess(code=7)
        calls = []

        def popen(command, **kwargs):
            calls.append((command, kwargs))
            if len(calls) == 1:
                self.assertFalse(self.router.exists())
                self.assertIn('--port', command)
                self.assertEqual(command[command.index('--port') + 1], '0')
                return server
            metadata = json.loads(self.router.read_text())
            self.assertEqual(metadata['base_url'], 'http://127.0.0.1:19234/v1')
            self.assertEqual(metadata['env_key'], 'NANOATTENTION_API_KEY')
            self.assertEqual(set(metadata), {'id', 'name', 'base_url', 'env_key',
                                            'default_model', 'takes_effort'})
            self.assertNotIn('ephemeral-test-secret', self.router.read_text())
            self.assertEqual(self.router.stat().st_mode & 0o777, 0o600)
            return agent

        with patch.object(launcher.subprocess, 'Popen', side_effect=popen), \
                patch.object(launcher.secrets, 'token_urlsafe', return_value='ephemeral-test-secret'), \
                patch.dict(os.environ, {'EXISTING_AUTH': 'preserved'}, clear=True):
            self.assertEqual(launcher.run(self.args), 7)
        self.assertEqual(calls[1][0], ['/fake/graff', '--model', 'nanoattention/flint',
                                     '--no-subagent-tier', '--subagent-provider', 'nanoattention',
                                     '--subagent-model', 'flint', '-p', 'synthetic task'])
        for command, options in calls:
            self.assertIs(options['start_new_session'], False)
            self.assertNotIn('process_group', options)
            self.assertNotIn('preexec_fn', options)
            self.assertNotIn('ephemeral-test-secret', ' '.join(command))
            self.assertEqual(options['env']['NANOATTENTION_API_KEY'], 'ephemeral-test-secret')
            self.assertEqual(options['env']['GRAFF_CONTEXT'], '32768')
            self.assertEqual(options['env']['EXISTING_AUTH'], 'preserved')
        self.assertEqual(calls[1][1]['cwd'], self.workspace)
        self.assertEqual(calls[0][1]['stderr'], subprocess.DEVNULL)
        self.assertTrue(server.terminated)
        self.assertTrue(server.stdout.closed)
        self.assertGreater(agent.waits, 1)  # Already-exited children are reaped too.
        self.assertFalse(agent.terminated)
        self.assertFalse(self.router.exists())
        self.assertNotIn('NANOATTENTION_API_KEY', os.environ)

    def test_opt_in_bf16_is_forwarded_only_to_owned_adapter(self):
        self.args.kv_bf16 = True
        server, agent = FakeProcess(ready()), FakeProcess()
        with patch.object(launcher.subprocess, 'Popen', side_effect=[server, agent]) as spawn:
            self.assertEqual(launcher.run(self.args), 0)
        self.assertIn('--kv-bf16', spawn.call_args_list[0].args[0])
        self.assertNotIn('--kv-bf16', spawn.call_args_list[1].args[0])
        self.assertFalse(self.router.exists())
        self.assertTrue(server.terminated)

    def test_opt_in_speculation_reaches_only_owned_adapter(self):
        self.args.speculative_tokens = 4
        self.args.speculative_query_group = 4
        self.args.speculative_expert_tile = 1
        self.args.speculative_weight_reuse = True
        server, agent = FakeProcess(ready()), FakeProcess()
        with patch.object(launcher.subprocess, 'Popen', side_effect=[server, agent]) as spawn:
            self.assertEqual(launcher.run(self.args), 0)
        adapter = spawn.call_args_list[0].args[0]
        self.assertEqual(adapter[adapter.index('--speculative-tokens') + 1], '4')
        self.assertEqual(adapter[adapter.index('--speculative-query-group') + 1], '4')
        self.assertEqual(adapter[adapter.index('--speculative-expert-tile') + 1], '1')
        self.assertIn('--speculative-weight-reuse', adapter)
        for flag in ('--speculative-tokens', '--speculative-query-group', '--speculative-expert-tile', '--speculative-weight-reuse'):
            self.assertNotIn(flag, spawn.call_args_list[1].args[0])
        self.assertFalse(self.router.exists())
        self.assertTrue(server.terminated)

    def test_router_conflict_never_starts_server(self):
        self.router.parent.mkdir()
        self.router.write_text('user configuration')
        with patch.object(launcher.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(launcher.LaunchError, 'refusing to overwrite'):
                launcher.run(self.args)
        spawn.assert_not_called()
        self.assertEqual(self.router.read_text(), 'user configuration')

    def test_router_lease_preserves_user_edits_and_replacements(self):
        for replacement in (False, True):
            with self.subTest(replacement=replacement):
                lease = launcher.RouterLease(self.workspace)
                lease.acquire(1234)
                if replacement:
                    other = self.workspace / 'replacement'
                    other.write_bytes(lease.content)
                    other.replace(self.router)
                else:
                    self.router.write_text('edited by user')
                expected = self.router.read_bytes()
                lease.release()
                self.assertEqual(self.router.read_bytes(), expected)
                self.router.unlink()

    def test_symlink_directory_and_router_are_conflicts(self):
        elsewhere = self.workspace / 'elsewhere'
        elsewhere.mkdir()
        directory = self.workspace / '.graff'
        directory.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaises(launcher.LaunchError):
            launcher.RouterLease(self.workspace).check()
        directory.unlink()
        directory.mkdir()
        self.router.symlink_to(self.workspace / 'nonexistent')
        with self.assertRaises(launcher.LaunchError):
            launcher.RouterLease(self.workspace).acquire(1234)
        self.assertTrue(self.router.is_symlink())

    def test_readiness_rejects_eof_malformed_wrong_context_and_nonloopback(self):
        for output in ('', 'not json\n', ready(context=4096), ready(host='0.0.0.0'),
                       ready(port=True), ready(port=65536), ready(model='other')):
            with self.subTest(output=output):
                server = FakeProcess(output)
                with patch.object(launcher.subprocess, 'Popen', return_value=server) as spawn:
                    with self.assertRaises(launcher.LaunchError):
                        launcher.run(self.args)
                self.assertEqual(spawn.call_count, 1)
                self.assertTrue(server.terminated)
                self.assertFalse(self.router.exists())

    def test_readiness_waits_on_single_event_with_timeout(self):
        process = FakeProcess(ready())
        with patch.object(launcher.queue, 'Queue') as queues:
            queues.return_value.get.side_effect = launcher.queue.Empty
            with self.assertRaisesRegex(launcher.LaunchError, 'timed out'):
                launcher.receive_ready(process, 0.01, 32768)
            queues.return_value.get.assert_called_once_with(timeout=0.01)
        process.stdout.close()

    def test_child_launch_error_and_interruptions_clean_both_resources(self):
        failures = [OSError('synthetic launch error'), KeyboardInterrupt(),
                    launcher.Interrupted(signal.SIGTERM),
                    subprocess.TimeoutExpired('fake', 1)]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                server = FakeProcess(ready())
                agent = FakeProcess(failure=failure)
                values = [server, failure] if isinstance(failure, OSError) else [server, agent]
                handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
                with patch.object(launcher.subprocess, 'Popen', side_effect=values):
                    with self.assertRaises(type(failure)):
                        launcher.run(self.args)
                self.assertTrue(server.terminated)
                if not isinstance(failure, OSError):
                    self.assertTrue(agent.terminated)
                self.assertFalse(self.router.exists())
                for signum, handler in handlers.items():
                    self.assertEqual(signal.getsignal(signum), handler)

    def test_conflict_appearing_after_readiness_stops_only_owned_server(self):
        server = FakeProcess(ready())

        def startup(process, timeout, context):
            self.router.parent.mkdir()
            self.router.write_text('concurrent user route')
            return 1234

        with patch.object(launcher.subprocess, 'Popen', return_value=server) as spawn, \
                patch.object(launcher, 'receive_ready', side_effect=startup):
            with self.assertRaises(launcher.LaunchError):
                launcher.run(self.args)
        self.assertEqual(spawn.call_count, 1)
        self.assertTrue(server.terminated)
        self.assertEqual(self.router.read_text(), 'concurrent user route')

    def test_stubborn_owned_process_is_killed_and_reaped(self):
        process = FakeProcess(ready(), stubborn=True)
        launcher.stop_owned(process)
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertEqual(process.waits, 2)
        self.assertTrue(process.stdout.closed)

    def test_cli_validates_actual_checkpoint_capacity_and_pinned_model(self):
        checkpoint = self.workspace / 'checkpoint'
        checkpoint.mkdir()
        config = checkpoint / 'config.json'
        config.write_text('{"max_position_embeddings":32768}')
        binary = self.workspace / 'executable'
        binary.write_text('not executed')
        binary.chmod(0o700)
        command = [str(checkpoint), '--graff', str(binary), '--binary', str(binary),
                   '--workspace', str(self.workspace)]
        args = launcher.parse_args(command + ['--', '-p', 'synthetic task'])
        self.assertEqual(args.context, 32768)
        self.assertEqual(args.max_tokens, 4096)
        self.assertEqual(args.graff_args, ['-p', 'synthetic task'])
        for extra in (['--context', '65536'], ['--max-tokens', '32768'],
                      ['--timeout', 'nan'], ['--', '--model=hosted'],
                      ['--', '-m', 'hosted']):
            with self.subTest(extra=extra), patch('sys.stderr', new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    launcher.parse_args(command + extra)

    def test_main_reports_conflict_and_sanitizes_third_party_errors(self):
        cases = [(launcher.LaunchError('workspace router conflict'), 1, 'workspace router conflict'),
                 (OSError('sensitive-third-party-text'), 1, 'local process failed'),
                 (launcher.Interrupted(signal.SIGTERM), 143, ''),
                 (KeyboardInterrupt(), 130, '')]
        for failure, expected, message in cases:
            with self.subTest(failure=type(failure).__name__), \
                    patch.object(launcher, 'parse_args', return_value=self.args), \
                    patch.object(launcher, 'run', side_effect=failure), \
                    patch('sys.stderr', new_callable=io.StringIO) as stderr:
                self.assertEqual(launcher.main([]), expected)
                self.assertIn(message, stderr.getvalue())
                self.assertNotIn('sensitive-third-party-text', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
