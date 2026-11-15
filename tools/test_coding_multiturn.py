#!/usr/bin/env python3
"""CPU-only coding protocol tests: no checkpoint, MLX, native GPU, or server loads.

Run: python3 -m unittest discover -s tools -p test_coding_multiturn.py -v
"""
import argparse
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().with_name("coding_multiturn.py")
SPEC = importlib.util.spec_from_file_location("coding_multiturn", MODULE_PATH)
coding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coding)
try:
    import numpy as np
except ImportError:
    np = None


class Tokenizer:
    """Byte-like mock template, with a special terminal EOS token preserved."""
    eos_token_id = 3
    chat_template = "synthetic test template"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert clean_up_tokenization_spaces is False
        return "".join(chr(x) for x in ids if not (skip_special_tokens and x == 3))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **unused):
        parts = ["<bos>"]
        for message in messages:
            if message["role"] == "user":
                parts.append("<user>" + message["content"] + "</user>")
            else:
                parts.append("<assistant>" + message["content"] + "\x03")
        if add_generation_prompt:
            parts.append("<assistant>")
        text = "".join(parts)
        return self.encode(text) if tokenize else text


class PrefixTests(unittest.TestCase):
    def test_incomplete_reasoning_code_is_not_a_final_answer(self):
        tokenizer = Tokenizer()
        reasoning = '<think>```python\ndef plan_order(graph):\n return []\n```'
        answer, complete = coding.final_answer(tokenizer, tokenizer.encode(reasoning))
        self.assertFalse(complete)
        self.assertEqual(answer, '')
        final = '```python\ndef plan_order(graph):\n return sorted(graph)\n```'
        answer, complete = coding.final_answer(tokenizer, tokenizer.encode(reasoning+'</think>'+final)+[3])
        self.assertTrue(complete)
        self.assertEqual(answer, final)
        self.assertTrue(coding.syntax_status(answer)['has_plan_order_definition'])

    def test_checkpoint_eos_list_takes_priority_over_single_tokenizer_eos(self):
        tokenizer = Tokenizer()
        self.assertEqual(coding.eos_token_ids(tokenizer), [3])
        tokenizer._flint_eos_token_ids = [3, 4, 5, 18]
        self.assertEqual(coding.eos_token_ids(tokenizer), [3, 4, 5, 18])

    def test_raw_eos_and_incremental_suffix_never_replayed(self):
        conversation = coding.Conversation(Tokenizer())
        suffix, first = conversation.user("initial task")
        generated = list(map(ord, "answer")) + [3]
        text, completion = conversation.assistant(generated)
        cached = suffix + generated
        self.assertEqual(conversation.cached_ids, cached)
        self.assertEqual(text, "answer\x03")
        # Prefer the canonical template-provided EOS, while preserving the raw
        # generated EOS ID in the cached prefix. No duplicate EOS is replayed.
        suffix2, second = conversation.user("please refine")
        self.assertEqual(second["cached_prefix_tokens"], len(cached))
        self.assertEqual(second["full_render_tokens"], len(cached) + len(suffix2))
        self.assertTrue(second["strict_id_prefix_verified"])
        self.assertTrue(second["strict_byte_prefix_verified"])
        full = conversation.pending[2]
        self.assertEqual(full, cached + suffix2)
        self.assertEqual(first["cached_prefix_tokens"], 0)
        self.assertTrue(completion["assistant_roundtrip_verified"])
        self.assertTrue(completion["terminal_eos_supplied_by_template"])
        self.assertEqual(completion["canonical_closure_tokens_not_yet_cached"], 0)

    def test_canonical_eos_insertion_without_duplicate(self):
        class StrictTokenizer(Tokenizer):
            def apply_chat_template(self, messages, **kwargs):
                if any(m["role"] == "assistant" and "\x03" in m["content"] for m in messages):
                    # A template that rejects embedded EOS would throw rather
                    # than render, but normal templates simply strip it.
                    messages = [dict(m, content=m["content"].replace("\x03", "")) for m in messages]
                return super().apply_chat_template(messages, **kwargs)
        conversation = coding.Conversation(StrictTokenizer())
        initial, _ = conversation.user("task")
        generated = list(map(ord, "answer")) + [3]
        _, boundary = conversation.assistant(generated)
        next_ids, _ = conversation.user("refine")
        self.assertEqual(next_ids[0], ord("<"))  # No additional EOS.
        self.assertEqual(conversation.pending[2], initial + generated + next_ids)
        self.assertEqual(boundary["canonical_closure_tokens_not_yet_cached"], 0)

    def test_length_limited_assistant_closure_is_new_input(self):
        conversation = coding.Conversation(Tokenizer())
        old, _ = conversation.user("task")
        _, boundary = conversation.assistant(list(map(ord, "unfinished")))
        self.assertEqual(boundary["canonical_closure_tokens_not_yet_cached"], 1)
        suffix, _ = conversation.user("continue")
        self.assertEqual(suffix[0], 3)
        self.assertEqual(conversation.pending[2], old + list(map(ord, "unfinished")) + suffix)

    def test_prefix_mismatch_fails_without_mutating_cached_ids(self):
        tokenizer = Tokenizer()
        conversation = coding.Conversation(tokenizer)
        conversation.user("task")
        conversation.assistant(list(map(ord, "answer")) + [3])
        cached = list(conversation.cached_ids)
        original = tokenizer.apply_chat_template
        def broken(messages, **kwargs):
            rendered = original(messages, **kwargs)
            return ([42] + rendered) if isinstance(rendered, list) else "*" + rendered
        tokenizer.apply_chat_template = broken
        with self.assertRaisesRegex(coding.ProtocolError, "prefix mismatch"):
            conversation.user("followup")
        self.assertEqual(conversation.cached_ids, cached)
        self.assertIsNone(conversation.pending)

    def test_decode_loss_fails_closed_but_keeps_raw_history(self):
        class Lossy(Tokenizer):
            def decode(self, ids, **kwargs):
                return super().decode(ids, **kwargs).replace("x", "y")
        conversation = coding.Conversation(Lossy())
        initial, _ = conversation.user("task")
        with self.assertRaisesRegex(coding.ProtocolError, "round trip"):
            conversation.assistant([ord("x"), 3])
        self.assertEqual(conversation.cached_ids, initial + [ord("x"), 3])

    def test_template_text_id_mismatch_fails(self):
        class Broken(Tokenizer):
            def apply_chat_template(self, messages, tokenize=True, **kwargs):
                value = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
                return [123] + value if tokenize else value
        with self.assertRaisesRegex(coding.ProtocolError, "text/ID"):
            coding.Conversation(Broken()).user("task")

    def test_unique_structured_context_and_honest_target(self):
        tokenizer = Tokenizer()
        text, sizing = coding.build_first_turn(tokenizer, 16000)
        self.assertIn("SYNTHETIC SOURCE/CONTEXT", text)
        self.assertIn("component/0000_extract.py", text)
        self.assertIn("manifest targets.", text)
        self.assertIn("USER TASK", text)
        self.assertGreater(sizing["synthetic_components"], 1)
        _, ids = coding.render(tokenizer, [{"role": "user", "content": text}], True)
        self.assertEqual(sizing["actual_prompt_tokens"], len(ids))
        self.assertEqual(sizing["target_prompt_tokens"], 16000)
        for i in range(sizing["synthetic_components"]):
            self.assertEqual(text.count("def describe_{:04d}".format(i)), 1)
        self.assertLess(abs(len(ids) - 16000), 1000)

    def test_static_checks_never_execute(self):
        text = "```python\nraise RuntimeError('must not execute')\ndef plan_order(graph):\n    return []\n```"
        status = coding.syntax_status(text)
        self.assertEqual(status["syntax"], "valid")
        self.assertTrue(status["has_plan_order_definition"])
        self.assertFalse(status["executed"])
        self.assertEqual(status["functional"], "not_tested")
        self.assertEqual(coding.syntax_status("def broken(")["syntax"], "invalid")


@unittest.skipIf(np is None, "numpy unavailable; no GPU dependencies are required")
class ReferenceTests(unittest.TestCase):
    def make_transcript(self):
        tokenizer = Tokenizer()
        conversation = coding.Conversation(tokenizer)
        transcript = {"schema": 1, "protocol": coding.PROTOCOL, "status": "complete",
                      "context": 10000, "turns": [], "conversation_token_ids": []}
        arrays = {}
        for index, content in enumerate(["task", "missing deps and cycles", "no mutation and lexical ties"]):
            suffix, boundary = conversation.user(content)
            before = len(conversation.cached_ids)
            generated = [65, 3]
            request = {"id": index + 1, "tokens": suffix, "max_tokens": 4,
                       "reset": False, "return_logits": True}
            prefill = np.zeros(256, dtype=np.float32)
            prefill[65] = 1
            final = np.zeros(256, dtype=np.float32)
            final[66] = 1
            response = {"id": index + 1, "tokens": generated, "eos": True,
                        "cache_before": before, "cache_after": before + len(suffix) + 2,
                        "prefill_tokens": len(suffix), "decode_forward_tokens": 2,
                        "reused_prefix_tokens": before, "prefill_seconds": 0.125,
                        "decode_seconds": 0.25, "kv_allocated_bytes": 100000,
                        "kv_valid_bytes": 1000}
            _, completion = conversation.assistant(generated)
            boundary.update(completion)
            transcript["turns"].append({"request": request, "response": response, "boundary": boundary})
            transcript["conversation_token_ids"].extend(suffix + generated)
            arrays["turn_{}_prefill_logits".format(index + 1)] = prefill
            arrays["turn_{}_final_logits".format(index + 1)] = final
        return transcript, arrays

    def test_validate_exact_incremental_counts_including_final_eos(self):
        transcript, arrays = self.make_transcript()
        turns = coding.validate_transcript(transcript, arrays, np)
        self.assertEqual(len(turns), 3)
        final = turns[-1]["response"]["cache_after"]
        self.assertEqual(final, len(transcript["conversation_token_ids"]))
        self.assertEqual(transcript["conversation_token_ids"][-1], 3)

    def test_reject_replay_reset_missing_final_commit_and_wrong_history(self):
        transcript, arrays = self.make_transcript()
        mutations = [
            lambda t: t["turns"][1]["response"].update(reused_prefix_tokens=0),
            lambda t: t["turns"][1]["response"].update(decode_forward_tokens=1),
            lambda t: t["turns"][1]["response"].update(cache_after=1),
            lambda t: t["turns"][1]["request"].update(reset=True),
            lambda t: t["conversation_token_ids"].pop(),
            lambda t: t["turns"][1]["boundary"].update(strict_id_prefix_verified=False),
            lambda t: t.update(status="incomplete"),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(transcript)
            mutate(bad)
            with self.subTest(mutate=mutate), self.assertRaises(coding.ProtocolError):
                coding.validate_transcript(bad, arrays, np)

    def test_reject_logits_shape_nonfinite_dtype_and_unexpected_keys(self):
        transcript, arrays = self.make_transcript()
        for value in (np.zeros((1, 256), dtype=np.float32),
                      np.zeros(256, dtype=np.float64),
                      np.full(256, np.nan, dtype=np.float32)):
            bad = dict(arrays, turn_1_final_logits=value)
            with self.assertRaises(coding.ProtocolError):
                coding.validate_transcript(transcript, bad, np)
        with self.assertRaises(coding.ProtocolError):
            coding.validate_transcript(transcript, dict(arrays, surprise=np.zeros(1)), np)

    def test_sequential_reference_preserves_one_stream_and_commits_eos(self):
        transcript, arrays = self.make_transcript()
        turns = coding.validate_transcript(transcript, arrays, np)
        endpoints, expected = {}, []
        for i, turn in enumerate(turns):
            expected.extend(turn["request"]["tokens"])
            endpoints[len(expected)] = arrays["turn_{}_prefill_logits".format(i + 1)]
            expected.extend(turn["response"]["tokens"])
            endpoints[len(expected)] = arrays["turn_{}_final_logits".format(i + 1)]
        seen = []
        def forward(token):
            self.assertIs(type(token), int)  # Exactly one token, never batched.
            seen.append(token)
            if len(seen) in endpoints:
                return endpoints[len(seen)]
            result = np.zeros(256, dtype=np.float32)
            result[3] = 1
            return result
        reports = coding.sequential_reference(turns, arrays, forward, np)
        self.assertEqual(seen, expected)
        self.assertEqual(seen, transcript["conversation_token_ids"])
        self.assertEqual(seen[-1], 3)
        self.assertTrue(all(r[b]["allclose"] and r[b]["bit_exact"]
                            for r in reports for b in ("prefill", "final")))
        self.assertEqual(reports[1]["cache_before"], reports[0]["cache_after"])

    def test_reference_tolerance_and_bit_exact_are_distinct(self):
        native = np.array([1, 2, 3], dtype=np.float32)
        close = coding.compare_boundary(native, native + 1e-4, np)
        far = coding.compare_boundary(native, native + 0.1, np)
        self.assertTrue(close["allclose"])
        self.assertFalse(close["bit_exact"])
        self.assertTrue(close["greedy_top1_agreement"])
        self.assertFalse(far["allclose"])
        self.assertTrue(coding.compare_boundary(native, native.copy(), np)["bit_exact"])

    def test_reference_invalid_transcript_rejected_before_mlx_import(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            coding.save_json(output / "transcript.json", {"status": "incomplete"})
            np.savez(output / "boundaries.npz")
            args = argparse.Namespace(output=output, checkpoint=Path("not-a-checkpoint"))
            with patch.dict(sys.modules, {"mlx": None, "mlx.core": None, "mlx_lm": None}):
                with self.assertRaises(coding.ProtocolError):
                    coding.reference_run(args)


@unittest.skipIf(np is None, "numpy unavailable")
class DriverTests(unittest.TestCase):
    def run_mock(self, output, corrupt_second=False):
        sessions = []
        class MockNative:
            def __init__(self, command, timeout):
                self.command = command
                self.timeout = timeout
                self.offset = 0
                self.requests = []
                self.closed = False
                sessions.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *unused):
                self.closed = True

            def request(self, request):
                self.requests.append(copy.deepcopy(request))
                generated = list(map(ord, "def plan_order(graph):\n    return []\n")) + [3]
                prefill = np.zeros(256, dtype=np.float32)
                prefill[generated[0]] = 1
                final = np.zeros(256, dtype=np.float32)
                final[9] = 1
                before = self.offset
                self.offset += len(request["tokens"]) + len(generated)
                response = dict(id=request["id"], tokens=generated, eos=True,
                                cache_before=before, cache_after=self.offset,
                                prefill_tokens=len(request["tokens"]),
                                decode_forward_tokens=len(generated),
                                reused_prefix_tokens=before, prefill_seconds=0.1,
                                decode_seconds=0.2, kv_allocated_bytes=10000,
                                kv_valid_bytes=5000, prefill_logits=prefill.tolist(),
                                final_logits=final.tolist())
                if corrupt_second and len(self.requests) == 2:
                    response["decode_forward_tokens"] -= 1
                return response
        args = argparse.Namespace(checkpoint=Path("unused-private-checkpoint"),
                                  binary=Path("unused-private-binary"), output=output,
                                  prompt_tokens=3000, max_tokens=64, context=10000, timeout=5)
        with patch.object(coding, "load_tokenizer", return_value=Tokenizer()), \
                patch.object(coding, "NativeSession", MockNative), \
                patch("sys.stdout", new=io.StringIO()):
            if corrupt_second:
                with self.assertRaises(coding.ProtocolError):
                    coding.native_run(args)
            else:
                coding.native_run(args)
        return sessions

    def test_native_complete_artifacts_and_one_session(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            sessions = self.run_mock(output)
            self.assertEqual(len(sessions), 1)
            self.assertTrue(sessions[0].closed)
            self.assertEqual(len(sessions[0].requests), 3)
            self.assertEqual(sessions[0].command[1], "session")
            self.assertEqual(sessions[0].command[-4:], ["--mlx-attention", "1", "--staged-row-tile", "2"])
            self.assertTrue(all(request["reset"] is False for request in sessions[0].requests))
            text = (output / "transcript.json").read_text()
            self.assertNotIn("unused-private", text)
            transcript = json.loads(text)
            with np.load(output / "boundaries.npz", allow_pickle=False) as arrays:
                coding.validate_transcript(transcript, arrays, np)
            self.assertEqual(transcript["status"], "complete")
            self.assertTrue(all(turn["static_check"]["syntax"] == "valid"
                                for turn in transcript["turns"]))
            self.assertTrue(all(turn["static_check"]["functional"] == "not_tested"
                                for turn in transcript["turns"]))
            self.assertIn("dependency absent", transcript["turns"][1]["user_text"])
            self.assertIn("lexicographically", transcript["turns"][2]["user_text"])

    def test_counter_failure_stops_session_and_saves_sanitized_partial_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            sessions = self.run_mock(output, corrupt_second=True)
            self.assertTrue(sessions[0].closed)
            self.assertEqual(len(sessions[0].requests), 2)
            text = (output / "transcript.json").read_text()
            self.assertNotIn("unused-private", text)
            transcript = json.loads(text)
            self.assertEqual(transcript["status"], "incomplete")
            self.assertEqual(transcript["failure_type"], "ProtocolError")
            self.assertEqual(len(transcript["turns"]), 1)
            with np.load(output / "boundaries.npz", allow_pickle=False) as arrays:
                self.assertEqual(len(arrays.files), 2)


class NativeOwnershipTests(unittest.TestCase):
    def session(self, code, timeout=5):
        # Real CPU-only disposable mock; never runs flint or loads a checkpoint.
        return coding.NativeSession([sys.executable, "-u", "-c", code], timeout=timeout)

    def assert_stopped(self, process):
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)

    def test_good_jsonl_uses_one_owned_child_and_stops_it(self):
        code = "import sys,json\nfor line in sys.stdin:\n x=json.loads(line); print(json.dumps({'id':x['id'],'tokens':x['tokens']}),flush=True)"
        with self.session(code) as native:
            process = native.process
            self.assertEqual(os.getpgid(process.pid), os.getpgrp())
            for i in range(3):
                request = {"id": i, "tokens": [i + 10], "reset": False}
                self.assertEqual(native.request(request), {"id": i, "tokens": [i + 10]})
                self.assertIs(native.process, process)
        self.assert_stopped(process)

    def test_abnormal_exit_after_successful_response_is_not_a_pass(self):
        code = "import sys,json\nfor line in sys.stdin:\n print(json.dumps({'id':json.loads(line)['id']}),flush=True)\nsys.exit(7)"
        with self.assertRaisesRegex(coding.ProtocolError, "abnormally after EOF"):
            with self.session(code) as native:
                process = native.process
                self.assertEqual(native.request({'id': 1}), {'id': 1})
        self.assert_stopped(process)

    def test_malformed_json_cleans_up(self):
        with self.assertRaises(coding.ProtocolError):
            with self.session("import sys,time\nsys.stdin.readline();print('not json',flush=True);time.sleep(60)") as native:
                process = native.process
                native.request({"id": 1, "tokens": [10]})
        self.assert_stopped(process)

    def test_interrupt_cleans_up(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.session("import time;time.sleep(60)") as native:
                process = native.process
                raise KeyboardInterrupt()
        self.assert_stopped(process)

    def test_timeout_cleans_up(self):
        with self.assertRaisesRegex(coding.ProtocolError, "timed out"):
            with self.session("import time;time.sleep(60)", timeout=0.05) as native:
                process = native.process
                native.request({"id": 1})
        self.assert_stopped(process)

    def test_eof_and_child_failure_cleanup(self):
        with self.assertRaises(coding.ProtocolError):
            with self.session("import sys;sys.stdin.readline();sys.exit(7)") as native:
                process = native.process
                native.request({"id": 1})
        self.assert_stopped(process)

    def test_launch_failure_closes_owned_stderr(self):
        captured = []
        def fail(*unused, **kwargs):
            captured.append(kwargs["stderr"])
            raise OSError("synthetic launch failure")
        with self.assertRaises(OSError):
            with coding.NativeSession(["not-run"], popen=fail):
                self.fail("must not enter")
        self.assertTrue(captured[0].closed)


if __name__ == "__main__":
    unittest.main()
