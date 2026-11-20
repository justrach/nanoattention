"""CPU-only adapter contract tests; no model, subprocess, GPU or downloads.

Run: /usr/bin/python3 -m unittest discover -s nanoattention/tools \
    -p test_codegraff_adapter.py -v
The system Python on the development machine is Python 3.9.
"""

import copy
import http.client
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch


TOOLS = Path(__file__).resolve().parent
# Resolve adapter dependencies from its sibling tools, never the outer repo.
sys.path.insert(0, str(TOOLS))
try:
    SPEC = importlib.util.spec_from_file_location(
        "codegraff_adapter_under_test", TOOLS / "codegraff_adapter.py")
    adapter = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(adapter)
finally:
    sys.path.pop(0)


EOS = "\x03"
TOOL = {"type": "function", "function": {
    "name": "read_file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"], "additionalProperties": False}}}
RAW_CALL = ("<think>inspect first\nkeep these exact bytes</think>"
            "<tool_call>read_file<arg_key>path</arg_key>"
            "<arg_value>src/main.py</arg_value></tool_call>")


class CharacterTokenizer:
    """One character per token, with an exactly reversible terminal EOS.

    Normal assistant history renders reasoning/tool objects canonically. Native
    output deliberately uses different whitespace, so a prefix hit requires the
    adapter's raw-byte restoration, not an accidentally equivalent template.
    """
    def __init__(self):
        self.renders = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=False,
               clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token) for token in ids)

    def apply_chat_template(self, messages, tools, tokenize,
                            add_generation_prompt, return_dict=False):
        assert add_generation_prompt in (True, False)
        assert return_dict is False
        self.renders.append(copy.deepcopy(messages))
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
        for message in messages:
            text += "<" + message["role"] + ">"
            if message.get("reasoning_content"):
                text += "<think>" + message["reasoning_content"] + "</think>"
            text += message["content"]
            for call in message.get("tool_calls", []):
                function = call["function"]
                assert isinstance(function["arguments"], dict)
                text += "<tool_call>" + function["name"]
                for key, value in function["arguments"].items():
                    text += ("\n<arg_key>" + key + "</arg_key><arg_value>" +
                             str(value) + "</arg_value>")
                text += "</tool_call>"
            if not text.endswith(EOS):
                text += EOS
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


class FakeNativeSession:
    """Models the native protocol: truncate replays from the newest checkpoint
    at or below N; checkpoint_at and --prefix-cache keep exact token prefixes."""
    def __init__(self, factory, command, timeout):
        self.factory = factory
        self.command, self.timeout = command, timeout
        self.requests, self.close_calls, self.cache = [], [], []
        self.checkpoints = set()
        self.entered = False
        self.mutate = None
        self.failure = None

    def __enter__(self):
        self.entered = True
        return self

    def request(self, request):
        assert self.entered and not self.close_calls
        self.requests.append(copy.deepcopy(request))
        if self.failure is not None:
            raise self.failure
        extra = {}
        disk = "--prefix-cache" in self.command
        restored = 0
        if request["reset"]:
            assert "truncate" not in request
            self.cache, self.checkpoints = [], set()
            if disk:
                tokens = request["tokens"]
                restored = max([len(k) for k in self.factory.disk
                                if len(k) < len(tokens) and tuple(tokens[:len(k)]) == k] + [0])
                self.cache = list(tokens[:restored])
                if restored:
                    self.checkpoints.add(restored)
        elif "truncate" in request:
            n = request["truncate"]
            assert n <= len(self.cache)
            best = max([p for p in self.checkpoints if p <= n] + [0])
            extra.update(truncated_from=len(self.cache), rewind_checkpoint=best, replayed_tokens=n - best,
                         replay_seconds=0.0625, rewind_source="memory" if best else "start")
            self.cache = self.cache[:n]
            self.checkpoints = set(p for p in self.checkpoints if p <= n)
        before = len(self.cache) - restored
        taken, written = 0, 0
        for p in request.get("checkpoint_at", []):
            assert before < p <= before + len(request["tokens"])
            if p < len(self.cache):
                continue
            self.checkpoints.add(p)
            taken += 1
            if disk and p > len(self.cache) and tuple(request["tokens"][:p - before]) not in self.factory.disk:
                self.factory.disk[tuple(self.cache + request["tokens"][len(self.cache) - before:p - before])] = True
                written += 1
        self.cache.extend(request["tokens"][restored:])
        self.checkpoints.add(len(self.cache))
        generated = []
        if request["max_tokens"]:
            generated = [ord(char) for char in self.factory.outputs.pop(0) + EOS]
            generated = generated[:request["max_tokens"]]
        self.cache.extend(generated)
        self.checkpoints.add(len(self.cache))
        if disk:
            extra.update(restored_prefix_tokens=restored, restore_seconds=0.03125,
                         snapshots_written=written, snapshot_bytes=written * 4096,
                         snapshot_seconds=0.0, snapshot_failures=0)
        if request.get("checkpoint_at"):
            extra["checkpoints_taken"] = taken
        response = dict(id=request["id"], tokens=generated,
                        cache_before=before, cache_after=len(self.cache),
                        prefill_tokens=len(request["tokens"]),
                        decode_forward_tokens=len(generated),
                        reused_prefix_tokens=before,
                        eos=bool(generated and generated[-1] == ord(EOS)),
                        prefill_seconds=0.125, decode_seconds=0.25 if generated else 0.0,
                        kv_allocated_bytes=8192 * 8, kv_valid_bytes=len(self.cache) * 8, **extra)
        if self.mutate is not None:
            self.mutate(response)
        return response

    def close(self, graceful=True):
        self.close_calls.append(graceful)


class FakeFactory:
    def __init__(self, outputs):
        self.outputs, self.workers = list(outputs), []
        self.disk = {}

    def __call__(self, command, timeout):
        worker = FakeNativeSession(self, command, timeout)
        self.workers.append(worker)
        return worker


class AdapterTests(unittest.TestCase):
    def make_backend(self, outputs=("done",), context=8192, maximum=512, prefix_cache=False):
        tokenizer = CharacterTokenizer()
        factory = FakeFactory(outputs)
        command = ["not-a-real-executable"] + (["--prefix-cache", "target/unused"] if prefix_cache else [])
        backend = adapter.Backend(tokenizer, command, context,
                                  [ord(EOS)], 256, maximum=maximum, timeout=3,
                                  worker_factory=factory, prefix_cache=prefix_cache)
        self.addCleanup(backend.close)
        return backend, tokenizer, factory

    def payload(self, **options):
        result = dict(model="flint", messages=[dict(role="user", content="inspect")],
                      tools=[copy.deepcopy(TOOL)], max_tokens=512)
        result.update(options)
        return result

    def followup(self, payload, result):
        echoed = copy.deepcopy(result["choices"][0]["message"])
        echoed.pop("reasoning_content", None)
        call_id = echoed["tool_calls"][0]["id"]
        return dict(payload, messages=payload["messages"] + [
            echoed, dict(role="tool", tool_call_id=call_id, content="file contents")])

    def test_tool_turn_reuses_worker_and_exact_raw_reasoning_and_eos_prefix(self):
        backend, tokenizer, factory = self.make_backend([RAW_CALL, "finished"])
        first_payload = self.payload()
        first = backend.complete(first_payload)
        cached = list(backend.cached_ids)
        self.assertEqual(cached[-1], ord(EOS))
        self.assertEqual(first["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(first["choices"][0]["message"]["tool_calls"][0]
                                   ["function"]["arguments"]), {"path": "src/main.py"})
        second_payload = self.followup(first_payload, first)
        plan = backend.prepare(second_payload)
        self.assertFalse(plan["reset"])
        self.assertEqual(plan["reused"], len(cached))
        self.assertEqual(plan["ids"][:len(cached)], cached)
        self.assertEqual(tokenizer.renders[-1][1],
                         dict(role="assistant", content=RAW_CALL + EOS))
        result = backend.execute(plan)
        self.assertEqual(len(factory.workers), 1)
        worker = factory.workers[0]
        self.assertEqual([r["reset"] for r in worker.requests], [True, False])
        self.assertEqual(worker.requests[1]["tokens"], plan["suffix"])
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], len(cached))
        self.assertEqual(result["choices"][0]["message"]["content"], "finished")
        self.assertEqual(worker.cache, backend.cached_ids)
        self.assertEqual(result["nanoattention"]["kv_valid_bytes"], len(worker.cache) * 8)
        backend.close()
        self.assertEqual(worker.close_calls, [True])

    def system_payload(self, user="inspect", tools=None, system="You are a careful coder."):
        return self.payload(messages=[dict(role="system", content=system),
                                      dict(role="user", content=user)],
                            tools=[copy.deepcopy(TOOL)] if tools is None else tools)

    def test_changed_earlier_history_truncates_at_exact_common_token_prefix(self):
        backend, tokenizer, factory = self.make_backend([RAW_CALL, "new history"])
        payload = self.payload()
        first = backend.complete(payload)
        cached = list(backend.cached_ids)
        changed = self.followup(payload, first)
        changed["messages"][0] = dict(role="user", content="different instruction")
        plan = backend.prepare(changed)
        self.assertEqual(plan["action"], "truncate")
        self.assertFalse(plan["reset"])
        common = next(i for i, (a, b) in enumerate(zip(plan["ids"], cached)) if a != b)
        self.assertEqual(plan["reused"], common)
        self.assertEqual(plan["ids"][:common], cached[:common])
        self.assertNotEqual(plan["ids"][common], cached[common])
        self.assertEqual(plan["suffix"], plan["ids"][common:])
        self.assertIn("tool_calls", tokenizer.renders[-1][1])
        result = backend.execute(plan)
        worker = factory.workers[0]
        self.assertEqual(len(factory.workers), 1)
        request = worker.requests[-1]
        self.assertEqual((request["reset"], request["truncate"]), (False, common))
        self.assertEqual(request["tokens"], plan["suffix"])
        self.assertEqual(worker.cache, backend.cached_ids)
        meta = result["nanoattention"]
        self.assertFalse(meta["cache_reset"])
        self.assertEqual(meta["cache_action"], "truncate")
        self.assertEqual(meta["truncated_from"], len(cached))
        self.assertEqual(meta["reused_prefix_tokens"], common)
        self.assertEqual(meta["rewind_checkpoint"] + meta["replayed_tokens"], common)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"],
                         common - meta["replayed_tokens"])

    def test_retried_request_rewinds_to_last_prompt_token(self):
        backend, _, factory = self.make_backend(["first", "again"])
        payload = self.payload()
        backend.complete(payload)
        plan = backend.prepare(payload)
        self.assertEqual(plan["action"], "truncate")
        self.assertEqual(plan["reused"], len(plan["ids"]) - 1)
        self.assertEqual(plan["suffix"], plan["ids"][-1:])
        result = backend.execute(plan)
        self.assertEqual(result["choices"][0]["message"]["content"], "again")
        self.assertEqual(factory.workers[0].cache, backend.cached_ids)

    def test_first_request_marks_exact_system_and_tool_prefix(self):
        backend, tokenizer, factory = self.make_backend(["one", "two"])
        payload = self.system_payload()
        first = backend.complete(payload)
        stable = tokenizer.apply_chat_template(payload["messages"][:1], tools=payload["tools"],
                                               tokenize=True, add_generation_prompt=False)
        request = factory.workers[0].requests[0]
        self.assertTrue(request["reset"])
        self.assertEqual(request["checkpoint_at"], [len(stable)])
        self.assertEqual(request["tokens"][:len(stable)], stable)
        self.assertEqual(first["nanoattention"]["stable_prefix_tokens"], len(stable))
        self.assertEqual(first["nanoattention"]["checkpoint_at"], [len(stable)])
        followup = dict(payload, messages=payload["messages"] + [
            dict(role="assistant", content="one"), dict(role="user", content="more")])
        backend.complete(followup)
        self.assertNotIn("checkpoint_at", factory.workers[0].requests[1])
        self.assertNotIn("truncate", factory.workers[0].requests[1])
        # A divergence after the stable prefix rewinds to a memory checkpoint.
        backend, _, factory = self.make_backend(["one", "two"])
        backend.complete(payload)
        result = backend.complete(self.system_payload(user="something else"))
        meta = result["nanoattention"]
        self.assertEqual(meta["cache_action"], "truncate")
        self.assertEqual(meta["rewind_checkpoint"], len(stable))
        self.assertEqual(meta["rewind_source"], "memory")
        self.assertEqual(meta["replayed_tokens"], meta["reused_prefix_tokens"] - len(stable))

    def test_prefix_cache_restores_stable_prefix_in_a_new_worker(self):
        backend, tokenizer, factory = self.make_backend(["one", "two", "three"], prefix_cache=True)
        payload = self.system_payload()
        first = backend.complete(payload)
        stable = len(tokenizer.apply_chat_template(payload["messages"][:1], tools=payload["tools"],
                                                   tokenize=True, add_generation_prompt=False))
        self.assertEqual(first["nanoattention"]["snapshots_written"], 1)
        self.assertEqual(first["nanoattention"]["restored_prefix_tokens"], 0)
        backend.abort()  # e.g. a failed request: the next one owns a fresh worker
        result = backend.complete(self.system_payload(user="next task"))
        self.assertEqual(len(factory.workers), 2)
        request = factory.workers[1].requests[0]
        self.assertTrue(request["reset"])
        self.assertEqual(request["checkpoint_at"], [stable])
        meta = result["nanoattention"]
        self.assertEqual((meta["cache_action"], meta["restored_prefix_tokens"]), ("reset", stable))
        self.assertEqual(meta["snapshots_written"], 0)
        self.assertEqual(result["usage"]["prompt_tokens_details"]["cached_tokens"], stable)
        self.assertEqual(factory.workers[1].cache, backend.cached_ids)
        # A change inside the stable prefix resets (a disk snapshot may match)
        # instead of rewinding into it; without a prefix cache it rewinds.
        changed = self.system_payload(user="next task", tools=[])
        self.assertEqual(backend.prepare(changed)["action"], "reset")
        plain, _, _ = self.make_backend(["one"])
        plain.complete(payload)
        self.assertEqual(plain.prepare(changed)["action"], "truncate")

    def test_command_passes_prefix_cache_only_when_configured(self):
        import argparse
        args = argparse.Namespace(binary=Path("zig-out/bin/flint"), checkpoint=Path("model"), context=8192,
                                  kv_bf16=False, speculative_tokens=0, prefix_cache=None, prefix_cache_bytes=None)
        self.assertNotIn("--prefix-cache", adapter.native_command(args))
        args.prefix_cache, args.prefix_cache_bytes = Path("target/prefix-cache"), 1 << 30
        command = adapter.native_command(args)
        index = command.index("--prefix-cache")
        self.assertEqual(Path(command[index + 1]), Path("target/prefix-cache").resolve())
        self.assertEqual(command[command.index("--prefix-cache-bytes") + 1], str(1 << 30))

    def test_json_string_tool_arguments_normalize_without_mutating_client_history(self):
        backend, tokenizer, factory = self.make_backend()
        payload = self.payload(messages=[dict(role="user", content="read"),
            dict(role="assistant", content=None, tool_calls=[dict(
                id="call_test", type="function", function=dict(
                    name="read_file", arguments='{"path":"a.py"}'))]),
            dict(role="tool", tool_call_id="call_test", content="ok")])
        original = copy.deepcopy(payload)
        plan = backend.prepare(payload)
        arguments = plan["messages"][1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments, {"path": "a.py"})
        self.assertEqual(tokenizer.renders[-1][1]["tool_calls"][0]["function"]["arguments"], arguments)
        self.assertEqual(payload, original)
        self.assertEqual(factory.workers, [])

    def test_invalid_budgets_context_and_options_never_mutate_native_worker(self):
        backend, _, factory = self.make_backend(["seed"])
        backend.complete(self.payload())
        worker = factory.workers[0]
        cached, requests = list(backend.cached_ids), copy.deepcopy(worker.requests)
        bad_options = [dict(max_tokens=0), dict(max_tokens=-1), dict(max_tokens=True),
                       dict(max_tokens=1.5), dict(max_tokens=2**32),
                       dict(max_tokens=2, max_completion_tokens=3), dict(model="other"),
                       dict(temperature=0.1), dict(temperature=float("nan")),
                       dict(top_p=0.5), dict(stream="yes"), dict(n=2),
                       dict(stop=["end"]), dict(response_format={"type": "json_object"}),
                       dict(tools=[TOOL, TOOL]), dict(tool_choice="required", tools=[]),
                       dict(messages=[dict(role="assistant", content="not final")]),
                       dict(messages=[dict(role="user", content="x" * 8192)])]
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(adapter.AdapterError):
                backend.complete(self.payload(**options))
            self.assertEqual(worker.requests, requests)
            self.assertEqual(backend.cached_ids, cached)
            self.assertIs(backend.worker, worker)
            self.assertFalse(worker.close_calls)
        self.assertEqual(len(factory.workers), 1)
        self.assertEqual(backend.prepare(self.payload(max_tokens=999))["budget"], 512)

    def test_chunked_prefill_reserves_output_before_any_native_request(self):
        backend, _, factory = self.make_backend(["ok"])
        payload = self.payload(max_tokens=8)
        plan = backend.prepare(payload)
        with patch.object(adapter, "NATIVE_CHUNK", 17):
            result = backend.execute(plan)
        worker = factory.workers[0]
        requests = worker.requests
        self.assertGreater(len(requests), 1)
        self.assertTrue(all(len(r["tokens"]) <= 17 for r in requests))
        self.assertEqual([r["max_tokens"] for r in requests], [0] * (len(requests)-1) + [8])
        self.assertEqual([r["reset"] for r in requests], [True] + [False] * (len(requests)-1))
        self.assertEqual([t for r in requests for t in r["tokens"]], plan["ids"])
        self.assertEqual([r["id"] for r in requests], list(range(1, len(requests)+1)))
        self.assertTrue(all(r["return_logits"] is False for r in requests))
        self.assertEqual(result["nanoattention"]["prefill_seconds"], len(requests) * 0.125)
        self.assertEqual(result["nanoattention"]["decode_seconds"], 0.25)
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        backend2, _, factory2 = self.make_backend(context=len(plan["ids"]) + 7)
        with patch.object(adapter, "NATIVE_CHUNK", 17), self.assertRaises(adapter.AdapterError):
            backend2.complete(payload)
        self.assertEqual(factory2.workers, [])

    def test_malformed_rewind_and_restore_accounting_aborts_owned_worker(self):
        bad_responses = [
            dict(truncated_from=1), dict(rewind_checkpoint=10**6), dict(replayed_tokens=0),
            dict(rewind_source="guess"), dict(replay_seconds=float("inf")),
            dict(restored_prefix_tokens=1), dict(checkpoints_taken=0)]
        for bad in bad_responses:
            with self.subTest(bad=bad):
                backend, _, factory = self.make_backend(["seed", "next"])
                backend.complete(self.system_payload())
                worker = factory.workers[0]
                worker.mutate = lambda response, bad=bad: response.update(bad)
                with self.assertRaises(adapter.AdapterError) as caught:
                    # Diverges inside the stable prefix: truncate plus checkpoint_at.
                    backend.complete(self.system_payload(system="You are a terse coder."))
                self.assertEqual(caught.exception.status, 502)
                self.assertEqual(worker.close_calls, [False])
                self.assertEqual(backend.cached_ids, [])
                self.assertTrue(backend.needs_reset)
        backend, _, factory = self.make_backend(["seed", "next"], prefix_cache=True)
        backend.complete(self.payload())
        factory.workers[0].mutate = lambda response: response.pop("snapshots_written")
        with self.assertRaises(adapter.AdapterError):
            backend.complete(self.payload(messages=[dict(role="user", content="another")]))

    def test_malformed_native_accounting_aborts_owned_worker(self):
        bad_responses = [
            dict(cache_after=-1), dict(cache_before=True), dict(prefill_tokens=0),
            dict(reused_prefix_tokens=-1), dict(decode_forward_tokens=0),
            dict(id=-1), dict(eos=False), dict(tokens=[256]), dict(tokens=[True]),
            dict(prefill_seconds=-1), dict(decode_seconds=float("nan")),
            dict(kv_allocated_bytes=-1), dict(kv_valid_bytes=2**30)]
        for bad in bad_responses:
            with self.subTest(bad=bad):
                backend, _, factory = self.make_backend(["seed", "next"])
                backend.complete(self.payload())
                worker = factory.workers[0]
                worker.mutate = lambda response, bad=bad: response.update(bad)
                with self.assertRaises(adapter.AdapterError) as caught:
                    backend.complete(self.payload(messages=[dict(role="user", content="another")]))
                self.assertEqual(caught.exception.status, 502)
                self.assertEqual(worker.close_calls, [False])
                self.assertIsNone(backend.worker)
                self.assertEqual(backend.cached_ids, [])
                self.assertTrue(backend.needs_reset)
                self.assertFalse(backend.active)

    def test_worker_failure_aborts_and_next_completion_owns_a_fresh_worker(self):
        backend, _, factory = self.make_backend(["seed", "recovered"])
        backend.complete(self.payload())
        worker = factory.workers[0]
        worker.failure = RuntimeError("synthetic worker failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
            backend.complete(self.payload())
        self.assertEqual(worker.close_calls, [False])
        self.assertIsNone(backend.worker)
        self.assertTrue(backend.needs_reset)
        result = backend.complete(self.payload())
        self.assertEqual(len(factory.workers), 2)
        self.assertTrue(factory.workers[1].requests[0]["reset"])
        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")

    def test_sse_reconstructs_tool_calls_reasoning_usage_and_finish(self):
        backend, _, _ = self.make_backend([RAW_CALL])
        result = backend.complete(self.payload())
        events = list(adapter.sse_events(result))
        rebuilt, calls, finish = {}, [], None
        for event in events:
            self.assertEqual(event["object"], "chat.completion.chunk")
            self.assertEqual(event["id"], result["id"])
            for choice in event["choices"]:
                for key, value in choice["delta"].items():
                    if key == "tool_calls":
                        calls.extend(dict((k, v) for k, v in call.items() if k != "index")
                                     for call in value)
                    else:
                        rebuilt[key] = rebuilt.get(key, "") + value
                if choice["finish_reason"] is not None:
                    finish = choice["finish_reason"]
        expected = result["choices"][0]["message"]
        self.assertEqual(rebuilt["role"], expected["role"])
        self.assertEqual(rebuilt["reasoning_content"], expected["reasoning_content"])
        self.assertEqual(rebuilt.get("content", ""), expected["content"])
        self.assertEqual(calls, expected["tool_calls"])
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(events[-1]["usage"], result["usage"])
        self.assertEqual(events[-1]["nanoattention"], result["nanoattention"])
        self.assertFalse(any("usage" in e for e in adapter.sse_events(result, False)))

    def test_sse_long_text_and_length_finish_are_not_truncated_or_executable(self):
        backend, _, _ = self.make_backend(["x" * 5000], context=12000, maximum=6000)
        result = backend.complete(self.payload(max_tokens=5000, tools=[]))
        self.assertEqual(result["choices"][0]["finish_reason"], "length")
        events = list(adapter.sse_events(result))
        text = "".join(c["delta"].get("content", "") for event in events for c in event["choices"])
        self.assertEqual(text, "x" * 5000)
        content_events = [e for e in events if e["choices"] and "content" in e["choices"][0]["delta"]]
        self.assertEqual(len(content_events), 2)
        self.assertEqual(events[-2]["choices"][0]["finish_reason"], "length")
        self.assertEqual(events[-1]["usage"]["completion_tokens"], 5000)

    def test_loopback_http_auth_models_and_streamed_completion_cleanup(self):
        backend, _, factory = self.make_backend([RAW_CALL])
        server = adapter.AdapterServer(("127.0.0.1", 0), backend, "test-key")
        thread = threading.Thread(target=server.serve_forever,
                                  kwargs=dict(poll_interval=0.02), daemon=True)
        started = False
        def request(method, path, payload=None, authorized=True):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                headers = {"Authorization": "Bearer test-key"} if authorized else {}
                body = None if payload is None else json.dumps(payload)
                if body is not None:
                    headers["Content-Type"] = "application/json"
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                return response.status, response.getheader("Content-Type"), response.read().decode()
            finally:
                connection.close()
        try:
            thread.start()
            started = True
            status, _, _ = request("GET", "/v1/models", authorized=False)
            self.assertEqual(status, 401)
            status, _, body = request("GET", "/v1/models")
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["data"][0]["id"], "flint")
            status, _, body = request("GET", "/health", authorized=False)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["context_length"], backend.context)
            status, _, _ = request("POST", "/v1/chat/completions", self.payload(), authorized=False)
            self.assertEqual(status, 401)
            self.assertEqual(factory.workers, [])
            status, _, _ = request("POST", "/v1/chat/completions", self.payload(temperature=1))
            self.assertEqual(status, 400)
            self.assertEqual(factory.workers, [])
            status, content_type, body = request("POST", "/v1/chat/completions",
                self.payload(stream=True, stream_options=dict(include_usage=True)))
            self.assertEqual(status, 200)
            self.assertEqual(content_type, "text/event-stream")
            data = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
            self.assertEqual(data[-1], "[DONE]")
            events = [json.loads(line) for line in data[:-1]]
            self.assertFalse(any("error" in event for event in events))
            calls = [call for event in events for choice in event["choices"]
                     for call in choice["delta"].get("tool_calls", [])]
            self.assertEqual(calls[0]["function"]["name"], "read_file")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "src/main.py"})
            self.assertEqual(events[-2]["choices"][0]["finish_reason"], "tool_calls")
            self.assertEqual(events[-1]["usage"]["completion_tokens"], len(RAW_CALL) + 1)
        finally:
            if started:
                server.shutdown()
            server.server_close()
            if started:
                thread.join(timeout=5)
            backend.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(factory.workers[0].close_calls, [True])


if __name__ == "__main__":
    unittest.main()
