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
        assert add_generation_prompt is True
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
        text += "<assistant>"
        return self.encode(text) if tokenize else text


class FakeNativeSession:
    def __init__(self, factory, command, timeout):
        self.factory = factory
        self.command, self.timeout = command, timeout
        self.requests, self.close_calls, self.cache = [], [], []
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
        if request["reset"]:
            self.cache = []
        before = len(self.cache)
        generated = []
        if request["max_tokens"]:
            generated = [ord(char) for char in self.factory.outputs.pop(0) + EOS]
            generated = generated[:request["max_tokens"]]
        self.cache.extend(request["tokens"] + generated)
        response = dict(id=request["id"], tokens=generated,
                        cache_before=before, cache_after=len(self.cache),
                        prefill_tokens=len(request["tokens"]),
                        decode_forward_tokens=len(generated),
                        reused_prefix_tokens=before,
                        eos=bool(generated and generated[-1] == ord(EOS)),
                        prefill_seconds=0.125, decode_seconds=0.25 if generated else 0.0,
                        kv_allocated_bytes=8192 * 8, kv_valid_bytes=len(self.cache) * 8)
        if self.mutate is not None:
            self.mutate(response)
        return response

    def close(self, graceful=True):
        self.close_calls.append(graceful)


class FakeFactory:
    def __init__(self, outputs):
        self.outputs, self.workers = list(outputs), []

    def __call__(self, command, timeout):
        worker = FakeNativeSession(self, command, timeout)
        self.workers.append(worker)
        return worker


class AdapterTests(unittest.TestCase):
    def make_backend(self, outputs=("done",), context=8192, maximum=512):
        tokenizer = CharacterTokenizer()
        factory = FakeFactory(outputs)
        backend = adapter.Backend(tokenizer, ["not-a-real-executable"], context,
                                  [ord(EOS)], 256, maximum=maximum, timeout=3,
                                  worker_factory=factory)
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

    def test_changed_earlier_history_resets_instead_of_reusing_stale_prefix(self):
        backend, tokenizer, factory = self.make_backend([RAW_CALL, "new history"])
        payload = self.payload()
        first = backend.complete(payload)
        changed = self.followup(payload, first)
        changed["messages"][0] = dict(role="user", content="different instruction")
        plan = backend.prepare(changed)
        self.assertTrue(plan["reset"])
        self.assertEqual(plan["reused"], 0)
        self.assertEqual(plan["suffix"], plan["ids"])
        self.assertIn("tool_calls", tokenizer.renders[-1][1])
        result = backend.execute(plan)
        self.assertTrue(factory.workers[0].requests[-1]["reset"])
        self.assertTrue(result["nanoattention"]["cache_reset"])
        self.assertEqual(factory.workers[0].cache, backend.cached_ids)

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


class InklingVisionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(TOOLS))
        try:
            spec = importlib.util.spec_from_file_location(
                "inkling_adapter_under_test", TOOLS / "inkling_adapter.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            cls.inkling = mod
        finally:
            sys.path.pop(0)

    def make_inkling_backend(self, outputs=("ok",), engine="native", max_image_tokens=1024):
        IMG_TOK = self.inkling.IMAGE_TOKEN_ID
        EOS_ID = 200006

        class InklingFakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                assert add_special_tokens is False
                ids = []
                i = 0
                marker = "<|unused_200054|>"
                eos_tok = "<|content_model_end_sampling|>"
                while i < len(text):
                    if text.startswith(marker, i):
                        ids.append(IMG_TOK)
                        i += len(marker)
                    elif text.startswith(eos_tok, i):
                        ids.append(EOS_ID)
                        i += len(eos_tok)
                    else:
                        ids.append(ord(text[i]))
                        i += 1
                return ids

            def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                out = []
                for t in ids:
                    if t == IMG_TOK:
                        out.append("<|unused_200054|>")
                    elif t == EOS_ID:
                        out.append("<|content_model_end_sampling|>")
                    else:
                        out.append(chr(t))
                return "".join(out)

            def apply_chat_template(self, messages, tools=None, tokenize=False,
                                    add_generation_prompt=True, reasoning_effort="low"):
                assert tokenize is False
                parts = []
                for m in messages:
                    role = m["role"]
                    content = m["content"]
                    if isinstance(content, list):
                        for p in content:
                            if p["type"] == "image_url":
                                parts.append(f"<|message_{role}|><|content_image|><|unused_200054|><|end_message|>")
                            else:
                                parts.append(f"<|message_{role}|><|content_text|>{p['text']}<|end_message|>")
                    elif role == "assistant":
                        parts.append(f"<|message_model|><|content_text|>{content}<|end_message|><|content_model_end_sampling|>")
                    else:
                        parts.append(f"<|message_{role}|><|content_text|>{content}<|end_message|>")
                return "".join(parts)

        class VisionFakeWorker:
            def __init__(self, outs):
                self.outs = list(outs)
                self.requests = []
                self.cache = []
                self.closed = False

            def __enter__(self):
                return self

            def request(self, req):
                self.requests.append(copy.deepcopy(req))
                if req["reset"]:
                    self.cache = []
                before = len(self.cache)
                tokens = list(req["tokens"])
                images = req.get("images") or []
                counts = None
                if images:
                    # Simulate 4 visual tokens per image (2 for low detail budget <= 256)
                    per_img = 2 if req.get("max_image_tokens", 1024) <= 256 else 4
                    counts = [per_img] * len(images)
                    expanded = []
                    for t in tokens:
                        if t == IMG_TOK:
                            expanded.extend([IMG_TOK] * per_img)
                        else:
                            expanded.append(t)
                    tokens = expanded
                if req["id"] == 0:
                    generated = [ord("w")]
                elif req["max_tokens"]:
                    reply = "<|message_model|><|content_text|>" + self.outs.pop(0) + "<|end_message|>"
                    generated = tok.encode(reply) + [EOS_ID]
                    generated = generated[:req["max_tokens"]]
                else:
                    generated = []
                self.cache.extend(tokens + generated)
                return dict(
                    id=req["id"],
                    tokens=generated,
                    cache_before=before,
                    cache_after=len(self.cache),
                    prefill_tokens=len(tokens),
                    decode_forward_tokens=len(generated),
                    reused_prefix_tokens=before,
                    eos=bool(generated and generated[-1] == EOS_ID),
                    prefill_seconds=0.1,
                    decode_seconds=0.1 if generated else 0.0,
                    kv_allocated_bytes=8192 * 8,
                    kv_valid_bytes=len(self.cache) * 8,
                    image_token_counts=counts,
                )

            def close(self, graceful=True):
                self.closed = True

        tok = InklingFakeTokenizer()
        workers = []

        def factory(command, timeout):
            w = VisionFakeWorker(outputs)
            workers.append(w)
            return w

        backend = self.inkling.InklingBackend(
            tok, ["fake-nano"], 4096, [EOS_ID], 201024, 512, 5,
            worker_factory=factory, max_image_tokens=max_image_tokens, engine=engine,
        )
        self.addCleanup(backend.close)
        return backend, workers

    def test_multimodal_validation_accepts_openai_parts_and_rejects_bad_inputs(self):
        data_uri = "data:image/png;base64,iVBORw0KGgo="
        msgs = adapter.normalize_messages([
            dict(role="user", content=[
                dict(type="text", text="what is this?"),
                dict(type="image_url", image_url=dict(url=data_uri, detail="low")),
                dict(type="input_image", image_url="https://example.com/a.png"),
                dict(type="image", image="file:///tmp/x.jpg"),
            ]),
        ], allow_images=True)
        self.assertEqual(len(msgs[0]["content"]), 4)
        self.assertEqual(msgs[0]["content"][1]["image_url"]["detail"], "low")
        self.assertEqual(msgs[0]["content"][2]["image_url"]["url"], "https://example.com/a.png")
        self.assertEqual(msgs[0]["content"][3]["image_url"]["url"], "file:///tmp/x.jpg")

        # Tool messages may also include images (matching chat_template.jinja)
        tool_msgs = adapter.normalize_messages([
            dict(role="user", content="snap"),
            dict(role="tool", tool_call_id="call_1", content=[
                dict(type="image_url", image_url=dict(url=data_uri)),
            ]),
        ], allow_images=True)
        self.assertEqual(tool_msgs[1]["content"][0]["type"], "image_url")

        # System/assistant images, non-base64 data URI, javascript scheme, and text-only Flint reject
        for bad in (
            [dict(role="system", content=[dict(type="image_url", image_url=dict(url=data_uri))]),
             dict(role="user", content="hi")],
            [dict(role="user", content=[dict(type="image_url", image_url=dict(url="javascript:alert(1)"))])],
            [dict(role="user", content=[dict(type="image_url", image_url=dict(url="data:image/png,raw"))])],
            [dict(role="user", content=[dict(type="image_url", image_url=dict(url=data_uri, detail="ultra"))])],
        ):
            with self.assertRaises(adapter.AdapterError):
                adapter.normalize_messages(bad, allow_images=True)
        with self.assertRaises(adapter.AdapterError):
            adapter.normalize_messages([dict(role="user", content=[dict(type="image_url", image_url=dict(url=data_uri))])],
                                       allow_images=False)

    def test_inkling_adapter_image_turn_kv_reuse_and_invalidation(self):
        img_a = "data:image/png;base64,AAAA"
        img_b = "data:image/png;base64,BBBB"
        backend, workers = self.make_inkling_backend(outputs=["red pixel", "still red", "blue pixel"])

        # Turn 1: image A with detail=low -> max_image_tokens=256, 2 visual tokens expanded
        p1 = dict(model="flint", max_tokens=128, messages=[
            dict(role="user", content=[
                dict(type="image_url", image_url=dict(url=img_a, detail="low")),
                dict(type="text", text="color?"),
            ]),
        ])
        r1 = backend.complete(p1)
        self.assertEqual(r1["choices"][0]["message"]["content"], "red pixel")
        worker = workers[0]
        # requests[0] is warm-up, requests[1] is Turn 1
        req1 = worker.requests[1]
        self.assertEqual(req1["images"], [img_a])
        self.assertEqual(req1["max_image_tokens"], 256)
        self.assertEqual(r1["usage"]["prompt_tokens"], len(req1["tokens"]) + 1)  # 1 placeholder -> 2 visual tokens
        cached_after_t1 = len(backend.cached_ids)

        # Turn 2: follow-up text referencing same Turn 1 image A -> 100% KV cache hit, no images re-sent
        p2 = dict(model="flint", max_tokens=128, messages=p1["messages"] + [
            r1["choices"][0]["message"],
            dict(role="user", content="sure?"),
        ])
        r2 = backend.complete(p2)
        self.assertEqual(r2["choices"][0]["message"]["content"], "still red")
        req2 = worker.requests[2]
        self.assertFalse(req2["reset"])
        self.assertNotIn("images", req2)
        self.assertEqual(r2["usage"]["prompt_tokens_details"]["cached_tokens"], cached_after_t1)

        # Turn 3: swap image A for image B (same visual token count!) -> MUST invalidate KV cache
        p3 = dict(model="flint", max_tokens=128, messages=[
            dict(role="user", content=[
                dict(type="image_url", image_url=dict(url=img_b, detail="high")),
                dict(type="text", text="color?"),
            ]),
        ])
        r3 = backend.complete(p3)
        self.assertEqual(r3["choices"][0]["message"]["content"], "blue pixel")
        req3 = worker.requests[3]
        self.assertTrue(req3["reset"])
        self.assertEqual(req3["images"], [img_b])
        self.assertEqual(req3["max_image_tokens"], 1024)
        self.assertEqual(r3["usage"]["prompt_tokens_details"]["cached_tokens"], 0)

    def test_inkling_adapter_mlx_engine_rejects_images(self):
        backend, _ = self.make_inkling_backend(engine="mlx")
        with self.assertRaises(self.inkling.AdapterError) as ctx:
            backend.complete(dict(model="flint", max_tokens=32, messages=[
                dict(role="user", content=[
                    dict(type="image_url", image_url=dict(url="https://example.com/pic.png")),
                ]),
            ]))
        self.assertEqual(ctx.exception.status, 400)


if __name__ == "__main__":
    unittest.main()
