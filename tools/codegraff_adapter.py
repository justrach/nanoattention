#!/usr/bin/env python3
"""Loopback OpenAI chat adapter for the native Zig/Metal Flint session worker.

Transformers handles tokenization only; no MLX/PyTorch model is loaded.
SSE is buffered generation with heartbeat comments, not token-live streaming.
Run under guarded_run.py to own the inference lock and bound resources.
"""
import argparse
import copy
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import signal
import threading
import time
import uuid

from coding_multiturn import NativeSession
from flint_tool_protocol import ProtocolError, parse_completion

for _key, _value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                     'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false'}.items():
    os.environ[_key] = _value

MAX_BODY = 8 * 1024 * 1024
NATIVE_CHUNK = 65536  # At most 459 KiB of u32 IDs in the native 1 MiB JSONL frame.
MODEL_ID = 'flint'


class AdapterError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def positive(value, name):
    if type(value) is not int or not 0 < value <= 0xffffffff:
        raise AdapterError('{} must be a positive u32'.format(name))
    return value


def content_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, dict) and v.get('type') == 'text'
                                       and isinstance(v.get('text'), str) for v in value):
        return '\n'.join(v['text'] for v in value)
    raise AdapterError('only text message content is supported')


def normalize_messages(messages):
    if not isinstance(messages, list) or not 0 < len(messages) <= 4096:
        raise AdapterError('messages must be a nonempty bounded array')
    result = []
    for raw in messages:
        if not isinstance(raw, dict) or raw.get('role') not in ('system', 'developer', 'user', 'assistant', 'tool'):
            raise AdapterError('unsupported message role')
        message = dict(role=raw['role'], content=content_text(raw.get('content')))
        if 'reasoning_content' in raw:
            if not isinstance(raw['reasoning_content'], str):
                raise AdapterError('reasoning_content must be text')
            message['reasoning_content'] = raw['reasoning_content']
        if raw.get('tool_calls'):
            if message['role'] != 'assistant' or not isinstance(raw['tool_calls'], list):
                raise AdapterError('tool_calls require an assistant message')
            calls = []
            for call in raw['tool_calls']:
                if not isinstance(call, dict) or call.get('type') != 'function' or not isinstance(call.get('function'), dict):
                    raise AdapterError('unsupported tool call')
                function = call['function']
                arguments = function.get('arguments', {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (ValueError, TypeError):
                        raise AdapterError('tool arguments must be a JSON object')
                if not isinstance(arguments, dict) or not isinstance(function.get('name'), str) or not isinstance(call.get('id'), str):
                    raise AdapterError('invalid function call history')
                calls.append(dict(id=call['id'], type='function', function=dict(name=function['name'], arguments=arguments)))
            message['tool_calls'] = calls
        if message['role'] == 'tool':
            if not isinstance(raw.get('tool_call_id'), str):
                raise AdapterError('tool observations require tool_call_id')
            message['tool_call_id'] = raw['tool_call_id']
        result.append(message)
    if result[-1]['role'] not in ('user', 'tool'):
        raise AdapterError('last message must be a user message or tool observation')
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def assistant_signature(message):
    result = copy.deepcopy(message)
    result.pop('reasoning_content', None)
    return result


def validate_options(payload, maximum):
    if not isinstance(payload, dict) or payload.get('model', MODEL_ID) != MODEL_ID:
        raise AdapterError('unknown model; use flint')
    if type(payload.get('n', 1)) is not int or payload.get('n', 1) != 1 or type(payload.get('stream', False)) is not bool:
        raise AdapterError('only n=1 and boolean stream are supported')
    stream_options = payload.get('stream_options', {})
    if not isinstance(stream_options, dict) or type(stream_options.get('include_usage', False)) is not bool:
        raise AdapterError('invalid stream_options')
    if type(payload.get('parallel_tool_calls', True)) is not bool:
        raise AdapterError('parallel_tool_calls must be boolean')
    for name, default in (('temperature', 0), ('top_p', 1), ('presence_penalty', 0), ('frequency_penalty', 0)):
        value = payload.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value != default:
            raise AdapterError('native greedy inference requires {}={}'.format(name, default))
    if payload.get('stop') not in (None, [], '') or payload.get('logprobs') or payload.get('reasoning_effort'):
        raise AdapterError('stop, logprobs and reasoning_effort controls are not supported')
    if payload.get('response_format', {'type': 'text'}) != {'type': 'text'}:
        raise AdapterError('only text response_format is supported')
    budget = payload.get('max_completion_tokens', payload.get('max_tokens', maximum))
    positive(budget, 'max_tokens')
    if 'max_tokens' in payload and 'max_completion_tokens' in payload and payload['max_tokens'] != budget:
        raise AdapterError('conflicting token budgets')
    tools = payload.get('tools') or []
    if not isinstance(tools, list) or len(tools) > 256:
        raise AdapterError('tools must be a bounded array')
    names = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
            raise AdapterError('only function tools are supported')
        function = tool['function']
        if not isinstance(function.get('name'), str) or not isinstance(function.get('parameters', {}), dict):
            raise AdapterError('invalid function schema')
        names.append(function['name'])
    if len(names) != len(set(names)):
        raise AdapterError('duplicate function names')
    choice = payload.get('tool_choice', 'auto')
    if choice == 'none':
        tools = []
    elif choice not in ('auto', 'required'):
        if not isinstance(choice, dict) or choice.get('type') != 'function' or not isinstance(choice.get('function'), dict):
            raise AdapterError('unsupported tool_choice')
        tools = [tool for tool in tools if tool['function']['name'] == choice['function'].get('name')]
        if len(tools) != 1:
            raise AdapterError('forced function is not available')
    if choice == 'required' and not tools:
        raise AdapterError('required tool_choice has no tools')
    return normalize_messages(payload.get('messages')), tools, min(budget, maximum)


class Backend:
    def __init__(self, tokenizer, command, context, eos, vocab_size, maximum=4096,
                 timeout=900, worker_factory=NativeSession):
        self.tokenizer, self.command = tokenizer, command
        self.context, self.eos, self.vocab_size = context, set(eos), vocab_size
        self.maximum, self.timeout, self.worker_factory = maximum, timeout, worker_factory
        self.worker = None
        self.cached_ids, self.records = [], []
        self.needs_reset, self.closed, self.active = True, False, False
        self.epoch, self.sequence = 0, 0
        self.gate = threading.Semaphore(1)
        self.metrics, self.worker_starts, self.failures = [], 0, 0

    def prepare(self, payload):
        if self.closed:
            raise AdapterError('adapter is closed', 503)
        messages, tools, budget = validate_options(payload, self.maximum)
        rendered_messages = copy.deepcopy(messages)
        # Restore our exact prior native assistant bytes when the client echoes
        # the corresponding OpenAI message. Never fake a matching token prefix.
        for record in self.records:
            index = record['index']
            if index >= len(messages) or fingerprint(messages[:index]) != record['prefix']:
                continue
            current = messages[index]
            if assistant_signature(current) != record['signature']:
                continue
            if 'reasoning_content' in current and current['reasoning_content'] != record['reasoning']:
                continue
            rendered_messages[index] = dict(role='assistant', content=record['raw'])
        options = dict(tools=tools, add_generation_prompt=True)
        text = self.tokenizer.apply_chat_template(rendered_messages, tokenize=False, **options)
        ids = list(self.tokenizer.apply_chat_template(rendered_messages, tokenize=True,
                                                      return_dict=False, **options))
        if list(self.tokenizer.encode(text, add_special_tokens=False)) != ids:
            raise AdapterError('checkpoint template tokenization disagrees with its text', 500)
        if not ids or any(type(token) is not int or not 0 <= token < self.vocab_size for token in ids):
            raise AdapterError('invalid template token IDs', 500)
        if len(ids) + budget > self.context:
            raise AdapterError('prompt plus reserved committed output exceeds context')
        hit = (not self.needs_reset and len(ids) >= len(self.cached_ids)
               and ids[:len(self.cached_ids)] == self.cached_ids)
        reused = len(self.cached_ids) if hit else 0
        suffix = ids[reused:]
        if not suffix:
            raise AdapterError('request has no new template suffix')
        return dict(payload=payload, messages=messages, tools=tools, budget=budget,
                    ids=ids, suffix=suffix, reused=reused, reset=not hit, epoch=self.epoch)

    def _check_native(self, response, request, before):
        tokens = response.get('tokens') if isinstance(response, dict) else None
        if (not isinstance(tokens, list) or len(tokens) > request['max_tokens'] or
                any(type(t) is not int or not 0 <= t < self.vocab_size for t in tokens)):
            raise AdapterError('invalid native token response', 502)
        expected = dict(id=request['id'], cache_before=before,
                        cache_after=before+len(request['tokens'])+len(tokens),
                        prefill_tokens=len(request['tokens']), decode_forward_tokens=len(tokens),
                        reused_prefix_tokens=before)
        if any(type(response.get(k)) is not int or response[k] != v for k, v in expected.items()):
            raise AdapterError('native cache accounting mismatch', 502)
        if type(response.get('eos')) is not bool or response['eos'] != bool(tokens and tokens[-1] in self.eos):
            raise AdapterError('native EOS accounting mismatch', 502)
        if any(t in self.eos for t in tokens[:-1]):
            raise AdapterError('native generated after EOS', 502)
        if response['cache_after'] > self.context:
            raise AdapterError('native exceeded context', 502)
        for key in ('prefill_seconds', 'decode_seconds'):
            value = response.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise AdapterError('invalid native timing', 502)
        for key in ('kv_allocated_bytes', 'kv_valid_bytes'):
            if type(response.get(key)) is not int or response[key] < 0:
                raise AdapterError('invalid native KV accounting', 502)
        if response['kv_valid_bytes'] > response['kv_allocated_bytes']:
            raise AdapterError('invalid native KV validity', 502)
        return tokens

    def execute(self, plan):
        self.active = True
        try:
            if plan['epoch'] != self.epoch or self.closed:
                raise AdapterError('request cancelled', 499)
            if self.worker is None:
                self.worker = self.worker_factory(self.command, timeout=self.timeout).__enter__()
                self.worker_starts += 1
            prefill_seconds, decode_seconds = 0.0, 0.0
            before = plan['reused']
            suffix = plan['suffix']
            response = None
            for start in range(0, len(suffix), NATIVE_CHUNK):
                if plan['epoch'] != self.epoch or self.closed:
                    raise AdapterError('request cancelled', 499)
                chunk = suffix[start:start+NATIVE_CHUNK]
                final = start + len(chunk) == len(suffix)
                self.sequence += 1
                request = dict(id=self.sequence, tokens=chunk,
                               max_tokens=plan['budget'] if final else 0,
                               reset=plan['reset'] and start == 0, return_logits=False)
                response = self.worker.request(request)
                generated = self._check_native(response, request, before)
                before = response['cache_after']
                prefill_seconds += response['prefill_seconds']
                decode_seconds += response['decode_seconds']
            if plan['epoch'] != self.epoch or self.closed:
                raise AdapterError('request cancelled', 499)
            self.cached_ids = plan['ids'] + generated
            self.needs_reset = False
            raw = self.tokenizer.decode(generated, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
            visible_ids = generated[:-1] if generated and generated[-1] in self.eos else generated
            text = self.tokenizer.decode(visible_ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
            parsed = parse_completion(text, plan['tools'], stopped=response['eos'])
            if response['eos'] and plan['payload'].get('tool_choice') == 'required' and not parsed['tool_calls']:
                raise AdapterError('model did not produce the required function call', 502)
            if plan['payload'].get('parallel_tool_calls') is False and len(parsed['tool_calls']) > 1:
                raise AdapterError('model produced multiple calls when disabled', 502)
            request_id = 'chatcmpl-' + uuid.uuid4().hex
            for index, call in enumerate(parsed['tool_calls']):
                call['id'] = 'call_{}_{}'.format(request_id[9:], index)
            message = dict(role='assistant', content=parsed['content'])
            if parsed['reasoning_content']:
                message['reasoning_content'] = parsed['reasoning_content']
            if parsed['tool_calls']:
                message['tool_calls'] = parsed['tool_calls']
            normalized = normalize_messages(plan['messages'] + [message, dict(role='user', content='')])[-2]
            if list(self.tokenizer.encode(raw, add_special_tokens=False)) == generated:
                self.records.append(dict(index=len(plan['messages']), prefix=fingerprint(plan['messages']),
                                         signature=assistant_signature(normalized), raw=raw,
                                         reasoning=parsed['reasoning_content']))
                self.records = self.records[-256:]
            result = dict(id=request_id, object='chat.completion', created=int(time.time()), model=MODEL_ID,
                        choices=[dict(index=0, message=message, finish_reason=parsed['finish_reason'])],
                        usage=dict(prompt_tokens=len(plan['ids']), completion_tokens=len(generated),
                                   total_tokens=len(plan['ids'])+len(generated),
                                   prompt_tokens_details=dict(cached_tokens=plan['reused'])),
                        nanoattention=dict(cache_reset=plan['reset'], reused_prefix_tokens=plan['reused'],
                                           new_prefill_tokens=len(suffix), cache_after=response['cache_after'],
                                           prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
                                           kv_allocated_bytes=response['kv_allocated_bytes'],
                                           kv_valid_bytes=response['kv_valid_bytes'],
                                           effective_max_tokens=plan['budget'], sampling='greedy'))
            self.metrics.append(dict(result['nanoattention'], prompt_tokens=len(plan['ids']),
                                     completion_tokens=len(generated), finish_reason=parsed['finish_reason'],
                                     tool_call_count=len(parsed['tool_calls']), worker_starts=self.worker_starts))
            self.metrics = self.metrics[-256:]
            return result
        except BaseException:
            self.failures += 1
            self.abort()
            raise
        finally:
            self.active = False

    def complete(self, payload):
        if not self.gate.acquire(blocking=False):
            raise AdapterError('one native sequence is already running', 429)
        try:
            return self.execute(self.prepare(payload))
        finally:
            self.gate.release()

    def abort(self):
        self.epoch += 1
        self.needs_reset = True
        self.cached_ids = []
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.close(graceful=False)

    def close(self):
        self.closed = True
        worker, self.worker = self.worker, None
        self.needs_reset = True
        if worker is not None:
            worker.close(graceful=not self.active)


def sse_events(result, include_usage=True):
    base = {key: result[key] for key in ('id', 'created', 'model')}
    base['object'] = 'chat.completion.chunk'
    def chunk(delta, finish=None):
        return dict(base, choices=[dict(index=0, delta=delta, finish_reason=finish)])
    yield chunk(dict(role='assistant'))
    message = result['choices'][0]['message']
    for field in ('reasoning_content', 'content'):
        value = message.get(field, '')
        for start in range(0, len(value), 4096):
            yield chunk({field: value[start:start+4096]})
    for index, call in enumerate(message.get('tool_calls', [])):
        yield chunk(dict(tool_calls=[dict(call, index=index)]))
    yield chunk({}, result['choices'][0]['finish_reason'])
    if include_usage:
        yield dict(base, choices=[], usage=result['usage'], nanoattention=result['nanoattention'])


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, backend, api_key):
        self.backend, self.api_key = backend, api_key
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *unused):
        pass  # No prompt/argument/access logging by default.

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def json_response(self, status, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True

    def authenticated(self):
        return hmac.compare_digest(self.headers.get('Authorization', ''),
                                   'Bearer ' + self.server.api_key)

    def do_GET(self):
        if self.path == '/health':
            self.json_response(200, dict(status='listening', model=MODEL_ID,
                                         context_length=self.server.backend.context))
        elif self.path == '/v1/nanoattention/metrics' and self.authenticated():
            self.json_response(200, dict(requests=list(self.server.backend.metrics),
                               worker_starts=self.server.backend.worker_starts,
                               failures=self.server.backend.failures))
        elif self.path == '/v1/models' and self.authenticated():
            self.json_response(200, dict(object='list', data=[dict(id=MODEL_ID, object='model',
                               owned_by='nanoattention', context_length=self.server.backend.context)]))
        else:
            self.json_response(401 if self.path == '/v1/models' else 404,
                               dict(error=dict(message='unauthorized or unknown endpoint')))

    def do_POST(self):
        backend = self.server.backend
        acquired, started, worker = False, False, None
        try:
            if self.path != '/v1/chat/completions':
                raise AdapterError('unknown endpoint', 404)
            if not self.authenticated():
                raise AdapterError('unauthorized', 401)
            if self.headers.get('Transfer-Encoding') or len(self.headers.get_all('Content-Length', [])) != 1:
                raise AdapterError('one Content-Length is required')
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                raise AdapterError('invalid Content-Length')
            if not 0 < length <= MAX_BODY:
                raise AdapterError('request body exceeds limit', 413)
            data = self.rfile.read(length)
            if len(data) != length:
                raise AdapterError('incomplete request body')
            try:
                def invalid_constant(unused):
                    raise ValueError('invalid JSON constant')
                payload = json.loads(data, parse_constant=invalid_constant)
            except (ValueError, UnicodeError):
                raise AdapterError('invalid JSON request')
            if not backend.gate.acquire(blocking=False):
                raise AdapterError('one native sequence is already running', 429)
            acquired = True
            plan = backend.prepare(payload)
            if not payload.get('stream', False):
                result = backend.execute(plan)
                self.json_response(200, result)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            started = True
            result_queue = queue.Queue(maxsize=1)
            def run():
                try:
                    result_queue.put((True, backend.execute(plan)))
                except BaseException as error:
                    result_queue.put((False, error))
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            while True:
                try:
                    ok, result = result_queue.get(timeout=5)
                    break
                except queue.Empty:
                    self.wfile.write(b': native inference in progress\n\n')
                    self.wfile.flush()
            if not ok:
                message = str(result) if isinstance(result, AdapterError) else 'native inference or tool protocol failed'
                self.wfile.write(('data: ' + json.dumps(dict(error=dict(message=message))) + '\n\n').encode())
            else:
                include_usage = payload.get('stream_options', {}).get('include_usage', False)
                for event in sse_events(result, include_usage):
                    self.wfile.write(('data: ' + json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionError, TimeoutError, OSError):
            if acquired:
                backend.abort()
        except Exception as error:
            if started:
                backend.abort()
            else:
                status = error.status if isinstance(error, AdapterError) else 502
                message = str(error) if isinstance(error, AdapterError) else 'native inference or tool protocol failed'
                self.json_response(status, dict(error=dict(message=message, type='invalid_request_error' if status < 500 else 'server_error')))
        finally:
            if worker is not None and worker.is_alive():
                backend.abort()
                worker.join(timeout=10)
            if acquired:
                backend.gate.release()
            self.close_connection = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--binary', type=Path, default=Path(__file__).resolve().parents[1]/'zig-out/bin/flint')
    parser.add_argument('--port', type=int, default=8080, help='loopback only; 0 chooses a disposable port')
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--max-tokens', type=int, default=4096, help='server output cap, including committed EOS')
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    config = json.loads((args.checkpoint/'config.json').read_text())
    if not 0 < args.context <= config['max_position_embeddings'] or not 0 < args.max_tokens < args.context:
        parser.error('invalid context or output cap')
    if not 0 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('invalid port or timeout')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True,
                                              trust_remote_code=False, fix_mistral_regex=False)
    tokenizer.chat_template = (args.checkpoint/'chat_template.jinja').read_text()
    eos = config['eos_token_id']
    eos = eos if isinstance(eos, list) else [eos]
    command = [str(args.binary.resolve()), 'session', str(args.checkpoint.resolve()),
               '--context', str(args.context), '--mlx-attention', '1', '--staged-row-tile', '2']
    backend = Backend(tokenizer, command, args.context, eos, config['vocab_size'], args.max_tokens, args.timeout)
    server = AdapterServer(('127.0.0.1', args.port), backend, os.environ.get('NANOATTENTION_API_KEY', 'local'))
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *unused: stopped.set())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(json.dumps(dict(event='ready', host='127.0.0.1', port=server.server_port,
                          model=MODEL_ID, context_length=args.context)), flush=True)
    try:
        stopped.wait()
    finally:
        backend.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    main()
