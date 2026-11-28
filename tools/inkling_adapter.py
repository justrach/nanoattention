#!/usr/bin/env python3
"""Loopback OpenAI chat adapter for Inkling 2 Mini, on nanoattention's engine or on mlx-lm.

Same adapter contract as codegraff_adapter.py (one request at a time, KV
prefix reuse across turns, greedy, buffered SSE), which this file imports and
changes in four places:

  * the worker is `nano session` (--engine native, the Inkling knob set of
    tools/nano_inkling_tasks.py) or tools/inkling_mlx_session.py (--engine mlx,
    mlx-lm's own prefill and decode), both speaking the same JSONL protocol;
  * the model's turns are parsed with tools/inkling_tool_protocol.py;
  * a previous model turn that the client echoes back is spliced into the
    prompt as the exact token IDs the engine generated (the checkpoint's template
    would wrap raw text as a text message), so a follow-up turn finds its prefix
    in the KV cache instead of re-reading it;
  * each worker is warmed on a throwaway request before the first real one.

The route id stays `flint`: Graff's built-in `nanoattention` provider asks for
that model name and nothing else. The engine behind it is Inkling.

    python3 tools/inkling_adapter.py CHECKPOINT --engine native --binary zig-out/bin/nano
    python3 tools/inkling_adapter.py CHECKPOINT --engine mlx --mlx-python PY

`--record PATH` appends one JSON line per request (the IDs sent and generated,
reuse, seconds); the replay tool feeds those back to either engine.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
import uuid

import codegraff_adapter as base
from codegraff_adapter import AdapterError, AdapterServer, Backend
from coding_multiturn import NativeSession
import inkling_tool_protocol

base.parse_completion = inkling_tool_protocol.parse_completion  # Backend.execute's parser

EFFORT = "low"
CHUNK = 4096  # prefill rows a forward, for both engines (tools/nano_inkling_tasks.py)
WARM_TOKENS = 300
MAX_BODY = 64 * 1024 * 1024  # allow high-res base64 image_url payloads
IMAGE_TOKEN_ID = 200054
DEFAULT_MAX_IMAGE_TOKENS = 1024
LOW_DETAIL_MAX_IMAGE_TOKENS = 256
SENTINEL = "\x00recorded-{}\x00"
WRAPPER = "<|message_model|><|content_text|>{}<|end_message|><|content_model_end_sampling|>"


def extract_images(messages):
    images = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                spec = part["image_url"]
                url = spec["url"]
                detail = spec.get("detail", "auto")
                images.append(dict(
                    url=url,
                    detail=detail,
                    fingerprint=hashlib.sha256(url.encode("utf-8")).hexdigest(),
                ))
    return images


def expand_image_tokens(tokens, counts, token_id=IMAGE_TOKEN_ID):
    placeholders = sum(1 for t in tokens if t == token_id)
    if placeholders != len(counts):
        raise AdapterError("native image token count mismatch", 502)
    out = []
    img_idx = 0
    for t in tokens:
        if t == token_id:
            out.extend([token_id] * counts[img_idx])
            img_idx += 1
        else:
            out.append(t)
    return out


def native_command(
    binary,
    checkpoint,
    context,
    *,
    knobs=None,
    draft=None,
    loop_passes=None,
    loopcd=None,
    loopcd_omega=None,
    loopcd_weak_pass=None,
    lsa=False,
):
    sys_path = Path(__file__).resolve().parent
    repo_root = sys_path.parent
    drop = set(filter(None, os.environ.get("NANO_DROP_KNOBS", "").split(",")))  # experiments: knobs to leave out
    command = [str(binary), "session", str(checkpoint), "--context", str(context)]
    inkling_preset = repo_root / "src" / "mega" / "presets" / "inkling.json"
    if knobs is not None:
        command += ["--knobs", str(knobs)]
    elif inkling_preset.is_file() and not drop:
        command += ["--knobs", str(inkling_preset)]
    else:
        for knob in _knobs(sys_path):
            if knob.split("=")[0] not in drop:
                command += ["--knob", knob]
    command += ["--knob", "batch=" + str(CHUNK)]
    if drop and inkling_preset.is_file() and knobs is not None:
        for name in sorted(drop):
            command += ["--knob", f"{name}=false"]
    if lsa:
        command += ["--knob", "lsa=true", "--knob", "lsa_cross_loop_reuse=true"]
    if loop_passes is not None:
        command += ["--loop-passes", str(loop_passes)]
    if loopcd is not None:
        command += ["--loopcd", str(loopcd)]
    if loopcd_omega is not None:
        command += ["--loopcd-omega", str(loopcd_omega)]
    if loopcd_weak_pass is not None:
        command += ["--loopcd-weak-pass", str(loopcd_weak_pass)]
    if draft is not None:
        command += ["--draft", str(draft)]
    return command


def _knobs(tools_dir):
    import importlib.util
    spec = importlib.util.spec_from_file_location("nano_inkling_tasks", tools_dir / "nano_inkling_tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.KNOBS)


def mlx_command(python, checkpoint, context):
    return [str(python), "-B", str(Path(__file__).with_name("inkling_mlx_session.py")), str(checkpoint),
            "--chunk", str(CHUNK), "--context", str(context)]


class InklingBackend(Backend):
    allow_images = True

    def __init__(self, *args, record=None, max_image_tokens=DEFAULT_MAX_IMAGE_TOKENS, engine="native", **kwargs):
        super().__init__(*args, **kwargs)
        self.record = record
        self.max_image_tokens = max_image_tokens
        self.engine = engine
        self.cached_images = []  # [(start_pos, count, fingerprint, max_image_tokens), ...]

    def abort(self):
        self.cached_images = []
        super().abort()

    def close(self):
        self.cached_images = []
        super().close()

    def prepare(self, payload):
        if self.closed:
            raise AdapterError("adapter is closed", 503)
        messages, tools, budget = base.validate_options(payload, self.maximum, allow_images=True)
        images = extract_images(messages)
        if images and self.engine != "native":
            raise AdapterError("image input requires --engine native", 400)
        req_max_image_tokens = (
            min(self.max_image_tokens, LOW_DETAIL_MAX_IMAGE_TOKENS)
            if images and all(img["detail"] == "low" for img in images)
            else self.max_image_tokens
        )
        rendered = copy.deepcopy(messages)
        splices = {}
        for record in self.records:
            index = record["index"]
            if index >= len(messages) or base.fingerprint(messages[:index]) != record["prefix"]:
                continue
            current = messages[index]
            if base.assistant_signature(current) != record["signature"]:
                continue
            if "reasoning_content" in current and current["reasoning_content"] != record["reasoning"]:
                continue
            rendered[index] = dict(role="assistant", content=SENTINEL.format(index))
            splices[index] = record["raw"]
        options = dict(tools=tools, add_generation_prompt=True, reasoning_effort=EFFORT)
        text = self.tokenizer.apply_chat_template(rendered, tokenize=False, **options)
        for index, raw in splices.items():
            wrapper = WRAPPER.format(SENTINEL.format(index))
            if text.count(wrapper) != 1:
                raise AdapterError("recorded model turn did not render as expected", 500)
            text = text.replace(wrapper, raw)
        raw_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if not raw_ids or any(type(t) is not int or not 0 <= t < self.vocab_size for t in raw_ids):
            raise AdapterError("invalid template token IDs", 500)
        placeholder_count = sum(1 for t in raw_ids if t == IMAGE_TOKEN_ID)
        if placeholder_count != len(images):
            raise AdapterError("image placeholder count does not match image inputs", 400)

        # Expand leading images that match the currently cached KV prefix so
        # multi-turn follow-ups reuse cached image KV without re-encoding.
        expanded_ids = []
        matched_images = 0
        prefix_ok = not self.needs_reset
        img_cursor = 0
        for tok in raw_ids:
            if tok == IMAGE_TOKEN_ID:
                img = images[img_cursor]
                if prefix_ok and img_cursor < len(self.cached_images):
                    c_start, c_count, c_fp, c_max = self.cached_images[img_cursor]
                    if c_start == len(expanded_ids) and c_fp == img["fingerprint"] and c_max == req_max_image_tokens:
                        expanded_ids.extend([IMAGE_TOKEN_ID] * c_count)
                        matched_images += 1
                    else:
                        prefix_ok = False
                        expanded_ids.append(tok)
                else:
                    expanded_ids.append(tok)
                img_cursor += 1
            else:
                expanded_ids.append(tok)

        hit = (prefix_ok and matched_images == len(self.cached_images)
               and len(expanded_ids) >= len(self.cached_ids)
               and expanded_ids[:len(self.cached_ids)] == self.cached_ids)
        if hit:
            reused = len(self.cached_ids)
            ids = expanded_ids
            suffix = ids[reused:]
            suffix_images = images[matched_images:]
            prefix_images = list(self.cached_images)
        else:
            reused = 0
            ids = raw_ids
            suffix = raw_ids
            suffix_images = images
            prefix_images = []
        if len(ids) + budget > self.context:
            raise AdapterError("prompt plus reserved committed output exceeds context")
        if not suffix:
            raise AdapterError("request has no new template suffix")
        return dict(payload=payload, messages=messages, tools=tools, budget=budget, ids=ids, suffix=suffix,
                    reused=reused, reset=not hit, epoch=self.epoch, suffix_images=suffix_images,
                    prefix_images=prefix_images, max_image_tokens=req_max_image_tokens)

    def _warm(self):
        """A throwaway request on a fresh worker: kernels built, weights paged in."""
        ids = [(7 * i + 3) % 20000 + 1000 for i in range(WARM_TOKENS)]
        response = self.worker.request(dict(id=0, tokens=ids, max_tokens=4, reset=True, return_logits=False))
        if not isinstance(response, dict) or "tokens" not in response:
            raise AdapterError("warm-up failed", 502)

    def _execute_multimodal(self, plan):
        self.active = True
        try:
            if plan["epoch"] != self.epoch or self.closed:
                raise AdapterError("request cancelled", 499)
            prefill_seconds, decode_seconds = 0.0, 0.0
            before = plan["reused"]
            suffix = plan["suffix"]
            suffix_images = plan["suffix_images"]
            actual_ids = list(plan["ids"][:plan["reused"]])
            actual_images = list(plan["prefix_images"])
            img_cursor = 0
            response = None
            generated = []
            for start in range(0, len(suffix), base.NATIVE_CHUNK):
                if plan["epoch"] != self.epoch or self.closed:
                    raise AdapterError("request cancelled", 499)
                chunk = suffix[start:start + base.NATIVE_CHUNK]
                final = start + len(chunk) == len(suffix)
                n_chunk_imgs = sum(1 for t in chunk if t == IMAGE_TOKEN_ID)
                chunk_images = suffix_images[img_cursor:img_cursor + n_chunk_imgs]
                img_cursor += n_chunk_imgs
                self.sequence += 1
                request = dict(
                    id=self.sequence,
                    tokens=chunk,
                    max_tokens=plan["budget"] if final else 0,
                    reset=plan["reset"] and start == 0,
                    return_logits=False,
                )
                if chunk_images:
                    request["images"] = [img["url"] for img in chunk_images]
                    request["max_image_tokens"] = plan["max_image_tokens"]
                response = self.worker.request(request)
                if chunk_images:
                    counts = response.get("image_token_counts") if isinstance(response, dict) else None
                    if (not isinstance(counts, list) or len(counts) != len(chunk_images) or
                            any(type(c) is not int or c <= 0 for c in counts)):
                        raise AdapterError("invalid native image token accounting", 502)
                    expanded_chunk = expand_image_tokens(chunk, counts)
                    pos = len(actual_ids)
                    ci = 0
                    for tok in chunk:
                        if tok == IMAGE_TOKEN_ID:
                            cnt = counts[ci]
                            actual_images.append((pos, cnt, chunk_images[ci]["fingerprint"], plan["max_image_tokens"]))
                            pos += cnt
                            ci += 1
                        else:
                            pos += 1
                else:
                    expanded_chunk = chunk
                generated = self._check_native(response, request, before, expected_prefill=len(expanded_chunk))
                actual_ids.extend(expanded_chunk)
                before = response["cache_after"]
                prefill_seconds += response["prefill_seconds"]
                decode_seconds += response["decode_seconds"]
            if img_cursor != len(suffix_images):
                raise AdapterError("unconsumed image inputs in multimodal prefill", 502)
            if plan["epoch"] != self.epoch or self.closed:
                raise AdapterError("request cancelled", 499)
            plan["ids"] = actual_ids
            plan["suffix"] = actual_ids[plan["reused"]:]
            self.cached_ids = actual_ids + generated
            self.cached_images = actual_images
            self.needs_reset = False
            raw = self.tokenizer.decode(generated, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
            visible_ids = generated[:-1] if generated and generated[-1] in self.eos else generated
            text = self.tokenizer.decode(visible_ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
            parsed = base.parse_completion(text, plan["tools"], stopped=response["eos"])
            if response["eos"] and plan["payload"].get("tool_choice") == "required" and not parsed["tool_calls"]:
                raise AdapterError("model did not produce the required function call", 502)
            if plan["payload"].get("parallel_tool_calls") is False and len(parsed["tool_calls"]) > 1:
                raise AdapterError("model produced multiple calls when disabled", 502)
            request_id = "chatcmpl-" + uuid.uuid4().hex
            for index, call in enumerate(parsed["tool_calls"]):
                call["id"] = "call_{}_{}".format(request_id[9:], index)
            message = dict(role="assistant", content=parsed["content"])
            if parsed["reasoning_content"]:
                message["reasoning_content"] = parsed["reasoning_content"]
            if parsed["tool_calls"]:
                message["tool_calls"] = parsed["tool_calls"]
            normalized = base.normalize_messages(
                plan["messages"] + [message, dict(role="user", content="")],
                allow_images=True,
            )[-2]
            if list(self.tokenizer.encode(raw, add_special_tokens=False)) == generated:
                self.records.append(dict(
                    index=len(plan["messages"]),
                    prefix=base.fingerprint(plan["messages"]),
                    signature=base.assistant_signature(normalized),
                    raw=raw,
                    reasoning=parsed["reasoning_content"],
                ))
                self.records = self.records[-256:]
            result = dict(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=base.MODEL_ID,
                choices=[dict(index=0, message=message, finish_reason=parsed["finish_reason"])],
                usage=dict(
                    prompt_tokens=len(actual_ids),
                    completion_tokens=len(generated),
                    total_tokens=len(actual_ids) + len(generated),
                    prompt_tokens_details=dict(cached_tokens=plan["reused"]),
                ),
                nanoattention=dict(
                    cache_reset=plan["reset"],
                    reused_prefix_tokens=plan["reused"],
                    new_prefill_tokens=len(plan["suffix"]),
                    cache_after=response["cache_after"],
                    prefill_seconds=prefill_seconds,
                    decode_seconds=decode_seconds,
                    kv_allocated_bytes=response["kv_allocated_bytes"],
                    kv_valid_bytes=response["kv_valid_bytes"],
                    kv_storage=response.get("kv_storage", "fp32"),
                    speculative_batches=response.get("speculative_batches", 0),
                    speculative_proposed=response.get("speculative_proposed", 0),
                    speculative_accepted=response.get("speculative_accepted", 0),
                    effective_max_tokens=plan["budget"],
                    sampling="greedy",
                ),
            )
            self.metrics.append(dict(
                result["nanoattention"],
                prompt_tokens=len(actual_ids),
                completion_tokens=len(generated),
                finish_reason=parsed["finish_reason"],
                tool_call_count=len(parsed["tool_calls"]),
                worker_starts=self.worker_starts,
            ))
            self.metrics = self.metrics[-256:]
            return result
        except BaseException:
            self.failures += 1
            self.abort()
            raise
        finally:
            self.active = False

    def execute(self, plan):
        if self.worker is None and not self.closed:
            self.worker = self.worker_factory(self.command, timeout=self.timeout).__enter__()
            self.worker_starts += 1
            try:
                self._warm()
            except BaseException:
                self.abort()
                raise
        if plan.get("suffix_images"):
            result = self._execute_multimodal(plan)
        else:
            result = super().execute(plan)
            self.cached_images = list(plan.get("prefix_images", []))
        if self.record:
            generated = self.cached_ids[len(plan["ids"]):]
            na = result["nanoattention"]
            line = dict(index=len(self.metrics), prompt_ids=plan["ids"], suffix_ids=plan["suffix"], generated_ids=generated,
                        reused_prefix_tokens=plan["reused"], reset=plan["reset"],
                        prefill_seconds=na["prefill_seconds"],
                        decode_seconds=na["decode_seconds"],
                        speculative_batches=na.get("speculative_batches", 0),
                        speculative_proposed=na.get("speculative_proposed", 0),
                        speculative_accepted=na.get("speculative_accepted", 0),
                        finish_reason=result["choices"][0]["finish_reason"],
                        tool_calls=[c["function"]["name"] for c in result["choices"][0]["message"].get("tool_calls", [])])
            with open(self.record, "a") as stream:
                stream.write(json.dumps(line) + "\n")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--engine", choices=("native", "mlx"), required=True)
    parser.add_argument("--binary", type=Path, help="nano, for --engine native")
    parser.add_argument("--knobs", type=Path, help="knobs JSON preset for --engine native (default: src/mega/presets/inkling.json)")
    parser.add_argument("--draft", type=Path, help="DFlash 2 draft checkpoint directory (for --engine native)")
    parser.add_argument("--loop-passes", type=int, help="Inkling core loop passes (1..3, for --engine native)")
    parser.add_argument("--loopcd", choices=("off", "hidden", "logits", "logits-adaptive"), help="Inkling LoopCD mode (--engine native)")
    parser.add_argument("--loopcd-omega", type=float, help="LoopCD contrast scale omega >= 0 (--engine native)")
    parser.add_argument("--loopcd-weak-pass", type=int, help="1-based LoopCD weak pass index (--engine native)")
    parser.add_argument("--lsa", action="store_true", help="Enable Q4 Multi-Head LSA + cross-loop reuse (--engine native)")
    parser.add_argument("--mlx-python", type=Path, help="a Python with mlx-lm, for --engine mlx")
    parser.add_argument("--port", type=int, default=8080, help="loopback only; 0 chooses a disposable port")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=2048, help="server output cap, including committed EOS")
    parser.add_argument("--max-image-tokens", type=int, default=DEFAULT_MAX_IMAGE_TOKENS,
                        help="visual token budget per image (default: 1024)")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    text = config["text_config"]
    if not 0 < args.context <= text["model_max_length"] or not 0 < args.max_tokens < args.context:
        parser.error("invalid context or output cap")
    if args.max_image_tokens <= 0:
        parser.error("--max-image-tokens must be positive")
    if not 0 <= args.port <= 65535 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("invalid port or timeout")
    if args.engine == "native":
        if not args.binary:
            parser.error("--engine native needs --binary")
        command = native_command(
            args.binary.resolve(),
            args.checkpoint.resolve(),
            args.context,
            knobs=args.knobs.resolve() if args.knobs else None,
            draft=args.draft.resolve() if args.draft else None,
            loop_passes=args.loop_passes,
            loopcd=args.loopcd,
            loopcd_omega=args.loopcd_omega,
            loopcd_weak_pass=args.loopcd_weak_pass,
            lsa=args.lsa,
        )
    else:
        if any(x is not None for x in (args.knobs, args.draft, args.loop_passes, args.loopcd, args.loopcd_omega, args.loopcd_weak_pass)) or args.lsa:
            parser.error("--knobs/--draft/--loop-passes/--loopcd/--lsa are only supported with --engine native")
        if not args.mlx_python:
            parser.error("--engine mlx needs --mlx-python")
        command = mlx_command(args.mlx_python.absolute(), args.checkpoint.resolve(), args.context)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), local_files_only=True, trust_remote_code=False,
                                              fix_mistral_regex=False)
    tokenizer.chat_template = (args.checkpoint / "chat_template.jinja").read_text()
    eos = config["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    backend = InklingBackend(tokenizer, command, args.context, eos, text["unpadded_vocab_size"], args.max_tokens,
                             args.timeout, record=args.record, max_image_tokens=args.max_image_tokens, engine=args.engine)
    server = AdapterServer(("127.0.0.1", args.port), backend, os.environ.get("NANOATTENTION_API_KEY", "local"),
                           max_body=MAX_BODY)
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *unused: stopped.set())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(json.dumps(dict(event="ready", host="127.0.0.1", port=server.server_port, model=base.MODEL_ID,
                          context_length=args.context, engine=args.engine)), flush=True)
    try:
        stopped.wait()
    finally:
        backend.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
