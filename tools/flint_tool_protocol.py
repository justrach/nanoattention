"""CPU-only parsing of Flint's native tool-call syntax (stdlib only).

parse_completion(text, tools, stopped=True) accepts decoded text without native
EOS and OpenAI function definitions. It never executes calls. Truncated output
(including an unclosed <think>) returns finish_reason='length' and no calls;
its visible content retains untrusted, possibly partial tool markup. Completed
calls are removed from content, with surrounding whitespace preserved verbatim.

String arguments are literal, not JSON-unquoted. Other explicit types require
strict JSON. Untyped arguments prefer JSON when valid, otherwise literal text;
ambiguous typed alternatives are rejected rather than guessed. Reserved closing
delimiters cannot occur literally inside their corresponding raw values. This
is a protocol parser, not an XML parser; no entity expansion or XML repair occurs.
"""

import json
import math
import re


class ProtocolError(ValueError):
    """The completion or its tool schema cannot safely describe a tool call."""


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError("Duplicate JSON object key: " + key)
            result[key] = value
        return result

    def number(raw_number):
        value = float(raw_number)
        if not math.isfinite(value):
            raise ProtocolError("Non-finite JSON number")
        return value

    def constant(value):
        raise ProtocolError("Invalid JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                      parse_constant=constant)


def _is_type(value, kind):
    checks = {
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "null": lambda: value is None,
    }
    if kind not in checks:
        raise ProtocolError("Unsupported schema type: " + str(kind))
    return checks[kind]()


def _same(left, right):
    # JSON booleans and numbers are distinct even though Python True == 1.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _validate(value, schema):
    if schema is True:
        return
    if schema is False:
        raise ProtocolError("Value forbidden by schema")
    if not isinstance(schema, dict):
        raise ProtocolError("Schema must be an object or boolean")
    supported = {
        "type", "properties", "required", "additionalProperties", "items",
        "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
        "minLength", "maxLength", "pattern", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "enum", "const", "anyOf",
        "oneOf", "allOf", "not", "title", "description", "default", "examples",
        "deprecated", "readOnly", "writeOnly", "format", "$schema", "$id",
        "$defs", "definitions",
    }
    if any(key not in supported for key in schema):
        raise ProtocolError("Unsupported schema keyword")
    kinds = schema.get("type")
    if kinds is not None:
        kinds = kinds if isinstance(kinds, list) else [kinds]
        if not kinds or not any(_is_type(value, kind) for kind in kinds):
            raise ProtocolError("Argument has the wrong JSON type")
    if "enum" in schema and not any(_same(value, x) for x in schema["enum"]):
        raise ProtocolError("Argument is not an allowed enum value")
    if "const" in schema and not _same(value, schema["const"]):
        raise ProtocolError("Argument differs from schema const")
    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword in schema:
            branches = schema[keyword]
            if not isinstance(branches, list) or not branches:
                raise ProtocolError("Schema alternatives must be nonempty lists")
            matches = sum(_matches(value, branch) for branch in branches)
            if ((keyword == "anyOf" and not matches)
                    or (keyword == "oneOf" and matches != 1)
                    or (keyword == "allOf" and matches != len(branches))):
                raise ProtocolError("Argument does not satisfy " + keyword)
    if "not" in schema and _matches(value, schema["not"]):
        raise ProtocolError("Argument satisfies forbidden schema")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ProtocolError("Schema properties must be an object")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(not isinstance(k, str) for k in required):
            raise ProtocolError("Schema required must be a list of keys")
        if any(key not in value for key in required):
            raise ProtocolError("Missing required argument")
        for key, item in value.items():
            _validate(item, properties.get(key, schema.get("additionalProperties", True)))
        if len(value) < schema.get("minProperties", 0):
            raise ProtocolError("Too few object properties")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            raise ProtocolError("Too many object properties")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ProtocolError("Too few array items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ProtocolError("Too many array items")
        if schema.get("uniqueItems") and any(
                _same(item, other) for i, item in enumerate(value) for other in value[:i]):
            raise ProtocolError("Duplicate array item")
        for item in value:
            _validate(item, schema.get("items", True))
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ProtocolError("String too short")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ProtocolError("String too long")
        if "pattern" in schema:
            try:
                matched = re.search(schema["pattern"], value)
            except (re.error, TypeError) as exc:
                raise ProtocolError("Invalid schema pattern") from exc
            if matched is None:
                raise ProtocolError("String does not match pattern")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        for key, fails in (("minimum", lambda n: value < n),
                           ("maximum", lambda n: value > n),
                           ("exclusiveMinimum", lambda n: value <= n),
                           ("exclusiveMaximum", lambda n: value >= n)):
            if key in schema and fails(schema[key]):
                raise ProtocolError("Number violates " + key)


def _matches(value, schema):
    try:
        _validate(value, schema)
        return True
    except ProtocolError:
        return False


def _type_hints(schema):
    if not isinstance(schema, dict):
        return []
    if "type" in schema:
        kinds = schema["type"]
        return kinds if isinstance(kinds, list) else [kinds]
    result = []
    for keyword in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(keyword, [])
        if not isinstance(branches, list):
            raise ProtocolError("Schema alternatives must be lists")
        for branch in branches:
            hints = _type_hints(branch)
            # An unconstrained alternative can accept decoded JSON too. An
            # untyped allOf clause merely adds constraints, not alternatives.
            result.extend(hints or ([None] if keyword != "allOf" else []))
    return result


def _coerce(raw, schema):
    kinds = _type_hints(schema)
    candidates = []
    if not kinds or any(kind != "string" for kind in kinds):
        try:
            candidates.append(_json(raw))
        except ValueError:
            if not kinds or None in kinds:
                candidates.append(raw)
    if "string" in kinds:
        candidates.append(raw)
    valid = []
    for candidate in candidates:
        if _matches(candidate, schema) and not any(_same(candidate, old) for old in valid):
            valid.append(candidate)
    if len(valid) != 1:
        raise ProtocolError("Argument is invalid or ambiguous for its schema")
    return valid[0]


def _schemas(tools):
    result = {}
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ProtocolError("Only OpenAI function tools are supported")
        function = tool.get("function", {})
        if not isinstance(function, dict):
            raise ProtocolError("Invalid function definition")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise ProtocolError("Invalid or duplicate function name")
        schema = function.get("parameters", {"type": "object"})
        if not isinstance(schema, dict):
            raise ProtocolError("Function parameters must be an object schema")
        result[name] = schema
    return result


def _parse_call(body, schemas, index):
    first_tag = body.find("<")
    name = (body if first_tag < 0 else body[:first_tag]).strip()
    if name not in schemas:
        raise ProtocolError("Unknown or non-whitelisted function: " + name)
    schema = schemas[name]
    position = len(body) if first_tag < 0 else first_tag
    arguments = {}
    while body[position:].strip():
        position += len(body[position:]) - len(body[position:].lstrip())
        if not body.startswith("<arg_key>", position):
            raise ProtocolError("Expected <arg_key>")
        end = body.find("</arg_key>", position + len("<arg_key>"))
        if end < 0:
            raise ProtocolError("Unclosed argument key")
        key = body[position + len("<arg_key>"):end]
        if not key or "<" in key or key in arguments:
            raise ProtocolError("Empty, malformed, or duplicate argument key")
        position = end + len("</arg_key>")
        position += len(body[position:]) - len(body[position:].lstrip())
        if not body.startswith("<arg_value>", position):
            raise ProtocolError("Expected <arg_value>")
        start = position + len("<arg_value>")
        end = body.find("</arg_value>", start)
        if end < 0:
            raise ProtocolError("Unclosed argument value")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ProtocolError("Schema properties must be an object")
        argument_schema = properties.get(key, schema.get("additionalProperties", True))
        arguments[key] = _coerce(body[start:end], argument_schema)
        position = end + len("</arg_value>")
    _validate(arguments, schema)
    return {"id": "call_" + str(index), "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False,
                                                                  allow_nan=False)}}


def parse_completion(text, tools, stopped=True):
    """Return content, reasoning_content, tool_calls, and finish_reason.

    ProtocolError rejects malformed *completed* output. A length-capped result
    is intentionally never validated into executable calls, even if it contains
    a fully closed call. Unclosed thinking likewise invalidates every call.
    IDs are local to this completion and start at call_0.
    """
    if not isinstance(text, str):
        raise ProtocolError("Completion text must be a string")
    visible = []
    reasoning = []
    bodies = []
    untrusted = []
    incomplete_thinking = False
    position = 0
    markers = re.compile(r"<think>|</think>|</?(?:tool_call|arg_key|arg_value)\b[^>]*(?:>|$)")
    malformed = False
    while position < len(text):
        match = markers.search(text, position)
        if match is None:
            visible.append(text[position:])
            untrusted.append(text[position:])
            break
        visible.append(text[position:match.start()])
        untrusted.append(text[position:match.start()])
        tag = match.group()
        if tag == "<think>":
            end = text.find("</think>", match.end())
            if end < 0:
                reasoning.append(text[match.end():])
                incomplete_thinking = True
                break
            thought = text[match.end():end]
            if "<think>" in thought:
                malformed = True
            reasoning.append(thought)
            position = end + len("</think>")
        elif tag == "<tool_call>":
            end = text.find("</tool_call>", match.end())
            if end < 0:
                malformed = True
                untrusted.append(text[match.start():])
                break
            bodies.append(text[match.end():end])
            untrusted.append(text[match.start():end + len("</tool_call>")])
            position = end + len("</tool_call>")
        else:
            malformed = True
            untrusted.append(tag)
            position = match.end()
    if not stopped or incomplete_thinking:
        return {"content": "".join(untrusted), "reasoning_content": "".join(reasoning),
                "tool_calls": [], "finish_reason": "length"}
    if malformed:
        raise ProtocolError("Malformed tool or thinking delimiters")
    schemas = _schemas(tools)
    calls = [_parse_call(body, schemas, index) for index, body in enumerate(bodies)]
    return {"content": "".join(visible), "reasoning_content": "".join(reasoning),
            "tool_calls": calls, "finish_reason": "tool_calls" if calls else "stop"}
