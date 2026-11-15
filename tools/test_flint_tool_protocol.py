"""CPU-only Flint protocol tests; no tokenizer/model/server imports.

Run with Python 3.9: python3 -m unittest discover -s tools -p test_flint_tool_protocol.py -v
"""

import importlib.util
import json
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().with_name("flint_tool_protocol.py")
SPEC = importlib.util.spec_from_file_location("flint_tool_protocol", PATH)
protocol = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(protocol)
ProtocolError = protocol.ProtocolError
parse_completion = protocol.parse_completion


def tool(name="write_file", properties=None, required=None, **constraints):
    schema = {"type": "object", "properties": properties or {},
              "required": required or [], "additionalProperties": False}
    schema.update(constraints)
    return {"type": "function", "function": {"name": name, "parameters": schema}}


def call(name="write_file", pairs=()):
    return "<tool_call>" + name + "".join(
        "<arg_key>" + key + "</arg_key><arg_value>" + raw + "</arg_value>"
        for key, raw in pairs) + "</tool_call>"


def arguments(result, index=0):
    return json.loads(result["tool_calls"][index]["function"]["arguments"])


class FlintProtocolTests(unittest.TestCase):
    def setUp(self):
        self.write = tool(properties={"path": {"type": "string"},
                                      "content": {"type": "string"}},
                          required=["path", "content"])

    def test_plain_text_and_reasoning_are_separate_without_trimming(self):
        result = parse_completion(" before <think>plan\ncarefully</think> after ", [])
        self.assertEqual(result, {"content": " before  after ",
                                  "reasoning_content": "plan\ncarefully",
                                  "tool_calls": [], "finish_reason": "stop"})
        self.assertEqual(parse_completion("", None)["finish_reason"], "stop")

    def test_real_multiline_file_content_is_preserved(self):
        content = ('\n#!/usr/bin/env python3\n'
                   '"""A file with quotes, tabs, Unicode π and XML-ish text."""\n'
                   'def greet(name):\n'
                   '    payload = {"name": name, "enabled": True}\n'
                   '    print("hello\\n", payload)\n'
                   '    return "<think>literal, not reasoning</think> &amp;"\n\n  ')
        result = parse_completion(call(pairs=[("path", "src/greet.py"),
                                              ("content", content)]), [self.write])
        self.assertEqual(arguments(result), {"path": "src/greet.py", "content": content})
        self.assertEqual(result["reasoning_content"], "")
        self.assertEqual(result["content"], "")
        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertIsInstance(result["tool_calls"][0]["function"]["arguments"], str)

    def test_string_values_are_raw_even_if_they_look_like_json(self):
        schema = tool(properties={"s": {"type": "string"}}, required=["s"])
        for raw in ['"quoted"', ' true ', '123', '{"x":1}', '  \n ']:
            with self.subTest(raw=raw):
                result = parse_completion(call(pairs=[("s", raw)]), [schema])
                self.assertEqual(arguments(result)["s"], raw)

    def test_typed_flags_arrays_nested_objects_and_null(self):
        schema = tool("run", properties={
            "count": {"type": "integer", "minimum": 1},
            "ratio": {"type": "number"}, "enabled": {"type": "boolean"},
            "empty": {"type": "null"},
            "files": {"type": "array", "items": {"type": "string"}},
            "options": {"type": "object", "properties": {
                "retries": {"type": "integer"},
                "flags": {"type": "array", "items": {"type": "boolean"}}},
                "required": ["retries"], "additionalProperties": False},
        })
        result = parse_completion(call("run", [
            ("count", " 3 "), ("ratio", "0.125"), ("enabled", "false"),
            ("empty", "null"), ("files", '["a.py", "b.py"]'),
            ("options", '{"retries":2,"flags":[true,false]}'),
        ]), [schema])
        self.assertEqual(arguments(result), {
            "count": 3, "ratio": 0.125, "enabled": False, "empty": None,
            "files": ["a.py", "b.py"], "options": {"retries": 2, "flags": [True, False]}})

    def test_multiple_calls_are_ordered_with_visible_text_before_and_after(self):
        tools = [tool("first"), tool("second")]
        result = parse_completion("before\n" + call("first") + "between" + call("second") + "\nafter", tools)
        self.assertEqual(result["content"], "before\nbetween\nafter")
        self.assertEqual([x["id"] for x in result["tool_calls"]], ["call_0", "call_1"])
        self.assertEqual([x["function"]["name"] for x in result["tool_calls"]], ["first", "second"])
        self.assertTrue(all(x["type"] == "function" for x in result["tool_calls"]))
        self.assertEqual(arguments(result), {})

    def test_fake_or_malformed_calls_inside_reasoning_are_ignored(self):
        thought = call("not_whitelisted") + "<tool_call>broken<arg_key>"
        result = parse_completion("<think>" + thought + "</think>answer", [])
        self.assertEqual(result["reasoning_content"], thought)
        self.assertEqual(result["content"], "answer")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(result["finish_reason"], "stop")

    def test_only_calls_outside_reasoning_execute(self):
        result = parse_completion("<think>" + call("fake") + "</think>" + call("real"), [tool("real")])
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "real")

    def test_unclosed_thinking_blocks_even_preceding_completed_calls(self):
        for text in ("<think>" + call("go"), call("go") + "<think>still planning"):
            with self.subTest(text=text):
                result = parse_completion(text, [tool("go")])
                self.assertEqual(result["tool_calls"], [])
                self.assertEqual(result["finish_reason"], "length")
                self.assertTrue(result["reasoning_content"])

    def test_length_cap_never_emits_calls_or_repairs_partial_markup(self):
        for text in (call("go"), call("unknown"), "<tool_call>go<arg_key>x",
                     "visible<think>private", "<tool_call>broken</tool_call>"):
            with self.subTest(text=text):
                result = parse_completion(text, [tool("go")], stopped=False)
                self.assertEqual(result["tool_calls"], [])
                self.assertEqual(result["finish_reason"], "length")
        capped = parse_completion("prefix" + call("go"), [tool("go")], stopped=False)
        self.assertEqual(capped["content"], "prefix" + call("go"))

    def test_unknown_and_nonwhitelisted_tools_are_rejected(self):
        for tools in ([], [tool("other")]):
            with self.assertRaises(ProtocolError):
                parse_completion(call("go"), tools)

    def test_malformed_completed_calls_are_rejected(self):
        malformed = [
            "<tool_call>write_file", "</tool_call>", "<tool_call", "<arg_key",
            "<tool_call name='write_file'></tool_call>",
            "<tool_call>write_file{\"path\":\"x\"}</tool_call>",
            "<tool_call>write_file<arg_key>path</arg_key></tool_call>",
            "<tool_call>write_file<arg_key>path<arg_value>x</arg_value></tool_call>",
            "<tool_call>write_file<arg_value>x</arg_value></tool_call>",
            "<tool_call>write_file<arg_key>path</arg_key><arg_value>x</tool_call>",
            "<tool_call>write_file<arg_key></arg_key><arg_value>x</arg_value></tool_call>",
            call(pairs=[("path", "x"), ("content", "y")])[:-12] + "junk</tool_call>",
            "<arg_key>x</arg_key>", "</think>",
        ]
        for text in malformed:
            with self.subTest(text=text), self.assertRaises(ProtocolError):
                parse_completion(text, [self.write])

    def test_duplicate_and_missing_required_arguments_are_rejected(self):
        for pairs in ([('path', 'x'), ('path', 'y'), ('content', 'z')],
                      [('path', 'x')], []):
            with self.subTest(pairs=pairs), self.assertRaises(ProtocolError):
                parse_completion(call(pairs=pairs), [self.write])

    def test_non_object_parameter_schemas_are_rejected(self):
        for schema in ({"type": "array"}, {"type": "string"}, {"type": "null"}):
            tools = [{"type": "function", "function": {"name": "go", "parameters": schema}}]
            with self.subTest(schema=schema), self.assertRaises(ProtocolError):
                parse_completion(call("go"), tools)

    def test_wrong_types_and_non_json_values_are_rejected(self):
        bad = [("integer", "true"), ("integer", "1.5"), ("number", '"2"'),
               ("boolean", "True"), ("boolean", "1"), ("object", "[]"),
               ("array", "{}"), ("null", '"null"'), ("object", "{'a':1}"),
               ("number", "NaN"), ("number", "Infinity"), ("number", "1e999"),
               ("object", '{"x":1,"x":2}')]
        for kind, raw in bad:
            with self.subTest(kind=kind, raw=raw), self.assertRaises(ProtocolError):
                parse_completion(call(pairs=[("x", raw)]), [tool(properties={"x": {"type": kind}})])

    def test_nested_validation_and_additional_properties(self):
        value_schema = {"type": "object", "properties": {
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 1}},
            "required": ["items"], "additionalProperties": False}
        schema = tool(properties={"x": value_schema})
        for raw in ('{}', '{"items":[true]}', '{"items":[]}', '{"items":[1],"extra":2}'):
            with self.subTest(raw=raw), self.assertRaises(ProtocolError):
                parse_completion(call(pairs=[("x", raw)]), [schema])
        with self.assertRaises(ProtocolError):
            parse_completion(call(pairs=[("unknown", "x")]), [self.write])

    def test_alternatives_nullable_and_allof(self):
        schemas = [({"type": ["object", "null"]}, 'null', None),
                   ({"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, 'true', True),
                   ({"oneOf": [{"type": "object"}, {"type": "array"}]}, '[]', []),
                   ({"allOf": [{"type": "integer"}, {"minimum": 2}]}, '3', 3),
                   ({"allOf": [{"type": "string"}, {"minLength": 1}]}, '"raw"', '"raw"')]
        for schema, raw, expected in schemas:
            with self.subTest(schema=schema):
                result = parse_completion(call(pairs=[("x", raw)]), [tool(properties={"x": schema})])
                self.assertEqual(arguments(result)["x"], expected)

    def test_ambiguous_alternatives_and_oneof_overlap_are_rejected(self):
        for schema in ({"type": ["string", "integer"]},
                       {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                       {"anyOf": [{"type": "string"}, {}]},
                       {"type": ["string", "null"]},
                       {"oneOf": [{"type": "integer"}, {"type": "number"}]}):
            with self.subTest(schema=schema), self.assertRaises(ProtocolError):
                raw = "null" if schema == {"type": ["string", "null"]} else "1"
                parse_completion(call(pairs=[("x", raw)]), [tool(properties={"x": schema})])

    def test_untyped_values_use_json_or_raw_text_without_eval(self):
        schema = tool(properties={"x": {}})
        for raw, expected in [('true', True), ('[1,2]', [1, 2]),
                              ('hello\n world', 'hello\n world'), ('1 + 2', '1 + 2')]:
            with self.subTest(raw=raw):
                self.assertEqual(arguments(parse_completion(call(pairs=[("x", raw)]), [schema]))["x"], expected)

    def test_schema_constraints_and_unsupported_references_fail_closed(self):
        for schema, raw in [({"type": "string", "enum": ["a"]}, 'b'),
                            ({"type": "boolean", "enum": [1]}, 'true'),
                            ({"type": "array", "uniqueItems": True}, '[1,1]'),
                            ({"type": "string", "pattern": "^ok$"}, 'no'),
                            ({"$ref": "#/$defs/x"}, '1')]:
            with self.subTest(schema=schema), self.assertRaises(ProtocolError):
                parse_completion(call(pairs=[("x", raw)]), [tool(properties={"x": schema})])

    def test_duplicate_tool_definitions_are_rejected(self):
        with self.assertRaises(ProtocolError):
            parse_completion(call("go"), [tool("go"), tool("go")])


if __name__ == "__main__":
    unittest.main()
