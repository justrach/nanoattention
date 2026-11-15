#!/usr/bin/env python3
"""CPU-only protocol tests; no checkpoints, native binaries, servers, or MLX GPU work."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark
import compare_mlx
import tune_attention

try:
    import numpy as np
except ImportError:
    np = None


class FakeTokenizer:
    chat_template = "fake local template"

    def __init__(self):
        self.chat_calls = []
        self.encode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return [101, 102, 103]

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        return list(map(ord, text))


def rows(repeats=2, reused=False):
    result = []
    for engine in benchmark.ENGINES:
        for repeat in range(repeats):
            row = dict(engine=engine, repeat=repeat, prefill_seconds=2.0,
                       decode_seconds=1.0, checksum=7)
            if engine == "zig_fp32" and reused:
                row.update(prefill_reused=True, prefill_samples=1)
            result.append(row)
    return result


class WorkloadTests(unittest.TestCase):
    def test_original_short_workload_and_hash_unchanged(self):
        tokenizer = FakeTokenizer()
        prompt, decode = benchmark.workload(tokenizer, 64)
        self.assertEqual(prompt, [101, 102, 103])
        self.assertEqual(decode, list(map(ord, benchmark.CONTINUATION * 4))[:64])
        self.assertEqual(tokenizer.encode_calls,
                         [(benchmark.CONTINUATION * 4, {"add_special_tokens": False})])
        self.assertEqual(tokenizer.chat_calls, [(
            [{"role": "user", "content": benchmark.PROMPT}],
            dict(tokenize=True, add_generation_prompt=True, return_dict=False))])
        report = benchmark.summarize(rows(), prompt, decode, 2)
        # Captured from the original short-workload implementation with this tokenizer.
        self.assertEqual(report["input_sha256"],
                         "b5827bb84023bd0bcbc347404099d09cb26b127b50eab380eeb8518077705a0e")
        self.assertNotIn("prompt_protocol", report)
        self.assertNotIn("reuse_prefill", report["native_execution"])
        self.assertNotIn("attention_implementation", report["native_execution"])
        self.assertNotIn("attention_implementation", report["engines"]["zig_fp32"])
        self.assertEqual(benchmark.workload(FakeTokenizer(), 64, None), (prompt, decode))

    def test_exact_real_prefix_lengths_and_unchanged_continuation(self):
        original_decode = benchmark.workload(FakeTokenizer(), 64)[1]
        stock = list(map(ord, benchmark.CONTINUATION))
        for size in (1, 1024, 4096, 8192):
            with self.subTest(size=size):
                tokenizer = FakeTokenizer()
                prompt, decode = benchmark.workload(tokenizer, 64, size)
                self.assertEqual(len(prompt), size)
                self.assertEqual(prompt, [stock[i % len(stock)] for i in range(size)])
                self.assertEqual(decode, original_decode)
                self.assertEqual(tokenizer.chat_calls, [])
                self.assertEqual(prompt, benchmark.code_prompt(FakeTokenizer(), size))

    def test_invalid_prefix_and_empty_or_invalid_stock_rejected(self):
        for size in (0, -1, 2**32, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                benchmark.code_prompt(FakeTokenizer(), size)
        for stock in ([], [-1], [2**32], [True], ["1"]):
            with self.subTest(stock=stock), self.assertRaises(ValueError):
                benchmark.code_prompt(types.SimpleNamespace(encode=lambda *a, **k: stock), 8)

    def test_long_report_records_actual_context_and_same_input_hash(self):
        prompt, decode = benchmark.workload(FakeTokenizer(), 64, 1024)
        report = benchmark.summarize(rows(), prompt, decode, 2, prompt_tokens=1024)
        self.assertEqual(report["context_length"], 1088)
        self.assertEqual(report["mlx_prefill_chunk_size"], 512)
        self.assertIn("not quality evaluation", report["prompt_protocol"])
        self.assertEqual(report["input_sha256"],
                         benchmark.summarize(rows(), prompt, decode, 2)["input_sha256"])
        with self.assertRaises(ValueError):
            benchmark.summarize(rows(), prompt, decode, 2, prompt_tokens=1025)

    def test_reuse_prefill_is_one_sample_not_repeated_samples(self):
        report = benchmark.summarize(rows(reused=True), [1, 2], [3], 2, reuse_prefill=True)
        native = report["engines"]["zig_fp32"]
        self.assertEqual(native["prefill_run_seconds"], [2.0])
        self.assertEqual(native["prefill_samples"], 1)
        self.assertTrue(native["prefill_reused"])
        self.assertEqual(report["engines"]["mlx_fp32"]["prefill_run_seconds"], [2.0, 2.0])
        self.assertIn("one sample", report["prefill_note"])
        with self.assertRaises(ValueError):
            benchmark.summarize(rows(), [1], [2], 2, reuse_prefill=True)
        invalid = rows(reused=True)
        invalid[1]["prefill_seconds"] = 3.0
        with self.assertRaises(ValueError):
            benchmark.summarize(invalid, [1], [2], 2, reuse_prefill=True)

    def test_native_reuse_input_is_explicit_and_default_omits_it(self):
        for reuse in (False, True):
            captured = []

            def run(argv, **kwargs):
                captured.append(json.loads(Path(argv[1]).read_text()))
                selected = [row for row in rows(reused=reuse) if row["engine"] == "zig_fp32"]
                return types.SimpleNamespace(stdout="\n".join(map(json.dumps, selected)))

            with patch.object(benchmark, "subprocess", types.SimpleNamespace(run=run, PIPE=-1)):
                benchmark.native_runs(Path("fake-binary"), Path("fake-checkpoint"),
                                      [1, 2, 3], [4], 2, reuse_prefill=reuse)
            self.assertEqual(captured[0]["prompt"], [1, 2, 3])
            if reuse:
                self.assertIs(captured[0]["reuse_prefill"], True)
            else:
                self.assertNotIn("reuse_prefill", captured[0])

    def test_native_mlx_attention_is_opt_in_and_requires_observed_labels(self):
        for enabled in (False, True):
            for reuse in (False, True):
                captured = []
                selected = [row for row in rows(reused=reuse) if row["engine"] == "zig_fp32"]
                if enabled:
                    for row in selected:
                        row["attention_implementation"] = benchmark.MLX_ATTENTION_IMPLEMENTATION

                def run(argv, **kwargs):
                    captured.append(json.loads(Path(argv[1]).read_text()))
                    return types.SimpleNamespace(stdout="\n".join(map(json.dumps, selected)))

                with patch.object(benchmark, "subprocess", types.SimpleNamespace(run=run, PIPE=-1)):
                    observed = benchmark.native_runs(Path("fake-binary"), Path("fake-checkpoint"),
                                                     [1, 2, 3], [4], 2,
                                                     reuse_prefill=reuse, mlx_attention=enabled)
                    self.assertEqual(observed, selected)
                    if enabled:
                        self.assertIs(captured[0]["mlx_attention"], True)
                        for invalid in (None, "baseline"):
                            selected[-1]["attention_implementation"] = invalid
                            with self.assertRaisesRegex(ValueError, "label MLX-layout"):
                                benchmark.native_runs(Path("fake-binary"), Path("fake-checkpoint"),
                                                      [1], [2], 2, mlx_attention=True)
                    else:
                        self.assertNotIn("mlx_attention", captured[0])
                self.assertEqual(captured[0]["prompt"], [1, 2, 3])
                self.assertNotIn("attention_candidates", captured[0])

    def test_native_mlx_attention_conflicts_before_launch(self):
        with patch.object(benchmark.subprocess, "run") as launch:
            for kwargs in ({"resident_workers": 1}, *({"attention_chunk": chunk}
                                                   for chunk in benchmark.ATTENTION_CHUNKS)):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "conflicts"):
                    benchmark.native_runs(Path("unused"), Path("unused"), [1], [2], 2,
                                          mlx_attention=True, **kwargs)
            launch.assert_not_called()

    def test_mlx_attention_summary_labels_native_only_and_keeps_protocol(self):
        for reuse in (False, True):
            observed = rows(reused=reuse)
            for row in observed:
                if row["engine"] == "zig_fp32":
                    row["attention_implementation"] = benchmark.MLX_ATTENTION_IMPLEMENTATION
            report = benchmark.summarize(observed, [1, 2], [3], 2, prompt_tokens=2,
                                         reuse_prefill=reuse, mlx_attention=True)
            self.assertEqual(report["native_execution"]["attention_implementation"],
                             benchmark.MLX_ATTENTION_IMPLEMENTATION)
            self.assertEqual(report["engines"]["zig_fp32"]["attention_implementation"],
                             benchmark.MLX_ATTENTION_IMPLEMENTATION)
            self.assertEqual(report["input_sha256"], benchmark.summarize(
                rows(reused=reuse), [1, 2], [3], 2, prompt_tokens=2,
                reuse_prefill=reuse)["input_sha256"])
            self.assertEqual(report["context_length"], 3)
            self.assertEqual(report["mlx_prefill_chunk_size"], 512)
            for engine in ("mlx_fp32", "mlx_bf16"):
                self.assertNotIn("attention_implementation", report["engines"][engine])
            observed[0].pop("attention_implementation")
            with self.assertRaisesRegex(ValueError, "label MLX-layout"):
                benchmark.summarize(observed, [1, 2], [3], 2, mlx_attention=True)

    def test_native_attention_chunk_is_only_sent_when_explicit(self):
        for chunk in (None, *benchmark.ATTENTION_CHUNKS):
            captured = []

            def run(argv, **kwargs):
                captured.append(json.loads(Path(argv[1]).read_text()))
                selected = [row for row in rows() if row["engine"] == "zig_fp32"]
                return types.SimpleNamespace(stdout="\n".join(map(json.dumps, selected)))

            with patch.object(benchmark, "subprocess", types.SimpleNamespace(run=run, PIPE=-1)):
                benchmark.native_runs(Path("fake-binary"), Path("fake-checkpoint"),
                                      [1], [2], 2, attention_chunk=chunk)
            if chunk is None:
                self.assertNotIn("attention_chunk", captured[0])
            else:
                self.assertEqual(captured[0]["attention_chunk"], chunk)
        for chunk in (-1, 1, 2048, True):
            with self.assertRaises(ValueError):
                benchmark.native_runs(Path("unused"), Path("unused"), [1], [2], 2,
                                      attention_chunk=chunk)

    def test_native_attention_summary_uses_observed_rows_and_keeps_old_rows_compatible(self):
        self.assertNotIn("attention_chunk", benchmark.summarize(rows(), [1], [2], 2)["native_execution"])
        observed = rows()
        for row in observed:
            if row["engine"] == "zig_fp32":
                row["attention_chunk"] = 0
        report = benchmark.summarize(observed, [1], [2], 2)
        self.assertEqual(report["native_execution"]["attention_chunk"], 0)
        self.assertEqual(report["engines"]["zig_fp32"]["attention_chunk"], [0, 0])
        observed[1]["attention_chunk"] = 256
        with self.assertRaises(ValueError):
            benchmark.summarize(observed, [1], [2], 2)

    def test_cli_rejects_invalid_attention_choices_and_resident_conflicts(self):
        for module in (benchmark, compare_mlx):
            for flags in (["--attention-chunk", "1"],
                          ["--attention-chunk", "0", "--resident-workers", "1"],
                          ["--mlx-attention", "--resident-workers", "1"],
                          ["--mlx-attention", "--attention-chunk", "0"],
                          ["--mlx-attention", "--attention-chunk", "128"]):
                with self.subTest(module=module.__name__, flags=flags), \
                     patch.object(sys, "argv", [module.__name__, "unused", *flags]), \
                     contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_invalid_prompt_lengths_before_importing_models(self):
        for module in (benchmark, compare_mlx):
            for value in ("0", "-1", "4294967296", "1.5"):
                with self.subTest(module=module.__name__, value=value):
                    with patch.object(sys, "argv", [module.__name__, "unused", "--prompt-tokens", value]), \
                         contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                        module.main()
                    self.assertEqual(raised.exception.code, 2)

    def test_benchmark_cli_threads_mlx_attention_without_changing_reference(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())
        with tempfile.TemporaryDirectory() as temp:
            checkpoint, binary, output = (Path(temp) / name for name in ("checkpoint", "binary", "report.json"))
            checkpoint.mkdir()
            binary.touch()
            native_rows = [row for row in rows(reused=True) if row["engine"] == "zig_fp32"]
            for row in native_rows:
                row["attention_implementation"] = benchmark.MLX_ATTENTION_IMPLEMENTATION
            mlx_rows = [row for row in rows() if row["engine"] != "zig_fp32"]
            argv = ["benchmark", str(checkpoint), "--binary", str(binary), "--prompt-tokens", "1024",
                    "--decode-tokens", "64", "--repeats", "2", "--reuse-prefill", "--mlx-attention",
                    "--output", str(output)]
            with patch.dict(sys.modules, {"transformers": transformers}), \
                 patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(benchmark, "native_runs", return_value=native_rows) as native, \
                 patch.object(benchmark, "mlx_runs", return_value=mlx_rows) as mlx:
                benchmark.main()
            self.assertIs(native.call_args.kwargs["mlx_attention"], True)
            self.assertIs(native.call_args.kwargs["reuse_prefill"], True)
            self.assertIsNone(native.call_args.kwargs["attention_chunk"])
            self.assertEqual(len(native.call_args.args[2]), 1024)
            self.assertEqual(mlx.call_args.kwargs, {"prefill_chunk_size": 512})
            report = json.loads(output.read_text())
            self.assertEqual(report["native_execution"]["attention_implementation"],
                             benchmark.MLX_ATTENTION_IMPLEMENTATION)
            self.assertEqual(report["context_length"], 1088)
            self.assertTrue(all(row.get("attention_implementation") == benchmark.MLX_ATTENTION_IMPLEMENTATION
                                for row in report["runs"] if row["engine"] == "zig_fp32"))

    def test_benchmark_cli_threads_long_protocol_and_guards_checkpoint(self):
        tokenizer = FakeTokenizer()
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "checkpoint"
            checkpoint.mkdir()
            binary = Path(temp) / "binary"
            binary.touch()
            output = Path(temp) / "report.json"
            native_rows = [row for row in rows(reused=True) if row["engine"] == "zig_fp32"]
            mlx_rows = [row for row in rows() if row["engine"] != "zig_fp32"]
            argv = ["benchmark", str(checkpoint), "--binary", str(binary), "--prompt-tokens", "1024",
                    "--decode-tokens", "64", "--repeats", "2", "--reuse-prefill",
                    "--attention-chunk", "512", "--output", str(output)]
            with patch.dict(sys.modules, {"transformers": transformers}), \
                 patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(benchmark, "native_runs", return_value=native_rows) as native, \
                 patch.object(benchmark, "mlx_runs", return_value=mlx_rows) as mlx:
                benchmark.main()
            self.assertTrue(native.call_args.kwargs["reuse_prefill"])
            self.assertEqual(native.call_args.kwargs["attention_chunk"], 512)
            self.assertEqual(len(native.call_args.args[2]), 1024)
            self.assertEqual(mlx.call_args.kwargs["prefill_chunk_size"], 512)
            self.assertEqual(json.loads(output.read_text())["context_length"], 1088)
            argv[-1] = str(checkpoint / "forbidden.json")
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                benchmark.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((checkpoint / "forbidden.json").exists())


@unittest.skipIf(np is None, "optional NumPy unavailable; no dependencies downloaded")
class PrefillAndComparisonTests(unittest.TestCase):
    def test_chunked_prefill_evaluates_all_real_tokens_in_one_cache(self):
        for chunk_size in (None, 1, 256, 512):
            with self.subTest(chunk_size=chunk_size):
                calls, evaluations, cache = [], [], object()

                def model(tokens, cache):
                    calls.append((tokens.tolist()[0], cache))
                    return np.repeat(tokens[:, :, None], 3, axis=2)

                mx = types.SimpleNamespace(array=np.array, int32=np.int32,
                                           eval=lambda logits: evaluations.append(logits.shape))
                prompt = list(range(1025))
                logits = benchmark.mlx_prefill(model, mx, prompt, cache, chunk_size)
                self.assertEqual([token for chunk, _ in calls for token in chunk], prompt)
                self.assertTrue(all(seen_cache is cache for _, seen_cache in calls))
                if chunk_size is None:
                    self.assertEqual(evaluations, [(1, 1025, 3)])
                else:
                    self.assertTrue(all(len(chunk) <= chunk_size for chunk, _ in calls))
                    self.assertTrue(all(shape == (1, 1, 3) for shape in evaluations))
                    self.assertEqual(logits[0, 0, 0], 1024)

    def run_comparison(self, prompt_tokens, attention_chunk=None, mlx_prefill_chunk_size=None,
                       mlx_attention=False):
        tokenizer, events, model_inputs, commands = FakeTokenizer(), [], [], []
        mx = types.ModuleType("mlx.core")
        mx.array, mx.int32, mx.float32, mx.argmax = np.array, np.int32, np.float32, np.argmax
        mx.eval = lambda *args: None
        mx.set_memory_limit = mx.set_cache_limit = lambda *args: None
        mlx = types.ModuleType("mlx")
        mlx.core = mx

        def logits_for(tokens):
            return np.eye(5, dtype=np.float32)[(np.asarray(tokens) + 1) % 5]

        class Model:
            def set_dtype(self, dtype):
                pass

            def __call__(self, tokens, cache):
                model_inputs.extend(tokens.tolist()[0])
                cache.extend(tokens.tolist()[0])
                return logits_for(tokens)

        def load(*args, **kwargs):
            events.append("mlx_load")
            return Model(), None

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
        mlx_lm, cache_module = types.ModuleType("mlx_lm"), types.ModuleType("mlx_lm.models.cache")
        mlx_lm.load = load
        cache_module.make_prompt_cache = lambda model: []

        def run(argv, **kwargs):
            commands.append(argv)
            mode = argv[1]
            events.append(mode)
            tokens = json.loads(Path(argv[argv.index("--tokens-file") + 1]).read_text())
            if mode == "generate":
                generated, token = [], tokens[-1]
                for _ in range(int(argv[argv.index("--max-tokens") + 1])):
                    token = (token + 1) % 5
                    generated.append(token)
                return types.SimpleNamespace(stdout=json.dumps({"tokens": generated}))
            start = int(argv[argv.index("--logits-from") + 1]) if "--logits-from" in argv else 0
            return types.SimpleNamespace(stdout="\n".join(json.dumps(row.tolist())
                                                          for row in logits_for(tokens[start:])))

        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "chat_template.jinja").write_text("fake")
            argv = ["compare_mlx", temp, "--decode-tokens", "16"]
            if prompt_tokens is not None:
                argv += ["--prompt-tokens", str(prompt_tokens)]
            if attention_chunk is not None:
                argv += ["--attention-chunk", str(attention_chunk)]
            if mlx_prefill_chunk_size is not None:
                argv += ["--mlx-prefill-chunk-size", str(mlx_prefill_chunk_size)]
            if mlx_attention:
                argv += ["--mlx-attention"]
            output = io.StringIO()
            modules = {"transformers": transformers, "mlx": mlx, "mlx.core": mx,
                       "mlx_lm": mlx_lm, "mlx_lm.models.cache": cache_module}
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                 patch.object(compare_mlx, "subprocess", types.SimpleNamespace(run=run, PIPE=-1)), \
                 contextlib.redirect_stdout(output):
                compare_mlx.main()
        report, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(events, ["generate", "eval", "mlx_load"])
        self.assertEqual(report["max_absolute_logit_error"], 0)
        self.assertEqual(report["greedy_decode_agreement"], 16)
        self.assertEqual(len(model_inputs), report["prompt_tokens"] + 15)
        return report, commands, tokenizer

    def test_long_compare_limits_output_but_really_prefills_full_prefix(self):
        for size in (1024, 4096, 8192):
            with self.subTest(size=size):
                report, commands, tokenizer = self.run_comparison(size)
                self.assertEqual(report["evaluated_positions"], 16)
                self.assertEqual(report["validated_position_range"]["start"], size - 1)
                self.assertEqual(report["validated_position_range"]["end_inclusive"], size + 14)
                self.assertEqual(report["context_length"], size + 16)
                self.assertEqual(report["mlx_prefill_chunk_size"], 256)
                self.assertNotIn("--logits-from", commands[0])
                self.assertEqual(commands[1][-2:], ["--logits-from", str(size - 1)])
                self.assertEqual(tokenizer.chat_calls, [])

    def test_comparison_forwards_only_explicit_chunk_to_generate_and_eval(self):
        for chunk in (None, 0, 256, 1024):
            report, commands, _ = self.run_comparison(1024, chunk)
            for command in commands:
                if chunk is None:
                    self.assertNotIn("--attention-chunk", command)
                else:
                    self.assertEqual(command[command.index("--attention-chunk") + 1], str(chunk))
            if chunk is None:
                self.assertNotIn("attention_chunk", report["native_execution"])
            else:
                self.assertEqual(report["native_execution"]["attention_chunk"], chunk)

    def test_comparison_mlx_attention_is_opt_in_and_labeled(self):
        for enabled in (False, True):
            report, commands, _ = self.run_comparison(1024, mlx_attention=enabled)
            for command in commands:
                if enabled:
                    self.assertEqual(command[command.index("--mlx-attention") + 1], "1")
                else:
                    self.assertNotIn("--mlx-attention", command)
            if enabled:
                self.assertEqual(report["native_execution"]["attention_implementation"],
                                 benchmark.MLX_ATTENTION_IMPLEMENTATION)
            else:
                self.assertNotIn("attention_implementation", report["native_execution"])

    def test_explicit_sequential_reference_is_labeled_without_changing_inputs(self):
        report, _, _ = self.run_comparison(1024, 128, mlx_prefill_chunk_size=1)
        self.assertEqual(report["mlx_prefill_chunk_size"], 1)
        self.assertEqual(report["reference_prefill_mode"], "sequential")
        self.assertEqual(report["evaluated_positions"], 16)

    def test_default_comparison_still_validates_every_prefill_position(self):
        report, commands, tokenizer = self.run_comparison(None)
        self.assertEqual(report["evaluated_positions"], 3 + 15)
        self.assertNotIn("validated_position_range", report)
        self.assertNotIn("--logits-from", commands[1])
        self.assertEqual(len(tokenizer.chat_calls), 1)


def attention_rows(chunks=(0, 128, 256), repeats=3):
    return [dict(engine="zig_fp32", repeat=repeat, attention_chunk=chunk,
                 prefill_seconds=8.0, prefill_gpu_seconds=7.0, decode_seconds=seconds,
                 decode_gpu_seconds=seconds * 0.8, checksum=3, predictions=[1, 2],
                 prefill_reused=True, prefill_samples=1,
                 submissions_per_token=10, dispatches_per_token=20)
            for chunk, seconds in ((c, {0: 1.0, 128: 0.98, 256: 0.9, 512: 0.8, 1024: 0.7}[c])
                                   for c in chunks)
            for repeat in range(repeats)]


class AttentionTuningTests(unittest.TestCase):
    def test_prediction_gated_rank_and_strict_three_percent_threshold(self):
        observed = attention_rows()
        with patch.object(benchmark, "validate_rows", wraps=benchmark.validate_rows) as validate:
            report = tune_attention.summarize_candidates(observed, [0, 128, 256], 3, 2)
        self.assertEqual(validate.call_count, 3)
        self.assertEqual(report["timing_rank"], [256, 128, 0])
        self.assertEqual(report["best_admitted"]["attention_chunk"], 256)
        close = attention_rows((0, 128))
        for row in close:
            row["decode_seconds"] = 103.0 if row["attention_chunk"] == 0 else 100.0
        report = tune_attention.summarize_candidates(close, [0, 128], 3, 2)
        self.assertIsNone(report["best_admitted"])

    def test_checksum_collision_and_changed_repeat_are_not_admitted(self):
        for index in (3, 4):
            observed = attention_rows()
            observed[index]["predictions"] = [2, 1]  # Same checksum, different positions.
            report = tune_attention.summarize_candidates(observed, [0, 128, 256], 3, 2)
            self.assertEqual(report["candidates"]["128"]["status"], "rejected")
            self.assertNotIn(128, report["timing_rank"])
        observed = attention_rows()
        observed[1]["predictions"] = [2, 1]
        with self.assertRaisesRegex(ValueError, "baseline failed"):
            tune_attention.summarize_candidates(observed, [0, 128, 256], 3, 2)

    def test_counts_predictions_and_shared_real_prefill_are_required(self):
        mutations = [lambda r: r.pop(),
                     lambda r: r[3].update(repeat=2),
                     lambda r: r[3].update(predictions=[1]),
                     lambda r: r[3].update(predictions=[True, 2])]
        for mutate in mutations:
            observed = attention_rows((0, 128))
            mutate(observed)
            report = tune_attention.summarize_candidates(observed, [0, 128], 3, 2)
            self.assertEqual(report["candidates"]["128"]["status"], "rejected")
            self.assertIsNone(report["best_admitted"])
        for field, value in (("prefill_reused", False), ("prefill_samples", 3),
                             ("prefill_seconds", 9.0), ("prefill_gpu_seconds", 9.0),
                             ("attention_chunk", 999)):
            observed = attention_rows()
            observed[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                tune_attention.summarize_candidates(observed, [0, 128, 256], 3, 2)

    def test_stream_uses_one_process_and_cleans_up_on_parse_failure(self):
        for invalid in (False, True):
            captured, delivered = [], []

            class Child:
                def __init__(self, argv, **kwargs):
                    captured.append(json.loads(Path(argv[1]).read_text()))
                    self.stdout = io.StringIO("not-json\n" if invalid else
                                             "\n".join(map(json.dumps, attention_rows())))
                    self.returncode = None
                    self.terminated = False

                def wait(self, **kwargs):
                    self.returncode = 0
                    return self.returncode

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated = True

            child = []

            def launch(*args, **kwargs):
                child.append(Child(*args, **kwargs))
                return child[-1]

            with patch.object(tune_attention.subprocess, "Popen", side_effect=launch) as popen:
                if invalid:
                    with self.assertRaises(json.JSONDecodeError):
                        tune_attention.stream_native(Path("not-executed"), {"test": True}, delivered.append)
                    self.assertTrue(child[0].terminated)
                else:
                    tune_attention.stream_native(Path("not-executed"), {"test": True}, delivered.append)
                    self.assertEqual(len(delivered), 9)
                    self.assertFalse(child[0].terminated)
                self.assertEqual(popen.call_count, 1)
                self.assertEqual(captured, [{"test": True}])
                self.assertTrue(child[0].stdout.closed)

    def test_cli_saves_incremental_real_workload_and_one_multicandidate_request(self):
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTokenizer())
        with tempfile.TemporaryDirectory() as temp:
            checkpoint, binary, output = (Path(temp) / name for name in ("checkpoint", "binary", "report.json"))
            checkpoint.mkdir()
            binary.touch()
            argv = ["tune_attention", str(checkpoint), "--binary", str(binary),
                    "--prompt-tokens", "1024", "--decode-tokens", "2", "--output", str(output),
                    "--chunks", "256", "128", "256"]

            def stream(native_binary, payload, on_row):
                self.assertEqual(native_binary, binary.resolve())
                self.assertEqual(payload["attention_candidates"], [0, 256, 128])
                self.assertTrue(payload["reuse_prefill"])
                self.assertTrue(payload["emit_predictions"])
                self.assertEqual(payload["staged_row_tile"], 2)
                expected_prompt, expected_decode = benchmark.workload(FakeTokenizer(), 2, 1024)
                self.assertEqual(payload["prompt"], expected_prompt)
                self.assertEqual(payload["decode"], expected_decode)
                for index, row in enumerate(attention_rows((0, 256, 128))):
                    on_row(row)
                    saved = json.loads(output.read_text())
                    self.assertEqual(saved["status"], "running")
                    self.assertEqual(len(saved["runs"]), index + 1)
                    self.assertIn("Shared real baseline prefill", saved["prefill_note"])

            with patch.dict(sys.modules, {"transformers": transformers}), \
                 patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(tune_attention, "stream_native", side_effect=stream) as native:
                tune_attention.main()
            self.assertEqual(native.call_count, 1)
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "native_decode_tuning_complete")
            self.assertEqual(report["best_admitted"]["attention_chunk"], 256)
            self.assertEqual(report["runs"][0]["submissions_per_token"], 10)
            self.assertEqual(report["shared_baseline_prefill_seconds"], 8.0)
            self.assertEqual(report["mlx_logit_verification"], "not performed")
            self.assertFalse((output.with_suffix(".json.tmp")).exists())

            def malformed_stream(native_binary, payload, on_row):
                row = attention_rows()[0]
                row["decode_seconds"] = float("inf")
                on_row(row)

            with patch.dict(sys.modules, {"transformers": transformers}), \
                 patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(tune_attention, "stream_native", side_effect=malformed_stream), \
                 self.assertRaises(ValueError):
                tune_attention.main()
            failed = json.loads(output.read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertIsNone(failed["best_admitted"])
            self.assertEqual(failed["runs"], [])
            argv[argv.index("--output") + 1] = str(checkpoint / "forbidden.json")
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                tune_attention.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse((checkpoint / "forbidden.json").exists())

    def test_cli_rejects_bad_lengths_and_chunks_before_tokenizer_import(self):
        for flags in ([], ["--prompt-tokens", "0"], ["--prompt-tokens", "-1"],
                      ["--prompt-tokens", "4294967296"], ["--prompt-tokens", "1.5"],
                      ["--prompt-tokens", "1024", "--chunks", "1"],
                      ["--prompt-tokens", "1024", "--repeats", "0"]):
            with patch.object(sys, "argv", ["tune_attention", "unused", *flags]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                tune_attention.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
