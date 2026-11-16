#!/usr/bin/env python3
"""CPU-only tests: python3 -m unittest discover -s tools -p test_bf16_kv_metrics.py.

No MLX, GPU, native binary, real checkpoint, server, build, or downloads required.
The CLI protocol tests replace both the tokenizer and native subprocesses.
"""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bf16_kv_comparison", TOOL_DIR / "compare_bf16_kv.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class LogitMetricsTests(unittest.TestCase):
    def test_identical_arrays_have_zero_errors_and_kl(self):
        values = np.array([[1, 4, 2], [0, 0, 0]], dtype=np.float32)
        result = comparison.compare_logits(values, values, expected_shape=(2, 3))
        self.assertEqual(result["status"], "measured")
        for key in ("max_absolute_logit_error", "rms_logit_error", "relative_l2_logit_error",
                    "mean_softmax_kl_nats"):
            self.assertEqual(result[key], 0.0)
        self.assertEqual(result["top1_agreement_count"], 2)
        self.assertEqual(result["top1_agreement_fraction"], 1.0)
        self.assertEqual([row["fp32_top1_margin"] for row in result["rows"]], [2.0, 0.0])
        self.assertEqual(result["fp32"]["finite_count"], 6)
        self.assertEqual(result["bf16"]["shape"], [2, 3])
        json.dumps(result, allow_nan=False)

    def test_known_errors_top1_margins_and_kl_direction(self):
        reference = [[2, 0], [1, 3]]
        candidate = [[0, 2], [1, 4]]
        result = comparison.compare_logits(reference, candidate)
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["max_absolute_logit_error"], 2)
        self.assertEqual(result["rms_logit_error"], 1.5)
        self.assertAlmostEqual(result["relative_l2_logit_error"], 3 / math.sqrt(14))
        self.assertEqual(result["top1_agreement_count"], 1)
        self.assertEqual([row["top1_agrees"] for row in result["rows"]], [False, True])
        self.assertEqual([row["fp32_top1"] for row in result["rows"]], [0, 1])
        self.assertEqual([row["bf16_top1"] for row in result["rows"]], [1, 1])
        self.assertEqual([row["bf16_top1_margin"] for row in result["rows"]], [2, 3])
        first_kl = 2 * math.tanh(1)
        p = 1 / (1 + math.exp(2))
        q = 1 / (1 + math.exp(3))
        second_kl = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        self.assertAlmostEqual(result["rows"][0]["softmax_kl_nats"], first_kl)
        self.assertAlmostEqual(result["mean_softmax_kl_nats"], (first_kl + second_kl) / 2)

    def test_constant_shift_has_zero_softmax_kl_but_nonzero_logit_error(self):
        reference = np.array([[1, 2, -3], [10, -4, 2]], dtype=np.float64)
        result = comparison.compare_logits(reference, reference + 10000)
        self.assertEqual(result["max_absolute_logit_error"], 10000)
        self.assertEqual(result["mean_softmax_kl_nats"], 0)
        self.assertEqual(result["top1_agreement_count"], 2)

    def test_extreme_logits_use_stable_float64_log_softmax(self):
        result = comparison.compare_logits([[10000, -10000]], [[-10000, 10000]])
        self.assertEqual(result["status"], "measured")
        self.assertAlmostEqual(result["mean_softmax_kl_nats"], 20000)
        json.dumps(result, allow_nan=False)

    def test_zero_reference_norm_is_explicit_not_infinity(self):
        same = comparison.compare_logits([[0, 0]], [[0, 0]])
        self.assertEqual(same["relative_l2_logit_error"], 0)
        changed = comparison.compare_logits([[0, 0]], [[1, 0]])
        self.assertEqual(changed["status"], "measured")
        self.assertIsNone(changed["relative_l2_logit_error"])
        json.dumps(changed, allow_nan=False)

    def test_nonfinite_fails_with_counts_without_nan_in_report(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for side in (0, 1):
                arrays = [[[1, 2]], [[1, 2]]]
                arrays[side] = [[1, bad]]
                with self.subTest(value=bad, side=side):
                    result = comparison.compare_logits(*arrays)
                    self.assertEqual(result["status"], "failed")
                    key = "fp32" if side == 0 else "bf16"
                    self.assertEqual(result[key]["finite_count"], 1)
                    self.assertEqual(result[key]["nonfinite_count"], 1)
                    self.assertIn(key + "_nonfinite_logits", result["issues"])
                    self.assertNotIn("rows", result)
                    json.dumps(result, allow_nan=False)

    def test_shapes_counts_and_ragged_input_fail(self):
        cases = [([], [], None), ([1, 2], [1, 2], None),
                 ([[1]], [[1]], None), ([[1, 2]], [[1, 2], [1, 2]], None),
                 ([[1, 2]], [[1, 2]], (2, 2)),
                 ([[1, 2], [3]], [[1, 2]], None),
                 ([[1, 2]], [[1, 2, 3]], None)]
        for reference, candidate, expected in cases:
            with self.subTest(reference=reference, candidate=candidate, expected=expected):
                result = comparison.compare_logits(reference, candidate, expected_shape=expected)
                self.assertEqual(result["status"], "failed")
                self.assertTrue(result["issues"])
                json.dumps(result, allow_nan=False)

    def test_large_finite_drift_is_measured_not_fake_accuracy_pass(self):
        result = comparison.compare_logits([[1, 0]], [[-100, 100]])
        self.assertEqual(result["status"], "measured")
        self.assertEqual(result["top1_agreement_count"], 0)
        self.assertGreater(result["mean_softmax_kl_nats"], 100)
        self.assertNotIn("passed", result)


class AllocationAndProtocolTests(unittest.TestCase):
    def test_default_16k_allocation_includes_both_caches_and_sliding_fp32(self):
        result = comparison.analytic_kv_allocation({}, 16384)
        global_elements = 2 * 16384 * 14 * 4 * 128
        sliding_bytes = 2 * 513 * 35 * 4 * 128 * 4
        self.assertEqual(result["global_layers"], 14)
        self.assertEqual(result["sliding_layers"], 35)
        self.assertEqual(result["fp32"]["total_bytes"], global_elements * 4 + sliding_bytes)
        self.assertEqual(result["hybrid_bf16"]["total_bytes"], global_elements * 2 + sliding_bytes)
        self.assertEqual(result["hybrid_bf16"]["sliding_bytes"], sliding_bytes)
        self.assertIn("not RSS", result["scope"])

    def test_local_config_and_short_context_change_capacities(self):
        config = dict(num_hidden_layers=2, layer_types=["full_attention", "sliding_attention"],
                      num_key_value_heads=2, head_dim=8, sliding_window=7, vocab_size=4)
        result = comparison.analytic_kv_allocation(config, 3)
        self.assertEqual(result["sliding_capacity_per_layer"], 3)
        self.assertEqual(result["fp32"]["total_bytes"], 2 * 3 * 2 * 8 * 4 * 2)
        self.assertEqual(result["saved_bytes"], 2 * 3 * 2 * 8 * 2)
        for bad in (dict(config, layer_types=["unknown", "sliding_attention"]),
                    dict(config, num_key_value_heads=0), dict(config, use_sliding_window=False)):
            with self.assertRaises(ValueError):
                comparison.analytic_kv_allocation(bad, 3)

    def test_output_parser_keeps_full_vocabulary_rows_and_nonfinite_for_diagnosis(self):
        self.assertEqual(comparison.parse_logits("[1,2,3]\n[4,5,6]\n"), [[1, 2, 3], [4, 5, 6]])
        self.assertTrue(math.isnan(comparison.parse_logits("[NaN, 1]\n")[0][0]))
        for value in ('{"logits":[1,2]}', '[true, 2]', '["secret", 2]', 'log text'):
            with self.assertRaises((ValueError, TypeError)):
                comparison.parse_logits(value)

    def run_mock_cli(self, *, native_output=None, native_error=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "private-checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(json.dumps({"vocab_size": 3}))
            before = (checkpoint / "config.json").read_bytes()
            binary = root / "fake-flint"
            binary.write_bytes(b"not executable; mocked native child only")
            output = root / "report.json"
            calls = []
            fake_tokenizer = types.SimpleNamespace(encode=lambda text, **kwargs: [0, 1, 2] * 4)
            auto = types.SimpleNamespace(from_pretrained=mock.Mock(return_value=fake_tokenizer))

            def fake_run(command, **kwargs):
                tokens = json.loads(Path(command[command.index("--tokens-file") + 1]).read_text())
                calls.append((command, tokens, kwargs))
                if native_error is not None:
                    raise native_error
                stdout = native_output if native_output is not None else "[1, 2, 3]\n" * 3
                return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="private diagnostic")

            with mock.patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=auto)}), \
                    mock.patch.object(sys, "path", [str(TOOL_DIR)] + sys.path), \
                    mock.patch.object(comparison.subprocess, "run", side_effect=fake_run), \
                    mock.patch.dict(comparison.os.environ, {}, clear=False), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = comparison.main([str(checkpoint), "--binary", str(binary),
                                        "--prompt-tokens", "5", "--decode-tokens", "2",
                                        "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(before, (checkpoint / "config.json").read_bytes())
            self.assertEqual(sorted(path.name for path in checkpoint.iterdir()), ["config.json"])
            self.assertNotIn(str(root), output.read_text())
            self.assertNotIn("private diagnostic", output.read_text())
            self.assertNotIn("private-checkpoint", output.read_text())
            self.assertTrue(auto.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertFalse(auto.from_pretrained.call_args.kwargs["trust_remote_code"])
            return code, report, calls

    def test_mocked_cli_replays_identical_real_prefix_then_bf16_and_sanitizes(self):
        code, report, calls = self.run_mock_cli()
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "measured")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], [0, 1, 2, 0, 1, 0, 1])
        self.assertEqual(calls[0][1], calls[1][1])
        baseline, candidate = calls[0][0], calls[1][0]
        self.assertNotIn("--kv-bf16", baseline)
        self.assertEqual(candidate, baseline + ["--kv-bf16", "1"])
        self.assertEqual(baseline[1], "eval")
        for flag, value in (("--context", "7"), ("--logits-from", "4"),
                            ("--mlx-attention", "1"), ("--staged-row-tile", "2")):
            self.assertEqual(baseline[baseline.index(flag) + 1], value)
        self.assertEqual(report["positions"]["count"], 3)
        self.assertEqual(report["positions"]["end_inclusive"], 6)
        self.assertEqual(report["binary"]["name"], "fake-flint")
        self.assertEqual(len(report["input_sha256"]), 64)
        self.assertEqual(len(report["binary"]["sha256"]), 64)
        self.assertTrue(all(call[2]["timeout"] >= 180 for call in calls))
        self.assertTrue(all(call[2]["capture_output"] and call[2]["check"] for call in calls))
        self.assertFalse(Path(baseline[baseline.index("--tokens-file") + 1]).exists())

    def test_mocked_cli_fails_shape_nonfinite_and_native_errors_without_leaks(self):
        for stdout in ("[1,2,3]\n", "[NaN,2,3]\n" * 3):
            code, report, _ = self.run_mock_cli(native_output=stdout)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["logits"]["status"], "failed")
        for error, reason in ((subprocess.CalledProcessError(2, ["private"], stderr="secret"), "native_nonzero_exit"),
                              (subprocess.TimeoutExpired(["private"], 180, output="secret"), "native_timeout")):
            code, report, calls = self.run_mock_cli(native_error=error)
            self.assertEqual(code, 1)
            self.assertEqual(report["failure"]["reason"], reason)
            self.assertEqual(report["failure"]["phase"], "fp32_eval")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("secret", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
