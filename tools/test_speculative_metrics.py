#!/usr/bin/env python3
"""CPU-only protocol tests: python3 -m unittest discover -s tools -p test_speculative_metrics.py.

Synthetic reports, a fake tokenizer, and mocked native subprocesses only.
No GPU, native build, checkpoint, server, network, or tokenizer install needed.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

TOOL_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("speculative_metrics", TOOL_DIR / "check_speculative.py")
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def config(**overrides):
    result = dict(prompt_tokens=[10, 20, 30, 10, 20], max_tokens=64,
                  draft_tokens=4, draft_candidates=[2, 4, 8], query_groups=[2], expert_tiles=[0], repeats=3, kv_bf16=False, weight_reuse=False, mlp_weight_reuse=False, compare_mlp_weight_reuse=False)
    result.update(overrides)
    return result


def native_report(workload, *, count=None, eos=False):
    """Independent synthetic schema fixture; order is a left rotation per round."""
    candidates = workload.get("draft_candidates") or [workload["draft_tokens"]]
    mlp_modes = [False, True] if workload.get("compare_mlp_weight_reuse", False) else [workload.get("mlp_weight_reuse", False)]
    combinations = list(itertools.product(candidates, workload.get("query_groups", [2]), workload.get("expert_tiles", [0]), mlp_modes))
    count = workload["max_tokens"] if count is None else count
    forced = min(32, count)
    result = dict(
        schema_version=3, engine="native_matched_greedy", exact_tokens=True,
        exact_final_logit_bits=True, exact_batch_row_bits=True,
        forced_rows_correctness_only=True, speed_workload="free_running_history_draft",
        reference_untimed=True, warmup_timed=False,
        warmup_max_tokens=min(32, workload["max_tokens"]), timing_order="rotated_by_round",
        actual_prefix_tokens=len(workload["prompt_tokens"]), max_tokens=workload["max_tokens"],
        draft_candidates=candidates[:], query_groups=workload.get("query_groups", [2]),
        expert_tiles=workload.get("expert_tiles", [0]), prefill_seconds=1.25, prefill_samples=1,
        prefix_restore_timed=False, forced_validation_rows=forced,
        partial_commit_checks=(0 if forced == 1 else (1 if forced == 2 else 2)) * len(combinations),
        wrapped_sliding_layers=0, kv_allocated_bytes=4096,
        kv_storage="bf16_global_fp32_sliding" if workload["kv_bf16"] else "fp32",
        weight_reuse=workload.get("weight_reuse", False), mlp_weight_reuse=workload.get("mlp_weight_reuse", False),
        compare_mlp_weight_reuse=workload.get("compare_mlp_weight_reuse", False), weight_reuse_scope="speculative_modes_only",
        ordinary=[], speculative=[])
    modes = [(0, 2, 0, False)] + combinations
    for round_number in range(workload["repeats"]):
        rotation = round_number % len(modes)
        order = modes[rotation:] + modes[:rotation]
        for index, (draft, query, tile, mlp_reuse) in enumerate(order):
            rate = 80 + 20 * round_number + 6 * draft + query - 2 + tile + 3 * mlp_reuse
            trial = dict(draft_tokens=draft, query_group=query, expert_tile=tile, mlp_weight_reuse=mlp_reuse, round=round_number, order=index,
                         decode_seconds=count / rate, gpu_seconds=0.75 * count / rate, tokens_per_second=rate,
                         committed_tokens=count, eos=eos, batches=0, proposed=0, accepted=0)
            result["ordinary" if draft == 0 else "speculative"].append(trial)
    return result


class ReportTests(unittest.TestCase):
    def test_v3_rounds_include_rotated_baseline_and_every_candidate(self):
        workload = config(repeats=6)
        report = native_report(workload)
        metrics.validate_report(report, workload)
        self.assertEqual(len(report["ordinary"]), 6)
        self.assertEqual(len(report["speculative"]), 18)
        trials = report["ordinary"] + report["speculative"]
        for round_number in range(6):
            ordered = sorted((trial for trial in trials if trial["round"] == round_number),
                             key=lambda trial: trial["order"])
            modes = [0, 2, 4, 8]
            rotation = round_number % 4
            self.assertEqual([trial["draft_tokens"] for trial in ordered],
                             modes[rotation:] + modes[:rotation])
        json.dumps(report, allow_nan=False)

    def test_legacy_input_fallback_and_candidate_one(self):
        for candidates in ([], [1], [8, 2, 4]):
            workload = config(draft_candidates=candidates)
            metrics.validate_report(native_report(workload), workload)
        workload.pop("draft_candidates")
        metrics.validate_report(native_report(workload), workload)

    def test_weight_reuse_requires_exact_boolean_config_match_and_candidate_only_scope(self):
        for enabled in (False, True):
            workload = config(weight_reuse=enabled)
            metrics.validate_report(native_report(workload), workload)
            for bad in (not enabled, 0, 1, None, "false", [], {}):
                report = native_report(workload)
                report["weight_reuse"] = bad
                with self.subTest(enabled=enabled, bad=bad), self.assertRaises(ValueError):
                    metrics.validate_report(report, workload)
            for scope in ("all_modes", "ordinary", "private", False):
                report = native_report(workload)
                report["weight_reuse_scope"] = scope
                with self.subTest(scope=scope), self.assertRaises(ValueError):
                    metrics.validate_report(report, workload)
        for bad in (0, 1, None, "true"):
            workload = config(weight_reuse=bad)
            with self.subTest(config=bad), self.assertRaises(ValueError):
                metrics.validate_report(native_report(workload), workload)
        workload = config()
        workload.pop("weight_reuse")
        metrics.validate_report(native_report(workload), workload)

    def test_mlp_flags_are_strict_booleans_with_matching_top_level_request(self):
        for key in ("mlp_weight_reuse", "compare_mlp_weight_reuse"):
            for enabled in (False, True):
                workload = config(**{key: enabled})
                metrics.validate_report(native_report(workload), workload)
                for bad in (not enabled, 0, 1, None, "false", [], {}):
                    report = native_report(workload)
                    report[key] = bad
                    with self.subTest(key=key, enabled=enabled, bad=bad), self.assertRaises(ValueError):
                        metrics.validate_report(report, workload)
            for bad in (0, 1, None, "true"):
                workload = config(**{key: bad})
                report = native_report(config())
                with self.subTest(config=key, bad=bad), self.assertRaises(ValueError):
                    metrics.validate_report(report, workload)
        workload = config()
        workload.pop("mlp_weight_reuse")
        workload.pop("compare_mlp_weight_reuse")
        metrics.validate_report(native_report(workload), workload)
        for kind in ("ordinary", "speculative"):
            for bad in (0, 1, None, "false"):
                with self.subTest(kind=kind, bad=bad):
                    self.assert_mutation_rejected(lambda report: report[kind][0].update(mlp_weight_reuse=bad))
        self.assert_mutation_rejected(lambda report: report["ordinary"][0].update(mlp_weight_reuse=True))
        self.assert_mutation_rejected(lambda report: report["speculative"][0].update(mlp_weight_reuse=True))

    def test_short_budgets_eos_and_bf16_keep_row_rollback_gates(self):
        for count in (1, 2, 3, 8, 16, 31, 32, 33, 64):
            for eos in (False, True):
                workload = config(max_tokens=count, kv_bf16=True)
                metrics.validate_report(native_report(workload, eos=eos), workload)
        workload = config()
        metrics.validate_report(native_report(workload, count=2, eos=True), workload)

    def assert_mutation_rejected(self, mutate):
        workload = config()
        report = native_report(workload)
        mutate(report)
        with self.assertRaises(ValueError):
            metrics.validate_report(report, workload)

    def test_schema_is_closed_and_v1_is_not_silently_admitted(self):
        mutations = [lambda report: report.update(schema_version=1),
                     lambda report: report.update(schema_version=True),
                     lambda report: report.update(private_diagnostic="private"),
                     lambda report: report.pop("reference_untimed"),
                     lambda report: report.update(ordinary=report["ordinary"][0]),
                     lambda report: report["speculative"][0].update(logits=[1, 2]),
                     lambda report: report.update(draft_candidates=[True, 4, 8]),
                     lambda report: report.update(kv_storage="unexpected"),
                     lambda report: report.update(actual_prefix_tokens=6)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.assert_mutation_rejected(mutate)

    def test_every_correctness_and_untimed_protocol_gate_is_required(self):
        for key, bad in (("exact_tokens", False), ("exact_final_logit_bits", False),
                         ("exact_batch_row_bits", False), ("forced_rows_correctness_only", False),
                         ("reference_untimed", False), ("warmup_timed", True),
                         ("prefix_restore_timed", True), ("prefill_samples", 2),
                         ("warmup_max_tokens", 31), ("timing_order", "grouped"),
                         ("speed_workload", "forced_future"), ("forced_validation_rows", 7),
                         ("partial_commit_checks", 0)):
            with self.subTest(key=key):
                self.assert_mutation_rejected(lambda report: report.update({key: bad}))

    def test_missing_duplicate_unrotated_and_mismatched_trials_fail(self):
        mutations = [lambda report: report["ordinary"].pop(),
                     lambda report: report["speculative"].pop(),
                     lambda report: report["ordinary"].__setitem__(1, copy.deepcopy(report["ordinary"][0])),
                     lambda report: report["speculative"][0].update(round=99),
                     lambda report: report["ordinary"][1].update(order=0),
                     lambda report: report["speculative"][0].update(draft_tokens=3),
                     lambda report: report["speculative"][0].update(committed_tokens=63),
                     lambda report: report["speculative"][0].update(eos=True),
                     lambda report: report["speculative"][0].update(round=True)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.assert_mutation_rejected(mutate)

    def test_gpu_command_time_is_required_finite_nonnegative_and_not_bounded_by_wall(self):
        workload = config()
        for value in (0, 0.0, 0.01, 1000.0):
            report = native_report(workload)
            for trial in report["ordinary"] + report["speculative"]:
                trial["gpu_seconds"] = value
            metrics.validate_report(report, workload)
        for kind in ("ordinary", "speculative"):
            for value in (-0.01, float("nan"), float("inf"), -float("inf"), True, False, None, "private"):
                with self.subTest(kind=kind, value=value):
                    self.assert_mutation_rejected(lambda report: report[kind][0].update(gpu_seconds=value))
            with self.subTest(missing=kind):
                self.assert_mutation_rejected(lambda report: report[kind][0].pop("gpu_seconds"))

    def test_numeric_and_proposal_accounting_fail_closed(self):
        for key, bad in (("decode_seconds", 0), ("decode_seconds", float("nan")),
                         ("tokens_per_second", float("inf")), ("tokens_per_second", "private"),
                         ("tokens_per_second", 999), ("committed_tokens", True),
                         ("eos", 1), ("accepted", 1), ("batches", 1),
                         ("proposed", -1), ("order", -1)):
            with self.subTest(key=key, bad=bad):
                self.assert_mutation_rejected(lambda report: report["speculative"][0].update({key: bad}))
        self.assert_mutation_rejected(lambda report: report["ordinary"][0].update(
            batches=1, proposed=2, accepted=1))
        workload = config()
        report = native_report(workload)
        report["speculative"][0].update(batches=1, proposed=2, accepted=1)
        metrics.validate_report(report, workload)


class CliTests(unittest.TestCase):
    def run_cli(self, extra=(), *, transform=None, raw_stdout=None, returncode=0,
                tokenizer_error=None, mutate_binary=False):
        """Run the real wrapper and validator, intercepting only external dependencies."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "private-binary"
            binary.write_bytes(b"mocked child; never executed")
            checkpoint = root / "private-checkpoint"
            checkpoint.mkdir()
            output = root / "report.json"
            calls = []
            auto = types.SimpleNamespace(from_pretrained=mock.Mock())
            auto.from_pretrained.return_value = types.SimpleNamespace(
                encode=mock.Mock(return_value=[10, 20, 30]))
            def fake_tokenizer(*args, **kwargs):
                print("private tokenizer warning /private/token-secret")
                print("private tokenizer diagnostic /private/token-secret", file=sys.stderr)
                if tokenizer_error is not None:
                    raise tokenizer_error
                return auto.from_pretrained.return_value

            auto.from_pretrained.side_effect = fake_tokenizer

            def fake_run(command, **kwargs):
                input_path = Path(command[1])
                workload = json.loads(input_path.read_text())
                calls.append((command, workload, kwargs))
                report = native_report(workload)
                if transform:
                    transform(report)
                if mutate_binary:
                    binary.write_bytes(b"changed")
                return subprocess.CompletedProcess(command, returncode,
                                                   stdout=json.dumps(report) if raw_stdout is None else raw_stdout,
                                                   stderr="private child diagnostics /private/token-secret")

            stdout, stderr = io.StringIO(), io.StringIO()
            argv = ["--checkpoint", str(checkpoint), "--binary", str(binary),
                    "--prompt-tokens", "5", "--max-tokens", "64", "--repeats", "3",
                    "--output", str(output)] + list(extra)
            with mock.patch.dict(sys.modules, {"transformers": types.SimpleNamespace(AutoTokenizer=auto)}), \
                    mock.patch.object(metrics.subprocess, "run", side_effect=fake_run), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = metrics.cli(argv)
            encoded = stdout.getvalue()
            report = json.loads(encoded) if encoded else None
            for text in (encoded, stderr.getvalue(), output.read_text() if output.exists() else ""):
                for private in (str(root), "private-binary", "private-checkpoint", "private child",
                                "token-secret", "/private"):
                    self.assertNotIn(private, text)
            for command, _, _ in calls:
                self.assertFalse(Path(command[1]).exists(), "temporary config must be removed")
            self.assertEqual(list(checkpoint.iterdir()), [])
            self.assertEqual(output.exists(), code == 0)
            if code == 0:
                self.assertEqual(output.read_text(), encoded)
            return code, report, calls, auto, stderr.getvalue()

    def test_cli_multi_candidate_medians_use_repeated_baseline(self):
        code, report, calls, auto, stderr = self.run_cli(["--draft-candidates", "2,4,8"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(report["schema_version"], 3)
        self.assertEqual(report["ordinary_samples"], 3)
        for trial in report["ordinary"] + report["speculative"]:
            self.assertGreaterEqual(trial["gpu_seconds"], 0)
            self.assertGreater(trial["decode_seconds"], 0)
        self.assertIn("approximate unprofiled residual, not pure CPU time", report["speed_note"])
        self.assertEqual(report["ordinary_tokens_per_second_median"], 100)
        self.assertEqual(report["speculative_tokens_per_second_median"], {"d2_q2_t0": 112, "d4_q2_t0": 124, "d8_q2_t0": 148})
        self.assertEqual(report["speedup_median"], {"d2_q2_t0": 1.12, "d4_q2_t0": 1.24, "d8_q2_t0": 1.48})
        self.assertEqual(len(calls), 1)
        _, workload, kwargs = calls[0]
        self.assertEqual(workload["prompt_tokens"], [10, 20, 30, 10, 20])
        self.assertEqual(workload["draft_candidates"], [2, 4, 8])
        self.assertEqual(kwargs["stdout"], subprocess.PIPE)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertFalse(kwargs["check"])
        workload = {key: value for key, value in workload.items() if key != "checkpoint"}
        self.assertEqual(report["input_sha256"], hashlib.sha256(metrics.canonical_bytes(workload)).hexdigest())
        self.assertEqual(len(report["binary_sha256"]), 64)
        self.assertEqual(auto.from_pretrained.call_args.kwargs,
                         dict(local_files_only=True, trust_remote_code=False, fix_mistral_regex=False))
        auto.from_pretrained.return_value.encode.assert_called_once_with(metrics.CODE, add_special_tokens=False)

    def test_weight_reuse_cli_hash_labels_and_repeated_baseline_protocol(self):
        extra = ["--draft-candidates", "8", "--query-groups", "2", "--expert-tiles", "0"]
        off_code, off, off_calls, _, _ = self.run_cli(extra)
        on_code, on, on_calls, _, _ = self.run_cli(extra + ["--weight-reuse"])
        self.assertEqual((off_code, on_code), (0, 0))
        self.assertIs(off["weight_reuse"], False)
        self.assertIs(on["weight_reuse"], True)
        self.assertIs(off_calls[0][1]["weight_reuse"], False)
        self.assertIs(on_calls[0][1]["weight_reuse"], True)
        self.assertEqual(on["weight_reuse_scope"], "speculative_modes_only")
        self.assertEqual(set(on["speculative_tokens_per_second_median"]), {"d8_q2_t0"})
        self.assertNotEqual(off["input_sha256"], on["input_sha256"])
        for report, calls in ((off, off_calls), (on, on_calls)):
            workload = {key: value for key, value in calls[0][1].items() if key != "checkpoint"}
            self.assertEqual(report["input_sha256"], hashlib.sha256(metrics.canonical_bytes(workload)).hexdigest())
            self.assertEqual(report["ordinary_samples"], 3)
            self.assertIs(report["reference_untimed"], True)
            self.assertIs(report["warmup_timed"], False)
            self.assertIn("ordinary reference and baselines always disable it", report["speed_note"])
        # The wrapper keeps all repeated baseline samples and their unchanged
        # zero-draft/default-query/default-tile identities when reuse is enabled.
        # This is a protocol check with synthetic native results, not GPU proof.
        self.assertEqual(off["ordinary"], on["ordinary"])
        self.assertEqual([trial["round"] for trial in on["ordinary"]], [0, 1, 2])
        for trial in on["ordinary"]:
            self.assertEqual((trial["draft_tokens"], trial["query_group"], trial["expert_tile"]), (0, 2, 0))
        for transform in (lambda report: report.update(weight_reuse=False),
                          lambda report: report.update(weight_reuse=1),
                          lambda report: report.update(weight_reuse_scope="private")):
            code, report, _, _, stderr = self.run_cli(extra + ["--weight-reuse"], transform=transform)
            self.assertEqual(code, 1)
            self.assertIsNone(report)
            self.assertIn("no speed report admitted", stderr)

    def test_independent_mlp_reuse_cli_and_hash_keep_ordinary_disabled(self):
        hashes = set()
        ordinary = None
        for weight, mlp in itertools.product((False, True), repeat=2):
            extra = ["--draft-candidates", "8"]
            if weight:
                extra.append("--weight-reuse")
            if mlp:
                extra.append("--mlp-weight-reuse")
            code, report, calls, _, _ = self.run_cli(extra)
            self.assertEqual(code, 0)
            for key, value in (("weight_reuse", weight), ("mlp_weight_reuse", mlp)):
                self.assertIs(report[key], value)
                self.assertIs(calls[0][1][key], value)
            workload = {key: value for key, value in calls[0][1].items() if key != "checkpoint"}
            self.assertEqual(report["input_sha256"], hashlib.sha256(metrics.canonical_bytes(workload)).hexdigest())
            hashes.add(report["input_sha256"])
            self.assertEqual(set(report["speculative_tokens_per_second_median"]), {"d8_q2_t0"})
            for trial in report["ordinary"]:
                self.assertIs(trial["mlp_weight_reuse"], False)
            for trial in report["speculative"]:
                self.assertIs(trial["mlp_weight_reuse"], mlp)
            if ordinary is None:
                ordinary = report["ordinary"]
            self.assertEqual(ordinary, report["ordinary"])
        self.assertEqual(len(hashes), 4)

    def test_compare_mlp_reuse_rotates_both_settings_from_one_prefix(self):
        for requested in (False, True):
            extra = ["--draft-candidates", "8", "--query-groups", "2", "--expert-tiles", "0",
                     "--weight-reuse", "--compare-mlp-weight-reuse"]
            if requested:
                extra.append("--mlp-weight-reuse")
            code, report, calls, _, _ = self.run_cli(extra)
            self.assertEqual(code, 0)
            self.assertEqual(len(calls), 1)
            self.assertIs(report["mlp_weight_reuse"], requested)
            self.assertIs(report["compare_mlp_weight_reuse"], True)
            self.assertEqual(report["prefill_samples"], 1)
            self.assertEqual(len(report["ordinary"]), 3)
            self.assertEqual(len(report["speculative"]), 6)
            self.assertEqual(report["partial_commit_checks"], 4)
            self.assertEqual(report["speculative_tokens_per_second_median"], {"d8_q2_t0_m0": 148, "d8_q2_t0_m1": 151})
            trials = report["ordinary"] + report["speculative"]
            for round_number in range(3):
                ordered = sorted((trial for trial in trials if trial["round"] == round_number), key=lambda trial: trial["order"])
                modes = [(0, False), (8, False), (8, True)]
                self.assertEqual([(trial["draft_tokens"], trial["mlp_weight_reuse"]) for trial in ordered],
                                 modes[round_number:] + modes[:round_number])
            native = native_report(calls[0][1])
            native["speculative"][0]["mlp_weight_reuse"] = not native["speculative"][0]["mlp_weight_reuse"]
            with self.assertRaises(ValueError):
                metrics.validate_report(native, calls[0][1])
        for transform in (lambda report: report.update(mlp_weight_reuse=True),
                          lambda report: report.update(compare_mlp_weight_reuse=False),
                          lambda report: report["ordinary"][0].update(mlp_weight_reuse=True)):
            code, report, _, _, stderr = self.run_cli(
                ["--draft-candidates", "8", "--compare-mlp-weight-reuse"], transform=transform)
            self.assertEqual(code, 1)
            self.assertIsNone(report)
            self.assertIn("no speed report admitted", stderr)

    def test_comparison_counts_toward_the_sixteen_mode_limit(self):
        boundary = config(draft_candidates=[2, 4], query_groups=[2, 4], expert_tiles=[0, 1], compare_mlp_weight_reuse=True)
        self.assertEqual(len(metrics.requested_modes(boundary)), 16)
        metrics.validate_report(native_report(boundary), boundary)
        with self.assertRaises(ValueError):
            metrics.requested_modes(config(draft_candidates=[2, 4, 8], query_groups=[2, 4], expert_tiles=[0, 1], compare_mlp_weight_reuse=True))
        stderr = io.StringIO()
        with mock.patch.object(metrics.subprocess, "run") as run, contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exited:
            metrics.cli(["--checkpoint", "/private/token-secret", "--draft-candidates", "2,4,8", "--query-groups", "2,4",
                         "--expert-tiles", "0,1", "--compare-mlp-weight-reuse"])
        self.assertEqual(exited.exception.code, 2)
        run.assert_not_called()
        self.assertNotIn("private", stderr.getvalue())

    def test_cli_default_and_explicit_single_draft_compatibility(self):
        for extra, expected in (([], 4), (["--draft-tokens", "2"], 2),
                                (["--draft-tokens", "2", "--draft-candidates", "8"], 8)):
            with self.subTest(extra=extra):
                code, report, _, _, _ = self.run_cli(extra)
                self.assertEqual(code, 0)
                self.assertEqual(report["draft_candidates"], [expected])
                self.assertEqual(set(report["speculative_tokens_per_second_median"]), {"d{}_q2_t0".format(expected)})

    def test_native_failure_parity_failure_unknown_payload_and_binary_mutation_are_sanitized(self):
        cases = [dict(returncode=9), dict(raw_stdout="/private/token-secret: native error"),
                 dict(transform=lambda report: report.update(exact_tokens=False)),
                 dict(transform=lambda report: report.update(private="/private/token-secret")),
                 dict(transform=lambda report: report["speculative"][0].update(eos=True)),
                 dict(transform=lambda report: report["ordinary"][0].update(gpu_seconds=-1)),
                 dict(transform=lambda report: report["speculative"][0].update(gpu_seconds="/private/token-secret")),
                 dict(mutate_binary=True),
                 dict(tokenizer_error=RuntimeError("/private/token-secret tokenizer error")),
                 dict(tokenizer_error=AttributeError("/private/token-secret tokenizer error"))]
        for case in cases:
            with self.subTest(case=case):
                code, report, _, _, stderr = self.run_cli(**case)
                self.assertEqual(code, 1)
                self.assertIsNone(report)
                self.assertIn("no speed report admitted", stderr)

    def test_duplicate_json_nonfinite_and_extra_stdout_are_rejected(self):
        valid = json.dumps(native_report(config(draft_candidates=[])))
        for raw in (valid.replace('"schema_version": 3', '"schema_version": 3, "schema_version": 3'),
                    valid.replace('"prefill_seconds": 1.25', '"prefill_seconds": NaN'),
                    valid + "\n/private/token-secret", "[]", "null"):
            with self.subTest(raw=raw[:30]):
                code, report, _, _, _ = self.run_cli(raw_stdout=raw)
                self.assertEqual(code, 1)
                self.assertIsNone(report)

    def test_joint_sweep_six_modes_rounds_and_medians(self):
        code, report, calls, _, _ = self.run_cli([
            "--draft-candidates", "8", "--query-groups", "2,4,8", "--expert-tiles", "0,1"])
        self.assertEqual(code, 0)
        self.assertEqual(len(report["ordinary"]), 3)
        self.assertEqual(len(report["speculative"]), 18)
        self.assertEqual(report["partial_commit_checks"], 12)
        expected = ["d8_q{}_t{}".format(q, t) for q in (2, 4, 8) for t in (0, 1)]
        self.assertEqual(set(report["speculative_tokens_per_second_median"]), set(expected))
        metrics.validate_report({key: value for key, value in report.items()
                                 if key in native_report(calls[0][1])}, calls[0][1])
        for key in ("query_group", "expert_tile"):
            bad = native_report(calls[0][1])
            bad["ordinary"][0][key] = 4
            with self.assertRaises(ValueError):
                metrics.validate_report(bad, calls[0][1])

    def test_joint_axes_empty_duplicate_invalid_and_over_budget_fail_before_native(self):
        for overrides in (dict(query_groups=[]), dict(expert_tiles=[]),
                          dict(query_groups=[2, 2]), dict(expert_tiles=[0, 0]),
                          dict(query_groups=[1]), dict(expert_tiles=[3]),
                          dict(query_groups=[True]), dict(expert_tiles=[False]),
                          dict(query_groups=[2, 4, 8], expert_tiles=[0, 1])):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                metrics.requested_modes(config(**overrides))
        for argv in (["--query-groups", ""], ["--expert-tiles", ""],
                     ["--query-groups", "2,2"], ["--expert-tiles", "0,0"],
                     ["--query-groups", "3"], ["--expert-tiles", "3"],
                     ["--draft-candidates", "2,4,8", "--query-groups", "2,4,8", "--expert-tiles", "0,1"]):
            stderr = io.StringIO()
            with self.subTest(argv=argv), mock.patch.object(metrics.subprocess, "run") as run, \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exited:
                metrics.cli(["--checkpoint", "/private/token-secret"] + argv)
            self.assertEqual(exited.exception.code, 2)
            run.assert_not_called()
            self.assertNotIn("private", stderr.getvalue())
        self.assertEqual(metrics.query_groups("8,2,4"), [8, 2, 4])
        self.assertEqual(metrics.expert_tiles("0,1,2,4,8"), [0, 1, 2, 4, 8])
        self.assertEqual(len(metrics.requested_modes(config(draft_candidates=[1, 2, 4, 8],
                                                           query_groups=[2, 4], expert_tiles=[0, 1]))), 16)

    def test_extended_draft_sweep_preserves_rotation_gpu_metrics_and_full_row_admission(self):
        code, report, calls, _, _ = self.run_cli([
            "--draft-candidates", "8,16,32", "--query-groups", "2", "--expert-tiles", "0",
            "--weight-reuse", "--repeats", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["draft_candidates"], [8, 16, 32])
        self.assertEqual(report["forced_validation_rows"], 32)
        self.assertEqual(report["partial_commit_checks"], 6)
        self.assertEqual(len(report["ordinary"]), 3)
        self.assertEqual(len(report["speculative"]), 9)
        self.assertEqual(set(report["speculative_tokens_per_second_median"]), {"d8_q2_t0", "d16_q2_t0", "d32_q2_t0"})
        trials = report["ordinary"] + report["speculative"]
        for round_number in range(3):
            ordered = sorted((trial for trial in trials if trial["round"] == round_number), key=lambda trial: trial["order"])
            modes = [0, 8, 16, 32]
            self.assertEqual([trial["draft_tokens"] for trial in ordered], modes[round_number:] + modes[:round_number])
        for trial in trials:
            self.assertIs(trial["mlp_weight_reuse"], False)
            self.assertGreaterEqual(trial["gpu_seconds"], 0)
        for draft in (16, 32):
            code, report, _, _, _ = self.run_cli(["--draft-tokens", str(draft)])
            self.assertEqual(code, 0)
            self.assertEqual(report["draft_candidates"], [draft])
        for bad_config in (config(draft_candidates=[33]), config(draft_candidates=[], draft_tokens=33)):
            with self.assertRaises(ValueError):
                metrics.requested_modes(bad_config)
        stderr = io.StringIO()
        with mock.patch.object(metrics.subprocess, "run") as run, contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exited:
            metrics.cli(["--checkpoint", "/private/token-secret", "--draft-tokens", "33"])
        self.assertEqual(exited.exception.code, 2)
        run.assert_not_called()
        self.assertNotIn("private", stderr.getvalue())

    def test_admission_requires_up_to_32_rows_even_for_eight_token_drafts(self):
        workload = config(draft_candidates=[8])
        for count in (1, 2, 8, 16, 31, 32, 33, 64):
            report = native_report(workload, count=count, eos=True)
            self.assertEqual(report["forced_validation_rows"], min(32, count))
            metrics.validate_report(report, workload)
        report = native_report(workload)
        for rows in (8, 16, 31, 33):
            report["forced_validation_rows"] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                metrics.validate_report(report, workload)

    def test_candidate_parser_accepts_order_and_rejects_invalid_input(self):
        self.assertEqual(metrics.draft_candidates("8, 2,4"), [8, 2, 4])
        self.assertEqual(metrics.draft_candidates("1"), [1])
        self.assertEqual(metrics.draft_candidates("8,16,32"), [8, 16, 32])
        self.assertEqual(metrics.draft_candidates("9"), [9])
        for raw in ("", "0", "33", "32,32", "2,2", "2,", ",2", "2,,4", "2.0", "-2", "+2", "٢", "private-secret"):
            with self.subTest(raw=raw), self.assertRaises(argparse.ArgumentTypeError):
                metrics.draft_candidates(raw)

    def test_invalid_cli_candidates_are_rejected_before_external_work_without_echoing(self):
        for raw in ("", "0", "33", "32,32", "2,2", "2,", "private-secret"):
            stderr = io.StringIO()
            with self.subTest(raw=raw), mock.patch.object(metrics.subprocess, "run") as run, \
                    contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exited:
                metrics.cli(["--checkpoint", "/private/token-secret", "--draft-candidates", raw])
            self.assertEqual(exited.exception.code, 2)
            run.assert_not_called()
            self.assertNotIn("private", stderr.getvalue())
            self.assertNotIn("secret", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
