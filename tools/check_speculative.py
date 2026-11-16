#!/usr/bin/env python3
"""Verify exact native greedy speculation, then time real free-running decode.

No MLX inference or downloads. Requires a locally installed Transformers tokenizer
and a built flint-speculative-bench. Run under the project's GPU guard, e.g.:

  python3 tools/guarded_run.py --timeout 3600 -- python3 tools/check_speculative.py \
      --checkpoint LOCAL_CHECKPOINT --binary zig-out/bin/flint-speculative-bench \
      --prompt-tokens 16384 --max-tokens 256 --draft-candidates 2,4,8 --repeats 3 \
      --query-groups 2 --expert-tiles 0 --output speculative-16k.json

Repeat with --prompt-tokens 32768. A >513-token prefix exercises wrapped sliding
KV on the real checkpoint. Forced rows are untimed correctness checks ONLY;
speed measures history-only drafts without access to ordinary future tokens.
The temporary native config is outside the repository and removed on exit.
Schema v3 has ordinary/speculative arrays with zero-based round/order and
per-trial draft_tokens/query_group/expert_tile (draft 0 means ordinary).
Query groups: 2/4/8; expert tiles: 0/1/2/4/8 (0 inherits staged tile 2).
The Cartesian product is unique and limited to 16 modes. Full-vocabulary row
and rollback checks run per mode over min(32, committed tokens) reference
inputs, independent of draft length; partial_commit_checks sums across modes.
--weight-reuse enables reuse only for speculative modes; ordinary reference,
warmup, timed baselines, and serial row admission always keep it disabled.
--mlp-weight-reuse independently enables candidate-only MLP reuse under the
same scope. Top-level weight_reuse/mlp_weight_reuse flags identify the run;
mode labels stay unchanged unless --compare-mlp-weight-reuse is used. That
flag includes both MLP settings per mode regardless of --mlp-weight-reuse,
using labels such as d8_q2_t0_m0/d8_q2_t0_m1 within the same 16-mode limit.
Ordinary runs and serial admission always disable both reuse paths.
Every timed trial includes wall decode_seconds and GPU command gpu_seconds.
Wall minus GPU command time is only an approximate unprofiled residual, not
pure CPU time or a performance gate; the clocks have no enforced ordering.
Each round rotates baseline and all
candidates from the same restored prefix after untimed reference/warmups.
Wrapper medians and speedups are keyed by d8_q4_t1-style mode labels, relative to the median
of all ordinary trials. Reports contain no checkpoint/binary paths, prompt
IDs/text, raw logits, or child diagnostics.
"""
import argparse
import contextlib
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_DRAFT_TOKENS = 32  # Native Model.max_speculative_tokens admission capacity.

CODE = '''from dataclasses import dataclass
from typing import Optional

@dataclass
class Node:
    key: int
    left: Optional["Node"] = None
    right: Optional["Node"] = None


def insert(root, key):
    """Insert a key into a binary search tree, preserving sorted order."""
    if root is None:
        return Node(key)
    if key < root.key:
        root.left = insert(root.left, key)
    elif key > root.key:
        root.right = insert(root.right, key)
    return root


def inorder(root):
    if root is not None:
        yield from inorder(root.left)
        yield root.key
        yield from inorder(root.right)


def test_tree():
    root = None
    for key in (8, 3, 10, 1, 6, 14, 4, 7, 13):
        root = insert(root, key)
    assert list(inorder(root)) == [1, 3, 4, 6, 7, 8, 10, 13, 14]

'''


def positive_u32(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive u32")
    if not 0 < number <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("must be in 1..4294967295")
    return number


def draft_candidates(value):
    """Parse a nonempty, ordered set of native draft lengths without echoing input."""
    try:
        parts = value.split(",")
        if any(not part.strip().isascii() or not part.strip().isdecimal() for part in parts):
            raise ValueError
        candidates = [int(part) for part in parts]
        if (not candidates or any(not 1 <= candidate <= MAX_DRAFT_TOKENS for candidate in candidates) or
                len(set(candidates)) != len(candidates)):
            raise ValueError
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("must be distinct comma-separated draft lengths in 1..32") from None
    return candidates


def requested_candidates(config):
    candidates = config.get("draft_candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("invalid requested candidates")
    candidates = candidates or [config["draft_tokens"]]
    if (any(type(candidate) is not int or not 1 <= candidate <= MAX_DRAFT_TOKENS for candidate in candidates) or
            len(set(candidates)) != len(candidates)):
        raise ValueError("invalid requested candidates")
    return candidates


def mode_axis(value, allowed):
    try:
        parts = value.split(",")
        if any(not part.strip().isascii() or not part.strip().isdecimal() for part in parts):
            raise ValueError
        values = [int(part) for part in parts]
        if not values or any(item not in allowed for item in values) or len(set(values)) != len(values):
            raise ValueError
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("invalid or duplicate mode axis values") from None
    return values


def query_groups(value):
    return mode_axis(value, (2, 4, 8))


def expert_tiles(value):
    return mode_axis(value, (0, 1, 2, 4, 8))


def requested_modes(config):
    axes = [requested_candidates(config), config.get("query_groups", [2]), config.get("expert_tiles", [0])]
    for axis, allowed in zip(axes, (range(1, MAX_DRAFT_TOKENS + 1), (2, 4, 8), (0, 1, 2, 4, 8))):
        if (not isinstance(axis, list) or not axis or
                any(type(value) is not int or value not in allowed for value in axis) or
                len(set(axis)) != len(axis)):
            raise ValueError("invalid requested mode axis")
    for key in ("mlp_weight_reuse", "compare_mlp_weight_reuse"):
        if type(config.get(key, False)) is not bool:
            raise ValueError("invalid requested MLP reuse flag")
    mlp_modes = [False, True] if config.get("compare_mlp_weight_reuse", False) else [config.get("mlp_weight_reuse", False)]
    modes = list(itertools.product(*axes, mlp_modes))
    if len(modes) > 16:
        raise ValueError("too many requested modes")
    return modes


def mode_label(mode, compare_mlp=False):
    label = "d{}_q{}_t{}".format(*mode[:3])
    return label + "_m{}".format(int(mode[3])) if compare_mlp else label


def code_prompt(tokenizer, count):
    """Exact-length real code-token repetition, without padding/chat templates."""
    stock = list(tokenizer.encode(CODE, add_special_tokens=False))
    if not stock or any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF
                        for token in stock):
        raise ValueError("tokenizer returned invalid code tokens")
    copies, remainder = divmod(count, len(stock))
    return stock * copies + stock[:remainder]


def binary_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def validate_report(report, config):
    """Reject unexpected payloads rather than accidentally publishing diagnostics."""
    candidates = requested_candidates(config)
    combinations = requested_modes(config)
    fields = {
        "schema_version", "engine", "exact_tokens", "exact_final_logit_bits",
        "exact_batch_row_bits", "forced_rows_correctness_only", "speed_workload",
        "actual_prefix_tokens", "max_tokens", "draft_candidates", "prefill_seconds",
        "reference_untimed", "warmup_timed", "warmup_max_tokens", "timing_order",
        "prefill_samples", "prefix_restore_timed", "forced_validation_rows",
        "partial_commit_checks", "wrapped_sliding_layers", "kv_allocated_bytes",
        "kv_storage", "ordinary", "speculative", "query_groups", "expert_tiles",
        "weight_reuse", "mlp_weight_reuse", "compare_mlp_weight_reuse", "weight_reuse_scope",
    }
    if not isinstance(report, dict) or set(report) != fields:
        raise ValueError("unexpected native report schema")
    if (type(report["schema_version"]) is not int or report["schema_version"] != 3 or
            report["engine"] != "native_matched_greedy"):
        raise ValueError("unexpected native benchmark engine")
    for key in ("weight_reuse", "mlp_weight_reuse", "compare_mlp_weight_reuse"):
        expected_reuse = config.get(key, False)
        if (type(expected_reuse) is not bool or type(report[key]) is not bool or
                report[key] is not expected_reuse):
            raise ValueError("native candidate weight reuse differs from request")
    if report["weight_reuse_scope"] != "speculative_modes_only":
        raise ValueError("unexpected native weight reuse scope")
    for key in ("exact_tokens", "exact_final_logit_bits", "exact_batch_row_bits",
                "forced_rows_correctness_only"):
        if report[key] is not True:
            raise ValueError("native correctness admission did not pass")
    if (report["speed_workload"] != "free_running_history_draft" or
            report["prefix_restore_timed"] is not False or report["prefill_samples"] != 1 or
            report["reference_untimed"] is not True or report["warmup_timed"] is not False or
            report["warmup_max_tokens"] != min(32, config["max_tokens"]) or
            report["timing_order"] != "rotated_by_round"):
        raise ValueError("unexpected native timing protocol")
    if (report["actual_prefix_tokens"] != len(config["prompt_tokens"]) or
            report["max_tokens"] != config["max_tokens"] or
            report["draft_candidates"] != candidates):
        raise ValueError("native workload differs from request")
    if (not isinstance(report["draft_candidates"], list) or
            any(type(candidate) is not int for candidate in report["draft_candidates"])):
        raise ValueError("invalid native candidates")
    for key, default in (("query_groups", [2]), ("expert_tiles", [0])):
        if (not isinstance(report[key], list) or report[key] != config.get(key, default) or
                any(type(value) is not int for value in report[key])):
            raise ValueError("native mode axes differ from request")
    for key in ("actual_prefix_tokens", "max_tokens", "warmup_max_tokens", "prefill_samples",
                "forced_validation_rows", "partial_commit_checks",
                "wrapped_sliding_layers", "kv_allocated_bytes"):
        if type(report[key]) is not int or report[key] < 0:
            raise ValueError("invalid native count")
    expected_storage = "bf16_global_fp32_sliding" if config["kv_bf16"] else "fp32"
    if report["kv_storage"] != expected_storage:
        raise ValueError("unexpected native KV storage")
    if not finite_positive(report["prefill_seconds"]):
        raise ValueError("invalid native prefill timing")
    if (not isinstance(report["ordinary"], list) or len(report["ordinary"]) != config["repeats"] or
            not isinstance(report["speculative"], list) or
            len(report["speculative"]) != config["repeats"] * len(combinations)):
        raise ValueError("unexpected native repeat count")
    trial_fields = {"decode_seconds", "gpu_seconds", "committed_tokens", "tokens_per_second",
                    "eos", "batches", "proposed", "accepted", "draft_tokens", "query_group", "expert_tile", "mlp_weight_reuse", "round", "order"}
    seen = set()
    reference = None
    modes = [(0, 2, 0, False)] + combinations
    trials = [(kind, trial) for kind in ("ordinary", "speculative") for trial in report[kind]]
    for kind, trial in trials:
        if not isinstance(trial, dict) or set(trial) != trial_fields:
            raise ValueError("unexpected native trial schema")
        for key in ("decode_seconds", "tokens_per_second"):
            if not finite_positive(trial[key]):
                raise ValueError("invalid native decode timing")
        if type(trial["mlp_weight_reuse"]) is not bool:
            raise ValueError("invalid native trial MLP reuse flag")
        if not finite_nonnegative(trial["gpu_seconds"]):
            raise ValueError("invalid native GPU command timing")
        for key in ("committed_tokens", "batches", "proposed", "accepted", "draft_tokens", "query_group", "expert_tile", "round", "order"):
            if type(trial[key]) is not int or trial[key] < 0:
                raise ValueError("invalid native trial count")
        count = trial["committed_tokens"]
        if not 0 < count <= config["max_tokens"] or type(trial["eos"]) is not bool:
            raise ValueError("invalid native committed count or EOS")
        if not trial["eos"] and count != config["max_tokens"]:
            raise ValueError("native trial did not use full decode budget")
        if reference is None:
            reference = (count, trial["eos"])
        if (count, trial["eos"]) != reference:
            raise ValueError("native exact greedy result differs between trials")
        draft = trial["draft_tokens"]
        mode = (draft, trial["query_group"], trial["expert_tile"], trial["mlp_weight_reuse"])
        if mode not in ([(0, 2, 0, False)] if kind == "ordinary" else combinations):
            raise ValueError("unexpected native trial candidate")
        round_number, order = trial["round"], trial["order"]
        if (round_number >= config["repeats"] or order >= len(modes) or
                modes[(order + round_number % len(modes)) % len(modes)] != mode or
                (round_number, mode) in seen):
            raise ValueError("invalid native rotated trial schedule")
        seen.add((round_number, mode))
        if (not trial["accepted"] <= trial["proposed"] or trial["accepted"] > count or
                not trial["batches"] <= trial["accepted"] or
                not 2 * trial["batches"] <= trial["proposed"] <= draft * trial["batches"] or
                (draft < 2 and any(trial[key] for key in ("batches", "proposed", "accepted")))):
            raise ValueError("invalid native proposal accounting")
        if not math.isclose(trial["tokens_per_second"], count / trial["decode_seconds"],
                            rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("inconsistent native decode timing")
    forced = min(MAX_DRAFT_TOKENS, reference[0])
    expected_partial = 0 if forced == 1 else (1 if forced == 2 else 2)
    if report["forced_validation_rows"] != forced or report["partial_commit_checks"] != expected_partial * len(combinations):
        raise ValueError("missing native row/partial rollback checks")


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native JSON field")
        result[key] = value
    return result


def reject_json_constant(value):
    raise ValueError("nonfinite native JSON value")


def run_native(binary, config):
    with tempfile.TemporaryDirectory(prefix="flint-speculative-") as temporary:
        path = Path(temporary) / "input.json"
        path.write_bytes(canonical_bytes(config))
        # Capture stderr: error logs may contain private paths. Never copy those
        # logs into the sanitized report. The outer guard owns runtime limits.
        result = subprocess.run([str(binary), str(path)], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode:
        raise RuntimeError("native speculative verification failed (exit {})".format(result.returncode))
    try:
        report = json.loads(result.stdout, object_pairs_hook=unique_json_object,
                            parse_constant=reject_json_constant)
    except (ValueError, TypeError):
        raise ValueError("native benchmark did not return one JSON report") from None
    validate_report(report, config)
    return report


class SanitizedArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default embeds rejected input (potentially private paths).
        self.exit(2, "Invalid speculative benchmark arguments; use --help.\n")


def main(argv=None):
    parser = SanitizedArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/flint-speculative-bench"))
    parser.add_argument("--prompt-tokens", type=positive_u32, default=16384)
    parser.add_argument("--max-tokens", type=positive_u32, default=256)
    parser.add_argument("--draft-tokens", type=int, choices=range(1, MAX_DRAFT_TOKENS + 1), default=4)
    parser.add_argument("--draft-candidates", type=draft_candidates, default=[],
                        help="distinct comma-separated lengths in 1..32; overrides --draft-tokens")
    parser.add_argument("--query-groups", type=query_groups, default=[2])
    parser.add_argument("--expert-tiles", type=expert_tiles, default=[0])
    parser.add_argument("--repeats", type=positive_u32, default=3)
    parser.add_argument("--kv-bf16", action="store_true")
    parser.add_argument("--compare-mlp-weight-reuse", action="store_true",
                        help="test both MLP reuse settings per speculative mode, overriding --mlp-weight-reuse")
    parser.add_argument("--mlp-weight-reuse", action="store_true",
                        help="enable MLP weight reuse for speculative modes only; ordinary always disables it")
    parser.add_argument("--weight-reuse", action="store_true",
                        help="enable batch weight reuse for speculative modes only; ordinary always disables it")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.prompt_tokens + args.max_tokens > 0xFFFFFFFF:
        parser.error("prompt plus decode exceeds native u32 context")
    try:
        requested_modes(dict(draft_tokens=args.draft_tokens, draft_candidates=args.draft_candidates,
                             query_groups=args.query_groups, expert_tiles=args.expert_tiles,
                             mlp_weight_reuse=args.mlp_weight_reuse,
                             compare_mlp_weight_reuse=args.compare_mlp_weight_reuse))
    except ValueError:
        parser.error("invalid mode combination budget")
    # Import only the tokenizer, never load model weights or run MLX inference.
    # Dependency warnings/errors may embed local paths; retain only our fixed
    # failure message. The explicit regex flag preserves existing tokenization.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint.resolve()),
                                                  local_files_only=True,
                                                  trust_remote_code=False,
                                                  fix_mistral_regex=False)
        prompt = code_prompt(tokenizer, args.prompt_tokens)
    workload = dict(prompt_tokens=prompt, max_tokens=args.max_tokens,
                    draft_tokens=args.draft_tokens, draft_candidates=args.draft_candidates,
                    query_groups=args.query_groups, expert_tiles=args.expert_tiles, repeats=args.repeats,
                    kv_bf16=args.kv_bf16, weight_reuse=args.weight_reuse,
                    mlp_weight_reuse=args.mlp_weight_reuse,
                    compare_mlp_weight_reuse=args.compare_mlp_weight_reuse)
    binary = args.binary.resolve()
    before_hash = binary_hash(binary)
    report = run_native(binary, dict(workload, checkpoint=str(args.checkpoint.resolve())))
    if binary_hash(binary) != before_hash:
        raise RuntimeError("native binary changed during verification")
    baseline_median = statistics.median(trial["tokens_per_second"] for trial in report["ordinary"])
    candidate_medians = {
        mode_label(mode, args.compare_mlp_weight_reuse): statistics.median(
            trial["tokens_per_second"] for trial in report["speculative"]
            if (trial["draft_tokens"], trial["query_group"], trial["expert_tile"], trial["mlp_weight_reuse"]) == mode)
        for mode in requested_modes(workload)
    }
    report.update({
        "binary_sha256": before_hash,
        "input_sha256": hashlib.sha256(canonical_bytes(workload)).hexdigest(),
        "input_hash_scope": "canonical_workload_json_excluding_checkpoint_path",
        "prompt_recipe": "repeat_local_tokenizer_code_ids_then_truncate_no_special_tokens_v1",
        "tokenizer_local_only": True,
        "ordinary_tokens_per_second_median": baseline_median,
        "speculative_tokens_per_second_median": candidate_medians,
        "ordinary_samples": len(report["ordinary"]),
        "speed_note": "Free-running history-only draft; forced rows are untimed correctness only. Weight reuse applies only to speculative modes; ordinary reference and baselines always disable it. MLP weight reuse independently follows the same candidate-only scope. GPU seconds measure command time; wall minus GPU is an approximate unprofiled residual, not pure CPU time or a performance gate. Not a quality evaluation.",
    })
    report["speedup_median"] = {draft: rate / baseline_median
                                for draft, rate in candidate_medians.items()}
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


def cli(argv=None):
    try:
        main(argv)
    except Exception:
        # Do not leak tokenizer exceptions, child stderr, or local paths.
        print("Speculative check failed; no speed report admitted. Check local tokenizer, binary, and native correctness.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
