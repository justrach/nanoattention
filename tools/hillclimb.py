#!/usr/bin/env python3
"""Bounded, parity-gated resident scheduling search; never edits kernel defaults.

From the repository root, use a Python environment with the local tokenizer and MLX installed:
  python3 tools/guarded_run.py --timeout 3600 -- \
      python3 tools/hillclimb.py CHECKPOINT --output .zig-cache/hillclimb.json

The parent imports only the tokenizer, not MLX/full weights. Every native/MLX
engine runs in a sequential child and exits before the next engine starts. The
outer guard owns resource limits/timeout; this driver acquires no additional lock.
Coarse --max-trials counts distinct configurations, including the seed. At most
four additional 64-token/3-repeat native confirmations (seed plus top three), two
16-token logit comparisons, and one contemporary MLX benchmark follow. These
confirmation costs do not consume the coarse search budget. No global-optimum or
MLX-speed claim is made from coarse results or checksum agreement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

import benchmark

WORKERS = (8, 16, 24, 32)
ROW_TILES = (1, 2, 4, 8)
SEED = (32, 1)
IMPROVEMENT = 0.03
FULL_TOKENS = 64
FULL_REPEATS = 3
HELDOUT_PROMPT = "Write a Python function that reverses a linked list."


def bounded_int(maximum):
    def parse(value):
        number = int(value)
        if not 1 <= number <= maximum:
            raise argparse.ArgumentTypeError("must be in 1..{}".format(maximum))
        return number
    return parse


def json_safe(value):
    """Keep rejected nonfinite measurements inspectable in standards-valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def config_key(config):
    return "workers{}_tile{}".format(*config)


def neighbors(config):
    wi, ti = WORKERS.index(config[0]), ROW_TILES.index(config[1])
    for w, t in ((wi - 1, ti), (wi + 1, ti), (wi, ti - 1), (wi, ti + 1)):
        if 0 <= w < len(WORKERS) and 0 <= t < len(ROW_TILES):
            yield WORKERS[w], ROW_TILES[t]


def admitted_rows(rows, tokens, repeats, expected=None):
    """Token-position parity is mandatory; equal checksums alone admit nothing."""
    benchmark.validate_rows(rows, "zig_fp32", repeats, resident_workers=1)
    predictions = None
    for row in rows:
        if row.get("error"):
            raise ValueError("native row reports an error: {}".format(row["error"]))
        actual = row.get("predictions")
        if (not isinstance(actual, list) or len(actual) != tokens or
                any(type(token) is not int or not 0 <= token <= 0xFFFFFFFF for token in actual)):
            raise ValueError("missing/invalid per-position u32 predictions")
        if predictions is not None and actual != predictions:
            raise ValueError("per-position predictions changed between identical repeats")
        predictions = actual
        if expected is not None and actual != expected:
            mismatches = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            raise ValueError("prediction mismatch against seed at positions {}".format(mismatches[:16]))
    if expected is not None and len(expected) != tokens:
        raise ValueError("baseline prediction count does not match workload")
    median = statistics.median(row["decode_seconds"] for row in rows)
    if not math.isfinite(tokens / median):
        raise ValueError("nonfinite derived decode throughput")
    return predictions, median


class HillClimber:
    """Pure control plane; run_native(config, token_count, repeats) is injectable."""

    def __init__(self, report, run_native, save):
        self.report, self.run_native, self.save = report, run_native, save

    def evaluate(self, config, phase, tokens, repeats, expected=None):
        key = config_key(config)
        candidate = self.report["candidates"].setdefault(key, {
            "resident_workers": config[0], "resident_row_tile": config[1],
        })
        if phase in candidate:
            return candidate[phase]
        result = {"status": "running", "decode_tokens": tokens, "repeats": repeats}
        candidate[phase] = result
        self.save()
        try:
            rows = self.run_native(config, tokens, repeats)
            result["runs"] = rows  # Keep even rejected outputs for inspection.
            predictions, median = admitted_rows(rows, tokens, repeats, expected)
            seconds = [row["decode_seconds"] for row in rows]
            result.update(status="valid", predictions=predictions,
                          median_decode_seconds=median,
                          decode_tokens_per_second=tokens / median,
                          relative_timing_range=(max(seconds) - min(seconds)) / median)
        except Exception as exc:
            result.update(status="rejected", error="{}: {}".format(type(exc).__name__, exc))
        self.save()
        return result

    def search(self, max_trials, tokens, repeats):
        self.report["stage"] = "coarse_search"
        baseline = self.evaluate(SEED, "coarse", tokens, repeats)
        if baseline["status"] != "valid":
            raise RuntimeError("seed failed coarse admission; no timing candidate may be promoted")
        expected = baseline["predictions"]
        current = SEED
        path = self.report["search_path"]
        path.append(config_key(current))
        while True:
            adjacent = list(neighbors(current))
            exhausted = False
            for config in adjacent:
                key = config_key(config)
                if key in self.report["candidates"]:
                    continue
                if len(self.report["candidates"]) >= max_trials:
                    exhausted = True
                    break
                self.evaluate(config, "coarse", tokens, repeats, expected)
            valid = [config for config in adjacent
                     if self.report["candidates"].get(config_key(config), {}).get(
                         "coarse", {}).get("status") == "valid"]
            current_seconds = self.report["candidates"][config_key(current)]["coarse"]["median_decode_seconds"]
            best = min(valid, key=lambda c: self.report["candidates"][config_key(c)]["coarse"]["median_decode_seconds"],
                       default=current)
            best_seconds = self.report["candidates"][config_key(best)]["coarse"]["median_decode_seconds"]
            if best_seconds < current_seconds * (1 - IMPROVEMENT):
                current = best
                path.append(config_key(current))
                continue
            self.report["search_stop"] = (
                "coarse budget exhausted; neighborhood incomplete" if exhausted else
                "local plateau: no admitted adjacent candidate improves median by more than 3%")
            break
        ranked = sorted((key for key, value in self.report["candidates"].items()
                         if value["coarse"]["status"] == "valid"),
                        key=lambda key: self.report["candidates"][key]["coarse"]["median_decode_seconds"])
        self.report["coarse_ranking"] = ranked
        self.report["coarse_trials"] = len(self.report["candidates"])
        self.report["coarse_local_recommendation"] = config_key(current)
        self.save()
        return ranked

    def grid(self, max_trials, tokens, repeats):
        """A bounded sweep avoids mistaking a hill-climb plateau for a global one."""
        self.report["stage"] = "coarse_search"
        baseline = self.evaluate(SEED, "coarse", tokens, repeats)
        if baseline["status"] != "valid":
            raise RuntimeError("seed failed coarse admission; no timing candidate may be promoted")
        for workers in reversed(WORKERS):
            for tile in ROW_TILES:
                config = workers, tile
                if config == SEED:
                    continue
                if len(self.report["candidates"]) >= max_trials:
                    break
                self.evaluate(config, "coarse", tokens, repeats, baseline["predictions"])
        ranked = sorted((key for key, value in self.report["candidates"].items()
                         if value["coarse"]["status"] == "valid"),
                        key=lambda key: self.report["candidates"][key]["coarse"]["median_decode_seconds"])
        self.report["coarse_ranking"] = ranked
        self.report["coarse_trials"] = len(self.report["candidates"])
        self.report["search_stop"] = ("complete finite grid" if max_trials >= len(WORKERS) * len(ROW_TILES)
                                      else "grid budget exhausted; grid incomplete")
        self.save()
        return ranked

    def confirm(self, ranked, coarse_tokens):
        self.report["stage"] = "full_confirmation"
        if len(ranked) < 3:
            raise RuntimeError("fewer than three valid coarse candidates; insufficient confirmation coverage")
        baseline = self.evaluate(SEED, "full", FULL_TOKENS, FULL_REPEATS)
        if baseline["status"] != "valid":
            raise RuntimeError("seed failed full confirmation")
        coarse_seed = self.report["candidates"][config_key(SEED)]["coarse"]
        if baseline["predictions"][:coarse_tokens] != coarse_seed["predictions"]:
            baseline.update(status="rejected", error="full seed predictions differ from coarse seed prefix")
            self.save()
            raise RuntimeError(baseline["error"])
        for key in ranked[:3]:
            candidate = self.report["candidates"][key]
            config = candidate["resident_workers"], candidate["resident_row_tile"]
            full = self.evaluate(config, "full", FULL_TOKENS, FULL_REPEATS, baseline["predictions"])
            if full["status"] == "valid":
                coarse_speedup = coarse_seed["median_decode_seconds"] / candidate["coarse"]["median_decode_seconds"]
                full_speedup = baseline["median_decode_seconds"] / full["median_decode_seconds"]
                candidate["confirmation"] = {
                    "coarse_speedup_over_seed": coarse_speedup,
                    "full_speedup_over_seed": full_speedup,
                    "noisy_regression": coarse_speedup > 1 / (1 - IMPROVEMENT) and
                                        not full_speedup > 1 / (1 - IMPROVEMENT),
                    "slower_than_seed_on_full": full_speedup < 1,
                }
        full_ranked = sorted((key for key, value in self.report["candidates"].items()
                              if value.get("full", {}).get("status") == "valid"),
                             key=lambda key: self.report["candidates"][key]["full"]["median_decode_seconds"])
        self.report["full_ranking"] = full_ranked
        best = full_ranked[0]
        if self.report["candidates"][best]["full"]["median_decode_seconds"] >= baseline["median_decode_seconds"] * (1 - IMPROVEMENT):
            best = config_key(SEED)
        self.report["provisional_winner"] = best
        self.save()
        return self.report["candidates"][best]


def child_run(command):
    # No Popen/background work: exit and release weights before the next child.
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def json_prefix(text):
    """compare_mlx prints one JSON report followed by a human PASS line."""
    return json.JSONDecoder().raw_decode(text.lstrip())[0]


def final_validation(report, checkpoint, binary, engine_binary, winner, save, run_child=child_run):
    tools = Path(__file__).resolve().parent
    flags = ["--resident-workers", str(winner["resident_workers"]),
             "--resident-row-tile", str(winner["resident_row_tile"])]
    report["stage"] = "winner_logit_validation"
    report["parity_checks"] = []
    save()
    for name, prompt in (("coding", benchmark.PROMPT), ("heldout", HELDOUT_PROMPT)):
        check = {"name": name, "prompt": prompt, "status": "running"}
        report["parity_checks"].append(check)
        save()
        result = run_child([
            sys.executable, str(tools / "compare_mlx.py"), str(checkpoint),
            "--binary", str(engine_binary), "--decode-tokens", "16", "--prompt", prompt, *flags,
        ])
        check.update(result)
        if result["returncode"] != 0:
            check["status"] = "failed"
            save()
            raise RuntimeError("{} winner logit/parity child failed".format(name))
        try:
            parity = json_prefix(result["stdout"])
            # The child enforces unchanged FP32 allclose rtol/atol=2e-3, all
            # greedy positions, and final top1; require its explicit completion.
            if not isinstance(parity, dict) or "PASS:" not in result["stdout"]:
                raise ValueError("comparison did not report completed parity validation")
            if (parity["generated_tokens"] <= 0 or
                    parity["greedy_decode_agreement"] != parity["generated_tokens"] or
                    parity["last_native_top1"] != parity["last_mlx_top1"] or
                    any(not math.isfinite(parity[field]) for field in
                        ("max_absolute_logit_error", "rms_logit_error"))):
                raise ValueError("comparison returned invalid/incomplete parity evidence")
            check.update(status="passed", report=parity)
        except Exception as exc:
            check.update(status="failed", error=str(exc))
            save()
            raise RuntimeError("invalid {} parity report".format(name)) from exc
        save()
    report["stage"] = "contemporary_mlx_benchmark"
    save()
    with tempfile.TemporaryDirectory(prefix="flint-hillclimb-reference-") as temp:
        output = Path(temp) / "reference.json"
        result = run_child([
            sys.executable, str(tools / "benchmark.py"), str(checkpoint),
            "--binary", str(binary), "--decode-tokens", str(FULL_TOKENS),
            "--repeats", str(FULL_REPEATS), "--output", str(output), *flags,
        ])
        report["reference_child"] = result
        save()
        if result["returncode"] != 0:
            raise RuntimeError("contemporary MLX benchmark child failed")
        reference = json.loads(output.read_text(encoding="utf-8"))
    report["contemporary_benchmark"] = reference
    if (reference["input"] != report["input"] or
            reference["decode_tokens"] != FULL_TOKENS or reference["repeats"] != FULL_REPEATS):
        raise ValueError("contemporary benchmark workload differs from confirmed workload")
    if (reference["native_execution"]["resident_workers"] != winner["resident_workers"] or
            reference["native_execution"]["resident_row_tile"] != winner["resident_row_tile"]):
        raise ValueError("contemporary benchmark did not use the winning configuration")
    for engine in benchmark.ENGINES:
        benchmark.validate_rows([row for row in reference["runs"] if row["engine"] == engine],
                                engine, FULL_REPEATS, resident_workers=winner["resident_workers"])
    rates = {engine: reference["engines"][engine]["decode_tokens_per_second"]
             for engine in benchmark.ENGINES}
    if any(not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0
           for rate in rates.values()):
        raise ValueError("contemporary benchmark has invalid throughput")
    report["mlx_decode_target"] = {
        "scope": "this machine, checkpoint, 64-token teacher-forced workload; medians of 3 runs",
        "decode_tokens_per_second": rates,
        "zig_speedup_over": {engine: rates["zig_fp32"] / rates[engine]
                             for engine in benchmark.ENGINES[1:]},
        "matched_or_exceeded": {engine: rates["zig_fp32"] >= rates[engine]
                                for engine in benchmark.ENGINES[1:]},
        "note": "Speed matches require measured native throughput >= MLX; no tolerance or extrapolation.",
    }
    report["winner"] = {"resident_workers": winner["resident_workers"],
                        "resident_row_tile": winner["resident_row_tile"],
                        "admitted": True, "kernel_defaults_changed": False}
    report["stage"] = "complete"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("zig-out/bin/flint-bench"))
    parser.add_argument("--engine-binary", type=Path, default=Path("zig-out/bin/flint"))
    parser.add_argument("--search", choices=("hillclimb", "grid"), default="hillclimb",
                        help="neighboring hill climb or bounded grid sweep")
    parser.add_argument("--max-trials", type=bounded_int(16), default=12,
                        help="distinct coarse configurations including seed (1..16)")
    parser.add_argument("--coarse-tokens", type=bounded_int(64), default=32)
    parser.add_argument("--repeats", type=bounded_int(10), default=2)
    parser.add_argument("--output", type=Path, default=Path(".zig-cache/hillclimb.json"))
    args = parser.parse_args()
    checkpoint, binary, engine_binary = (p.resolve() for p in
                                          (args.checkpoint, args.binary, args.engine_binary))
    output = args.output.resolve()
    if not checkpoint.is_dir():
        parser.error("checkpoint must be an existing local directory")
    if not binary.is_file() or not engine_binary.is_file():
        parser.error("build flint-bench and flint first, or supply both binary paths")
    if output == checkpoint or checkpoint in output.parents:
        parser.error("--output must not write inside the checkpoint directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "stage": "tokenizer", "winner": None,
        "artifacts": {
            "benchmark_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "engine_binary_sha256": hashlib.sha256(engine_binary.read_bytes()).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256((Path(__file__).resolve().parents[1] / "src" / name).read_bytes()).hexdigest()
                for name in ("resident_plan.zig", "resident_plan.metal", "kernels.metal", "row_tiling.metal", "resident.metal")
            },
        },
        "search": {"mode": args.search, "workers": list(WORKERS), "row_tiles": list(ROW_TILES),
                   "seed": config_key(SEED), "max_trials": args.max_trials,
                   "coarse_tokens": args.coarse_tokens, "coarse_repeats": args.repeats,
                   "minimum_median_improvement": IMPROVEMENT,
                   "full_tokens": FULL_TOKENS, "full_repeats": FULL_REPEATS,
                   "confirmation_top_k": 3,
                   "scope": "bounded finite-parameter search, not a global kernel optimum"},
        "candidates": {}, "search_path": [],
    }

    def save():
        # Atomic same-directory replacement leaves the previous complete report
        # intact if interrupted during serialization; never touches checkpoints.
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    save()
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=False,
            fix_mistral_regex=False,
        )
        template = checkpoint / "chat_template.jinja"
        if template.is_file():
            tokenizer.chat_template = template.read_text(encoding="utf-8")
        if not tokenizer.chat_template:
            raise ValueError("checkpoint has no local chat template")
        prompt, decode = benchmark.workload(tokenizer, FULL_TOKENS)
        del tokenizer
        report["input"] = {"prompt": prompt, "decode": decode}
        encoded = json.dumps(report["input"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        report["input_sha256"] = hashlib.sha256(encoded).hexdigest()

        def run_native(config, token_count, repeats):
            return benchmark.native_runs(binary, checkpoint, prompt, decode[:token_count], repeats,
                                         resident_workers=config[0], resident_row_tile=config[1],
                                         emit_predictions=True)

        climb = HillClimber(report, run_native, save)
        search = climb.grid if args.search == "grid" else climb.search
        ranked = search(args.max_trials, args.coarse_tokens, args.repeats)
        winner = climb.confirm(ranked, args.coarse_tokens)
        final_validation(report, checkpoint, binary, engine_binary, winner, save)
    except BaseException as exc:
        report["failed_stage"] = report["stage"]
        report["stage"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = "{}: {}".format(type(exc).__name__, exc)
        save()
        raise
    print(json.dumps({"report": str(output), "winner": report["winner"],
                      "mlx_decode_target": report["mlx_decode_target"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
