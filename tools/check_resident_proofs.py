#!/usr/bin/env python3
"""Build kernel-checked Lean proofs and compare the generated native bindings.

Default mode never rewrites an artifact. --generate intentionally writes the
emitter output directly; subsequent native builds should run the default mode.
Requires the pinned Lean toolchain and lake, with Std only (no dependencies).
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="write generated bindings")
    args = parser.parse_args()
    zig = Path(__file__).resolve().parents[1]
    proofs = zig / "proofs"
    try:
        subprocess.run(["lake", "build"], cwd=proofs, check=True)
        subprocess.run(["lake", "env", "lean", "Speculative.lean"], cwd=proofs, check=True)
        subprocess.run(["lake", "env", "lean", "Prefetch.lean"], cwd=proofs, check=True)
        subprocess.run(["lake", "env", "lean", "Transpose.lean"], cwd=proofs, check=True)
        subprocess.run(["lake", "env", "lean", "Prepared.lean"], cwd=proofs, check=True)
        emitter = proofs / ".lake" / "build" / "bin" / "resident_emit"
        # Obtain all results before touching any destination.
        generated = [
            (zig / "src" / filename, subprocess.run(
                [str(emitter), mode], cwd=proofs, check=True,
                stdout=subprocess.PIPE).stdout)
            for filename, mode in (("resident_plan.zig", "--zig"),
                                   ("resident_plan.metal", "--metal"),
                                   ("replay_limits.h", "--replay-c"))
        ]
        failed = False
        for path, expected in generated:
            if args.generate:
                path.write_bytes(expected)
                print(f"Generated {path.relative_to(zig)}")
                continue
            actual = path.read_bytes() if path.exists() else b""
            if actual != expected:
                failed = True
                print(f"Binding drift: {path.relative_to(zig)}", file=sys.stderr)
                diff = difflib.unified_diff(
                    actual.decode("utf-8", errors="replace").splitlines(),
                    expected.decode("utf-8").splitlines(),
                    fromfile="checked-in artifact", tofile="Lean emitter", lineterm="")
                for line in diff:
                    print(line, file=sys.stderr)
        if failed:
            print("Run python3 tools/check_resident_proofs.py --generate to regenerate.",
                  file=sys.stderr)
            return 1
        print("Resident proofs and bindings verified." if not args.generate
              else "Resident proofs built; bindings generated.")
        return 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Resident proof check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
