#!/usr/bin/env python3
"""Bound a local macOS inference check without touching other apps or model files.

Optionally holds an existing inference lock so native and MLX workers do not
compete. Monitors the owned process group, memory pressure, RSS, and headroom.
"""
import argparse
import fcntl
import os
import re
import signal
import subprocess
import time

GIB = 1024 ** 3


def command(*args):
    return subprocess.check_output(args, text=True, timeout=3)


def snapshot(group=None):
    raw = command("/usr/bin/vm_stat")
    size = int(re.search(r"page size of (\d+) bytes", raw).group(1))
    pages = dict((name, int(value)) for name, value in re.findall(r"^([^:\n]+):\s+(\d+)\.", raw, re.M))
    available = sum(pages[n] for n in ("Pages free", "Pages inactive", "Pages speculative")) * size
    pressure = int(command("/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"))
    rss = 0
    if group is not None:
        for line in command("/bin/ps", "-axo", "pgid=,rss=").splitlines():
            pgid, kb = map(int, line.split())
            if pgid == group:
                rss += kb * 1024
    return available, pressure, rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", help="Existing lock file shared with another local Flint runtime")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.command
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        parser.error("a command is required after --")
    total = int(command("/usr/sbin/sysctl", "-n", "hw.memsize"))
    rss_limit, reserve = min(48 * GIB, total // 4), max(16 * GIB, total // 4)
    lock = open(args.lock, "rb") if args.lock else None
    try:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        available, pressure, _ = snapshot()
        if pressure != 1 or available < reserve + rss_limit:
            raise RuntimeError("Not enough normal-pressure memory headroom for a full-model check")
        child = subprocess.Popen(argv, start_new_session=True)
        deadline = time.monotonic() + args.timeout
        try:
            while child.poll() is None:
                available, pressure, rss = snapshot(child.pid)
                if pressure != 1 or available < reserve or rss >= rss_limit:
                    raise RuntimeError("Owned inference worker exceeded the memory safety budget")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Owned inference worker exceeded the time budget")
                time.sleep(0.5)
            if child.returncode:
                raise SystemExit(child.returncode)
        finally:
            # Only this newly created group, including any descendants we started.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    finally:
        if lock:
            lock.close()


if __name__ == "__main__":
    main()
