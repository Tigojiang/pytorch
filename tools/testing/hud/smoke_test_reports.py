"""Run a few test files of every kind and show the test reports they produce.

Clears test/test-reports and run_test.py's stepcurrent cache, runs the Python
and C++ sets below through test/run_test.py with CI=1 (which turns the reports
on), then parses every *.report.xml and prints one line per report.

    python tools/testing/hud/smoke_test_reports.py
    python tools/testing/hud/smoke_test_reports.py --python test_complex -- --inductor

Arguments after "--" are passed to both run_test.py invocations.
"""

from __future__ import annotations

import argparse
import collections
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "test" / "test-reports"
STEPCURRENT_CACHE = REPO_ROOT / ".pytest_cache" / "v" / "cache" / "stepcurrent"
# A CPU-only file, a device-generic file (CPU and CUDA classes in one process)
# and a CUDA file whose handler runs every test in its own process.
PYTHON_TESTS = ["test_type_info", "test_complex", "test_cuda_nvml_based_avail"]
# gtest binaries, run through pytest-cpp with xdist workers.
CPP_TESTS = ["cpp/atest", "cpp/c10_intrusive_ptr_test"]


def run_test(args: list[str]) -> int:
    cmd = [sys.executable, "test/run_test.py", *args]
    print(f"$ CI=1 {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, env={**os.environ, "CI": "1"}).returncode


def describe(report: Path) -> str:
    closed = report.read_bytes().rstrip().endswith(b"</report>")
    try:
        root = ET.parse(report).getroot()
    except ET.ParseError as e:
        return f"PARSE ERROR {e}"
    env = root.find("environment")
    if env is None:
        return "NO <environment> ELEMENT"
    device = f"{env.get('accelerator')} {env.get('device_name') or '-'}"
    device += f" x{env.get('device_count')}"
    flags = " ".join(f"{f.get('name')}={f.get('value')}" for f in env.iter("flag"))
    outcomes = collections.Counter(a.get("outcome") for a in root.iter("attempt"))
    counts = " ".join(f"{k}={v}" for k, v in sorted(outcomes.items()))
    state = "closed" if closed else "OPEN"
    summary = f"{state:6} {sum(outcomes.values()):4} attempts  {counts or '-':28}"
    return f"{summary} {device}  flags: {flags or '-'}"


def main() -> int:
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)
    parser.add_argument("--python", nargs="*", default=PYTHON_TESTS, help="test files")
    parser.add_argument("--cpp", nargs="*", default=CPP_TESTS, help="test binaries")
    parser.add_argument("run_test_args", nargs=argparse.REMAINDER, help="after --")
    args = parser.parse_args()
    extra = args.run_test_args
    if extra[:1] == ["--"]:
        extra = extra[1:]

    for path in (REPORTS_DIR, STEPCURRENT_CACHE):
        shutil.rmtree(path, ignore_errors=True)
    print(f"cleared {REPORTS_DIR} and {STEPCURRENT_CACHE}")

    exit_codes = []
    if args.python:
        exit_codes.append(run_test(["-i", *args.python, *extra]))
    if args.cpp:
        exit_codes.append(run_test(["--cpp", "-i", *args.cpp, *extra]))

    reports = sorted(REPORTS_DIR.rglob("*.report.xml"))
    print(f"\n{len(reports)} reports under {REPORTS_DIR}:")
    for report in reports:
        print(f"  {report.relative_to(REPORTS_DIR)}\n      {describe(report)}")
    print(f"\nrun_test.py exit codes: {exit_codes}")
    return 1 if any(exit_codes) else 0


if __name__ == "__main__":
    sys.exit(main())
