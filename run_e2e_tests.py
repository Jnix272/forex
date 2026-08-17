"""
E2E Test Runner for the Forex Scaling Model.
Invokes E2E test suites and compiles a summary report.

MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine.
Do not hardcode test results.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# MANDATORY INTEGRITY WARNING: DO NOT CHEAT. All implementations must be genuine.
# Do not hardcode test results.


def main():
    print("=" * 70)
    print("Forex Scaling Model - E2E Test Runner")
    print("=" * 70)

    root_dir = Path(__file__).resolve().parent
    tests_to_run = ["tests/test_api.py", "tests/test_dashboard.py", "tests/test_visualizer.py"]

    # Check which test files exist
    existing_tests = []
    for t in tests_to_run:
        p = root_dir / t
        if p.exists():
            existing_tests.append(t)
        else:
            print(f"Warning: Test file {t} not found.")

    if not existing_tests:
        print("Error: No E2E test files found. Exiting.")
        sys.exit(1)

    xml_report = root_dir / "e2e_report.xml"
    if xml_report.exists():
        try:
            os.remove(xml_report)
        except OSError:
            pass

    # Prepare command
    python_exe = sys.executable
    cmd = [python_exe, "-m", "pytest", *existing_tests, f"--junitxml={xml_report}", "-v"]

    # Try adding coverage if pytest-cov/coverage is installed
    try:
        import pytest_cov  # noqa: F401

        # Try adding coverage for the targeted components
        cmd.extend(["--cov=api", "--cov=dashboard", "--cov=visualize_backtest", "--cov-report=term-missing"])
        print("pytest-cov detected. Running with coverage analysis...")
    except ImportError:
        print("pytest-cov not found. Running tests without coverage.")

    print(f"Running command: {' '.join(cmd)}")
    print("Executing tests...")
    print("-" * 70)

    # Run pytest
    result = subprocess.run(cmd, cwd=str(root_dir), capture_output=True, text=True)

    # Print stdout/stderr
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print("-" * 70)
    print("E2E Test Execution Summary:")
    print("-" * 70)

    # Parse JUnit XML to build a custom report
    if xml_report.exists():
        try:
            tree = ET.parse(xml_report)
            root = tree.getroot()

            total_tests = 0
            passed = 0
            failed = 0
            skipped = 0

            failures_list = []
            skips_list = []

            # Root tag is usually <testsuite> or <testsuites>
            suites = [root] if root.tag == "testsuite" else root.findall("testsuite")

            for suite in suites:
                for case in suite.findall("testcase"):
                    total_tests += 1
                    name = case.get("name")
                    classname = case.get("classname").split(".")[-1]

                    fail_el = case.find("failure")
                    err_el = case.find("error")
                    skip_el = case.find("skipped")

                    if fail_el is not None:
                        failed += 1
                        msg = fail_el.get("message") or "Assertion failed"
                        failures_list.append((classname, name, msg))
                    elif err_el is not None:
                        failed += 1
                        msg = err_el.get("message") or "Error during execution"
                        failures_list.append((classname, name, msg))
                    elif skip_el is not None:
                        skipped += 1
                        msg = skip_el.get("message") or skip_el.text or "Skipped"
                        skips_list.append((classname, name, msg))
                    else:
                        passed += 1

            print(f"Total Tests Run: {total_tests}")
            print(f"  - Passed:      {passed}")
            print(f"  - Failed:      {failed} (Gating check failures are expected if files/modules are missing)")
            print(f"  - Skipped:     {skipped} (Tests will skip if implementation is missing)")

            if failures_list:
                print("\nFailed Tests Detail:")
                for cname, name, msg in failures_list:
                    print(f"  [FAIL] {cname}::{name}")
                    print(f"         Reason: {msg}")

            if skips_list:
                print("\nSkipped Tests Detail:")
                for cname, name, msg in skips_list:
                    # Clean up long messages
                    msg_short = msg.strip().split("\n")[0]
                    print(f"  [SKIP] {cname}::{name} -> {msg_short}")

        except Exception as e:
            print(f"Error parsing JUnit XML report: {e}")
    else:
        print("Error: JUnit XML report was not generated. Pytest execution might have crashed.")

    print("=" * 70)

    # Return pytest exit code
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
