#!/usr/bin/env python3
"""
Test runner for fur2amk.
Runs fur2amk on all Furnace projects in the tests directory.
"""

import subprocess
import sys
from pathlib import Path


def run_tests() -> bool:
    """Run fur2amk on all .fur files in tests directory. Returns True if all pass."""
    script_dir = Path(__file__).parent
    tests_dir = script_dir / "tests"
    fur2amk_script = script_dir / "fur2amk.py"

    if not tests_dir.exists():
        print(f"Error: tests directory not found: {tests_dir}")
        return False

    fur_files = sorted(tests_dir.glob("*.fur"))
    if not fur_files:
        print("No .fur files found in tests directory")
        return True

    print(f"Running fur2amk on {len(fur_files)} test files...\n")

    passed = 0
    failed = 0
    failures = []

    for fur_file in fur_files:
        test_name = fur_file.stem
        print(f"  {test_name}... ", end="", flush=True)

        try:
            result = subprocess.run(
                [sys.executable, str(fur2amk_script), str(fur_file), "--nosmpl"],
                capture_output=True,
                text=True,
                cwd=script_dir,
            )

            if result.returncode == 0:
                print("OK")
                passed += 1
            else:
                print("FAILED")
                failed += 1
                failures.append((test_name, result.stderr or result.stdout))

        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            failures.append((test_name, str(e)))

    print(f"\n{passed} passed, {failed} failed")

    if failures:
        print("\nFailure details:")
        for name, error in failures:
            print(f"\n  {name}:")
            for line in error.strip().split("\n"):
                print(f"    {line}")

    return failed == 0


def main() -> int:
    success = run_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
