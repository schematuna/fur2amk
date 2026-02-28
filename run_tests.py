#!/usr/bin/env python3
"""
Test runner for fur2amk.
Runs fur2amk on all Furnace projects in the tests directory,
and compares example outputs against control files in tests/control.
"""

import difflib
import subprocess
import sys
from pathlib import Path


def _run_fur2amk(fur2amk_script: Path, fur_file: Path, script_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(fur2amk_script), str(fur_file), "--nosmpl"],
        capture_output=True,
        text=True,
        cwd=script_dir,
    )


def _normalize(text: str) -> list[str]:
    """Strip lines that are expected to vary between versions before comparison."""
    return [
        line for line in text.splitlines(keepends=True)
        if not line.startswith("; created with fur2amk")
    ]


def _diff_output(generated_path: Path, control_path: Path) -> str | None:
    """Return a unified diff string if files differ, or None if they match."""
    generated = _normalize(generated_path.read_text(encoding="utf-8"))
    control = _normalize(control_path.read_text(encoding="utf-8"))
    if generated == control:
        return None
    return "".join(difflib.unified_diff(
        control,
        generated,
        fromfile=f"control/{control_path.name}",
        tofile=f"music/{generated_path.name}",
    ))


def run_conversion_tests(script_dir: Path, fur2amk_script: Path) -> tuple[int, int, list[tuple[str, str]]]:
    """Run fur2amk on all .fur files in tests/. Returns (passed, failed, failures)."""
    tests_dir = script_dir / "tests"

    if not tests_dir.exists():
        print(f"Error: tests directory not found: {tests_dir}")
        return 0, 0, []

    fur_files = sorted(tests_dir.glob("*.fur"))
    if not fur_files:
        print("No .fur files found in tests/")
        return 0, 0, []

    print(f"Conversion tests ({len(fur_files)} files in tests/):\n")

    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for fur_file in fur_files:
        name = fur_file.stem
        print(f"  {name}... ", end="", flush=True)
        try:
            result = _run_fur2amk(fur2amk_script, fur_file, script_dir)
            if result.returncode == 0:
                print("OK")
                passed += 1
            else:
                print("FAILED")
                failed += 1
                failures.append((name, result.stderr or result.stdout))
        except Exception as e:
            print(f"ERROR")
            failed += 1
            failures.append((name, str(e)))

    return passed, failed, failures


def run_example_tests(script_dir: Path, fur2amk_script: Path) -> tuple[int, int, list[tuple[str, str]]]:
    """Run fur2amk on examples/ and diff output against tests/control. Returns (passed, failed, failures)."""
    examples_dir = script_dir / "examples"
    control_dir = script_dir / "tests" / "control"
    music_dir = script_dir / "music"

    if not examples_dir.exists():
        print(f"Error: examples directory not found: {examples_dir}")
        return 0, 0, []

    fur_files = sorted(examples_dir.glob("*.fur"))
    if not fur_files:
        print("No .fur files found in examples/")
        return 0, 0, []

    print(f"Control tests ({len(fur_files)} files in examples/control):\n")

    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for fur_file in fur_files:
        name = fur_file.stem
        print(f"  {name}... ", end="", flush=True)

        control_file = control_dir / f"{name}.txt"
        if not control_file.exists():
            print(f"SKIP (no control file at tests/control/{name}.txt)")
            continue

        try:
            result = _run_fur2amk(fur2amk_script, fur_file, script_dir)
            if result.returncode != 0:
                print("FAILED (conversion error)")
                failed += 1
                failures.append((name, result.stderr or result.stdout))
                continue

            generated_file = music_dir / f"{name}.txt"
            if not generated_file.exists():
                print("FAILED (output file not found)")
                failed += 1
                failures.append((name, f"Expected output not found: music/{name}.txt"))
                continue

            diff = _diff_output(generated_file, control_file)
            if diff is None:
                print("OK")
                passed += 1
            else:
                print("FAILED (output differs)")
                failed += 1
                failures.append((name, diff))

        except Exception as e:
            print("ERROR")
            failed += 1
            failures.append((name, str(e)))

    return passed, failed, failures


def _print_failures(failures: list[tuple[str, str]]) -> None:
    if not failures:
        return
    print("\nFailure details:")
    for name, detail in failures:
        print(f"\n  {name}:")
        for line in detail.rstrip().splitlines():
            print(f"    {line}")


def main() -> int:
    script_dir = Path(__file__).parent
    fur2amk_script = script_dir / "fur2amk.py"

    conv_passed, conv_failed, conv_failures = run_conversion_tests(script_dir, fur2amk_script)
    print()
    ex_passed, ex_failed, ex_failures = run_example_tests(script_dir, fur2amk_script)

    _print_failures(conv_failures + ex_failures)

    total_passed = conv_passed + ex_passed
    total_failed = conv_failed + ex_failed
    print(f"\n{total_passed} passed, {total_failed} failed")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
