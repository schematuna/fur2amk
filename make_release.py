#!/usr/bin/env python3
"""
Release script for fur2amk.
Creates a distributable zip file with all necessary files.
"""

import argparse
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


def read_version() -> tuple[int, int, int]:
    """Read the current version from src/version.py."""
    script_dir = Path(__file__).parent
    version_file = script_dir / "src" / "version.py"

    content = version_file.read_text()

    major_match = re.search(r"VERSION_MAJOR\s*=\s*(\d+)", content)
    minor_match = re.search(r"VERSION_MINOR\s*=\s*(\d+)", content)
    build_match = re.search(r"VERSION_BUILD\s*=\s*(\d+)", content)

    if not major_match or not minor_match or not build_match:
        print("Error: Could not parse version from src/version.py")
        sys.exit(1)

    return int(major_match.group(1)), int(minor_match.group(1)), int(build_match.group(1))


def write_version(major: int, minor: int, build: int) -> None:
    """Write the new version to src/version.py."""
    script_dir = Path(__file__).parent
    version_file = script_dir / "src" / "version.py"

    lines = [
        '"""Version information for fur2amk."""',
        "",
        f"VERSION_MAJOR = {major}",
        f"VERSION_MINOR = {minor}",
        f"VERSION_BUILD = {build}",
        "",
        'VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"',
    ]
    version_file.write_text("\n".join(lines) + "\n")


def run_tests() -> bool:
    """Run the test suite. Returns True if all tests pass."""
    script_dir = Path(__file__).parent
    test_script = script_dir / "run_tests.py"

    if not test_script.exists():
        print("Warning: run_tests.py not found, skipping tests")
        return True

    print("Running tests...")
    result = subprocess.run([sys.executable, str(test_script)], cwd=script_dir)
    return result.returncode == 0


def build_gui() -> Path:
    """Build the GUI exe with PyInstaller. Returns the path to the built exe."""
    script_dir = Path(__file__).parent
    gui_dir = script_dir / "gui"
    print("Building GUI executable...")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--onefile", "--windowed", "--icon", "fuzzy.ico", "--add-data", "fuzzy.ico;.", "--paths", "..", "fur2amk_gui.py"],
        cwd=gui_dir,
    )
    if result.returncode != 0:
        print("Error: PyInstaller build failed.")
        sys.exit(1)
    exe = gui_dir / "dist" / "fur2amk_gui.exe"
    if not exe.exists():
        print(f"Error: Expected exe not found at {exe}")
        sys.exit(1)
    print(f"Built: {exe}")
    return exe


def make_release(version: str, gui_exe: Path) -> None:
    """Create a release package for fur2amk."""
    script_dir = Path(__file__).parent
    releases_dir = script_dir / "releases"
    release_name = f"fur2amk_{version}"
    release_dir = releases_dir / release_name
    python_dir = release_dir / "python"

    releases_dir.mkdir(exist_ok=True)

    if release_dir.exists():
        shutil.rmtree(release_dir)

    release_dir.mkdir()
    python_dir.mkdir()

    # Top-level: docs and exe only
    for filename in ["README.md", "ADVANCED.md"]:
        src = script_dir / filename
        if not src.exists():
            print(f"Warning: {filename} not found, skipping")
            continue
        shutil.copy2(src, release_dir / filename)

    shutil.copy2(gui_exe, release_dir / gui_exe.name)

    # python/ subfolder: all CLI scripts and supporting files
    for filename in ["fur2amk.py", "copy_to_amk.py"]:
        src = script_dir / filename
        if not src.exists():
            print(f"Warning: {filename} not found, skipping")
            continue
        shutil.copy2(src, python_dir / filename)

    config_template = script_dir / "fur2amk_config.json.template"
    if config_template.exists():
        shutil.copy2(config_template, python_dir / "fur2amk_config.json")
    else:
        print("Warning: fur2amk_config.json.template not found")

    shutil.copytree(
        script_dir / "src",
        python_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    for dirname in ["examples", "templates"]:
        src = script_dir / dirname
        if not src.exists():
            print(f"Warning: {dirname}/ not found, skipping")
            continue
        shutil.copytree(
            src,
            release_dir / dirname,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    zip_path = releases_dir / release_name
    shutil.make_archive(str(zip_path), "zip", release_dir)

    shutil.rmtree(release_dir)

    print(f"Release created: {zip_path}.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fur2amk release package")
    parser.add_argument("--major", action="store_true", help="Increment major version (resets minor and build to 0)")
    parser.add_argument("--minor", action="store_true", help="Increment minor version (resets build to 0)")
    parser.add_argument("--build", action="store_true", help="Increment build number")
    parser.add_argument("--nightly", action="store_true", help="Create a nightly build (appends date, doesn't modify version file)")
    args = parser.parse_args()

    major, minor, build = read_version()

    if args.nightly:
        version = f"{major}.{minor}.{build}-nightly.{date.today().strftime('%Y%m%d')}"
        print(f"Creating nightly build: {version}")
    elif args.major:
        major += 1
        minor = 0
        build = 0
        write_version(major, minor, build)
        print(f"Version updated to {major}.{minor}.{build}")
        version = f"{major}.{minor}.{build}"
    elif args.minor:
        minor += 1
        build = 0
        write_version(major, minor, build)
        print(f"Version updated to {major}.{minor}.{build}")
        version = f"{major}.{minor}.{build}"
    elif args.build:
        build += 1
        write_version(major, minor, build)
        print(f"Version updated to {major}.{minor}.{build}")
        version = f"{major}.{minor}.{build}"
    else:
        print(f"Using current version {major}.{minor}.{build} (not incrementing)")
        version = f"{major}.{minor}.{build}"

    if not run_tests():
        print("\nTests failed. Aborting release.")
        sys.exit(1)

    exe = build_gui()

    print()  # blank line after test output
    make_release(version, exe)


if __name__ == "__main__":
    main()
