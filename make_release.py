#!/usr/bin/env python3
"""
Release script for fur2amk.
Creates a distributable zip file with all necessary files.
"""

import argparse
import shutil
import sys
from pathlib import Path


def make_release(version: str) -> None:
    """Create a release package for fur2amk."""
    script_dir = Path(__file__).parent
    releases_dir = script_dir / "releases"
    release_name = f"fur2amk_{version}"
    release_dir = releases_dir / release_name

    # Create releases directory if it doesn't exist
    releases_dir.mkdir(exist_ok=True)

    # Remove existing release directory if it exists
    if release_dir.exists():
        shutil.rmtree(release_dir)

    release_dir.mkdir()

    # Files to copy
    files_to_copy = [
        "fur2amk.py",
        "README.md",
        "ADVANCED.md",
        "copy_to_amk.py",
    ]

    # Copy individual files
    for filename in files_to_copy:
        src = script_dir / filename
        if not src.exists():
            print(f"Warning: {filename} not found, skipping")
            continue
        shutil.copy2(src, release_dir / filename)

    # Copy and rename config template
    config_template = script_dir / "fur2amk_config.json.template"
    if config_template.exists():
        shutil.copy2(config_template, release_dir / "fur2amk_config.json")
    else:
        print("Warning: fur2amk_config.json.template not found")

    # Copy directories (excluding __pycache__)
    dirs_to_copy = ["src", "examples"]
    for dirname in dirs_to_copy:
        src = script_dir / dirname
        if not src.exists():
            print(f"Warning: {dirname}/ not found, skipping")
            continue
        shutil.copytree(
            src,
            release_dir / dirname,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    # Create zip archive (files at root, no nested folder)
    zip_path = releases_dir / release_name
    shutil.make_archive(str(zip_path), "zip", release_dir)

    # Remove the unzipped folder
    shutil.rmtree(release_dir)

    print(f"Release created: {zip_path}.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fur2amk release package")
    parser.add_argument("version", help="Version string (e.g., 1.0.0)")
    args = parser.parse_args()

    make_release(args.version)


if __name__ == "__main__":
    main()
