#!/usr/bin/env python3
"""
Copy fur2amk outputs (MML and BRR samples) into an AddmusicK directory.

- Scans fur2amk/music/*.txt for songs unless --song is specified
- Reads #path from the MML to determine the target samples subfolder
- Copies the MML into <amk_dir>/music/<song>.txt
- Copies the BRRs into <amk_dir>/samples/<path>/

Usage examples (PowerShell):
  python .\copy_to_amk.py --amk-dir ..\AddmusicK_1.0.11 --song "Sunken Lights"
  python .\copy_to_amk.py --amk-dir ..\AddmusicK_1.0.11
"""
import argparse
import os
import re
import shutil
import sys
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AMK_DIR = os.path.normpath(os.path.join(HERE, '..', 'AddmusicK_1.0.11'))
MUSIC_SRC_DIR = os.path.join(HERE, 'music')

PATH_RE = re.compile(r'^\s*#path\s*"([^"]+)"', re.IGNORECASE)


def parse_args(argv):
    ap = argparse.ArgumentParser(description='Copy fur2amk outputs to AddmusicK.')
    ap.add_argument('--amk-dir', default=DEFAULT_AMK_DIR, help='Path to AddmusicK directory (contains AddmusicK.exe)')
    ap.add_argument('--song', default=None, help='Song name (stem of .txt). If omitted, copy all .txt in fur2amk/music')
    ap.add_argument('--dry-run', action='store_true', help='Show planned copies without writing')
    return ap.parse_args(argv)


def find_mmls(song: str | None):
    if song:
        src = os.path.join(MUSIC_SRC_DIR, f'{song}.txt')
        return [src] if os.path.exists(src) else []
    return sorted(glob(os.path.join(MUSIC_SRC_DIR, '*.txt')))


def extract_path_from_mml(mml_path: str) -> str | None:
    try:
        with open(mml_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                m = PATH_RE.match(line)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


def copy_one(mml_path: str, amk_dir: str, dry_run: bool = False) -> tuple[bool, str]:
    song = os.path.splitext(os.path.basename(mml_path))[0]
    path_str = extract_path_from_mml(mml_path) or song

    # Destinations
    music_dst_dir = os.path.join(amk_dir, 'music')
    samples_dst_dir = os.path.join(amk_dir, 'samples', path_str)

    # Sources
    samples_src_dir = os.path.join(MUSIC_SRC_DIR, song)

    actions: list[str] = []
    # Copy MML
    mml_dst = os.path.join(music_dst_dir, f'{song}.txt')
    actions.append(f'MML: {mml_path} -> {mml_dst}')
    # Copy BRRs
    brrs = []
    if os.path.isdir(samples_src_dir):
        brrs = sorted(glob(os.path.join(samples_src_dir, '*.brr')))
    else:
        # No sample dir is okay if song uses only default samples
        pass
    for b in brrs:
        actions.append(f'BRR: {b} -> {os.path.join(samples_dst_dir, os.path.basename(b))}')

    if dry_run:
        for a in actions:
            print('[dry-run]', a)
        return True, f'{song} (dry-run)'

    try:
        os.makedirs(music_dst_dir, exist_ok=True)
        os.makedirs(samples_dst_dir, exist_ok=True)
        shutil.copy2(mml_path, mml_dst)
        for b in brrs:
            shutil.copy2(b, os.path.join(samples_dst_dir, os.path.basename(b)))
        return True, song
    except Exception as e:
        return False, f'{song}: {e}'


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    amk_dir = os.path.abspath(args.amk_dir)
    if not os.path.isdir(amk_dir):
        print(f'Error: AMK dir not found: {amk_dir}')
        return 2
    mmls = find_mmls(args.song)
    if not mmls:
        target = f'"{args.song}"' if args.song else 'any .txt in music'
        print(f'No MMLs found for {target}.')
        return 1
    ok = 0
    for m in mmls:
        res, msg = copy_one(m, amk_dir, args.dry_run)
        if res:
            print(f'Copied: {msg}')
            ok += 1
        else:
            print(f'Failed: {msg}')
    if ok == 0:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
