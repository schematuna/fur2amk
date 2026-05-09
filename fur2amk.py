"""
fur2amk - Convert Furnace SNES modules to AddmusicK format

See README.md for quick start and ADVANCED.md for detailed usage notes.
"""

from __future__ import annotations

import os
import logging
import sys
import json
from typing import List, Optional
import argparse


from src.reader.FurnaceParser import FurnaceParser
from src.FurnaceConverter.FurnaceConverter import FurnaceConverter
from src.AMKConverter.AMKConverter import AMKConverter
from src.writer.AMKWriter import AMKWriter
from src.writer.BRRHandler import BRRConverter, BRRSample
from copy_to_amk import main as copy_to_amk_main

# TODO: support mid-sample loop points in BRR validation/writing
#       get game name from Furnace module metadata if available
#       support global tuning
#       preserve furnace channel names
#       "Divider" BPM control
#       Recommended furnace pre-emphasis settings?
#       support virtual tempo (simple tempo multiplier)

# --------------------------------------------------------------------------------------


def load_config() -> dict:
    """Load configuration from fur2amk_config.json in the script directory."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fur2amk_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load config.json: {e}. Using defaults.")
    return {}


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert Furnace .fur files to AddmusicK format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Positional argument: furnace file
    parser.add_argument(
        'furnace_file',
        type=str,
        metavar='FURNACE_FILE',
        help='Path to the Furnace .fur file to convert'
    )
    
    # Boolean flags (store_true/store_false)
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--nosmpl', '-ns',
        action='store_true',
        default=False,
        help='Skip sample conversion/dumping'
    )
    
    args = parser.parse_args(argv[1:])
    
    # Validate furnace file exists
    if not os.path.exists(args.furnace_file):
        parser.error(f"Furnace file does not exist: {args.furnace_file}")
    
    return args


def run_conversion(furnace_file: str, out_dir: str, nosmpl: bool = False):
    """Parse, convert, and write outputs. Returns (mml_path, sample_dir or None)."""
    song_name = os.path.splitext(os.path.basename(furnace_file.replace('\\', '/')))[0]

    module = FurnaceParser().parse(furnace_file)
    chiptune_data = FurnaceConverter().convert(module)
    amk_data = AMKConverter().convert(chiptune_data)
    amk_writer = AMKWriter(amk_data, song_name)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{song_name}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(amk_writer.get_text())
    logging.info(f'Wrote {out_path}')

    sample_dir = None
    if not nosmpl:
        sample_dir = os.path.join(out_dir, song_name)
        os.makedirs(sample_dir, exist_ok=True)
        samples = [BRRSample(name=s.name, index=s.index, brr_data=s.brr_raw,
                             loop_start=s.loop_start, loop_end=s.loop_end)
                   for s in module.Samples]
        BRRConverter().dump_samples_to_brr(sample_dir, samples)

    return out_path, sample_dir


def main() -> None:
    args = parse_args(sys.argv)
    config = load_config()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(levelname)-7s %(message)s', level=log_level)

    song_name = os.path.splitext(os.path.basename(args.furnace_file))[0]
    out_path, sample_dir = run_conversion(args.furnace_file, 'music', args.nosmpl)

    amk_dir = config.get('amk_dir')
    if amk_dir:
        copy_to_amk_main(['--amk-dir', amk_dir, '--song', song_name])

if __name__ == "__main__":
    main()
