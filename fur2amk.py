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


from src.FurnaceParser import FurnaceParser
from src.FurnaceConverter import FurnaceConverter
from src.AMKWriter import AMKWriter
from src.BRRHandler import BRRConverter, BRRSample
from copy_to_amk import main as copy_to_amk_main

# TODO: support mid-sample loop points in BRR validation/writing
#       get game name from Furnace module metadata if available
#       support global tuning
#       preserve furnace channel names
#       look into alternative ADSR handling (Furnace has more options than AMK)
#       "Divider" BPM control
#       Recommended furnace pre-emphasis settings?
#       legato by default
#       support virtual tempo (simple tempo multiplier)
#       filter special characters in comments
#       sample fine tune

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


def main() -> None:
    args = parse_args(sys.argv)
    
    # Load configuration
    config = load_config()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(levelname)-7s %(message)s', level=log_level)

    furnace_file = args.furnace_file
    path_name = os.path.splitext(os.path.basename(furnace_file.replace('\\', '/')))[0]

    # Load module (Furnace)
    parser = FurnaceParser()
    module = parser.parse(furnace_file)

    # Build AMK object
    converter = FurnaceConverter()
    amk_data = converter.convert(module)

    amk_writer = AMKWriter(amk_data, path_name)

    # Output txt file
    song_name = os.path.splitext(os.path.basename(furnace_file))[0]
    
    out_path = os.path.join('music', f'{song_name}.txt')
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(amk_writer.get_text())
        logging.info(f"Wrote {out_path}")

    # Attempt to dump samples to BRR files (unless disabled)
    if not args.nosmpl:
        sample_dir = os.path.join('music', path_name)
        os.makedirs(sample_dir, exist_ok=True)
        samples = list[BRRSample]()
        for s in module.Samples:
            samples.append(BRRSample(name=s.name, index=s.index, brr_data=s.brr_raw, loop_start=s.loop_start, loop_end=s.loop_end))
        brr_converter = BRRConverter()
        brr_converter.dump_samples_to_brr(sample_dir, samples)

    # Copy to AddmusicK if configured
    amk_dir = config.get('amk_dir')
    if amk_dir:
        copy_to_amk_main([
            '--amk-dir', amk_dir,
            '--song', song_name
        ])

if __name__ == "__main__":
    main()
