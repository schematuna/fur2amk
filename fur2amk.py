"""
fur2amk

Requires furnace files saved in Furnace 0.6pre5 or later

Requires all samples to be converted to BRR format prior to use.

Furnace projects may require optimization if AMK throws an error about ARAM.
There are a few ways to do this:
    1. decrease the SNES echo delay in the chip manager
    2. reduce sample sizes by downsampling or trimming
        - need to switch to 8 or 16 bit PCM first, edit, then back to BRR
    3. replace interpolated commands with slide commands

Gain handling:
    If the gain macro is used in Furnace, the first gain value is used as the primary gain setting for the instrument. 
    Any additional gain values are handled via remote commands.
    If the gain macro is unused then the gain setting in the instrument SNES tab is used.

Jump commands:
    You can use one instance of the "Jump to Order" command 0Bxx. 
    The last instance of the command will be used to place the intro marker in the amk output.

Wavetables are not supported. All instruments must use samples or noise.
    
"""

from __future__ import annotations

import os
import logging
import sys
from typing import List
import argparse


from src.FurnaceParser import FurnaceParser
from src.FurnaceConverter import FurnaceConverter
from src.AMKWriter import AMKWriter
from src.BRRHandler import BRRConverter, BRRSample 
from copy_to_amk import main as copy_to_amk_main

# TODO: support mid-sample loop points in BRR validation/writing
#       get game name from Furnace module metadata if available
#       support global tuning
#       support 0D, skip to next order command
#       preserve furnace channel names
#       look into alternative ADSR handling (Furnace has more options than AMK)
#       "Divider" BPM control
#       Recommended furnace pre-emphasis settings?

# --------------------------------------------------------------------------------------


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

    parser.add_argument(
        '-c', '--copy',
        action='store_true',
        default=False,
        help='Run copy_to_amk.py after conversion'
    )
    
    args = parser.parse_args(argv[1:])
    
    # Validate furnace file exists
    if not os.path.exists(args.furnace_file):
        parser.error(f"Furnace file does not exist: {args.furnace_file}")
    
    return args


def main() -> None:
    args = parse_args(sys.argv)
    
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

    if args.copy:
        copy_to_amk_main([
            '--amk-dir', '..\AddmusicK_1.0.11',
            '--song', song_name
        ])

if __name__ == "__main__":
    main()
