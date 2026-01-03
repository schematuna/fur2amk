"""
fur2amk

Requires furnace files saved in Furnace 0.6pre5 or later

Requires all samples to be converted to BRR format prior to use.

Furnace projects may require optimization if AMK throws an error about ARAM.
There are two ways to do this:
    1. decrease the SNES echo delay in the chip manager
    2. reduce sample sizes by downsampling or trimming
        - need to switch to 8 or 16 bit PCM first, edit, then back to BRR

Gain handling:
    If the gain macro is used in Furnace, the first gain value is used as the primary gain setting for the instrument. 
    Any additional gain values are handled via remote commands.
    If the gain macro is unused then the gain setting in the instrument SNES tab is used.

Jump commands:
    You can use one instance of the "Jump to Order" command 0Bxx. 
    The last instance of the command will be used to place the intro marker in the amk output.

Volume/pan slide commands:
    To avoid volume/pan spam, you should use the volume/pan slide commands built in to Furnace rather than the interpolate option.

Wavetables are not supported. All instruments must use samples or noise.

Fades:  
    Prefer gain and volume slides over interpolate actions. This will save space in the output.
    
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

from src.FurnaceParser import FurnaceParser
from src.FurnaceConverter import FurnaceConverter
from src.AMKWriter import AMKWriter
from src.BRRHandler import BRRConverter, BRRSample 


# TODO: support mid-sample loop points in BRR validation/writing
#       warn if tick rate is not 60Hz (NTSC)... is PAL supported?
#       get game name from Furnace module metadata if available
#       support global tuning
#       support 0D, skip to next order command
#       preserve furnace channel names
#       look into alternative ADSR handling (Furnace has more options than AMK)
#       "Divider" BPM control
#       Recommended furnace pre-emphasis settings?

# --------------------------------------------------------------------------------------

class Config:
    flags: Dict[str, List[Any]] = {
        'nosmpl': [False, 'bool'],        # Skip sample conversion/dumping
        'diag': [False, 'bool'],          # Diagnostic logging
        'legato': [True, 'bool'],         # Whether or not to apply $F4 $02
        # ARAM checking
        'aram_check': [True, 'bool'],           # Emit an ARAM usage warning after generation
        'aram_sample_budget_kb': [52, 'int'],   # Conservative sample budget in KB (approx)
    }

    flag_aliases: Dict[str, str] = {
        'ns': 'nosmpl',
    }

    @staticmethod
    def flag(name: str) -> Any:
        if name in Config.flags:
            return Config.flags[name][0]
        # try alias lookup
        if name in Config.flag_aliases:
            return Config.flags[Config.flag_aliases[name]][0]
        raise KeyError(f"Unknown flag '{name}'")

    @staticmethod
    def set_flag(flag: str, value: str) -> None:
        # alias expansion
        key = Config.flag_aliases.get(flag, flag)
        if key not in Config.flags:
            raise KeyError(f"Unknown flag '{flag}'")

        current = Config.flags[key]
        default_val, ftype = current[0], current[1]

        if ftype == 'bool':
            if isinstance(value, bool):
                current[0] = value
            else:
                v = str(value).strip().lower()
                if v in ('1', 'true', 'yes', 'y', 'on'):
                    current[0] = True
                elif v in ('0', 'false', 'no', 'n', 'off'):
                    current[0] = False
                else:
                    raise ValueError(f"Invalid bool for {key}: {value}")
        elif ftype == 'int':
            current[0] = int(value)
        elif ftype == 'real':
            current[0] = float(value)
        elif ftype == 'string' or ftype == 'time':
            current[0] = str(value)
        elif ftype == 'hex':
            # enforce exact hex length if provided (third entry)
            hex_len = current[2] if len(current) > 2 else None
            vv = value.strip().lower().removeprefix('0x').replace(' ', '')
            if hex_len is not None and len(vv) not in (hex_len, hex_len * 2):
                # allow bytes (space-less) or nibble count; keep simple
                # we won’t normalize here; we just store the string
                pass
            # basic validate
            int(vv or '0', 16)
            current[0] = vv
        else:
            current[0] = value


# --------------------------------------------------------------------------------------
# Main


def parse_cli(argv: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    if len(argv) < 2:
        usage = (
            'Usage: python fur2amk.py <furnace_file.fur> <flags>'
        )
        print(usage)
        sys.exit(1)

    module_path = argv[1]
    if not os.path.exists(module_path):
        print(f"Error: {module_path} does not exist.")
        sys.exit(1)

    if len(argv) >= 2 and len(argv) % 2 != 0:
        print('Error: Missing flag argument (flags must be in pairs).')
        sys.exit(1)

    pairs: List[Tuple[str, str]] = []
    i = 2
    while i < len(argv):
        pairs.append((argv[i], argv[i + 1]))
        i += 2
    return module_path, pairs


def main() -> None:
    module_path, flag_pairs = parse_cli(sys.argv)

    # Apply CLI flags
    for flag, arg in flag_pairs:
        name = flag.lstrip('-').strip()
        try:
            Config.set_flag(name, arg)
        except (ValueError, KeyError) as e:
            print(f"Flag error for '{flag}': {e}")
            sys.exit(1)

    # Load module (Furnace)
    parser = FurnaceParser()
    module = parser.parse(module_path)

    # Build AMK object
    converter = FurnaceConverter()
    amk_data = converter.convert(module)

    amk_writer = AMKWriter(amk_data, module_path)

    # Output txt file
    song_name = os.path.splitext(os.path.basename(module_path))[0]
    out_path = os.path.join('music', f'{song_name}.txt')
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(amk_writer.get_text())
        print(f"Wrote {out_path}")

    # Attempt to dump samples to BRR files (unless disabled)
    if not bool(Config.flag('nosmpl')):
        path_name = os.path.splitext(os.path.basename(module_path.replace('\\', '/')))[0]
        sample_dir = os.path.join('music', path_name)
        os.makedirs(sample_dir, exist_ok=True)
        samples = list[BRRSample]()
        for s in module.Samples:
            samples.append(BRRSample(name=s.name, index=s.index, brr_data=s.brr_raw, loop_start=s.loop_start, loop_end=s.loop_end))
        brr_converter = BRRConverter()
        brr_converter.dump_samples_to_brr(sample_dir, samples)


if __name__ == "__main__":
    main()
