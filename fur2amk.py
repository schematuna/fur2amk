"""
fur2amk

Requires furnace files saved in Furnace 0.6pre5 or later

Requires all samples to be converted to BRR format prior to use.

"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from furnace_parser import (
    FurnaceParser,
    FurnaceModule,
    FurnacePatternRow,
)

# TODO: support mid-sample loop points in BRR validation/writing
#       warn if tick rate is not 60Hz (NTSC)... is PAL supported?
#       get game name from Furnace module metadata if available
#       support global tuning
#       support 0B and 0D, special furnace order jumps for advanced looping
#       preserve furnace channel names
#       grab SNES chip flags for echo/FIR

# --------------------------------------------------------------------------------------

class Config:
    flags: Dict[str, List[Any]] = {
        'nosmpl': [False, 'bool'],        # Skip sample conversion/dumping
        'diag': [False, 'bool'],          # Diagnostic logging
        'game': ['', 'string'],           # Game title
        'length': ['', 'time'],           # SPC length
        'vcurve': ['accurate', 'string'], # accurate, linear, x^2
        'panning': ['accurate', 'string'],# linear, accurate
        'tspeed': [False, 'bool'],        # Use txxx for Axx commands
        'legato': [True, 'bool'],         # Whether or not to apply $F4 $02
        'vcmd': ['v', 'string'],          # Which volume command to use for the v column
        'mcmd': ['v', 'string'],          # Which volume command to use for the M effect
        'svcmd': ['v', 'string'],         # Which volume command to use for global sample volume
        'ivcmd': ['v', 'string'],         # Which volume command to use for global instrument volume
        # ARAM checking
        'aram_check': [True, 'bool'],           # Emit an ARAM usage warning after generation
        'aram_sample_budget_kb': [52, 'int'],   # Conservative sample budget in KB (approx)
    }

    flag_aliases: Dict[str, str] = {
        'ns': 'nosmpl',
        'gm': 'game',
        'ln': 'length',
        'vc': 'vcurve',
        'p': 'panning',
        'ts': 'tspeed',
        'l': 'legato',
        'v': 'vcmd',
        'm': 'mcmd',
        'sv': 'svcmd',
        'iv': 'ivcmd',
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
# Event model (simplified, compatible shape for MML)


class EventState:
    def __init__(self) -> None:
        # Mirror keys used by it2amk where possible
        self.state_d: Dict[str, Any] = {
            '': None, 'M': None, 'S': 0x90, 'X': 0x80,
            'E': 0x00, 'H': 0x00, 'I': 0x00, 'J': 0x00,
            'Q': 0x00, 'R': 0x00, 'v': None, '@': None,
            'IV': None, 'SV': None, 'EV': None, 'EX': 32, 'EE': None,
            'eflag': False, 'pflag': False, 'H': 0x00, 'Hon': False,
            'Z1': None,
            'a': 0x00, 'b': 0x00, 'c': 0x00, 'd': 0x00, 'l': 0x00, 'r': 0x00,
            'D': 0x00, 'N': 0x00, 'P': 0x00,
        }


class Event:
    def __init__(self, tick: int, effect: str, value: Any, visible: bool = True) -> None:
        self.tick = tick
        self.effect = effect
        self.value = value
        self.visible = visible


class EventTable:
    """Build an event list from the FurnaceModule.    """

    def __init__(self, module: FurnaceModule) -> None:
        self.events: List[List[Event]] = [[] for _ in range(8)]
        self.g_events: List[Event] = []
        self.module = module
        self.states = [EventState() for _ in range(8)]
        self.g_state_d: Dict[str, Any] = {'T': None, 'V': None}
        self.used_samples: set[int] = set()
        # map sample index -> (filename, tuning_hex)
        self.sample_dict: Dict[int, Tuple[str, str]] = {}
        self.ins_dict: Dict[int, Any] = {}
        # List of (instrument_index, sample_index) pairs to emit in #instruments
        self.ins_list: List[Tuple[int, int]] = []
        self.loop_tick: int = 0
        self.convert()

    def convert(self) -> None:
        # Build trivial instrument/sample dictionary from module
        for s in self.module.Samples:
            self.used_samples.add(s.index)
            # Build sample filename and tuning string
            fname = f"{s.index:02d}_" + (s.name or f"Sample{s.index}").replace(' ', '_') + '.brr'
            tuning_word = 0x0100
            try:
                if s.c4_rate and s.c4_rate > 0:
                    # MAGIC NUMBERS to convert from c4_rate to AMK instrument tuning value
                    # stolen from it2amk's SampConv
                    val = int(round(float(s.c4_rate) * 768 / 12539))
                    tuning_word = max(0, min(0xFFFF, val))
            except Exception:
                tuning_word = 0x0100
            tune_str = f"${(tuning_word >> 8) & 0xFF:02X} ${(tuning_word & 0xFF):02X}"
            self.sample_dict[s.index] = (fname, tune_str)
        # For each instrument, gather unique samples referenced by its sample map
        for ins in self.module.Instruments:
            samples_for_ins: List[int] = []
            try:
                if ins.use_sample_map and ins.sample_table:
                    # Collect unique, non-zero sample indices from the 120-entry map
                    uniq: List[int] = []
                    seen = set()
                    for (_note_to_play, samp_to_play) in ins.sample_table:
                        # Furnace SM uses 0-based sample indices
                        try:
                            sidx1 = int(samp_to_play)
                        except Exception:
                            sidx1 = -1
                        if sidx1 < 0:
                            continue
                        if sidx1 not in seen:
                            seen.add(sidx1)
                            uniq.append(sidx1)
                    samples_for_ins = uniq
                else:
                    # Use initial_sample (0-based) if available
                    try:
                        if ins.initial_sample is not None and int(ins.initial_sample) >= 0:
                            sidx1 = int(ins.initial_sample)
                        else:
                            sidx1 = 0
                    except Exception:
                        sidx1 = 0
                    samples_for_ins = [sidx1]
            except Exception:
                try:
                    sidx1 = int(ins.initial_sample) if (ins.initial_sample is not None and int(ins.initial_sample) >= 0) else 0
                except Exception:
                    sidx1 = 0
                samples_for_ins = [sidx1]
            # Track used samples and populate instrument entries
            for sidx in samples_for_ins:
                self.used_samples.add(sidx)
                self.ins_list.append((ins.index, sidx))


# --------------------------------------------------------------------------------------
# MML writer (streamlined for now)


class MMLState:
    def __init__(self) -> None:
        self.state_d: Dict[str, Any] = {
            'o': None, 'h': 0, 'v': None, 'q': None,
            'tune': 0x00, 'y': (10, 0, 0), 'p': (0, 0, 0), 'trem': (0, 0, 0),
            'echo': 0x00, '@': 0, 'dgain': None, 'note': None,
            'echof': False, 'n': None, 'amp': 0x00, 'gain': None,
        }
        self.hstate_d: Dict[str, Any] = {
            '': None, 'M': None, 'S': 0x90, 'X': 0x80,
            'E': 0x00, 'H': 0x00, 'I': 0x00, 'J': 0x00,
            'Q': 0x00, 'R': 0x00, 'v': None, '@': None,
            'IV': None, 'SV': None, 'EV': None, 'EX': None, 'EE': None, 'H': 0x00,
            'Z1': None,
        }


class MML:
    def __init__(self, event_table: EventTable, module_path: str) -> None:
        self.txt: str = ''
        self.event_table = event_table
        self.module_path = module_path
        self.states = [MMLState() for _ in range(8)]
        self.g_state: Dict[str, Any] = {'evoll': 0, 'evolr': 0}
        self.echo_set = False

        self.add_amk_header()
        self.add_spc_info()
        self.add_sample_info()
        self.add_ins_info()
        self.convert()

    # Sections
    def add_amk_header(self) -> None:
        self.txt += '#amk 2\n\n'

    # --- helpers ---
    def _row_kind(self, row: FurnacePatternRow) -> str:
        """Classify a Furnace row for emission.

        Returns: 'note' | 'off' | 'cut' | 'empty'.
        OFF = 255, CUT = 254 as per parser comment.
        """
        n = row.Note
        if n is None:
            return 'empty'
        try:
            v = int(n)
        except Exception:
            return 'empty'
        if v == 255:
            return 'off'
        if v == 254:
            return 'cut'
        if 0 <= v <= 179:
            return 'note'
        return 'empty'

    def _divisors(self, n: int) -> List[int]:
        n = int(n)
        if n <= 0:
            return [1]
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return sorted(divs)

    def _run_to_denoms(self, run_rows: int, base_den: int) -> List[int]:
        """Decompose a run of rows into a list of AMK length denominators to tie.

        Each row is 1/base_den. We choose chunks that are divisors of base_den
        and sum to run_rows. For each chunk, the length number is base_den/chunk.
        Example: base_den=16, run=3 -> chunks [2,1] => denoms [8,16] -> c8^16.
        """
        run = max(1, int(run_rows))
        bd = max(1, int(base_den))
        divs = self._divisors(bd)
        # allowed chunks are divisors of base_den
        chunks = sorted(divs, reverse=True)
        out: List[int] = []
        rem = run
        while rem > 0:
            # pick largest chunk <= rem
            pick = None
            for c in chunks:
                if c <= rem:
                    pick = c
                    break
            if pick is None:
                # fallback to 1-row chunks (shouldn't happen since 1 divides bd)
                pick = 1
            out.append(bd // pick)
            rem -= pick
        return out

    def add_spc_info(self) -> None:
        # Emit AddmusicK readme-style #spc block with #title/#game/#author/#length
        lines = ['#spc', '{']
        mod = self.event_table.module
        title = getattr(mod, 'SongName', '') or ''
        author = getattr(mod, 'Author', '') or ''
        info_align_width = 8
        if title:
            lines.append(f'    {'#title':<{info_align_width}} "{title}"')
        if Config.flag('game'):
            lines.append(f'    {'#game':<{info_align_width}} "{Config.flag("game")}"')
        if author:
            lines.append(f'    {'#author':<{info_align_width}} "{author}"')
        if Config.flag('length'):
            lines.append(f'    {'#length':<{info_align_width}} "{Config.flag("length")}"')
        # Optional comment: use first line of Message if present
        msg = str(getattr(mod, 'Message', '') or '').strip()
        if msg:
            first_line = msg.splitlines()[0]
            lines.append(f'    {'#comment':<{info_align_width}} "{first_line}"')
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_sample_info(self) -> None:
        path_name = os.path.splitext(os.path.basename(self.module_path.replace('\\', '/')))[0]
        sample_dir = os.path.join('music', path_name)
        os.makedirs(sample_dir, exist_ok=True)
        # Attempt to dump samples to BRR files (unless disabled)
        if not bool(Config.flag('nosmpl')):
            self._dump_samples_to_brr(sample_dir)
        sample_lines = [f'#path "{path_name}"', '', '#samples', '{', '    #optimized']
        # Prefer listing only BRRs we actually generated to avoid missing files
        mod = self.event_table.module
        for samp in sorted(mod.Samples, key=lambda x: x.index):
            base = f"{samp.index:02d}_" + (samp.name or f'Sample{samp.index}').replace(' ', '_')
            brr_rel = f'{base}.brr'
            brr_abs = os.path.join(sample_dir, brr_rel)
            if os.path.exists(brr_abs) and os.path.getsize(brr_abs) > 0:
                # Match AMK style: list quoted filenames only
                sample_lines.append(f'    "{brr_rel}"')
        sample_lines.append('}')
        # Even if no extra BRRs, we still want #samples { #optimized } for clarity
        self.txt += '\n'.join(sample_lines) + '\n\n'

    def add_ins_info(self) -> None:
        if not self.event_table.ins_list:
            return
        lines = ['#instruments', '{']
        # Assign AMK instrument numbers starting at 30 in the order we emit
        # Map of (instrument_index, sample_index) -> AMK instrument number
        self.insnum_map: Dict[Tuple[int, int], int] = {}
        next_num = 30
        name_col = max(len(name) for name, _ in self.event_table.sample_dict.values())
        # get max sample name length for alignment
        name_field_width = name_col + 2  # account for quotes
        for ins_idx, samp_idx in self.event_table.ins_list:
            # Resolve sample filename and tuning
            samp_entry = self.event_table.sample_dict.get(samp_idx)
            if not samp_entry:
                # Fallback to first sample
                samp_entry = next(iter(self.event_table.sample_dict.values()), ("Sample1.brr", "$01 $00"))
            samp_name, samp_tuning = samp_entry
            # ADSR/GAIN
            ins = self.event_table.module.Instruments[ins_idx]
            # Default: no envelope -> $00 $00 $7F
            da = 0x00
            sr = 0x00
            ga = 0x7F
            try:
                if ins.sn_flags is not None and (ins.sn_flags & 0x10):
                    d = int(ins.sn_decay or 0)
                    a = int(ins.sn_attack or 0)
                    ssv = int(ins.sn_sustain or 0)
                    rv = int(ins.sn_release or 0)
                    da = ((d & 0x7) | 0x8) << 4 | (a & 0xF)
                    sr = ((ssv & 0x7) << 5) | (rv & 0x1F)
                    # When ADSR is on, GA is ignored by AMK; keep placeholder $7F.
                else:
                    # ADSR off: use raw GAIN value if provided (clamped to 7-bit)
                    if ins.sn_gain is not None:
                        ga = max(0, min(0x7F, int(ins.sn_gain)))
            except Exception:
                pass
            quoted_name = f'"{samp_name}"'
            lines.append(f'    {quoted_name:<{name_field_width}} ${da:02X} ${sr:02X} ${ga:02X} {samp_tuning} ;@{next_num}')
            self.insnum_map[(ins_idx, samp_idx)] = next_num
            next_num += 1
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    # Conversion
    def convert(self) -> None:
        # If we have parsed orders/patterns, emit simple note streams with basic durations per channel.
        mod = self.event_table.module

        # Global tempo and volume
        base_den = 4 * (mod.HighlightA or 4)
        if base_den <= 0:
            base_den = 16
        tps = float(getattr(mod, 'TicksPerSecond', 0.0) or 0.0)
        spd = int(getattr(mod, 'Speed1', 0) or 0)
        if spd <= 0:
            spd = 6
        if tps > 0:
            bpm = max(1, int(round(240.0 * tps / (base_den * spd))))
        else:
            bpm = int(getattr(mod, 'IT', 125) or 125)

        amk_tempo = bpm * 8192 // 20025

        # get global volume if set
        gvol = getattr(mod, 'GV', None)
        # treat w200 as 100%
        amk_volume = min(int(gvol * 200), 255)

        self.txt += f'w{amk_volume} t{amk_tempo}\n\n'

        if getattr(mod, 'OrdersPerChannel', None) and getattr(mod, 'PatternsByChannel', None) and any(mod.OrdersPerChannel):
            for c in range(mod.NumChannels):
                self.txt += f'#%d\n' % c
                current_oct = None
                current_ins = None  # Furnace instrument index
                current_amk_ins: Optional[int] = None  # AMK @ number actually in use
                current_vol: Optional[int] = None  # 0..255
                tokens: List[str] = []
                orders = mod.OrdersPerChannel[c] if c < len(mod.OrdersPerChannel) else []
                patmap = mod.PatternsByChannel[c] if c < len(mod.PatternsByChannel) else {}
                # Flatten rows for this channel
                flat_rows: List[FurnacePatternRow] = []
                for pat in orders:
                    rows = patmap.get(pat)
                    if rows:
                        flat_rows.extend(rows)
                    else:
                        flat_rows.extend([FurnacePatternRow() for _ in range(mod.PatternLength)])

                i = 0
                N = len(flat_rows)
                cur_order_num = -1
                # TODO: use AMK group labels for identical patterns/lines
                while i < N:
                    orderNum = i // mod.PatternLength
                    if cur_order_num != orderNum:
                        cur_order_num = orderNum
                        tokens.append(f'\n; order {orderNum}\n')
                    row = flat_rows[i]
                    kind = self._row_kind(row)
                    # Track instrument changes (don’t emit @ yet; defer until note to choose sample variant)
                    if row.Ins is not None and row.Ins != current_ins and row.Ins != 255:
                        current_ins = int(row.Ins)
                        current_amk_ins = None  # force re-select on next note

                    # Determine if this is a note or rest
                    if kind == 'note':
                        note_idx = int(row.Note)  # type: ignore[arg-type]
                        # Ensure we have some instrument context
                        if current_ins is None or current_ins == 255:
                            current_ins = 0
                            current_amk_ins = None
                        # Determine which sample this note should use for this instrument
                        amk_num = self._resolve_amk_instrument_for_note(current_ins, note_idx)
                        if amk_num is not None and amk_num != current_amk_ins:
                            tokens.append(f'@{amk_num}')
                            current_amk_ins = amk_num
                        name, octv = self._note_name_and_octave(note_idx)
                        if current_oct != octv:
                            tokens.append(f'o{octv}')
                            current_oct = octv
                        # Apply volume if present on this row and changed
                        if row.Vol is not None:
                            v = max(0, min(255, int(row.Vol)))
                            if current_vol != v:
                                tokens.append(f'v{v}')
                                current_vol = v
                        # Count run length of same note continuing (no new note starts)
                        run = 1
                        j = i + 1
                        while j < N:
                            r2 = flat_rows[j]
                            k2 = self._row_kind(r2)
                            # stop if next row starts a new note, OFF/CUT, or instrument change
                            if (k2 in ('note','off','cut')) or (r2.Ins is not None and r2.Ins != current_ins and r2.Ins != 255):
                                break
                            run += 1
                            j += 1
                        # Emit note with duration expressed as ties if run>1
                        # Always emit explicit duration numbers and numeric ties
                        denoms = self._run_to_denoms(run, base_den)
                        note_token = f'{name}{denoms[0]}'
                        for d in denoms[1:]:
                            note_token += f'^{d}'
                        # First segment includes note name
                        tokens.append(note_token)
                        i = j
                        continue
                    else:
                        # Rest or OFF/CUT run
                        run = 1
                        j = i + 1
                        while j < N:
                            r2 = flat_rows[j]
                            if self._row_kind(r2) == 'note':
                                break
                            run += 1
                            j += 1
                        # Always emit explicit rest duration and numeric ties
                        denoms = self._run_to_denoms(run, base_den)
                        rest_token = f'r{denoms[0]}'
                        for d in denoms[1:]:
                            rest_token += f'^{d}'
                        tokens.append(rest_token)
                        i = j
                        continue
                self.txt += ' '.join(tokens) + '\n\n'
            return
        # Fallback: emit 8 empty channels as before
        for c in range(8):
            self.txt += f'#%d\n' % c
            self.txt += '\n'

    def validate_and_fix_brr_data(self, data: bytes, loop_end: int) -> bytes:
        """Validate BRR data and fix invalid nibbles if needed.

        Args:
            data: Raw BRR data (multiple of 9 bytes).
            loop_end: Loop sample offset
        Returns:
            Validated/fixed BRR data.
        """
        # check that last block has end flag set
        if len(data) % 9 != 0:
            raise ValueError("BRR data length is not a multiple of 9")
        fixed_data = bytearray(data)

        loop_end_byte = (loop_end // 16 * 9) - 9
        # loop over every 9-byte block and set loop and end flags appropriately
        for i in range(0, len(fixed_data), 9):
            # check loop flag
            loop_flag = (fixed_data[i] & 0x02) != 0

            # debug
            # if loop_flag:
            #     print(f"byte {i}: loop flag is set, loop_end_byte={loop_end_byte}")
            # if (i==loop_end_byte):
            #     print(f"byte {i}: expected loop end byte")

            end_flag = (fixed_data[i] & 0x01) != 0
            if (i == loop_end_byte) and not loop_flag:
                print(f"[diag] warning: BRR loop end block missing loop flag; fixing")
                fixed_data[i] |= 2

            # TODO : furnace seems to set loop flag on EVERY block
            # not even sure why this works. Removing them breaks things.
            # elif (i != loop_end_byte) and loop_flag:
            #     print(f"[diag] warning: BRR block erroneously has loop flag; fixing")
            #     fixed_data[i] &= 0xFD # 0xFF ^ 0x02

            # debug
            # elif (i == loop_end_byte) and loop_flag:
            #     print(f"[diag] info: BRR loop end block has correct loop flag")

            # end block can be missing for some furnace BRR samples
            # seems to happen for samples that are converted to BRR from PCM
            if (i + 9 >= len(fixed_data)) and not end_flag:
                print(f"[diag] warning: BRR last block missing end flag; fixing")
                fixed_data[i] |= 1
            
            # debug
            # elif (i + 9 >= len(fixed_data)) and end_flag:
            #     print(f"[diag] info: BRR last block has correct end flag")
                
        return fixed_data

    def _dump_samples_to_brr(self, out_dir: str) -> None:
        mod = self.event_table.module
        if not getattr(mod, 'Samples', None):
            return
        total = 0
        created = 0
        for s in mod.Samples:
            total += 1
            # Target BRR path
            # Prefix with index to avoid name collisions and keep ordering stable
            fname_base = (f"{s.index:02d}_" + (f"{s.name}".strip() or f"Sample{s.index}")).replace(' ', '_')
            brr_path = os.path.join(out_dir, fname_base + '.brr')
            s.brr_path = brr_path
            # Always overwrite existing BRR: remove it first if present
            try:
                if os.path.exists(brr_path):
                    os.remove(brr_path)
            except OSError:
                pass
            # If the sample already contains raw BRR data, wrap it with AMK 2-byte loop header and write
            if s.brr_raw:
                try:
                    data = s.brr_raw
                    # Ensure len % 9 == 0 by adding header; raw data itself should be multiple of 9
                    if len(data) % 9 != 0:
                        # Truncate to the nearest lower whole block to satisfy AMK; log diagnostics
                        trunc = (len(data) // 9) * 9
                        if bool(Config.flag('diag')):
                            print(f"[diag] warning: BRR raw not block-aligned ({len(data)}); truncating to {trunc}")
                        data = data[:trunc]
                    
                    data = self.validate_and_fix_brr_data(data, s.loop_end)

                    loop_off = 0
                    if s.loop_start is not None and s.loop_start >= 0:
                        # Convert PCM loop start (samples) to BRR byte offset: floor(loop/16)*9
                        loop_off = (int(s.loop_start) // 16) * 9
                    header = bytes((loop_off & 0xFF, (loop_off >> 8) & 0xFF))
                    with open(brr_path, 'wb') as f:
                        f.write(header + data)
                    created += 1
                    if bool(Config.flag('diag')):
                        print(f"[diag] wrote BRR (raw+hdr): {os.path.basename(brr_path)} loop_off={loop_off}")
                    continue
                except Exception:
                    if bool(Config.flag('diag')):
                        print(f"[diag] failed to write raw BRR for {s.index:02d} {s.name}")
            else:
                print(f"[diag] info: sample {s.index:02d} {s.name} has no raw BRR data, skipping")
            
        if bool(Config.flag('diag')):
            print(f"[diag] summary: samples={total} brr_created={created}")

    def _note_name_and_octave(self, i: int) -> Tuple[str, int]:
        # Map Furnace note index (0=C-0) to AMK note name and octave using oN
        names = ['c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+', 'a', 'a+', 'b']
        note = i % 12
        octave = i // 12 - 5  # align with fur2tad convention
        return names[note], octave

    def _resolve_amk_instrument_for_note(self, ins_idx: int, note_idx: int) -> Optional[int]:
        """Pick the AMK instrument number for this Furnace instrument at a given note.

        Uses the instrument's sample map (INS2 'SM') when present; else the initial sample.
        """
        try:
            mod = self.event_table.module
            if ins_idx <= 0 or ins_idx > len(mod.Instruments):
                return None
            ins = mod.Instruments[ins_idx]

            # Determine sample index to use
            samp_idx: Optional[int] = None
            n = int(note_idx)
            if ins.use_sample_map and ins.sample_table:
                # Furnace provides 120 entries; clamp into range
                n120 = n
                if n120 < 0:
                    n120 = 0
                if n120 >= len(ins.sample_table):
                    n120 = n120 % len(ins.sample_table)
                _note_to_play, samp_to_play = ins.sample_table[n120]
                try:
                    sidx_raw = int(samp_to_play)
                except Exception:
                    sidx_raw = -1
                if sidx_raw >= 0:
                    samp_idx = sidx_raw + 1
            if samp_idx is None:
                try:
                    if ins.initial_sample is not None and int(ins.initial_sample) >= 0:
                        samp_idx = int(ins.initial_sample)
                    else:
                        samp_idx = 0
                except Exception:
                    samp_idx = 1
            # Map to AMK instrument number
            if hasattr(self, 'insnum_map') and isinstance(self.insnum_map, dict):
                return self.insnum_map.get((ins_idx, int(samp_idx)))
            # Fallback: sequential mapping (unlikely to be correct, but avoids crash)
            return 30 + int(ins_idx)
        except Exception:
            return None

    # Output
    def save(self, filename: str) -> None:
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.txt)


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

    # Build events and MML
    evtbl = EventTable(module)
    mml = MML(evtbl, module_path)

    # Output
    song_name = os.path.splitext(os.path.basename(module_path))[0]
    out_path = os.path.join('music', f'{song_name}.txt')
    mml.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
