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
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from furnace_parser import (
    FurnaceParser,
    FurnaceModule,
    FurnacePatternRow,
)

# TODO: support mid-sample loop points in BRR validation/writing
#       warn if tick rate is not 60Hz (NTSC)... is PAL supported?
#       get game name from Furnace module metadata if available
#       support global tuning
#       support 0D, skip to next order command
#       preserve furnace channel names

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

class RemoteCommandTiming(Enum):
    DISABLE = 0
    AFTER_START = 1
    BEFORE_END = 2
    KEY_OFF = 3
    RUN_NOW = 4
    KEY_ON = -1

class RemoteCommandTypes(Enum):
    GAIN = 0

class EventRemoteCommand:
    def __init__(
        self,
        command_idx: int,
        event_type: RemoteCommandTiming,
        amk_command_type: RemoteCommandTypes,
        remote_command_arg: Optional[Any] = None,
        amk_command_args: Optional[List[Any]] = None,
    ) -> None:
        self.command_idx = command_idx
        self.event_type = event_type
        self.amk_command_type = amk_command_type
        self.remote_command_arg = remote_command_arg # if there is an extra argument for remote command
        self.amk_command_args = amk_command_args if amk_command_args is not None else []


# subclass for different instrument types?
class EventInstrument:
    def __init__(
        self,
        index: int,
        sample_index: int = None,
        is_noise: bool = False,
        noise_freq: int = 0,
    ) -> None:
        self.index = index
        self.is_noise = is_noise
        if is_noise:
            self.noise_freq = noise_freq
            self.sample_index = None
        else:
            self.sample_index = sample_index
            self.noise_freq = 0
        self.remote_commands: List["EventRemoteCommand"] = []

    @classmethod
    def noise(cls, index: int, noise_freq: int) -> "EventInstrument":
        return cls(index=index, is_noise=True, noise_freq=noise_freq)

    @classmethod
    def sample(cls, index: int, sample_index: int) -> "EventInstrument":
        return cls(index=index, sample_index=sample_index, is_noise=False)


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
        self.ins_list: List[Tuple[int, EventInstrument]] = []
        # which order to place the intro marker at
        self.intro_order: int = None
        self.label_start: int = 1
        self.convert()

    def convert(self) -> None:
        # Build trivial instrument/sample dictionary from module
        for s in self.module.Samples:
            self.used_samples.add(s.index)
            # Build sample filename and tuning string
            fname = f"{s.index:02d}_" + (s.name or f"Sample{s.index}").replace(' ', '_') + '.brr'
            tuning_word = 0x0100
            if s.c4_rate and s.c4_rate > 0:
                # MAGIC NUMBERS to convert from c4_rate to AMK instrument tuning value
                # stolen from it2amk's SampConv
                val = int(round(float(s.c4_rate) * 768 / 12539))
                tuning_word = max(0, min(0xFFFF, val))
            tune_str = f"${(tuning_word >> 8) & 0xFF:02X} ${(tuning_word & 0xFF):02X}"
            self.sample_dict[s.index] = (fname, tune_str)
        # For each instrument, gather unique samples referenced by its sample map
        for ins in self.module.Instruments:
            instruments: List[EventInstrument] = []
            # first, check if this is a noise instrument
            if ins.snes_macro_data.is_noise:
                if ins.snes_macro_data.noise_freq is None:
                    noise_freq = 29  # default noise freq if unset
                    print(f"Warning: Instrument {ins.index} is a noise instrument but has no noise frequency set; You should set it explicitly in Furnace.", file=sys.stderr)
                instruments.append(EventInstrument.noise(index=ins.index, noise_freq=ins.snes_macro_data.noise_freq))
            elif ins.use_sample_map and ins.sample_table:
                # Collect unique, non-zero sample indices from the 120-entry map
                uniq: List[int] = []
                seen = set()
                for (_note_to_play, samp_to_play) in ins.sample_table:
                    # Furnace SM uses 0-based sample indices
                    sidx1 = int(samp_to_play)
                    if sidx1 < 0:
                        continue
                    if sidx1 not in seen:
                        seen.add(sidx1)
                        uniq.append(sidx1)
                for sidx in uniq:
                    instruments.append(EventInstrument.sample(index=ins.index, sample_index=sidx))
            else:
                # Use initial_sample (0-based) if available
                if ins.initial_sample is not None and int(ins.initial_sample) >= 0:
                    sidx1 = int(ins.initial_sample)
                else:
                    sidx1 = 0
                instruments.append(EventInstrument.sample(index=ins.index, sample_index=sidx1))
            # Track used samples and populate instrument entries
            for inst in instruments:
                self.used_samples.add(inst.sample_index)
                self.ins_list.append((ins.index, inst))

        # start at 1, 0 will be reserved for stop remote commands
        command_num = 1
        def add_remote_command(inst: EventInstrument, command: EventRemoteCommand) -> None:
            inst.remote_commands.append(command)
            nonlocal command_num
            command_num += 1

        # gather remote commands and associate with an instrument
        for ins in self.module.Instruments:
            gmacro = ins.snes_macro_data.gain_values
            if gmacro and len(gmacro) > 1:
                # just support one gain change for now.
                # I think amk would allow no more than 2 remote commands at once anyways
                # find this instrument in ins_list
                for (ins_index, inst) in self.ins_list:
                    if ins_index == ins.index:
                        add_remote_command(inst, EventRemoteCommand(command_num, 
                                                                    RemoteCommandTiming.AFTER_START, 
                                                                    RemoteCommandTypes.GAIN, 
                                                                    f"={ins.snes_macro_data.gain_speed}",
                                                                    [gmacro[1]]))
        # need to indicate where to pick up with labels
        # loop labels and remote command labels can't overlap
        self.label_start = command_num
         
        # iterate all rows for command 0Bxx (jump to order)
        # This will be interpreted as the intro marker position
        for c in range(self.module.NumChannels):
            patmap = self.module.PatternsByChannel[c] if c < len(self.module.PatternsByChannel) else {}
            orders = self.module.OrdersPerChannel[c] if c < len(self.module.OrdersPerChannel) else []
            for pat_idx in orders:
                rows = patmap.get(pat_idx)
                if rows:
                    for row in rows:
                        for effect in (row.Effects or []):
                            if effect[0] == 0x0B:
                                self.intro_order = int(effect[1])


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

class MMLLine:
    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens
        self.label: Optional[int] = None
        self.isRepeat: bool = False
    
    def __str__(self) -> str:
        if self.label is not None:
            if self.isRepeat:
                return f"({self.label})"
            else:
                return f"({self.label})[" + ' '.join(self.tokens) + "]"
        
        return ' '.join(self.tokens)

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
        self.add_volume_tempo_info()
        self.add_echo_info()
        self.add_remote_commands()
        self.convert()

    # Convert -128->127 ranged values to 2's complement hex
    @staticmethod
    def to_hex(val):
        return f"{(val & 0xFF):02X}" if val >= 0 else f"{((val + 256) & 0xFF):02X}"

    # Sections
    def add_amk_header(self) -> None:
        self.txt += '#amk 2\n\n'

    # --- helpers ---
    def _row_kind(self, row: FurnacePatternRow) -> str:
        """Classify a Furnace row for emission.

        Returns: 'note' | 'off' | 'release' | 'empty'.
        OFF = 180, RELEASE = 181, MACRO RELEASE = 182
        """
        n = row.Note
        if n is None:
            return 'empty'
        try:
            v = int(n)
        except Exception:
            return 'empty'
        if v == 180:
            return 'off'
        if v == 181:
            return 'release'
        if v == 182:
            return 'macro_release'
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

    def _run_to_denoms(self, run_rows: int, base_den: int, no_whole_notes: bool = False) -> List[int]:
        """Decompose a run of rows into a list of AMK length denominators to tie.

        Each row is 1/base_den. We choose chunks that are divisors of base_den
        and sum to run_rows. For each chunk, the length number is base_den/chunk.
        Example: base_den=16, run=3 -> chunks [2,1] => denoms [8,16] -> c8^16.
        """
        run = max(1, int(run_rows))
        bd = max(1, int(base_den))
        divs = self._divisors(bd)
        # remove divisor of 16 if no_whole_notes
        if no_whole_notes:
            divs = [d for d in divs if d < 16]
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
        # TODO: change insnum_map structure to account for noise instruments
        # rather than relying on negative sample indices
        # map instrument string
        # Map (instrument_index, sample_index) -> AMK instrument number.
        # For noise instruments, sample_index will be None.
        self.insnum_map: Dict[Tuple[int, Optional[int]], int] = {}
        next_num = 30
        name_col = max(len(name) for name, _ in self.event_table.sample_dict.values())
        # get max sample name length for alignment
        name_field_width = name_col + 2  # account for quotes
        # if using sample maps, each sample for an instrument gets its own AMK instrument
        for ins_idx, event_ins in self.event_table.ins_list:
            if event_ins.is_noise:
                # Noise instrument
                samp_name = f'n{(event_ins.noise_freq):02X}'
                print(f"Info: Emitting noise instrument {samp_name} for instrument {self.to_hex(ins_idx)}.", file=sys.stderr)
            else:
                # Resolve sample filename and tuning
                samp_entry = self.event_table.sample_dict.get(event_ins.sample_index)
                if not samp_entry:
                    # Fallback to first sample
                    samp_entry = next(iter(self.event_table.sample_dict.values()), ("Sample1.brr", "$01 $00"))
                samp_name, samp_tuning = samp_entry
                samp_name = f'"{samp_name}"'
            # ADSR/GAIN
            ins = self.event_table.module.Instruments[ins_idx]
            # Default: no envelope -> $00 $00
            da = 0x00
            sr = 0x00
            # Default to no GAIN
            ga = 0x00
            if ins.sn_envelope_on:
                # ADSR on: build ADSR values
                d = int(ins.sn_decay or 0)
                a = int(ins.sn_attack or 0)
                ssv = int(ins.sn_sustain or 0)
                rv = int(ins.sn_release or 0)
                da = ((d & 0x7) | 0x8) << 4 | (a & 0xF)
                sr = ((ssv & 0x7) << 5) | (rv & 0x1F)
            else:
                if ins.snes_macro_data.gain_values:
                    # set primary GAIN to first gain value, other will be handled by remote commands
                    ga = ins.snes_macro_data.gain_values[0]
                elif ins.sn_gain is not None:
                    ga = ins.sn_gain
                else:
                    print(f"Info: Instrument {ins_idx} uses gain mode but has no SNES gain set; defaulting to 0.", file=sys.stderr)
                    ga = 0x00
            lines.append(f'    {samp_name:<{name_field_width}} ${da:02X} ${sr:02X} ${ga:02X} {samp_tuning} ;@{next_num}')
            self.insnum_map[(ins_idx, event_ins.sample_index)] = next_num
            next_num += 1
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_volume_tempo_info(self) -> None:
        mod = self.event_table.module

        # Global tempo and volume
        base_num = mod.HighlightA
        if (base_num <= 0):
            base_num = 4
        base_den = mod.HighlightB
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

        # global volume is average of left/right furnace volumes
        # volumes also stored inversely for some reason.
        Lvol = 127 - mod.SNESFlags.volScaleL
        Rvol = 127 - mod.SNESFlags.volScaleR
        # map 127 -> w255
        gvol = Lvol + Rvol
        amk_volume = min(int(gvol), 255)

        self.txt += f'w{amk_volume} t{amk_tempo}\n\n'

    def add_echo_info(self) -> None:
        mod = self.event_table.module

        # make echo commands
        sn = mod.SNESFlags
        mask = sn.echoMask
        # furnace volume ranges from -128..127
        # not entirely clear how negative volumes are handled in furnace
        # but AMK treats negative volumes as surround volume
        evoll = sn.echoVolL
        evolr = sn.echoVolR
        # echo delay is already 00->0F
        edl = sn.echoDelay
        # feedback, AKA "reverb". Negative numbers are surround reverb.
        efb = sn.echoFeedback
        echoOn = sn.echo
        fir_idx = 0x01 if echoOn else 0x00

        self.txt += f'$EF ${self.to_hex(mask)} ${self.to_hex(evoll)} ${self.to_hex(evolr)}\n'
        self.txt += f'$F1 ${self.to_hex(edl)} ${self.to_hex(efb)} ${self.to_hex(fir_idx)}\n'

        coeffs_hex = ' '.join(f'${self.to_hex(c)}' for c in sn.echoFilterCoeffs)
        self.txt += f'$F5 {coeffs_hex}\n\n'

    def add_remote_commands(self) -> None:
        def make_remote_command(num, command):
            return f"(!{num})[{command}]"
        # add remote code definitions
        # definition for any gain macros
        for (ins_index, inst) in self.event_table.ins_list:
            # just support one gain change for now.
            # I think amk would allow no more than 2 remote commands at once anyways
            for command in inst.remote_commands:
                if command.amk_command_type == RemoteCommandTypes.GAIN:
                    if len(command.amk_command_args) > 0:
                        amk_command = f"$FA$01${self.to_hex(command.amk_command_args[0])}"
                    else:
                        print(f"No gain value present for remote command. Not creating remote command for instrument {ins_index}")
                        continue
                else:
                    print(f"Unrecognized AMK command type {command.amk_command_type}")
                    continue

                self.txt += make_remote_command(command.command_idx, amk_command) + f" ;for furnace inst {self.to_hex(ins_index)}\n"

        self.txt += "\n\n"

    # wild amk volume mapping function stol from it2amk
    def find_v(self, level):
        if level == 0:
            return 0
    
        mindiff = 256
        minval = -1
        
        for v in range(0, 256):
            vv = (v * 0xFF) >> 8
            vv = (vv * vv) >> 8
            vv = (vv * 0x51) >> 8
            vv = (vv * 0xFC) >> 8
            l = vv * 0xFF / 0x4D
            
            if abs(l - level) <= mindiff:
                mindiff = abs(l - level)
                minval = v

        return minval
    
    def channel_has_remote_commands(self, channel: int) -> bool:
        mod = self.event_table.module
        orders = mod.OrdersPerChannel[channel] if channel < len(mod.OrdersPerChannel) else []
        patmap = mod.PatternsByChannel[channel] if channel < len(mod.PatternsByChannel) else {}
        for pat in orders:
            rows = patmap.get(pat)
            if rows:
                for row in rows:
                    kind = self._row_kind(row)
                    if kind == "note":
                        event_insts = self.event_table.ins_list
                        for (ins_index, event_inst) in event_insts:
                            if ins_index == row.Ins:
                                if event_inst.remote_commands:
                                    return True
        return False

    def _convert_effects(self, row: FurnacePatternRow, delay: int, note_idx: int) -> List[str]:
        amk_delay = self.to_hex(delay * 8) # $08 = 1 eighth note
        effect_tokens = []
        # for effect in row.Effects:
        
        return effect_tokens
    
    # Pitchbend is handled specially since it is placed after the note
    def _convert_pitchbend(self, row: FurnacePatternRow, delay: int, note_idx: int, current_octave: int) -> str:
        amk_delay = self.to_hex(delay * 8) # $08 = 1 eighth note
        for effect in row.Effects:
            # TODO: support pitch slide down
            if effect[0] == 0xE1:  # pitch slide up
                # speed is first value of nibble, note is second
                # convert max $0F Fruance to quarter note $30 AMK
                # TODO: figure out precise speed scaling, I just earballed it
                speed = int(48 * (effect[1] >> 4) / 15)
                note = note_idx + (effect[1] & 0x0F)
                name, octave = self._note_name_and_octave(note)  # validate
                bend_note = name
                if (octave != current_octave):
                    bend_note = f'o{octave}{bend_note}'
                    self.current_octave = octave
                return f"$DD${amk_delay}${self.to_hex(speed)} {bend_note}"
        
        return ""

    def _optimize_loops(self, channel_lines: Dict[int, MMLLine], label_count: int) -> int:
        # Identify and label loops in the channel lines
        labels_assigned: Dict[int, List[str]] = {}
        unique_lines: Dict[int, List[str]] = {}
        for order_num, line in channel_lines.items():
            # Check for repeated patterns
            if line.tokens not in unique_lines.values():
                unique_lines[order_num] = line.tokens
            elif line.tokens not in labels_assigned.values():
                # Assign a label to this repeated pattern
                labels_assigned[label_count] = line.tokens
                line.label = label_count
                line.isRepeat = True
                # and mark the first occurrence
                for order, tokens in unique_lines.items():
                    if tokens == line.tokens:
                        channel_lines[order].label = label_count
                        break
                label_count += 1
            else:
                # Find the existing label for this pattern
                for lbl, tokens in labels_assigned.items():
                    if tokens == line.tokens:
                        line.label = lbl
                        line.isRepeat = True
                        break
        return label_count

    # Conversion
    def convert(self) -> None:
        # If we have parsed orders/patterns, emit simple note streams with basic durations per channel.
        mod = self.event_table.module
        if getattr(mod, 'OrdersPerChannel', None) and getattr(mod, 'PatternsByChannel', None) and any(mod.OrdersPerChannel):
            # track global loop labels
            label_count = self.event_table.label_start
            for c in range(mod.NumChannels):
                self.txt += f'#%d\n' % c
                current_oct = None
                current_ins = None  # Furnace instrument index
                current_echo = True # start with echo enabled by default
                has_remote_commands = self.channel_has_remote_commands(c)
                current_remote_gain: Optional[int] = None 
                current_amk_ins: Optional[int] = None  # AMK @ number actually in use
                current_vol: Optional[int] = None  # 0..255
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
                cur_order_num = 0
                cur_measure_num = 0
                base_den = mod.HighlightB
                # TODO: use AMK group labels for identical patterns/measures
                #       line length-dependent breaks by measure
                # I don't see a good way to separate tokenization logic from formatting logic
                # once the amk tokens are created there's not a good way to back out what their pattern associations were
                # I think we just have to process by output line here
                # We can also store prior processed lines in a buffer and detect repeats as we go for labelled loops
                #     - need to remove volume/instrument info when detecting dupes
                # Will implement these formatting nuances after rest of effects are done, in case they add complexity.
                # Probably want a new class MMLLine
                # 1st pass is line breaks informed by pattern length, makes MMLLine objects
                # 2nd pass is line deduplication with labels
                # also need to handle inner/outer loops, cause I'll want to take care of stuff like r1^1^1^1^1^1
                # Somewhere in there too-long lines need to be handled, probably before de-dupe step
                introJustEnded = False
                line_tokens: List[str] = []
                # map of pattern num -> line tokens
                channel_lines = dict()
                # logic to ensure each line always sets the octave explicitly before the first note
                line_octave_set = False
                while i < N:
                    orderNum, rem = divmod(i, mod.PatternLength)
                    measureNum = (i // base_den)
                    if cur_order_num != orderNum and orderNum > 0:
                        channel_lines[cur_order_num] = MMLLine(line_tokens)
                        cur_measure_num = measureNum
                        cur_order_num = orderNum
                        line_octave_set = False
                        line_tokens = []
                        if cur_order_num == self.event_table.intro_order:
                            if rem != 0:
                                print(f"Warning: Expected perfect pattern alignment for intro marker at order {orderNum}, channel {c}.", file=sys.stderr)
                            if has_remote_commands:
                                current_remote_gain = None
                    if cur_measure_num != measureNum:
                        cur_measure_num = measureNum
                        # not desirable for shorter measures, comment out for now
                        # line_tokens = []
                        # self.txt += ' '.join(line_tokens) + '\n\n'
                        # self.txt += f'\n'
                    row = flat_rows[i]
                    kind = self._row_kind(row)
                    # Track instrument changes (don’t emit @ yet; defer until note to choose sample variant)
                    if row.Ins is not None and row.Ins != current_ins and row.Ins != 255:
                        current_ins = int(row.Ins)
                        current_amk_ins = None  # force re-select on next note

                    # Determine if this is a note or rest
                    if kind == 'note':
                        note_token_ties = ''
                        note_idx = int(row.Note)  # type: ignore[arg-type]
                        # Ensure we have some instrument context
                        if current_ins is None or current_ins == 255:
                            current_ins = 0
                            current_amk_ins = None
                        # Determine which sample this note should use for this instrument
                        amk_num = self._resolve_amk_instrument_for_note(current_ins, note_idx)

                        # Begin new instrument and run any instrument-specific commands
                        # make sure to reset instrument if intro just ended, so state is correct on loop
                        if amk_num is not None and (amk_num != current_amk_ins or introJustEnded):
                            line_tokens.append(f'@{amk_num}')
                            current_amk_ins = amk_num
                            ins_echo = mod.Instruments[current_ins].snes_macro_data.is_echo
                            # if instrument echo setting differs from previous, toggle echo
                            if ins_echo != current_echo:
                                current_echo = ins_echo
                                line_tokens.append('$F4$03')

                            event_insts = self.event_table.ins_list
                            for (ins_index, event_inst) in event_insts:
                                if ins_index == current_ins:
                                    has_gain = False
                                    for cmd in event_inst.remote_commands:
                                        if cmd.remote_command_arg is not None:
                                            remote_command = f'(!{cmd.command_idx},{int(cmd.event_type.value)},{cmd.remote_command_arg})'
                                        else:
                                            remote_command = f'(!{cmd.command_idx},{int(cmd.event_type.value)})'

                                        if cmd.amk_command_type == RemoteCommandTypes.GAIN:
                                            has_gain = True
                                            if current_remote_gain != remote_command:
                                                current_remote_gain = remote_command
                                                line_tokens.append(remote_command)
                                    
                                    # turn off remote gain if it's on but next instrument has no gain macro
                                    # also proactively turn it off if we're just ending the intro, so state is reset on loop
                                    if not has_gain and current_remote_gain is not None:
                                        current_remote_gain = None
                                        # kill all remote commands on this channel, ending any gain effects
                                        # TODO: restart any other remote commands that were active before?
                                        line_tokens.append('(!99, 0)')
                                            
                        name, octv = self._note_name_and_octave(note_idx)
                        if current_oct != octv or not line_octave_set:
                            line_tokens.append(f'o{octv}')
                            line_octave_set = True
                            current_oct = octv

                        # Apply volume if present on this row and changed
                        if row.Vol is not None:
                            amk_v = self.find_v(min(int(row.Vol), 255))
                            # TODO: push any overflow to volume scale
                            # TODO: incorporate panning in volume calculation?
                            # so basically figure out if we need to steal more things from it2amk
                            v = max(0, min(255, int(amk_v)))
                            if current_vol != v:
                                line_tokens.append(f'v{v}')
                                current_vol = v
                        
                        effect_tokens = self._convert_effects(row or [], 0, note_idx)
                        bend_token = self._convert_pitchbend(row or [], 0, note_idx, current_oct)
                        # Count run length of same note continuing (no new note starts)
                        run = 1
                        j = i + 1
                        while j < N:
                            r2 = flat_rows[j]
                            k2 = self._row_kind(r2)
                            # stop if next row starts a new note, OFF/RELEASE, or instrument change
                            if (k2 in ('note','off','release')) or (r2.Ins is not None and r2.Ins != current_ins and r2.Ins != 255):
                                break
                            effect_tokens.extend(self._convert_effects(r2 or [], run, note_idx))
                            bend = self._convert_pitchbend(r2 or [], run, note_idx, current_oct)
                            if bend:
                                bend_token = bend  # use latest bend token in run
                            # break note if we're at the end of the intro
                            orderNum, rem = divmod(j, mod.PatternLength)
                            if rem == 0 and orderNum == self.event_table.intro_order:
                                print(f'Note tie across intro marker at order {orderNum}, channel {c}. Breaking tie.')
                                break
                            run += 1
                            j += 1
                            
                        line_tokens.extend(effect_tokens)

                        # Emit note with duration expressed as ties if run>1
                        # Always emit explicit duration numbers and numeric ties
                        denoms = self._run_to_denoms(run, base_den, bend_token != '')
                        note_token = f'{name}{denoms[0]}'

                        for d in denoms[1:]:
                            note_token_ties += f'^{d}'

                        if not bend_token:
                            note_token += note_token_ties

                        # First segment includes note name
                        line_tokens.append(note_token)

                        # pitch bend goes after the note
                        # TODO: support pitchbends at arbitrary points in a note, not just beginning
                        if bend_token:
                            bend_token += note_token_ties
                            line_tokens.append(bend_token)
                        i = j
                        continue
                    else:
                        # Rest or OFF/RELEASE run
                        run = 1
                        j = i + 1
                        while j < N:
                            # break rests at pattern boundaries for readability (and potential loop points)
                            if j % mod.PatternLength == 0:
                                break
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
                        line_tokens.append(rest_token)
                        i = j
                        continue
                # get last one we missed when we broke out of the loop
                channel_lines[orderNum] = MMLLine(line_tokens)

                label_count = self._optimize_loops(channel_lines, label_count)

                # emit channel lines
                for order_num in sorted(channel_lines.keys()):
                    if order_num == self.event_table.intro_order:
                        self.txt += '/\n'
                        if has_remote_commands:
                            self.txt += "(!99, 0) ; reset remote state for loop\n"
                    self.txt += f'; order {order_num}\n'
                    line = channel_lines[order_num]
                    self.txt += str(line) + '\n'

                self.txt += '\n\n'
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
        mod = self.event_table.module
        if ins_idx < 0 or ins_idx > len(mod.Instruments):
            return None
        ins = mod.Instruments[ins_idx]

        # Determine sample index to use
        samp_idx: Optional[int] = None
        n = int(note_idx)
        if not ins.snes_macro_data.is_noise:
            if ins.use_sample_map and ins.sample_table:
                # Furnace provides 120 entries; clamp into range
                n120 = n
                if n120 < 0:
                    n120 = 0
                if n120 >= len(ins.sample_table):
                    n120 = n120 % len(ins.sample_table)
                _note_to_play, samp_to_play = ins.sample_table[n120]
                sidx_raw = int(samp_to_play)
                if sidx_raw >= 0:
                    samp_idx = sidx_raw + 1
            if samp_idx is None:
                if ins.initial_sample is not None and int(ins.initial_sample) >= 0:
                    samp_idx = int(ins.initial_sample)
                else:
                    samp_idx = 0
        # Map to AMK instrument number
        if hasattr(self, 'insnum_map') and isinstance(self.insnum_map, dict):
            return self.insnum_map.get((ins_idx, samp_idx))

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
