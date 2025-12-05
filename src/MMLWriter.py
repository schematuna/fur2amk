from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum, auto

from AMKData import AMKData, AMKInstrument, AMKRemoteCommand, AMKRemoteCommandType, AMKRemoteCommandTiming
from AMKData import Event, EventTable, EventType

if TYPE_CHECKING:
    from fur2amk import Config

# --------------------------------------------------------------------------------------
# MML writer

class MMLState:
    class Type(Enum):
        OCTAVE = auto()

    def __init__(self) -> None:
        self.state_d: Dict[str, Any] = {
            MMLState.Type.OCTAVE: None
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
    def __init__(self, amk_data: AMKData, module_path: str) -> None:
        self.txt: str = ''
        self.amk_data = amk_data
        self.states = [MMLState() for _ in range(8)]

        # for aggregated warning
        self.pitch_warn_count: int = 0       # number of notes that exceeded maximum pitch

        self.add_amk_header()
        self.add_spc_info()
        self.add_sample_info(module_path)
        self.add_ins_info()
        self.add_volume_tempo_info()
        self.add_echo_info()
        self.add_remote_commands()
        self.convert()

        # After conversion, emit aggregated pitch warning if needed
        if self.pitch_warn_count > 0:
            print(
                f"Warning: {self.pitch_warn_count} notes exceeded AMK max pitch.",
                file=sys.stderr,
            )

    # Convert -128->127 ranged values to 2's complement hex
    @staticmethod
    def to_hex(val):
        return f"{(val & 0xFF):02X}" if val >= 0 else f"{((val + 256) & 0xFF):02X}"

    # Sections
    def add_amk_header(self) -> None:
        self.txt += f'#amk {self.amk_data.version}\n\n'

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
        info_align_width = 8
        if self.amk_data.title:
            lines.append(f'    {'#title':<{info_align_width}} "{self.amk_data.title}"')
        if self.amk_data.game:
            lines.append(f'    {'#game':<{info_align_width}} "{self.amk_data.game}"')
        if self.amk_data.author:
            lines.append(f'    {'#author':<{info_align_width}} "{self.amk_data.author}"')
        if self.amk_data.length:
            lines.append(f'    {'#length':<{info_align_width}} "{self.amk_data.length}"')
        # Optional comment: use first line of Message if present
        msg = (self.amk_data.comment or '').strip()
        if msg:
            first_line = msg.splitlines()[0]
            lines.append(f'    {'#comment':<{info_align_width}} "{first_line}"')
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_sample_info(self, module_path: str) -> None:
        path_name = os.path.splitext(os.path.basename(module_path.replace('\\', '/')))[0]
        sample_lines = [f'#path "{path_name}"', '', '#samples', '{', '    #optimized']
        for _, (samp_name, _) in self.amk_data.samples.items():
            brr_rel = f'{samp_name}.brr'
            sample_lines.append(f'    "{brr_rel}"')
        sample_lines.append('}')
        self.txt += '\n'.join(sample_lines) + '\n\n'

    def add_ins_info(self) -> None:
        if not self.amk_data.instruments:
            return
        lines = ['#instruments', '{']
        # Assign AMK instrument numbers starting at 30 in the order we emit
        # Map of (instrument_index, sample_index) -> AMK instrument number
        self.insnum_map: Dict[Tuple[int, Optional[int]], int] = {}
        next_num = 30
        name_col = max(len(name) for name, _ in self.amk_data.samples.values())
        # get max sample name length for alignment
        name_field_width = name_col + 2  # account for quotes
        # if using sample maps, each sample for an instrument gets its own AMK instrument
        for ins_idx, amk_ins in self.amk_data.instruments:
            if amk_ins.is_noise:
                # Noise instrument
                samp_name = f'n{(amk_ins.noise_freq):02X}'
                print(f"Info: Emitting noise instrument {samp_name} for instrument {self.to_hex(ins_idx)}.", file=sys.stderr)
            else:
                # Resolve sample filename and tuning
                samp_entry = self.amk_data.samples.get(amk_ins.sample_index)
                if not samp_entry:
                    # Fallback to first sample
                    samp_entry = next(iter(self.amk_data.samples.values()), ("Sample1.brr", "$01 $00"))
                samp_name, samp_tuning = samp_entry
                samp_name = f'"{samp_name}"'
            # ADSR/GAIN
            ins = self.amk_data.instruments[ins_idx]
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
            self.insnum_map[(ins_idx, amk_ins.sample_index)] = next_num
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
                if command.amk_command_type == AMKRemoteCommandType.GAIN:
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
    
    # Pitchbend is handled specially since it is placed after the note
    def _convert_pitchbend(self, row: FurnaceRow, delay: int, note_idx: int, current_octave: int) -> str:
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
    
    def _get_note_delay(self, row: FurnaceRow) -> int:
        for effect in row.Effects:
            if effect[0] == 0xED:  # note delay
                # value is delay in ticks
                delay_ticks = effect[1]

                return delay_ticks
        
        return 0

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
        a = 1

    def _note_name_and_octave(self, i: int) -> Tuple[str, int]:
        a = 1

    def _resolve_amk_instrument_for_note(self, ins_idx: int, note_idx: int) -> Optional[int]:
        a = 1

    # Output
    def save(self, filename: str) -> None:
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.txt)