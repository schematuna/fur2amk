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
        OCTAVE      = auto()
        ECHO        = auto()
        REMOTE_GAIN = auto()
        VOL         = auto()

    def __init__(self) -> None:
        self.state_d: Dict[str, Any] = {
            MMLState.Type.OCTAVE: None,
            MMLState.Type.ECHO: True,
            MMLState.Type.REMOTE_GAIN: None,
            MMLState.Type.VOL: None
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

    def _run_to_denoms(self, num_subdivisions: int, base_den: int, no_whole_notes: bool = False) -> List[int]:
        """Decompose a number of base_den subdivisions into a list of AMK length denominators to tie.

        Each subdivision represents 1/base_den of a whole note. We choose chunks that are divisors of base_den
        and sum to num_subdivisions. For each chunk, the length number is base_den/chunk.
        Example: base_den=16, num_subdivisions=3 -> chunks [2,1] => denoms [8,16] -> c8^16.
        """
        num = max(1, int(num_subdivisions))
        bd = max(1, int(base_den))
        divs = self._divisors(bd)
        # remove divisor of 16 if no_whole_notes
        if no_whole_notes:
            divs = [d for d in divs if d < 16]
        # allowed chunks are divisors of base_den
        chunks = sorted(divs, reverse=True)
        out: List[int] = []
        rem = num
        while rem > 0:
            # pick largest chunk <= rem
            pick = None
            for c in chunks:
                if c <= rem:
                    pick = c
                    break
            if pick is None:
                # fallback to 1-subdivision chunks (shouldn't happen since 1 divides bd)
                pick = 1
            out.append(bd // pick)
            rem -= pick
        return out

    def add_spc_info(self) -> None:
        # Emit AddmusicK readme-style #spc block with #title/#game/#author/#length
        lines = ['#spc', '{']
        info_align_width = 8
        info = self.amk_data.spc_info
        if info.title:
            lines.append(f'    {'#title':<{info_align_width}} "{info.title}"')
        if info.game:
            lines.append(f'    {'#game':<{info_align_width}} "{info.game}"')
        if info.author:
            lines.append(f'    {'#author':<{info_align_width}} "{info.author}"')
        if info.length:
            lines.append(f'    {'#length':<{info_align_width}} "{info.length}"')
        # Optional comment: use first line of Message if present
        msg = (info.comment or '').strip()
        if msg:
            first_line = msg.splitlines()[0]
            lines.append(f'    {'#comment':<{info_align_width}} "{first_line}"')
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_sample_info(self, module_path: str) -> None:
        path_name = os.path.splitext(os.path.basename(module_path.replace('\\', '/')))[0]
        sample_lines = [f'#path "{path_name}"', '', '#samples', '{', '    #optimized']
        for _, (samp_name, _) in self.amk_data.samples.items():
            brr_rel = f'{samp_name}'
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
        for idx, amk_ins in enumerate(self.amk_data.instruments):
            if amk_ins.is_noise:
                # Noise instrument
                samp_name = f'n{(amk_ins.noise_freq):02X}'
                print(f"Info: Emitting noise instrument {samp_name} for instrument {self.to_hex(idx)}.", file=sys.stderr)
            else:
                # Resolve sample filename and tuning
                samp_entry = self.amk_data.samples[amk_ins.sample_index]
                if not samp_entry:
                    # Fallback to first sample
                    samp_entry = next(iter(self.amk_data.samples.values()), ("Sample1.brr", "$01 $00"))
                samp_name, samp_tuning = samp_entry
                samp_name = f'"{samp_name}"'
            # ADSR/GAIN
            # Default: no envelope -> $00 $00
            da = 0x00
            sr = 0x00
            # Default to no GAIN
            ga = 0x00
            if amk_ins.uses_envelope:
                # ADSR on: build ADSR values
                d = int(amk_ins.envelope.decay or 0)
                a = int(amk_ins.envelope.attack or 0)
                ssv = int(amk_ins.envelope.sustain or 0)
                rv = int(amk_ins.envelope.release or 0)
                da = ((d & 0x7) | 0x8) << 4 | (a & 0xF)
                sr = ((ssv & 0x7) << 5) | (rv & 0x1F)
            else:
                if amk_ins.gain_values:
                    # set primary GAIN to first gain value, other will be handled by remote commands
                    ga = amk_ins.gain_values[0]
                elif amk_ins.gain is not None:
                    ga = amk_ins.gain
                else:
                    print(f"Info: Instrument {idx} uses gain mode but has no SNES gain set; defaulting to 0.", file=sys.stderr)
                    ga = 0x00
            lines.append(f'    {samp_name:<{name_field_width}} ${da:02X} ${sr:02X} ${ga:02X} {samp_tuning} ;@{next_num}')
            next_num += 1
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_volume_tempo_info(self) -> None:
        self.txt += f'w{self.amk_data.volume} t{self.amk_data.tempo}\n\n'

    def add_echo_info(self) -> None:
        echo_data = self.amk_data.echo_data
        if echo_data:
            self.txt += f'$EF ${self.to_hex(echo_data.echoMask)} ${self.to_hex(echo_data.echoVolL)} ${self.to_hex(echo_data.echoVolR)}\n'
            self.txt += f'$F1 ${self.to_hex(echo_data.echoDelay)} ${self.to_hex(echo_data.echoFeedback)} ${self.to_hex(echo_data.firIdx)}\n'

        if echo_data.echoFilterCoeffs:
            coeffs_hex = ' '.join(f'${self.to_hex(c)}' for c in echo_data.echoFilterCoeffs)
            self.txt += f'$F5 {coeffs_hex}\n\n'

    def add_remote_commands(self) -> None:
        def make_remote_command(num, command):
            return f"(!{num})[{command}]"

        for command in self.amk_data.remote_commands:
            if command.amk_command_type == AMKRemoteCommandType.GAIN:
                if len(command.amk_command_args) > 0:
                    amk_command = f"$FA$01${self.to_hex(command.amk_command_args[0])}"
                else:
                    print(f"No gain value present for remote command. Not creating remote command for instrument")
                    continue
            else:
                print(f"Unrecognized AMK command type {command.amk_command_type}")
                continue

            self.txt += make_remote_command(command.command_idx, amk_command) + f" ;for furnace inst\n"

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
        for event in self.amk_data.event_table.events[channel]:
            if event.effect == EventType.INS_CHANGE:
                for (ins_index, inst) in self.amk_data.instruments:
                    if ins_index == event.value:
                        if inst.remote_commands:
                            return True
        return False
    
    # Pitchbend is handled specially since it is placed after the note
    # def _convert_pitchbend(self, event: int, note_idx: int, current_octave: int) -> str:
    #     amk_delay = self.to_hex(delay * 8) # $08 = 1 eighth note
    #     speed = event.value
    #     note = note_idx + event.value2  # semitones
    #     name, octave = self._note_name_and_octave(note)  # validate
    #     bend_note = name
    #     if (octave != current_octave):
    #         bend_note = f'o{octave}{bend_note}'
    #         self.current_octave = octave
    #     return f"$DD${amk_delay}${self.to_hex(speed)} {bend_note}"

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

    def _format_duration_token(self, base_token: str, duration_ticks: int, ticks_per_subdivision: int, base_den: int) -> str:
        """Format a note or rest token with duration and ties.
        
        Args:
            base_token: Base token (e.g., 'c', 'r', 'c+')
            duration_ticks: Duration in ticks
            ticks_per_subdivision: Number of ticks per base_den subdivision (Speed1)
            base_den: Base denominator (e.g., 16 for 16th note grid, from measure_length)
        
        Returns:
            Formatted token with duration (e.g., 'c16', 'r8^16', 'c1^2^4')
        """
        if duration_ticks <= 0:
            return base_token
        
        # Convert ticks to number of base_den subdivisions
        # Each subdivision = ticks_per_subdivision ticks
        if ticks_per_subdivision <= 0:
            ticks_per_subdivision = 1  # fallback to avoid division by zero
        num_subdivisions = duration_ticks / ticks_per_subdivision
        
        # Use _run_to_denoms to convert subdivisions to MML duration denominators
        denoms = self._run_to_denoms(int(round(num_subdivisions)), base_den)
        
        if len(denoms) == 0:
            return base_token
        
        # First duration is attached directly to the note/rest
        token = f'{base_token}{denoms[0]}'
        
        # Additional durations use tie syntax
        for d in denoms[1:]:
            token += f'^{d}'
        
        return token

    # Conversion
    def convert(self) -> None:
        # track global loop labels
        label_count = self.amk_data.label_start
        
        # Determine base denominator from measure_length
        # This represents the subdivision grid (e.g., 16 = 16th note grid)
        base_den = self.amk_data.measure_length
        
        # Calculate ticks per base_den subdivision (Speed1) from ticks_per_beat
        # ticks_per_beat = Speed1 * measure_length, so Speed1 = ticks_per_beat / measure_length
        ticks_per_subdivision = self.amk_data.ticks_per_beat // base_den
        
        for c in range(self.amk_data.num_channels):
            mml_state = self.states[c]
            self.txt += f'#%d\n' % c

            if not self.amk_data.event_table.events[c]:
                print(f"Info: Channel {c} has no events.", file=sys.stderr)
                continue

            events = self.amk_data.event_table.events[c]
            current_octave = None
            
            # Find the first note/rest event to check if we need an initial rest
            first_note_event = None
            for event in events:
                if event.type in (EventType.NOTE, EventType.NOTE_OFF):
                    first_note_event = event
                    break
            
            # If channel doesn't start with a note, emit a rest until the first note
            if first_note_event is not None and first_note_event.tick > 0:
                # Calculate rest duration from tick 0 to first note
                rest_duration_ticks = first_note_event.tick
                rest_token = self._format_duration_token('r', rest_duration_ticks, ticks_per_subdivision, base_den)
                self.txt += f'{rest_token} '
            
            for i, event in enumerate(events):
                # Find the next note/rest event for duration calculation
                next_note_event = None
                for j in range(i + 1, len(events)):
                    if events[j].type in (EventType.NOTE, EventType.NOTE_OFF):
                        next_note_event = events[j]
                        break
                
                # Calculate duration to next note/rest event
                if next_note_event is not None:
                    duration_ticks = next_note_event.tick - event.tick
                else:
                    # Last note/rest event - use one subdivision as default duration
                    duration_ticks = ticks_per_subdivision
                
                # Ensure duration is at least 1
                duration_ticks = max(1, duration_ticks)
                
                # print(f"Event: {event.type}, tick={event.tick}, duration={duration_ticks}", file=sys.stderr)
                
                if event.type == EventType.NOTE:
                    note_idx = event.value
                    note_name, note_octave = self.note_name_and_octave(note_idx)
                    
                    # Emit octave change if needed
                    if current_octave is None or current_octave != note_octave:
                        self.txt += f'o{note_octave} '
                        current_octave = note_octave
                        mml_state.state_d[MMLState.Type.OCTAVE] = note_octave
                    
                    # Format note with duration
                    note_token = self._format_duration_token(note_name, duration_ticks, ticks_per_subdivision, base_den)
                    self.txt += f'{note_token} '
                    
                elif event.type == EventType.NOTE_OFF:
                    # Format rest with duration
                    rest_token = self._format_duration_token('r', duration_ticks, ticks_per_subdivision, base_den)
                    self.txt += f'{rest_token} '
                    
                elif event.type == EventType.INS_CHANGE:
                    # Instrument change - emit immediately, no duration
                    ins_idx = event.value
                    self.txt += f'@{ins_idx + 30} '
                    
                elif event.type == EventType.VOLUME:
                    # Volume change - emit immediately, no duration
                    vol = event.value
                    mml_state.state_d[MMLState.Type.VOL] = vol
                    vol_mml = self.find_v(vol)
                    self.txt += f'v{vol_mml} '
                    
                elif event.type == EventType.PITCH_BEND:
                    # Pitch bend is handled specially - it modifies the previous note
                    # For now, we'll skip it as the conversion method is commented out
                    # TODO: Implement pitch bend handling
                    pass
            
            # Add newline at the end of each channel's data
            self.txt += '\n\n'

        return

    def note_name_and_octave(self, i: int) -> Tuple[str, int]:
        # highest allowed AMK pitch is o6 a
        # TODO: use pitch bend or something to fix automatically?
        while i > 141:
            i -= 12
            self.pitch_warn_count += 1
        # Map Furnace note index (0=C-0) to AMK note name and octave using oN
        names = ['c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+', 'a', 'a+', 'b']
        note = i % 12
        octave = i // 12 - 5  # align with fur2tad convention
        return names[note], octave

    # Output
    def save(self, filename: str) -> None:
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.txt)