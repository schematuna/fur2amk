from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass

from AMKData import AMKData, AMKInstrument, AMKRemoteCommand, AMKRemoteCommandType, AMKRemoteCommandTiming
from AMKData import Event, EventTable, EventType

from MMLUtil import DurationFormatter, MMLUtil

# --------------------------------------------------------------------------------------
# MML writer

@dataclass
class MMLState:
    octave: Optional[int]       = None
    echo: bool                  = False
    remote_gain: Optional[int]  = None
    vol: Optional[int]          = None

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

        # assume sixteenth notes for now
        base_den = 16
        self.durForamtter = DurationFormatter(self.amk_data.ticks_per_subdivision, base_den)

        self.add_amk_header()
        self.add_spc_info()
        self.add_sample_info(module_path)
        self.add_ins_info()
        self.add_volume_tempo_info()
        self.add_echo_info()
        self.add_remote_commands()
        self.convert()

    # Sections
    def add_amk_header(self) -> None:
        self.txt += f'#amk {self.amk_data.version}\n\n'

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
                print(f"Info: Emitting noise instrument {samp_name} for instrument {MMLUtil.to_hex(idx)}.", file=sys.stderr)
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
            self.txt += f'$EF ${MMLUtil.to_hex(echo_data.echoMask)} ${MMLUtil.to_hex(echo_data.echoVolL)} ${MMLUtil.to_hex(echo_data.echoVolR)}\n'
            self.txt += f'$F1 ${MMLUtil.to_hex(echo_data.echoDelay)} ${MMLUtil.to_hex(echo_data.echoFeedback)} ${MMLUtil.to_hex(echo_data.firIdx)}\n'

        if echo_data.echoFilterCoeffs:
            coeffs_hex = ' '.join(f'${MMLUtil.to_hex(c)}' for c in echo_data.echoFilterCoeffs)
            self.txt += f'$F5 {coeffs_hex}\n\n'

    def add_remote_commands(self) -> None:
        def make_remote_command(num, command):
            return f"(!{num})[{command}]"

        for command in self.amk_data.remote_commands:
            if command.amk_command_type == AMKRemoteCommandType.GAIN:
                if len(command.amk_command_args) > 0:
                    amk_command = f"$FA$01${MMLUtil.to_hex(command.amk_command_args[0])}"
                else:
                    print(f"No gain value present for remote command. Not creating remote command for instrument")
                    continue
            else:
                print(f"Unrecognized AMK command type {command.amk_command_type}")
                continue

            self.txt += make_remote_command(command.command_idx, amk_command) + f" ;for furnace inst\n"

        self.txt += "\n\n"
    
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
    #     amk_delay = MMLUtil.to_hex(delay * 8) # $08 = 1 eighth note
    #     speed = event.value
    #     note = note_idx + event.value2  # semitones
    #     name, octave = self._note_name_and_octave(note)  # validate
    #     bend_note = name
    #     if (octave != current_octave):
    #         bend_note = f'o{octave}{bend_note}'
    #         self.current_octave = octave
    #     return f"$DD${amk_delay}${MMLUtil.to_hex(speed)} {bend_note}"

    def optimize_loops(self, channel_lines: Dict[int, MMLLine], label_count: int) -> int:
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

    def handle_initial_rest(self, events: List[Event]) -> None:
        first_note_event = None
        for event in events:
            if event.type in (EventType.NOTE, EventType.NOTE_OFF):
                first_note_event = event
                break
        
        if first_note_event is not None and first_note_event.tick > 0:
            rest_duration_ticks = first_note_event.tick
            rest_token = self.durForamtter.format('r', rest_duration_ticks)
            self.txt += f'{rest_token} '

    def handle_note_or_rest(self, event: Event, duration_ticks: int, mml_state: MMLState) -> None:
        if event.type == EventType.NOTE:
            note_idx = event.value
            note_name, note_octave = MMLUtil.note_name_and_octave(note_idx)
            
            # Emit octave change if needed
            if mml_state.octave != note_octave:
                self.txt += f'o{note_octave} '
                mml_state.octave = note_octave
            
            # Format note with duration
            note_token = self.durForamtter.format(note_name, duration_ticks)
            self.txt += f'{note_token} '
            
        elif event.type == EventType.NOTE_OFF:
            # Format rest with duration
            rest_token = self.durForamtter.format('r', duration_ticks)
            self.txt += f'{rest_token} '

    # Conversion
    def convert(self) -> None:
        # track global loop labels
        label_count = self.amk_data.label_start
        
        for c in range(self.amk_data.num_channels):
            mml_state = self.states[c]
            self.txt += f'#%d\n' % c

            if not self.amk_data.event_table.events[c]:
                print(f"Info: Channel {c} has no events.", file=sys.stderr)
                continue

            events = self.amk_data.event_table.events[c]
            
            self.handle_initial_rest(events)
            
            # TODO: filter events into two structures: one for notes/rests and one for instrument changes, volume changes, etc.
            # make clear distinciton between rest/note durations and other events. Can this distintion be clearly made?
            for i, event in enumerate(events):
                # print(f"Event: {event.type}, tick={event.tick}", file=sys.stderr)

                if (event.type == EventType.NOTE or event.type == EventType.NOTE_OFF):
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
                        # TODO: this should be to end of song
                        duration_ticks = self.amk_data.ticks_per_subdivision
                    
                    # Ensure duration is at least 1
                    duration_ticks = max(1, duration_ticks)

                    self.handle_note_or_rest(event, duration_ticks, mml_state)

                elif event.type == EventType.INS_CHANGE:
                    # Instrument change - emit immediately, no duration
                    ins_idx = event.value
                    self.txt += f'@{ins_idx + 30} '
                    
                elif event.type == EventType.VOLUME:
                    # Volume change - emit immediately, no duration
                    vol = event.value
                    mml_state.vol = vol
                    vol_mml = MMLUtil.find_v(vol)
                    self.txt += f'v{vol_mml} '
                    
                elif event.type == EventType.PITCH_BEND:
                    # Pitch bend is handled specially - it modifies the previous note
                    # For now, we'll skip it as the conversion method is commented out
                    # TODO: Implement pitch bend handling
                    pass
            
            # Add newline at the end of each channel's data
            self.txt += '\n\n'

        return

    # Output
    def save(self, filename: str) -> None:
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.txt)