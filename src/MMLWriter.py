from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field

from AMKData import AMKData, AMKInstrument, AMKRemoteCommand, AMKRemoteCommandType, AMKRemoteCommandTiming
from AMKData import AMKCommand, AMKDuration, MMLDurationType, MMLData, CommandType

from MMLUtil import DurationFormatter, MMLUtil

# --------------------------------------------------------------------------------------
# MML writer

@dataclass
class MMLState:
    octave: Optional[int]       = None
    echo: bool                  = False
    remote_gain: Optional[int]  = None
    vol: Optional[int]          = None

# A note or rest with its commands
@dataclass
class MMLToken:
    duration: AMKDuration
    commands: List[AMKCommand] = field(default_factory=list)

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
        self.durForamtter = DurationFormatter(self.amk_data.mml_data.ticks_per_subdivision, base_den)

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
        for event in self.amk_data.event_table.commands[channel]:
            if event.type == CommandType.INS_CHANGE:
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

    def make_tokens(self, durations: List[AMKDuration], commands: List[AMKCommand]) -> List[MMLToken]:
        tokens: List[MMLToken] = []
        # sort before iterating
        durations = sorted(durations, key=lambda dur : dur.tick)
        commands = sorted(commands, key=lambda cmd : cmd.tick)
        
        if not durations:
            print(f"Info: Channel has no notes.", file=sys.stderr)
            return []

        cmd_idx = 0
        for dur in durations:
            token = MMLToken(dur)
            while cmd_idx < len(commands):
                cmd_tick = commands[cmd_idx].tick
                if cmd_tick >= dur.tick and cmd_tick < dur.tick + dur.duration:
                    token.commands.append(commands[cmd_idx])
                    cmd_idx += 1
                else:
                    break
            tokens.append(token)
        return tokens

    def convert_command(self, command: AMKCommand, mml_state: MMLState) -> str:
        command_txt = ''
        if command.type == CommandType.INS_CHANGE:
            # Instrument change - emit immediately, no duration
            ins_idx = command.value
            command_txt = f'@{ins_idx + 30} '
            
        elif command.type == CommandType.VOLUME:
            # Volume change - emit immediately, no duration
            vol = command.value
            mml_state.vol = vol
            vol_mml = MMLUtil.find_v(vol)
            command_txt = f'v{vol_mml} '
            
        elif command.type == CommandType.PITCH_BEND:
            speed = command.value
            note = command.value2
            name, octave = MMLUtil.note_name_and_octave(note)
            bend_note = name
            if (octave != mml_state.octave):
                bend_note = f'o{octave}{bend_note}'
                mml_state.octave = octave
            # TODO: handling delay correctly here?
            return f"$DD${MMLUtil.to_hex(0)}${MMLUtil.to_hex(speed)} {bend_note}"

        return command_txt

    def handle_token(self, token: MMLToken, mml_state: MMLState) -> str:
        token_txt = ''
        command_idx = 0
        cur_tick = token.duration.tick
        
        # process any pre-note commands (at the same tick as note start)
        while command_idx < len(token.commands) and token.commands[command_idx].tick == token.duration.tick:
            token_txt += self.convert_command(token.commands[command_idx], mml_state) + ' '
            command_idx += 1

        # add note name and octave
        dur = token.duration
        if dur.type == MMLDurationType.NOTE:
            note_name, note_octave = MMLUtil.note_name_and_octave(dur.note)
            
            # Emit octave change if needed
            if mml_state.octave != note_octave:
                token_txt += f'o{note_octave} '
                mml_state.octave = note_octave
            
            token_txt += note_name
        elif dur.type == MMLDurationType.REST:
            token_txt += 'r'

        # Add initial duration (before any remaining commands)
        cont = False
        if command_idx < len(token.commands):
            first_cmd_tick = token.commands[command_idx].tick
            if first_cmd_tick > cur_tick:
                token_txt += self.durForamtter.format(first_cmd_tick - cur_tick, cont)
                cur_tick = first_cmd_tick
                cont = True
        
        # Interleave commands with duration
        while command_idx < len(token.commands):
            command = token.commands[command_idx]
            cmd_tick = command.tick
            token_txt += self.convert_command(command, mml_state)
            command_idx += 1
            
            # Update cur_tick to this command's tick
            cur_tick = cmd_tick
            
            # Add duration to next command
            if command_idx < len(token.commands):
                next_cmd_tick = token.commands[command_idx].tick
                if next_cmd_tick > cur_tick:
                    token_txt += self.durForamtter.format(next_cmd_tick - cur_tick, cont) + ' '
                    cur_tick = next_cmd_tick
                    cont = True

        # Add remaining duration to end of note
        end_tick = token.duration.tick + token.duration.duration
        if cur_tick < end_tick:
            token_txt += self.durForamtter.format(end_tick - cur_tick, cont) + ' '

        return token_txt

    # Conversion
    def convert(self) -> None:        
        # track global loop labels
        label_count = self.amk_data.label_start
        mml_data = self.amk_data.mml_data
        
        for c in range(mml_data.num_channels):
            token_txt = ''
            mml_state = self.states[c]
            self.txt += f'#%d\n' % c

            tokens = self.make_tokens(mml_data.durations[c], mml_data.commands[c])
            for token in tokens:
                token_txt += self.handle_token(token, mml_state)

            self.txt += token_txt + '\n\n'

        return

    # Output
    def save(self, filename: str) -> None:
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.txt)