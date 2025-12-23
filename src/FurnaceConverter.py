# Converts a furnace object to an AMK object

from __future__ import annotations

from operator import mod
import sys
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .model.FurnaceData import FurnaceInstrument, FurnaceModule, FurnaceRow
from .model.AMKData import *
from .model.MMLCommands import *

# persistent channel state
@dataclass
class FurnaceState:
    gain_remote: RemoteCommand = None
    fur_ins_idx: int = None
    echo: bool = True

class FurnaceConverter:
    def __init__(self) -> None:
        # default to quarter note but this will be choosen based on the tick rate
        self.amk_ticks_per_row = 12
        # ratio of amk tick length to furnace tick length
        self.tick_ratio = 1

        # keeps track of how an AMK instrument maps to a Furnace instrument
        self.ins_map: Dict[int, int] = {}
        # Keeps track of instruments that have an associated remote command
        # fur_ins_idx -> remote_command_idx
        self.ins_remote_map: Dict[int, int] = {}

    def convert_spc_info(self, module: FurnaceModule) -> SPCInfo:
        info = SPCInfo()
        info.title = module.SongName
        # info.game = module.Game
        info.author = module.Author
        # info.length = module.Length
        info.comment = module.Comment
        return info

    def convert_samples(self, module: FurnaceModule) -> Dict[int, Tuple[str, str]]:
        sample_dict = {}
        for s in module.Samples:
            # Build sample filename and tuning string
            fname = f"{s.index:02d}_" + (s.name or f"Sample{s.index}").replace(' ', '_') + '.brr'
            tuning_word = 0x0100
            if s.c4_rate and s.c4_rate > 0:
                # MAGIC NUMBERS to convert from c4_rate to AMK instrument tuning value
                # stolen from it2amk's SampConv
                val = int(round(float(s.c4_rate) * 768 / 12539))
                tuning_word = max(0, min(0xFFFF, val))
            tune_str = f"${(tuning_word >> 8) & 0xFF:02X} ${(tuning_word & 0xFF):02X}"
            sample_dict[s.index] = (fname, tune_str)

        return sample_dict
    
    def convert_instruments(self, module: FurnaceModule) -> List[AMKInstrument]:
        instruments: List[AMKInstrument] = []

        amk_ins_index = 0
        for ins in module.Instruments:
            amk_ins = AMKInstrument()
            # first, check if this is a noise instrument
            if ins.snes_macro_data.is_noise:
                amk_ins.is_noise = True
                amk_ins.noise_freq = ins.snes_macro_data.noise_freq
                if ins.snes_macro_data.noise_freq is None:
                    ins.snes_macro_data.noise_freq = 29  # default noise freq if unset
                    print(f"Warning: Instrument {ins.index} is a noise instrument but has no noise frequency set; You should set it explicitly in Furnace.", file=sys.stderr)
            else:
                amk_ins.sample_index = int(ins.initial_sample)

            if ins.sn_envelope_on:
                amk_ins.uses_envelope = True
                env = AMKEnvelope()
                env.attack = ins.sn_attack
                env.decay = ins.sn_decay
                env.sustain = ins.sn_sustain
                env.release = ins.sn_release
                amk_ins.envelope = env
            else:
                amk_ins.gain_values = ins.snes_macro_data.gain_values
                amk_ins.gain = ins.sn_gain
                if amk_ins.gain_values is None or amk_ins.gain is None:
                    print(f"Warning: Instrument {ins.index:02X} uses gain mode but does not have gain parameters set.")

            instruments.append(amk_ins)
            # remember how this AMK instrument maps to a Furnace instrument
            # needed for samples maps where a Furnace instrument can map to multiple AMK instruments
            self.ins_map[amk_ins_index] = ins.index
            amk_ins_index += 1


        return instruments
    
    def convert_remote_commands(self, module: FurnaceModule, amk_data: AMKData) -> int:
        # start at 1, 0 will be reserved for stop remote commands
        command_num = 1

        for fur_ins in module.Instruments:
            gmacro = fur_ins.snes_macro_data.gain_values
            if gmacro and len(gmacro) > 1:
                # just support one gain change for now.
                # I think amk would allow no more than 2 remote commands at once anyways
                comment = "Gain toggle for Furnace instrument " + str(fur_ins.index)+ ": " + fur_ins.name
                remote_def = AMKRemoteDef(command_num, EnableGainCommand(None, gmacro[1]), comment)
                amk_data.remote_defs.append(remote_def)
                self.ins_remote_map[fur_ins.index] = command_num
                command_num += 1

        # need to indicate where to pick up with labels
        # loop labels and remote command labels can't overlap
        return command_num

    def convert_loop_marker(self, module: FurnaceModule) -> int:
        # iterate all rows for command 0Bxx (jump to order)
        # This will be interpreted as the intro marker position
        intro_order = None
        for c in range(module.NumChannels):
            patmap = module.PatternsByChannel[c] if c < len(module.PatternsByChannel) else {}
            orders = module.OrdersPerChannel[c] if c < len(module.OrdersPerChannel) else []
            for pat_idx in orders:
                rows = patmap.get(pat_idx)
                if rows:
                    for row in rows:
                        for effect in (row.Effects or []):
                            if effect[0] == 0x0B:
                                intro_order = int(effect[1])

        return intro_order

    def convert_notes(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLNote]:
        notes: List[MMLNote] = []
        tick = 0
        # process notes
        cur_dur: Optional[MMLNote] = None
        for i, row in enumerate(flat_rows):
            note_kind = row.kind()
            if note_kind == FurnaceRow.NoteKind.NOTE:
                if cur_dur is not None:
                    cur_dur.duration = tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = MMLNote(tick=tick, duration=0, note=row.Note)
            elif note_kind == FurnaceRow.NoteKind.OFF or note_kind == FurnaceRow.NoteKind.RELEASE:
                if cur_dur is not None:
                    cur_dur.duration = tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = None
            
            tick += self.amk_ticks_per_row
        
        # Finalize possible final note
        if cur_dur is not None:
            cur_dur.duration = tick - cur_dur.tick
            notes.append(cur_dur)

        return notes

    def get_note_at_row(self, flat_rows: List[FurnaceRow], row_idx: int) -> Optional[int]:
        cur_note = None
        for i, row in enumerate(flat_rows):
            if row.kind() == FurnaceRow.NoteKind.NOTE:
                cur_note = row.Note
            if row_idx == i:
                return cur_note
        return None

    def convert_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        instruments = module.Instruments

        # process rows into commands
        commands: List[MMLCommand] = []
        tick = 0
        disable_commands_label_idx = 99
        state = FurnaceState()
        for i, row in enumerate(flat_rows):
            # Volume
            vol = row.Vol
            if vol is not None:
                commands.append(VolumeChange(tick, vol))
            
            # Instrument
            fur_ins = None
            for ins in instruments:
                if ins.index == row.Ins:
                    fur_ins = ins
                    break
            if fur_ins is not None:
                # instrument echo
                if fur_ins.snes_macro_data.is_echo != state.echo:
                    commands.append(EchoToggle(tick))
                    state.echo = fur_ins.snes_macro_data.is_echo


                # Instrument gain
                if fur_ins.index in self.ins_remote_map:
                    gain_speed = fur_ins.snes_macro_data.gain_speed
                    remote_comand_idx = self.ins_remote_map[fur_ins.index]
                    gain_remote = RemoteCommand(tick, remote_comand_idx, RemoteCommandTiming.AFTER_START, gain_speed)
                    if gain_remote is not state.gain_remote:
                        commands.append(gain_remote)
                        state.gain_remote = gain_remote
                elif state.gain_remote is not None: # turn off remote commands when gain is disabled
                    commands.append(RemoteCommand(tick, disable_commands_label_idx, RemoteCommandTiming.DISABLE))
                    state.gain_remote = None

                # TODO: handle sample maps here, AMK doesn't need to know about that
                amk_ins_idx = fur_ins.index
                commands.append(InstrumentChange(tick, amk_ins_idx))
            
            # Effects
            for effect in (row.Effects or []):
                effect_num = effect[0]
                value = effect[1]
                if effect_num == 0xE1: # Note slide up
                    # speed is first value of nibble, note is second
                    # convert max $0F Furnace to quarter note $30 AMK
                    # TODO: figure out precise speed scaling, I just earballed it
                    speed = int(48 * (value >> 4) / 15)
                    semitones = value & 0x0F
                    note = self.get_note_at_row(flat_rows, i)
                    if note is not None:
                        bent_note = note + semitones
                    else:
                        print(f"Warning: No note found at tick {tick} for note slide up effect {effect}.", file=sys.stderr)
                        continue
                    commands.append(PitchBend(tick, bent_note, speed))
                elif effect_num == 0xED: # note delay
                    delay_ticks = value
                    # TODO: note delay is not an event, it just modifies a note's tick value
                    # commands.append(NoteDelay(tick, delay_ticks))

            tick += self.amk_ticks_per_row

        return commands
    
    def convert_mml_data(self, module: FurnaceModule) -> MMLData:
        mml_data = MMLData()
        mml_data.num_channels = module.NumChannels
        # for formatting and duration calculations
        # lengths are in ticks
        mml_data.measure_length     = module.HighlightB * self.amk_ticks_per_row
        mml_data.section_length     = module.PatternLength * self.amk_ticks_per_row
        mml_data.song_length        = len(module.OrdersPerChannel[0]) * mml_data.section_length


        for ch in range(module.NumChannels):
            flat_rows: List[FurnaceRow] = []
            patmap = module.PatternsByChannel[ch] if ch < len(module.PatternsByChannel) else {}
            orders = module.OrdersPerChannel[ch] if ch < len(module.OrdersPerChannel) else []
            for pat in orders:
                rows = patmap.get(pat)
                if rows:
                    flat_rows.extend(rows)
                else:
                    print(f"Warning: Channel {ch} references missing pattern {pat}. Inserting empty pattern.", file=sys.stderr)
                    flat_rows.extend([FurnaceRow() for _ in range(module.PatternLength)])

            mml_data.notes[ch]      = self.convert_notes(flat_rows, module)
            mml_data.commands[ch]   = self.convert_commands(flat_rows, module)

        return mml_data

    def convert_tempo(self, module: FurnaceModule) -> int:
        rows_per_beat = MMLUtil.AMK_TICKS_PER_BEAT / self.amk_ticks_per_row
        fur_ticks_per_beat = rows_per_beat * module.Speed1
        beats_per_second = module.TicksPerSecond / fur_ticks_per_beat
        bpm = int(round(60 * beats_per_second))
        return bpm * 8192 // 20025

    def convert_volume(self, module: FurnaceModule) -> int:
        # global volume is average of left/right furnace volumes
        # volumes also stored inversely for some reason.
        Lvol = 127 - module.SNESFlags.volScaleL
        Rvol = 127 - module.SNESFlags.volScaleR
        # map 127 -> w255
        gvol = Lvol + Rvol
        return min(int(gvol), 255)
    
    def convert_echo(self, module: FurnaceModule) -> AMKEchoData:
        echo_data = AMKEchoData()
        echoOn = module.SNESFlags.echo
        echo_data.firIdx = 0x01 if echoOn else 0x00
        echo_data.echoDelay = module.SNESFlags.echoDelay
        echo_data.echoFeedback = module.SNESFlags.echoFeedback
        echo_data.echoMask = module.SNESFlags.echoMask
        echo_data.echoVolL = module.SNESFlags.echoVolL
        echo_data.echoVolR = module.SNESFlags.echoVolR
        echo_data.echoFilterCoeffs = module.SNESFlags.echoFilterCoeffs
        return echo_data

    def convert(self, module: FurnaceModule) -> AMKData:
        # determine musical duration to map to a furnace row
        # find first AMK tick value that is greater than or equal to the furnace tick rate
        for tick_value in MMLUtil.TICK_TO_DURATION.keys():
            if tick_value >= module.Speed1:
                self.amk_ticks_per_row = tick_value
                break

        self.tick_ratio = self.amk_ticks_per_row / module.Speed1
        if self.tick_ratio != round(self.tick_ratio):
            # TODO: For these situations just give up and do everything in ticks
            print("Warning: Furnace ticks not cleanly convertible to amk ticks.")
        print(f"One Furnace tick is {self.tick_ratio:.2g} AMK ticks.")

        amk_data = AMKData()

        amk_data.spc_info     = self.convert_spc_info(module)
        amk_data.samples      = self.convert_samples(module)
        amk_data.instruments  = self.convert_instruments(module)
        amk_data.label_start  = self.convert_remote_commands(module, amk_data)
        amk_data.tempo        = self.convert_tempo(module)
        amk_data.volume       = self.convert_volume(module)
        amk_data.echo_data    = self.convert_echo(module)
        amk_data.mml_data     = self.convert_mml_data(module)

        return amk_data