# Converts a furnace object to an AMK object

from __future__ import annotations

from operator import mod
import sys
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto

from FurnaceData import FurnaceModule, FurnaceRow
from AMKData import AMKData, SPCInfo, AMKInstrument, AMKEnvelope, AMKRemoteCommand, AMKRemoteCommandType, AMKRemoteCommandTiming, AMKEchoData
from AMKData import Event, EventTable, EventType

# enum for state types
class EventStateType(Enum):
    INST = auto()
    VOLUME = auto()
    PAN = auto()
    EFFECT = auto()
    
# persitent channel state, useful for avoiding repeat emission
class EventState:
    def __init__(self):
        self.state_d = { EventStateType.INST: None,
                         EventStateType.VOLUME: None,
                         EventStateType.PAN: None,
                         EventStateType.EFFECT: {} }


class FurnaceConverter:
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

            instruments.append(amk_ins)

        return instruments
    
    def convert_remote_commands(self, module: FurnaceModule, amk_data: AMKData) -> int:
        # start at 1, 0 will be reserved for stop remote commands
        command_num = 1

        # gather remote commands and associate with an instrument
        # TODO: remote commands should have their own events, not qualities of amk instruments
        # for fur_ins in module.Instruments:
        #     gmacro = fur_ins.snes_macro_data.gain_values
        #     if gmacro and len(gmacro) > 1:
        #         # just support one gain change for now.
        #         # I think amk would allow no more than 2 remote commands at once anyways
        #         for amk_ins in amk_data.instruments:
        #             if ins_index == fur_ins.index:
        #                 cmd = AMKRemoteCommand(command_num, 
        #                                         AMKRemoteCommandTiming.AFTER_START, 
        #                                         AMKRemoteCommandType.GAIN, 
        #                                         f"={fur_ins.snes_macro_data.gain_speed}",
        #                                         [gmacro[1]])
        #                 amk_ins.remote_commands.append(cmd)
        #                 command_num += 1

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
    
    def convert_events(self, module: FurnaceModule) -> EventTable:
        event_table = EventTable()
        self.states: List[EventState] = [EventState() for _ in range(module.NumChannels)]

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

            state = self.states[ch].state_d
            tick = 0
            ticksPerRow = module.Speed1
            for row in flat_rows:
                # process row into events
                # Order: volume, instrument, effects, then note
                # This ensures commands are emitted before the note in MML
                
                # Volume
                vol = row.Vol
                if vol is not None:
                    if vol != state[EventStateType.VOLUME]:
                        state[EventStateType.VOLUME] = vol
                        event_table.events[ch].append(Event(tick, EventType.VOLUME, vol))
                
                # Instrument
                # TODO: handle sample maps here, AMK doesn't need to know about that
                ins = row.Ins
                if ins is not None:
                    if ins != state[EventStateType.INST]:
                        state[EventStateType.INST] = ins
                        event_table.events[ch].append(Event(tick, EventType.INS_CHANGE, ins))
                
                # Effects
                for effect in (row.Effects or []):
                    effect_num = effect[0]
                    value = effect[1]
                    if effect_num == 0xE1: # Note slide up
                        # TODO: figure out precise speed scaling, I just earballed it
                        speed = int(48 * (value >> 4) / 15)
                        semitones = value & 0x0F
                        event_table.events[ch].append(Event(tick, EventType.PITCH_BEND, speed, semitones))
                    elif effect_num == 0xED: # note delay
                        delay_ticks = value
                        # TODO: note delay is not an event, it just modifies a note's tick value
                        # event_table.events[ch].append(Event(tick, EventType.NOTE_DELAY, delay_ticks))
                
                # Note (processed last)
                type = row._kind()
                if type == FurnaceRow.NoteKind.NOTE:
                    event_table.events[ch].append(Event(tick, EventType.NOTE, row.Note))
                elif type == FurnaceRow.NoteKind.OFF:
                    event_table.events[ch].append(Event(tick, EventType.NOTE_OFF, None))

                tick += ticksPerRow

        return event_table

    def convert_tempo(self, module: FurnaceModule) -> int:
        # Global tempo and volume
        base_num = module.HighlightA
        if (base_num <= 0):
            base_num = 4
        base_den = module.HighlightB
        if base_den <= 0:
            base_den = 16
        tps = float(getattr(module, 'TicksPerSecond', 0.0) or 0.0)
        spd = int(getattr(module, 'Speed1', 0) or 0)
        bpm = max(1, int(round(240.0 * tps / (base_den * spd))))

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
        amk_data = AMKData()

        amk_data.spc_info     = self.convert_spc_info(module)
        amk_data.samples      = self.convert_samples(module)
        amk_data.instruments  = self.convert_instruments(module)
        amk_data.label_start  = self.convert_remote_commands(module, amk_data)
        amk_data.intro_order  = self.convert_loop_marker(module)
        amk_data.tempo        = self.convert_tempo(module)
        amk_data.volume       = self.convert_volume(module)
        amk_data.echo_data    = self.convert_echo(module)
        amk_data.event_table  = self.convert_events(module)
        
        amk_data.num_channels = module.NumChannels
        # for formatting and duration calculations
        # lengths are in units of furnace rows
        amk_data.beat_length            = module.HighlightA
        amk_data.measure_length         = module.HighlightB
        amk_data.pattern_length         = module.PatternLength
        amk_data.ticks_per_subdivision  = module.Speed1

        return amk_data