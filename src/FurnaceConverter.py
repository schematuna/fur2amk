# Converts a furnace object to an AMK object

from __future__ import annotations

import sys
import logging
from typing import Dict, List, Tuple

from .model.FurnaceData import FurnaceModule, FurnaceRow
from .model.AMKData import *
from .model.MMLCommands import *
from .RowConverter import *

class FurnaceConverter:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.row_converter = None
        # mapping of furnace instrument index to conversion info
        self.instrument_info : Dict[int, InstrumentInfo] = {}

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
            ins_info = InstrumentInfo()
            amk_instruments: List[AMKInstrument] = []
            # first, check if this is a noise instrument
            if ins.snes_macro_data.is_noise:
                amk_ins = AMKInstrument()
                amk_ins.is_noise = True
                amk_ins.noise_freq = ins.snes_macro_data.noise_freq
                if amk_ins.noise_freq is None:
                    amk_ins.noise_freq = 29  # default noise freq if unset
                    self.logger.warning(f"Instrument {ins.index} is a noise instrument but has no noise frequency set; You should set it explicitly in Furnace.")
                amk_instruments.append(amk_ins)
                ins_info.amk_ins = amk_ins_index
                amk_ins_index += 1
            else:
                if ins.use_sample_map:
                    for i, mapping in enumerate(ins.sample_table):
                        idx = mapping[1]
                        if idx != 65535:
                            amk_ins = AMKInstrument()
                            amk_ins.sample_index = mapping[1]
                            amk_instruments.append(amk_ins)
                            # Store the note -> AMK instrument mapping
                            # Convert from 0:C-(-5) for furnace note to 0:C-0 for sample map
                            note = i + 60
                            note_to_play = mapping[0] + 60
                            ins_info.ins_map[note] = MappingInfo(amk_ins_index, note_to_play)
                            amk_ins_index += 1
                else:
                    amk_ins = AMKInstrument()
                    amk_ins.sample_index = int(ins.initial_sample)
                    amk_instruments.append(amk_ins)
                    ins_info.amk_ins = amk_ins_index
                    amk_ins_index += 1
                
            self.instrument_info[ins.index] = ins_info

            # Apply envelope/gain settings to all AMK instruments created from this Furnace instrument
            for amk_ins in amk_instruments:
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
                        self.logger.debug(f"Instrument {ins.index:02X} uses gain mode but does not have gain parameters set.")

                instruments.append(amk_ins)

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
                self.instrument_info[fur_ins.index].remote_commands.append(command_num)
                command_num += 1

        # need to indicate where to pick up with labels
        # loop labels and remote command labels can't overlap
        return command_num

    def convert_tempo(self, module: FurnaceModule) -> int:
        rows_per_beat = MMLUtil.AMK_TICKS_PER_BEAT / self.row_converter.amk_ticks_per_row
        fur_ticks_per_beat = rows_per_beat * module.Speed1
        beats_per_second = module.TicksPerSecond / fur_ticks_per_beat
        bpm = int(round(60 * beats_per_second))
        # Empirically measured linear relationship between BPM and AMK tempo
        amk_tempo = int(round((bpm - 3.7) / 2.175)) 
        # AMK docs say to use bpm * 8192 / 20025, but that's off
        return amk_tempo

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

    def convert_mml_data(self, module: FurnaceModule) -> MMLData:
        mml_data = MMLData()
        mml_data.num_channels = module.NumChannels
        # for formatting and duration calculations
        # lengths are in ticks
        mml_data.measure_length     = module.HighlightB * self.row_converter.amk_ticks_per_row
        mml_data.section_length     = module.PatternLength * self.row_converter.amk_ticks_per_row
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
                    self.logger.warning(f"Channel {ch} references missing pattern {pat}. Inserting empty pattern.")
                    flat_rows.extend([FurnaceRow() for _ in range(module.PatternLength)])

            loop_tick = self.row_converter.convert_loop_marker(flat_rows, module)
            if loop_tick is not None:
                mml_data.loop_tick = loop_tick
            mml_data.notes[ch], pitch_commands = self.row_converter.convert_notes(flat_rows, self.instrument_info, module.Instruments)
            mml_data.commands[ch] = pitch_commands
            mml_data.commands[ch].extend(self.row_converter.convert_commands(flat_rows, module))

        return mml_data

    def convert(self, module: FurnaceModule) -> AMKData:
        self.row_converter = RowConverter(module.Speed1)

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