# Converts a furnace object to an AMK object

from __future__ import annotations

import sys
import logging
from typing import Dict, List, Tuple

from .model.FurnaceData import FurnaceModule, FurnaceRow, SustainMode
from .model.AMKData import *
from .model.MMLCommands import *
from .RowConverter import *
from .util import *

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
    
    def convert_instruments(self, module: FurnaceModule, num_samples: int) -> List[AMKInstrument]:
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
                            sample_index = mapping[1]
                            if sample_index >= num_samples:
                                sample_index = 0
                                self.logger.warning(f"Instrument {ins.index} has sample index {sample_index} which is greater than the number of samples {num_samples}. Using sample index 0 instead. Please check your sample index in Furnace.")
                            amk_ins.sample_index = sample_index
                            amk_instruments.append(amk_ins)
                            # Store the note -> AMK instrument mapping
                            # Convert from 0:C-(-5) for furnace note to 0:C-0 for sample map
                            note = i + 60
                            note_to_play = mapping[0] + 60
                            ins_info.ins_map[note] = MappingInfo(amk_ins_index, note_to_play)
                            amk_ins_index += 1
                else:
                    amk_ins = AMKInstrument()
                    sample_index = int(ins.initial_sample)
                    if sample_index >= num_samples:
                        sample_index = 0
                        self.logger.warning(f"Instrument {ins.index} has initial sample index {sample_index} which is greater than the number of samples {num_samples}. Using sample index 0 instead. Please check your sample index in Furnace.")
                    amk_ins.sample_index = sample_index
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
                    # For sustain modes 1-3, d2 is used as the effective release rate during sustain
                    # For mode 0 (DIRECT), use the standard release value
                    if ins.sustain_mode != SustainMode.DIRECT:
                        env.release = ins.decay2
                    else:
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
                # If we want to support more we;ll have to set gain manually throughout the note.
                comment = "Gain toggle for Furnace instrument " + str(fur_ins.index)+ ": " + fur_ins.name
                command = EnableGainCommand(None, SnesGain.from_byte(gmacro[1]))
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.AFTER_START, fur_ins.snes_macro_data.gain_speed)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[fur_ins.index].remote_commands.append(remote_def)
                command_num += 1

            if fur_ins.sn_envelope_on and (fur_ins.sustain_mode == SustainMode.EFF_LINEAR or fur_ins.sustain_mode == SustainMode.EFF_EXP):
                gain_mode = GainMode.DEC_LINEAR if fur_ins.sustain_mode == SustainMode.EFF_LINEAR else GainMode.DEC_LOG
                gain = SnesGain(gain_mode, fur_ins.sn_release)
                comment = "Remote gain command for Furnace instrument " + str(fur_ins.index)+ ": " + fur_ins.name
                command = EnableGainCommand(None, gain)
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.KEY_OFF)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[fur_ins.index].remote_commands.append(remote_def)
                command_num += 1
                # and create the key on command to undo the changes
                comment = "Restore ADSR for Furnace instrument " + str(fur_ins.index)+ ": " + fur_ins.name
                command = CustomADSR(None, ADSR(fur_ins.sn_attack, fur_ins.sn_decay, fur_ins.sn_sustain, fur_ins.decay2)) # TODO: just enable adsr directly thru register?
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.KEY_ON)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[fur_ins.index].remote_commands.append(remote_def)
                command_num += 1

            # for delayed release, ADSR changes on note off. Have to restore it on key on.
            if fur_ins.sn_envelope_on and fur_ins.sustain_mode == SustainMode.DELAYED:
                comment = "Restore ADSR for Furnace instrument " + str(fur_ins.index)+ ": " + fur_ins.name
                command = CustomADSR(None, ADSR(fur_ins.sn_attack, fur_ins.sn_decay, fur_ins.sn_sustain, fur_ins.decay2)) # TODO: modify release directly thru register?
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.KEY_ON)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[fur_ins.index].remote_commands.append(remote_def)
                command_num += 1

        # need to indicate where to pick up with labels
        # loop labels and remote command labels can't overlap
        return command_num

    def convert_tempo(self, module: FurnaceModule) -> int:
        rows_per_beat = MMLUtil.AMK_TICKS_PER_BEAT / self.row_converter.amk_ticks_per_row
        fur_ticks_per_beat = rows_per_beat * module.Speed1
        beats_per_second = module.TicksPerSecond / fur_ticks_per_beat
        bpm = int(round(60 * beats_per_second))
        amk_tempo = int(round(bpm * 0.4096 - 1))
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
        mml_data.measure_length = module.HighlightB * self.row_converter.amk_ticks_per_row

        # Analyze pattern lengths and detect loop point
        pattern_lengths_rows, pattern_offsets, loop_tick = self.row_converter.analyze_pattern_lengths(module)

        # Convert to ticks and store
        mml_data.section_lengths = [length * self.row_converter.amk_ticks_per_row
                                     for length in pattern_lengths_rows]
        mml_data.song_length = sum(mml_data.section_lengths)
        mml_data.loop_tick = loop_tick

        for ch in range(module.NumChannels):
            flat_rows: List[FurnaceRow] = []
            patmap = module.PatternsByChannel[ch]
            orders = module.OrdersPerChannel[ch]

            for order_idx, pat in enumerate(orders):
                rows = patmap.get(pat)
                if rows:
                    start_offset = pattern_offsets[order_idx]
                    end_offset = start_offset + pattern_lengths_rows[order_idx]
                    flat_rows.extend(rows[start_offset:end_offset])
                else:
                    self.logger.warning(f"Channel {ch} references missing pattern {pat}. Inserting empty pattern.")
                    flat_rows.extend([FurnaceRow() for _ in range(pattern_lengths_rows[order_idx])])

            mml_data.notes[ch], mml_data.commands[ch] = self.row_converter.convert_notes(flat_rows, self.instrument_info, module.Instruments)
            mml_data.commands[ch].extend(self.row_converter.convert_commands(flat_rows, module))

        return mml_data

    def convert(self, module: FurnaceModule) -> AMKData:
        self.row_converter = RowConverter(module.Speed1)

        amk_data = AMKData()
        amk_data.spc_info     = self.convert_spc_info(module)
        amk_data.samples      = self.convert_samples(module)
        amk_data.instruments  = self.convert_instruments(module, len(amk_data.samples))
        amk_data.label_start  = self.convert_remote_commands(module, amk_data)
        amk_data.tempo        = self.convert_tempo(module)
        amk_data.volume       = self.convert_volume(module)
        amk_data.echo_data    = self.convert_echo(module)
        amk_data.mml_data     = self.convert_mml_data(module)

        return amk_data