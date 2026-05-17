# Converts a Chiptune object to an AMK object

from __future__ import annotations

import sys
import logging
from typing import Dict, List, Tuple

from ..model.ChiptuneData import *
from ..model.AMKData import *
from ..model.MMLCommands import *
from .TickDataConverter import *
from ..util import *

class AMKConverter:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.tickdata_converter = None
        # mapping of instrument index to conversion info
        self.instrument_info : Dict[int, InstrumentInfo] = {}

    def convert_spc_info(self, chiptune_data: ChiptuneData) -> SPCInfo:
        info = SPCInfo()
        info.title = chiptune_data.song_info.title
        # info.game = module.Game
        info.author = chiptune_data.song_info.author
        # info.length = module.Length
        info.comment = chiptune_data.song_info.comment
        return info

    def convert_samples(self, chiptune_data: ChiptuneData) -> Dict[int, Tuple[str, str]]:
        sample_dict = {}
        for s in chiptune_data.sample_info:
            # Build sample filename and tuning string
            tuning_word = 0x0100
            if s.c4_rate and s.c4_rate > 0:
                # MAGIC NUMBERS to convert from c4_rate to AMK instrument tuning value
                # stolen from it2amk's SampConv
                val = int(round(float(s.c4_rate) * 768 / 12539))
                tuning_word = max(0, min(0xFFFF, val))
            tune_str = f"${(tuning_word >> 8) & 0xFF:02X} ${(tuning_word & 0xFF):02X}"
            sample_dict[s.index] = AMKSample(filename=s.filename, tuning=tune_str)

        return sample_dict
    
    def convert_instruments(self, chiptune_data: ChiptuneData, num_samples: int) -> List[AMKInstrument]:
        instruments: List[AMKInstrument] = []

        for ins in chiptune_data.instruments:
            amk_ins = AMKInstrument()
            # first, check if this is a noise instrument
            if ins.snes_macro_data.is_noise:
                amk_ins.is_noise = True
                amk_ins.noise_freq = ins.snes_macro_data.noise_freq
                if amk_ins.noise_freq is None:
                    amk_ins.noise_freq = 29  # default noise freq if unset
                    self.logger.warning(f"Instrument {ins.index} is a noise instrument but has no noise frequency set; You should set it explicitly in Furnace.")
            else:
                sample_index = int(ins.initial_sample)
                if sample_index >= num_samples:
                    self.logger.warning(f"Instrument {ins.index} has initial sample index {sample_index} which is greater than the number of samples {num_samples}. Using sample index 0 instead. Please check your sample index in Furnace.")
                    sample_index = 0
                amk_ins.sample_index = sample_index
                
            # Apply envelope/gain settings to this instrument
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
    
    def convert_remote_commands(self, chiptune_data: ChiptuneData, amk_data: AMKData) -> int:
        # start at 1, 0 will be reserved for stop remote commands
        command_num = 1

        for chip_ins in chiptune_data.instruments:
            self.instrument_info[chip_ins.index] = InstrumentInfo()
            gmacro = chip_ins.snes_macro_data.gain_values
            if gmacro and len(gmacro) > 1:
                # just support one gain change for now.
                # If we want to support more we'll have to set gain manually throughout the note.
                comment = "Gain toggle for instrument " + str(chip_ins.index)+ ": " + chip_ins.name
                command = EnableGainCommand(None, SnesGain.from_byte(gmacro[1]))
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.AFTER_START, chip_ins.snes_macro_data.gain_speed)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[chip_ins.index].remote_commands.append(remote_def)
                command_num += 1

            if chip_ins.sn_envelope_on and (chip_ins.sustain_mode == SustainMode.EFF_LINEAR or chip_ins.sustain_mode == SustainMode.EFF_EXP):
                gain_mode = GainMode.DEC_LINEAR if chip_ins.sustain_mode == SustainMode.EFF_LINEAR else GainMode.DEC_LOG
                gain = SnesGain(gain_mode, chip_ins.sn_release)
                comment = "Remote gain command for instrument " + str(chip_ins.index)+ ": " + chip_ins.name
                command = EnableGainCommand(None, gain)
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.KEY_OFF)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[chip_ins.index].remote_commands.append(remote_def)
                command_num += 1
                # and create the key on command to undo the changes
                comment = "Restore ADSR for instrument " + str(chip_ins.index)+ ": " + chip_ins.name
                command = CustomADSR(None, ADSR(chip_ins.sn_attack, chip_ins.sn_decay, chip_ins.sn_sustain, chip_ins.decay2)) # TODO: just enable adsr directly thru register?
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.KEY_ON)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[chip_ins.index].remote_commands.append(remote_def)
                command_num += 1

            # for delayed release, ADSR changes on note off. Have to restore it on key on.
            if chip_ins.sn_envelope_on and chip_ins.sustain_mode == SustainMode.DELAYED:
                comment = "Restore ADSR for instrument " + str(chip_ins.index)+ ": " + chip_ins.name
                command = CustomADSR(None, ADSR(chip_ins.sn_attack, chip_ins.sn_decay, chip_ins.sn_sustain, chip_ins.decay2)) # TODO: modify release directly thru register?
                remote_def = AMKRemoteDef(command_num, command, comment, RemoteCommandTiming.KEY_ON)
                amk_data.remote_defs.append(remote_def)
                self.instrument_info[chip_ins.index].remote_commands.append(remote_def)
                command_num += 1

        # need to indicate where to pick up with labels
        # loop labels and remote command labels can't overlap
        return command_num

    def convert_tempo(self, chiptune_data: ChiptuneData) -> int:
        return AMKUtil.tick_rate_to_amk_tempo(chiptune_data.structure, self.tickdata_converter.amk_ticks_per_row, chiptune_data.tick_rate)

    def convert_mml_data(self, chiptune_data: ChiptuneData) -> MMLData:
        mml_data = MMLData()
        mml_data.num_channels = chiptune_data.structure.num_channels
        # for formatting and duration calculations
        # lengths are in ticks
        mml_data.measure_length = self.tickdata_converter.to_amk_ticks(chiptune_data.structure.measure_length)

        # Convert to ticks and store
        mml_data.section_lengths = [self.tickdata_converter.to_amk_ticks(length) for length in chiptune_data.structure.section_lengths]
        # TODO: consider changes in ticks_per_step when calculating song length
        mml_data.song_length = self.tickdata_converter.to_amk_ticks(chiptune_data.structure.song_length)
        mml_data.loop_tick = self.tickdata_converter.to_amk_ticks(chiptune_data.structure.loop_tick)

        for ch, ticks in enumerate(chiptune_data.tick_data):
            amk_ticks = self.tickdata_converter.expand_ticks(ticks)
            mml_data.notes[ch], mml_data.commands[ch] = self.tickdata_converter.convert(amk_ticks, chiptune_data, self.instrument_info, ch)

        return mml_data

    def convert(self, chiptune_data: ChiptuneData) -> AMKData:
        self.tickdata_converter = TickDataConverter(chiptune_data.structure.ticks_per_step[0])

        amk_data = AMKData()
        amk_data.spc_info     = self.convert_spc_info(chiptune_data)
        amk_data.samples      = self.convert_samples(chiptune_data)
        amk_data.instruments  = self.convert_instruments(chiptune_data, len(amk_data.samples))
        amk_data.label_start  = self.convert_remote_commands(chiptune_data, amk_data)
        amk_data.tempo        = self.convert_tempo(chiptune_data)
        amk_data.volume       = chiptune_data.global_volume
        amk_data.echo_data    = chiptune_data.echo_data
        amk_data.mml_data     = self.convert_mml_data(chiptune_data)

        return amk_data