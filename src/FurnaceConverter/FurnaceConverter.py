import logging

from ..model.FurnaceData import *
from ..model.ChiptuneData import *
from .RowConverter import *
from .InstrumentInfo import *
from .TickDataResolver import *
from .TickDataConverter import *
from ..util import *

class FurnaceConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

        self.instrument_info : Dict[int, FurInstrumentInfo] = {}
    
    def get_song_info(self, module: FurnaceModule):
        info = ChiptuneSongInfo()

        info.author = module.Author
        info.comment = module.Comment
        info.title = module.SongName

        return info
    
    def get_sample_info(self, module: FurnaceModule):
        samps: List[ChiptuneSampleInfo] = []
        for s in module.Samples:
            fname = f"{s.index:02d}_" + (s.name or f"Sample{s.index}").replace(' ', '_') + '.brr'
            samps.append(ChiptuneSampleInfo(s.index, fname, s.c4_rate))

        return samps
    
    def get_instruments(self, module: FurnaceModule, num_samples: int) -> List[ChiptuneInstrument]:
        '''Decomposes sample maps to get flat list of all SNES instruments needed for this port
           Also stores sample map info for later use when resolving furnace notes'''
        chip_instruments: List[ChiptuneInstrument] = []
        chip_ins_counter = 0
        for ins in module.Instruments:
            ins_info = FurInstrumentInfo()
            if ins.use_sample_map:
                # sample idx: chip ins idx
                used_samples: Dict[int, int] = {}
                chip_ins_idx: int = 0
                for i, mapping in enumerate(ins.sample_table):
                    idx = mapping[1]
                    if idx != 65535:
                        sample_index = idx
                        if sample_index >= num_samples:
                            sample_index = 0
                            self.logger.warning(f"Instrument {ins.index} has sample index {sample_index} which is greater than the number of samples {num_samples}. Using sample index 0 instead. Please check your sample index in Furnace.")
                        # only make a new chip instrument for unique samples
                        if sample_index not in used_samples:
                            chip_ins = self.fur_ins_to_chip_ins(ins, chip_ins_counter)
                            chip_ins.initial_sample = sample_index
                            chip_instruments.append(chip_ins)

                            used_samples[sample_index] = chip_ins.index
                            chip_ins_idx = chip_ins_counter
                            chip_ins_counter += 1
                        else:
                            chip_ins_idx = used_samples[sample_index]
                        
                        # Store the note -> AMK instrument mapping for later
                        # Convert from 0:C-(-5) for furnace note to 0:C-0 for sample map
                        note = i + 60
                        note_to_play = mapping[0] + 60
                        ins_info.ins_map[note] = MappingInfo(chip_ins_idx, note_to_play)
            else:
                chip_ins = self.fur_ins_to_chip_ins(ins, chip_ins_counter)
                sample_index = ins.initial_sample
                if sample_index >= num_samples:
                    sample_index = 0
                    self.logger.warning(f"Instrument {ins.index} has sample index {sample_index} which is greater than the number of samples {num_samples}. Using sample index 0 instead. Please check your sample index in Furnace.")
                chip_ins.initial_sample = sample_index
                chip_instruments.append(chip_ins)
                ins_info.default_ins = chip_ins_counter
                chip_ins_counter += 1

            self.instrument_info[ins.index] = ins_info

        return chip_instruments
    
    def fur_ins_to_chip_ins(self, fur_ins: FurnaceInstrument, chip_ins_idx: int) -> ChiptuneInstrument:
        chip_ins = ChiptuneInstrument(chip_ins_idx, fur_ins.name)
        chip_ins.sn_envelope_on = fur_ins.sn_envelope_on

        chip_ins.sn_attack = fur_ins.sn_attack
        chip_ins.sn_decay = fur_ins.sn_decay
        chip_ins.sn_sustain = fur_ins.sn_sustain
        chip_ins.sn_release = fur_ins.sn_release
        chip_ins.decay2 = fur_ins.decay2
        chip_ins.sustain_mode = fur_ins.sustain_mode

        chip_ins.gain_mode = fur_ins.gain_mode
        chip_ins.sn_gain = fur_ins.sn_gain

        chip_ins.initial_sample = fur_ins.initial_sample

        chip_macros = ChiptuneMacroData()
        chip_macros.is_noise = fur_ins.get_special_flag(SpecialFlag.Noise)
        if noise_freq_macro := fur_ins.get_macro(SNESMacroCode.NoiseFreq):
            chip_macros.noise_freq = noise_freq_macro.values[0]
        if gain_macro := fur_ins.get_macro(SNESMacroCode.Gain):
            chip_macros.gain_values = gain_macro.values
            chip_macros.gain_speed = gain_macro.speed
        chip_ins.snes_macro_data = chip_macros

        return chip_ins

    def get_global_volume(self, module: FurnaceModule):
        # global volume is average of left/right furnace volumes
        # volumes also stored inversely for some reason.
        Lvol = 127 - module.SNESFlags.volScaleL
        Rvol = 127 - module.SNESFlags.volScaleR
        # map 127 -> w255
        gvol = Lvol + Rvol
        return min(int(gvol), 255)
    
    def get_echo_data(self, module: FurnaceModule) -> SNESEchoData:
        echo_data = SNESEchoData()
        echoOn = module.SNESFlags.echo
        echo_data.firIdx = 0x01 if echoOn else 0x00
        echo_data.echoDelay = module.SNESFlags.echoDelay
        echo_data.echoFeedback = module.SNESFlags.echoFeedback
        echo_data.echoMask = module.SNESFlags.echoMask
        echo_data.echoVolL = module.SNESFlags.echoVolL
        echo_data.echoVolR = module.SNESFlags.echoVolR
        echo_data.echoFilterCoeffs = module.SNESFlags.echoFilterCoeffs
        return echo_data

    def convert(self, module: FurnaceModule):
        chiptune_data = ChiptuneData()

        chiptune_data.song_info = self.get_song_info(module)
        chiptune_data.sample_info = self.get_sample_info(module)
        chiptune_data.instruments = self.get_instruments(module, len(chiptune_data.sample_info))
        chiptune_data.tick_rate = module.TicksPerSecond
        chiptune_data.global_volume = self.get_global_volume(module)
        chiptune_data.echo_data = self.get_echo_data(module)

        # decompose all rows into ticks and structural information
        rowConverter = RowConverter(module)
        furnace_ticks = rowConverter.get_ticks()

        structure = ChiptuneStructure()
        structure.num_channels = module.NumChannels
        structure.ticks_per_step = rowConverter.get_ticks_per_row()
        structure.measure_length = module.HighlightB * structure.ticks_per_step[0] # TODO: variable measure lengths
        structure.section_lengths = rowConverter.get_pattern_lengths()
        structure.song_length = sum(structure.section_lengths)
        structure.loop_tick = rowConverter.get_loop_tick()
        chiptune_data.structure = structure

        # simplify ticks, resolving furnace-specific effects
        resolver = TickDataResolver()
        for ch, channel_ticks in enumerate(furnace_ticks):
            furnace_ticks[ch] = resolver.resolve_ticks(channel_ticks, module.Instruments)

        # convert furnace tickdata to chiptune tickdata
        tickDataConverter = TickDataConverter()
        for channel_ticks in furnace_ticks:
            chiptune_data.tick_data.append(tickDataConverter.convert(channel_ticks, module.Instruments, self.instrument_info))

        return chiptune_data