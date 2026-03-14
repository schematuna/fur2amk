import logging

from ..model.FurnaceData import *
from ..model.ChiptuneData import *
from .RowConverter import *
from .TickDataResolver import *
from ..util import *

class FurnaceConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
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
        chiptune_data.instruments = module.Instruments
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
        for channel_ticks in furnace_ticks:
            chiptune_data.tick_data.append(resolver.resolve_ticks(channel_ticks, module.Instruments))

        return chiptune_data