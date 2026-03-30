from ..model.FurnaceData import *
from ..model.FurnaceEffects import *
from ..model.ChiptuneData import *
from ..model.ChiptuneCommands import *

class TickDataConverter:
    def convert(self, furnace_ticks: List[FurnaceTickData]) -> List[ChiptuneTickData]:
        chiptune_ticks: List[ChiptuneTickData] = []
        for fur_tick in furnace_ticks:
            chip_tick = ChiptuneTickData()
            chip_tick.Note = fur_tick.Note
            chip_tick.Ins = fur_tick.Ins
            chip_tick.Vol = fur_tick.Vol
            for effect in fur_tick.Effects:
                chip_cmd = self.convert_effect(effect)
                if chip_cmd:
                    chip_tick.Commands.append(chip_cmd)
                
            chiptune_ticks.append(chip_tick)
        
        return chiptune_ticks
    
    def convert_effect(self, effect: FurnaceEffect) -> ChiptuneCommand | None:
        if isinstance(effect, PitchSlideEffect):
            return PitchSlideCommand(effect.change_per_tick)
        elif isinstance(effect, NoteSlideEffect):
            return NoteSlideCommand(effect.speed, effect.semitones)
        elif isinstance(effect, SetPitchEffect):
            return SetPitchCommand(effect.pitch)
        elif isinstance(effect, StereoPanEffect):
            return StereoPanCommand(effect.left_volume, effect.right_volume)
        elif isinstance(effect, PanEffect):
            return PanCommand(effect.pan_position)
        elif isinstance(effect, PanSlideEffect):
            return PanSlideCommand(effect.change_per_tick)
        elif isinstance(effect, LegatoEffect):
            return LegatoCommand(effect.legato_on)
        elif isinstance(effect, VolumeSlideEffect):
            return VolumeSlideCommand(effect.change_per_tick)
        elif isinstance(effect, FineVolumeSlideEffect):
            return FineVolumeSlideCommand(effect.change_per_tick)
        elif isinstance(effect, VibratoEffect):
            return VibratoCommand(effect.speed, effect.depth)
        elif isinstance(effect, SetTickRateEffect):
            return SetTickRateCommand(effect.tick_rate)
        elif isinstance(effect, SendExternalEffect):
            return SendExternalCommand(effect.value)
        return None