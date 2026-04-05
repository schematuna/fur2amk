from ..model.FurnaceData import *
from ..model.FurnaceEffects import *
from ..model.ChiptuneData import *
from ..model.ChiptuneCommands import *

class TickDataConverter:
    def convert(self, furnace_ticks: List[FurnaceTickData]) -> List[ChiptuneTickData]:
        chiptune_ticks: List[ChiptuneTickData] = []
        tuningConverter = TuningConverter()
        for fur_tick in furnace_ticks:
            chip_tick = ChiptuneTickData()
            chip_tick.Note = fur_tick.Note
            chip_tick.Ins = fur_tick.Ins
            chip_tick.Vol = fur_tick.Vol
            for effect in fur_tick.Effects:
                chip_cmd = self.convert_effect(effect)
                if chip_cmd:
                    chip_tick.Commands.append(chip_cmd)

            if tuning_command := tuningConverter.convert_tuning_effects(fur_tick):
                chip_tick.Commands.append(tuning_command)
                
            chiptune_ticks.append(chip_tick)
        
        return chiptune_ticks
    
    def convert_effect(self, effect: FurnaceEffect) -> ChiptuneCommand | None:
        if isinstance(effect, PitchSlideEffect):
            return PitchSlideCommand(effect.change_per_tick)
        elif isinstance(effect, NoteSlideEffect):
            return NoteSlideCommand(effect.speed, effect.semitones)
        elif isinstance(effect, StereoPanEffect):
            return StereoPanCommand(effect.left_volume, effect.right_volume)
        elif isinstance(effect, PanEffect):
            return PanCommand(effect.pan_position)
        elif isinstance(effect, PanSlideEffect):
            return PanSlideCommand(effect.change_per_tick)
        elif isinstance(effect, LegatoEffect):
            return LegatoEnableCommand(effect.legato_on)
        elif isinstance(effect, EchoEffect):
            return EchoEnableCommand(effect.echo_on)
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
    
class TuningConverter:
    def __init__(self):
        self.global_tuning = 0
        self.local_tuning = 0

    def convert_tuning_effects(self, fur_tick: FurnaceTickData):
        tuning_command = None
        if set_pitch_effect := fur_tick.get_effect(SetPitchEffect):
            shift = set_pitch_effect.pitch
            # normalize
            if shift < 0:
                shift = shift / 0x80
            else:
                shift = shift / 0x7F
            self.global_tuning = shift
            tuning_command = TuningCommand(self.global_tuning + self.local_tuning)
        
        if fur_tick.kind() == FurnaceTickData.NoteKind.NOTE:
            if self.local_tuning is not 0:
                tuning_command = TuningCommand(self.global_tuning)
            self.local_tuning = 0

        if single_tick_pitch_effect := fur_tick.get_effect(SingleTickPitchEffect):
            self.local_tuning += single_tick_pitch_effect.pitch_change / 0x20
            tuning_command = TuningCommand(self.global_tuning + self.local_tuning)

        return tuning_command

