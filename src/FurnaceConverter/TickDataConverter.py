import logging

from ..model.FurnaceData import *
from ..model.FurnaceEffects import *
from ..model.ChiptuneData import *
from ..model.ChiptuneCommands import *
from .InstrumentInfo import FurInstrumentInfo
from .MacroConverter import VolumeMacroConverter
from .FurnaceSliders import FurnaceVolumeSlider

class TickDataConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def convert(self, furnace_ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument], ins_info: Dict[int, FurInstrumentInfo]) -> List[ChiptuneTickData]:
        chiptune_ticks: List[ChiptuneTickData] = []
        tuning_converter = TuningConverter()
        vol_converter = VolumeConverter()
        active_ins: FurnaceInstrument = None

        for fur_tick in furnace_ticks:
            for ins in instruments:
                if ins.index == fur_tick.Ins:
                    active_ins = ins
                    break

            chip_tick = ChiptuneTickData()

            for tick_idx, cmd in vol_converter.process_tick(fur_tick):
                chiptune_ticks[tick_idx].Commands.append(cmd)

            for effect in fur_tick.Effects:
                chip_cmd = self.convert_effect(effect)
                if chip_cmd:
                    chip_tick.Commands.append(chip_cmd)

            if tuning_command := tuning_converter.convert_tuning_effects(fur_tick):
                chip_tick.Commands.append(tuning_command)

            chip_tick.Note, chip_tick.Ins = self.apply_sample_map(fur_tick, active_ins, ins_info)
            chiptune_ticks.append(chip_tick)

        if completed := vol_converter._slider.end_slide():
            tick_idx, cmd = completed
            chiptune_ticks[tick_idx].Commands.append(cmd)

        self.apply_volume_macros(chiptune_ticks, furnace_ticks, instruments)
        return chiptune_ticks

    def apply_volume_macros(self, chiptune_ticks: List[ChiptuneTickData], furnace_ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument]) -> None:
        vol_mac_converter = VolumeMacroConverter()
        active_ins: FurnaceInstrument = None
        for fur_tick, chip_tick in zip(furnace_ticks, chiptune_ticks):
            for ins in instruments:
                if ins.index == fur_tick.Ins:
                    active_ins = ins
                    break
            chip_tick.Vol = vol_mac_converter.get_volume_for_tick(fur_tick, active_ins)

    def apply_sample_map(self, fur_tick: FurnaceTickData, active_ins: FurnaceInstrument, ins_info: Dict[int, FurInstrumentInfo]) -> Tuple[any, any]:
        if fur_tick.kind() == FurnaceTickData.NoteKind.NOTE:
            if active_ins is None:
                self.logger.warning(f"No furnace instrument active in row with Note {fur_tick.Note}.")
            elif active_ins.use_sample_map:
                note = fur_tick.Note
                note_map = ins_info[active_ins.index].ins_map
                if note in note_map:
                    return note_map[note].note_to_play, note_map[note].amk_ins_idx
                else:
                    self.logger.warning(f"No instrument mapping found for Furnace instrument {active_ins.index}, note {note}.")
                    return fur_tick.Note, 0
            else:
                # still need to update instrument index for non-sample mapped instruments
                # since they may have been bumped up by prior sample-mapped instruments
                return fur_tick.Note, ins_info[active_ins.index].default_ins

        return fur_tick.Note, fur_tick.Ins

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
        elif isinstance(effect, VibratoEffect):
            return VibratoCommand(effect.speed, effect.depth)
        elif isinstance(effect, SetTickRateEffect):
            return SetTickRateCommand(effect.tick_rate)
        elif isinstance(effect, SendExternalEffect):
            return SendExternalCommand(effect.value)
        return None


class VolumeConverter:
    """Tracks volume slide state, emitting retroactive VolumeFadeCommands."""

    def __init__(self):
        self._slider = FurnaceVolumeSlider()

    def process_tick(self, fur_tick: FurnaceTickData) -> List[Tuple[int, ChiptuneCommand]]:
        """Returns a list of (tick_idx, command) pairs to be placed on a previously-emitted ChiptuneTickData."""
        retroactive: List[Tuple[int, ChiptuneCommand]] = []

        vol_effect = fur_tick.get_effect(VolumeSlideEffect) or fur_tick.get_effect(FineVolumeSlideEffect)

        if fur_tick.Vol is not None:
            if vol_effect is not None:
                # End the current slide first (captures accumulated target), then update base.
                # handle_new_effect will start the new slide cleanly since is_sliding will be False.
                if completed := self._slider.end_slide():
                    retroactive.append(completed)
                self._slider.set_target(fur_tick.Vol)
            else:
                if completed := self._slider.set_volume(fur_tick.Vol):
                    retroactive.append(completed)

        if vol_effect is not None:
            if completed := self._slider.handle_new_effect(vol_effect):
                retroactive.append(completed)

        if completed := self._slider.tick():
            retroactive.append(completed)
        return retroactive
    

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
            if self.local_tuning != 0:
                tuning_command = TuningCommand(self.global_tuning)
            self.local_tuning = 0

        if single_tick_pitch_effect := fur_tick.get_effect(SingleTickPitchEffect):
            self.local_tuning += single_tick_pitch_effect.pitch_change / 0x20
            tuning_command = TuningCommand(self.global_tuning + self.local_tuning)

        return tuning_command
