import logging

from ..model.FurnaceData import *
from ..model.FurnaceEffects import *
from ..model.ChiptuneData import *
from ..model.ChiptuneCommands import *
from .InstrumentInfo import FurInstrumentInfo
from .MacroConverter import *
from .FurnaceSliders import *
from .FurnaceUtil import FurnaceUtil

class TickDataConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def convert(self, furnace_ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument], ins_info: Dict[int, FurInstrumentInfo]) -> List[ChiptuneTickData]:
        chiptune_ticks: List[ChiptuneTickData] = []
        echo_converter = EchoConverter()
        tuning_converter = TuningConverter()
        vol_converter = VolumeConverter()
        pan_converter = PanConverter()
        pitchbend_converter = PitchBendConverter()
        active_ins: FurnaceInstrument = None
        vol_at_tick: List[float] = []

        for i, fur_tick in enumerate(furnace_ticks):
            for ins in instruments:
                if ins.index == fur_tick.Ins:
                    active_ins = ins
                    break

            chip_tick = ChiptuneTickData()

            # convert echo
            if echo_cmd := echo_converter.process_tick(fur_tick, active_ins):
                chip_tick.Commands.append(echo_cmd)

            # condense volume slides
            for tick_idx, cmd in vol_converter.process_tick(fur_tick):
                chiptune_ticks[tick_idx].Commands.append(cmd)

            # remember volume at every tick to help with volume macro conversion later
            vol_at_tick.append(vol_converter.current_vol)

            # condense pan slides
            for tick_idx, cmd in pan_converter.process_tick(fur_tick, i):
                if tick_idx == i:
                    chip_tick.Commands.append(cmd)
                else:                
                    chiptune_ticks[tick_idx].Commands.append(cmd)

            for effect in fur_tick.Effects:
                chip_cmd = self.convert_effect(effect)
                if chip_cmd:
                    chip_tick.Commands.append(chip_cmd)

            if tuning_command := tuning_converter.convert_tuning_effects(fur_tick):
                chip_tick.Commands.append(tuning_command)

            chip_tick.Note, chip_tick.Ins = self.apply_sample_map(fur_tick, active_ins, ins_info)

            # handle pitch slides after sample mapping so target note is correct
            for tick_idx, cmd in pitchbend_converter.process_tick(fur_tick, chip_tick.Note, i):
                if tick_idx == i:
                    chip_tick.Commands.append(cmd)
                else:
                    chiptune_ticks[tick_idx].Commands.append(cmd)

            chiptune_ticks.append(chip_tick)

        # finish potential final slides
        if completed := vol_converter.slider.end_slide():
            tick_idx, cmd = completed
            chiptune_ticks[tick_idx].Commands.append(cmd)
        if completed := pan_converter.slider.end_slide():
            tick_idx, cmd = completed
            chiptune_ticks[tick_idx].Commands.append(cmd)
        if completed := pitchbend_converter.slider.end_slide():
            tick_idx, cmd = completed
            chiptune_ticks[tick_idx].Commands.append(cmd)

        self.apply_volume_macros(chiptune_ticks, furnace_ticks, instruments, vol_at_tick)
        self.apply_arp_macros(chiptune_ticks, furnace_ticks, instruments)
        return chiptune_ticks

    def apply_volume_macros(self, chiptune_ticks: List[ChiptuneTickData], furnace_ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument], vol_at_tick: List[float]) -> None:
        vol_mac_converter = VolumeMacroConverter()
        active_ins: FurnaceInstrument = None
        macro_mults: List[float] = []

        # First pass: set chip_tick.Vol and record macro_mult at each tick
        for i, (fur_tick, chip_tick) in enumerate(zip(furnace_ticks, chiptune_ticks)):
            for ins in instruments:
                if ins.index == fur_tick.Ins:
                    active_ins = ins
                    break
            chip_tick.Vol = vol_mac_converter.get_volume_for_tick(fur_tick, active_ins, vol_at_tick[i])
            macro_mults.append(vol_mac_converter.macro_mult)

        # Second pass: scale VolumeFadeCommand targets by macro_mult at slide-start tick
        for i, chip_tick in enumerate(chiptune_ticks):
            if cmd := chip_tick.get_command(VolumeFadeCommand):
                macro_mult = macro_mults[i]
                if macro_mult != 1:
                    cmd.target = round(min(max(cmd.target * macro_mult, 0), 0xFE))

    def apply_arp_macros(self, chiptune_ticks: List[ChiptuneTickData], furnace_ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument]) -> None:
        arp_converter = ArpMacroConverter()
        active_ins: FurnaceInstrument = None
        for fur_tick, chip_tick in zip(furnace_ticks, chiptune_ticks):
            for ins in instruments:
                if ins.index == fur_tick.Ins:
                    active_ins = ins
                    break
            if cmd := arp_converter.get_arp_for_tick(fur_tick, active_ins):
                if existing := chip_tick.get_command(TuningCommand):
                    existing.tuning += cmd.tuning
                else:
                    chip_tick.Commands.append(cmd)

    def apply_sample_map(self, fur_tick: FurnaceTickData, active_ins: FurnaceInstrument, ins_info: Dict[int, FurInstrumentInfo]) -> Tuple[any, any]:
        if fur_tick.kind() == FurnaceTickData.NoteKind.NOTE:
            if active_ins is None:
                self.logger.warning(f"No furnace instrument active in row with Note {fur_tick.Note}.")
            elif active_ins.use_sample_map:
                note = fur_tick.Note
                note_map = ins_info[active_ins.index].ins_map
                if note in note_map:
                    return note_map[note].note_to_play, note_map[note].chiptune_ins_idx
                else:
                    self.logger.warning(f"No instrument mapping found for Furnace instrument {active_ins.index}, note {note}.")
                    return fur_tick.Note, 0
            else:
                # still need to update instrument index for non-sample mapped instruments
                # since they may have been bumped up by prior sample-mapped instruments
                return fur_tick.Note, ins_info[active_ins.index].default_ins

        return fur_tick.Note, fur_tick.Ins

    def convert_effect(self, effect: FurnaceEffect) -> ChiptuneCommand | None:
        if isinstance(effect, LegatoEffect):
            return LegatoEnableCommand(effect.legato_on)
        elif isinstance(effect, VibratoEffect):
            return VibratoCommand(effect.speed, effect.depth)
        elif isinstance(effect, SetTickRateEffect):
            return SetTickRateCommand(effect.tick_rate)
        elif isinstance(effect, SendExternalEffect):
            return SendExternalCommand(effect.value)
        return None


class PitchBendConverter:
    def __init__(self):
        self.slider = FurnacePitchSlider()
        self.cur_pitch = 0  # in semitones

    def process_tick(self, fur_tick: FurnaceTickData, new_note: int, tick_num: int) -> List[Tuple[int, ChiptuneCommand]]:
        """Returns a list of (tick_idx, command) pairs."""

        commands: List[Tuple[int, ChiptuneCommand]] = []

        note_slide_command = fur_tick.get_effect(NoteSlideEffect)
        pitch_slide_command = fur_tick.get_effect(PitchSlideEffect)

        if new_note is not None:
            self.cur_pitch = new_note
            if completed := self.slider.end_slide():
                commands.append(completed)
            self.slider.set_target(self.cur_pitch)
            # if this note interrupted a pitch slide, start a new one
            # unless there is a pitch slide command on this tick, in which case we'll let that handle it
            if completed and not note_slide_command and not pitch_slide_command:
                self.slider.start_slide()
        elif fur_tick.kind() == FurnaceTickData.NoteKind.RELEASE:
            self.cur_pitch = None
            if completed := self.slider.end_slide():
                commands.append(completed)

        if note_slide_command:
            target_note = self.cur_pitch + note_slide_command.semitones
            target_note = max(0,min(target_note, MMLUtil.AMK_MAX_PITCH))
            # for note slides, each speed unit is 4 pitch steps per tick
            ticks_to_slide = FurnaceUtil.ticks_from_speed(note_slide_command.speed * 4, abs(note_slide_command.semitones))
            commands.append((tick_num, PitchSlideCommand(ticks_to_slide, target_note)))
            self.cur_pitch = target_note
            self.slider.set_target(self.cur_pitch)

        if pitch_slide_command:
            if completed := self.slider.handle_new_effect(pitch_slide_command):
                commands.append(completed)
                _, cmd = completed
                self.cur_pitch = cmd.target

        if completed := self.slider.tick():
            commands.append(completed)
            _, cmd = completed
            self.cur_pitch = cmd.target

        return commands

class VolumeConverter:
    """Tracks volume slide state, emitting retroactive VolumeFadeCommands."""

    def __init__(self):
        self.slider = FurnaceVolumeSlider()

    @property
    def current_vol(self) -> float:
        return self.slider.target_val

    def process_tick(self, fur_tick: FurnaceTickData) -> List[Tuple[int, ChiptuneCommand]]:
        """Returns a list of (tick_idx, command) pairs."""

        commands: List[Tuple[int, ChiptuneCommand]] = []

        vol_effect = fur_tick.get_effect(VolumeSlideEffect) or fur_tick.get_effect(FineVolumeSlideEffect)

        if fur_tick.Vol is not None:
            if completed := self.slider.end_slide():
                commands.append(completed)
                if vol_effect is None:
                    self.slider.start_slide()
                
            self.slider.set_target(fur_tick.Vol)

        if vol_effect is not None:
            if completed := self.slider.handle_new_effect(vol_effect):
                commands.append(completed)

        if completed := self.slider.tick():
            commands.append(completed)

        return commands
    

class PanConverter:
    """Tracks pan slide state, emitting retroactive pan fade commands."""

    def __init__(self):
        self.slider = FurnacePanSlider()
        self.cur_pan = 0x80  # center pan

    def process_tick(self, fur_tick: FurnaceTickData, tick_num: int) -> List[Tuple[int, ChiptuneCommand]]:
        """Returns a list of (tick_idx, command) pairs."""

        commands: List[Tuple[int, ChiptuneCommand]] = []

        pan_slide_effect = fur_tick.get_effect(PanSlideEffect)

        if pan_effect := fur_tick.get_effect(PanEffect):
            effect_pan = pan_effect.pan_position
            commands.append((tick_num, PanCommand(effect_pan)))
            self.slider.set_target(effect_pan)
            self.cur_pan = effect_pan

        if stereo_pan_effect := fur_tick.get_effect(StereoPanEffect):
            effect_pan = FurnaceUtil.stereo_to_unity_pan(stereo_pan_effect.left_volume, stereo_pan_effect.right_volume)
            commands.append((tick_num, PanCommand(effect_pan)))
            self.slider.set_target(effect_pan)
            self.cur_pan = effect_pan

        if pan_slide_effect := fur_tick.get_effect(PanSlideEffect):
            if completed := self.slider.handle_new_effect(pan_slide_effect):
                commands.append(completed)

        if completed := self.slider.tick():
            commands.append(completed)

        return commands      


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

class EchoConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.echo_macro_converter = EchoMacroConverter()

    def process_tick(self, tick_data: FurnaceTickData, active_ins: FurnaceInstrument) -> ChiptuneCommand | None:
        echo_command = None

        if macro_echo_effect := self.echo_macro_converter.get_echo_for_tick(tick_data, active_ins):
            echo_command = EchoEnableCommand(macro_echo_effect.echo_on)

        if row_echo_effect := tick_data.get_effect(EchoEffect):
            # echo macro takes precedence over row effect
            if macro_echo_effect is None:
                echo_command = EchoEnableCommand(row_echo_effect.echo_on)

        return echo_command