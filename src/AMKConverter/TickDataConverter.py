from typing import List
import logging

from ..model.ChiptuneData import *
from .CommandConverter import *
from .NoteConverter import *
from ..util import *

class TickDataConverter:
    def __init__(self, ticks_per_step: int) -> None:
        # calc initial amk ticks per row based on chiptune ticks per step
        self.calc_amk_ticks_per_row(ticks_per_step)

    def calc_amk_ticks_per_row(self, ticks_per_step: int) -> None:
        self.logger = logging.getLogger(__name__)
        # determine musical duration to map to a chiptunedata row
        # find first AMK tick value that is greater than or equal to the chiptune tick rate
        self.amk_ticks_per_row = 12
        for tick_value in MMLUtil.TICK_TO_DURATION.keys():
            if tick_value >= ticks_per_step:
                self.amk_ticks_per_row = tick_value
                break

        # ratio of amk ticks to chiptune ticks
        self.tick_ratio = self.amk_ticks_per_row / ticks_per_step
        if self.tick_ratio != round(self.tick_ratio):
            self.logger.warning("Furnace ticks not cleanly convertible to amk ticks.")
        self.logger.info(f"One Furnace tick is {self.tick_ratio:.2g} AMK ticks.")

    def to_amk_ticks(self, chiptune_ticks: int):
        if chiptune_ticks is None:
            return None
        
        amk_ticks = chiptune_ticks * self.tick_ratio
        rounded_amk_ticks = round(amk_ticks)
        if (rounded_amk_ticks != amk_ticks):
            self.logger.debug("furnace to amk tick conversion was not clean")

        return rounded_amk_ticks

    # expands a set of ChiptuneData ticks out into AMK ticks
    # there will (or at least should) always be the same or more AMK ticks than ChiptuneData ticks per beat
    # so we iterate over all ChiptuneData ticks and place its notes and commands in the nearest AMK tick
    def expand_ticks(self, ticks: List[ChiptuneTickData]):
        song_length = self.to_amk_ticks(len(ticks))
        amk_ticks = [ChiptuneTickData() for _ in range(song_length)]
        for tick, tick_data in enumerate(ticks):
            amk_tick = self.to_amk_ticks(tick)
            # should never have to condense here - an AMK tick should always correspond to only one ChiptuneData tick
            if amk_ticks[amk_tick] != ChiptuneTickData():
                self.logger.warning("Two Furnace ticks round to one AMK tick. This should never happen.")

            amk_tick_data = copy.deepcopy(tick_data)

            # convert slide durations to amk ticks    
            if pitch_slide := amk_tick_data.get_command(PitchSlideCommand):
                pitch_slide.duration = max(1, self.to_amk_ticks(pitch_slide.duration))

            if pan_slide := amk_tick_data.get_command(PanFadeCommand):
                pan_slide.duration = max(1, self.to_amk_ticks(pan_slide.duration))
            
            if volume_slide := amk_tick_data.get_command(VolumeFadeCommand):
                volume_slide.duration = max(1, self.to_amk_ticks(volume_slide.duration))

            amk_ticks[amk_tick] = amk_tick_data
        
        return amk_ticks

    def convert(self, ticks: List[ChiptuneTickData], chiptune_data: ChiptuneData, ins_info: Dict[int, InstrumentInfo], channel: int) -> Tuple[List[MMLNote], List[MMLCommand]]:
        # process rows into notes and commands
        notes: List[MMLNote] = []
        commands: List[MMLCommand] = []
        proc_ticks = ticks
        tick = 0

        # convert notes
        note_converter      = NoteConverter()
        notes, commands = note_converter.convert(proc_ticks, ins_info, chiptune_data.instruments)

        bend_converter      = PitchBendConverter()
        bend_commands, notes, proc_ticks = bend_converter.convert(proc_ticks, notes)
        commands.extend(bend_commands)

        # convert fine tune commands 
        # since AMK can't change tuning mid-note, we also split notes and wrap in legato when fine tune changes mid-note
        # update ticks with legato effects here instead of making legato commands since they'll get properly taken care of by LegatoConverter anyways
        tuning_converter    = TuningConverter()
        tuning_commands, proc_ticks, notes = tuning_converter.convert(proc_ticks, notes)
        commands.extend(tuning_commands)

        legato_converter    = LegatoConverter()
        commands.extend(legato_converter.convert(proc_ticks, notes))    

        echo_converter      = EchoConverter(chiptune_data.echo_data, channel, chiptune_data.structure.loop_tick is not None)
        commands.extend(echo_converter.convert(proc_ticks))    
        
        tempo_converter     = TempoConverter(chiptune_data.structure, self.amk_ticks_per_row)
        volume_converter    = VolumeConverter(chiptune_data.structure.loop_tick)
        pan_converter       = PanConverter(chiptune_data.structure.loop_tick)
        vibrato_converter   = VibratoConverter(self.tick_ratio)
        state = AMKState()
        for tick_data in proc_ticks:
            commands.extend(tempo_converter.convert_tick(tick_data, tick, state))
            commands.extend(volume_converter.convert_tick(tick_data, tick))
            commands.extend(pan_converter.convert_tick(tick_data, tick, state))
            commands.extend(vibrato_converter.convert_tick(tick_data, tick, state))

            tick += 1

        # sort notes and commands by tick
        sorted_commands = sorted(commands, key=lambda x: x.tick)
        sorted_notes = sorted(notes, key=lambda x: x.tick)

        return sorted_notes, sorted_commands