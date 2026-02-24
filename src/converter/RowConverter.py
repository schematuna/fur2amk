from typing import List, Optional
from dataclasses import dataclass, field
import sys
import logging

from ..model.FurnaceData import *
from ..model.AMKData import *
from ..model.MMLCommands import *
from ..model.ChiptuneData import *
from ..model.FurnaceEffects import *
from .CommandConverter import *
from .NoteConverter import *
from ..util import *

class RowConverter:
    def __init__(self, fur_ticks_per_row: int) -> None:
        self.logger = logging.getLogger(__name__)
        # determine musical duration to map to a furnace row
        # find first AMK tick value that is greater than or equal to the furnace tick rate
        self.amk_ticks_per_row = 12
        for tick_value in MMLUtil.TICK_TO_DURATION.keys():
            if tick_value >= fur_ticks_per_row:
                self.amk_ticks_per_row = tick_value
                break

        # ratio of amk ticks to furnace ticks
        self.tick_ratio = self.amk_ticks_per_row / fur_ticks_per_row
        if self.tick_ratio != round(self.tick_ratio):
            self.logger.warning("Furnace ticks not cleanly convertible to amk ticks.")
        self.logger.info(f"One Furnace tick is {self.tick_ratio:.2g} AMK ticks.")

    def to_amk_ticks(self, furnace_ticks: int):
        if furnace_ticks is None:
            return None
        
        amk_ticks = furnace_ticks * self.tick_ratio
        rounded_amk_ticks = round(amk_ticks)
        if (rounded_amk_ticks != amk_ticks):
            self.logger.debug("furnace to amk tick conversion was not clean")

        return rounded_amk_ticks

    # expands a set of Furnace ticks out into AMK ticks
    # there will (or at least should) always be the same or more AMK ticks than Furnace ticks per beat
    # so we iterate over all Furnace ticks and place its notes and commands in the nearest AMK tick
    def expand_ticks(self, ticks: List[TickData]):
        song_length = self.to_amk_ticks(len(ticks))
        amk_ticks = [TickData() for _ in range(song_length)]
        for tick, tick_data in enumerate(ticks):
            amk_tick = self.to_amk_ticks(tick)
            # should never have to condense here - an AMK tick should always correspond to only one furnace tick
            if amk_ticks[amk_tick] != TickData():
                self.logger.warning("Two Furnace ticks round to one AMK tick. This should never happen.")
                                    
            amk_ticks[amk_tick] = tick_data
        
        return amk_ticks

    def convert(self, ticks: List[TickData], chiptune_data: ChiptuneData, ins_info: Dict[int, InstrumentInfo]) -> Tuple[List[MMLNote], List[MMLCommand]]:
        # process rows into notes and commands
        notes: List[MMLNote] = []
        commands: List[MMLCommand] = []

        tick = 0

        # initialize converters
        note_converter = NoteConverter(self.tick_ratio, self.amk_ticks_per_row)

        # convert notes
        new_notes, new_commands = note_converter.convert(ticks, ins_info, chiptune_data.instruments)
        commands.extend(new_commands)
        notes.extend(new_notes)

        # convert commands
        legato_converter    = LegatoConverter(self.amk_ticks_per_row)
        commands.extend(legato_converter.convert(ticks, notes))
        
        volume_converter    = VolumeConverter(self.tick_ratio)
        pan_converter       = PanConverter(self.tick_ratio)
        vibrato_converter   = VibratoConverter(self.tick_ratio)
        state = FurnaceState()
        for tick_data in ticks:
            # commands.extend(legato_converter.convert_row(row, tick, state))
            commands.extend(volume_converter.convert_tick(tick_data, tick, state))
            commands.extend(pan_converter.convert_tick(tick_data, tick, state))
            commands.extend(vibrato_converter.convert_tick(tick_data, tick, state))
            tick += 1

        # if necessary, toggle legato off before looping
        if state.quick_legato:
            last_note = notes[-1]
            # must be in the middle of a note duration to be effective
            length = last_note.duration
            last_tick = last_note.tick + last_note.duration
            legato_tick = last_tick - 1
            # use clean duration if possible
            if length > 12:
                legato_tick = last_tick - 12
            commands.append(LegatoToggle(legato_tick))
            state.quick_legato = False

        # sort notes and commands by tick
        sorted_commands = sorted(commands, key=lambda x: x.tick)
        sorted_notes = sorted(notes, key=lambda x: x.tick)

        return sorted_notes, sorted_commands