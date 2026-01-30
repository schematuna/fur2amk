from typing import List, Optional
from dataclasses import dataclass, field
import sys
import logging

from ..model.FurnaceData import *
from ..model.AMKData import *
from ..model.MMLCommands import *
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

    def analyze_pattern_lengths(self, module: FurnaceModule) -> tuple[List[int], List[int], Optional[int]]:
        """
        Analyze patterns to determine effective lengths considering jump commands.
        Also detects the loop point (0B command).

        Returns:
            - pattern_lengths: [order] -> row count for that pattern
            - pattern_start_offsets: [order] -> starting row offset
            - loop_tick: tick position where loop starts (destination of 0B jump, None if no loop)
        """
        num_orders = len(module.OrdersPerChannel[0])
        pattern_lengths = []
        pattern_start_offsets = []
        loop_target_order = None  # Track which order the loop jumps to

        next_start_row = 0
        accumulated_ticks = 0  # Track total ticks for loop calculation
        for order_idx in range(num_orders):
            pattern_start_offsets.append(next_start_row)

            # Default: pattern runs from start_row to end
            effective_length = module.PatternLength - next_start_row
            next_start_row = 0  # Reset for next pattern
            jump_found = False

            # Scan all channels for jump commands at this order
            for ch in range(module.NumChannels):
                orders = module.OrdersPerChannel[ch]

                pat_id = orders[order_idx]
                patmap = module.PatternsByChannel[ch]
                rows = patmap.get(pat_id, [])

                current_start = pattern_start_offsets[-1]

                # Check for jump commands in this pattern
                for row_idx in range(current_start, len(rows)):
                    row = rows[row_idx]

                    # 0D: Jump to next pattern at specific row
                    if effect := row.get_effect(JumpToNextPatternEffect):
                        effective_length = (row_idx - current_start) + 1
                        next_start_row = effect.row_number
                        jump_found = True
                        break

                    # 0B: Jump to order (also ends pattern and marks loop point)
                    if effect := row.get_effect(JumpToOrderEffect):
                        effective_length = (row_idx - current_start) + 1
                        next_start_row = 0
                        jump_found = True
                        # Store the target order for loop tick calculation
                        loop_target_order = effect.order_number
                        break

                # If we found a jump command, stop scanning other channels
                if jump_found:
                    break

            pattern_lengths.append(effective_length)
            # Accumulate ticks for next iteration
            accumulated_ticks += effective_length * self.amk_ticks_per_row

        # Calculate loop tick based on the target order
        loop_tick = None
        if loop_target_order is not None:
            loop_tick = 0
            for i in range(loop_target_order):
                if i < len(pattern_lengths):
                    loop_tick += pattern_lengths[i] * self.amk_ticks_per_row

        return pattern_lengths, pattern_start_offsets, loop_tick

    def convert(self, flat_rows: List[FurnaceRow], module: FurnaceModule, ins_info: Dict[int, InstrumentInfo]) -> Tuple[List[MMLNote], List[MMLCommand]]:
        # process rows into notes and commands
        notes: List[MMLNote] = []
        commands: List[MMLCommand] = []

        tick = 0

        # initialize converters
        note_converter = NoteConverter(self.tick_ratio, self.amk_ticks_per_row)

        # convert notes
        new_notes, new_commands = note_converter.convert(flat_rows, ins_info, module.Instruments)
        commands.extend(new_commands)
        notes.extend(new_notes)

        # convert commands
        legato_converter    = LegatoConverter(self.amk_ticks_per_row)
        commands.extend(legato_converter.convert(flat_rows, notes))
        
        volume_converter    = VolumeConverter(self.tick_ratio, self.amk_ticks_per_row)
        pan_converter       = PanConverter(self.tick_ratio, self.amk_ticks_per_row)
        vibrato_converter   = VibratoConverter(self.tick_ratio)
        state = FurnaceState()
        for row in flat_rows:
            # commands.extend(legato_converter.convert_row(row, tick, state))
            commands.extend(volume_converter.convert_row(row, tick, state))
            commands.extend(pan_converter.convert_row(row, tick, state))
            commands.extend(vibrato_converter.convert_row(row, tick, state))
            tick += self.amk_ticks_per_row

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