import logging

from ..model.FurnaceData import *
from ..model.ChiptuneData import *
from .MacroConverter import *
from ..util import *

# converts a Furnace module to a generic chiptune format
# this class abstracts away lots of Furnace-specific stuff like:
#   - quick legato
#   - delayed commands
#   - macros
#   - sample maps

class FurnaceConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

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
            accumulated_ticks += effective_length * module.Speed1

        # Calculate loop tick based on the target order
        loop_tick = None
        if loop_target_order is not None:
            loop_tick = 0
            for i in range(loop_target_order):
                if i < len(pattern_lengths):
                    loop_tick += pattern_lengths[i] * module.Speed1

        return pattern_lengths, pattern_start_offsets, loop_tick

    def get_ticks(self, flat_rows: List[FurnaceRow], instruments: List[FurnaceInstrument], ticks_per_row: int):
        vol_converter = VolumeMacroConverter()
        # currently active instrument
        active_ins = None 
        active_note = None
        total_ticks = ticks_per_row * len(flat_rows)
        furnace_ticks = [TickData() for _ in range(total_ticks)]
        for i, row in enumerate(flat_rows):
            row_tick = i * ticks_per_row

            note_kind = row.kind()
            is_new_note = False
            if note_kind == FurnaceRow.NoteKind.NOTE:
                is_new_note = True
                # TODO: this should take sample mapping into account
                # also should probably be setting active note to None on note releases
                # ALSO the active note should change with note slide commands
                active_note = row.Note
                new_fur_ins = None
                for ins in instruments:
                    if ins.index == row.Ins:
                        new_fur_ins = ins
                        break

                if new_fur_ins is not None:
                    active_ins = new_fur_ins

                if active_ins is None:
                    self.logger.warning(f"No furnace instrument active in row with Note {row.Note}.")

            row_vol = vol_converter.get_volume_for_row(row.Vol, is_new_note, active_ins)

            tick = TickData()
            tick.Note = row.Note
            tick.Ins = row.Ins
            tick.Vol = row_vol
            tick.Effects = row.Effects

            if row.kind() == FurnaceRow.NoteKind.OFF or row.kind() == FurnaceRow.NoteKind.RELEASE:
                tick.Type = TickData.NoteKind.RELEASE
            elif row.kind() == FurnaceRow.NoteKind.NOTE:
                tick.Type = TickData.NoteKind.NOTE
            else:
                tick.Type = TickData.NoteKind.EMPTY

            furnace_ticks[row_tick] = tick

            # check for quick legato, make a new note if found
            if effect := row.get_effect(QuickLegatoEffect):
                new_note = active_note + effect.semitones
                new_note_tick = row_tick + effect.delay
                if (new_note_tick < total_ticks):
                    furnace_ticks[new_note_tick].Note = new_note
                    furnace_ticks[new_note_tick].Type = TickData.NoteKind.NOTE
                    active_note = new_note

                # and insert legato effects
                furnace_ticks[row_tick].Effects.append(LegatoEffect(1))
                legato_end_tick = new_note_tick + 1
                if (legato_end_tick < total_ticks):
                    furnace_ticks[legato_end_tick].Effects.append(LegatoEffect(0))

        return furnace_ticks
    

    def get_song_info(self, module: FurnaceModule):
        info = ChiptuneSongInfo()

        info.author = module.Author
        info.comment = module.Comment
        info.title = module.SongName

        return info
    
    def get_structure(self, module: FurnaceModule):
        structure = ChiptuneStructure()

        structure.num_channels = module.NumChannels

        # Analyze pattern lengths and detect loop point
        pattern_lengths_rows, pattern_offsets, loop_tick = self.analyze_pattern_lengths(module)

        # for formatting and duration calculations
        # lengths are in ticks
        ticks_per_row = module.Speed1
        structure.ticks_per_step = ticks_per_row
        structure.measure_length = module.HighlightB * ticks_per_row
        structure.section_lengths = [length * ticks_per_row
                                    for length in pattern_lengths_rows]
        structure.song_length = sum(structure.section_lengths)

        # loop point
        structure.loop_tick = loop_tick

        return structure, pattern_offsets, pattern_lengths_rows

    
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
        chiptune_data.structure, pattern_offsets, pattern_lengths_rows = self.get_structure(module)
        chiptune_data.sample_info = self.get_sample_info(module)
        chiptune_data.instruments = module.Instruments
        chiptune_data.tick_rate = module.TicksPerSecond
        chiptune_data.global_volume = self.get_global_volume(module)
        chiptune_data.echo_data = self.get_echo_data(module)

        # process all rows into tick data
        for ch in range(module.NumChannels):
            flat_rows: List[FurnaceRow] = []
            patmap = module.PatternsByChannel[ch]
            orders = module.OrdersPerChannel[ch]

            for order_idx, pat in enumerate(orders):
                rows = patmap.get(pat)
                if rows:
                    start_offset = pattern_offsets[order_idx]
                    end_offset = start_offset + pattern_lengths_rows[order_idx]
                    flat_rows.extend(rows[start_offset:end_offset])
                else:
                    self.logger.warning(f"Channel {ch} references missing pattern {pat}. Inserting empty pattern.")
                    flat_rows.extend([FurnaceRow() for _ in range(pattern_lengths_rows[order_idx])])

            chiptune_data.tick_data.append(self.get_ticks(flat_rows, module.Instruments, chiptune_data.structure.ticks_per_step))

        return chiptune_data