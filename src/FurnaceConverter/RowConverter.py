# class for working with Furnace rows
# and decomposing them into ticks

from ..model.FurnaceData import *

class RowConverter():
    def __init__(self, module: FurnaceModule):
        self.module = module
        self.ticks_per_row: List[int] = None
        self.loop_tick: int = None
        self.pattern_lengths_ticks: List[int] = None
    
    def get_ticks(self) -> List[List[FurnaceTickData]]:
        # first, naively flatten rows
        flat_song_rows = self._flatten_rows()

        # Then, trim rows to account for jump commands, keeping track of the resulting pattern lengths and loop point
        condensed_rows, pattern_lengths_rows, loop_row = self._resolve_jumps(flat_song_rows, self.module.PatternLength)

        # Get the number of ticks per row, which may be variable
        self.ticks_per_row = self._resolve_speeds(condensed_rows, self.module.Speed1)

        # calc loop tick
        if loop_row:
            self.loop_tick = sum(self.ticks_per_row[:loop_row])

        # Convert pattern lengths to ticks
        self.pattern_lengths_ticks = self._get_pattern_lengths_ticks(pattern_lengths_rows, self.ticks_per_row)

        furnace_ticks: List[List[FurnaceTickData]] = []
        for channel_rows in condensed_rows:
            furnace_ticks.append(self._rows_to_ticks(channel_rows, self.ticks_per_row))

        return furnace_ticks
    
    def get_ticks_per_row(self):
        return self.ticks_per_row
    
    def get_loop_tick(self):
        return self.loop_tick
    
    def get_pattern_lengths(self):
        return self.pattern_lengths_ticks

    def _flatten_rows(self):
        flat_song_rows: List[List[FurnaceRow]] = []
        for ch in range(self.module.NumChannels):
            channel_rows: List[FurnaceRow] = []
            patmap = self.module.PatternsByChannel[ch]
            orders = self.module.OrdersPerChannel[ch]

            for pat in orders:
                rows = patmap.get(pat)
                if rows:
                    channel_rows.extend(rows)
                else:
                    self.logger.warning(f"Channel {ch} references missing pattern {pat}. Inserting empty pattern.")
                    channel_rows.extend([FurnaceRow() for _ in range(len(patmap))])

            flat_song_rows.append(channel_rows)

        return flat_song_rows
    
    def _get_pattern_lengths_ticks(self, pattern_lengths_rows: List[int], ticks_per_row: List[int]) -> List[int]:
        cur_row = 0
        pattern_lengths: List[int] = []
        for pattern_length in pattern_lengths_rows:
            pattern_length_ticks = 0
            for _ in range(pattern_length):
                pattern_length_ticks += ticks_per_row[cur_row]
                cur_row += 1

            pattern_lengths.append(pattern_length_ticks)
        
        return pattern_lengths

    def _rows_to_ticks(self, flat_rows: List[FurnaceRow], ticks_per_row: List[int]):
        # first, do basic expansion from rows to ticks
        furnace_ticks: List[FurnaceTickData] = []
        for i, row in enumerate(flat_rows):
            # copy row info into first tick of row
            first_tick = FurnaceTickData()
            first_tick.Note = row.Note
            first_tick.Ins = row.Ins
            first_tick.Vol = row.Vol
            first_tick.Effects = row.Effects
            furnace_ticks.append(first_tick)

            # and create empty ticks for rest of row
            for i in range(ticks_per_row[i] - 1):
                furnace_ticks.append(FurnaceTickData())

        return furnace_ticks


    def _resolve_jumps(self, flat_rows: List[List[FurnaceRow]], pattern_length: int) -> tuple[List[List[FurnaceRow]], List[int], Optional[int]]:
        """
        Analyze patterns to determine effective lengths considering jump commands.
        Also detects the loop point (0B command).
        """
        num_channels = len(flat_rows)
        num_sections = len(flat_rows[0]) // pattern_length
        if num_sections != int(num_sections):
            self.logger.warning("Rows not evenly divisible by section length. Truncating.")
            num_sections = int(num_sections)
        pattern_lengths = []
        pattern_start_offsets = []
        loop_target_section = None  # Track which section the loop jumps to

        # Trawl rows for pattern length and offset metadata
        next_start_row = 0
        for section_idx in range(num_sections):
            pattern_start_offsets.append(next_start_row)

            # Default: pattern runs from start_row to end
            effective_length = pattern_length - next_start_row
            next_start_row = 0  # Reset for next pattern
            jump_found = False

            # Scan all channels for jump commands in current section
            for ch in range(num_channels):
                # grab this section's rows
                rows = flat_rows[ch][section_idx * pattern_length: (section_idx + 1) * pattern_length]

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
                        # Store the target section for loop tick calculation
                        loop_target_section = effect.order_number
                        break

                # If we found a jump command, stop scanning other channels
                if jump_found:
                    break

            pattern_lengths.append(effective_length)

        # get condensed rows taking jumps into account
        condensed_rows: List[List[FurnaceRow]] = []
        for ch in range(num_channels):
            channel_rows: List[FurnaceRow] = []

            for section_idx in range(num_sections):
                rows = flat_rows[ch][section_idx * pattern_length: (section_idx + 1) * pattern_length]
                start_offset = pattern_start_offsets[section_idx]
                end_offset = start_offset + pattern_lengths[section_idx]
                channel_rows.extend(rows[start_offset:end_offset])

            condensed_rows.append(channel_rows)

        # Calculate loop row based on the target order
        loop_row = None
        if loop_target_section is not None:
            loop_row = 0
            for i in range(loop_target_section):
                if i < len(pattern_lengths):
                    loop_row += pattern_lengths[i]

        return condensed_rows, pattern_lengths, loop_row
    
    def _resolve_speeds(self, condensed_rows: List[List[FurnaceRow]], default_speed: int) -> List[int]:
        """ 
        Returns the number of ticks that should be in each row 

        Accounts for command 0F Set Speed
        TODO: account for 09, alternating speeds, and grooves
        """

        num_channels = len(condensed_rows)
        num_rows = len(condensed_rows[0])
        # First, round up all speed changes across all channels
        speed_changes: Dict[int, int] = dict() # dict tracking row num to speed change
        for ch in range(num_channels):
            channels_rows = condensed_rows[ch]
            for i, row in enumerate(channels_rows):
                if speed_effect := row.get_effect(SetSpeedEffect):
                    speed_changes[i] = speed_effect.ticks_per_row

        # Then create a list of speeds for all rows
        row_speeds: List[int] = []
        cur_speed = default_speed
        for i in range(num_rows):
            if i in speed_changes:
                cur_speed = speed_changes[i]

            row_speeds.append(cur_speed)
        
        return row_speeds