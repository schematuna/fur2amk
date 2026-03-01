import logging

from ..model.FurnaceData import *
from ..model.ChiptuneData import *
from .MacroConverter import *
from ..util import *

import copy

# converts a Furnace module to a generic chiptune format
# this class abstracts away lots of Furnace-specific stuff like:
#   - quick legato
#   - macros

class FurnaceConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def flatten_rows(self, module: FurnaceModule):
        flat_song_rows: List[List[FurnaceRow]] = []
        for ch in range(module.NumChannels):
            channel_rows: List[FurnaceRow] = []
            patmap = module.PatternsByChannel[ch]
            orders = module.OrdersPerChannel[ch]

            for pat in orders:
                rows = patmap.get(pat)
                if rows:
                    channel_rows.extend(rows)
                else:
                    self.logger.warning(f"Channel {ch} references missing pattern {pat}. Inserting empty pattern.")
                    channel_rows.extend([FurnaceRow() for _ in range(len(patmap))])

            flat_song_rows.append(channel_rows)

        return flat_song_rows

    def resolve_jumps(self, flat_rows: List[List[FurnaceRow]], pattern_length: int) -> tuple[List[List[FurnaceRow]], List[int], Optional[int]]:
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
    
    def resolve_speeds(self, condensed_rows: List[List[FurnaceRow]], default_speed: int) -> List[int]:
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
    
    def get_pattern_lengths_ticks(self, pattern_lengths_rows: List[int], ticks_per_row: List[int]) -> List[int]:
        cur_row = 0
        pattern_lengths: List[int] = []
        for pattern_length in pattern_lengths_rows:
            pattern_length_ticks = 0
            for _ in range(pattern_length):
                pattern_length_ticks += ticks_per_row[cur_row]
                cur_row += 1

            pattern_lengths.append(pattern_length_ticks)
        
        return pattern_lengths

    def resolve_quick_legato(self, furnace_ticks: List[TickData]):
        """
        Resolve Quick Legato commands into legato commands

        Quick legato starts at the effect and ends at the start of the destination note
        (the first note after the quick legato chain that doesn't have a quick legato effect).
        """
        # tracks gloabl legato command EA, separate from quick legato state
        global_legato_enabled: bool = False
        # tracks guick legato chains, which create localized legato regions
        in_ql_chain: bool = False
        num_ticks = len(furnace_ticks)
        # necessary to deep copy because we modify future elements as we iterate
        resolved_furnace_ticks = copy.deepcopy(furnace_ticks)
        for i, tick_data in enumerate(furnace_ticks):
            # track global legato changes
            legato_effect = tick_data.get_effect(LegatoEffect)
            if legato_effect:
                global_legato_enabled = legato_effect.legato_on

            # check for portamento
            portamento_effect = tick_data.get_effect(PortamentoEffect)
            # when portamento happens, there isn't actually a note onset to end the ql chain
            tick_has_note = tick_data.Note is not None and not portamento_effect
            
            # quick legato chains end when a note happens
            if in_ql_chain and tick_has_note:
                in_ql_chain = False
                # turn legato off if we aren't already in a global legato region
                if not global_legato_enabled:
                    if i - 1 > 0:
                        # turn it off on previous tick
                        resolved_furnace_ticks[i-1].Effects.append(LegatoEffect(0))
                    else:
                        self.logger.warning("Can't end legato because tick would be negative")

            quick_legato_effect = tick_data.get_effect(QuickLegatoEffect)
            # start quick legato chain
            if quick_legato_effect and not in_ql_chain:
                in_ql_chain = True
                # turn legato on if we aren't already in a global legato region
                if not global_legato_enabled:
                    # turn on legato on the tick before new note would start
                    # in Furnace, when you enable legato on the same row as a note onset, legato takes effect immediately
                    # the AMK Converter works with this behavior, moving the legato to the beginning of the previous note
                    # when it lies on a note boundary. So push ql-induced legatos later to work around this.
                    # Ideally should standardize how legato works...
                    resolved_furnace_ticks[i + quick_legato_effect.delay - 1].Effects.append(LegatoEffect(1))

        # end open chains at end of song
        if in_ql_chain:
            resolved_furnace_ticks[num_ticks-1].Effects.append(LegatoEffect(0))
            in_ql_chain = False

        return resolved_furnace_ticks

    def get_ticks(self, flat_rows: List[FurnaceRow], instruments: List[FurnaceInstrument], ticks_per_row: List[int]):
        # first, do basic expansion from rows to ticks
        furnace_ticks: List[TickData] = []
        for i, row in enumerate(flat_rows):
            # copy row info into first tick of row
            first_tick = TickData()
            first_tick.Note = row.Note
            first_tick.Ins = row.Ins
            first_tick.Vol = row.Vol
            first_tick.Effects = row.Effects
            furnace_ticks.append(first_tick)

            # and create empty ticks for rest of row
            for i in range(ticks_per_row[i] - 1):
                furnace_ticks.append(TickData())

        # abstract away quick legato
        # this just handles legato commands, corresponding notes are added in loop below
        furnace_ticks = self.resolve_quick_legato(furnace_ticks)

        vol_converter = VolumeMacroConverter()
        # currently active instrument
        active_ins = None 
        active_note = None
        total_ticks = len(furnace_ticks)
        for i, tick_data in enumerate(furnace_ticks):
            note_kind = tick_data.kind()
            is_new_note = False
            if note_kind == TickData.NoteKind.NOTE:
                is_new_note = True
                # TODO: this should take sample mapping into account
                # also should probably be setting active note to None on note releases
                # ALSO the active note should change with note slide commands
                active_note = tick_data.Note
                new_fur_ins = None
                for ins in instruments:
                    if ins.index == tick_data.Ins:
                        new_fur_ins = ins
                        break

                if new_fur_ins is not None:
                    active_ins = new_fur_ins

                if active_ins is None:
                    self.logger.warning(f"No furnace instrument active in row with Note {tick_data.Note}.")

            tick_data.Vol = vol_converter.get_volume_for_tick(tick_data.Vol, is_new_note, active_ins)

            # check for quick legato, make a new note if found
            if effect := tick_data.get_effect(QuickLegatoEffect):
                if active_note is None:
                    self.logger.warning("Quick Legato found but no note active. Ignoring.")
                else:
                    new_note = active_note + effect.semitones
                    new_note_tick = i + effect.delay
                    if (new_note_tick < total_ticks):
                        furnace_ticks[new_note_tick].Note = new_note
                        active_note = new_note

        return furnace_ticks
    
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

        # first, naively flatten rows
        flat_song_rows = self.flatten_rows(module)

        # Then, trim rows to account for jump commands, keeping track of the resulting pattern lengths and loop point
        condensed_rows, pattern_lengths_rows, loop_row = self.resolve_jumps(flat_song_rows, module.PatternLength)

        # Get the number of ticks per row, which may be variable
        ticks_per_row = self.resolve_speeds(condensed_rows, module.Speed1)

        # Convert pattern lengths to ticks
        pattern_lengths_ticks = self.get_pattern_lengths_ticks(pattern_lengths_rows, ticks_per_row)

        structure = ChiptuneStructure()
        structure.num_channels = module.NumChannels
        structure.ticks_per_step = ticks_per_row
        structure.measure_length = module.HighlightB * ticks_per_row[0] # TODO: variable measure lengths
        structure.section_lengths = pattern_lengths_ticks
        structure.song_length = sum(pattern_lengths_ticks)
        if loop_row is not None:
            structure.loop_tick = sum(ticks_per_row[:loop_row])
        chiptune_data.structure = structure

        # decompose all rows into ticks
        for channel_rows in condensed_rows:
            chiptune_data.tick_data.append(self.get_ticks(channel_rows, module.Instruments, ticks_per_row))

        return chiptune_data