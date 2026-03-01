import logging

from ..model.FurnaceData import *
from ..model.ChiptuneData import *
from .MacroConverter import *
from ..util import *

import copy

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

        # Calculate loop tick based on the target order
        loop_tick = None
        if loop_target_order is not None:
            loop_tick = 0
            for i in range(loop_target_order):
                if i < len(pattern_lengths):
                    loop_tick += pattern_lengths[i] * module.Speed1

        return pattern_lengths, pattern_start_offsets, loop_tick
    
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

    def get_ticks(self, flat_rows: List[FurnaceRow], instruments: List[FurnaceInstrument], ticks_per_row: int):
        # first, do basic expansion from rows to ticks
        # TODO: support grooves here
        furnace_ticks: List[TickData] = []
        num_ticks_per_row = ticks_per_row
        for row in flat_rows:
            # copy row info into first tick of row
            first_tick = TickData()
            first_tick.Note = row.Note
            first_tick.Ins = row.Ins
            first_tick.Vol = row.Vol
            first_tick.Effects = row.Effects
            furnace_ticks.append(first_tick)

            if set_speed_effect := row.get_effect(SetSpeedEffect):
                val = set_speed_effect.ticks_per_row
                if val > 0:
                    num_ticks_per_row = set_speed_effect.ticks_per_row

            # and create empty ticks for rest of row
            for i in range(num_ticks_per_row - 1):
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

        # get flat rows taking jumps into account
        flat_song_rows: List[List[FurnaceRow]] = []
        for ch in range(module.NumChannels):
            channel_rows: List[FurnaceRow] = []
            patmap = module.PatternsByChannel[ch]
            orders = module.OrdersPerChannel[ch]

            for order_idx, pat in enumerate(orders):
                rows = patmap.get(pat)
                if rows:
                    start_offset = pattern_offsets[order_idx]
                    end_offset = start_offset + pattern_lengths_rows[order_idx]
                    channel_rows.extend(rows[start_offset:end_offset])
                else:
                    self.logger.warning(f"Channel {ch} references missing pattern {pat}. Inserting empty pattern.")
                    channel_rows.extend([FurnaceRow() for _ in range(pattern_lengths_rows[order_idx])])

            flat_song_rows.append(channel_rows)

        # copy global effects to all rows
        for flat_rows in flat_song_rows:
            for i, row in enumerate(flat_rows):
                if set_speed_effect := row.get_effect(SetSpeedEffect):
                    for flat_rows in flat_song_rows:
                        if set_speed_effect not in flat_rows[i].Effects:
                            flat_rows[i].Effects.append(set_speed_effect)

        # decompose all rows into ticks
        for channel_rows in flat_song_rows:
            chiptune_data.tick_data.append(self.get_ticks(channel_rows, module.Instruments, chiptune_data.structure.ticks_per_step))

        return chiptune_data