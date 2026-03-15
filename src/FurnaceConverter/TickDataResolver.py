from ..model.ChiptuneData import *
from .MacroConverter import *

import copy

# this class abstracts away lots of Furnace-specific stuff like:
#   - quick legato
#   - macros

class TickDataResolver():
    def __init__(self):
        pass

    def resolve_ticks(self, tick_data: List[TickData], instruments: List[FurnaceInstrument]) -> List[TickData]:
        furnace_ticks = tick_data
        # Speed changes and jump commands are already handled by the time we get here
        # virtual tempo changes must also be resolved before note delay is handled
        furnace_ticks = self.resolve_note_delay(furnace_ticks)

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

            tick_data.Vol = vol_converter.get_volume_for_tick(tick_data, is_new_note, active_ins)

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
    
    def resolve_note_delay(self, furnace_ticks: List[TickData]) -> List[TickData]:
        out_ticks: List[TickData] = [TickData()] * len(furnace_ticks)
        for i, tick_data in enumerate(furnace_ticks):
            if note_delay_effect := tick_data.get_effect(NoteDelayEffect):
                delay = note_delay_effect.delay_ticks
                out_ticks[i + delay] = tick_data
            else:
                # only populate if this tick isn't already a delayed tick
                if out_ticks[i] == TickData():
                    out_ticks[i] = tick_data

        return out_ticks

    def resolve_quick_legato(self, furnace_ticks: List[TickData]) -> List[TickData]:
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
            if legato_effect := tick_data.get_effect(LegatoEffect):
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