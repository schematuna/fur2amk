from .MacroConverter import *
from ..util.MMLUtil import *

import copy
import math

# this class abstracts away lots of Furnace-specific stuff like:
#   - quick legato
#   - macros

@dataclass
class MappingInfo:
    amk_ins_idx: int
    note_to_play: int

# conversion helper with sample mapping info
# TODO: just make a utility function for this that takes a note and a FurnaceInstrument?
@dataclass
class FurInstrumentInfo:
    # for non-sample map instruments, the chiptune instrument index
    default_ins: int = 0

    # sample map data
    # note -> mapping_info
    ins_map: Dict[int, MappingInfo] = field(default_factory=dict)

class TickDataResolver():
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def resolve_ticks(self, ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument], ins_info: Dict[int, FurInstrumentInfo]) -> List[FurnaceTickData]:
        furnace_ticks = ticks
        # Speed changes and jump commands are already handled by the time we get here
        # virtual tempo changes must also be resolved before note delay is handled
        furnace_ticks = self.resolve_note_delay(furnace_ticks)

        # abstract away quick legato
        # this just handles legato commands, corresponding notes are added in loop below
        furnace_ticks = self.resolve_quick_legato(furnace_ticks)

        # convert portamento into note slides so AMK Converter doesn't need to worry about porta logic
        # Note: portamentos are unaffected by smaple mapped note changes. This is how it works in Furnace.
        furnace_ticks = self.resolve_portamento(furnace_ticks)

        # convert macros into plain commands
        furnace_ticks = self.resolve_macros(furnace_ticks, instruments)

        # flatten sample maps into separate instruments
        # Note: after this point the instrument numbers in tick data represent chiptune instrument index, not furnace instrument index
        furnace_ticks = self.resolve_sample_maps(furnace_ticks, instruments, ins_info)

        return furnace_ticks
    
    def resolve_note_delay(self, furnace_ticks: List[FurnaceTickData]) -> List[FurnaceTickData]:
        out_ticks: List[FurnaceTickData] = [FurnaceTickData()] * len(furnace_ticks)
        for i, tick_data in enumerate(furnace_ticks):
            if note_delay_effect := tick_data.get_effect(NoteDelayEffect):
                delay = note_delay_effect.delay_ticks
                out_ticks[i + delay] = tick_data
            else:
                # only populate if this tick isn't already populated by a delayed tick
                if out_ticks[i] == FurnaceTickData():
                    out_ticks[i] = tick_data

        return out_ticks

    def resolve_ql_legato(self, furnace_ticks: List[FurnaceTickData]) -> List[FurnaceTickData]:
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
    
    def resolve_ql_notes(self, furnace_ticks: List[FurnaceTickData]) -> List[FurnaceTickData]:
        # currently active instrument
        active_note = None
        total_ticks = len(furnace_ticks)
        new_ticks = furnace_ticks
        for i, tick_data in enumerate(furnace_ticks):
            note_kind = tick_data.kind()
            if note_kind == FurnaceTickData.NoteKind.NOTE:
                # TODO: this should take sample mapping into account
                # also should probably be setting active note to None on note releases
                # ALSO the active note should change with note slide commands
                active_note = tick_data.Note

            # check for quick legato, make a new note if found
            if effect := tick_data.get_effect(QuickLegatoEffect):
                if active_note is None:
                    self.logger.warning("Quick Legato found but no note active. Ignoring.")
                else:
                    new_note = active_note + effect.semitones
                    new_note_tick = i + effect.delay
                    if (new_note_tick < total_ticks):
                        new_ticks[new_note_tick].Note = new_note
                        active_note = new_note

        return new_ticks

    def resolve_quick_legato(self, furnace_ticks: List[FurnaceTickData]) -> List[FurnaceTickData]:
        resolved_ticks = furnace_ticks
        resolved_ticks = self.resolve_ql_legato(resolved_ticks)
        resolved_ticks = self.resolve_ql_notes(resolved_ticks)
        return resolved_ticks
    
    def resolve_portamento(self, furnace_ticks: List[FurnaceTickData]):
        '''Convert portamentos to note slides for simplicity'''
        # TODO: this conversion loses precision. Convert all pitch slides to some standard ChiptuneData format instead
        out_ticks: List[FurnaceTickData] = []
        # the current active pitch. Determines starting pitch for portamentos
        active_note = None
        for i, tick_data in enumerate(furnace_ticks):
            new_tick = FurnaceTickData()
            if portamento_effect := tick_data.get_effect(PortamentoEffect):
                if tick_data.kind() == FurnaceTickData.NoteKind.NOTE:
                    new_tick = FurnaceTickData()
                    # don't copy note to new tick
                    new_note = tick_data.Note
                    new_tick.Ins = tick_data.Ins
                    new_tick.Vol = tick_data.Vol
                    for effect in tick_data.Effects:
                        if not isinstance(effect, PortamentoEffect):
                            new_tick.Effects.append(effect)
                        else:
                            note_slide = NoteSlideEffect(0, 0)
                            note_slide.speed = math.ceil(portamento_effect.speed / 4)
                            note_slide.semitones = new_note - active_note
                            new_tick.Effects.append(note_slide)

                    active_note = new_note
                else:
                    self.logger.warning("Portamento effect found on non-note row, ignoring.")
                    new_tick = tick_data
            else:
                new_tick = tick_data
                if tick_data.kind() == FurnaceTickData.NoteKind.NOTE:
                    active_note = tick_data.Note

                # note slides affect portamento starting pitch 
                if noteslide_effect := tick_data.get_effect(NoteSlideEffect):
                    target_note = active_note + noteslide_effect.semitones
                    target_note = max(0,min(target_note, MMLUtil.AMK_MAX_PITCH))
                    active_note = target_note

            out_ticks.append(new_tick)
                
        return out_ticks
    

    def resolve_macros(self, furnace_ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument]):
        vol_converter = VolumeMacroConverter()
        echo_converter = EchoMacroConverter()
        # currently active instrument
        active_ins = None 
        new_ticks = furnace_ticks
        for tick_data in new_ticks:
            note_kind = tick_data.kind()
            is_new_note = False
            if note_kind == FurnaceTickData.NoteKind.NOTE:
                is_new_note = True
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

            if echo_effect := echo_converter.get_echo_for_tick(tick_data, is_new_note, active_ins):
                tick_data.Effects.append(echo_effect)

        return new_ticks
    
    def resolve_sample_maps(self, furnace_ticks: List[FurnaceTickData], instruments: List[FurnaceInstrument], ins_info: Dict[int, FurInstrumentInfo]):
        '''Updates FurnaceTickData with sample-mapped instruments and notes, abstracting away sample maps'''
        # currently active instrument
        active_ins: FurnaceInstrument = None 
        new_ticks: List[FurnaceTickData] = furnace_ticks
        for i, tick_data in enumerate(furnace_ticks):
            new_fur_ins = None
            for ins in instruments:
                if ins.index == tick_data.Ins:
                    new_fur_ins = ins
                    break

            if new_fur_ins is not None:
                active_ins = new_fur_ins

            if tick_data.kind() == FurnaceTickData.NoteKind.NOTE:
                if active_ins is None:
                    self.logger.warning(f"No furnace instrument active in row with Note {tick_data.Note}, tick {i}.")
                    break
                note = tick_data.Note
                # Get the chiptune instrument index using ins_map
                if ins.use_sample_map:
                    note_map = ins_info[active_ins.index].ins_map
                    # Try to find exact note match first
                    if note in note_map:
                        chip_ins_idx = note_map[note].amk_ins_idx
                        note_to_play = note_map[note].note_to_play
                    else:
                        self.logger.warning(f"No instrument mapping found for Furnace instrument {chip_ins_idx}, note {note}.")
                        chip_ins_idx = 0
                        note_to_play = note

                    new_ticks[i].Note = note_to_play
                    # FIXME: this assignment fundamentally changes the meaning of FurnaceTickData::Ins from furnace ins to chiptune ins
                    # should really be a new data structure for clarity
                    new_ticks[i].Ins = chip_ins_idx
                else:
                    # still need to update instrument index for non-sample mapped instruments
                    # since they may ave been bumped up by prior sample-mapped instruments
                    new_ticks[i].Ins = ins_info[active_ins.index].default_ins

        return new_ticks
