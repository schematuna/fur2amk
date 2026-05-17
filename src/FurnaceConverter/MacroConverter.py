import logging
import sys

from ..model.FurnaceData import *
from ..model.ChiptuneData import *

class MacroTimer():
    def __init__(self, macro: FurnaceMacro):
        self.logger = logging.getLogger(__name__)
        if macro.type != 0:
            sys.exit("Cannot initialize MacroTimer with non-sequence macro.")
        self.delay = macro.delay
        self.speed = macro.speed
        self.values = macro.values

        self.cur_tick = 0
        self.cur_step = None

    def tick(self) -> int:
        '''Increments timer by one. 
           Returns the macro value corresponding to the prior tick, if one exists.'''
        if self.cur_tick < self.delay:
            cur_step = 0
        else:
            cur_step = (self.cur_tick - self.delay) // self.speed

        retVal = None
        if cur_step != self.cur_step and cur_step < len(self.values):
            retVal = self.values[cur_step]
            self.cur_step = cur_step

        self.cur_tick += 1

        return retVal

class VolumeMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # default starting volume is 7F
        self.primary_vol = 127
        self.macro_mult = 1
        self.timer = None

    def get_volume_for_tick(self, tick_data: FurnaceTickData, active_ins: FurnaceInstrument, resolved_vol: float = None):
        emit_vol_change = False
        if (new_vol := tick_data.Vol) is not None:
            self.primary_vol = new_vol
            emit_vol_change = True
        elif resolved_vol is not None:
            self.primary_vol = resolved_vol  # silently track; emission driven by macro state changes only

        if tick_data.kind() == FurnaceTickData.NoteKind.NOTE:
            vol_macro = active_ins.get_macro(SNESMacroCode.Volume)
            if vol_macro and vol_macro.type == SNESMacroTypes.Sequence.value:
                self.timer = MacroTimer(vol_macro)
            elif self.timer is not None:
                self.timer = None

        if self.timer is not None:
            if (new_macro_val := self.timer.tick()) is not None:
                new_macro_mult = new_macro_val / 127
                if new_macro_mult != self.macro_mult:
                    self.macro_mult = new_macro_mult
                    emit_vol_change = True
        elif self.macro_mult != 1:
            self.macro_mult = 1
            emit_vol_change = True

        if emit_vol_change:
            new_vol = self.primary_vol * self.macro_mult
            # must limit 0->254
            return min(max(0, new_vol), 254)

        return None
    
class ArpMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.timer = None
        self.primary_note = 0
        self.in_legato_seq = False

    def get_arp_for_tick(self, tick_data: FurnaceTickData, active_ins: FurnaceInstrument, chip_note: int) -> Tuple[int | None, LegatoEnableCommand | None]:
        if chip_note is not None:
            arp_macro = active_ins.get_macro(SNESMacroCode.Arpeggio) if active_ins else None
            if arp_macro and arp_macro.type == SNESMacroTypes.Sequence.value:
                self.timer = MacroTimer(arp_macro)
            else:
                self.timer = None
            
            self.primary_note = chip_note

        is_note_release = tick_data.kind() == ChiptuneTickData.NoteKind.RELEASE
        if is_note_release:
            self.timer = None

        new_note = None
        legato_cmd = None
        if self.timer is not None:
            val = self.timer.tick()
            if val is not None:
                new_note = self.primary_note + val
                # turn on legato on second note.
                if self.timer.cur_step == 1:
                    legato_cmd = LegatoEnableCommand(True)
                    self.in_legato_seq = True

        if chip_note is not None and self.in_legato_seq:
            # turn off legato on note after arp macro ends
            legato_cmd = LegatoEnableCommand(False)
            self.in_legato_seq = False

        # note slides affect pitch throughout entire note duration
        if note_slide_command := tick_data.get_effect(NoteSlideEffect):
            self.primary_note += note_slide_command.semitones

        return new_note, legato_cmd

    
class EchoMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ins_echo: bool = None

    def get_echo_for_tick(self, tick_data: FurnaceTickData, active_ins: FurnaceInstrument):
        is_new_note = tick_data.kind() == FurnaceTickData.NoteKind.NOTE 
        echo_effect = None
        if is_new_note:
            new_echo = active_ins.get_special_flag(SpecialFlag.Echo)
            if new_echo is not None:
                if new_echo != self.ins_echo:
                    self.ins_echo = new_echo
                    echo_effect = EchoEffect(new_echo)

        return echo_effect