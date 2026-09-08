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

class TremoloMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.last_ins_index = None
        self.tremolo_active = False

    def get_tremolo_for_tick(self, active_ins: FurnaceInstrument) -> TremoloCommand | None:
        if active_ins is None or active_ins.index == self.last_ins_index:
            return None
        self.last_ins_index = active_ins.index

        vol_macro = active_ins.get_macro(SNESMacroCode.Volume)
        if vol_macro and vol_macro.type == SNESMacroTypes.LFO.value:
            depth = vol_macro.lfo_top - vol_macro.lfo_bottom
            ticks = self._speed_to_ticks(vol_macro.lfo_speed, depth, vol_macro.lfo_shape)
            self.tremolo_active = True
            cmd = TremoloCommand(vol_macro.delay, ticks, depth)
            return cmd

        if self.tremolo_active:
            self.tremolo_active = False
            cmd = TremoloCommand(0, 0, 0)
            return cmd

        return None

    def _speed_to_ticks(self, speed: int, depth: int, shape: int) -> int:
        '''Converts Furnace's LFO speed parameter to a period length in ticks,
           matching the tooltip shown in Furnace's macro editor (insEdit.cpp).'''
        if speed is None or speed < 1:
            return 0
        lfo_range = abs(depth)
        param_max = 65536 if shape == LFOShape.Square.value else ((lfo_range << 8) | 0xFF)
        return (param_max // speed) * (2 if shape == LFOShape.Triangle.value else 1)

class ArpMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.timer = None
        self.primary_note = 0
        self.in_legato_seq = False
        self.last_val = None

    def get_arp_for_tick(self, tick_data: FurnaceTickData, active_ins: FurnaceInstrument, chip_note: int) -> Tuple[int | None, LegatoEnableCommand | None]:
        if chip_note is not None:
            arp_macro = active_ins.get_macro(SNESMacroCode.Arpeggio) if active_ins else None
            if arp_macro and arp_macro.type == SNESMacroTypes.Sequence.value:
                self.timer = MacroTimer(arp_macro)
            else:
                self.timer = None
            
            self.primary_note = chip_note
            self.last_val = None

        is_note_release = tick_data.kind() == FurnaceTickData.NoteKind.RELEASE
        if is_note_release:
            self.timer = None

        new_note = None
        legato_cmd = None
        if self.timer is not None:
            val = self.timer.tick()
            # make a new note for new arp vals
            if val is not None and val != self.last_val:
                self.last_val = val
                new_note = self.primary_note + val
                # turn on legato on second note.
                if self.timer.cur_step >= 1 and not self.in_legato_seq:
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