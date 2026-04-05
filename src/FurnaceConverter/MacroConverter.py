import logging

from ..model.FurnaceData import *

class VolumeMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # default starting volume is 7F
        self.primary_vol = 127
        self.macro_mult = 1

    def get_volume_for_tick(self, tick_data: FurnaceTickData, is_new_note: bool, active_ins: FurnaceInstrument):
        new_vol = tick_data.Vol
        if tick_data.Vol is not None:
            self.primary_vol = new_vol

        if vol_change_effect := tick_data.get_effect(SingleTickVolumeEffect):
            self.primary_vol += vol_change_effect.vol_change
            new_vol = self.primary_vol * self.macro_mult

        if is_new_note:
            # only consider first tick of volume macro for now
            new_macro_mult = self.macro_mult
            if active_ins.snes_macro_data.vol_values:
                new_macro_mult = active_ins.snes_macro_data.vol_values[0] / 127
            else:
                # mult resets to "normal" if no volume macro for this instrument
                new_macro_mult = 1

            # note onsets with new macro mult trigger a volume change on this row
            if new_macro_mult != self.macro_mult:
                self.macro_mult = new_macro_mult
                new_vol = self.primary_vol * self.macro_mult
            
        # all volume commands are affected by any active volue macros.
        if tick_data.Vol is not None:
            new_vol = self.primary_vol * self.macro_mult

        # must limit 0->254
        if new_vol:
            new_vol = min(max(0, new_vol), 254)

        return new_vol
    
class EchoMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # TODO: implement echo command 12XX, accounting for channel echo initialization in chip settings
        # self.global_echo: bool = 1
        self.ins_echo: bool = None

    def get_echo_for_tick(self, tick_data: FurnaceTickData, is_new_note: bool, active_ins: FurnaceInstrument):
        echo_effect = None
        if is_new_note:
            echo_macro = active_ins.snes_macro_data.is_echo
            if echo_macro is not None:
                new_ins_echo = active_ins.snes_macro_data.is_echo

                if new_ins_echo != self.ins_echo:
                    self.ins_echo = new_ins_echo
                    echo_effect = EchoEffect(new_ins_echo)

        return echo_effect