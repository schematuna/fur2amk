import logging

from ..model.FurnaceData import *

class VolumeMacroConverter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # default starting volume is 7F
        self.primary_vol = 127
        self.macro_mult = 1

    def get_volume_for_row(self, row_vol: int, is_new_note: bool, active_ins: FurnaceInstrument):
        new_vol = row_vol
        if row_vol is not None:
            self.primary_vol = new_vol

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
        if row_vol is not None:
            new_vol = self.primary_vol * self.macro_mult

        return new_vol

        
        