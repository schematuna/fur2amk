import logging

from ..model.FurnaceData import *
from ..util import *

# preprocessor for Furnace row data
# abstracts away macros and other commands that can be reduced to more elemental commands
# this simplifies the conversion to AMK commands

class RowPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def convert(self, flat_rows: List[FurnaceRow], instruments: List[FurnaceInstrument]):
        # default starting volume is 7F
        primary_vol = 127
        macro_mult = 1
        # currently active instrument
        fur_ins = None 
        processed_rows: List[FurnaceRow] = []
        for row in flat_rows:
            new_vol = row.Vol
            if new_vol is not None:
                primary_vol = new_vol

            note_kind = row.kind()
            if note_kind == FurnaceRow.NoteKind.NOTE:                    
                new_fur_ins = None
                for ins in instruments:
                    if ins.index == row.Ins:
                        new_fur_ins = ins
                        break

                if new_fur_ins is not None:
                    fur_ins = new_fur_ins

                if fur_ins is None:
                    self.logger.error(f"No furnace instrument active in row with Note {row.Note}.")
                    continue

                # only consider first tick of volume macro for now
                new_macro_mult = macro_mult
                if fur_ins.snes_macro_data.vol_values:
                    new_macro_mult = fur_ins.snes_macro_data.vol_values[0] / 127
                else:
                    # mult resets to "normal" if no volume macro for this instrument
                    new_macro_mult = 1

                # note onsets with new macro mult trigger a volume change on this row
                if new_macro_mult != macro_mult:
                    macro_mult = new_macro_mult
                    row.Vol = primary_vol * macro_mult
                
            # all volume commands are affected by any active volue macros.
            if new_vol is not None:
                row.Vol = primary_vol * macro_mult

            processed_rows.append(row)

        return processed_rows