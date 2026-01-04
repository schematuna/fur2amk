from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging


class MMLUtil:
    AMK_TICKS_PER_BEAT = 48
    TICK_TO_DURATION = {
        int(AMK_TICKS_PER_BEAT / 16): 64,
        int(AMK_TICKS_PER_BEAT / 8): 32,
        int(AMK_TICKS_PER_BEAT / 4): 16,
        int(AMK_TICKS_PER_BEAT / 2): 8,
        int(AMK_TICKS_PER_BEAT): 4,
        int(AMK_TICKS_PER_BEAT * 2): 2,
        int(AMK_TICKS_PER_BEAT * 4): 1
    }

    AMK_MIN_PITCH = 72  # o1 c
    AMK_MAX_PITCH = 141  # o6 a

    # Track out-of-range notes
    _out_of_range_count = 0
    # Convert -128->127 ranged values to 2's complement hex
    @staticmethod
    def to_hex(val: int) -> str:
        return f"{(val & 0xFF):02X}" if val >= 0 else f"{((val + 256) & 0xFF):02X}"

    # wild amk volume mapping function stol from it2amk
    @staticmethod
    def find_v(level: int) -> int:
        if level == 0:
            return 0
    
        mindiff = 256
        minval = -1
        
        # amk -> fur mapping function l ≈ v² * constant_factor
        # not easily reversible, so we do a brute force reverse lookup
        for v in range(0, 256):
            vv = (v * 0xFF) >> 8
            vv = (vv * vv) >> 8
            vv = (vv * 0x51) >> 8
            vv = (vv * 0xFC) >> 8
            l = vv * 0xFF / 0x4D
            
            if abs(l - level) <= mindiff:
                mindiff = abs(l - level)
                minval = v

        return minval

    # also stol from it2amk
    @staticmethod
    def find_y(fur_pan: int) -> int:
        smw_pan_tbl = [0x00, 0x01, 0x03, 0x07, 0x0D, 0x15, 0x1E, 0x29, 0x34, 0x42,
                        0x51, 0x5E, 0x67, 0x6E, 0x73, 0x77, 0x7A, 0x7C, 0x7D, 0x7E, 0x7F]
        rvol = max(fur_pan, 1) - 0x01
        lvol = 0xFF - max(fur_pan, 1)
        
        diff, base_pan, rnorm = 1000000, 1.0, None
        for p in range(len(smw_pan_tbl)):
            plvol = smw_pan_tbl[p]
            prvol = smw_pan_tbl[20 - p]
            sum = plvol + prvol
            norm = 254.0 / sum
            
            plvol *= norm
            prvol *= norm
            
            tdiff = abs(plvol - lvol) + abs(prvol - rvol)
            
            if tdiff < diff:
                base_pan = p
                diff = tdiff
                # for normalizing volume, not used for now
                rnorm = norm

        return base_pan

    @staticmethod
    def note_name_and_octave(i: int) -> Tuple[str, int]:
        original_i = i
        # highest allowed AMK pitch is o6 a
        # TODO: use pitch bend or something to fix automatically?
        while i > MMLUtil.AMK_MAX_PITCH:
            i -= 12
        # lowest allowed AMK pitch is o1 c
        while i < MMLUtil.AMK_MIN_PITCH:
            i += 12

        # Track if note was out of range
        if i != original_i:
            MMLUtil._out_of_range_count += 1

        # Map Furnace note index (0=C-0) to AMK note name and octave using oN
        names = ['c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+', 'a', 'a+', 'b']
        note = i % 12
        octave = i // 12 - 5  # align with fur2tad convention
        return names[note], octave

    @staticmethod
    def reset_out_of_range_count():
        """Reset the out-of-range note counter."""
        MMLUtil._out_of_range_count = 0

    @staticmethod
    def print_out_of_range_warning():
        """Print a warning if any notes were out of range."""
        if MMLUtil._out_of_range_count > 0:
            logger = logging.getLogger(__name__)
            logger.warning(f"{MMLUtil._out_of_range_count} note(s) were out of AMK's supported range (o1 c to o6 a) and were shifted by octaves to fit.")

@dataclass
class MMLState:
    octave: Optional[int]       = None
    ins: Optional[int]          = None

class DurationFormatter:
    # a "beat" is a quarter note, and the song is assumed to be in 4/4
    # This is how AMK works - a whole note is always 4 quarter notes
    # TODO: support triplets for relevant time signatures and/or beat subdivisions

    @staticmethod
    def format(duration_ticks: int, continuation: bool = False) -> str:
        """Format a note or rest token with duration and ties.
        
        Args:
            duration_ticks: Duration in ticks
        
        Returns:
            Formatted token with duration (e.g., 'c16', 'r8^16', 'c1^2^4')
        """
        if duration_ticks <= 0:
            return ''
        
        # Use run_to_denoms to convert ticks to MML duration denominators
        denoms, remainder = DurationFormatter.run_to_denoms(duration_ticks)
        
        token = ''
    
        if continuation:
            token += '^'

        if len(denoms) > 0:
            token += str(denoms[0])
            
            # Additional durations use tie syntax
            for d in denoms[1:]:
                token += f'^{d}'
        
        # Handle remainder using ={ticks} notation
        if remainder > 0:
            if len(denoms) > 0:
                token += '^'
            token += f'={remainder}'

        return token

    @staticmethod
    def run_to_denoms(ticks: int) -> Tuple[List[int], int]:
        """Decompose ticks into a list of AMK duration denominators to tie.

        Uses TICK_TO_DURATION to map tick values to duration denominators.
        Returns (denoms_list, remainder_ticks) where remainder_ticks should be formatted as ={ticks}.
        Example: 27 ticks -> 24 ticks (8th note) + 3 ticks remainder -> ([8], 3) -> c8^=3.
        """
        if ticks <= 0:
            return ([], 0)
        
        # Get sorted tick values in descending order for greedy decomposition
        tick_values = sorted([int(k) for k in MMLUtil.TICK_TO_DURATION.keys()], reverse=True)
        
        out: List[int] = []
        remaining = ticks
        
        # Greedily decompose: pick largest tick value that fits
        for tick_value in tick_values:
            count = remaining // tick_value
            if count > 0:
                duration = MMLUtil.TICK_TO_DURATION[tick_value]
                # Add the duration for each occurrence
                out.extend([duration] * count)
                remaining -= tick_value * count
        
        return (out, remaining)