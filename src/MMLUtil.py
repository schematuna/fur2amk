from typing import List, Tuple, Optional
from dataclasses import dataclass



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

    @staticmethod
    def note_name_and_octave(i: int) -> Tuple[str, int]:
        # highest allowed AMK pitch is o6 a
        # TODO: use pitch bend or something to fix automatically?
        while i > 141:
            i -= 12
        # Map Furnace note index (0=C-0) to AMK note name and octave using oN
        names = ['c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+', 'a', 'a+', 'b']
        note = i % 12
        octave = i // 12 - 5  # align with fur2tad convention
        return names[note], octave


@dataclass
class MMLState:
    octave: Optional[int]       = None
    ins: Optional[int]          = None

class DurationFormatter:
    # a "beat" is a quarter note, and the song is assumed to be in 4/4
    # This is how AMK works - a whole note is always 4 quarter notes
    # TODO: support triplets for relevant time signatures and/or beat subdivisions

    def format(self, duration_ticks: int, continuation: bool = False) -> str:
        """Format a note or rest token with duration and ties.
        
        Args:
            duration_ticks: Duration in ticks
        
        Returns:
            Formatted token with duration (e.g., 'c16', 'r8^16', 'c1^2^4')
        """
        if duration_ticks <= 0:
            return ''
        
        # Use run_to_denoms to convert ticks to MML duration denominators
        denoms, remainder = self.run_to_denoms(duration_ticks)
        
        token = ''
        
        if len(denoms) > 0:
            if continuation:
                token += '^'
            
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

    def divisors(self, n: int) -> List[int]:
        n = int(n)
        if n <= 0:
            return [1]
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return sorted(divs)

    def run_to_denoms(self, ticks: int) -> Tuple[List[int], int]:
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