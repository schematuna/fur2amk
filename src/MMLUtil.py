from typing import List, Tuple, Optional
from dataclasses import dataclass

class MMLUtil:
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
    echo: bool                  = False
    remote_gain: Optional[int]  = None
    vol: Optional[int]          = None

class DurationFormatter:
    def __init__(self, ticks_per_subdivision, base_den) -> None:
        self.ticks_per_subdivision = ticks_per_subdivision
        self.base_den = base_den

    def format(self, duration_ticks: int, continuation: bool = False) -> str:
        """Format a note or rest token with duration and ties.
        
        Args:
            duration_ticks: Duration in ticks
        
        Returns:
            Formatted token with duration (e.g., 'c16', 'r8^16', 'c1^2^4')
        """
        if duration_ticks <= 0:
            return ''
        
        # Convert ticks to number of base_den subdivisions
        # Each subdivision = ticks_per_subdivision ticks
        num_subdivisions = duration_ticks / self.ticks_per_subdivision
        
        # Use run_to_denoms to convert subdivisions to MML duration denominators
        denoms = self.run_to_denoms(int(round(num_subdivisions)), self.base_den)
        
        if len(denoms) == 0:
            return ''
        
        token = ''
        if continuation:
            token += '^'
        
        token += str(denoms[0])
        
        # Additional durations use tie syntax
        for d in denoms[1:]:
            token += f'^{d}'
        
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

    def run_to_denoms(self, num_subdivisions: int, base_den: int, no_whole_notes: bool = False) -> List[int]:
        """Decompose a number of base_den subdivisions into a list of AMK length denominators to tie.

        Each subdivision represents 1/base_den of a whole note. We choose chunks that are divisors of base_den
        and sum to num_subdivisions. For each chunk, the length number is base_den/chunk.
        Example: base_den=16, num_subdivisions=3 -> chunks [2,1] => denoms [8,16] -> c8^16.
        """
        num = max(1, int(num_subdivisions))
        bd = max(1, int(base_den))
        divs = self.divisors(bd)
        # remove divisor of 16 if no_whole_notes
        if no_whole_notes:
            divs = [d for d in divs if d < 16]
        # allowed chunks are divisors of base_den
        chunks = sorted(divs, reverse=True)
        out: List[int] = []
        rem = num
        while rem > 0:
            # pick largest chunk <= rem
            pick = None
            for c in chunks:
                if c <= rem:
                    pick = c
                    break
            if pick is None:
                # fallback to 1-subdivision chunks (shouldn't happen since 1 divides bd)
                pick = 1
            out.append(bd // pick)
            rem -= pick
        return out