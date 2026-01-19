from dataclasses import dataclass
from enum import Enum
import logging

# common music utilities

@dataclass
class ADSR:
    attack: int = None
    decay: int = None
    sustain: int = None
    release: int = None

class GainMode(Enum):
    """SNES GAIN register modes

    When useEnv is False, the GAIN register controls the envelope instead of ADSR.
    """
    DIRECT = 0       # Direct volume level (0-127)
    DEC_LINEAR = 4   # Linear decrease (0-31 rate)
    DEC_LOG = 5      # Exponential/logarithmic decrease (0-31 rate)
    INC_LINEAR = 6   # Linear increase (0-31 rate)
    INC_INVLOG = 7   # Bent/inverse-log increase (0-31 rate)

class SnesGain:
    def __init__(self, mode: GainMode, gain: int):
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.gain = gain

    def to_byte(self) -> int:
        gain_byte = self.gain
        if self.mode == GainMode.DIRECT:
            if self.gain > 0x7F:
                self.logger.warning(f"Gain value {self.gain} is too high for direct mode.")
                gain_byte = 0x7F
        else:
            if self.gain > 0x1F:
                self.logger.warning(f"Gain value {self.gain} is too high for {self.mode} mode.")
                gain_byte = 0x1F
        if self.mode == GainMode.DEC_LINEAR:
            gain_byte = self.gain + 0x80
        elif self.mode == GainMode.DEC_LOG:
            gain_byte = self.gain + 0xA0
        elif self.mode == GainMode.INC_LINEAR:
            gain_byte = self.gain + 0xC0
        elif self.mode == GainMode.INC_INVLOG:
            gain_byte = self.gain + 0xE0

        return gain_byte

    @staticmethod
    def from_byte(byte: int) -> 'SnesGain':
        if byte >= 0x80 and byte <= 0x9F:
            return SnesGain(GainMode.DEC_LINEAR, byte - 0x80)
        elif byte >= 0xA0 and byte <= 0xBF:
            return SnesGain(GainMode.DEC_LOG, byte - 0xA0)
        elif byte >= 0xC0 and byte <= 0xDF:
            return SnesGain(GainMode.INC_LINEAR, byte - 0xC0)
        elif byte >= 0xE0 and byte <= 0xFF:
            return SnesGain(GainMode.INC_INVLOG, byte - 0xE0)
        else:
            return SnesGain(GainMode.DIRECT, byte)