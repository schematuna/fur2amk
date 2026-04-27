from typing import Optional
from enum import Enum
from dataclasses import dataclass, field

from ..util.SNESUtil import *
from ..util.MMLUtil import *

class RemoteCommandTiming(Enum):
    DISABLE = 0
    AFTER_START = 1
    BEFORE_END = 2
    KEY_OFF = 3
    RUN_NOW = 4
    DISABLE_EXCEPT_KEY_ON = 7
    DISABLE_KEY_ON = 8
    KEY_ON = -1

@dataclass
class MMLCommand:
    # can be relative to song start or relative to the note start, depending on the command
    tick: int = field(compare=False)

    # override to remove spaces from the command text
    def add_spaces(self, text: str) -> str:
        return text + ' '

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        raise NotImplementedError("Subclasses must implement to_mml")

    def get_text(self, mml_state: 'MMLState' = None) -> str:
        return self.add_spaces(self.to_mml(mml_state))


@dataclass
class LegatoToggle(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$F4$01"

@dataclass
class LightStaccatoToggle(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$F4$02"

@dataclass
class EchoToggle(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$F4$03"

@dataclass
class VolumeTableToggle(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$F4$08"

@dataclass
class InstrumentChange(MMLCommand):
    instrument_index: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f'@{self.instrument_index + 30}'

@dataclass
class VolumeChange(MMLCommand):
    volume: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f'v{self.volume}'

@dataclass
class PanChange(MMLCommand):
    pan: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"y{self.pan}"

@dataclass
class PitchBend(MMLCommand):
    # AMK Pitchbend command has a delay field but we never use that here
    # instead MMLWriter will figure out the correct placement of the command automatically
    duration: int
    note: int

    # no space after target note
    def add_spaces(self, text: str) -> str:
        return text

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        bend_note, octave = MMLUtil.note_name_and_octave(self.note)
        if octave != mml_state.octave:
            bend_note = f'o{octave} {bend_note}'
            mml_state.octave = octave
        return f"$DD$00${MMLUtil.to_hex(self.duration)} {bend_note}"

@dataclass
class TempPitchBend(MMLCommand):
    '''Intermediary pitchbend command
       After conversion from chiptune data
       but before conversion to PitchEnvelope'''
    
    duration: int
    target_note: float

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return "PLACEHOLDER"

@dataclass
class PitchEnvelope(MMLCommand):
    # Attack Pitch Envelope
    # Prefer this to regular PitchBend command because it doesn't require special duration formatting

    delay: int     # 0 -> 255
    duration: int  # 0 -> 255
    semitones: int # -127 -> 128

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$EB${MMLUtil.to_hex(self.delay)}${MMLUtil.to_hex(self.duration)}${MMLUtil.to_hex(self.semitones)}"

   

@dataclass
class VolumeFade(MMLCommand):
    duration: int = 0
    target_volume: int = 0

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$E8${MMLUtil.to_hex(self.duration)}${MMLUtil.to_hex(self.target_volume)}"

@dataclass
class PanFade(MMLCommand):
    duration: int = 0
    target_pan: int = 0

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$DC${MMLUtil.to_hex(self.duration)}${MMLUtil.to_hex(self.target_pan)}"

@dataclass
class EnableGainCommand(MMLCommand):
    gain: SnesGain

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$FA$01${MMLUtil.to_hex(self.gain.to_byte())}"

@dataclass
class RemoteCommand(MMLCommand):
    command_idx: int
    timing: RemoteCommandTiming
    wait_ticks: Optional[int] = None

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        if self.wait_ticks is not None:
            return f"(!{self.command_idx}, {self.timing.value}, ={self.wait_ticks})"
        else:
            return f"(!{self.command_idx}, {self.timing.value})"

@dataclass
class Vibrato(MMLCommand):
    duration: int  # speed
    amplitude: int  # depth

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        # Delay is always 0 since Furnace doesn't have a delay parameter for vibrato
        return f"$DE$00${MMLUtil.to_hex(self.duration)}${MMLUtil.to_hex(self.amplitude)}"

@dataclass
class DisableVibrato(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return "$DF"

@dataclass
class CustomADSR(MMLCommand):
    adsr: ADSR

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        # $ED $DA $SR
        # $DA = %0dddaaaa (decay 3 bits, attack 4 bits)
        # $SR = %sssrrrrr (sustain 3 bits, release 5 bits)
        da_byte = ((self.adsr.decay & 0x07) << 4) | (self.adsr.attack & 0x0F)
        sr_byte = ((self.adsr.sustain & 0x07) << 5) | (self.adsr.release & 0x1F)
        return f"$ED${MMLUtil.to_hex(da_byte)}${MMLUtil.to_hex(sr_byte)}"
    
@dataclass
class TempoChange(MMLCommand):
    tempo: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        # Could equivalently use $E2 here but t is cleaner
        return f"t{self.tempo}"

@dataclass
class FineTune(MMLCommand):
    # tunes the channel, 0 -> +1 semitone (0 -> 0xFF)
    tuning: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$EE${MMLUtil.to_hex(self.tuning)}"
    
@dataclass
class SemitoneTune(MMLCommand):
    # tunes the channel by a number of semitones
    # uses 2's complement
    semitones: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$FA$02${MMLUtil.to_hex(self.semitones)}"
