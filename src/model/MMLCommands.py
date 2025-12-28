from typing import Optional
from enum import Enum
from dataclasses import dataclass, field

from ..MMLUtil import MMLUtil, MMLState


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
    tick: int = field(compare=False)

    # override to remove spaces from the command text
    def add_spaces(self, text: str) -> str:
        return text + ' '

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        raise NotImplementedError("Subclasses must implement to_mml")

    def get_text(self, mml_state: 'MMLState' = None) -> str:
        return self.add_spaces(self.to_mml(mml_state))


@dataclass
class EchoToggle(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$F4$03"

@dataclass
class LegatoToggle(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$F4$01"

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

    def add_spaces(self, text: str) -> str:
        return text

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        bend_note, octave = MMLUtil.note_name_and_octave(self.note)
        if octave != mml_state.octave:
            bend_note = f'o{octave}{bend_note}'
            mml_state.octave = octave
        return f"$DD$00${MMLUtil.to_hex(self.duration)} {bend_note}"

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
    gain: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$FA$01${MMLUtil.to_hex(self.gain)}"

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