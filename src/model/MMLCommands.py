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

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        raise NotImplementedError("Subclasses must implement to_mml")

@dataclass
class EchoToggle(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"$F4$03"

@dataclass
class InstrumentChange(MMLCommand):
    instrument_index: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f'@{self.instrument_index + 30}'

@dataclass
class VolumeChange(MMLCommand):
    volume: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        vol_mml = MMLUtil.find_v(self.volume)
        return f'v{vol_mml}'

@dataclass
class PanChange(MMLCommand):
    pan: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return f"y{self.pan}"

@dataclass
class PitchBend(MMLCommand):
    note: int
    speed: int

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        bend_note, octave = MMLUtil.note_name_and_octave(self.note)
        if octave != mml_state.octave:
            bend_note = f'o{octave}{bend_note}'
            mml_state.octave = octave
        # TODO: handling delay correctly here?
        # amk_delay = MMLUtil.to_hex(delay * 8) # $08 = 1 eighth note
        return f"$DD${MMLUtil.to_hex(0)}${MMLUtil.to_hex(self.speed)} {bend_note}"

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