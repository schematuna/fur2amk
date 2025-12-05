from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Any, Optional

from AMKData import EventTable

class AMKRemoteCommandType(Enum):
    GAIN = 0

class AMKRemoteCommandTiming(Enum):
    DISABLE = auto()
    AFTER_START = auto()
    BEFORE_END = auto()
    KEY_OFF = auto()
    RUN_NOW = auto()
    KEY_ON = auto()

class EventType(Enum):
    NOTE = auto()
    NOTE_OFF = auto()
    INS_CHANGE = auto()
    VOLUME = auto()
    EFFECT = auto()

@dataclass
class AMKRemoteCommand:
    command_idx: int
    event_type: AMKRemoteCommandTiming
    amk_command_type: AMKRemoteCommandType
    remote_command_arg: Optional[Any] = None
    amk_command_args: List[Any] = field(default_factory=list)


class AMKInstrument:
    def __init__(self, index: int, sample_index: int = None, is_noise: bool = False, noise_freq: int = 0) -> None:
        self.index = index
        self.is_noise = is_noise
        if is_noise:
            self.noise_freq = noise_freq
            self.sample_index = None
        else:
            self.sample_index = sample_index
            self.noise_freq = 0
        self.remote_commands: List[AMKRemoteCommand] = []

    @classmethod
    def noise(cls, index: int, noise_freq: int) -> 'AMKInstrument':
        return cls(index=index, is_noise=True, noise_freq=noise_freq)

    @classmethod
    def sample(cls, index: int, sample_index: int) -> 'AMKInstrument':
        return cls(index=index, sample_index=sample_index, is_noise=False)

@dataclass
class Event:
    tick: int
    effect: str
    value: Any

@dataclass
class EventTable:
    events: List[List['Event']] = field(default_factory=lambda: [[] for _ in range(8)])
    intro_order: Optional[int] = None

@dataclass
class AMKData:
    version: int = 2
    title: str = ""
    author: str = ""
    game: str = ""
    length: int = 0
    comment: str = ""

    sample_path: str = ""
    # index -> filename, tuning string
    samples: Dict[int, Tuple[str, str]] = {}
    # List of (instrument_index, sample_index) pairs to emit in #instruments
    instruments: List[Tuple[int, AMKInstrument]] = []

    # Events for MML emission
    event_table: EventTable = None
    # which order to place the intro marker before
    intro_order: int = None

    label_start: int = 1
