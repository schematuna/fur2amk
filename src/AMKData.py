from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Any, Optional

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
    PITCH_BEND = auto()

@dataclass
class AMKRemoteCommand:
    command_idx: int
    event_type: AMKRemoteCommandTiming
    amk_command_type: AMKRemoteCommandType
    remote_command_arg: Optional[Any] = None
    amk_command_args: List[Any] = field(default_factory=list)

@dataclass
class AMKEnvelope:
    attack: int = 0
    decay: int = 0
    sustain: int = 0
    release: int = 0

@dataclass
class AMKInstrument:
    sample_index: Optional[int] = None

    is_noise: bool = False
    noise_freq: int = 0

    uses_envelope: bool = False
    envelope: Optional[AMKEnvelope] = None

    gain_values: List[int] = field(default_factory=list)

    gain: int = 0

@dataclass
class AMKEchoData:
    firIdx: Optional[bool] = None
    echoDelay: Optional[int] = None
    echoFeedback: Optional[int] = None
    echoMask: Optional[int] = None
    echoVolL: Optional[int] = None
    echoVolR: Optional[int] = None
    echoFilterCoeffs: Optional[List[int]] = None

@dataclass
class Event:
    tick: int
    type: EventType
    value: Any
    value2: Any = None

@dataclass
class EventTable:
    events: List[List['Event']] = field(default_factory=lambda: [[] for _ in range(8)])
    intro_order: Optional[int] = None

@dataclass
class SPCInfo:
    title: str = ""
    author: str = ""
    game: str = ""
    length: int = 0
    comment: str = ""

@dataclass
class AMKData:
    version: int = 2

    spc_info: SPCInfo = field(default_factory=SPCInfo)

    sample_path: str = ""
    # index -> filename, tuning string
    samples: Dict[int, Tuple[str, str]] = field(default_factory=dict)
    instruments: List[AMKInstrument] = field(default_factory=list)

    tempo: int = 0
    volume: int = 0

    echo_data: AMKEchoData = None

    remote_commands: List[AMKRemoteCommand] = field(default_factory=list)

    num_channels: int = 8
    # Events for MML emission
    event_table: EventTable = field(default_factory=EventTable)
    # which order to place the intro marker before
    intro_order: int = None

    # song data for formatting
    pattern_length: int = 0
    measure_length: int = 0
    ticks_per_beat: int = 0  # Speed1 * measure_length (ticks per row * rows per beat)

    label_start: int = 1
