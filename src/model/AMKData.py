from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .MMLData import *

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
    firIdx: Optional[int] = None
    echoDelay: Optional[int] = None
    echoFeedback: Optional[int] = None
    echoMask: Optional[int] = None
    echoVolL: Optional[int] = None
    echoVolR: Optional[int] = None
    echoFilterCoeffs: Optional[List[int]] = None

@dataclass
class AMKRemoteDef():
    command_idx: int
    amk_command: MMLCommand
    comment: str = ""

    timing: RemoteCommandTiming = RemoteCommandTiming.AFTER_START
    wait_ticks: Optional[int] = None

@dataclass
class SPCInfo:
    title: str = ""
    author: str = ""
    game: str = ""
    length: int = 0
    comment: str = ""

    @staticmethod
    def sanitize_for_amk(s: str) -> str:
        """Sanitize a string for use in AMK #spc info fields.

        AMK's preprocessor treats ';' as a comment delimiter and erases
        everything from ';' to end of line, even inside quoted strings.
        Backslash is only valid as '\"' escape sequence.
        """
        # Remove semicolons (would be treated as comment start)
        # Remove backslashes (only valid escape is \" which we don't need)
        # Quotes could be escaped as \" but simpler to just remove them
        return s.translate(str.maketrans('', '', ';\\"'))

    def get_title(self) -> str:
        return self.sanitize_for_amk(self.title)

    def get_author(self) -> str:
        return self.sanitize_for_amk(self.author)

    def get_game(self) -> str:
        return self.sanitize_for_amk(self.game)

    def get_comment(self) -> str:
        return self.sanitize_for_amk(self.comment)

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

    remote_defs: List[AMKRemoteDef] = field(default_factory=list)

    mml_data: MMLData = field(default_factory=MMLData)

    label_start: int = 1
