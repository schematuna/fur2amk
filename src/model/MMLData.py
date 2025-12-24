from dataclasses import dataclass, field
from typing import List, Optional

from .MMLCommands import MMLCommand

@dataclass
class MMLNote:
    tick: int
    duration: int = None
    note: int = None
    instrument: int = None

@dataclass
class MMLData:
    num_channels: int = 8
    # number of ticks in the song
    song_length: int = 0

    # list of notes for each channel. Rests are handled automatically by the MMLWriter.
    notes: List[List['MMLNote']] = field(default_factory=lambda: [[] for _ in range(8)])
    commands: List[List['MMLCommand']] = field(default_factory=lambda: [[] for _ in range(8)])
    loop_tick: Optional[int] = None

    # song data for formatting, in ticks
    measure_length: int = 192
    section_length: int = 192 * 4