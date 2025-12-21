from dataclasses import dataclass, field
from typing import List, Optional

from .MMLCommands import MMLCommand

@dataclass
class MMLNote:
    tick: int
    duration: int = None
    note: int = None

@dataclass
class MMLData:
    num_channels: int = 8
    # number of ticks in the song
    song_length: int = 0

    # list of notes for each channel. Rests are handled automatically by the MMLWriter.
    notes: List[List['MMLNote']] = field(default_factory=lambda: [[] for _ in range(8)])
    commands: List[List['MMLCommand']] = field(default_factory=lambda: [[] for _ in range(8)])
    intro_order: Optional[int] = None

    # song data for formatting
    beat_length: int = 4
    measure_length: int = 16
    pattern_length: int = 64
    ticks_per_subdivision: int = 0