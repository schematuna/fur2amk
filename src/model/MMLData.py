from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .MMLCommands import *

@dataclass
class MMLNote:
    tick: int
    duration: int = None
    note: int = None
    instrument: int = None

    # commands that are qualities of the note
    # e.g. setting/resetting note state
    pre_note_commands: List[MMLCommand] = field(default_factory=lambda: [])

    # pitchbends must occur within the duration of the note
    pitch_bends: List[PitchBend] = field(default_factory=lambda: [])

    # whether this note should use the MML trick to avoid a 1-tick gap before the next note
    no_gap: bool = False

@dataclass
class MMLData:
    num_channels: int = 8
    # number of ticks in the song
    song_length: int = 0

    # list of notes for each channel. Rests are handled automatically by the MMLWriter.
    notes: Dict[int, List['MMLNote']] = field(default_factory=dict)
    # free floating commands that are not necessarily attached to a note
    commands: Dict[int, List['MMLCommand']] = field(default_factory=dict)
    loop_tick: Optional[int] = None

    # song data for formatting, in ticks
    measure_length: int = 192
    section_lengths: List[int] = field(default_factory=list)  # [order] -> ticks per section