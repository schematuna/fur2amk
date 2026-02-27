from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import logging

from .FurnaceEffects import *
from .FurnaceData import *
from ..util.SNESUtil import *

# generic chiptune data format
# acts as an intermediary between Furnace and AMK

@dataclass
class ChiptuneSongInfo:
    title: str = ""
    author: str = ""
    comment: str = ""

@dataclass
class ChiptuneStructure:
    num_channels: int = 8
    # for formatting and duration calculations
    # lengths are in ticks
    measure_length: int = 64
    section_lengths: List[int] = None
    song_length: int = 64

    # usually ticks per tracker row
    # used to decide what AMK base musical length should be
    ticks_per_step: int = 8

    # loop point
    loop_tick: int = None

@dataclass
class ChiptuneSampleInfo:
    index: int
    filename: str
    c4_rate: str

# class representing all events that occur during a single tick
@dataclass
class TickData:
    Note: Optional[int] = None
    Ins: Optional[int] = None
    Vol: Optional[int] = None   # 0..64
    # TODO: make this furnace-agnostic
    Effects: List[FurnaceEffect] = field(default_factory=list)

    # enum for note kinds
    class NoteKind(Enum):
        NOTE = 0
        RELEASE = 1
        EMPTY = 2

    # Classify a Furnace row by note type
    def kind(self) -> NoteKind:
        n = self.Note
        if n is None:
            return self.NoteKind.EMPTY
        
        try:
            v = int(n)
        except Exception:
            return self.NoteKind.EMPTY
        
        # for snes, note release and note off are the same
        if v == 180 or v == 181:
            return self.NoteKind.RELEASE
        if 0 <= v <= 179:
            return self.NoteKind.NOTE
        
        # no macro releases considered here, those are abstracted away at this point
        return self.NoteKind.EMPTY

    def get_effect(self, command_type: type[FurnaceEffect]) -> Optional[FurnaceEffect]:
        for effect in self.Effects:
            if isinstance(effect, command_type):
                return effect
        return None

@dataclass
class ChiptuneData:
    song_info: ChiptuneSongInfo = field(default_factory=ChiptuneSongInfo)
    structure: ChiptuneStructure = field(default_factory=ChiptuneStructure)
    sample_info: List[ChiptuneSampleInfo] = field(default_factory=list)
    # I'm lazy, just pass FurnaceInstrument object through
    instruments: List[FurnaceInstrument] = field(default_factory=list)
    echo_data: SNESEchoData = field(default_factory=list)
    # per-channel tick data
    tick_data: List[List[TickData]] = field(default_factory=list)
    # ticks per second
    tick_rate: int = 60
    # song volume from 0 -> 255
    global_volume: int = 0