from typing import Optional, List
from dataclasses import dataclass, field

from ..util.MMLUtil import MMLUtil

from ..model.MMLData import *
from ..model.ChiptuneData import *

import copy

# persistent channel state for conversion process
@dataclass
class AMKState:
    remote_commands: List[RemoteCommand] = field(default_factory=list)
    is_echo: bool = True
    adsr: ADSR = None
    ins_idx: int = None

class AMKUtil:
    # Convert from ChiptuneData pan format (00=left, 80=center, FF=right)
    # to AMK format (0=right, 10=center, 20=left)
    @staticmethod
    def unity_to_amk_pan(pan: int) -> int:
        pan = max(0, min(255, pan))
        return MMLUtil.find_y(pan)
    
    @staticmethod
    def tick_rate_to_amk_tempo(structure: ChiptuneStructure, amk_ticks_per_row: int, tick_rate: int) -> int:
        rows_per_beat = MMLUtil.AMK_TICKS_PER_BEAT / amk_ticks_per_row
        fur_ticks_per_beat = rows_per_beat * structure.ticks_per_step[0]
        beats_per_second = tick_rate / fur_ticks_per_beat
        bpm = int(round(60 * beats_per_second))
        amk_tempo = int(round(bpm * 0.4096 - 1))
        return amk_tempo
    
    @staticmethod
    def get_note_active_at(tick: int, notes: List[MMLNote]) -> Tuple[Optional[MMLNote], Optional[int]]:
        """Find the note that is active (playing) at the given tick."""
        for i, note in enumerate(notes):
            if note.duration is None:
                continue
            # at note boundaries, defer to the earlier note
            if note.tick < tick <= note.tick + note.duration:
                return note, i
        return None, None

    @staticmethod
    def get_note_starting_at(tick: int, notes: List[MMLNote]) -> Optional[MMLNote]:
        """Find the note that starts at the given tick."""
        for note in notes:
            if note.tick == tick:
                return note
        return None
    
    @staticmethod
    def split_note(note, tick) -> Tuple[MMLNote, MMLNote]:
        note1 = copy.deepcopy(note)
        note1.duration = tick - note1.tick
        note2 = copy.deepcopy(note)
        note2.tick = tick
        note2.duration = note.duration - note1.duration
        # only first note keeps the pre-note commands
        note2.pre_note_commands = []

        return note1, note2
