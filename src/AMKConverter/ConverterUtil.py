from typing import Optional, List
from dataclasses import dataclass, field

from ..util.MMLUtil import MMLUtil

from ..model.FurnaceEffects import *
from ..model.MMLData import *
from ..model.FurnaceData import *
from ..model.ChiptuneData import *

from .SlideHelpers import Slide, VolumeSlide, SlideHelper

import copy

# persistent channel state for conversion process
@dataclass
class FurnaceState:
    remote_commands: List[RemoteCommand] = field(default_factory=list)
    is_echo: bool = True
    adsr: ADSR = None
    fur_ins_idx: int = None

class FurnaceUtil:
    PITCH_STEPS_PER_OCTAVE = 384

    @staticmethod
    def fur_pitch_change_to_semitones(change: int) -> float:
        semitones = change * 12 / FurnaceUtil.PITCH_STEPS_PER_OCTAVE
        return semitones

    @staticmethod
    def ticks_from_speed(speed: int, semitones: int) -> float:
        ticks_per_octave = FurnaceUtil.PITCH_STEPS_PER_OCTAVE / speed
        octaves_to_slide = abs(semitones) / 12
        return ticks_per_octave * octaves_to_slide

    # Convert from Furnace unity pan format (00=left, 80=center, FF=right)
    # to AMK format (0=right, 10=center, 20=left)
    @staticmethod
    def unity_to_amk_pan(pan: int) -> int:
        pan = max(0, min(255, pan))
        return MMLUtil.find_y(pan)

    # Convert from Furnace stereo pan format (left and right, 0->15)
    # to AMK format (0=right, 10=center, 20=left)
    @staticmethod
    def stereo_to_amk_pan(left: int, right: int) -> int:
        unity_pan = FurnaceUtil.stereo_to_unity_pan(left, right)
        return FurnaceUtil.unity_to_amk_pan(unity_pan)

    # Convert from Furnace stereo pan format (left and right, both 0->15)
    # to Furnace unity pan format (00=left, 80=center, FF=right)
    @staticmethod
    def stereo_to_unity_pan(left: int, right: int) -> int:
        # Clamp to valid range
        left = max(0, min(15, left))
        right = max(0, min(15, right))
        
        # Handle edge cases
        if left == 0 and right == 0:
            return 0x80
        
        # Calculate linear pan based on relative balance
        total = left + right
        level = round(255 * right / total)
        
        return max(0, min(255, level))
    
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

# Slide objects created by the helpers
@dataclass
class PitchSlide(Slide):
    target: int = 0

@dataclass
class PanSlide(Slide):
    target: int = 0

class PitchSlider(SlideHelper):
    @staticmethod
    def get_max_duration() -> int:
        # EB command can got to $FF but just do C0 for cleanliness
        return 0xC0

    def _get_target_amk(self) -> float:
        return self.target_val

    def _get_change_per_tick(self, effect: FurnaceEffect) -> float:
        if effect.change_per_tick is not None:
            return FurnaceUtil.fur_pitch_change_to_semitones(effect.change_per_tick / self.tick_ratio)
        else:
            return None

    def _limit_target_val(self, target_val: float) -> float:
        return max(MMLUtil.AMK_MIN_PITCH, min(target_val, MMLUtil.AMK_MAX_PITCH))

    def _get_command(self, tick: int, duration: int, target_note: float) -> Slide:
        return PitchSlide(tick, duration, target_note)


class PanSlider(SlideHelper):
    def __init__(self, tick_ratio: int, starting_tick: int) -> None:
        super().__init__(tick_ratio, starting_tick)
        self.stop_on_limit = False

    def _get_target_amk(self) -> int:
        return FurnaceUtil.unity_to_amk_pan(round(self.target_val))

    def _limit_target_val(self, target_val: float) -> float:
        return max(0, min(target_val, 0xFF))

    def _get_command(self, tick: int, duration: int, target_pan: int) -> Slide:
        return PanSlide(tick, duration, target_pan)

class VolumeSlider(SlideHelper):
    def _get_target_amk(self) -> int:
        return MMLUtil.find_v(round(self.target_val))

    def _limit_target_val(self, target_val: float) -> float:
        # max in Furnace is 7F, stored in binary as val * 2
        return max(0, min(target_val, 0xFE))

    def _get_command(self, tick: int, duration: int, target_volume: int) -> Slide:
        return VolumeSlide(tick, duration, target_volume)