from typing import Optional, List
from dataclasses import dataclass, field

from ..util.MMLUtil import MMLUtil

from ..model.FurnaceEffects import *
from ..model.MMLCommands import *
from ..model.MMLData import *
from ..model.FurnaceData import *
from ..model.ChiptuneData import *

import copy

import logging

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

# Create this before iterating through rows, and call tick() for each row
# call handle_new_command whenever a relevant slide command is encountered
# call set_target to manually set the target value
class SlideHelper:
    def __init__(self, tick_ratio: int, starting_tick: int) -> None:
        # ratio of amk ticks to furnace ticks
        self.tick_ratio: float = tick_ratio
        # change per amk tick in furnace units
        self.change_per_tick: Optional[float] = 0
        # target value in furnace units
        self.target_val: float = 0

        # starting tick in amk ticks
        self.cur_tick: int = starting_tick
        # slide start in amk ticks
        self.slide_start: Optional[int] = None
        # are we currently in a slide
        self.is_sliding: bool = False

        # whether this slide should stop when limit is reached
        self.stop_on_limit: bool = True

    # return the target value in amk units
    def _get_target_amk(self) -> int:
        return self.target_val

    def _limit_target_val(self, target_val: int) -> int:
        return target_val

    def _get_command(self, tick: int) -> MMLCommand:
        return None

    def _get_change_per_tick(self, effect: FurnaceEffect) -> float:
        if effect.change_per_tick is not None:
            # effect is in furnace units per furnace tick, convert to furnace units per amk tick
            return effect.change_per_tick / self.tick_ratio
        else:
            return None

    @staticmethod
    def get_max_duration() -> int:
        return max(MMLUtil.TICK_TO_DURATION.keys())

    #########################################################

    def set_target(self, target: int) -> None:
        self.target_val = target

    def end_slide(self, duration: int = None) -> Optional[MMLCommand]:
        if not self.is_sliding:
            return None
        if duration is None:
            duration = self.cur_tick - self.slide_start

        new_command = None
        if duration != 0:
            new_command = self._get_command(self.slide_start, duration, self._get_target_amk())
        else:
            logging.info(f"Ignoring slide command with duration 0, target {self._get_target_amk()}")

        self.is_sliding = False
        return new_command

    def start_slide(self) -> None:
        self.slide_start = self.cur_tick
        self.is_sliding = True

    def handle_new_effect(self, effect: FurnaceEffect) -> Optional[MMLCommand]:
        # this could be another slide or a stop slide command. Either way, we wrap up any current slide
        new_command = None
        new_command = self.end_slide(None)

        if change_per_tick := self._get_change_per_tick(effect):
            self.change_per_tick = change_per_tick
            self.start_slide()
        
        return new_command

    # increase slide length, provide number of amk ticks to increment by
    def tick(self, ticks: int) -> Optional[MMLCommand]:
        new_command = None
        if self.is_sliding:
            LONGEST_DURATION = self.get_max_duration()
            cur_duration = self.cur_tick - self.slide_start
            if cur_duration >= LONGEST_DURATION:
                new_command = self.end_slide(LONGEST_DURATION)
                self.start_slide()

            self.target_val += self.change_per_tick * ticks
            bounded_target = self._limit_target_val(self.target_val)
            target_was_limited = bounded_target != self.target_val
            self.target_val = bounded_target

            # Check if we've reached a limit and should stop
            # Furnace automatically stops most slides when they reach a limit
            if self.stop_on_limit and target_was_limited:
                new_command = self.end_slide(None)

        self.cur_tick += ticks

        return new_command

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

    def _get_command(self, tick: int, duration: int, target_note: float) -> MMLCommand:
        return TempPitchBend(tick, duration, target_note)


class PanSlider(SlideHelper):
    def __init__(self, tick_ratio: int, starting_tick: int) -> None:
        super().__init__(tick_ratio, starting_tick)
        self.stop_on_limit = False

    def _get_target_amk(self) -> int:
        return FurnaceUtil.unity_to_amk_pan(round(self.target_val))

    def _limit_target_val(self, target_val: float) -> float:
        return max(0, min(target_val, 0xFF))

    def _get_command(self, tick: int, duration: int, target_pan: int) -> MMLCommand:
        return PanFade(tick, duration, target_pan)

class VolumeSlider(SlideHelper):
    def _get_target_amk(self) -> int:
        return MMLUtil.find_v(round(self.target_val))

    def _limit_target_val(self, target_val: float) -> float:
        # max in Furnace is 7F, stored in binary as val * 2
        return max(0, min(target_val, 0xFE))

    def _get_command(self, tick: int, duration: int, target_volume: int) -> MMLCommand:
        return VolumeFade(tick, duration, target_volume)