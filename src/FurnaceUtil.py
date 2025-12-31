from typing import Optional
from enum import Enum
import sys

from .model.MMLCommands import *
from .MMLUtil import MMLUtil
from .model.FurnaceData import FurnaceCommandType

class FurnaceUtil:
    PITCH_STEPS_PER_OCTAVE = 384

    @staticmethod
    def fur_pitch_change_to_semitones(change: int) -> int:
        semitones = round(change * 12 / FurnaceUtil.PITCH_STEPS_PER_OCTAVE)
        return int(semitones)

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


# Create this before iterating through rows, and call tick() for each row
# call handle_new_command whenever a relevant slide command is encountered
# call set_target to manually set the target value
class SlideHelper:
    def __init__(self, starting_tick: int) -> None:
        self.cur_tick: int = starting_tick
        self.change_per_tick: Optional[int] = 0
        self.slide_start: Optional[int] = None
        self.target_val: int = 0

        self.active_note: Optional[int] = None

    def _get_target_amk(self) -> int:
        return self.target_val

    def _limit_target_val(self, target_val: int) -> int:
        return target_val

    @staticmethod
    def _get_change_per_tick(effect_num: int, value: int) -> int:
        return None

    def _get_command(self, tick: int) -> MMLCommand:
        return None

    def _is_target_relative(self) -> bool:
        return False

    @staticmethod
    def get_max_duration() -> int:
        return max(MMLUtil.TICK_TO_DURATION.keys())

    def handle_new_command(self, effect_num: int, value: int) -> Optional[MMLCommand]:
        # this could be another slide or a stop slide command. Either way, we wrap up any current slide
        new_command = None
        if self.slide_start is not None:
            new_command = self._get_command(self.slide_start, 
                                            self.cur_tick - self.slide_start, 
                                            self._get_target_amk())
            if self._is_target_relative():
                self.target_val = 0

        self.change_per_tick = self._get_change_per_tick(effect_num, value)

        if self.change_per_tick is not None:
            self.slide_start = self.cur_tick
        else:
            self.slide_start = None
        
        return new_command

    # increase slide length, provide number of amk ticks to increment by
    def tick(self, ticks: int) -> None:
        new_command = None
        if self.slide_start is not None:
            LONGEST_DURATION = SlideHelper.get_max_duration()
            cur_duration = self.cur_tick - self.slide_start
            if cur_duration >= LONGEST_DURATION:
                new_command = self._get_command(self.slide_start, LONGEST_DURATION, self._get_target_amk())
                self.slide_start = self.cur_tick
                if self._is_target_relative():
                    self.target_val = 0

            self.target_val += self.change_per_tick * ticks
            self.target_val = self._limit_target_val(self.target_val)

        self.cur_tick += ticks

        return new_command

    def set_target(self, target: int) -> None:
        self.target_val = target

class PitchSlider(SlideHelper):
    # Set the currently active note
    # Needed to figure out what note to slide to
    def set_active_note(self, note: int) -> None:
        self.active_note = note

    # pitchbend can't operate on a whole note, since 1 = 2^2 under the hood
    @staticmethod
    def get_max_duration() -> int:
        return int(SlideHelper.get_max_duration() / 2)

    def _get_target_amk(self) -> int:
        semitones = FurnaceUtil.fur_pitch_change_to_semitones(self.target_val)
        target_note = self.active_note + semitones
        target_note = max(0, min(target_note, MMLUtil.AMK_MAX_PITCH))
        return target_note

    @staticmethod
    def _get_change_per_tick(effect_num: int, value: int) -> int:
        if value == 0:
            return None
        else:
            if effect_num == FurnaceCommandType.PITCH_SLIDE_UP.value:
                return value
            elif effect_num == FurnaceCommandType.PITCH_SLIDE_DOWN.value:
                return -value
            else:
                print(f"Warning: Invalid pitch slide effect number {effect_num}.", file=sys.stderr)
                return None

    def _get_command(self, tick: int, duration: int, target_note: int) -> MMLCommand:
        return PitchBend(tick, duration, target_note)

    def _is_target_relative(self) -> bool:
        return True
        

class PanSlider(SlideHelper):
    def _get_target_amk(self) -> int:
        return FurnaceUtil.unity_to_amk_pan(round(self.target_val))

    def _limit_target_val(self, target_val: int) -> int:
        return max(0, min(target_val, 0xFF))

    @staticmethod
    def _get_change_per_tick(effect_num: int, value: int) -> int:
        left = value >> 4
        right = value & 0x0F
        if right == 0 and left == 0:
            return None
        elif right == 0:
            # halved because pan is spread across both channels in Furnace
            return -left / 2
        elif left == 0:
            return right / 2
        else:
            print(f"Warning: Invalid pan slide effect value {value}.", file=sys.stderr)
            return None

    def _get_command(self, tick: int, duration: int, target_pan: int) -> MMLCommand:
        return PanFade(tick, duration, target_pan)

class VolumeSlider(SlideHelper):
    def _get_target_amk(self) -> int:
        return MMLUtil.find_v(round(self.target_val))

    def _limit_target_val(self, target_val: int) -> int:
        return max(0, min(target_val, 0x7F))

    @staticmethod
    def _get_change_per_tick(effect_num: int, value: int) -> Optional[int]:
        vol_change_per_tick = None
        if effect_num == FurnaceCommandType.VOLUME_SLIDE.value or effect_num == FurnaceCommandType.FAST_VOLUME_SLIDE.value:
            rate_divisor = 4
            if effect_num == FurnaceCommandType.FAST_VOLUME_SLIDE.value:
                # fast volume slides are 4 times faster than normal volume slides
                rate_divisor = 1

            up = value >> 4
            down = value & 0x0F
            if down == 0 and up == 0:
                vol_change_per_tick = None
            elif down == 0:
                vol_change_per_tick = up / rate_divisor
            elif up == 0:
                vol_change_per_tick = -down / rate_divisor
            else:
                print("Warning: Invalid volume slide effect value.", file=sys.stderr)
        # fine volume slides are 64 times slower than normal volume slides
        elif effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_UP.value:
            if value == 0:
                vol_change_per_tick = None
            else:
                vol_change_per_tick = value / 64
        elif effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_DOWN.value:
            if value == 0:
                vol_change_per_tick = None
            else:
                vol_change_per_tick = -value / 64
        else:
            print(f"Warning: Invalid volume slide effect number {effect_num}.", file=sys.stderr)
        
        return vol_change_per_tick

    def _get_command(self, tick: int, duration: int, target_volume: int) -> MMLCommand:
        return VolumeFade(tick, duration, target_volume)