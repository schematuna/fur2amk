from typing import Optional

from .MMLUtil import MMLUtil

from .model.FurnaceEffects import *
from .model.MMLCommands import *


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
    def __init__(self, tick_ratio: int, starting_tick: int) -> None:
        # ratio of amk ticks to furnace ticks
        self.tick_ratio: int = tick_ratio
        # change per amk tick in furnace units
        self.change_per_tick: Optional[int] = 0
        # target value in furnace units
        self.target_val: int = 0

        # starting tick in amk ticks
        self.cur_tick: int = starting_tick
        # slide start in amk ticks
        self.slide_start: Optional[int] = None

        # whether this slide should stop when limit is reached
        self.stop_on_limit: bool = False

    # return the target value in amk units
    def _get_target_amk(self) -> int:
        return self.target_val

    def _limit_target_val(self, target_val: int) -> int:
        return target_val

    def _get_command(self, tick: int) -> MMLCommand:
        return None

    def _is_target_relative(self) -> bool:
        return False

    def _is_at_limit(self) -> bool:
        """Returns True if target_val has reached a limit and cannot continue."""
        return False

    def _complete_slide(self, duration: int) -> Optional[MMLCommand]:
        new_command = self._get_command(self.slide_start, duration, self._get_target_amk())
        if self._is_target_relative():
            self.target_val = 0
        return new_command

    def _stop_sliding(self) -> None:
        self.change_per_tick = None
        self.slide_start = None

    @staticmethod
    def get_max_duration() -> int:
        return max(MMLUtil.TICK_TO_DURATION.keys())

    def handle_new_effect(self, effect: FurnaceEffect) -> Optional[MMLCommand]:
        # this could be another slide or a stop slide command. Either way, we wrap up any current slide
        new_command = None
        if self.slide_start is not None:
            new_command = self._complete_slide(self.cur_tick - self.slide_start)

        if effect.change_per_tick is not None:
            # effect is in furnace units per furnace tick, convert to furnace units per amk tick
            self.change_per_tick = effect.change_per_tick / self.tick_ratio
            self.slide_start = self.cur_tick
        else:
            self._stop_sliding()
        
        return new_command

    # increase slide length, provide number of amk ticks to increment by
    def tick(self, ticks: int) -> Optional[MMLCommand]:
        new_command = None
        if self.slide_start is not None:
            LONGEST_DURATION = self.get_max_duration()
            cur_duration = self.cur_tick - self.slide_start
            if cur_duration >= LONGEST_DURATION:
                new_command = self._complete_slide(LONGEST_DURATION)
                self.slide_start = self.cur_tick

            self.target_val += self.change_per_tick * ticks
            self.target_val = self._limit_target_val(self.target_val)

            # Check if we've reached a limit and should stop
            # Furnace automatically stops most slides when they reach a limit
            if self.stop_on_limit and self._is_at_limit():
                # Emit final command for the slide that reached the limit
                if self.slide_start is not None:
                    new_command = self._complete_slide(self.cur_tick - self.slide_start)

                # Stop the slide
                self._stop_sliding()

        self.cur_tick += ticks

        return new_command

    def set_target(self, target: int) -> None:
        self.target_val = target

class PitchSlider(SlideHelper):

    def __init__(self, tick_ratio: int, starting_tick: int) -> None:
        super().__init__(tick_ratio, starting_tick)
        self.active_note: Optional[int] = None
        self.stop_on_limit = True

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
        if self.active_note is None:
            raise ValueError("Active note not set for PitchSlider")
        target_note = self.active_note + semitones
        target_note = max(MMLUtil.AMK_MIN_PITCH, min(target_note, MMLUtil.AMK_MAX_PITCH))
        return target_note

    def _get_command(self, tick: int, duration: int, target_note: int) -> MMLCommand:
        return PitchBend(tick, duration, target_note)

    def _is_target_relative(self) -> bool:
        return True

    def _is_at_limit(self) -> bool:
        """Check if pitch slide has reached min or max pitch limit."""
        if self.active_note is None:
            return False
        semitones = FurnaceUtil.fur_pitch_change_to_semitones(self.target_val)
        target_note = self.active_note + semitones
        # We're at limit if clamping would occur
        return target_note <= MMLUtil.AMK_MIN_PITCH or target_note >= MMLUtil.AMK_MAX_PITCH


class PanSlider(SlideHelper):
    def _get_target_amk(self) -> int:
        return FurnaceUtil.unity_to_amk_pan(round(self.target_val))

    def _limit_target_val(self, target_val: int) -> int:
        return max(0, min(target_val, 0xFF))

    def _get_command(self, tick: int, duration: int, target_pan: int) -> MMLCommand:
        return PanFade(tick, duration, target_pan)

class VolumeSlider(SlideHelper):
    def __init__(self, tick_ratio: int, starting_tick: int) -> None:
        super().__init__(tick_ratio, starting_tick)
        self.stop_on_limit = True

    def _get_target_amk(self) -> int:
        return MMLUtil.find_v(round(self.target_val))

    def _limit_target_val(self, target_val: int) -> int:
        return max(0, min(target_val, 0x7F))

    def _get_command(self, tick: int, duration: int, target_volume: int) -> MMLCommand:
        return VolumeFade(tick, duration, target_volume)