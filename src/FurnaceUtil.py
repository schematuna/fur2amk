from typing import Optional
from enum import Enum
import sys
from .model.MMLCommands import MMLCommand
from .MMLUtil import MMLUtil
from .model.MMLCommands import *

class FurnaceCommandType(Enum):
    PITCH_SLIDE_UP = 0x01
    PITCH_SLIDE_DOWN = 0x02
    PORTAMENTO = 0x03
    STEREO_PAN = 0x08
    VOLUME_SLIDE = 0x0A
    PAN = 0x80
    PAN_SLIDE = 0x83
    NOTE_SLIDE_UP = 0xE1
    NOTE_SLIDE_DOWN = 0xE2
    QUICK_LEGATO = 0xE6 # basically another note within a row
    QUICK_LEGATO_UP = 0xE8
    QUICK_LEGATO_DOWN = 0xE9
    NOTE_DELAY = 0xED
    FINE_VOLUME_SLIDE_UP = 0xF3
    FINE_VOLUME_SLIDE_DOWN = 0xF4
    FAST_VOLUME_SLIDE = 0xFA

# Create this before iterating through rows, and call tick() for each row
# call handle_new_command whenever a relevant slide command is encountered
class SlideHelper:
    def __init__(self, starting_tick: int) -> None:
        self.cur_tick: int = starting_tick
        self.change_per_tick: Optional[int] = 0
        self.current_slide: Optional[MMLCommand] = None
        self.target_val: int = 0

        self.active_note: Optional[int] = None

    def handle_new_command(self, effect_num: int, value: int) -> Optional[MMLCommand]:
        return None

    # increase slide length, provide number of amk ticks to increment by
    def tick(self, ticks: int) -> None:
        pass

    def set_target(self, target: int) -> None:
        self.target_val = target

class PitchSlider(SlideHelper):
    def set_active_note(self, note: int) -> None:
        self.active_note = note

    def handle_new_command(self, effect_num: int, value: int) -> Optional[MMLCommand]:
        # this could be another slide or a stop slide command. Either way, we wrap up any current slide
        new_command = None
        if self.current_slide is not None:
            self.current_slide.duration = self.cur_tick - self.current_slide.tick
            semitones = MMLUtil.fur_pitch_change_to_semitones(self.target_val)
            target_note = self.active_note + semitones
            if target_note > 141:
                target_note = 141
            if target_note < 0:
                target_note = 0

            self.current_slide.note = target_note
            new_command = self.current_slide

        if value == 0:
            self.change_per_tick = None
        else:
            if effect_num == FurnaceCommandType.PITCH_SLIDE_UP.value:
                self.change_per_tick = value
            elif effect_num == FurnaceCommandType.PITCH_SLIDE_DOWN.value:
                self.change_per_tick = -value

        if self.change_per_tick is not None:
            self.current_slide = PitchBend(self.cur_tick, None, None)
            self.target_val = 0
        else:
            self.current_slide = None
        
        return new_command

    def tick(self, ticks: int) -> Optional[MMLCommand]:
        new_command = None
        if self.current_slide is not None:
            # pitchbend can't operate on a whole note, since 1 = 2^2 under the hood
            LONGEST_DURATION = int(max(MMLUtil.TICK_TO_DURATION.keys()) / 2)
            cur_duration = self.cur_tick - self.current_slide.tick
            if cur_duration >= LONGEST_DURATION:
                self.current_slide.duration = LONGEST_DURATION
                semitones = MMLUtil.fur_pitch_change_to_semitones(self.target_val)
                target_note = self.active_note + semitones
                target_note = max(0, min(target_note, 141))

                self.current_slide.note = target_note
                new_command = self.current_slide
                self.current_slide = PitchBend(self.cur_tick, None, None)
                self.target_val = 0

            # increment target pitch
            self.target_val += self.change_per_tick * ticks

        self.cur_tick += ticks

        return new_command

class PanSlider(SlideHelper):
    def handle_new_command(self, effect_num: int, value: int) -> Optional[MMLCommand]:
        # this could be another slide or a stop slide command. Either way, we wrap up any current slide
        new_command = None
        if self.current_slide is not None:
            self.current_slide.duration = self.cur_tick - self.current_slide.tick
            self.current_slide.target_pan = MMLUtil.fur_pan_to_amk(round(self.target_val))
            new_command = self.current_slide

        left = value >> 4
        right = value & 0x0F
        if right == 0 and left == 0:
            self.change_per_tick = None
        elif right == 0:
            # halved because pan is spread across both channels in Furnace
            self.change_per_tick = -left / 2
        elif left == 0:
            self.change_per_tick = right / 2
        else:
            print(f"Warning: Invalid pan slide effect value {value}.", file=sys.stderr)

        if self.change_per_tick is not None:
            self.current_slide = PanFade(self.cur_tick, None, None)
        else:
            self.current_slide = None

        return new_command

    def tick(self, ticks: int) -> Optional[MMLCommand]:
        new_command = None
        if self.current_slide is not None:
            # if slide is too long, split it into multiple slides
            cur_duration = self.cur_tick - self.current_slide.tick
            LONGEST_DURATION = max(MMLUtil.TICK_TO_DURATION.keys())
            if cur_duration >= LONGEST_DURATION:
                self.current_slide.duration = LONGEST_DURATION
                self.current_slide.target_pan = MMLUtil.fur_pan_to_amk(round(self.target_val))
                new_command = self.current_slide
                self.current_slide = PanFade(self.cur_tick, None, None)

            # increment target pan
            self.target_val += self.change_per_tick * ticks
            self.target_val = max(0, min(self.target_val, 0xFF))

        self.cur_tick += ticks

        return new_command

class VolumeSlider(SlideHelper):
    @staticmethod
    def get_volume_slide_change(effect_num: int, value: int) -> Optional[int]:
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
            vol_change_per_tick = value / 64
        elif effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_DOWN.value:
            vol_change_per_tick = -value / 64
        else:
            print(f"Warning: Invalid volume slide effect number {effect_num}.", file=sys.stderr)
        
        return vol_change_per_tick

    def handle_new_command(self, effect_num: int, value: int) -> Optional[MMLCommand]:
        # this could be another slide or a stop slide command. Either way, we wrap up any current slide
        new_command = None
        if self.current_slide is not None:
            self.current_slide.duration = self.cur_tick - self.current_slide.tick
            self.current_slide.target_volume = MMLUtil.find_v(round(self.target_val))
            new_command = self.current_slide

        self.change_per_tick = self.get_volume_slide_change(effect_num, value)

        if self.change_per_tick is not None:
            self.current_slide = VolumeFade(self.cur_tick, None, None)
        else:
            self.current_slide = None

        return new_command

    def tick(self, ticks: int) -> Optional[MMLCommand]:
        new_command = None
        if self.current_slide is not None:
            # if slide is too long, split it into multiple slides
            cur_duration = self.cur_tick - self.current_slide.tick
            LONGEST_DURATION = max(MMLUtil.TICK_TO_DURATION.keys())
            if cur_duration >= LONGEST_DURATION:
                self.current_slide.duration = LONGEST_DURATION
                self.current_slide.target_volume = MMLUtil.find_v(round(self.target_val))
                new_command = self.current_slide
                self.current_slide = VolumeFade(self.cur_tick, None, None)

            self.target_val += self.change_per_tick * ticks
            self.target_val = max(0, min(self.target_val, 0x7F))

        self.cur_tick += ticks

        return new_command