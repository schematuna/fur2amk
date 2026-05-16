from typing import Optional, Tuple

from ..model.ChiptuneCommands import *


class FurnaceSlider:
    """Tracks slide state, emitting Chiptune slide commands."""

    def __init__(self):
        self.change_per_tick: float = 0
        self.target_val: float = 0xFE  # Furnace default volume is 0x7F (doubled to 0xFE)
        self.slide_start: int = None
        self.is_sliding: bool = False
        self.cur_tick: int = 0
        # whether this slide should stop when limit is reached
        self.stop_on_limit: bool = True

    def limit_target_val(self, target_val: float) -> float:
        return target_val

    def get_command(self, duration: int, target: int) -> ChiptuneCommand:
        return None

    def set_target(self, target: float) -> None:
        self.target_val = target

    def end_slide(self) -> Optional[Tuple[int, VolumeFadeCommand]]:
        """Ends the current slide. Returns (slide_start_tick, command) for retroactive placement."""
        if not self.is_sliding:
            return None
        duration = self.cur_tick - self.slide_start
        start = self.slide_start
        self.is_sliding = False
        if duration == 0:
            return None
        return start, self.get_command(duration, round(self.target_val))

    def start_slide(self) -> None:
        self.slide_start = self.cur_tick
        self.is_sliding = True

    def handle_new_effect(self, effect) -> Optional[Tuple[int, VolumeFadeCommand]]:
        """Handles a volume slide effect. A 0-value effect (change_per_tick is None) ends the slide.
        A non-zero effect ends the current slide and starts a new one.
        Returns (slide_start_tick, command) if a slide was completed, else None."""
        if effect.change_per_tick is None:
            return self.end_slide()
        completed = self.end_slide()
        self.change_per_tick = effect.change_per_tick
        if not self.is_sliding:
            self.start_slide()
        return completed

    def tick(self) -> Optional[Tuple[int, VolumeFadeCommand]]:
        result = None
        if self.is_sliding:
            self.target_val += self.change_per_tick
            bounded = self.limit_target_val(self.target_val)
            if bounded != self.target_val and self.stop_on_limit:
                self.target_val = bounded
                result = self.end_slide()
            else:
                self.target_val = bounded
        self.cur_tick += 1
        return result

class FurnaceVolumeSlider(FurnaceSlider):
    def __init__(self):
        super().__init__()

    def limit_target_val(self, target_val: float) -> float:
        return max(0, min(target_val, 0xFE))

    def get_command(self, duration: int, target: int) -> VolumeFadeCommand:
        return VolumeFadeCommand(duration, round(target))

class FurnacePanSlider(FurnaceSlider):
    def __init__(self):
        super().__init__()
        self.stop_on_limit = False

    def limit_target_val(self, target_val: float) -> float:
        return max(0, min(target_val, 0xFF))

    def get_command(self, duration: int, target: int) -> PanFadeCommand:
        return PanFadeCommand(duration, round(target))