from typing import Optional, Tuple

from ..model.ChiptuneCommands import VolumeFadeCommand


class FurnaceVolumeSlider:
    """Tracks volume slide state in furnace units, emitting VolumeFadeCommands."""

    def __init__(self):
        self.change_per_tick: float = 0
        self.target_val: float = 0xFE  # Furnace default volume is 0x7F (doubled to 0xFE)
        self.slide_start: int = None
        self.is_sliding: bool = False
        self.cur_tick: int = 0

    def set_target(self, target: float) -> None:
        self.target_val = target

    def set_volume(self, vol: float) -> Optional[Tuple[int, VolumeFadeCommand]]:
        """Called when an explicit volume is set on a row. Ends any in-progress slide,
        resets target_val, and restarts the slide from the new volume."""
        completed = self.end_slide() if self.is_sliding else None
        self.target_val = vol
        if completed is not None:
            self._start_slide()
        return completed

    def end_slide(self) -> Optional[Tuple[int, VolumeFadeCommand]]:
        """Ends the current slide. Returns (slide_start_tick, command) for retroactive placement."""
        if not self.is_sliding:
            return None
        duration = self.cur_tick - self.slide_start
        start = self.slide_start
        self.is_sliding = False
        if duration == 0:
            return None
        return start, VolumeFadeCommand(duration, round(self.target_val))

    def _start_slide(self) -> None:
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
            self._start_slide()
        return completed

    def tick(self) -> Optional[Tuple[int, VolumeFadeCommand]]:
        result = None
        if self.is_sliding:
            self.target_val += self.change_per_tick
            bounded = max(0, min(self.target_val, 0xFE))
            if bounded != self.target_val:
                self.target_val = bounded
                result = self.end_slide()
            else:
                self.target_val = bounded
        self.cur_tick += 1
        return result
