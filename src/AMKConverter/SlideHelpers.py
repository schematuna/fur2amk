from typing import Optional
from dataclasses import dataclass
import logging

from ..util.MMLUtil import MMLUtil


@dataclass
class Slide:
    tick: int = 0
    duration: int = 0

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

    def _get_target_amk(self) -> int:
        return self.target_val

    def _limit_target_val(self, target_val: int) -> int:
        return target_val

    def _get_command(self, tick: int, duration: int, target) -> Slide:
        return None

    def _get_change_per_tick(self, effect) -> float:
        if effect.change_per_tick is not None:
            # effect is in furnace units per furnace tick, convert to furnace units per amk tick
            return effect.change_per_tick / self.tick_ratio
        else:
            return None

    @staticmethod
    def get_max_duration() -> int:
        return max(MMLUtil.TICK_TO_DURATION.keys())

    def set_target(self, target: int) -> None:
        self.target_val = target

    def end_slide(self, duration: int = None) -> Optional[Slide]:
        if not self.is_sliding:
            return None
        if duration is None:
            duration = self.cur_tick - self.slide_start

        new_slide = None
        if duration != 0:
            new_slide = self._get_command(self.slide_start, duration, self._get_target_amk())
        else:
            logging.info(f"Ignoring slide command with duration 0, target {self._get_target_amk()}")

        self.is_sliding = False
        return new_slide

    def start_slide(self) -> None:
        self.slide_start = self.cur_tick
        self.is_sliding = True

    def handle_new_effect(self, effect) -> Optional[Slide]:
        # this could be another slide or a stop slide command. Either way, we wrap up any current slide
        new_slide = self.end_slide(None)

        if change_per_tick := self._get_change_per_tick(effect):
            self.change_per_tick = change_per_tick
            self.start_slide()

        return new_slide

    def tick(self, ticks: int) -> Optional[Slide]:
        new_slide = None
        if self.is_sliding:
            LONGEST_DURATION = self.get_max_duration()
            cur_duration = self.cur_tick - self.slide_start
            if cur_duration >= LONGEST_DURATION:
                new_slide = self.end_slide(LONGEST_DURATION)
                self.start_slide()

            self.target_val += self.change_per_tick * ticks
            bounded_target = self._limit_target_val(self.target_val)
            target_was_limited = bounded_target != self.target_val
            self.target_val = bounded_target

            if self.stop_on_limit and target_was_limited:
                new_slide = self.end_slide(None)

        self.cur_tick += ticks
        return new_slide
