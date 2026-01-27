from typing import List, Optional
from dataclasses import dataclass
import logging

from ..model.FurnaceData import *
from ..model.MMLCommands import *
from ..model.MMLData import *

from .ConverterUtil import *

class VolumeConverter():
    def __init__(self, tick_ratio: float, amk_ticks_per_row: int) -> None:
        self.amk_ticks_per_row = amk_ticks_per_row
        self.volume_slider = VolumeSlider(tick_ratio, 0)

    def convert_row(self, row: FurnaceRow, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        new_vol = row.Vol
        slide_command = None
        if new_vol is not None:
            slide_command = self.volume_slider.end_slide()
            if slide_command is not None:
                commands.append(slide_command)
            self.volume_slider.set_target(new_vol)
            commands.append(VolumeChange(tick, MMLUtil.find_v(new_vol)))

        if effect := row.get_effect(VolumeSlideEffect) or row.get_effect(FineVolumeSlideEffect):
            if new_command := self.volume_slider.handle_new_effect(effect):
                commands.append(new_command)
        elif slide_command is not None:
            # restart slide if it was interrupted by a volume command
            self.volume_slider.start_slide()
            
        if new_command := self.volume_slider.tick(self.amk_ticks_per_row):
            commands.append(new_command)

        return commands

class PanConverter():
    def __init__(self, tick_ratio: float, amk_ticks_per_row: int) -> None:
        self.amk_ticks_per_row = amk_ticks_per_row
        self.pan_slider = PanSlider(tick_ratio, 0)

    def convert_row(self, row: FurnaceRow, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := row.get_effect(PanEffect):
            self.pan_slider.set_target(effect.pan_position)
            amk_pan = FurnaceUtil.unity_to_amk_pan(effect.pan_position)
            commands.append(PanChange(tick, amk_pan))
        
        if effect := row.get_effect(StereoPanEffect):
            cur_pan = FurnaceUtil.stereo_to_unity_pan(effect.left_volume, effect.right_volume)
            self.pan_slider.set_target(cur_pan)
            amk_pan = FurnaceUtil.stereo_to_amk_pan(effect.left_volume, effect.right_volume)
            commands.append(PanChange(tick, amk_pan))
        
        if effect := row.get_effect(PanSlideEffect):
            if new_command := self.pan_slider.handle_new_effect(effect):
                commands.append(new_command)
                    
        if new_command := self.pan_slider.tick(self.amk_ticks_per_row):
            commands.append(new_command)

        return commands

class VibratoConverter():
    def convert_row(self, row: FurnaceRow, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := row.get_effect(VibratoEffect):
            # Vibrato off when both speed and depth are 0
            if effect.speed == 0 and effect.depth == 0:
                commands.append(DisableVibrato(tick))
            else:
                if effect.speed > 0:
                    # Furnace speed (vibratoRate) controls how many positions in the 64-entry sine table
                    # to advance per tick. One complete cycle = 64 positions.
                    amk_ticks_per_cycle = (64 * self.tick_ratio) / effect.speed
                    # 256 scalar seems to make it sounds closer to Furnace vibrato rates
                    amk_speed = 256 / amk_ticks_per_cycle
                    # Clamp to valid range (1-255)
                    speed = max(1, min(255, int(round(amk_speed))))
                else:
                    speed = 0

                # Furnace depth (0-15) represents vibrato depth where 15 = ±1 semitone
                # Map linearly: 15 in Furnace -> 0xC0 in AMK, which sounds about right
                amplitude = (effect.depth * 0xC0) // 15

                commands.append(Vibrato(tick, speed, amplitude))

        return commands


@dataclass
class LegatoRegion:
    """A region where legato should be active."""
    start_tick: int
    end_tick: int = None  # None means open-ended (until end of song or next event)


class LegatoConverter:
    def __init__(self, amk_ticks_per_row: int) -> None:
        self.amk_ticks_per_row = amk_ticks_per_row
        self.logger = logging.getLogger(__name__)

    def convert(self, flat_rows: List[FurnaceRow], notes: List[MMLNote]) -> List[MMLCommand]:
        # Pass 1: Build legato regions
        regions = self._build_legato_regions(flat_rows, notes)

        # Pass 2: Emit toggle commands
        commands = self._emit_toggle_commands(regions, notes)

        return commands

    def _build_legato_regions(self, flat_rows: List[FurnaceRow], notes: List[MMLNote]) -> List[LegatoRegion]:
        """
        Build a list of tick ranges where legato should be active.

        Builds global and quick legato regions separately, then merges them.
        """
        global_regions = self._build_global_legato_regions(flat_rows)
        quick_regions = self._build_quick_legato_regions(flat_rows, notes)

        # Combine and merge overlapping regions
        all_regions = sorted(global_regions + quick_regions, key=lambda r: r.start_tick)
        return self._merge_adjacent_regions(all_regions)

    def _build_global_legato_regions(self, flat_rows: List[FurnaceRow]) -> List[LegatoRegion]:
        """Build regions from LegatoEffect (simple on/off that persists)."""
        regions: List[LegatoRegion] = []
        current_region: LegatoRegion = None
        legato_on = False

        tick = 0
        for row in flat_rows:
            if legato_effect := row.get_effect(LegatoEffect):
                if legato_effect.legato_on and not legato_on:
                    # Legato turning ON
                    legato_on = True
                    current_region = LegatoRegion(start_tick=tick)
                elif not legato_effect.legato_on and legato_on:
                    # Legato turning OFF
                    legato_on = False
                    current_region.end_tick = tick
                    regions.append(current_region)
                    current_region = None

            tick += self.amk_ticks_per_row

        # Close any open region at end of song
        if current_region:
            current_region.end_tick = tick
            regions.append(current_region)

        return regions

    def _build_quick_legato_regions(self, flat_rows: List[FurnaceRow], notes: List[MMLNote]) -> List[LegatoRegion]:
        """
        Build regions from QuickLegatoEffect.

        Quick legato starts at the effect and ends at the start of the destination note
        (the first note after the quick legato chain that doesn't have a quick legato effect).
        """
        regions: List[LegatoRegion] = []
        current_region: LegatoRegion = None

        tick = 0
        for row in flat_rows:
            row_end = tick + self.amk_ticks_per_row
            quick_legato_effect = row.get_effect(QuickLegatoEffect)
            note_in_row = self._get_note_starting_in_range(tick, row_end, notes)

            if quick_legato_effect:
                # Start a new region if not already in one
                if current_region is None:
                    current_region = LegatoRegion(start_tick=tick)
            elif current_region is not None and note_in_row:
                # No quick legato on this row, but we're in a region and a note starts here
                # This note is the destination - end the region at its start
                current_region.end_tick = note_in_row.tick
                regions.append(current_region)
                current_region = None

            tick += self.amk_ticks_per_row

        # Close any open region at end of song
        if current_region:
            current_region.end_tick = tick
            regions.append(current_region)

        return regions

    def _merge_adjacent_regions(self, regions: List[LegatoRegion]) -> List[LegatoRegion]:
        """Merge regions that are adjacent or overlapping."""
        if not regions:
            return []

        merged = [regions[0]]
        for region in regions[1:]:
            last = merged[-1]
            if region.start_tick <= last.end_tick:
                # Overlapping or adjacent, extend the last region
                last.end_tick = max(last.end_tick, region.end_tick)
            else:
                merged.append(region)

        return merged

    def _emit_toggle_commands(self, regions: List[LegatoRegion], notes: List[MMLNote]) -> List[MMLCommand]:
        """
        Emit toggle commands for each region.

        ON toggle: added to pre_note_commands of the note active at region start.
        OFF toggle: placed one tick before the end of the note active at region end.
        """
        commands: List[MMLCommand] = []

        for region in regions:
            # Find note active at region start and add ON toggle
            start_note = self._get_note_active_at(region.start_tick, notes)
            if start_note:
                start_note.pre_note_commands.append(LegatoToggle(start_note.tick))
            else:
                # No note active, emit as standalone command
                commands.append(LegatoToggle(region.start_tick))

            # Find note that starts at region end and add OFF toggle to its pre_note_commands
            # AMK docs say we have to turn legato off in the middle of the previous note, but that
            # doesn't seem to be necessary.
            if region.end_tick is not None:
                end_note = self._get_note_starting_at(region.end_tick, notes)
                if end_note:
                    end_note.pre_note_commands.append(LegatoToggle(end_note.tick))
                else:
                    # No note starts at end_tick, emit as standalone command
                    commands.append(LegatoToggle(region.end_tick))

        return commands

    def _get_note_active_at(self, tick: int, notes: List[MMLNote]) -> Optional[MMLNote]:
        """Find the note that is active (playing) at the given tick."""
        for note in notes:
            if note.duration is None:
                continue
            if note.tick <= tick < note.tick + note.duration:
                return note
        return None

    def _get_note_starting_in_range(self, start_tick: int, end_tick: int, notes: List[MMLNote]) -> Optional[MMLNote]:
        """Find the first note that starts within the given tick range [start_tick, end_tick)."""
        for note in notes:
            if start_tick <= note.tick < end_tick:
                return note
        return None

    def _get_note_starting_at(self, tick: int, notes: List[MMLNote]) -> Optional[MMLNote]:
        """Find the note that starts at the given tick."""
        for note in notes:
            if note.tick == tick:
                return note
        return None