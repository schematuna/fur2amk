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

        Global legato (LegatoEffect): Simple on/off that persists.
        Quick legato (QuickLegatoEffect): Temporary, covers transition from current note to next.
            - Starts at the quick legato command tick
            - Ends when: another note starts (not via quick legato) or note ends,
              unless another quick legato extends it.
        """
        regions: List[LegatoRegion] = []

        # State tracking
        global_legato = False           # Persistent legato from LegatoEffect
        quick_legato_active = False     # Quick legato is active (waiting to resolve)
        current_region: LegatoRegion = None

        tick = 0
        for row in flat_rows:
            row_end = tick + self.amk_ticks_per_row

            # Check for effects
            legato_effect = row.get_effect(LegatoEffect)
            quick_legato_effect = row.get_effect(QuickLegatoEffect)

            # Find note that starts within this row (notes can start mid-row)
            note_in_row = self._get_note_starting_in_range(tick, row_end, notes)

            # Handle LegatoEffect (global on/off)
            if legato_effect:
                if legato_effect.legato_on and not global_legato:
                    # Global legato turning ON
                    global_legato = True
                    if current_region is None:
                        current_region = LegatoRegion(start_tick=tick)
                elif not legato_effect.legato_on and global_legato:
                    # Global legato turning OFF
                    global_legato = False
                    if current_region and not quick_legato_active:
                        # Close the region
                        current_region.end_tick = tick
                        regions.append(current_region)
                        current_region = None

            # Handle QuickLegatoEffect
            # A row with quick legato is guaranteed to have a note
            if quick_legato_effect:
                quick_legato_active = True
                if current_region is None:
                    current_region = LegatoRegion(start_tick=tick)

            # Check if quick legato should end
            elif note_in_row and quick_legato_active:
                # This note is the destination of the quick legato chain
                quick_legato_active = False

                # If global legato is off, end the region at this note's start
                if not global_legato and current_region:
                    current_region.end_tick = note_in_row.tick
                    regions.append(current_region)
                    current_region = None

            tick += self.amk_ticks_per_row

        # Close any open region at end of song
        if current_region:
            current_region.end_tick = tick
            regions.append(current_region)

        return self._merge_adjacent_regions(regions)

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

            # Find note active at region end and add OFF toggle
            if region.end_tick is not None:
                end_note = self._get_note_active_at(region.end_tick - 1, notes)
                if end_note:
                    # Place toggle one tick before the note ends
                    off_tick = end_note.tick + end_note.duration - 1 if end_note.duration else region.end_tick - 1
                    commands.append(LegatoToggle(off_tick))
                else:
                    # No note active, emit at region end - 1
                    off_tick = max(0, region.end_tick - 1)
                    commands.append(LegatoToggle(off_tick))

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