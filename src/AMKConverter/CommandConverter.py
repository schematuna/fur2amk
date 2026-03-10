from typing import List, Optional
from dataclasses import dataclass
import logging

from ..model.FurnaceData import *
from ..model.MMLCommands import *
from ..model.MMLData import *
from ..model.ChiptuneData import *

from .ConverterUtil import *

class VolumeConverter():
    def __init__(self, tick_ratio: float) -> None:
        self.tick_ratio = tick_ratio
        self.volume_slider = VolumeSlider(tick_ratio, 0)

    def convert_tick(self, tick_data: TickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        new_vol = tick_data.Vol
        slide_command = None
        if new_vol is not None:
            slide_command = self.volume_slider.end_slide()
            if slide_command is not None:
                commands.append(slide_command)
            self.volume_slider.set_target(new_vol)
            commands.append(VolumeChange(tick, MMLUtil.find_v(new_vol)))

        if effect := tick_data.get_effect(VolumeSlideEffect) or tick_data.get_effect(FineVolumeSlideEffect):
            if new_command := self.volume_slider.handle_new_effect(effect):
                commands.append(new_command)
        elif slide_command is not None:
            # restart slide if it was interrupted by a volume command
            self.volume_slider.start_slide()
            
        if new_command := self.volume_slider.tick(1):
            commands.append(new_command)

        return commands

class PanConverter():
    def __init__(self, tick_ratio: float) -> None:
        self.tick_ratio = tick_ratio
        self.pan_slider = PanSlider(tick_ratio, 0)

    def convert_tick(self, tick_data: TickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := tick_data.get_effect(PanEffect):
            self.pan_slider.set_target(effect.pan_position)
            amk_pan = FurnaceUtil.unity_to_amk_pan(effect.pan_position)
            commands.append(PanChange(tick, amk_pan))
        
        if effect := tick_data.get_effect(StereoPanEffect):
            cur_pan = FurnaceUtil.stereo_to_unity_pan(effect.left_volume, effect.right_volume)
            self.pan_slider.set_target(cur_pan)
            amk_pan = FurnaceUtil.stereo_to_amk_pan(effect.left_volume, effect.right_volume)
            commands.append(PanChange(tick, amk_pan))
        
        if effect := tick_data.get_effect(PanSlideEffect):
            if new_command := self.pan_slider.handle_new_effect(effect):
                commands.append(new_command)
                    
        if new_command := self.pan_slider.tick(1):
            commands.append(new_command)

        return commands

class VibratoConverter():
    def __init__(self, tick_ratio: float) -> None:
        self.tick_ratio = tick_ratio
        
    def convert_tick(self, tick_data: TickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := tick_data.get_effect(VibratoEffect):
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
    
class TempoConverter():
    def __init__(self, structure: ChiptuneStructure, amk_ticks_per_row: int) -> None:
        self.structure = structure
        self.amk_ticks_per_row = amk_ticks_per_row

    def convert_tick(self, tick_data: TickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := tick_data.get_effect(SetTickRateEffect):
            commands.append(TempoChange(tick, FurnaceUtil.tick_rate_to_amk_tempo(self.structure, self.amk_ticks_per_row, effect.tick_rate)))

        return commands
    
class TuningConverter():
    def convert_tick(self, tick_data: TickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := tick_data.get_effect(SetPitchEffect):
            # get into AMK range, 0->FF
            pitch_change = effect.pitch * 2
            semitone_tune = 0
            fine_tune = 0
            if pitch_change < 0:
                # compress to -255 -> 0
                pitch_change = round(pitch_change * 255/256)
                semitone_tune = -1
                fine_tune = (0xFF + pitch_change)
            else:
                # expand to 255 -> 0
                pitch_change = round(pitch_change * 255/254)
                fine_tune = pitch_change

            if state.fine_tune != fine_tune:
                commands.append(FineTune(tick, fine_tune))
                state.fine_tune = fine_tune

            if state.semitone_tune != semitone_tune:
                commands.append(SemitoneTune(tick, semitone_tune))
                state.semitone_tune = semitone_tune

        return commands

@dataclass
class LegatoRegion:
    """A region where legato should be active."""
    start_tick: int
    end_tick: int = None  # None means open-ended (until end of song or next event)


class LegatoConverter:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def convert(self, ticks: List[TickData], notes: List[MMLNote]) -> List[MMLCommand]:
        # Build legato regions
        regions = self.build_legato_regions(ticks)

        # Emit toggle commands
        commands = self.emit_toggle_commands(regions, notes)

        return commands

    def build_legato_regions(self, ticks: List[TickData]) -> List[LegatoRegion]:
        """
        Build a list of tick ranges where legato should be active.
        """
        regions: List[LegatoRegion] = []
        current_region: LegatoRegion = None
        legato_on = False

        tick = 0
        for tick_data in ticks:
            if legato_effect := tick_data.get_effect(LegatoEffect):
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

            tick += 1

        # Close any open region at end of song
        if current_region:
            current_region.end_tick = tick - 1
            regions.append(current_region)

        return regions

    def emit_toggle_commands(self, regions: List[LegatoRegion], notes: List[MMLNote]) -> List[MMLCommand]:
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
            # at note boundaries, defer to the earlier note
            if note.tick < tick <= note.tick + note.duration:
                return note
        return None

    def _get_note_starting_at(self, tick: int, notes: List[MMLNote]) -> Optional[MMLNote]:
        """Find the note that starts at the given tick."""
        for note in notes:
            if note.tick == tick:
                return note
        return None