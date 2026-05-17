from typing import List, Optional
from dataclasses import dataclass
from math import floor
import logging

from ..model.FurnaceData import *
from ..model.MMLCommands import *
from ..model.MMLData import *
from ..model.ChiptuneData import *

from .ConverterUtil import *

import copy

class VolumeConverter():
    def __init__(self) -> None:
        self.cur_vol: float = 0xFE  # Furnace default volume (0x7F * 2)

    def convert_tick(self, tick_data: ChiptuneTickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []

        if tick_data.Vol is not None:
            self.cur_vol = tick_data.Vol
            commands.append(VolumeChange(tick, MMLUtil.find_v(self.cur_vol)))

        if cmd := tick_data.get_command(VolumeFadeCommand):
            amk_duration = cmd.duration
            fur_start = self.cur_vol
            fur_target = cmd.target
            max_duration = max(MMLUtil.TICK_TO_DURATION.keys())
            if amk_duration <= max_duration:
                commands.append(VolumeFade(tick, amk_duration, MMLUtil.find_v(fur_target)))
            else:
                # Split into multiple fades with linearly interpolated targets in Furnace units
                cur_tick = tick
                remaining = amk_duration
                elapsed = 0
                while remaining > 0:
                    chunk = min(remaining, max_duration)
                    elapsed += chunk
                    t = elapsed / amk_duration
                    interp_fur = fur_start + t * (fur_target - fur_start)
                    commands.append(VolumeFade(cur_tick, chunk, MMLUtil.find_v(round(interp_fur))))
                    cur_tick += chunk
                    remaining -= chunk
            self.cur_vol = fur_target
        return commands

class PanConverter():
    def __init__(self) -> None:
        self.cur_pan: float = 0x80  # center (Furnace default)

    def convert_tick(self, tick_data: ChiptuneTickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := tick_data.get_command(PanCommand):
            self.cur_pan = effect.pan_position
            amk_pan = FurnaceUtil.unity_to_amk_pan(effect.pan_position)
            commands.append(PanChange(tick, amk_pan))

        if cmd := tick_data.get_command(PanFadeCommand):
            amk_duration = cmd.duration
            fur_start = self.cur_pan
            fur_target = cmd.target
            max_duration = max(MMLUtil.TICK_TO_DURATION.keys())
            if amk_duration <= max_duration:
                commands.append(PanFade(tick, amk_duration, FurnaceUtil.unity_to_amk_pan(fur_target)))
            else:
                # Split into multiple fades with linearly interpolated targets in Furnace units
                cur_tick = tick
                remaining = amk_duration
                elapsed = 0
                while remaining > 0:
                    chunk = min(remaining, max_duration)
                    elapsed += chunk
                    t = elapsed / amk_duration
                    interp_fur = fur_start + t * (fur_target - fur_start)
                    commands.append(PanFade(cur_tick, chunk, FurnaceUtil.unity_to_amk_pan(round(interp_fur))))
                    cur_tick += chunk
                    remaining -= chunk
            self.cur_pan = fur_target

        return commands

class VibratoConverter():
    def __init__(self, tick_ratio: float) -> None:
        self.tick_ratio = tick_ratio
        
    def convert_tick(self, tick_data: ChiptuneTickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := tick_data.get_command(VibratoCommand):
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

    def convert_tick(self, tick_data: ChiptuneTickData, tick: int, state: FurnaceState) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        if effect := tick_data.get_command(SetTickRateCommand):
            commands.append(TempoChange(tick, FurnaceUtil.tick_rate_to_amk_tempo(self.structure, self.amk_ticks_per_row, effect.tick_rate)))

        return commands
    
class TuningConverter():
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def convert(self, ticks: List[ChiptuneTickData], notes: List[MMLNote]) -> Tuple[List[MMLCommand], List[ChiptuneTickData], List[MMLNote]]:
        tuning_commands = self.convert_finetune(ticks)
        legatofied_ticks = self.convert_legato(ticks, notes)
        # TODO: split notes won't account for pitchbends across the split.
        split_notes = self.convert_notes(ticks, notes)

        return tuning_commands, legatofied_ticks, split_notes
    
    def convert_finetune(self, ticks: List[ChiptuneTickData]) -> List[MMLCommand]:
        fine_tune_state: int = 0
        semitone_tune_state: int = 0
        commands: List[MMLCommand] = []
        for tick, tick_data in enumerate(ticks):
            if effect := tick_data.get_command(TuningCommand):
                # can only fine-tune upwards in AMK so shift to nearest semitone value and fine-tune up from there
                semitone_tune = floor(effect.tuning)
                # get tuning into AMK range, 0->FF
                fine_tune = int((effect.tuning - semitone_tune) * 0xFF)

                if fine_tune != fine_tune_state:
                    commands.append(FineTune(tick, fine_tune))
                    fine_tune_state = fine_tune

                if semitone_tune != semitone_tune_state:
                    commands.append(SemitoneTune(tick, semitone_tune))
                    semitone_tune_state = semitone_tune

        return commands

    def convert_legato(self, ticks: List[ChiptuneTickData], notes: List[MMLNote]) -> List[ChiptuneTickData]:
        global_legato_enabled: bool = False
        # tracks fine tune chains, which create localized legato regions
        in_fine_tune_chain: bool = False
        for tick, tick_data in enumerate(ticks):
            # track global legato changes
            if legato_effect := tick_data.get_command(LegatoEnableCommand):
                global_legato_enabled = legato_effect.legato_on

            tick_has_note = tick_data.Note is not None

            # fine tune chain ends when a note happens
            if in_fine_tune_chain and tick_has_note:
                in_fine_tune_chain = False
                # turn legato off if we aren't already in a global legato region
                if not global_legato_enabled:
                    ticks[tick].Commands.append(LegatoEnableCommand(0))

            fine_tune_effect = tick_data.get_command(TuningCommand)
            retuned_note, _ = FurnaceUtil.get_note_active_at(tick, notes)
            is_mid_note = retuned_note and tick_data.kind() != ChiptuneTickData.NoteKind.NOTE
            # only turn on legato if a pitch effect happens for the first time in a note duration
            if fine_tune_effect and is_mid_note and not in_fine_tune_chain:
                in_fine_tune_chain = True
                # turn legato on if we aren't already in a global legato region
                if not global_legato_enabled:
                    # turn on legato on the tick before new note would start
                    if tick - 1 > 0:
                        ticks[tick - 1].Commands.append(LegatoEnableCommand(1))
                    else:
                        self.logger.warning("Cannot convert fine tune because we cannot enable legato before tick 0.")

        return ticks
    
    def convert_notes(self, ticks: List[ChiptuneTickData], notes: List[MMLNote]) -> List[MMLNote]:
        split_notes: List[MMLNote] = notes
        for tick, tick_data in enumerate(ticks):
            if tick_data.get_command(TuningCommand):
                retuned_note, idx = FurnaceUtil.get_note_active_at(tick, split_notes)
                if not retuned_note:
                    continue
                # only want to split the note if fine tune actually happened in middle of note
                if retuned_note.tick == tick or retuned_note.duration + retuned_note.tick == tick:
                    continue
                note1, note2 = FurnaceUtil.split_note(retuned_note, tick)
                split_notes.pop(idx)
                split_notes.insert(idx, note1)
                split_notes.insert(idx + 1, note2)

        return split_notes


@dataclass
class LegatoRegion:
    """A region where legato should be active."""
    start_tick: int
    end_tick: int = None  # None means open-ended (until end of song or next event)


class LegatoConverter:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def convert(self, ticks: List[ChiptuneTickData], notes: List[MMLNote]) -> List[MMLCommand]:
        # Build legato regions
        regions = self.build_legato_regions(ticks)

        # Emit toggle commands
        commands = self.emit_toggle_commands(regions, notes)

        return commands

    def build_legato_regions(self, ticks: List[ChiptuneTickData]) -> List[LegatoRegion]:
        """
        Build a list of tick ranges where legato should be active.
        """
        regions: List[LegatoRegion] = []
        current_region: LegatoRegion = None
        legato_on = False

        tick = 0
        for tick_data in ticks:
            if legato_effect := tick_data.get_command(LegatoEnableCommand):
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
            start_note, _ = FurnaceUtil.get_note_active_at(region.start_tick, notes)
            if start_note:
                start_note.pre_note_commands.append(LegatoToggle(start_note.tick))
            else:
                # No note active, emit as standalone command
                commands.append(LegatoToggle(region.start_tick))

            # Find note that starts at region end and add OFF toggle to its pre_note_commands
            # AMK docs say we have to turn legato off in the middle of the previous note, but that
            # doesn't seem to be necessary.
            if region.end_tick is not None:
                end_note = FurnaceUtil.get_note_starting_at(region.end_tick, notes)
                if end_note:
                    end_note.pre_note_commands.append(LegatoToggle(end_note.tick))
                else:
                    # No note starts at end_tick, emit as standalone command
                    commands.append(LegatoToggle(region.end_tick))

        return commands
    
class EchoConverter:
    def __init__(self, echo_data: SNESEchoData, channel: int, has_loop: bool) -> None:
        self.logger = logging.getLogger(__name__)
        # initial state is set per-channel by the chip config
        self.default_channel_echo = echo_data.echoMask & (0x01 << channel) != 0
        self.echo_state = self.default_channel_echo
        self.has_loop = has_loop

    def convert(self, ticks: List[ChiptuneTickData]) -> List[MMLCommand]:
        echo_commands: List[EchoToggle] = []
        # is echo toggled from channel default?
        for i, tick_data in enumerate(ticks):
            if echo_command := tick_data.get_command(EchoEnableCommand):
                if echo_command.echo_on != self.echo_state:
                    self.echo_state = echo_command.echo_on
                    echo_commands.append(EchoToggle(i))

        # Make sure to reset echo on loop
        # only necessary if there is an explicit loop point. AMK resets echo for us on natural loops
        if self.echo_state != self.default_channel_echo and self.has_loop:
            if len(ticks) > 0:
                echo_commands.append(EchoToggle(len(ticks) - 1))
            else:
                self.logger.warning("Can't process echo on zero-tick channel.")

        return echo_commands
    
# intermediary slide object
@dataclass
class PitchSlide():
    tick: int = 0
    duration: int = 0
    target: int = 0

class PitchBendConverter:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def convert(self, ticks: List[ChiptuneTickData], notes: List[MMLNote]) -> Tuple[List[MMLCommand], List[MMLNote], List[ChiptuneTickData]]:
        # figure out EB commands from notes and bend info
        # create new notes for multiple bends in the same note, and wrap in legato.
        commands: List[PitchEnvelope] = []
        split_notes: List[MMLNote] = []
        env_active = False
        current_pitch: int = None
        global_legato_enabled: bool = False
        out_ticks = copy.deepcopy(ticks)

        # can technically go to FF but choose C0 for neatness
        MAX_BEND_DURATION = 0xC0
        cur_pitch = None
        all_pitch_slides: List[PitchSlide] = []
        # get all pitch slides, splitting if too long
        for i, tick in enumerate(ticks):
            if tick.kind() == ChiptuneTickData.NoteKind.NOTE:
                cur_pitch = tick.Note
            if pitchbend_command := tick.get_command(PitchSlideCommand):
                duration = pitchbend_command.duration
                target = pitchbend_command.target
                if duration <= MAX_BEND_DURATION:
                    all_pitch_slides.append(PitchSlide(i, duration, target))
                else:
                    # Split into chunks with linearly interpolated targets
                    start_pitch = cur_pitch if cur_pitch is not None else target
                    remaining = duration
                    elapsed = 0
                    cur_tick = i
                    while remaining > 0:
                        chunk = min(remaining, MAX_BEND_DURATION)
                        elapsed += chunk
                        t = elapsed / duration
                        interp_target = start_pitch + t * (target - start_pitch)
                        all_pitch_slides.append(PitchSlide(cur_tick, chunk, interp_target))
                        cur_tick += chunk
                        remaining -= chunk
                cur_pitch = target

        cur_note: MMLNote = None
        # then split notes that contain multiple bends and wrap in legato
        # since can't apply pitch envelope mid-note
        for i, tick in enumerate(ticks):
            if tick_legato := tick.get_command(LegatoEnableCommand):
                global_legato_enabled = tick_legato.legato_on

            cur_note = next((note for note in notes if (i >= note.tick and i < note.tick + note.duration)), None)

            if cur_note is None or i != cur_note.tick:
                continue

            current_pitch = cur_note.note
            bends_in_note = [b for b in all_pitch_slides if (b.tick >= cur_note.tick and b.tick < cur_note.tick + cur_note.duration)]
            note_notes: List[MMLNote] = []
            if len(bends_in_note) > 0:
                env_active = True
                first_bend = bends_in_note[0]
                delay = first_bend.tick - cur_note.tick
                bend_amt = round(first_bend.target) - current_pitch
                commands.append(PitchEnvelope(cur_note.tick, delay, first_bend.duration, bend_amt))
                current_pitch += bend_amt
                if len(bends_in_note) > 1:
                    we_enabled_legato = False
                    if not global_legato_enabled:
                        # start legato 1 tick after cause we still want inital attack
                        out_ticks[cur_note.tick + 1].Commands.append(LegatoEnableCommand(1))
                        we_enabled_legato = True
                    global_legato_in_note = any(
                        (cmd := ticks[t].get_command(LegatoEnableCommand)) and cmd.legato_on
                        for t in range(cur_note.tick + 1, min(cur_note.tick + cur_note.duration + 1, len(ticks)))
                    )
                    if we_enabled_legato and not global_legato_in_note:
                        if len(out_ticks) > cur_note.tick + cur_note.duration:
                            out_ticks[cur_note.tick + cur_note.duration].Commands.append(LegatoEnableCommand(0))
                    last_note = cur_note
                    for b in bends_in_note[1:]:
                        note1, last_note = FurnaceUtil.split_note(last_note, b.tick)
                        last_note.note += bend_amt
                        bend_amt = round(b.target) - current_pitch
                        commands.append(PitchEnvelope(b.tick, 0, b.duration, bend_amt))
                        current_pitch += bend_amt
                        note_notes.append(note1)
                    note_notes.append(last_note)
                else:
                    note_notes.append(cur_note)
            else:
                if env_active:
                    commands.append(PitchEnvelope(cur_note.tick, 0, 0, 0))
                    # commands.append(PitchEnvelopeOff(cur_note.tick))
                    env_active = False
                note_notes.append(cur_note)

            split_notes.extend(note_notes)

        return commands, split_notes, out_ticks