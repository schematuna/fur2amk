from typing import List, Optional
from dataclasses import dataclass, field
import sys
import logging

from .model.FurnaceData import FurnaceInstrument, FurnaceModule, FurnaceRow
from .model.AMKData import *
from .model.MMLCommands import *
from .model.FurnaceEffects import *
from .MMLUtil import *
from .FurnaceUtil import *

# persistent channel state
@dataclass
class FurnaceState:
    gain_remote: RemoteCommand = None
    fur_ins_idx: int = None
    echo: bool = True
    is_legato: bool = False

@dataclass
class MappingInfo:
    amk_ins_idx: int
    note_to_play: int

# conversion helper data class
@dataclass
class InstrumentInfo:
    # default amk instrument
    amk_ins: Optional[int] = None

    # sample map data
    # note -> mapping_info
    ins_map: Dict[int, MappingInfo] = field(default_factory=dict)
    # list of remote command indices associated with this instrument
    remote_commands: List[int] = field(default_factory=list)

class RowConverter:
    def __init__(self, fur_ticks_per_row: int) -> None:
        self.logger = logging.getLogger(__name__)
        # determine musical duration to map to a furnace row
        # find first AMK tick value that is greater than or equal to the furnace tick rate
        self.amk_ticks_per_row = 12
        for tick_value in MMLUtil.TICK_TO_DURATION.keys():
            if tick_value >= fur_ticks_per_row:
                self.amk_ticks_per_row = tick_value
                break

        # ratio of amk ticks to furnace ticks
        self.tick_ratio = self.amk_ticks_per_row / fur_ticks_per_row
        if self.tick_ratio != round(self.tick_ratio):
            self.logger.warning("Furnace ticks not cleanly convertible to amk ticks.")
        self.logger.info(f"One Furnace tick is {self.tick_ratio:.2g} AMK ticks.")

    def convert_loop_marker(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> Optional[int]:
        # iterate all rows for command 0Bxx (jump to order)
        # This will be interpreted as the intro marker position
        intro_order = None
        for row in flat_rows:
            if effect := row.get_effect(JumpToOrderEffect):
                intro_order = effect.order_number
                return intro_order * module.PatternLength * self.amk_ticks_per_row

        return None

    def get_pre_note_commands(self, fur_ins: FurnaceInstrument, ins_info: Dict[int, InstrumentInfo], state: FurnaceState, note_tick: int) -> List[MMLCommand]:
        # instrument echo
        pre_note_commands = []
        if fur_ins.snes_macro_data.is_echo != state.echo:
            pre_note_commands.append(EchoToggle(note_tick))
            state.echo = fur_ins.snes_macro_data.is_echo

        # mid-note gain change, handled by remote command
        if fur_ins.index in ins_info and len(ins_info[fur_ins.index].remote_commands) > 0:
            gain_speed = fur_ins.snes_macro_data.gain_speed
            # just assume first remote command is gain for now
            remote_comand_idx = ins_info[fur_ins.index].remote_commands[0]
            gain_remote = RemoteCommand(note_tick, remote_comand_idx, RemoteCommandTiming.AFTER_START, gain_speed)
            if gain_remote is not state.gain_remote:
                pre_note_commands.append(gain_remote)
                state.gain_remote = gain_remote
        elif state.gain_remote is not None: # turn off remote commands when gain is disabled
            pre_note_commands.append(RemoteCommand(note_tick, 99, RemoteCommandTiming.DISABLE))
            state.gain_remote = None

        return pre_note_commands

    def convert_portamento(self, tick: int, active_note: int, target_note: int, speed: int) -> Tuple[int, MMLCommand]:
        semitones = target_note - active_note
        ticks_to_slide = FurnaceUtil.ticks_from_speed(speed, semitones)
        amk_duration = max(2, int(ticks_to_slide * self.tick_ratio))
        max_duration = PitchSlider.get_max_duration()
        if amk_duration > max_duration:
            self.logger.warning(f"Portamento duration {amk_duration} is greater than the longest tick duration {max_duration}.")
        command = PitchBend(tick, amk_duration, target_note)

        return target_note, command

    def convert_slides(self, tick: int, row: FurnaceRow, slide_helper: PitchSlider, active_note: Optional[int]) -> Tuple[Optional[int], List[MMLCommand]]:
        commands: List[MMLCommand] = []
        new_active_note = active_note
        if effect := row.get_effect(NoteSlideEffect):
            if active_note is None:
                self.logger.warning("Pitch slide effect found on non-note row, ignoring.")
            else:
                target_note = active_note + effect.semitones
                target_note = max(0,min(target_note, MMLUtil.AMK_MAX_PITCH))
                # for note slides, each speed unit is 4 pitch steps per tick
                ticks_to_slide = FurnaceUtil.ticks_from_speed(effect.speed * 4, abs(effect.semitones))
                amk_duration = max(2, int(ticks_to_slide * self.tick_ratio))
                max_duration = slide_helper.get_max_duration()
                if amk_duration > max_duration:
                    self.logger.warning(f"Pitch slide duration {amk_duration} is greater than the longest tick duration {max_duration}.")
                commands.append(PitchBend(tick, amk_duration, target_note))
                new_active_note = target_note
                slide_helper.set_target(new_active_note)

        if effect := row.get_effect(PitchSlideEffect):
            if new_command := slide_helper.handle_new_effect(effect):
                new_active_note = new_command.note
                commands.append(new_command)
                        
        if new_command := slide_helper.tick(self.amk_ticks_per_row):
            new_active_note = new_command.note
            commands.append(new_command)

        return new_active_note, commands

    def convert_notes(self, flat_rows: List[FurnaceRow], ins_info: Dict[int, InstrumentInfo], instruments: List[FurnaceInstrument]) -> List[MMLNote]:
        notes: List[MMLNote] = []
        commands: List[MMLCommand] = []
        tick = 0
        state = FurnaceState()
        slide_helper = PitchSlider(self.tick_ratio, tick)

        # the current note duration
        cur_dur: Optional[MMLNote] = None
        # the current active pitch, considering both pitch commands and explicit notes
        active_note = None
        # process notes and pitch commands
        for row in flat_rows:
            has_portamento = False
            if effect := row.get_effect(PortamentoEffect):
                has_portamento = True
                if row.kind() == FurnaceRow.NoteKind.NOTE:
                    active_note, portamento_command = self.convert_portamento(tick, active_note, row.Note, effect.speed)
                    if portamento_command is not None:
                        commands.append(portamento_command)
                else:
                    self.logger.warning("Portamento effect found on non-note row, ignoring.")

            note_kind = row.kind()
            if note_kind == FurnaceRow.NoteKind.OFF or note_kind == FurnaceRow.NoteKind.RELEASE:
                # finish any pitch slides that are still active
                # necessary to end the slide before we tick again
                pitch_command = slide_helper.end_slide(None)
                if pitch_command is not None:
                    commands.append(pitch_command)
            active_note, pitch_commands = self.convert_slides(tick, row, slide_helper, active_note)
            commands.extend(pitch_commands)

            # check for note delay (this also affects note offs)
            note_tick = tick
            if effect := row.get_effect(NoteDelayEffect):
                note_tick += int(effect.delay_ticks * self.tick_ratio)

            found_legato = False
            if effect := row.get_effect(QuickLegatoEffect):
                found_legato = True
                
            # don't make a new note for portamento rows, pitchbend will handle that
            if note_kind == FurnaceRow.NoteKind.NOTE and not has_portamento:
                fur_ins = None
                for ins in instruments:
                    if ins.index == row.Ins:
                        fur_ins = ins
                        break
                pre_note_commands = self.get_pre_note_commands(fur_ins, ins_info, state, note_tick)

                if found_legato != state.is_legato:
                    pre_note_commands.append(LegatoToggle(note_tick))
                    state.is_legato = found_legato
                    
                note_to_play = row.Note
                amk_ins_idx = None
                if row.Ins not in ins_info:
                    self.logger.error(f"No instrument info found for Furnace instrument {row.Ins}, this is not right.")
                    continue
                # Get the AMK instrument index using ins_map
                if fur_ins.use_sample_map:
                    note_map = ins_info[row.Ins].ins_map
                    # Try to find exact note match first
                    if row.Note in note_map:
                        amk_ins_idx = note_map[row.Note].amk_ins_idx
                        note_to_play = note_map[row.Note].note_to_play
                    else:
                        # No mapping found, use Furnace instrument index as fallback
                        self.logger.warning(f"No instrument mapping found for Furnace instrument {row.Ins}, note {row.Note}.")
                        amk_ins_idx = 0
                else:
                    amk_ins_idx = ins_info[row.Ins].amk_ins

                if cur_dur is not None:
                    cur_dur.duration = note_tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = MMLNote(note_tick, 0, note_to_play, amk_ins_idx, pre_note_commands)
                active_note = note_to_play
                slide_helper.set_target(active_note)

            elif note_kind == FurnaceRow.NoteKind.OFF or note_kind == FurnaceRow.NoteKind.RELEASE:
                if cur_dur is not None:
                    if found_legato != state.is_legato:
                        cur_dur.pre_note_commands.append(LegatoToggle(note_tick))
                        state.is_legato = found_legato  
                    cur_dur.duration = note_tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = None
                active_note = None
            

            # check for quick legato, make a new note if found
            if effect := row.get_effect(QuickLegatoEffect):
                pre_note_commands = []
                new_note_onset = note_tick + int(effect.delay * self.tick_ratio)
                if not state.is_legato:
                    pre_note_commands.append(LegatoToggle(note_tick))
                    state.is_legato = True
                if cur_dur is not None:
                    cur_dur.duration = new_note_onset - cur_dur.tick
                    notes.append(cur_dur)
                new_note = active_note + effect.semitones
                new_note = max(0,min(new_note, MMLUtil.AMK_MAX_PITCH))
                cur_dur = MMLNote(new_note_onset, 0, new_note, cur_dur.instrument, pre_note_commands)
                active_note = new_note
                slide_helper.set_target(active_note)

            tick += self.amk_ticks_per_row
        
        # Finalize possible final note
        if cur_dur is not None:
            cur_dur.duration = tick - cur_dur.tick
            notes.append(cur_dur)

        # if necessary, toggle legato off before looping
        if state.is_legato:
            notes[-1].pre_note_commands.append(LegatoToggle(tick))

        return notes, commands


    def convert_volume_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        # amk ticks
        tick = 0
        new_vol = None
        # furnace volume units per furnace tick
        slide_helper = VolumeSlider(self.tick_ratio, tick)
        for row in flat_rows:
            new_vol = row.Vol
            if new_vol is not None:
                commands.append(VolumeChange(tick, MMLUtil.find_v(new_vol)))

            if effect := row.get_effect(VolumeSlideEffect) or row.get_effect(FineVolumeSlideEffect):
                if new_command := slide_helper.handle_new_effect(effect):
                    commands.append(new_command)
                    
            # set row volume after ending previous command but before ticking new one
            if new_vol is not None:
                slide_helper.set_target(new_vol)
                
            if new_command := slide_helper.tick(self.amk_ticks_per_row):
                commands.append(new_command)
   
            tick += self.amk_ticks_per_row

        return commands

    def convert_pan_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        # amk ticks
        tick = 0
        slide_helper = PanSlider(self.tick_ratio, tick)
        for row in flat_rows:
            if effect := row.get_effect(PanEffect):
                slide_helper.set_target(effect.pan_position)
                amk_pan = FurnaceUtil.unity_to_amk_pan(effect.pan_position)
                commands.append(PanChange(tick, amk_pan))
            
            if effect := row.get_effect(StereoPanEffect):
                cur_pan = FurnaceUtil.stereo_to_unity_pan(effect.left_volume, effect.right_volume)
                slide_helper.set_target(cur_pan)
                amk_pan = FurnaceUtil.stereo_to_amk_pan(effect.left_volume, effect.right_volume)
                commands.append(PanChange(tick, amk_pan))
            
            if effect := row.get_effect(PanSlideEffect):
                if new_command := slide_helper.handle_new_effect(effect):
                    commands.append(new_command)
                        
            if new_command := slide_helper.tick(self.amk_ticks_per_row):
                commands.append(new_command)
   
            tick += self.amk_ticks_per_row

        return commands

    def convert_other_commands(self, flat_rows: List[FurnaceRow]) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        tick = 0

        for row in flat_rows:
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

            tick += self.amk_ticks_per_row

        return commands

    def convert_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        # process rows into commands
        commands: List[MMLCommand] = []

        commands.extend(self.convert_volume_commands(flat_rows, module))
        commands.extend(self.convert_pan_commands(flat_rows, module))
        commands.extend(self.convert_other_commands(flat_rows))

        # sort commands by tick
        sorted_commands = sorted(commands, key=lambda x: x.tick)

        return sorted_commands