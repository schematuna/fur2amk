from typing import List, Optional
from dataclasses import dataclass
import sys

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

class RowConverter:
    def __init__(self, fur_ticks_per_row: int) -> None:
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
            print("Warning: Furnace ticks not cleanly convertible to amk ticks.")
        print(f"One Furnace tick is {self.tick_ratio:.2g} AMK ticks.")

    def convert_loop_marker(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> Optional[int]:
        # iterate all rows for command 0Bxx (jump to order)
        # This will be interpreted as the intro marker position
        intro_order = None
        for row in flat_rows:
            for effect in row.Effects:
                if isinstance(effect, JumpToOrderEffect):
                    intro_order = effect.order_number
                    return intro_order * module.PatternLength * self.amk_ticks_per_row

        return None

    def get_pre_note_commands(self, fur_ins: FurnaceInstrument, ins_remote_map: Dict[int, int], state: FurnaceState, note_tick: int) -> List[MMLCommand]:
        # instrument echo
        pre_note_commands = []
        if fur_ins.snes_macro_data.is_echo != state.echo:
            pre_note_commands.append(EchoToggle(note_tick))
            state.echo = fur_ins.snes_macro_data.is_echo

        # mid-note gain change, handled by remote command
        if fur_ins.index in ins_remote_map:
            gain_speed = fur_ins.snes_macro_data.gain_speed
            remote_comand_idx = ins_remote_map[fur_ins.index]
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
            print(f"Warning: Portamento duration {amk_duration} is greater than the longest tick duration {max_duration}.", file=sys.stderr)
        command = PitchBend(tick, amk_duration, target_note)

        return target_note, command

    def convert_slides(self, tick: int, row: FurnaceRow, slide_helper: PitchSlider, active_note: Optional[int]) -> Tuple[Optional[int], List[MMLCommand]]:
        commands: List[MMLCommand] = []
        new_active_note = active_note
        for effect in (row.Effects or []):
            if isinstance(effect, NoteSlideEffect):
                semitones = effect.semitones
                speed = effect.speed
                if active_note is None:
                    print(f"Warning: Pitch slide effect found on non-note row, ignoring.", file=sys.stderr)
                    continue
                target_note = active_note + semitones
                target_note = max(0,min(target_note, MMLUtil.AMK_MAX_PITCH))
                # for note slides, each speed unit is 4 pitch steps per tick
                ticks_to_slide = FurnaceUtil.ticks_from_speed(speed * 4, abs(semitones))
                amk_duration = max(2, int(ticks_to_slide * self.tick_ratio))
                max_duration = slide_helper.get_max_duration()
                if amk_duration > max_duration:
                    print(f"Warning: Pitch slide duration {amk_duration} is greater than the longest tick duration {max_duration}.", file=sys.stderr)
                commands.append(PitchBend(tick, amk_duration, target_note))
                new_active_note = target_note
            elif isinstance(effect, PitchSlideEffect):
                new_command = slide_helper.handle_new_command(effect.change_per_tick)
                if new_command is not None:
                    new_active_note = new_command.note
                    commands.append(new_command)
                        
        new_command = slide_helper.tick(self.amk_ticks_per_row)
        if new_command is not None:
            new_active_note = new_command.note
            commands.append(new_command)

        return new_active_note, commands

    def convert_notes(self, flat_rows: List[FurnaceRow], ins_remote_map: Dict[int, int], instruments: List[FurnaceInstrument]) -> List[MMLNote]:
        notes: List[MMLNote] = []
        commands: List[MMLCommand] = []
        tick = 0
        state = FurnaceState()
        slide_helper = PitchSlider(tick)

        # the current note duration
        cur_dur: Optional[MMLNote] = None
        # the current active pitch, considering both pitch commands and explicit notes
        active_note = None
        # process notes and pitch commands
        for i, row in enumerate(flat_rows):
            portamento_speed = None
            for effect in (row.Effects or []):
                if isinstance(effect, PortamentoEffect):
                    portamento_speed = effect.speed
                    break
            
            if portamento_speed is not None:
                # handle portamento specially
                if row.kind() == FurnaceRow.NoteKind.NOTE:
                    active_note, portamento_command = self.convert_portamento(tick, active_note, row.Note, portamento_speed)
                    if portamento_command is not None:
                        commands.append(portamento_command)
                else:
                    print(f"Warning: Portamento effect found on non-note row, ignoring.", file=sys.stderr)


            # check for note delay (this also affects note offs)
            note_tick = tick
            for effect in (row.Effects or []):
                if isinstance(effect, NoteDelayEffect):
                    note_tick += int(effect.delay_ticks * self.tick_ratio)

            found_legato = False
            for effect in (row.Effects or []):
                if isinstance(effect, QuickLegatoEffect):
                    found_legato = True
                    break
                
            note_kind = row.kind()
            # don't make a new note for portamento rows, pitchbend will handle that
            if note_kind == FurnaceRow.NoteKind.NOTE and portamento_speed is None:
                fur_ins = None
                for ins in instruments:
                    if ins.index == row.Ins:
                        fur_ins = ins
                        break
                pre_note_commands = self.get_pre_note_commands(fur_ins, ins_remote_map, state, note_tick)

                if found_legato != state.is_legato:
                    pre_note_commands.append(LegatoToggle(note_tick))
                    state.is_legato = found_legato
                    
                # TODO: handle sample maps here, AMK doesn't need to know about that
                amk_ins_idx = row.Ins

                if cur_dur is not None:
                    cur_dur.duration = note_tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = MMLNote(note_tick, 0, row.Note, amk_ins_idx, pre_note_commands)
                active_note = row.Note

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
            for effect in row.Effects:
                if isinstance(effect, QuickLegatoEffect):
                    pre_note_commands = []
                    semitones = effect.semitones
                    delay = effect.delay
                    new_note_onset = note_tick + int(delay * self.tick_ratio)
                    if not state.is_legato:
                        pre_note_commands.append(LegatoToggle(note_tick))
                        state.is_legato = True
                    if cur_dur is not None:
                        cur_dur.duration = new_note_onset - cur_dur.tick
                        notes.append(cur_dur)
                    new_note = active_note + semitones
                    new_note = max(0,min(new_note, MMLUtil.AMK_MAX_PITCH))
                    cur_dur = MMLNote(new_note_onset, 0, new_note, cur_dur.instrument, pre_note_commands)
                    active_note = new_note

            slide_helper.set_active_note(active_note)
            active_note, pitch_commands = self.convert_slides(tick, row, slide_helper, active_note)
            commands.extend(pitch_commands)

            tick += self.amk_ticks_per_row
        
        # Finalize possible final note
        if cur_dur is not None:
            cur_dur.duration = tick - cur_dur.tick
            notes.append(cur_dur)

        if state.is_legato:
            notes[-1].pre_note_commands.append(LegatoToggle(tick))

        return notes, commands


    def convert_volume_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        # amk ticks
        tick = 0
        new_vol = None
        # furnace volume units per furnace tick
        slide_helper = VolumeSlider(tick)
        for row in flat_rows:
            vol = row.Vol
            if vol is not None:
                new_vol = vol
                commands.append(VolumeChange(tick, MMLUtil.find_v(vol)))
            else:
                new_vol = None

            for effect in (row.Effects or []):
                if isinstance(effect, VolumeSlideEffect):
                    new_command = slide_helper.handle_new_command(effect.change_per_tick)
                    if new_command is not None:
                        commands.append(new_command)
                elif isinstance(effect, FineVolumeSlideEffect):
                    new_command = slide_helper.handle_new_command(effect.change_per_tick)
                    if new_command is not None:
                        commands.append(new_command)
                        
            # set row volume after ending previous command but before ticking new one
            if new_vol is not None:
                slide_helper.set_target(new_vol)
            new_command = slide_helper.tick(self.amk_ticks_per_row)
            if new_command is not None:
                commands.append(new_command)
   
            tick += self.amk_ticks_per_row

        return commands

    def convert_pan_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        # amk ticks
        tick = 0
        slide_helper = PanSlider(tick)
        for row in flat_rows:
            for effect in (row.Effects or []):
                if isinstance(effect, PanEffect):
                    slide_helper.set_target(effect.pan_position)
                    amk_pan = FurnaceUtil.unity_to_amk_pan(effect.pan_position)
                    commands.append(PanChange(tick, amk_pan))
                elif isinstance(effect, StereoPanEffect):
                    cur_pan = FurnaceUtil.stereo_to_unity_pan(effect.left_volume, effect.right_volume)
                    slide_helper.set_target(cur_pan)
                    amk_pan = FurnaceUtil.stereo_to_amk_pan(effect.left_volume, effect.right_volume)
                    commands.append(PanChange(tick, amk_pan))
                elif isinstance(effect, PanSlideEffect):
                    new_command = slide_helper.handle_new_command(effect.change_per_tick)
                    if new_command is not None:
                        commands.append(new_command)
                        
            new_command = slide_helper.tick(self.amk_ticks_per_row)
            if new_command is not None:
                commands.append(new_command)
   
            tick += self.amk_ticks_per_row

        return commands

    def convert_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        # process rows into commands
        commands: List[MMLCommand] = []

        commands.extend(self.convert_volume_commands(flat_rows, module))
        commands.extend(self.convert_pan_commands(flat_rows, module))

        # sort commands by tick
        sorted_commands = sorted(commands, key=lambda x: x.tick)

        return sorted_commands