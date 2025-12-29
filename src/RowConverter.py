from typing import List, Optional
from dataclasses import dataclass
import sys

from .model.FurnaceData import FurnaceInstrument, FurnaceModule, FurnaceRow
from .model.AMKData import *
from .model.MMLCommands import *
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
            # TODO: For these situations just give up and do everything in ticks
            print("Warning: Furnace ticks not cleanly convertible to amk ticks.")
        print(f"One Furnace tick is {self.tick_ratio:.2g} AMK ticks.")

    def convert_loop_marker(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> Optional[int]:
        # iterate all rows for command 0Bxx (jump to order)
        # This will be interpreted as the intro marker position
        intro_order = None
        for row in flat_rows:
            for effect in row.Effects:
                if effect[0] == 0x0B:
                    intro_order = int(effect[1])
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


    def convert_notes(self, flat_rows: List[FurnaceRow], ins_remote_map: Dict[int, int], instruments: List[FurnaceInstrument]) -> List[MMLNote]:
        notes: List[MMLNote] = []
        tick = 0
        state = FurnaceState()
        # process notes
        cur_dur: Optional[MMLNote] = None
        for i, row in enumerate(flat_rows):
            # check for note delay (this also affects note offs)
            note_tick = tick
            # does this row require legato? Check all effects
            found_legato = False
            for effect in (row.Effects or []):
                effect_num = effect[0]
                value = effect[1]
                if effect_num == FurnaceCommandType.NOTE_DELAY.value:
                    note_tick += int(value * self.tick_ratio)
                # Check if this effect is a legato effect
                found_legato = found_legato or self.is_quick_legato(effect_num)

            note_kind = row.kind()
            if note_kind == FurnaceRow.NoteKind.NOTE:
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

            elif note_kind == FurnaceRow.NoteKind.OFF or note_kind == FurnaceRow.NoteKind.RELEASE:
                if cur_dur is not None:
                    if found_legato != state.is_legato:
                        cur_dur.pre_note_commands.append(LegatoToggle(note_tick))
                        state.is_legato = found_legato  
                    cur_dur.duration = note_tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = None
            

            # check for quick legato, make a new note if found
            for effect in row.Effects:
                effect_num = effect[0]
                value = effect[1]
                if self.is_quick_legato(effect_num):
                    pre_note_commands = []
                    note, delay = self.get_quick_legato_note(effect_num, value, cur_dur.note)
                    new_note_onset = note_tick + int(delay * self.tick_ratio)
                    if not state.is_legato:
                        pre_note_commands.append(LegatoToggle(note_tick))
                        state.is_legato = True
                    if cur_dur is not None:
                        cur_dur.duration = new_note_onset - cur_dur.tick
                        notes.append(cur_dur)
                    cur_dur = MMLNote(new_note_onset, 0, note, cur_dur.instrument, pre_note_commands)

            tick += self.amk_ticks_per_row
        
        # Finalize possible final note
        if cur_dur is not None:
            cur_dur.duration = tick - cur_dur.tick
            notes.append(cur_dur)

        if state.is_legato:
            notes[-1].pre_note_commands.append(LegatoToggle(tick))

        return notes

    def is_volume_slide(self, effect_num: int) -> bool:
        return effect_num == FurnaceCommandType.VOLUME_SLIDE.value \
            or effect_num == FurnaceCommandType.FAST_VOLUME_SLIDE.value \
            or effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_UP.value \
            or effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_DOWN.value

    def is_quick_legato(self, effect_num: int) -> bool:
        return effect_num == FurnaceCommandType.QUICK_LEGATO.value

    def get_pitch_slide_info(self, effect_num: int, value: int) -> Tuple[Optional[int], Optional[int]]:
        if effect_num == FurnaceCommandType.NOTE_SLIDE_DOWN.value or \
           effect_num == FurnaceCommandType.NOTE_SLIDE_UP.value:
            # speed is first value of nibble, note is second+
            # convert max $0F Furnace to quarter note $30 AMK
            # TODO: figure out precise speed scaling, I just earballed it
            speed = value >> 4
            semitones = value & 0x0F
            if effect_num == FurnaceCommandType.NOTE_SLIDE_DOWN.value:
                semitones = -semitones
        else:
            print(f"Warning: Invalid pitch slide effect number {effect_num}.", file=sys.stderr)
            return None, None

        return semitones, speed

    def get_quick_legato_note(self, effect_num: int, value: int, cur_note: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
        if effect_num == FurnaceCommandType.QUICK_LEGATO.value:
            x = value >> 4
            semitones = value & 0x0F
            delay = 0
            if x < 8:
                delay = x
            else:
                delay = x - 8
                semitones = -semitones

            new_note = cur_note + semitones

            return new_note, delay

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
                effect_num = effect[0]
                value = effect[1]
                if self.is_volume_slide(effect_num):
                    new_command = slide_helper.handle_new_command(effect_num, value)
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
                effect_num = effect[0]
                value = effect[1]

                if effect_num == FurnaceCommandType.PAN.value:
                    slide_helper.set_target(value)
                    amk_pan = MMLUtil.fur_pan_to_amk(value)
                    commands.append(PanChange(tick, amk_pan))
                elif effect_num == FurnaceCommandType.STEREO_PAN.value:
                    left_volume = value >> 4
                    right_volume = value & 0x0F
                    # normalize pan state to linear pan
                    cur_pan = MMLUtil.fur_stereo_pan_to_amk(left_volume, right_volume)
                    slide_helper.set_target(cur_pan)
                    amk_pan = MMLUtil.stereo_to_unity_pan(left_volume, right_volume)
                    commands.append(PanChange(tick, amk_pan))

                if effect_num == FurnaceCommandType.PAN_SLIDE.value:
                    new_command = slide_helper.handle_new_command(effect_num, value)
                    if new_command is not None:
                        commands.append(new_command)
                        
            new_command = slide_helper.tick(self.amk_ticks_per_row)
            if new_command is not None:
                commands.append(new_command)
   
            tick += self.amk_ticks_per_row

        return commands

    def convert_pitch_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        tick = 0
        slide_helper = PitchSlider(tick)
        for row in flat_rows:           
            if row.kind() == FurnaceRow.NoteKind.NOTE:
                if row.Note is not None:
                    slide_helper.set_active_note(row.Note)
                    active_note = row.Note
                
            # Effects
            for effect in (row.Effects or []):
                effect_num = effect[0]
                value = effect[1]
                if effect_num == FurnaceCommandType.NOTE_SLIDE_UP.value or effect_num == FurnaceCommandType.NOTE_SLIDE_DOWN.value:
                    semitones, speed = self.get_pitch_slide_info(effect_num, value)
                    target_note = active_note + semitones
                    if target_note > MMLUtil.AMK_MAX_PITCH:
                        target_note = MMLUtil.AMK_MAX_PITCH
                    if target_note < 0:
                        target_note = 0
                    # Empirical formula to convert speed to pitch change rate
                    ticks_per_octave = 96 / speed
                    octaves_to_slide = abs(semitones) / 12
                    ticks_to_slide = ticks_per_octave * octaves_to_slide
                    duration = int(ticks_to_slide * self.tick_ratio)
                    LONGEST_DURATION = int(max(MMLUtil.TICK_TO_DURATION.keys()) / 2)
                    if duration > LONGEST_DURATION:
                        print(f"Warning: Pitch slide duration {duration} is greater than the longest tick duration {MMLUtil.TICK_TO_DURATION.keys()[-1]}. Things might break.", file=sys.stderr)
                    commands.append(PitchBend(tick, duration, target_note))
                elif effect_num == FurnaceCommandType.PITCH_SLIDE_UP.value or effect_num == FurnaceCommandType.PITCH_SLIDE_DOWN.value:
                    new_command = slide_helper.handle_new_command(effect_num, value)
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
        commands.extend(self.convert_pitch_commands(flat_rows, module))

        # sort commands by tick
        sorted_commands = sorted(commands, key=lambda x: x.tick)

        return sorted_commands