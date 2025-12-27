from typing import List, Optional
from dataclasses import dataclass
import sys

from .model.FurnaceData import FurnaceInstrument, FurnaceModule, FurnaceRow
from .model.AMKData import *
from .model.MMLCommands import *
from .MMLUtil import *

# persistent channel state
@dataclass
class FurnaceState:
    gain_remote: RemoteCommand = None
    fur_ins_idx: int = None
    echo: bool = True

class FurnaceCommandType(Enum):
    STEREO_PAN = 0x08
    VOLUME_SLIDE = 0x0A
    PAN = 0x80
    PAN_SLIDE = 0x83
    NOTE_SLIDE_UP = 0xE1
    NOTE_SLIDE_DOWN = 0xE2
    NOTE_DELAY = 0xED
    FINE_VOLUME_SLIDE_UP = 0xF3
    FINE_VOLUME_SLIDE_DOWN = 0xF4
    FAST_VOLUME_SLIDE = 0xFA

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
            for effect in row.Effects:
                effect_num = effect[0]
                value = effect[1]
                if effect_num == FurnaceCommandType.NOTE_DELAY.value:
                    note_tick += int(value * self.tick_ratio)
                    break

            note_kind = row.kind()
            if note_kind == FurnaceRow.NoteKind.NOTE:
                fur_ins = None
                for ins in instruments:
                    if ins.index == row.Ins:
                        fur_ins = ins
                        break
                pre_note_commands = self.get_pre_note_commands(fur_ins, ins_remote_map, state, note_tick)
                    
                # TODO: handle sample maps here, AMK doesn't need to know about that
                amk_ins = row.Ins

                if cur_dur is not None:
                    cur_dur.duration = note_tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = MMLNote(note_tick, 0, row.Note, amk_ins, pre_note_commands)
            elif note_kind == FurnaceRow.NoteKind.OFF or note_kind == FurnaceRow.NoteKind.RELEASE:
                if cur_dur is not None:
                    cur_dur.duration = note_tick - cur_dur.tick
                    notes.append(cur_dur)
                cur_dur = None
            
            tick += self.amk_ticks_per_row
        
        # Finalize possible final note
        if cur_dur is not None:
            cur_dur.duration = tick - cur_dur.tick
            notes.append(cur_dur)

        return notes

    def get_active_note(self, flat_rows: List[FurnaceRow], row_idx: int) -> Optional[int]:
        cur_note = None
        for i, row in enumerate(flat_rows):
            if row.kind() == FurnaceRow.NoteKind.NOTE:
                cur_note = row.Note
            if row_idx == i:
                return cur_note
        return None

    def is_volume_slide(self, effect_num: int) -> bool:
        return effect_num == FurnaceCommandType.VOLUME_SLIDE.value \
            or effect_num == FurnaceCommandType.FAST_VOLUME_SLIDE.value \
            or effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_UP.value \
            or effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_DOWN.value

    def get_volume_slide_change(self, effect_num: int, value: int) -> Optional[int]:
        vol_change_per_tick = None
        if effect_num == FurnaceCommandType.VOLUME_SLIDE.value or effect_num == FurnaceCommandType.FAST_VOLUME_SLIDE.value:
            rate_divisor = 4
            if effect_num == FurnaceCommandType.FAST_VOLUME_SLIDE.value:
                # fast volume slides are 4 times faster than normal volume slides
                rate_divisor = 1

            up = value >> 4
            down = value & 0x0F
            if down == 0 and up == 0:
                vol_change_per_tick = None
            elif down == 0:
                vol_change_per_tick = up / rate_divisor
            elif up == 0:
                vol_change_per_tick = -down / rate_divisor
            else:
                print("Warning: Invalid volume slide effect value.", file=sys.stderr)
        # fine volume slides are 64 times slower than normal volume slides
        elif effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_UP.value:
            vol_change_per_tick = value / 64
        elif effect_num == FurnaceCommandType.FINE_VOLUME_SLIDE_DOWN.value:
            vol_change_per_tick = -value / 64
        else:
            print(f"Warning: Invalid volume slide effect number {effect_num}.", file=sys.stderr)
        
        return vol_change_per_tick

    def convert_volume_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        # amk ticks
        tick = 0
        # furnace volume units per furnace tick
        vol_change_per_tick: Optional[int] = None
        current_slide: Optional[VolumeFade] = None
        vol_target: int = 0
        cur_vol: int = 0
        for i, row in enumerate(flat_rows):
            vol = row.Vol
            if vol is not None:
                cur_vol = vol
                commands.append(VolumeChange(tick, MMLUtil.find_v(vol)))

            for effect in (row.Effects or []):
                effect_num = effect[0]
                value = effect[1]
                if self.is_volume_slide(effect_num):
                    # this could be another slide or a stop slide command. Either way, we wrap up any current slide
                    if current_slide is not None:
                        slide_duration = tick - current_slide.tick
                        LONGEST_DURATION = max(MMLUtil.TICK_TO_DURATION.keys())
                        if slide_duration > LONGEST_DURATION:
                            print(f"Warning: Volume slide duration {slide_duration} is greater than the longest tick duration {LONGEST_DURATION}. Things might break.", file=sys.stderr)
                        current_slide.duration = slide_duration
                        current_slide.target_volume = MMLUtil.find_v(round(vol_target))
                        commands.append(current_slide)

                    vol_change_per_tick = self.get_volume_slide_change(effect_num, value)

                    if vol_change_per_tick is not None:
                        current_slide = VolumeFade(tick, None, None)
                    else:
                        current_slide = None
                        
            if current_slide is not None:
                # if slide is too long, split it into multiple slides
                cur_duration = tick - current_slide.tick
                LONGEST_DURATION = max(MMLUtil.TICK_TO_DURATION.keys())
                if cur_duration >= LONGEST_DURATION:
                    current_slide.duration = LONGEST_DURATION
                    current_slide.target_volume = MMLUtil.find_v(round(vol_target))
                    commands.append(current_slide)
                    current_slide = VolumeFade(tick, None, None)

                vol_target = cur_vol + vol_change_per_tick * module.Speed1
                if vol_target > 0x7F:
                    vol_target = 0x7F
                if vol_target < 0:
                    vol_target = 0

                # track current volume separate from target volume
                # this allows volume changes and volume slides to coexist on the same row
                cur_vol = vol_target
   
            tick += self.amk_ticks_per_row

        return commands

    def convert_pan_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        # amk ticks
        tick = 0
        # furnace pan units per furnace tick
        pan_change_per_tick: Optional[int] = None
        current_slide: Optional[PanFade] = None
        pan_target: int = 0
        cur_pan: int = 0
        for i, row in enumerate(flat_rows):
            for effect in (row.Effects or []):
                effect_num = effect[0]
                value = effect[1]

                if effect_num == FurnaceCommandType.PAN.value:
                    cur_pan = value
                    amk_pan = MMLUtil.fur_pan_to_amk(value)
                    commands.append(PanChange(tick, amk_pan))
                elif effect_num == FurnaceCommandType.STEREO_PAN.value:
                    left_volume = value >> 4
                    right_volume = value & 0x0F
                    # normalize pan state to linear pan
                    cur_pan = MMLUtil.fur_stereo_pan_to_amk(left_volume, right_volume)

                    amk_pan = MMLUtil.stereo_to_unity_pan(left_volume, right_volume)
                    commands.append(PanChange(tick, amk_pan))

                if effect_num == FurnaceCommandType.PAN_SLIDE.value:
                    # this could be another slide or a stop slide command. Either way, we wrap up any current slide
                    if current_slide is not None:
                        slide_duration = tick - current_slide.tick
                        LONGEST_DURATION = max(MMLUtil.TICK_TO_DURATION.keys())
                        if slide_duration > LONGEST_DURATION:
                            print(f"Warning: Pan slide duration {slide_duration} is greater than the longest tick duration {LONGEST_DURATION}. Things might break.", file=sys.stderr)
                        current_slide.duration = slide_duration
                        current_slide.target_pan = MMLUtil.fur_pan_to_amk(round(pan_target))
                        commands.append(current_slide)

                    left = value >> 4
                    right = value & 0x0F
                    if right == 0 and left == 0:
                        pan_change_per_tick = None
                    elif right == 0:
                        # halved because pan is spread across both channels in Furnace
                        pan_change_per_tick = -left / 2
                    elif left == 0:
                        pan_change_per_tick = right / 2
                    else:
                        print(f"Warning: Invalid pan slide effect value {value}.", file=sys.stderr)
                    if pan_change_per_tick is not None:
                        current_slide = PanFade(tick, None, None)
                    else:
                        current_slide = None
                        
            if current_slide is not None:
                # if slide is too long, split it into multiple slides
                cur_duration = tick - current_slide.tick
                LONGEST_DURATION = max(MMLUtil.TICK_TO_DURATION.keys())
                if cur_duration >= LONGEST_DURATION:
                    current_slide.duration = LONGEST_DURATION
                    current_slide.target_pan = MMLUtil.fur_pan_to_amk(round(pan_target))
                    commands.append(current_slide)
                    current_slide = PanFade(tick, None, None)

                # increment target pan
                pan_target = cur_pan + pan_change_per_tick * module.Speed1
                if pan_target > 0xFF:
                    pan_target = 0xFF
                if pan_target < 0:
                    pan_target = 0

                # track current pan separate from target pan
                # this allows volume changes and volume slides to coexist on the same row
                cur_pan = pan_target
   
            tick += self.amk_ticks_per_row

        return commands

    def convert_other_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        commands: List[MMLCommand] = []
        tick = 0
        for i, row in enumerate(flat_rows):            
            # Effects
            for effect in (row.Effects or []):
                effect_num = effect[0]
                value = effect[1]
                if effect_num == FurnaceCommandType.NOTE_SLIDE_UP.value:
                    # speed is first value of nibble, note is second+
                    # convert max $0F Furnace to quarter note $30 AMK
                    # TODO: figure out precise speed scaling, I just earballed it
                    speed = int(48 * (value >> 4) / 15)
                    semitones = value & 0x0F
                    note = self.get_active_note(flat_rows, i)
                    if note is not None:
                        bent_note = note + semitones
                    else:
                        print(f"Warning: No note found at tick {tick} for note slide up effect {effect}.", file=sys.stderr)
                        continue
                    commands.append(PitchBend(tick, bent_note, speed))

            tick += self.amk_ticks_per_row

        return commands

    def convert_commands(self, flat_rows: List[FurnaceRow], module: FurnaceModule) -> List[MMLCommand]:
        # process rows into commands
        commands: List[MMLCommand] = []

        commands.extend(self.convert_volume_commands(flat_rows, module))
        commands.extend(self.convert_pan_commands(flat_rows, module))
        commands.extend(self.convert_other_commands(flat_rows, module))

        # sort commands by tick
        sorted_commands = sorted(commands, key=lambda x: x.tick)

        return sorted_commands