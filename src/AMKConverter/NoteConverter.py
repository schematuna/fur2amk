from typing import List, Optional, Tuple
import logging

from ..model.FurnaceData import *
from ..model.MMLCommands import *
from ..model.MMLData import *
from ..model.ChiptuneData import *
from ..model.AMKData import *
from .ConverterUtil import *
from ..util.MMLUtil import *

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
    remote_commands: List[AMKRemoteDef] = field(default_factory=list)

# Convert notes and tightly-coupled commands
# This includes:
# - pitch slides, which must be contained within a note
# - envelope-related remote commands
class NoteConverter():
    def __init__(self, tick_ratio: float) -> None:
        self.tick_ratio = tick_ratio
        self.pitch_slider = PitchSlider(tick_ratio, 0)
        #init logger
        self.logger = logging.getLogger(__name__)

    def get_pre_note_commands(self, fur_ins: FurnaceInstrument, ins_info: Dict[int, InstrumentInfo], state: FurnaceState, note_tick: int) -> List[MMLCommand]:
        # instrument echo
        pre_note_commands = []
        if fur_ins.snes_macro_data.is_echo != state.is_echo:
            pre_note_commands.append(EchoToggle(note_tick))
            state.is_echo = fur_ins.snes_macro_data.is_echo

        # if new instrument, set up remote commands for this instrument
        remote_commands = []
        if fur_ins.index in ins_info:
            for remote_cmd in ins_info[fur_ins.index].remote_commands:
                if remote_cmd.wait_ticks is not None:
                    remote_commands.append(RemoteCommand(note_tick, remote_cmd.command_idx, remote_cmd.timing, remote_cmd.wait_ticks))
                else:
                    remote_commands.append(RemoteCommand(note_tick, remote_cmd.command_idx, remote_cmd.timing))

        if len(remote_commands) > 2:
            self.logger.warning(f"Too many remote commands for Furnace instrument {fur_ins.index}, only 2 can be active at a time (one key on and one other)")

        # TODO: use (!!) syntax and only stop events that need to be stopped
        # Only emit remote command if it changed
        if remote_commands != state.remote_commands and len(state.remote_commands) > 0:
            pre_note_commands.append(RemoteCommand(note_tick, 99, RemoteCommandTiming.DISABLE))

        if len(remote_commands) > 0:
            pre_note_commands.extend(remote_commands)

        state.remote_commands = remote_commands

        return pre_note_commands

    def get_note_info(self, fur_ins_idx: int, note: int, ins_info: Dict[int, InstrumentInfo], use_sample_map: bool) -> Tuple[int, int]:
        note_to_play = note
        amk_ins_idx = None
        if fur_ins_idx not in ins_info:
            self.logger.error(f"No instrument info found for Furnace instrument {fur_ins_idx}, this is not right.")
            return None, None
        # Get the AMK instrument index using ins_map
        if use_sample_map:
            note_map = ins_info[fur_ins_idx].ins_map
            # Try to find exact note match first
            if note in note_map:
                amk_ins_idx = note_map[note].amk_ins_idx
                note_to_play = note_map[note].note_to_play
            else:
                # No mapping found, use Furnace instrument index as fallback
                self.logger.warning(f"No instrument mapping found for Furnace instrument {fur_ins_idx}, note {note}.")
                amk_ins_idx = 0
        else:
            amk_ins_idx = ins_info[fur_ins_idx].amk_ins

        return note_to_play, amk_ins_idx

    def convert_portamento(self, tick: int, active_note: int, target_note: int, speed: int) -> Tuple[int, MMLCommand]:
        semitones = target_note - active_note
        ticks_to_slide = FurnaceUtil.ticks_from_speed(speed, semitones)
        amk_duration = max(2, int(ticks_to_slide * self.tick_ratio))
        max_duration = PitchSlider.get_max_duration()
        if amk_duration > max_duration:
            self.logger.warning(f"Portamento duration {amk_duration} is greater than the longest tick duration {max_duration}.")
        command = PitchBend(tick, amk_duration, target_note)

        return target_note, command

    def convert_pitch_slides(self, tick: int, row: FurnaceRow, slide_helper: PitchSlider, active_note: Optional[int]) -> Tuple[Optional[int], List[MMLCommand]]:
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
                        
        if new_command := slide_helper.tick(1):
            new_active_note = new_command.note
            commands.append(new_command)

        return new_active_note, commands

    # get notes and pitch-related commands
    def convert(self, ticks: List[TickData], ins_info: Dict[int, InstrumentInfo], instruments: List[FurnaceInstrument]) -> Tuple[List[MMLNote], List[MMLCommand]]:
        notes: List[MMLNote] = []
        commands: List[MMLCommand] = []
        tick = 0
        state = FurnaceState()
        slide_helper = PitchSlider(self.tick_ratio, tick)

        # the current note duration
        cur_dur: Optional[MMLNote] = None
        # the current active pitch, considering both pitch commands and explicit notes
        active_note = None
        # active furnace instrument
        fur_ins = None
        # process notes and pitch commands
        for tick_data in ticks:
            # handle portamento before reading this row's note info
            has_portamento = False
            if effect := tick_data.get_effect(PortamentoEffect):
                has_portamento = True
                if tick_data.kind() == TickData.NoteKind.NOTE:
                    active_note, portamento_command = self.convert_portamento(tick, active_note, tick_data.Note, effect.speed)
                    if portamento_command is not None:
                        if cur_dur is not None:
                            cur_dur.pitch_bends.append(portamento_command)
                        else:
                            self.logger.warning("no active note to portamento from, ignoring portamento command.")
                else:
                    self.logger.warning("Portamento effect found on non-note row, ignoring.")
                
            # don't make a new note for portamento rows, pitchbend will handle that
            note_kind = tick_data.kind()
            if note_kind == TickData.NoteKind.NOTE and not has_portamento:                    
                new_fur_ins = None
                for ins in instruments:
                    if ins.index == tick_data.Ins:
                        new_fur_ins = ins
                        break

                if new_fur_ins is not None:
                    fur_ins = new_fur_ins

                if fur_ins is None:
                    self.logger.error(f"No furnace instrument active in row with Note {tick_data.Note}.")
                    continue

                pre_note_commands = []
                # we only have to set up pre-note commands for new instruments
                if fur_ins.index != state.fur_ins_idx:
                    pre_note_commands = self.get_pre_note_commands(fur_ins, ins_info, state, tick)
                    state.fur_ins_idx = fur_ins.index   

                note_to_play, amk_ins_idx = self.get_note_info(fur_ins.index, tick_data.Note, ins_info, fur_ins.use_sample_map)
                if note_to_play is None or amk_ins_idx is None:
                    self.logger.error(f"No note to play or AMK instrument index found for Furnace instrument {fur_ins.index}, note {tick_data.Note}.")
                    continue

                pitch_command = None
                if cur_dur is not None:
                    cur_dur.duration = tick - cur_dur.tick
                    pitch_command = slide_helper.end_slide(None)
                    if pitch_command is not None:
                        cur_dur.pitch_bends.append(pitch_command)
                    notes.append(cur_dur)
                cur_dur = MMLNote(tick, 0, note_to_play, amk_ins_idx, pre_note_commands)
                active_note = note_to_play

                slide_helper.set_target(active_note)
                # if this note interrupted a pitch slide, start a new one
                # unless there is a pitch slide effect on this row, in which case we'll let that handle it
                if pitch_command is not None and not tick_data.get_effect(PitchSlideEffect):
                    slide_helper.start_slide()

            elif note_kind == TickData.NoteKind.RELEASE:
                if fur_ins is not None and fur_ins.sn_envelope_on and fur_ins.sustain_mode == SustainMode.DELAYED:
                    # the delayed adsr mode sets the release time to the release value on key off
                    adsr = ADSR(fur_ins.sn_attack, fur_ins.sn_decay, fur_ins.sn_sustain, fur_ins.sn_release)
                    commands.append(CustomADSR(tick, adsr))
                    state.adsr = adsr
                else:
                    # finish any pitch slides that are still active
                    pitch_command = slide_helper.end_slide(None)
                    if cur_dur is not None:
                        if pitch_command is not None:
                            cur_dur.pitch_bends.append(pitch_command)
                        cur_dur.duration = tick - cur_dur.tick
                        notes.append(cur_dur)
                    else:
                        self.logger.debug(f"Note off or release was found but no note was playing.")
                        if pitch_command is not None:
                            self.logger.warning("Lost pitch slide on note off.")
                    cur_dur = None
                    active_note = None

            if effect := tick_data.get_effect(SendExternalEffect):
                if effect.value == 0 and cur_dur is not None:
                    cur_dur.no_gap = True
                else:
                    self.logger.warning("Send external effect found outside of a note, ignoring.")

            # handle pitch slides after processing this row's note info
            active_note, pitch_commands = self.convert_pitch_slides(tick, tick_data, slide_helper, active_note)
            if cur_dur is not None:
                cur_dur.pitch_bends.extend(pitch_commands)

            # increment tick before next tick
            tick += 1
        
        # Finalize possible final note
        if cur_dur is not None:
            cur_dur.duration = tick - cur_dur.tick
            notes.append(cur_dur)

        return notes, commands