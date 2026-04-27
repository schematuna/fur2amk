from typing import List, Optional, Tuple
import logging

from ..model.FurnaceData import *
from ..model.MMLCommands import *
from ..model.MMLData import *
from ..model.ChiptuneData import *
from ..model.AMKData import *
from .ConverterUtil import *
from ..util.MMLUtil import *

# conversion helper data class
@dataclass
class InstrumentInfo:
    # list of remote command indices associated with this instrument
    remote_commands: List[AMKRemoteDef] = field(default_factory=list)

# Convert notes and tightly-coupled commands
# This includes:
# - pitch slides, which must be contained within a note
# - envelope-related remote commands
# - remove 1-tick-gap
class NoteConverter():
    def __init__(self, tick_ratio: float) -> None:
        self.tick_ratio = tick_ratio
        self.pitch_slider = PitchSlider(tick_ratio, 0)
        #init logger
        self.logger = logging.getLogger(__name__)

    def get_pre_note_commands(self, chip_ins: ChiptuneInstrument, ins_info: Dict[int, InstrumentInfo], state: FurnaceState, note_tick: int) -> List[MMLCommand]:
        # if new instrument, set up remote commands for this instrument
        remote_commands = []
        if chip_ins.index in ins_info:
            for remote_cmd in ins_info[chip_ins.index].remote_commands:
                if remote_cmd.wait_ticks is not None:
                    remote_commands.append(RemoteCommand(note_tick, remote_cmd.command_idx, remote_cmd.timing, remote_cmd.wait_ticks))
                else:
                    remote_commands.append(RemoteCommand(note_tick, remote_cmd.command_idx, remote_cmd.timing))

        if len(remote_commands) > 2:
            self.logger.warning(f"Too many remote commands for instrument {chip_ins.index}, only 2 can be active at a time (one key on and one other)")

        # TODO: use (!!) syntax and only stop events that need to be stopped
        # Only emit remote command if it changed
        pre_note_commands = []
        if remote_commands != state.remote_commands and len(state.remote_commands) > 0:
            pre_note_commands.append(RemoteCommand(note_tick, 99, RemoteCommandTiming.DISABLE))

        if len(remote_commands) > 0:
            pre_note_commands.extend(remote_commands)

        state.remote_commands = remote_commands

        return pre_note_commands

    def convert_pitch_slides(self, tick: int, row: ChiptuneTickData, slide_helper: PitchSlider) -> List[TempPitchBend]:
        commands: List[MMLCommand] = []
        if effect := row.get_command(NoteSlideCommand):
            # for note slides, each speed unit is 4 pitch steps per tick
            ticks_to_slide = FurnaceUtil.ticks_from_speed(effect.speed * 4, abs(effect.semitones))
            amk_duration = max(2, int(ticks_to_slide * self.tick_ratio))
            max_duration = slide_helper.get_max_duration()
            if amk_duration > max_duration:
                self.logger.warning(f"Pitch slide duration {amk_duration} is greater than the longest tick duration {max_duration}.")
            commands.append(TempPitchBend(tick, amk_duration, effect.semitones))
            slide_helper.set_target(0)

        if effect := row.get_command(PitchSlideCommand):
            if new_command := slide_helper.handle_new_effect(effect):
                commands.append(new_command)
                slide_helper.set_target(0)
                        
        if new_command := slide_helper.tick(1):
            commands.append(new_command)
            slide_helper.set_target(0)

        return commands

    # get notes and pitch-related commands
    def convert(self, ticks: List[ChiptuneTickData], ins_info: Dict[int, InstrumentInfo], instruments: List[ChiptuneInstrument]) -> Tuple[List[MMLNote], List[MMLCommand], List[TempPitchBend]]:
        notes: List[MMLNote] = []
        commands: List[MMLCommand] = []
        pitchbends: List[TempPitchBend] = []
        tick = 0
        state = FurnaceState()
        slide_helper = PitchSlider(self.tick_ratio, tick)

        # the current note duration
        cur_dur: Optional[MMLNote] = None
        # active chiptune instrument
        chip_ins = None
        # process notes and pitch commands
        for tick_data in ticks:
            note_kind = tick_data.kind()
            if note_kind == ChiptuneTickData.NoteKind.NOTE:   
                new_chip_ins = None
                for ins in instruments:
                    if ins.index == tick_data.Ins:
                        new_chip_ins = ins
                        break

                if new_chip_ins is not None:
                    chip_ins = new_chip_ins

                if chip_ins is None:
                    self.logger.error(f"No instrument active in row with Note {tick_data.Note}.")
                    continue

                pre_note_commands = []
                # we only have to set up pre-note commands for new instruments
                if chip_ins.index != state.fur_ins_idx:
                    pre_note_commands = self.get_pre_note_commands(chip_ins, ins_info, state, tick)
                    state.fur_ins_idx = chip_ins.index   

                pitch_command = None
                if cur_dur is not None:
                    cur_dur.duration = tick - cur_dur.tick
                    pitch_command = slide_helper.end_slide(None)
                    if pitch_command is not None:
                        pitchbends.append(pitch_command)
                    notes.append(cur_dur)
                
                cur_dur = MMLNote(tick, 0, tick_data.Note, chip_ins.index, pre_note_commands)

                slide_helper.set_target(0)
                # if this note interrupted a pitch slide, start a new one
                # unless there is a pitch slide command on this row, in which case we'll let that handle it
                if pitch_command is not None and not tick_data.get_command(PitchSlideCommand) and not tick_data.get_command(NoteSlideCommand):
                    slide_helper.start_slide()

            elif note_kind == ChiptuneTickData.NoteKind.RELEASE:
                if chip_ins is not None and chip_ins.sn_envelope_on and chip_ins.sustain_mode == SustainMode.DELAYED:
                    # the delayed adsr mode sets the release time to the release value on key off
                    adsr = ADSR(chip_ins.sn_attack, chip_ins.sn_decay, chip_ins.sn_sustain, chip_ins.sn_release)
                    commands.append(CustomADSR(tick, adsr))
                    state.adsr = adsr
                else:
                    # finish any pitch slides that are still active
                    pitch_command = slide_helper.end_slide(None)
                    if cur_dur is not None:
                        if pitch_command is not None:
                            pitchbends.append(pitch_command)
                        cur_dur.duration = tick - cur_dur.tick
                        notes.append(cur_dur)
                    else:
                        self.logger.debug(f"Note off or release was found but no note was playing.")
                        if pitch_command is not None:
                            self.logger.warning("Lost pitch slide on note off.")
                    cur_dur = None

            if effect := tick_data.get_command(SendExternalCommand):
                if effect.value == 0 and cur_dur is not None:
                    cur_dur.no_gap = True
                else:
                    self.logger.warning("Send external effect found outside of a note, ignoring.")

            # handle pitch slides after processing this row's note info
            pitch_commands = self.convert_pitch_slides(tick, tick_data, slide_helper)
            if cur_dur is not None:
                pitchbends.extend(pitch_commands)

            # increment tick before next tick
            tick += 1
        
        # Finalize possible final note
        if cur_dur is not None:
            cur_dur.duration = tick - cur_dur.tick
            notes.append(cur_dur)
            pitch_command = slide_helper.end_slide(None)
            if pitch_command is not None:
                pitchbends.append(pitch_command)

        return notes, commands, pitchbends