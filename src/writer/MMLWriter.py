from typing import List
from dataclasses import replace
import logging

from ..model.MMLData import *
from ..model.MMLCommands import *
from .LoopOptimizer import *
from .MMLWriterData import *

from ..util import *

class MMLWriter:
    def __init__(self, mml_data: MMLData, label_start: int, optimize_loops: bool = True) -> None:
        self.logger = logging.getLogger(__name__)
        self.mml_data = mml_data
        self.label_count = label_start
        self.optimize_loops = optimize_loops

    def get_section_num(self, tick: int) -> int:
        """
        Calculate which section (order) a given tick falls into.
        Accounts for variable-length sections from pattern jump commands.
        """
        if not self.mml_data.section_lengths:
            return 0

        accumulated_ticks = 0
        for section_idx, section_ticks in enumerate(self.mml_data.section_lengths):
            if tick < accumulated_ticks + section_ticks:
                return section_idx
            accumulated_ticks += section_ticks

        # If tick is beyond all sections, return the last section
        return len(self.mml_data.section_lengths) - 1

    def split_durations_at_loop(self, durations: List) -> List:
        """Split any duration that spans the loop point into two parts."""
        loop_point = self.mml_data.loop_tick
        if loop_point is None:
            return durations

        result = []
        for dur in durations:
            if dur.tick < loop_point < dur.tick + dur.duration:
                first_len = loop_point - dur.tick
                second_len = dur.tick + dur.duration - loop_point
                if isinstance(dur, MMLRest):
                    result.append(MMLRest(dur.tick, first_len))
                    result.append(MMLRest(loop_point, second_len))
                else:
                    result.append(replace(dur, duration=first_len))
                    result.append(MMLRest(loop_point, second_len))
            else:
                result.append(dur)
        return result

    # get rests between notes
    def get_rests(self, notes: List[MMLNote]) -> List[MMLRest]:
        rests: List[MMLRest] = []
        # special case for no notes
        if len(notes) == 0:
            rests.append(MMLRest(0, self.mml_data.song_length))
            return rests
        # add initial rest if there is one
        if notes[0].tick > 0:
            rests.append(MMLRest(0, notes[0].tick))
        # add rests between notes
        for i, note in enumerate(notes):
            if i + 1 < len(notes) and notes[i+1].tick > note.tick + note.duration:
                rest_duration = notes[i+1].tick - (note.tick + note.duration)
                rests.append(MMLRest(note.tick + note.duration, rest_duration))
        # add final rest if there is one
        if notes[-1].tick + notes[-1].duration < self.mml_data.song_length:
            rest_duration = self.mml_data.song_length - (notes[-1].tick + notes[-1].duration)
            rests.append(MMLRest(notes[-1].tick + notes[-1].duration, rest_duration))
        return rests

    def make_words(self, notes: List[MMLNote], commands: List[MMLCommand]) -> List[MMLWord]:
        words: List[MMLWord] = []
        # sort before iterating
        notes = sorted(notes, key=lambda note : note.tick)
        commands = sorted(commands, key=lambda cmd : cmd.tick)

        cur_ins = None
        cmd_idx = 0
        rests = self.get_rests(notes)
        durations: List[MMLNote | MMLRest] = sorted(notes + rests, key=lambda dur : dur.tick)
        # can't have the loop point in the middle of a duration
        durations = self.split_durations_at_loop(durations)
        for duration in durations:
            if isinstance(duration, MMLRest):
                word = MMLWord(duration.tick, duration.duration, None)
            else:
                word = MMLWord(duration.tick, duration.duration, duration.note, duration.pre_note_commands)

                if duration.instrument != cur_ins:
                    word.commands.append(InstrumentChange(duration.tick, duration.instrument))
                    cur_ins = duration.instrument

                for pitch_bend in duration.pitch_bends:
                    # check that pitchbend is contained within this note
                    if pitch_bend.tick < duration.tick:
                        self.logger.warning(f"Pitchbend starts before the note. Ignoring. section: {self.get_section_num(pitch_bend.tick)}")
                        continue
                    if pitch_bend.tick + pitch_bend.duration > duration.tick + duration.duration:
                        self.logger.warning(f"Pitchbend duration exceeds the duration of the note. Trimming to fit.")
                        pitch_bend.duration = duration.tick + duration.duration - pitch_bend.tick
                    # special pitchbend handling
                    if pitch_bend.tick > duration.tick:
                        # place silent tiebreak command if pitchbend starts in the middle of a duration
                        word.commands.append(TieBreakCommand(pitch_bend.tick))
                    # Pitchbend commands are placed after the duration to be modulated
                    pitch_bend.tick = pitch_bend.tick + pitch_bend.duration
                    word.commands.append(pitch_bend)

                # handle no-gap notes. Not compatible with pitchbends.
                if duration.no_gap and len(duration.pitch_bends) == 0:
                    # place hard tiebreak command 1 tick before the end of the note to remove 1-tick gap before next note
                    word.commands.append(HardTieBreakCommand(duration.tick + duration.duration - 1))
                    
            while cmd_idx < len(commands):
                cmd_tick = commands[cmd_idx].tick
                if cmd_tick >= duration.tick and cmd_tick < duration.tick + duration.duration:
                    word.commands.append(commands[cmd_idx])
                    cmd_idx += 1
                else:
                    break

            words.append(word)
        return words

    def get_loop_state_commands(self, words: List[MMLWord]) -> tuple[List[MMLCommand], List[MMLCommand]]:
        """
        Get commands needed to handle state at the loop point.

        Returns:
            Tuple of (pre_loop_commands, post_loop_commands)
            - pre_loop_commands: Commands to emit just before the loop point
            - post_loop_commands: Commands to emit just after the loop point
        """
        # If there are remote commands after the loop point, we need to reset them at the loop point
        has_remote_commands_after_loop = False
        for word in words:
            for cmd in word.commands:
                if isinstance(cmd, RemoteCommand):
                    is_post_loop = self.mml_data.loop_tick is None or cmd.tick >= self.mml_data.loop_tick
                    if is_post_loop:
                        has_remote_commands_after_loop = True
                        break

        pre_loop_commands = []
        post_loop_commands = []

        # state tracking
        cur_ins = None
        cur_remote_commands = []
        cur_pitch_envelope: PitchEnvelope | None = None
        legato_on = False
        legato_on_at_loop = False  # was legato on when we crossed the loop point?
        echo_toggled = False
        echo_toggled_at_loop = False  # was echo toggled when we crossed the loop point?

        for word in words:
            is_post_loop = self.mml_data.loop_tick is None or word.tick >= self.mml_data.loop_tick
            first_note_after_loop = is_post_loop and word.note is not None

            # Capture toggle states at first word after loop
            if first_note_after_loop:
                legato_on_at_loop = legato_on
                echo_toggled_at_loop = echo_toggled

            # Track toggle states
            for command in word.commands:
                if isinstance(command, LegatoToggle):
                    legato_on = not legato_on
                if isinstance(command, EchoToggle):
                    echo_toggled = not echo_toggled

            # need to explicitly handle instrument change at loop point, so it's correct on loop
            if first_note_after_loop:
                if cur_ins is not None:
                    post_loop_commands.append(InstrumentChange(word.tick, cur_ins))

            # also handle remote command state for loop
            if first_note_after_loop and has_remote_commands_after_loop:
                # could filter these out if they're already in the word, but will let it be simple for now
                post_loop_commands.insert(0, RemoteCommand(word.tick, 99, RemoteCommandTiming.DISABLE))
                for cmd in cur_remote_commands:
                    post_loop_commands.append(replace(cmd, tick=word.tick))

            if first_note_after_loop and cur_pitch_envelope is not None:
                post_loop_commands.append(replace(cur_pitch_envelope, tick=word.tick))

            if first_note_after_loop:
                break

            # track remote command state
            for command in word.commands:
                if isinstance(command, RemoteCommand):
                    if command.timing is RemoteCommandTiming.DISABLE:
                        cur_remote_commands = []
                    else:
                        cur_remote_commands.append(command)

            # track pitch envelope state
            for command in word.commands:
                if isinstance(command, PitchEnvelope):
                    cur_pitch_envelope = command
                if isinstance(command, PitchEnvelopeOff):
                    cur_pitch_envelope = None

            # track instrument state
            for command in word.commands:
                if isinstance(command, InstrumentChange):
                    cur_ins = command.instrument_index

        # Handle legato state at loop point
        # If legato is on in the intro as it passes the loop point, we need to:
        # 1. Turn it off before the loop point (pre_loop_commands)
        # 2. Turn it back on after the loop point (post_loop_commands)
        if legato_on_at_loop and self.mml_data.loop_tick is not None:
            pre_loop_commands.append(LegatoToggle(self.mml_data.loop_tick - 1))
            post_loop_commands.insert(0, LegatoToggle(self.mml_data.loop_tick))

        # Handle echo state at loop point
        # If echo is toggled from channel default in the intro as it passes the loop point, we need to
        # toggle it before and after the loop point
        if echo_toggled_at_loop and self.mml_data.loop_tick is not None:
            pre_loop_commands.append(EchoToggle(self.mml_data.loop_tick - 1))
            post_loop_commands.insert(0, EchoToggle(self.mml_data.loop_tick))

        # TODO: Handle pitch envelope state at loop point

        return pre_loop_commands, post_loop_commands


    def write(self) -> None:
        has_loop_point = self.mml_data.loop_tick not in [None, 0]

        txt = ''
        for c in range(self.mml_data.num_channels):
            if len(self.mml_data.notes[c]) == 0 and len(self.mml_data.commands[c]) == 0:
                continue
            word_txt = ''
            txt += f'#{c}\n'

            # A "word" is a note or rest with its commands
            words = self.make_words(self.mml_data.notes[c], self.mml_data.commands[c])
            # sort again for good measure
            words = sorted(words, key=lambda words : words.tick)

            pre_loop_commands, post_loop_commands = self.get_loop_state_commands(words)

            sections: List[MMLSection] = []
            section_words: List[MMLWord] = []
            cur_section_num = 0
            for word in words:
                sectionNum = self.get_section_num(word.tick)
                # split line at loop point
                is_loop_point = has_loop_point and word.tick == self.mml_data.loop_tick
                if sectionNum != cur_section_num or is_loop_point:
                    sections.append(MMLSection(section_words, cur_section_num, self.mml_data.measure_length))
                    cur_section_num = sectionNum
                    section_words = []
                section_words.append(word)

            sections.append(MMLSection(section_words, cur_section_num, self.mml_data.measure_length))

            optimizer = LoopOptimizer()
            if self.optimize_loops:
                self.label_count = optimizer.label_repeated_sections(sections, self.label_count)
                optimizer.optimize_subloops(sections)
                self.label_count = optimizer.optimize_loops(sections, self.label_count)
            else:
                for section in sections:
                    section.loopInfo = [LoopInfo(list(range(len(section.sentences))))]

            mml_state = MMLState()
            # light staccato is a global toggle
            # enable it by default for all ports to reduce unwanted space between notes from 2 ticks to 1 tick
            if c == 0:
                staccato_cmd = LightStaccatoToggle(0)
                txt += "; enable light staccato\n"
                txt += staccato_cmd.get_text(mml_state) + '\n'

            def get_commands_text(commands: List[MMLCommand], comment: str) -> str:
                txt = ''
                if len(commands) > 0:
                    txt += f'; {comment}\n'
                    for cmd in commands:
                        txt += cmd.get_text(mml_state)
                    txt += '\n'
                return txt

            for i, section in enumerate(sections):
                if not has_loop_point and i == 0:
                    word_txt += get_commands_text(post_loop_commands, 'reset state on loop')
                elif has_loop_point and section.tick() == self.mml_data.loop_tick:
                    word_txt += get_commands_text(pre_loop_commands, 'reset state before loop')
                    word_txt += '/\n'
                    word_txt += get_commands_text(post_loop_commands, 'reset state on loop')
                word_txt += f"; section {MMLUtil.to_hex(section.section_num)}\n"
                word_txt += section.to_mml(mml_state) + '\n'

            txt += word_txt + '\n\n'

        # Print warning if any notes were out of range
        MMLUtil.print_out_of_range_warning()

        return txt