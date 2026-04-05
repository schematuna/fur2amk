from typing import Dict, List, Optional
from dataclasses import dataclass, replace
import logging

from ..model.MMLData import *
from ..model.MMLCommands import *

from ..util import *


################################
# INTERNAL MML WRITER CLASSES  #
################################

# silent instruction to break a tie
# useful for pitchbend commands that need to be placed after the duration to be modulated
@dataclass
class TieBreakCommand(MMLCommand):
    # no spaces around this command
    def add_spaces(self, text: str) -> str:
        return text
    
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return ''

# a silent instruction to break a tie and force the compiler to not optimize it away
# useful for forcing 1-tick ties for removing N-SPC 1-tick gap
@dataclass
class HardTieBreakCommand(MMLCommand):
    # no spaces around this command
    def add_spaces(self, text: str) -> str:
        return text

    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return '|'

@dataclass
class MMLRest:
    tick: int
    duration: int

# A note or rest with its commands
@dataclass
class MMLWord:
    tick: int = field(compare=False)
    duration: int
    note: Optional[int] = None
    commands: List[MMLCommand] = field(default_factory=list)

    def _want_space_after_duration(self, tick: int) -> str:
        # insert a space after the duration if the following tick contains any non-tiebreak commands
        space_after_duration = False
        for cmd in self.commands:
            if cmd.tick == tick and not isinstance(cmd, TieBreakCommand) and not isinstance(cmd, HardTieBreakCommand):
                space_after_duration = True
                break

        return space_after_duration
    
    def _want_space_after_pitchbend(self, tick: int, ensuing_commands: List[MMLCommand]) -> str:
        # insert a space after the pitchbend if there's another command on the same tick
        space_after_pitchbend = False
        for cmd in ensuing_commands:
            if cmd.tick == tick and not isinstance(cmd, PitchBend) and not isinstance(cmd, TieBreakCommand) and not isinstance(cmd, HardTieBreakCommand):
                space_after_pitchbend = True
                break

        return space_after_pitchbend

    def to_mml(self, mml_state: MMLState) -> str:
        word_txt = ''
        command_idx = 0
        cur_tick = self.tick
        # sort commands by tick, since we will iterate through them in order
        self.commands = sorted(self.commands, key=lambda cmd : cmd.tick)
        
        # process any pre-note commands (at the same tick as note start)
        while command_idx < len(self.commands) and self.commands[command_idx].tick == self.tick:
            word_txt += self.commands[command_idx].get_text(mml_state)
            command_idx += 1

        # add note name and octave
        if self.note is not None:
            note_name, note_octave = MMLUtil.note_name_and_octave(self.note)
            # Emit octave change if needed
            if mml_state.octave != note_octave:
                word_txt += f'o{note_octave} '
                mml_state.octave = note_octave
            
            word_txt += note_name
        else:
            word_txt += 'r'

        # Add initial duration (before any remaining commands)
        cont = False
        if command_idx < len(self.commands):
            first_cmd = self.commands[command_idx]
            first_cmd_tick = first_cmd.tick
            if first_cmd_tick > cur_tick:
                word_txt += DurationFormatter.format(first_cmd_tick - cur_tick, cont, isinstance(first_cmd, PitchBend))
                if self._want_space_after_duration(first_cmd_tick):
                    word_txt += ' '
                cur_tick = first_cmd_tick
                cont = True

        # Interleave commands with duration
        while command_idx < len(self.commands):
            command = self.commands[command_idx]
            cmd_tick = command.tick
            word_txt += command.get_text(mml_state)
            if isinstance(command, PitchBend) and command_idx + 1 < len(self.commands):
                # If there's another command on this tick after the pitchbend command, we need a space
                # if not, we don't want a space
                if self._want_space_after_pitchbend(cmd_tick, self.commands[command_idx+1:]):
                    word_txt += ' '
            command_idx += 1
            
            # Update cur_tick to this command's tick
            cur_tick = cmd_tick
            
            # Add duration to next command
            if command_idx < len(self.commands):
                next_cmd = self.commands[command_idx]
                next_cmd_tick = next_cmd.tick
                if next_cmd_tick > cur_tick:
                    word_txt += DurationFormatter.format(next_cmd_tick - cur_tick, cont, isinstance(next_cmd, PitchBend))
                    if self._want_space_after_duration(next_cmd_tick):
                        word_txt += ' '
                    cur_tick = next_cmd_tick
                    cont = True

        # Add remaining duration to end of note
        end_tick = self.tick + self.duration
        if cur_tick < end_tick:
            word_txt += DurationFormatter.format(end_tick - cur_tick, cont)

        return word_txt

class MMLSentence:
    def __init__(self, words: List[MMLWord]) -> None:
        self.words = words
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, MMLSentence):
            return False
        return self.words == other.words

    def to_mml(self, mml_state: MMLState) -> str:
        sentence_txt = ''
        # sort words by tick, since we will iterate through them in order
        self.words = sorted(self.words, key=lambda word : word.tick)
        for word in self.words:
            sentence_txt += word.to_mml(mml_state) + ' '
        return sentence_txt.rstrip()

# loop info for a group of sentences which may be:
#   - an unlooped set of sentences
#   - a looped set of sentences
#   - a labelled set of sentences
#   - a standalone loop label
@dataclass
class LoopInfo:
    # indices of the sentences that are in this group
    sentenceIndices: List[int]
    # index of label if this group has a label
    label: int = None
    # whether this is a repeat of a previous loop, in which case only the label will be shown
    isRepeat: int = False
    # how many times this group is looped, if at all
    numLoops: int = None

# a segment of MML usually representing a musical section
# the MMLSection will be labelled in the MML with the section number
# loops cannot cross section boundaries
class MMLSection:
    def __init__(self, words: List[MMLWord], section_num: int, measure_length: int) -> None:
        self.section_num = section_num
        # metadata representing the looping structure of an MMLSection
        self.loopInfo: List[LoopInfo] = []
        self.MAX_CHARS_PER_LINE = 80
        self.MIN_CHARS_PER_LINE = 10
        self.sentences: List[MMLSentence] = []
        self.make_sentences(words, measure_length)
        self.logger = logging.getLogger(__name__)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, MMLSection):
            return False
        return self.sentences == other.sentences

    def tick(self) -> int:
        if len(self.sentences) == 0:
            self.logger.warning(f"Line has no sentences. section: {self.section_num}")
            return 0
        if len(self.sentences[0].words) == 0:
            self.logger.warning(f"Line has no words. section: {self.section_num}")
            return 0
        return self.sentences[0].words[0].tick

    # convert all sentences without accounting for loops
    # useful for pre-looped formatting logic
    def convert_sentences_raw(self, mml_state: MMLState) -> List[str]:
        line_txt = ''
        for sentence in self.sentences:
            line_txt += sentence.to_mml(mml_state) + '\n'

        return line_txt.rstrip()

    # measure-informed sentence splitting to avoid overlong lines
    def make_sentences(self, words: List[MMLWord], measure_length: int) -> None:
        self.sentences: List[MMLSentence] = [MMLSentence(words)]
        mml_state = MMLState()
        if len(self.convert_sentences_raw(mml_state)) > self.MAX_CHARS_PER_LINE:
            self.sentences: List[MMLSentence] = []
            cur_measure_num = words[0].tick // measure_length
            cur_words: List[MMLWord] = []
            for word in words:
                measureNum = word.tick // measure_length
                if measureNum != cur_measure_num:
                    next_sentence = MMLSentence(cur_words)
                    # don't split line if it's too short
                    if not (len(next_sentence.to_mml(mml_state)) < self.MIN_CHARS_PER_LINE):
                        self.sentences.append(next_sentence)
                        cur_words = []
                    cur_measure_num = measureNum
                cur_words.append(word)
            next_sentence = MMLSentence(cur_words)
            if not (len(next_sentence.to_mml(mml_state)) < self.MIN_CHARS_PER_LINE):
                self.sentences.append(next_sentence)
            else:
                self.sentences[-1].words.extend(cur_words)

        # Break up any sentences that are still too long, splitting along beat boundaries
        i = 0
        while i < len(self.sentences):
            sentence = self.sentences[i]
            if len(sentence.to_mml(mml_state)) > self.MAX_CHARS_PER_LINE:
                # Split sentence along beat boundaries
                new_sentences = self._split_sentence_by_beats(sentence, mml_state)
                # If we couldn't split (all words in same beat), keep original and move on
                if len(new_sentences) <= 1:
                    i += 1
                else:
                    # Replace the original sentence with the split ones
                    self.sentences[i:i+1] = new_sentences
                    # Continue checking from the same index (don't increment) in case new sentences are also too long
            else:
                i += 1
    
    def _split_sentence_by_beats(self, sentence: MMLSentence, mml_state: MMLState) -> List[MMLSentence]:
        """Split a sentence into smaller sentences along beat boundaries."""
        if not sentence.words:
            return [sentence]
        
        new_sentences: List[MMLSentence] = []
        cur_words: List[MMLWord] = []
        cur_beat_num = None
        
        for word in sentence.words:
            beat_num = word.tick // MMLUtil.AMK_TICKS_PER_BEAT
            
            # If we hit a new beat and have accumulated words, start a new sentence
            if cur_beat_num is not None and beat_num != cur_beat_num and cur_words:
                new_sentence = MMLSentence(cur_words)
                # Only split if the sentence would still be too long
                # If splitting here would make it too short, continue accumulating
                if len(new_sentence.to_mml(mml_state)) >= self.MIN_CHARS_PER_LINE:
                    new_sentences.append(new_sentence)
                    cur_words = []
            
            cur_beat_num = beat_num
            cur_words.append(word)
        
        # Add the final group of words
        if cur_words:
            final_sentence = MMLSentence(cur_words)
            # If final sentence would be too short and we have previous sentences, merge with previous
            if len(final_sentence.to_mml(mml_state)) < self.MIN_CHARS_PER_LINE and new_sentences:
                # Merge final sentence with the previous one
                new_sentences[-1].words.extend(cur_words)
            else:
                new_sentences.append(final_sentence)
        
        # If we couldn't split (all words in one beat), return original sentence
        if len(new_sentences) <= 1:
            return [sentence]
        
        return new_sentences
    
    def to_mml(self, mml_state: MMLState) -> str:
        line_txt = ''
        for info in self.loopInfo:
            if info.label:
                if info.isRepeat:
                    line_txt += f"({info.label})"
                else:
                    line_txt += f"({info.label})[\n"
                    for idx in info.sentenceIndices:
                        line_txt += self.sentences[idx].to_mml(mml_state) + '\n'
                    line_txt += "]"
            else:
                for idx in info.sentenceIndices:
                    line_txt += self.sentences[idx].to_mml(mml_state) + '\n'

        # don't want last newline, strip it
        return line_txt.rstrip()

class MMLWriter:
    def __init__(self, mml_data: MMLData, label_start: int) -> None:
        self.logger = logging.getLogger(__name__)
        self.mml_data = mml_data
        self.label_count = label_start

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

    def optimize_loops(self, sections: List[MMLSection], label_count: int) -> int:
        # Identify and label loops in the channel lines
        labels_assigned: Dict[int, MMLSection] = {}
        unique_sections: Dict[int, MMLSection] = {}
        for i, section in enumerate(sections):
            # Check for repeated patterns
            if section not in unique_sections.values():
                unique_sections[i] = section
                section.loopInfo = [LoopInfo(range(len(section.sentences)))]
            elif section not in labels_assigned.values():
                # Assign a label to this repeated pattern
                labels_assigned[label_count] = section
                section.loopInfo = [LoopInfo(range(len(section.sentences)), label_count, True)]
                # and mark the first occurrence
                for order, section2 in unique_sections.items():
                    if section2 == section:
                        sections[order].loopInfo[0].label = label_count
                        break
                label_count += 1
            else:
                # Find the existing label for this pattern
                for lbl, section2 in labels_assigned.items():
                    if section2 == section:
                        section.loopInfo = [LoopInfo(range(len(section.sentences)), lbl, True)]
                        break
        return label_count

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

            if first_note_after_loop:
                break

            # track remote command state
            for command in word.commands:
                if isinstance(command, RemoteCommand):
                    if command.timing is RemoteCommandTiming.DISABLE:
                        cur_remote_commands = []
                    else:
                        cur_remote_commands.append(command)

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

            self.label_count = self.optimize_loops(sections, self.label_count)

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