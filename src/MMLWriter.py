from typing import Dict, List, Optional
from dataclasses import dataclass, replace
import logging

from .model.MMLData import *
from .model.MMLCommands import *

from .MMLUtil import *


################################
# INTERNAL MML WRITER CLASSES  #
################################

# silent instruction to break a tie
# useful for pitchbend commands that need to be placed after the duration to be modulated
@dataclass
class TieBreakCommand(MMLCommand):
    def to_mml(self, mml_state: 'MMLState' = None) -> str:
        return ''

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
                word_txt += DurationFormatter.format(first_cmd_tick - cur_tick, cont)
                if not isinstance(first_cmd, TieBreakCommand): # don't add space if first command is a tie break
                    word_txt += ' '
                cur_tick = first_cmd_tick
                cont = True
        
        # Interleave commands with duration
        while command_idx < len(self.commands):
            command = self.commands[command_idx]
            cmd_tick = command.tick
            word_txt += command.get_text(mml_state)
            command_idx += 1
            
            # Update cur_tick to this command's tick
            cur_tick = cmd_tick
            
            # Add duration to next command
            if command_idx < len(self.commands):
                next_cmd = self.commands[command_idx]
                next_cmd_tick = next_cmd.tick
                if next_cmd_tick > cur_tick:
                    word_txt += DurationFormatter.format(next_cmd_tick - cur_tick, cont)
                    if not isinstance(next_cmd, TieBreakCommand): # don't add space if next command is a tie break
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

class MMLLine:
    def __init__(self, words: List[MMLWord], section_num: int, measure_length: int) -> None:
        self.section_num = section_num
        self.label: Optional[int] = None
        self.isRepeat: bool = False
        self.MAX_CHARS_PER_LINE = 80
        self.MIN_CHARS_PER_LINE = 10
        self.sentences: List[MMLSentence] = []
        self.make_sentences(words, measure_length)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, MMLLine):
            return False
        return self.sentences == other.sentences

    def tick(self) -> int:
        return self.sentences[0].words[0].tick

    def convert_sentences(self, mml_state: MMLState) -> List[str]:
        line_txt = ''
        for sentence in self.sentences:
            line_txt += sentence.to_mml(mml_state) + '\n'

        return line_txt.rstrip()

    # measure-informed sentence splitting to avoid overlong lines
    def make_sentences(self, words: List[MMLWord], measure_length: int) -> None:
        self.sentences: List[MMLSentence] = [MMLSentence(words)]
        mml_state = MMLState()
        if len(self.convert_sentences(mml_state)) > self.MAX_CHARS_PER_LINE:
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
        if self.label is not None:
            if self.isRepeat:
                line_txt += f"({self.label})"
            else:
                line_txt += f"({self.label})[\n"
                line_txt += self.convert_sentences(mml_state)
                line_txt += "]"
        else:
            line_txt += self.convert_sentences(mml_state)
        return line_txt

class MMLWriter:
    def __init__(self, mml_data: MMLData, label_start: int) -> None:
        self.logger = logging.getLogger(__name__)
        self.mml_data = mml_data
        self.label_count = label_start

    def channel_has_remote_commands(self, channel: int) -> bool:
        for note in self.mml_data.notes[channel]:
            for command in note.pre_note_commands:
                if isinstance(command, RemoteCommand):
                    return True
        return False

    def optimize_loops(self, lines: List[MMLLine], label_count: int) -> int:
        # Identify and label loops in the channel lines
        labels_assigned: Dict[int, MMLLine] = {}
        unique_lines: Dict[int, MMLLine] = {}
        for i, line in enumerate(lines):
            # Check for repeated patterns
            if line not in unique_lines.values():
                unique_lines[i] = line
            elif line not in labels_assigned.values():
                # Assign a label to this repeated pattern
                labels_assigned[label_count] = line
                line.label = label_count
                line.isRepeat = True
                # and mark the first occurrence
                for order, line2 in unique_lines.items():
                    if line2 == line:
                        lines[order].label = label_count
                        break
                label_count += 1
            else:
                # Find the existing label for this pattern
                for lbl, line2 in labels_assigned.items():
                    if line2 == line:
                        line.label = lbl
                        line.isRepeat = True
                        break
        return label_count

    # get rests between notes
    def get_rests(self, notes: List[MMLNote]) -> List[MMLRest]:
        rests: List[MMLRest] = []
        if notes[0].tick > 0:
            rests.append(MMLRest(0, notes[0].tick))
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
        
        if not notes:
            self.logger.debug("Channel has no notes.")
            return []

        mml_state = MMLState()
        cmd_idx = 0
        rests = self.get_rests(notes)
        durations = sorted(notes + rests, key=lambda dur : dur.tick)
        for duration in durations:
            if isinstance(duration, MMLRest):
                word = MMLWord(duration.tick, duration.duration, None)
            else:
                word = MMLWord(duration.tick, duration.duration, duration.note, duration.pre_note_commands)
                if duration.instrument != mml_state.ins:
                    word.commands.append(InstrumentChange(duration.tick, duration.instrument))
                    mml_state.ins = duration.instrument
            while cmd_idx < len(commands):
                cmd_tick = commands[cmd_idx].tick
                if cmd_tick >= duration.tick and cmd_tick < duration.tick + duration.duration:
                    command = commands[cmd_idx]

                    # special pitchbend handling
                    if isinstance(commands[cmd_idx], PitchBend):
                        # Pitchbend commands are placed after the duration to be modulated
                        word.commands.append(TieBreakCommand(cmd_tick))
                        new_tick = cmd_tick + command.duration
                        if new_tick > duration.tick + duration.duration:
                            self.logger.warning(f"Pitchbend duration {command.duration} exceeds the duration of the note. The command will be ignored.")
                        command.tick = new_tick
                        
                    word.commands.append(command)
                    cmd_idx += 1
                else:
                    break

            words.append(word)
        return words

    def split_at_loop_point(self, words: List[MMLWord]) -> List[MMLWord]:
        loop_point = self.mml_data.loop_tick
        if loop_point is None:
            return words
        
        result: List[MMLWord] = []
        for word in words:
            word_start = word.tick
            word_end = word.tick + word.duration
            
            # Check if loop point falls in the middle of this word
            if word_start < loop_point < word_end:
                # Categorize commands upfront to avoid confusion
                pre_note_commands = [cmd for cmd in word.commands if cmd.tick == word_start]
                mid_note_commands = [cmd for cmd in word.commands if word_start < cmd.tick < loop_point]
                post_loop_commands = [cmd for cmd in word.commands if cmd.tick >= loop_point]
                
                # First word: note or rest from start to loop point
                first_commands = pre_note_commands + mid_note_commands
                first_duration = loop_point - word_start
                first_word = MMLWord(word_start, first_duration, word.note, first_commands)
                result.append(first_word)
                
                # Second word: rest from loop point to end
                second_commands = post_loop_commands
                second_duration = word_end - loop_point
                second_word = MMLWord(loop_point, second_duration, None, second_commands)
                result.append(second_word)
            else:
                # Keep the word as-is
                result.append(word)
        
        return result

    def write_loop_point(self, has_remote_commands: bool) -> str:
        loop_txt = '/\n'
        if has_remote_commands:
            loop_txt += "(!99, 0) ; reset remote state for loop\n"
        return loop_txt

    def write(self) -> None:
        has_loop_point = self.mml_data.loop_tick is not None

        txt = ''
        for c in range(self.mml_data.num_channels):
            word_txt = ''
            txt += f'#{c}\n'

            # A "word" is a note or rest with its commands
            words = self.make_words(self.mml_data.notes[c], self.mml_data.commands[c])
            # Cannot have the loop point mid-word
            if has_loop_point:
                words = self.split_at_loop_point(words)
            # sort again for good measure
            words = sorted(words, key=lambda words : words.tick)

            lines: List[MMLLine] = []
            line_words: List[MMLWord] = []
            cur_section_num = 0
            for word in words:
                sectionNum = word.tick // self.mml_data.section_length
                # split line at loop point
                is_loop_point = has_loop_point and word.tick == self.mml_data.loop_tick
                if sectionNum != cur_section_num or is_loop_point:
                    lines.append(MMLLine(line_words, cur_section_num, self.mml_data.measure_length))
                    cur_section_num = sectionNum
                    line_words = []
                line_words.append(word)

            lines.append(MMLLine(line_words, cur_section_num, self.mml_data.measure_length))

            self.label_count = self.optimize_loops(lines, self.label_count)

            mml_state = MMLState()
            for line in lines:
                if has_loop_point and line.tick() == self.mml_data.loop_tick:
                    word_txt += self.write_loop_point(self.channel_has_remote_commands(c))
                word_txt += f"; section {MMLUtil.to_hex(line.section_num)}\n"
                word_txt += line.to_mml(mml_state) + '\n'

            txt += word_txt + '\n\n'

        # Print warning if any notes were out of range
        MMLUtil.print_out_of_range_warning()

        return txt