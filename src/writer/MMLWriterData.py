from typing import List, Optional
from dataclasses import dataclass

from ..model.MMLData import *
from ..model.MMLCommands import *


################################
# INTERNAL MML WRITER CLASSES  #
################################

@dataclass
class MMLState:
    octave: Optional[int]                   = None

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
        # special case for 0-tick words
        if self.duration == 0:
            for cmd in self.commands:
                word_txt += cmd.get_text(mml_state)

            return word_txt

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

@dataclass
class SubLoopInfo:
    # indices of looped sentences, relative to parent loop
    sentenceIndices: List[int]
    # How many times this loops
    numLoops: int = 1

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
    numLoops: int = 1
    # subloops within this loop (uses AMK superloops)
    subLoops: List[SubLoopInfo] = None

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
                    line_txt += f"({info.label})["
                    if len(info.sentenceIndices) > 1:
                        line_txt += "\n"
                    for j, subloop in enumerate(info.subLoops):
                        if subloop.numLoops > 1:
                            line_txt += f"[["
                            if len(subloop.sentenceIndices) > 1:
                                line_txt += "\n"
                        for i, idx in enumerate(subloop.sentenceIndices):
                            line_txt += self.sentences[idx].to_mml(mml_state)
                            if i != len(subloop.sentenceIndices) - 1:
                                line_txt += "\n"
                        if subloop.numLoops > 1:
                            if len(subloop.sentenceIndices) > 1:
                                line_txt += "\n"
                            line_txt += f"]]{subloop.numLoops}"
                        if j != len(info.subLoops) - 1:
                            line_txt += "\n"
                    if len(info.sentenceIndices) > 1:
                        line_txt += "\n"
                    line_txt += "]"
            else:
                if info.numLoops > 1:
                    line_txt += "["
                    if len(info.sentenceIndices) > 1:
                        line_txt += "\n"
                for i, idx in enumerate(info.sentenceIndices):
                    line_txt += self.sentences[idx].to_mml(mml_state)
                    if i != len(info.sentenceIndices) - 1:
                        line_txt += "\n"
                if info.numLoops > 1:
                    if len(info.sentenceIndices) > 1:
                        line_txt += "\n"
                    line_txt += f"]{info.numLoops}"
                line_txt += "\n"

        # don't want last newline, strip it
        return line_txt.rstrip()