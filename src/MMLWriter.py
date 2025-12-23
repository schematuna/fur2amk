from typing import Dict, List, Optional
from dataclasses import dataclass

from .model.MMLData import *
from .model.MMLCommands import *

from .MMLUtil import *


################################
# INTERNAL MML WRITER CLASSES  #
################################

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

    def to_mml(self, mml_state: MMLState, durFormatter: DurationFormatter) -> str:
        word_txt = ''
        command_idx = 0
        cur_tick = self.tick
        
        # process any pre-note commands (at the same tick as note start)
        while command_idx < len(self.commands) and self.commands[command_idx].tick == self.tick:
            word_txt += self.commands[command_idx].to_mml(mml_state) + ' '
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
            first_cmd_tick = self.commands[command_idx].tick
            if first_cmd_tick > cur_tick:
                word_txt += durFormatter.format(first_cmd_tick - cur_tick, cont) + ' '
                cur_tick = first_cmd_tick
                cont = True
        
        # Interleave commands with duration
        while command_idx < len(self.commands):
            command = self.commands[command_idx]
            cmd_tick = command.tick
            word_txt += command.to_mml(mml_state) + ' '
            command_idx += 1
            
            # Update cur_tick to this command's tick
            cur_tick = cmd_tick
            
            # Add duration to next command
            if command_idx < len(self.commands):
                next_cmd_tick = self.commands[command_idx].tick
                if next_cmd_tick > cur_tick:
                    word_txt += durFormatter.format(next_cmd_tick - cur_tick, cont) + ' '
                    cur_tick = next_cmd_tick
                    cont = True

        # Add remaining duration to end of note
        end_tick = self.tick + self.duration
        if cur_tick < end_tick:
            word_txt += durFormatter.format(end_tick - cur_tick, cont)

        return word_txt


class MMLLine:
    def __init__(self, words: List[MMLWord]) -> None:
        self.words = words
        self.label: Optional[int] = None
        self.isRepeat: bool = False
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, MMLLine):
            return False
        return self.words == other.words

    def convert_words(self, mml_state: MMLState, durFormatter: DurationFormatter) -> List[str]:
        words_txt = ''
        for word in self.words:
            words_txt += word.to_mml(mml_state, durFormatter) + ' '
        return words_txt.rstrip()
    
    def to_mml(self, mml_state: MMLState, durFormatter: DurationFormatter) -> str:
        line_txt = ''
        if self.label is not None:
            if self.isRepeat:
                line_txt += f"({self.label})"
            else:
                line_txt += f"({self.label})["
                line_txt += self.convert_words(mml_state, durFormatter)
                line_txt += "]"
        else:
            line_txt += self.convert_words(mml_state, durFormatter)
        return line_txt

class MMLWriter:
    def __init__(self, mml_data: MMLData, label_start: int) -> None:
        self.mml_data = mml_data
        self.label_count = label_start

        self.durForamtter = DurationFormatter()

    def channel_has_remote_commands(self, channel: int) -> bool:
        for event in self.mml_data.commands[channel]:
            if isinstance(event, RemoteCommand):
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
            print(f"Info: Channel has no notes.")
            return []

        cmd_idx = 0
        rests = self.get_rests(notes)
        durations = sorted(notes + rests, key=lambda dur : dur.tick)
        for duration in durations:
            if isinstance(duration, MMLRest):
                word = MMLWord(duration.tick, duration.duration, None)
            else:
                word = MMLWord(duration.tick, duration.duration, duration.note)
            while cmd_idx < len(commands):
                cmd_tick = commands[cmd_idx].tick
                if cmd_tick >= duration.tick and cmd_tick < duration.tick + duration.duration:
                    word.commands.append(commands[cmd_idx])
                    cmd_idx += 1
                else:
                    break
            words.append(word)
        return words

    def write(self) -> None:
        txt = ''
        for c in range(self.mml_data.num_channels):
            word_txt = ''
            txt += f'#{c}\n'

            # A "word" is a note or rest with its commands
            words = self.make_words(self.mml_data.notes[c], self.mml_data.commands[c])
            # sort again for good measure
            words = sorted(words, key=lambda words : words.tick)
            lines: List[MMLLine] = []
            line_words: List[MMLWord] = []
            cur_section_num = 0
            for word in words:
                sectionNum = word.tick // self.mml_data.section_length
                if sectionNum != cur_section_num:
                    cur_section_num = sectionNum
                    lines.append(MMLLine(line_words))
                    line_words = []
                line_words.append(word)
            lines.append(MMLLine(line_words))

            self.label_count = self.optimize_loops(lines, self.label_count)

            mml_state = MMLState()
            for line in lines:
                word_txt += f"; section {line.words[0].tick // self.mml_data.section_length}\n"
                word_txt += line.to_mml(mml_state, self.durForamtter) + '\n'

            txt += word_txt + '\n\n'
        return txt