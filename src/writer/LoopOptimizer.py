from typing import List
import copy

from .MMLWriterData import *

class LoopOptimizer:
    def __init__(self):
        pass

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
    
    @staticmethod
    def behead_sentence(sentence: MMLSentence) -> Tuple[MMLSentence, List[MMLCommand]]:
        sentence_local = copy.deepcopy(sentence)

        first_word = sentence.words[0]
        initial_commands: List[MMLCommand] = []
        rem: List[MMLCommand] = []
        for cmd in first_word.commands:
            if cmd.tick == first_word.tick:
                initial_commands.append(cmd)
            else:
                rem.append(cmd)
        
        # change first word commands to just commands not on first tick
        sentence_local.words[0].commands = rem

        return sentence_local, initial_commands
    
    @staticmethod
    def rehead_sentence(sentence: MMLSentence, cmds: List[MMLCommand]) -> MMLSection:
        sentence.words[0].commands = cmds + sentence.words[0].commands
        return sentence
    
    @staticmethod 
    def fudge_equality(group1: List[MMLSentence], group2: List[MMLSentence]) -> Tuple[bool, List[MMLSentence], List[MMLSentence]]:
        '''Compares two MML sections, returns whether they're equal.
           If equal, also returns new MMLSections with unique initial commands split off into
           a new initial sentence. '''

        group1_local = copy.deepcopy(group1)
        group2_local = copy.deepcopy(group2)

        group1_local[0], initial_cmds1 = LoopOptimizer.behead_sentence(group1_local[0])
        group2_local[0], initial_cmds2 = LoopOptimizer.behead_sentence(group2_local[0])

        # check for equality without initial commands
        if group1_local == group2_local:
            common_cmds = [item for item in initial_cmds1 if item in initial_cmds2]

            # rebuild sections with split initial commands
            if len(common_cmds) > 0:
                group1_local[0] = LoopOptimizer.rehead_sentence(group1_local[0], common_cmds)
                group2_local[0] = LoopOptimizer.rehead_sentence(group2_local[0], common_cmds)

            unique1 = [item for item in initial_cmds1 if item not in initial_cmds2]
            unique2 = [item for item in initial_cmds2 if item not in initial_cmds1]

            if len(unique1) > 0:
                initial_word1 = MMLWord(group1_local[0].words[0].tick, 0, None, unique1)
                group1_local.insert(0, MMLSentence([initial_word1]))

            if len(unique2) > 0:
                initial_word2 = MMLWord(group2_local[0].words[0].tick, 0, None, unique2)
                group2_local.insert(0, MMLSentence([initial_word2]))

            return True, group1_local, group2_local
        else:
            return False, None, None
        
    @staticmethod
    def fudge_group_in_dict(group: List[MMLSentence], groups: Dict[int, List[MMLSentence]]) -> bool:
        for grp in groups.values():
            if LoopOptimizer.fudge_equality(grp, group)[0]:
                return True
            
        return False


    def optimize_loops_v2(self, sections: List[MMLSection], label_count: int) -> int:
        # Identify and label loops in the channel lines
        labels_assigned: Dict[int, List[MMLSentence]] = {}
        unique_groups: Dict[int, List[MMLSentence]] = {}
        for i, section in enumerate(sections):
            group = section.sentences
            # Check if this section matches any prior seen sections
            # accounting for any extra commands at start of first section
            if not LoopOptimizer.fudge_group_in_dict(group, unique_groups):
                unique_groups[i] = group
                section.loopInfo = [LoopInfo(range(len(group)))]
            elif not LoopOptimizer.fudge_group_in_dict(group, labels_assigned):
                # find the first occurrence
                for order, uniq_grp in unique_groups.items():
                    eq, uniq_grp, grp = self.fudge_equality(uniq_grp, group)
                    if eq:
                        # update sections with potential new first sentence
                        sections[order].sentences = uniq_grp
                        section.sentences = grp
                        # and mark initial unique section with label
                        core_sentences = uniq_grp
                        if uniq_grp[0].words[0].duration == 0:
                            # grab the common sentence group
                            core_sentences = uniq_grp[1:]
                            # and update the initial section's loop info
                            sections[order].loopInfo.insert(0, LoopInfo([0]))
                            sections[order].loopInfo[1].sentenceIndices = range(1, len(uniq_grp))
                        sections[order].loopInfo[-1].label = label_count

                        # Assign a label to this repeated pattern
                        labels_assigned[label_count] = core_sentences
                        # and configure this section's loop info
                        # separating out initial commands if applicable
                        if grp[0].words[0].duration == 0:
                            section.loopInfo = [LoopInfo([0]), LoopInfo([1, len(group)], label_count, True)]
                        else:
                            section.loopInfo = [LoopInfo([0, len(group)], label_count, True)]

                        break
                label_count += 1
            else:
                # Find the existing label for this pattern
                for lbl, assigned_grp in labels_assigned.items():
                    eq, _, grp = self.fudge_equality(assigned_grp, group)
                    if eq:
                        section.sentences = grp
                        # set the loop info for this repeated pattern
                        if grp[0].words[0].duration == 0:
                            section.loopInfo = [LoopInfo([0]), LoopInfo([1, len(grp)], lbl, True)]
                        else:
                            section.loopInfo = [LoopInfo([0, len(grp)], lbl, True)]

                        break
        return label_count