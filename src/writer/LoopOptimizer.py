from typing import List
import copy

from .MMLWriterData import *

class LoopOptimizer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

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
            tmp_cmds = initial_cmds2.copy()
            common_cmds: list[MMLCommand] = []
            for cmd in initial_cmds1:
                if cmd in tmp_cmds:
                    common_cmds.append(cmd)
                    tmp_cmds.remove(cmd)

            # rebuild sections with split initial commands
            if len(common_cmds) > 0:
                group1_local[0] = LoopOptimizer.rehead_sentence(group1_local[0], common_cmds)
                group2_local[0] = LoopOptimizer.rehead_sentence(group2_local[0], common_cmds)

            unique1 = initial_cmds1.copy()
            for cmd in common_cmds:
                if cmd in unique1:
                    unique1.remove(cmd)
                else:
                    print("Expected to find command in loop optimization. Something went wrong.")

            unique2 = initial_cmds2.copy()
            for cmd in common_cmds:
                if cmd in unique2:
                    unique2.remove(cmd)
                else:
                    print("Expected to find command in loop optimization. Something went wrong.")

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

    def label_repeated_sections(self, sections: List[MMLSection], label_count: int) -> int:
        """Iterates through the sections, identifying duplicates and assigning loop metadata.
           Uses 'fudge' comparisons between sentence groups, recognizing equality even if 
           initial 0-tick commands differ. These commands then get split out of the loop."""
        
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
    
    def optimize_subloops(self, sections: List[MMLSection]):
        """optimize finer tuned intra-section subloops
           Uses a modified LZ77 alg, only allowing consecutive repeats"""

        for i, section in enumerate(sections):
            for loop in section.loopInfo:
                # Only optimize if this is the initial labelled loop
                if loop.label is not None and not loop.isRepeat:
                    looped_sentences: List[MMLSentence] = []
                    for idx in loop.sentenceIndices:
                        looped_sentences.append(section.sentences[idx])

                    loopInfo = self.lz77(looped_sentences)
                    # apply loop sentence offset
                    idx_offset = loop.sentenceIndices[0]
                    for info in loopInfo:
                        info.sentenceIndices = [x + idx_offset for x in info.sentenceIndices]

                    loop.subLoops = loopInfo

    def optimize_loops(self, sections: List[MMLSection], label_count: int) -> int:
        """optimize finer tuned intra-section loops
           Uses a modified LZ77 alg, only allowing consecutive repeats
           assigns labels for repeated sentence groups across sections"""

        labels_assigned: Dict[int, List[MMLSentence]] = {}
        # links unique groups of sentences to the LoopInfo object from their first occurrence
        unique_groups: List[Tuple[List[MMLSentence], LoopInfo]] = []
        for section in sections:
            # Only optimize section if it hasn't been touched yet. i.e. doesn't have a label
            if len(section.loopInfo) == 1 and section.loopInfo[0].label is None:
                subloops = self.lz77(section.sentences)

                loopInfo: List[LoopInfo] = []
                for info in subloops:
                    newLoopInfo = LoopInfo(info.sentenceIndices, None, False, info.numLoops)
                    sentences = [section.sentences[idx] for idx in info.sentenceIndices]
                    if not any(g == sentences for g, _ in unique_groups):
                        unique_groups.append((sentences, newLoopInfo))
                    elif not any(g == sentences for g in labels_assigned.values()):
                        for group, original_loop_info in unique_groups:
                            if group == sentences:
                                # assign label to this repeated group
                                labels_assigned[label_count] = group
                                # mark the original LoopInfo object with the label directly
                                original_loop_info.label = label_count
                                newLoopInfo.label = label_count
                                newLoopInfo.isRepeat = True
                                newLoopInfo.numLoops = info.numLoops
                                label_count += 1
                    else:
                        # find existing label for this pattern
                        for lbl, group in labels_assigned.items():
                            if group == sentences:
                                newLoopInfo.label = lbl
                                newLoopInfo.isRepeat = True
                                newLoopInfo.numLoops = info.numLoops
                                break

                    loopInfo.append(newLoopInfo)

                section.loopInfo = loopInfo

        return label_count

    def lz77(self, sentences: List[MMLSentence]) -> List[SubLoopInfo]:
        """lz77 alg for MML sentences
           Identifies consecutive repeated sentences and returns loop info for them"""
        
        loopInfo: List[SubLoopInfo] = []

        search_buffer: List[MMLSentence] = []
        lookahead_buffer: List[MMLSentence] = copy.deepcopy(sentences)
        last_match: List[MMLSentence] = None
        # search buffer position
        cur_buffer_pos = 0
        while len(lookahead_buffer) > 0:
            search_sentence = lookahead_buffer[0]
            found_match = False
            # check if this is another consecutive repeat of the last match
            if last_match and len(lookahead_buffer) >= len(last_match) \
                          and lookahead_buffer[:len(last_match)] == last_match:
                # search buffer should be empty here since we're coming fresh off a match
                assert(len(search_buffer) == 0)
                loopInfo[-1].numLoops += 1
                found_match = True
            else:
                # end match chain as soon as it is broken
                # since matches must be consecutive
                last_match = None

            # check for any consecutive matches in search buffer
            for i, buffer_sentence in enumerate(search_buffer):
                if buffer_sentence == search_sentence:
                    # buffer must have a match from matched sentence to end of buffer
                    # since matches have to be consecutive
                    search_match = search_buffer[i:]
                    # can't match a pattern greater than the number of sentences left to check
                    match_len = len(search_match)
                    if match_len > len(lookahead_buffer):
                        continue
                    lookahead_match = lookahead_buffer[:match_len]

                    if search_match == lookahead_match:
                        # we have a consecutive match
                        last_match = search_match
                        # set loop info
                        relative_pos = cur_buffer_pos + i
                        if i > 0:
                            loopInfo.append(SubLoopInfo(list(range(cur_buffer_pos, relative_pos))))
                        loopInfo.append(SubLoopInfo(list(range(relative_pos, relative_pos + match_len)), 2))
                        found_match = True
                        break

            # update state after this check
            if found_match:
                # track buffer position relative to main loop start
                cur_buffer_pos += len(search_buffer) + len(last_match)
                # can't match anything but the last match from the existing search buffer
                search_buffer.clear()
                # remove match from lookahead
                lookahead_buffer = lookahead_buffer[len(last_match):]
            else:
                search_buffer.append(lookahead_buffer.pop(0))

        if len(search_buffer) > 0:
            loopInfo.append(SubLoopInfo(list(range(cur_buffer_pos, cur_buffer_pos + len(search_buffer)))))
                            
        return loopInfo