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

                    loopInfo = self._rle_lz(looped_sentences)
                    # apply loop sentence offset
                    idx_offset = loop.sentenceIndices[0]
                    for info in loopInfo:
                        info.sentenceIndices = [x + idx_offset for x in info.sentenceIndices]

                    loop.subLoops = loopInfo

    def optimize_loops(self, sections: List[MMLSection], label_count: int) -> int:
        """optimize finer tuned intra-section loops
           First pass uses a modified lz77 alg, only allowing consecutive repeats
           Second pass does full lz77 on all remaining sentence groups
           Both passes assign labels for repeated sentence groups across sections"""

        labels_assigned: Dict[int, List[MMLSentence]] = {}
        # links unique groups of sentences to the LoopInfo object from their first occurrence
        unique_groups: List[Tuple[List[MMLSentence], LoopInfo]] = []
        # loops that aren't optimized by lz77, tracked on this pass for later use
        unoptimized_loops: List[Tuple[int, int]] = {} # ec idx, loopinfo idx
        for i, section in enumerate(sections):
            # Only optimize section if it hasn't been touched yet. i.e. doesn't have a label
            if len(section.loopInfo) == 1 and section.loopInfo[0].label is None:
                subloops = self._rle_lz(section.sentences)

                loopInfo: List[LoopInfo] = []
                for j, info in enumerate(subloops):
                    newLoopInfo = LoopInfo(info.sentenceIndices, None, False, info.numLoops)
                    # TODO: calculate all loopInfo sentences outside the loop, reuse for both lz77 passes
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
                    # if unlooped/unlabelled, then lz77 didn't optimize it. Store it for future optimizations
                    if newLoopInfo.numLoops == 1 and newLoopInfo.label == None:
                        unoptimized_loops.append((i, j))

                section.loopInfo = loopInfo

        # resolve unoptimized loops to their actual sentences ahead of time
        if len(unoptimized_loops) == 0:
            # return early if everything miraculously got optimized
            self.logger.info("Optimized every loopInfo on the first compression pass. Nice!")
            return label_count
        unoptimized_sent_grps: List[Tuple[LoopInfo, List[MMLSentence]]] = []
        for section_idx, info_idx in unoptimized_loops:
            section = sections[section_idx]
            info = section.loopInfo[info_idx]
            loop_sentences = [section.sentences[idx] for idx in info.sentenceIndices]
            unoptimized_sent_grps.append((info, loop_sentences))

        # now do proper lz77 on all untouched sentence groups
        # reused labels_assigned data structure here, same purpose
        labels_assigned.clear()
        # lz77 principles hold here
        # we iterate through unoptimized loopInfos sequentially.
        # at each step, it looks through entire search buffer to see if any sentence group within any loopInfo 
        # matches any sentence group within the currently considered loopinfo.
        # If a match is found, it creates a new label and removes the loopInfo with the match from the search buffer.
        # TODO: this approach means there can only be one match per loopInfo object. Ideally we just remove the matched sentences
        #       and keep the loopInfo object around until all sentences are matched.
        # then, the cursor position is increments, the current loopInfo is added to the search buffer.
        # TODO: move this pass into its own function and unit test so you dont go crazy debugging
        search_buffer: List[Tuple[LoopInfo, List[MMLSentence]]] = []
        search_buffer.append(unoptimized_sent_grps[0])
        # current lookahead position relative to start of unoptimized_sent_grps
        for cur_info, cur_sentences in unoptimized_sent_grps:
            matched: bool = False
            for i, (search_info, search_sentences) in enumerate(search_buffer):
                # TODO: implement the duplex lz77
                matches = self._lz77_duplex(search_sentences, cur_sentences)
                if len(matches) == 0:
                    continue
                matched = True
                # just worry about 1-match case for now
                match = matches[0]
                matched_search_idcs = match[0]
                matched_cursor_idcs = match[1]
                newSearchLoopInfos = self._split_loopInfo(search_info, matched_search_idcs, label_count)
                newCurLoopInfos = self._split_loopInfo(cur_info, matched_cursor_idcs, label_count, True)
                matched_group = [search_info.sentenceIndices[idx] for idx in matched_search_idcs]
                labels_assigned[label_count] = matched_group
                label_count += 1
                # update infos in original data
                # TODO: need to use section/loopinfo idcs from unoptimized_loops object to find and replace relevant loopinfo

                # this loopinfo is no longer a candidate in search buffer
                # TODO: minor optimization - keep unmatched splits in search buffer
                search_buffer.pop(i)
                break
            if matched == False:
                # only add this info to search buffer if it had no matches
                # TODO: need to change this if we support unmatched splits
                search_buffer.append(cur_info, cur_sentences)
                    
        return label_count

    def condense_sections(self, sections: List[MMLSection], loop_tick: int):
        # a section that is a candidate for condensation is a section that is one self-contained labelled loop
        # if any following sections are just a repeat of that label, they should be folded into the first instance
        loop_candidate: LoopInfo = None
        # labels we condensed. Store for performant cleanup
        condensed_loops: List[LoopInfo] = []
        condensed_labels: set = set()
        for section in sections:
            # can't condense across the loop point
            if section.tick() == loop_tick:
                loop_candidate = None
            if loop_candidate and len(section.loopInfo) == 1 and section.loopInfo[0].label == loop_candidate.label:
                # fold this section into the candidate and skip writing it
                loop_candidate.numLoops += section.loopInfo[0].numLoops
                section.skip_write = True
                if loop_candidate not in condensed_loops and not loop_candidate.isRepeat:
                    condensed_loops.append(loop_candidate)
                    condensed_labels.add(loop_candidate.label)
            elif section.loopInfo[-1].label:
                loop_candidate = section.loopInfo[-1]
            else:
                loop_candidate = None

            # keep track track of standalone labels that will no longer be needed
            if not section.skip_write:
                section_labels = [loop.label for loop in section.loopInfo if loop.label is not None]
                repeated_labels = [label for label in section_labels if label in condensed_labels]
                for loop in condensed_loops:
                    if loop.label in repeated_labels:
                        condensed_loops.remove(loop)

        #finally, remove vestigial labels
        for loop in condensed_loops:
            loop.label = None


    def simplify_loops(self, sections: List[MMLSection]):
        # simplify case of single repeated subloop within a loop
        multed_labels: dict[int, int] = dict() # label, multiplier
        for section in sections:
            for loop in section.loopInfo:
                if loop.subLoops:
                    subloop = loop.subLoops[0]
                    if len(loop.subLoops) == 1 and subloop.numLoops > 1:
                        loop.sentenceIndices = subloop.sentenceIndices
                        loop.numLoops = subloop.numLoops * loop.numLoops
                        loop.subLoops = None

                        if loop.label is not None:
                            multed_labels[loop.label] = subloop.numLoops

        # propagate multipliers to repeated labels
        for section in sections:
            for loop in section.loopInfo:
                if loop.label in multed_labels and loop.isRepeat:
                    loop.numLoops *= multed_labels[loop.label]

    def _rle_lz(self, sentences: List[MMLSentence]) -> List[SubLoopInfo]:
        """compression alg for MML sentences
           Identifies consecutive repeated sentences and returns loop info for them
           Hybrid of run-length encoding and lz77
           """
        
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

    def _lz77_duplex(self, sentences1: List[MMLSentence], sentences2: List[MMLSentence]) -> List[Tuple[List[int], List[int]]]:
        """modified lz77 for finding repeated sentences groups between two sets of sentences
           Returns tuples of sentence group pairs, indexed relative to start of sentences passed in."""

        # well this is complicated to write
        # think it's just the same as lz77 expect the search buffer is limited to sentences1
        # and lookahead buffer is limited to sentences2
        # we pop sentences one by one from sentences1 into search buffer
        # some constraints made the original implementation simpler.
        # namely that matches had ot be consecutive. Computation time will go up by removing that constraint.
        # have to check all possibilities at every step.
        
        return []

    def _split_loopInfo(self, info: LoopInfo, split_idcs: List[int], label: int = None, isRepeat: bool = False) -> List[LoopInfo]:
        """Splits a loopInfo object into smaller loopinfo objects, given indices you want to split out
           split_idcs is relative to start of info
           Optionally labels the split section"""
        newLoopInfos: List[LoopInfo] = []

        first_info_sent_idx = info.sentenceIndices[0]
        if split_idcs[0] > 0:
            newLoopInfos.append(LoopInfo(list(range(first_info_sent_idx, first_info_sent_idx + split_idcs[0]))))
        split_loop_info = LoopInfo(list(range(first_info_sent_idx + split_idcs[0], first_info_sent_idx + split_idcs[-1] + 1)))
        split_loop_info.label = label
        split_loop_info.isRepeat = isRepeat
        newLoopInfos.append(split_loop_info)
        if split_idcs[-1] < len(info.sentenceIndices) - 1:
            newLoopInfos.append(LoopInfo(list(range(first_info_sent_idx + split_idcs[-1] + 1, info.sentenceIndices[-1] + 1))))

        return newLoopInfos