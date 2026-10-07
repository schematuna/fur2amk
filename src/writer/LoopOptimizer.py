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
                section.loopInfo = [LoopInfo(list(range(len(group))))]
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
                            sections[order].loopInfo[1].sentenceIndices = list(range(1, len(uniq_grp)))
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

    def _get_label_groups(self, sections: List[MMLSection], loop_tick: int) -> List[List[LabelInfo]]:
        group_candidate = None
        label_groups: List[List[LabelInfo]] = []
        group_idx: int = 0
        for i, section in enumerate(sections):
            if section.tick() == loop_tick:
                if group_candidate is not None:
                    group_candidate = None
                    group_idx += 1
            if group_candidate is not None and len(section.loopInfo) == 1 and section.loopInfo[0].label is not None:
                label_groups[group_idx].append(LabelInfo(i, 0, section.loopInfo[0].label))
            elif section.loopInfo[-1].label is not None and section.loopInfo[-1].numLoops == 1:
                if group_candidate is not None:
                    group_idx += 1
                group_candidate = section.loopInfo[-1]
                label_groups.append([LabelInfo(i, len(section.loopInfo) - 1, section.loopInfo[-1].label)])
            else:
                if group_candidate is not None:
                    group_candidate = None
                    group_idx += 1

        # only want sequences of labelled sections
        new_label_groups: List[List[LabelInfo]] = []
        for i, grp in enumerate(label_groups):
            if len(grp) > 1:
                new_label_groups.append(grp)

        return new_label_groups
    
    def optimize_subloops(self, sections: List[MMLSection]):
        """optimize finer tuned intra-section subloops
           Uses a modified LZ77 alg, only allowing consecutive repeats"""

        for i, section in enumerate(sections):
            for loop in section.loopInfo:
                # only optimize intial labelled sections and unlabelled repeated sections
                if (loop.label is not None and not loop.isRepeat) or (loop.label is None and loop.numLoops > 1):
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
        """optimize finer tuned intra-section loops"""

        labels_assigned: Dict[int, List[MMLSentence]] = {}
        # links unique groups of sentences to the LoopInfo object from their first occurrence
        unique_groups: List[Tuple[List[MMLSentence], LoopInfo]] = []
        for section in sections:
            # Only optimize section if it hasn't been touched yet
            if len(section.loopInfo) == 1 and section.loopInfo[0].label is None and section.loopInfo[0].numLoops == 1 and section.loopInfo[0].subLoops is None:
                matches = self._lz77(section.sentences)
                loopinfos = self._make_loop_info(section.sentences, matches)

                # assign labels to matching loopinfos
                loopInfo: List[LoopInfo] = []
                for j, info in enumerate(loopinfos):
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

    def optimize_repeats(self, sections: List[MMLSection], label_count: int) -> int:
        # find unoptimized sentence groups
        unoptimized_sent_grps: List[GroupInfo] = []
        for sec_idx, sec in enumerate(sections):
            for info_idx, info in enumerate(sec.loopInfo):
                if info.label == None and info.numLoops == 1 and info.subLoops == None:
                    # this is an untouched subloop, store it as an optimization candidate
                    loop_sentences = [sec.sentences[idx] for idx in info.sentenceIndices]
                    if len(loop_sentences) == 1 and len(loop_sentences[0].words) == 1:
                        # don't optimize ultra-short sentence groups
                        continue
                    unoptimized_sent_grps.append(GroupInfo(sec_idx, info_idx, info, loop_sentences))

        if len(unoptimized_sent_grps) == 0:
            # return early if everything miraculously got optimized
            self.logger.info("Optimized every loopInfo on the first compression pass. Nice!")
            return label_count
        
        # now do proper lz77 on all untouched sentence groups
        labels_assigned: Dict[int, List[MMLSentence]] = {}
        search_buffer: List[GroupInfo] = []
        # defer actual list mutation until every match has been found, since replacing one
        # group_info's loopInfo entry can shift the cached info_index of others in the same section
        pending_replacements: List[Tuple[GroupInfo, List[LoopInfo]]] = []
        # current lookahead position relative to start of unoptimized_sent_grps
        for cur_grp_info in unoptimized_sent_grps:
            matched: bool = False
            for search_idx, search_grp_info in enumerate(search_buffer):
                matches = self._lz77_duplex(search_grp_info.sentences, cur_grp_info.sentences)
                if len(matches) == 0:
                    continue
                matched = True
                # just worry about 1-match case for now
                match = matches[0]
                matched_search_idcs = match[0]
                matched_cursor_idcs = match[1]
                newSearchLoopInfos = self._split_loopInfo(search_grp_info.info, matched_search_idcs, label_count)
                newCurLoopInfos = self._split_loopInfo(cur_grp_info.info, matched_cursor_idcs, label_count, True)
                matched_group = [search_grp_info.info.sentenceIndices[idx] for idx in matched_search_idcs]
                # remember this matched sentence group
                # TODO: need to actually make use of this
                labels_assigned[label_count] = matched_group
                label_count += 1

                pending_replacements.append((cur_grp_info, newCurLoopInfos))
                pending_replacements.append((search_grp_info, newSearchLoopInfos))

                # this loopinfo is no longer a candidate in search buffer
                # TODO: minor optimization - keep unmatched splits in search buffer
                search_buffer.pop(search_idx)
                break
            if matched == False:
                # only add this info to search buffer if it had no matches
                # TODO: need to change this if we support unmatched splits
                search_buffer.append(cur_grp_info)

        # now apply all the replacements, end-to-start so loopinfo indices don't get messed up
        pending_replacements.sort(key=lambda item: (item[0].section_index, item[0].info_index), reverse=True)
        for group_info, new_loop_info in pending_replacements:
            self._replace_loopInfo(sections, group_info, new_loop_info)

        return label_count

    def rollup_sections(self, sections: List[MMLSection], loop_tick: int):
        # a section that is a candidate for condensation is a section that is one self-contained labelled loop
        # if any following sections are just a repeat of that label, they should be folded into the first instance
        loop_candidate: LoopInfo = None
        # labels we condensed. Store for performant cleanup
        condensed_loops: List[LoopInfo] = []
        condensed_labels: set = set()
        sections_to_remove: List[int] = []
        for i, section in enumerate(sections):
            # can't condense across the loop point
            if section.tick() == loop_tick:
                loop_candidate = None
            marked_for_removal: bool = False
            if loop_candidate and len(section.loopInfo) == 1 and section.loopInfo[0].label == loop_candidate.label:
                # fold this section into the candidate and skip writing it
                loop_candidate.numLoops += section.loopInfo[0].numLoops
                marked_for_removal = True
                sections_to_remove.append(i)
                if loop_candidate not in condensed_loops and not loop_candidate.isRepeat:
                    condensed_loops.append(loop_candidate)
                    condensed_labels.add(loop_candidate.label)
            elif section.loopInfo[-1].label:
                loop_candidate = section.loopInfo[-1]
            else:
                loop_candidate = None

            # keep track track of standalone labels that will no longer be needed
            if not marked_for_removal:
                section_labels = [loop.label for loop in section.loopInfo if loop.label is not None]
                repeated_labels = [label for label in section_labels if label in condensed_labels]
                for loop in condensed_loops:
                    if loop.label in repeated_labels:
                        condensed_loops.remove(loop)

        #finally, remove vestigial labels
        for loop in condensed_loops:
            loop.label = None

        for sec in reversed(sections_to_remove):
            sections.pop(sec)
            

    def condense_sections(self, sections: List[MMLSection], loop_tick: int):
        # first, gather groups of sections optimised by label_repeated_sections
        label_groups = self._get_label_groups(sections, loop_tick)

        raw_label_groups: List[List[int]] = []
        for grp in label_groups:
            int_grp: List[int] = []
            for labelled_sec in grp:
                int_grp.append(labelled_sec.label)
            raw_label_groups.append(int_grp)

        # need to use some combination of lz77 and lz77_duplex here...
        # with a special case that if a label is used outside of a previously recognized pattern,
        # the pattern is removed as an optimization candidate (or just shrunk if possible...)
        sections_to_remove: List[int] = []
        for i, grp in enumerate(raw_label_groups):
            matches = self._lz77(grp, min_match_len=2)
            if len(matches) == 0:
                continue
            matched_label_group = label_groups[i]

            for match in matches:
                start_idxs = match[0]
                match_len = match[1]

                # disqualify the match if a label defined in a removed section is still referenced by a surviving one
                removed_secs = set()
                defined_labels = set()
                for start_idx in start_idxs:
                    for match_idx in range(start_idx + 1, start_idx + match_len):
                        info = matched_label_group[match_idx]
                        removed_secs.add(info.section_index)
                        removed_loopinfo = sections[info.section_index].loopInfo[info.info_index]
                        if not removed_loopinfo.isRepeat:
                            defined_labels.add(removed_loopinfo.label)
                if any(loop.isRepeat and loop.label in defined_labels
                       for sec_idx, sec in enumerate(sections) if sec_idx not in removed_secs
                       for loop in sec.loopInfo):
                    continue

                for start_idx in start_idxs:
                    start_label_info = matched_label_group[start_idx]
                    start_section = sections[start_label_info.section_index]
                    start_loopinfo = start_section.loopInfo[start_label_info.info_index]
                    for match_idx in range(start_idx + 1, start_idx + match_len):
                        cur_label_info = matched_label_group[match_idx]
                        cur_section = sections[cur_label_info.section_index]
                        cur_loopinfo = cur_section.loopInfo[cur_label_info.info_index]

                        # remove condensed sections and add their sentences to the start of the match
                        sections_to_remove.append(cur_label_info.section_index)
                        # not necessary if setences won't be written out anyways
                        if not start_loopinfo.isRepeat:
                            new_indices = [j + len(start_section.sentences) for j in range(len(cur_section.sentences))]
                            start_loopinfo.sentenceIndices.extend(new_indices)
                            start_section.sentences.extend(cur_section.sentences)

        for sec in reversed(sections_to_remove):
            sections.pop(sec)

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

    def _make_loop_info(self, sentences: List[MMLSentence], matches: List[Tuple[List[int], int]]) -> List[LoopInfo]:
        """Crafts a set of loop info opbjects given a sentence group and a list of matches
        Handles consecutive repeat metadata, but not labels"""

        # where we are traversing through the sentences
        loopInfos: List[LoopInfo] = []
        for match in matches:
            match_idxs = match[0]
            match_len = match[1]
            if match_len == 0:
                print("WHAT")
            cur_idx = 0
            for idx in match_idxs:
                if cur_idx == idx and cur_idx != 0:
                    loopInfos[-1].numLoops += 1
                else:
                    cur_idx = idx
                    loopInfos.append(LoopInfo(list(range(idx, idx + match_len))))
                cur_idx += match_len

        loopInfos.sort(key=lambda info: info.sentenceIndices[0])

        # fill in the gaps
        filled_loopinfos = copy.deepcopy(loopInfos)
        cur_idx = 0
        for info in loopInfos:
            start_idx = info.sentenceIndices[0]
            length = len(info.sentenceIndices) * info.numLoops
            if cur_idx != start_idx:
                filled_loopinfos.append(LoopInfo(list(range(cur_idx, start_idx))))
            cur_idx = start_idx + length

        if cur_idx != len(sentences):
            filled_loopinfos.append(LoopInfo(list(range(cur_idx, len(sentences)))))

        filled_loopinfos.sort(key=lambda info: info.sentenceIndices[0])

        return filled_loopinfos

    def _lz77(self, items, min_match_len = 1) -> List[Tuple[List[int], int]]:
        """lz77 implementation for an arbitrary group of objects
           returns of match start indices and the match length for each match found"""
        
        match_info: List[Tuple[List[int], int]] = []

        search_buffer: List = []
        lookahead_buffer: List = copy.deepcopy(items)
        last_match: List = None
        # absolute buffer start indices
        search_pos = 0
        la_pos = 0
        while len(lookahead_buffer) > 0:
            found_match = False
            for s_idx in range(len(search_buffer)):
                matched_group: List = []
                _cur_idx = 0
                _search_idx = s_idx
                while (_cur_idx < len(lookahead_buffer)) \
                    and (_search_idx < len(search_buffer)) \
                    and (search_buffer[_search_idx] == lookahead_buffer[_cur_idx]):
                    matched_group.append(lookahead_buffer[_cur_idx])
                    _cur_idx += 1
                    _search_idx += 1

                match_len = len(matched_group)
                if match_len >= min_match_len:
                    # we have a match
                    last_match = matched_group
                    # set loop info
                    relative_pos = search_pos + s_idx
                    match_info.append(([relative_pos, la_pos], match_len))
                    found_match = True
                    break

            # update state after this check
            if found_match:
                lookahead_buffer = lookahead_buffer[len(last_match):]
                la_pos += len(last_match)
                # look ahead for more consecutive matches
                # TODO: do this for any future match, not just consecutive
                while len(lookahead_buffer) >= len(last_match) \
                        and lookahead_buffer[:len(last_match)] == last_match:
                    lookahead_buffer = lookahead_buffer[len(last_match):]
                    match_info[-1][0].append(la_pos)
                    la_pos += len(last_match)

                # start looking again beyond latest match
                # TODO: could keep unmatched lines in search buffer, and track matched lines and skip them
                search_buffer.clear()                
                search_pos = la_pos
            else:
                search_buffer.append(lookahead_buffer.pop(0))
                la_pos += 1
                            
        return match_info

    def _rle_lz(self, sentences: List[MMLSentence]) -> List[LoopInfo]:
        """compression alg for MML sentences
           Identifies consecutive repeated sentences and returns loop info for them
           Hybrid of run-length encoding and lz77
           """
        
        loopInfo: List[LoopInfo] = []

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
                            loopInfo.append(LoopInfo(list(range(cur_buffer_pos, relative_pos))))
                        loopInfo.append(LoopInfo(list(range(relative_pos, relative_pos + match_len)), None, False, 2))
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
            loopInfo.append(LoopInfo(list(range(cur_buffer_pos, cur_buffer_pos + len(search_buffer)))))
                            
        return loopInfo

    def _lz77_duplex(self, group1, group2) -> List[Tuple[List[int], List[int]]]:
        """modified lz77 for finding repeated groups between two sets of items
           Returns tuples of group pairs, indexed relative to start of groups passed in."""

        # shallow copy for nomenclature
        search_buffer = group1
        lookahead_buffer = group2
        matches: List[Tuple[List[int], List[int]]] = []
        for cur_idx in range(len((lookahead_buffer))):
            for search_idx in range(len(search_buffer)):
                matched_group: List[MMLSentence] = []
                _cur_idx = cur_idx
                _search_idx = search_idx
                while (_cur_idx < len(lookahead_buffer)) \
                      and (_search_idx < len(search_buffer)) \
                      and (search_buffer[_search_idx] == lookahead_buffer[_cur_idx]):
                    matched_group.append(lookahead_buffer[_cur_idx])
                    _cur_idx += 1
                    _search_idx += 1

                if len(matched_group) > 0:
                    # found one, now catalog it
                    matches.append((list(range(search_idx, search_idx + len(matched_group))), list(range(cur_idx, cur_idx + len(matched_group)))))
                    # just return the first match for now
                    # TODO: return all matches...
                    return matches
                    
        return matches

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

    def _replace_loopInfo(self, sections: List[MMLSection], group_info: GroupInfo, new_loop_info: List[LoopInfo]):
        sections[group_info.section_index].loopInfo.pop(group_info.info_index)
        sections[group_info.section_index].loopInfo[group_info.info_index:group_info.info_index] = new_loop_info