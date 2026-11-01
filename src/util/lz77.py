from typing import List, Tuple
import copy

class CompressionUtil:
    @staticmethod
    def _lz77(items, min_match_len = 1) -> List[Tuple[List[int], int]]:
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

    @staticmethod
    def _lz77_duplex(group1, group2) -> List[Tuple[List[int], List[int]]]:
        """modified lz77 for finding repeated groups between two sets of items
            Returns tuples of group pairs, indexed relative to start of groups passed in."""

        # shallow copy for nomenclature
        search_buffer = group1
        lookahead_buffer = group2
        matches: List[Tuple[List[int], List[int]]] = []
        for cur_idx in range(len((lookahead_buffer))):
            for search_idx in range(len(search_buffer)):
                matched_group: List = []
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