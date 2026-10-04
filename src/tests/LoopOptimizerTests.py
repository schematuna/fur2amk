from ..writer.MMLWriterData import *

from ..writer.LoopOptimizer import LoopOptimizer

# unit tests, am I a real coder now?

optimizer = LoopOptimizer()

def make_section(loopInfo: List[LoopInfo]) -> MMLSection:
    section = MMLSection([], 0, 16)
    section.loopInfo = loopInfo
    return section

def make_word(note: int, duration: int = 24, tick: int = 0) -> MMLWord:
    return MMLWord(tick, duration, note)

def make_sentence(notes: List[int], duration: int = 24) -> MMLSentence:
    return MMLSentence([make_word(note, duration, i * duration) for i, note in enumerate(notes)])

def check_match(expected: List[LoopInfo], actual: List[LoopInfo]) -> Tuple[bool, str]:
    ret_str = ''
    if expected != actual:
        ret_str = "Test failed, expected\n" + str(expected)
        ret_str += "\nbut instead got\n" + str(actual)
        return False, ret_str
    else:
        return True, ret_str

def split_test(info: LoopInfo, idcs: List[int], expected: List[LoopInfo]) -> bool:
    output = optimizer._split_loopInfo(info, idcs)
    success, err = check_match(expected, output)
    if not success:
        print(err)
    else:
        print("Pass")

def replace_test(sections: List[MMLSection], group_info: GroupInfo, new_loop_info: List[LoopInfo], expected: List[LoopInfo]) -> bool:
    optimizer._replace_loopInfo(sections, group_info, new_loop_info)
    success, err = check_match(expected, sections[group_info.section_index].loopInfo)
    if not success:
        print(err)
    else:
        print("Pass")

def duplex_test(sentences1: List[MMLSentence], sentences2: List[MMLSentence], expected: List[Tuple[List[int], List[int]]]) -> bool:
    output = optimizer._lz77_duplex(sentences1, sentences2)
    success, err = check_match(expected, output)
    if not success:
        print(err)
    else:
        print("Pass")

def label_group_test(sections: List[MMLSection], loop_tick: int, expected: List[List[LabelInfo]]):
    output = optimizer._get_label_groups(sections, loop_tick)
    success, err = check_match(expected, output)
    if not success:
        print(err)
    else:
        print("Pass")

def lz77_test(items: List[int], min_match_len: int, expected: List[Tuple[List[int], int]]):
    output = optimizer._lz77(items, min_match_len)
    success, err = check_match(expected, output)
    if not success:
        print(err)
    else:
        print("Pass")

if __name__ == "__main__":
    print("Testing _split_loopInfo")
    split_test(LoopInfo([0, 1, 2, 3, 4, 5, 6]), [2, 3, 4], [LoopInfo([0, 1]), LoopInfo([2, 3, 4]), LoopInfo([5, 6])])
    split_test(LoopInfo([9, 10, 11, 12, 13]), [0, 1, 2], [LoopInfo([9, 10, 11]), LoopInfo([12, 13])])
    split_test(LoopInfo([1]), [0], [LoopInfo([1])])
    split_test(LoopInfo([3, 4, 5, 6]), [1, 2, 3], [LoopInfo([3]), LoopInfo([4, 5, 6])])

    print("Testing _replace_loopInfo")
    # replace the sole loopInfo entry in a section
    sections = [make_section([LoopInfo([0, 1, 2, 3, 4, 5, 6])])]
    new_infos = [LoopInfo([0, 1]), LoopInfo([2, 3, 4], 5, True), LoopInfo([5, 6])]
    replace_test(sections, GroupInfo(0, 0, sections[0].loopInfo[0], []), new_infos, new_infos)

    # replace a middle loopInfo entry, surrounding entries should be preserved
    sections = [make_section([LoopInfo([0, 1]), LoopInfo([2, 3, 4]), LoopInfo([5, 6])])]
    new_infos = [LoopInfo([2]), LoopInfo([3, 4], 7, True)]
    expected = [LoopInfo([0, 1]), LoopInfo([2]), LoopInfo([3, 4], 7, True), LoopInfo([5, 6])]
    replace_test(sections, GroupInfo(0, 1, sections[0].loopInfo[1], []), new_infos, expected)

    print("Testing _lz77_duplex")
    sentence_a = make_sentence([60, 62, 64])
    sentence_b = make_sentence([65, 67])
    sentence_c = make_sentence([60, 62, 64])
    sentence_d = make_sentence([70, 71])

    duplex_test([sentence_a], [sentence_b], [])
    duplex_test([sentence_a, sentence_b], [sentence_a, sentence_c], [([0], [0])])
    duplex_test([sentence_b, sentence_a], [sentence_c], [([1], [0])])
    duplex_test([sentence_a, sentence_b], [sentence_c, sentence_b], [([0, 1], [0, 1])])
    duplex_test([sentence_d, sentence_a, sentence_b], [sentence_a, sentence_b, sentence_d], [([1, 2], [0, 1])])
    duplex_test([sentence_d, sentence_a, sentence_b, sentence_d], [sentence_d, sentence_a, sentence_b], [([0, 1, 2], [0, 1, 2])])

    print("Testing _get_label_groups")
    sections: List[MMLSection] = []
    section_1 = make_section([LoopInfo([0], 0)])
    section_2 = make_section([LoopInfo([0], 1)])
    section_3 = make_section([LoopInfo([0])])
    section_4 = make_section([LoopInfo([0], 2)])
    section_5 = make_section([LoopInfo([0], 3)])
    section_pre = make_section([LoopInfo([0]), LoopInfo([0], 4)])
    sections = [section_1, section_2, section_3, section_4, section_5]

    info1 = [LabelInfo(0, 0, 0), LabelInfo(1, 0, 1)]
    info2 = [LabelInfo(3, 0, 2), LabelInfo(4, 0, 3)]
    label_group_test(sections, None, [info1, info2])

    sections = [section_1, section_3, section_4, section_2]
    info3 = [LabelInfo(2, 0, 2), LabelInfo(3, 0, 1)]
    label_group_test(sections, None, [info3])

    sections = [section_pre, section_1, section_pre, section_2]
    info4 = [LabelInfo(0, 1, 4), LabelInfo(1, 0, 0)]
    info5 = [LabelInfo(2, 1, 4), LabelInfo(3, 0, 1)]
    label_group_test(sections, None, [info4, info5])


    print("testing _lz77")
    items1 = [0, 1, 2, 3, 0, 1];
    items2 = [0, 1, 1, 1, 2, 3, 3]
    items3 = [36, 37, 38, 39, 35, 40, 35, 41, 42, 43, 44, 36, 37, 38, 39, 35, 40, 35, 41, 42, 43, 44, 36, 37, 38]
    lz77_test(items1, 1, [([0, 4], 2)])
    lz77_test(items2, 1, [([1, 2, 3], 1), ([5, 6], 1)])
    lz77_test(items3, 2, [([0, 11], 11)])
