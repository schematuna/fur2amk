from .MMLWriterData import *

from .LoopOptimizer import LoopOptimizer

# unit tests, am I a real coder now?
# run from fur2amk directory with:
# python -m src.writer.LoopOptimizerTests

optimizer = LoopOptimizer()

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

print("Testing _split_loopInfo")

split_test(LoopInfo([0, 1, 2, 3, 4, 5, 6]), [2, 3, 4], [LoopInfo([0, 1]), LoopInfo([2, 3, 4]), LoopInfo([5, 6])])
split_test(LoopInfo([9, 10, 11, 12, 13]), [0, 1, 2], [LoopInfo([9, 10, 11]), LoopInfo([12, 13])])
split_test(LoopInfo([1]), [0], [LoopInfo([1])])
split_test(LoopInfo([3, 4, 5, 6]), [1, 2, 3], [LoopInfo([3]), LoopInfo([4, 5, 6])])

def make_section(loopInfo: List[LoopInfo]) -> MMLSection:
    section = MMLSection([], 0, 16)
    section.loopInfo = loopInfo
    return section

def replace_test(sections: List[MMLSection], group_info: GroupInfo, new_loop_info: List[LoopInfo], expected: List[LoopInfo]) -> bool:
    optimizer._replace_loopInfo(sections, group_info, new_loop_info)
    success, err = check_match(expected, sections[group_info.section_index].loopInfo)
    if not success:
        print(err)
    else:
        print("Pass")


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

def make_word(note: int, duration: int = 24, tick: int = 0) -> MMLWord:
    return MMLWord(tick, duration, note)

def make_sentence(notes: List[int], duration: int = 24) -> MMLSentence:
    return MMLSentence([make_word(note, duration, i * duration) for i, note in enumerate(notes)])

def duplex_test(sentences1: List[MMLSentence], sentences2: List[MMLSentence], expected: List[Tuple[List[int], List[int]]]) -> bool:
    output = optimizer._lz77_duplex(sentences1, sentences2)
    success, err = check_match(expected, output)
    if not success:
        print(err)
    else:
        print("Pass")

sentence_a = make_sentence([60, 62, 64])
sentence_b = make_sentence([65, 67])
sentence_c = make_sentence([60, 62, 64])
sentence_d = make_sentence([70, 71])

print("Testing _lz77_duplex")
duplex_test([sentence_a], [sentence_b], [])
duplex_test([sentence_a, sentence_b], [sentence_a, sentence_c], [([0], [0])])
duplex_test([sentence_b, sentence_a], [sentence_c], [([1], [0])])
duplex_test([sentence_a, sentence_b], [sentence_c, sentence_b], [([0, 1], [0, 1])])
duplex_test([sentence_d, sentence_a, sentence_b], [sentence_a, sentence_b, sentence_d], [([1, 2], [0, 1])])
duplex_test([sentence_d, sentence_a, sentence_b, sentence_d], [sentence_d, sentence_a, sentence_b], [([0, 1, 2], [0, 1, 2])])