from .MMLWriterData import *

from .LoopOptimizer import LoopOptimizer

# unit tests, am I a real coder now?
# run from fur2amk directory with:
# python -m src.writer.LoopOptimizerTests

optimizer = LoopOptimizer()

def check_output(expected: List[LoopInfo], actual: List[LoopInfo]) -> Tuple[bool, str]:
    ret_str = ''
    if expected != actual:
        ret_str = "Test failed, expected\n"
        for info in expected:
            ret_str += str(info.sentenceIndices)
        ret_str += "\nbut instead got\n"
        for info in actual:
            ret_str += str(info.sentenceIndices)
        return False, ret_str
    else:
        return True, ret_str

def split_test(info: LoopInfo, idcs: List[int], expected: List[LoopInfo]) -> bool:
    output = optimizer._split_loopInfo(info, idcs)
    success, err = check_output(expected, output)
    if not success:
        print(err)
    else:
        print("Pass")

split_test(LoopInfo([0, 1, 2, 3, 4, 5, 6]), [2, 3, 4], [LoopInfo([0, 1]), LoopInfo([2, 3, 4]), LoopInfo([5, 6])])
split_test(LoopInfo([9, 10, 11, 12, 13]), [0, 1, 2], [LoopInfo([9, 10, 11]), LoopInfo([12, 13])])
split_test(LoopInfo([1]), [0], [LoopInfo([1])])
split_test(LoopInfo([3, 4, 5, 6]), [1, 2, 3], [LoopInfo([3]), LoopInfo([4, 5, 6])])