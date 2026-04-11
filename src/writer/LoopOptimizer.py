from typing import List

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