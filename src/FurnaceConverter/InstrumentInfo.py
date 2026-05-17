from dataclasses import dataclass, field
from typing import Dict

@dataclass
class MappingInfo:
    chiptune_ins_idx: int
    note_to_play: int

@dataclass
class FurInstrumentInfo:
    # for non-sample map instruments, the chiptune instrument index
    default_ins: int = 0

    # sample map data
    # note -> mapping_info
    ins_map: Dict[int, MappingInfo] = field(default_factory=dict)
