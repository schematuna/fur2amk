from dataclasses import dataclass

# General music utilities

@dataclass
class ADSR:
    attack: int = None
    decay: int = None
    sustain: int = None
    release: int = None