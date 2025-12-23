from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum

@dataclass
class FurnaceSNESFlags:
    antiClick: Optional[bool] = None
    echo: Optional[bool] = None
    echoDelay: Optional[int] = None
    echoFeedback: Optional[int] = None
    echoFilterCoeffs: Optional[List[int]] = None
    echoMask: Optional[int] = None
    echoVolL: Optional[int] = None
    echoVolR: Optional[int] = None
    volScaleL: Optional[int] = None
    volScaleR: Optional[int] = None


@dataclass
class FurnaceSample:
    index: int
    name: str
    brr_raw: Optional[bytes] = None  # Raw BRR data if sample is stored as BRR
    c4_rate: Optional[int] = None  # Hz
    # Raw PCM payload and metadata from SMP2
    pcm16: List[int] = field(default_factory=list)  # mono 16-bit samples
    sample_rate: Optional[int] = None
    depth: int = 16
    # will be None if no loop
    loop_start: Optional[int] = None
    loop_end: Optional[int] = None

@dataclass
class FurnaceSNESMacroData:
    is_noise: bool = False
    is_echo: bool = True # per-instrument echo enablement
    is_pitch_mod: bool = False
    invert_right: bool = False # not sure
    invert_left: bool = False  # what these are for
    noise_freq: Optional[int] = None # ranges 0 to 32
    gain_values: Optional[List[int]] = None  # snes gain values
    gain_speed: Optional[int] = None  # ticks between each gain change

@dataclass
class FurnaceInstrument:
    index: int
    name: str
    # SNES ADSR/GAIN from INS2 'SN'
    sn_envelope_on: Optional[bool] = True  # whether adsr or gain is enabled

    # SNES ADSR fields
    # set defaults because furnace file won't have snes values if unchanged from default instrument
    sn_attack: Optional[int] = 15     # 0..15
    sn_decay: Optional[int] = 7      # 0..7
    sn_sustain: Optional[int] = 7    # 0..7
    sn_release: Optional[int] = 0    # 0..31
    decay2: Optional[int] = 0        # 0..31, special decay for some sustain modes
    sustain_mode: Optional[int] = 0  # 0: direct, 1: sustain (release with dec), 2: sustain (release with exp), 3: sustain (release with rel)

    # SNES gain fields
    gain_mode: Optional[int] = None  # 0: direct, 4: dec, 5: exp, 6: inc, 7: bent
    sn_gain: Optional[int] = None    # 0..127 for direct, 0..31 for others

    # Sample mapping from INS2 'SM'
    initial_sample: Optional[int] = 0  # sample 0 by default
    use_sample_map: bool = False
    sample_table: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 1)] * 120)
    
    # Instrument macros (INS2 'MA'): code -> macro definition
    macros: Dict[int, "FurnaceMacro"] = field(default_factory=dict)

    snes_macro_data: FurnaceSNESMacroData = field(default_factory=FurnaceSNESMacroData)

    # Parse SNES-specific macro flags from macros
    # call after all macros have been read
    def parse_snes_macro_flags(self):
        for macro in self.macros.values():
            if macro.code == 2:  # code 2 corresponds to pitch freq
                self.snes_macro_data.noise_freq = macro.values[0]
            if macro.code == 5:  # code 5 corresponds to extra1 macro
                special_snes_flags = macro.values[0]
                self.snes_macro_data.is_noise = (special_snes_flags & 0x01) != 0
                self.snes_macro_data.is_echo = (special_snes_flags & 0x02) != 0
                self.snes_macro_data.is_pitch_mod = (special_snes_flags & 0x04) != 0
                self.snes_macro_data.invert_right = (special_snes_flags & 0x08) != 0
                self.snes_macro_data.invert_left = (special_snes_flags & 0x10) != 0
            if macro.code == 6:  # code 6 corresponds to gain
                self.snes_macro_data.gain_values = macro.values
                self.snes_macro_data.gain_speed = macro.speed

            if self.snes_macro_data.is_noise and self.snes_macro_data.noise_freq is None:
                default_noise_freq = 29
                print(f"Info: Instrument {self.index:02X} is noise but has no noise_freq set; defaulting to {default_noise_freq}.")
                self.snes_macro_data.noise_freq = default_noise_freq
@dataclass
class FurnaceMacro:
    # Representation of a single macro as parsed from INS2 'MA'
    # for snes, codes are interpreted as:
    #  duty (2) = pitch freq
    #  wave (3) = waveform
    #  extra1 (5) = special
    #  extra2 (6) = gain
    code: int                   # macro code (0..21, 255=end)
    length: int                 # number of steps, basically the length of values list
    loop: int                   # loop position (unused)
    release: int                # release position (unused)
    type: int                   # 0=normal,1=ADSR,2=LFO
    instant_release: bool       # instant release flag (>=182) (unused)
    delay: int                  # macro delay
    speed: int                  # step length in ticks
    values: List[int]           # parsed integer values (length entries)


@dataclass
class FurnaceRow:
    Note: Optional[int] = None
    Ins: Optional[int] = None
    Vol: Optional[int] = None   # 0..64
    Effects: List[Tuple[int, int]] = field(default_factory=list)

    # enum for note kinds
    class NoteKind(Enum):
        NOTE = 0
        OFF = 1
        RELEASE = 2
        MACRO_RELEASE = 3
        EMPTY = 4

    # Classify a Furnace row by note type
    def kind(self) -> NoteKind:
        n = self.Note
        if n is None:
            return self.NoteKind.EMPTY
        
        try:
            v = int(n)
        except Exception:
            return self.NoteKind.EMPTY
        
        if v == 180:
            return self.NoteKind.OFF
        if v == 181:
            return self.NoteKind.RELEASE
        if v == 182:
            return self.NoteKind.MACRO_RELEASE
        if 0 <= v <= 179:
            return self.NoteKind.NOTE
        
        return self.NoteKind.EMPTY


@dataclass
class FurnacePattern:
    rows: List[List[FurnaceRow]] = field(default_factory=list)  # 64 x channels


@dataclass
class FurnaceModule:
    # A normalized adapter exposing the subset EventTable/MML expect
    SongName: str = ''
    Author: str = ''            # song author
    Comment: str = ''           # song comment
    GV: float = 1.0             # global volume (0..1)
    Instruments: List[FurnaceInstrument] = field(default_factory=list)
    Samples: List[FurnaceSample] = field(default_factory=list)
    NumChannels: int = 8
    PatternLength: int = 64
    OrdersPerChannel: List[List[int]] = field(default_factory=list)  # [ch][order_idx] -> pattern_id
    PatternsByChannel: List[Dict[int, List[FurnaceRow]]] = field(default_factory=list)  # [ch][pat_id] -> rows
    # Timing
    HighlightA: int = 4
    HighlightB: int = 16
    TicksPerSecond: float = 0.0
    Speed1: int = 6
    Speed2: int = 0
    SNESFlags: FurnaceSNESFlags = field(default_factory=FurnaceSNESFlags)