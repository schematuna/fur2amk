from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from enum import Enum
import logging

from .FurnaceEffects import *
from ..util.SNESUtil import *


class SustainMode(Enum):
    """SNES sustain/release modes from Furnace.

    Controls how the envelope behaves during sustain and on key-off.
    Modes 1-3 use 'd2' as the decay rate during sustain, and 'r' controls release behavior.
    """
    DIRECT = 0       # No sustain - key off sends hardware KOFF (fast release)
    EFF_LINEAR = 1   # Sustain with d2; key-off switches to GAIN linear decay at rate r
    EFF_EXP = 2      # Sustain with d2; key-off switches to GAIN exponential decay at rate r
    DELAYED = 3      # Sustain with d2; key-off updates ADSR R field to r


@dataclass
class FurnaceSNESFlags:
    antiClick: bool = True
    echo: bool = True
    echoDelay: int = 0
    echoFeedback: int = 0
    echoFilterCoeffs: List[int] = field(default_factory=lambda: [127, 0, 0, 0, 0, 0, 0, 0])
    echoMask: int = 0
    echoVolL: int = 127
    echoVolR: int = 127
    volScaleL: int = 0
    volScaleR: int = 0


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
    vol_values: Optional[List[int]] = None # macro volume values
    vol_speed: Optional[List[int]] = None # ticks between each volume change

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
    decay2: Optional[int] = 0        # 0..31, used as R during sustain in modes 1-3
    sustain_mode: SustainMode = SustainMode.DIRECT  # Controls key-off behavior

    # SNES gain fields
    gain_mode: GainMode = GainMode.DIRECT  # GAIN register mode when useEnv is False
    sn_gain: Optional[int] = None    # 0..127 for DIRECT, 0..31 for others

    # Sample mapping from INS2 'SM'
    initial_sample: Optional[int] = 0  # sample 0 by default
    use_sample_map: bool = False
    use_sample: bool = False  # bit 1 of SM flags
    use_wave: bool = False    # bit 2 of SM flags
    waveform_length: int = 0  # from SM block
    sample_table: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 1)] * 120)
    
    # Instrument macros (INS2 'MA'): code -> macro definition
    macros: Dict[int, "FurnaceMacro"] = field(default_factory=dict)

    snes_macro_data: FurnaceSNESMacroData = field(default_factory=FurnaceSNESMacroData)

    # Parse SNES-specific macro flags from macros
    # call after all macros have been read
    def parse_snes_macro_flags(self):
        self.logger = logging.getLogger(__name__)
        for macro in self.macros.values():
            if macro.code == 0:  # code 0 is volume
                self.snes_macro_data.vol_values = macro.values
                self.snes_macro_data.vol_speed = macro.speed
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
                self.logger.debug(f"Instrument {self.index:02X} is noise but has no noise_freq set; defaulting to {default_noise_freq}.")
                self.snes_macro_data.noise_freq = default_noise_freq

    def get_initial_adsr(self) -> ADSR:
        # For sustain modes 1-3, use decay2 as the release value during note-on
        # (sn_release is only used for DIRECT mode, or when note-off happens in DELAYED mode)
        release = self.sn_release if self.sustain_mode == SustainMode.DIRECT else self.decay2
        return ADSR(self.sn_attack, self.sn_decay, self.sn_sustain, release)

    def get_initial_gain(self) -> SnesGain:
        return SnesGain(self.gain_mode, self.sn_gain)

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

# class representing a chunk of time in the Furnace sequencer. Could be a row or just a tick
@dataclass
class FurnaceRow:
    Note: Optional[int] = None
    Ins: Optional[int] = None
    Vol: Optional[int] = None   # 0..64
    Effects: List[FurnaceEffect] = field(default_factory=list)

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

    def get_effect(self, command_type: type[FurnaceEffect]) -> Optional[FurnaceEffect]:
        num_effects = 0
        ret = None
        for effect in self.Effects:
            if isinstance(effect, command_type):
                num_effects += 1
                ret = effect

        self.logger = logging.getLogger(__name__)
        if num_effects > 1:
            self.logger.warning(f"{num_effects} effects of the type {command_type} found in row. This isn't right.")

        return ret


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