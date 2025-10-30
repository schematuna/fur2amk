"""
fur2amk

First-pass scaffold to merge:
- Input parsing: Furnace (.fur) parsing logic (to be ported from fur2tad)
- Output generation: AMK MML generation logic (inspired by it2amk)

This file intentionally includes lightweight stubs for the Furnace parser and
event conversion so the CLI and MML writer can run now. We’ll wire real parsing
and conversion next by porting from fur2tad and mapping to the AMK event model.
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import io
import struct
import zlib


# --------------------------------------------------------------------------------------
# CLI helpers (previous path cache)

previous_textfile_filename = "fur2amk_last_run.txt"


def _get_previous_module() -> Optional[str]:
    try:
        with open(previous_textfile_filename, "r", encoding="utf-8") as f:
            result = f.read().strip()
            return result or None
    except FileNotFoundError:
        return None


def _save_previous_module(path: str) -> None:
    try:
        with open(previous_textfile_filename, "w", encoding="utf-8") as f:
            f.write(path)
    except OSError:
        pass


# --------------------------------------------------------------------------------------
# Config (ported structure from it2amk, simplified implementation)


class CompileErrorException(Exception):
    pass


class Config:
    flags: Dict[str, List[Any]] = {
        'nosmpl': [False, 'bool'],        # Skip sample conversion/dumping
        'diag': [False, 'bool'],          # Diagnostic logging
        'addmml': [[], None],             # List of MML snippets to add
        'game': ['', 'string'],           # Game title
        'author': ['', 'string'],         # Author name
        'length': ['', 'time'],           # SPC length
        'tmult': [2, 'real'],             # Tempo multiplier
        'vmult': [1.0, 'real'],           # Volume multiplier
        'chipc': [1, 'int'],              # Number of SPC chip instances
        'vcurve': ['accurate', 'string'], # accurate, linear, x^2
        'panning': ['accurate', 'string'],# linear, accurate
        'tspeed': [False, 'bool'],        # Use txxx for Axx commands
        'legato': [True, 'bool'],         # Whether or not to apply $F4 $02
        'vcmd': ['v', 'string'],          # Which volume command to use for the v column
        'mcmd': ['v', 'string'],          # Which volume command to use for the M effect
        'svcmd': ['v', 'string'],         # Which volume command to use for global sample volume
        'ivcmd': ['v', 'string'],         # Which volume command to use for global instrument volume
        'resample': [1.0, 'real'],        # Constant resample ratio across all samples
        'amplify': [0.92, 'real'],        # Constant amplify ratio across all samples
        'echo': ['', 'hex', 8],           # Echo parameters
        'fir': ['', 'hex', 16],           # Fir parameters
        'master': ['', 'hex', 4],         # Master level (left and right)
        # ARAM checking
        'aram_check': [True, 'bool'],           # Emit an ARAM usage warning after generation
        'aram_sample_budget_kb': [52, 'int'],   # Conservative sample budget in KB (approx)
    }

    flag_aliases: Dict[str, str] = {
        'ns': 'nosmpl',
        'mm': 'addmml',
        'gm': 'game',
        'au': 'author',
        'ln': 'length',
        't': 'tmult',
        'vm': 'vmult',
        'c': 'chipc',
        'vc': 'vcurve',
        'p': 'panning',
        'ts': 'tspeed',
        'l': 'legato',
        'v': 'vcmd',
        'm': 'mcmd',
        'sv': 'svcmd',
        'iv': 'ivcmd',
        'r': 'resample',
        'a': 'amplify',
        'e': 'echo',
        'f': 'fir',
        'ml': 'master',
    }

    @staticmethod
    def flag(name: str) -> Any:
        if name in Config.flags:
            return Config.flags[name][0]
        # try alias lookup
        if name in Config.flag_aliases:
            return Config.flags[Config.flag_aliases[name]][0]
        raise KeyError(f"Unknown flag '{name}'")

    @staticmethod
    def set_flag(flag: str, value: str) -> None:
        # alias expansion
        key = Config.flag_aliases.get(flag, flag)
        if key not in Config.flags:
            raise KeyError(f"Unknown flag '{flag}'")

        current = Config.flags[key]
        default_val, ftype = current[0], current[1]

        if ftype is None:
            # addmml special-case: allow repeated values
            if key == 'addmml':
                current[0].append(value)
            else:
                current[0] = value
            return

        if ftype == 'bool':
            if isinstance(value, bool):
                current[0] = value
            else:
                v = str(value).strip().lower()
                if v in ('1', 'true', 'yes', 'y', 'on'):
                    current[0] = True
                elif v in ('0', 'false', 'no', 'n', 'off'):
                    current[0] = False
                else:
                    raise ValueError(f"Invalid bool for {key}: {value}")
        elif ftype == 'int':
            current[0] = int(value)
        elif ftype == 'real':
            current[0] = float(value)
        elif ftype == 'string' or ftype == 'time':
            current[0] = str(value)
        elif ftype == 'hex':
            # enforce exact hex length if provided (third entry)
            hex_len = current[2] if len(current) > 2 else None
            vv = value.strip().lower().removeprefix('0x').replace(' ', '')
            if hex_len is not None and len(vv) not in (hex_len, hex_len * 2):
                # allow bytes (space-less) or nibble count; keep simple
                # we won’t normalize here; we just store the string
                pass
            # basic validate
            int(vv or '0', 16)
            current[0] = vv
        else:
            current[0] = value

# --------------------------------------------------------------------------------------
# Furnace parsing (stub) and adapter types


@dataclass
class FurnaceSample:
    index: int
    name: str
    # Minimal fields needed for AMK sample list (expand later):
    brr_path: Optional[str] = None
    brr_raw: Optional[bytes] = None  # Raw BRR data if sample is stored as BRR
    c4_rate: Optional[int] = None  # Hz
    vol: int = 64  # 0..64
    pan: int = 128  # 0..255 center
    # Raw PCM payload and metadata from SMP2
    pcm16: List[int] = field(default_factory=list)  # mono 16-bit samples
    sample_rate: Optional[int] = None
    depth: int = 16
    loop_start: Optional[int] = None
    loop_end: Optional[int] = None


@dataclass
class FurnaceInstrument:
    index: int
    name: str
    gbv: int = 64  # instrument global volume
    dfp: int = 128 # default pan
    # SNES ADSR/GAIN from INS2 'SN'
    sn_attack: Optional[int] = None  # 0..15
    sn_decay: Optional[int] = None   # 0..7
    sn_sustain: Optional[int] = None # 0..7
    sn_release: Optional[int] = None # 0..31
    sn_flags: Optional[int] = None   # bit4 envelope on, bits0..2 gain mode
    sn_gain: Optional[int] = None    # 0..255 raw gain value
    sn_decay2susmode: Optional[int] = None
    # Sample mapping from INS2 'SM'
    initial_sample: Optional[int] = 0  # sample 0 by default
    use_sample_map: bool = False
    sample_table: List[Tuple[int, int]] = field(default_factory=lambda: [(0, 1)] * 120)


@dataclass
class FurnacePatternRow:
    # Extremely simplified row placeholder
    Note: Optional[int] = None  # 0..119, 254=cut, 255=off
    Ins: Optional[int] = None
    Vol: Optional[int] = None   # 0..64
    Pan: Optional[int] = None   # 0..255
    Effects: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class FurnacePattern:
    rows: List[List[FurnacePatternRow]] = field(default_factory=list)  # 64 x channels


@dataclass
class FurnaceModule:
    # A normalized adapter exposing the subset EventTable/MML expect
    SongName: str = ''
    Message: str = ''
    GV: int = 128               # global volume (0..128)
    IT: int = 125               # tempo
    IS: int = 6                 # speed (ticks per row)
    # Legacy placeholders (unused by the new path)
    Orders: List[int] = field(default_factory=list)
    Patterns: Dict[int, List[List[FurnacePatternRow]]] = field(default_factory=dict)  # legacy, not used
    Instruments: List[FurnaceInstrument] = field(default_factory=list)
    Samples: List[FurnaceSample] = field(default_factory=list)
    NumChannels: int = 8
    # New structures for pattern conversion
    PatternLength: int = 64
    OrdersPerChannel: List[List[int]] = field(default_factory=list)  # [ch][order_idx] -> pattern_id
    PatternsByChannel: List[Dict[int, List[FurnacePatternRow]]] = field(default_factory=list)  # [ch][pat_id] -> rows
    # Timing
    HighlightA: int = 4
    HighlightB: int = 16
    TicksPerSecond: float = 0.0
    Speed1: int = 6
    Speed2: int = 0


class FurnaceParser:
    """Placeholder: Wire real parsing from fur2tad.FurnaceFile next.

    Minimal reader: scan blocks, extract song name (INFO/SONG), samples (SMP2),
    and instrument placeholders (INS2). Patterns/orders are ignored for now.
    """

    def parse(self, filename: str) -> FurnaceModule:
        data = self._read_file_bytes(filename)
        # Try as-is, else zlib-decompress and retry
        if data[0] == 0x78:  # zlib magic byte
            data = zlib.decompress(data)
        if not data.startswith(b"-Furnace module-"):
            raise CompileErrorException("Not a Furnace .fur file (magic not found)")

        mod = FurnaceModule()
        mod.SongName = os.path.splitext(os.path.basename(filename))[0]
        mod.NumChannels = 8  # default for SNES

        # Keep data around for pointer-based seeks
        self._data = data
        bio = io.BytesIO(data)
        # Header (32 bytes)
        magic = bio.read(16)
        version = self._ru16(bio)
        bio.read(2)  # reserved
        info_ptr = self._ru32(bio)
        bio.read(8)  # reserved

        # First, try to read INFO at info_ptr
        inst_ptrs: List[int] = []
        samp_ptrs: List[int] = []
        try:
            if 0 < info_ptr < len(data) - 8:
                tag = data[info_ptr:info_ptr+4]
                size = int.from_bytes(data[info_ptr+4:info_ptr+8], 'little')
                if tag == b'INFO' and (info_ptr+8+size) <= len(data):
                    payload = data[info_ptr+8:info_ptr+8+size]
                    inst_ptrs, samp_ptrs = self._parse_INFO(mod, io.BytesIO(payload))
        except Exception:
            # Fall back to scanning for INFO in the stream
            pass

        # Pointer-driven parse of SMP2 and INS2 blocks
        for off in samp_ptrs:
            if 0 < off+8 <= len(data):
                tag = data[off:off+4]
                size = int.from_bytes(data[off+4:off+8], 'little')
                if tag == b'SMP2' and off+8+size <= len(data):
                    self._parse_SMP2(mod, io.BytesIO(data[off+8:off+8+size]))
        for off in inst_ptrs:
            if 0 < off+8 <= len(data):
                tag = data[off:off+4]
                size = int.from_bytes(data[off+4:off+8], 'little')
                if tag == b'INS2' and off+8+size <= len(data):
                    self._parse_INS2(mod, io.BytesIO(data[off+8:off+8+size]))

        # Finally, scan for PATN blocks to populate pattern data
        self._scan_and_parse_blocks(mod, data, b'PATN', lambda m, s: self._parse_PATN(m, s))

        # Ensure at least one pattern for downstream assumptions
        if not mod.Patterns:
            mod.Orders = [0]
            empty_row = FurnacePatternRow()
            mod.Patterns[0] = [[empty_row for _ in range(mod.NumChannels)] for __ in range(64)]

        return mod

    # ------------- block handlers -------------

    def _parse_INFO(self, mod: FurnaceModule, s: io.BytesIO) -> Tuple[List[int], List[int]]:
        # Read subset as per docs; many fields will be ignored.
        # time base, speed1, speed2, arp time
        tb = self._ru8(s); sp1 = self._ru8(s); sp2 = self._ru8(s); arp = self._ru8(s)
        tps = self._rf32(s)  # ticks per second
        pat_len = self._ru16(s)
        ord_len = self._ru16(s)
        hlA = self._ru8(s); hlB = self._ru8(s)
        inst_count = self._ru16(s); wavetable_count = self._ru16(s); sample_count = self._ru16(s)
        mod._inst_count = inst_count  # type: ignore[attr-defined]
        mod.PatternLength = max(1, int(pat_len) or 64)
        # store timing
        mod.HighlightA = int(hlA) or 4
        mod.HighlightB = int(hlB) or 16
        mod.TicksPerSecond = float(tps)
        mod.Speed1 = int(sp1)
        mod.Speed2 = int(sp2)
        # global pattern count
        gpat_count = self._ru32(s)
        chips = s.read(32)
        # If first chip is SNES (0x87), set channels to 8
        if len(chips) >= 1 and chips[0] == 0x87:
            mod.NumChannels = 8
        # Ensure containers sized by channels
        if not mod.OrdersPerChannel or len(mod.OrdersPerChannel) != mod.NumChannels:
            mod.OrdersPerChannel = [[] for _ in range(mod.NumChannels)]
        if not mod.PatternsByChannel or len(mod.PatternsByChannel) != mod.NumChannels:
            mod.PatternsByChannel = [dict() for _ in range(mod.NumChannels)]
        # legacy fields per spec
        s.read(32)   # sound chip volumes
        s.read(32)   # sound chip panning
        s.read(128)  # sound chip flag pointers / flags
        # Read song name and author
        name = self._rstr(s)
        author = self._rstr(s)
        if name:
            mod.SongName = name.replace('/', '-').replace('\\', '-')
        # tuning
        a4 = self._rf32(s)
        # Read the 20-ish 1-byte compatibility flags up to pointer tables (match fur2tad order)
        for _i in range(20):
            s.read(1)
        # Now read pointers to instruments/wavetables/samples/patterns
        inst_ptrs = list(self._read_u32_list(s, inst_count))
        wav_ptrs = list(self._read_u32_list(s, wavetable_count))
        samp_ptrs = list(self._read_u32_list(s, sample_count))
        # skip pattern pointers array
        _ = s.read(4 * int(gpat_count))
        # After pointers, read orders and channel metadata (common song data)
        # Orders: for each of 8 channels, ord_len bytes
        try:
            for ch in range(mod.NumChannels):
                col = []
                for _i in range(ord_len):
                    col.append(self._ru8(s))
                mod.OrdersPerChannel[ch] = col
            # Skip effect column counts, channel flags, and names
            s.read(mod.NumChannels)  # effect_column_count
            s.read(mod.NumChannels)  # channels_hidden
            s.read(mod.NumChannels)  # channels_collapsed
            for _ in range(mod.NumChannels):
                _ = self._rstr(s)  # channel name
            for _ in range(mod.NumChannels):
                _ = self._rstr(s)  # short channel name
        except Exception:
            # If orders parsing fails, leave OrdersPerChannel empty to avoid misleading output
            pass
        return inst_ptrs, samp_ptrs

    def _parse_SONG(self, mod: FurnaceModule, s: io.BytesIO) -> None:
        # Similar to INFO but for subsongs; we only care about the name as fallback.
        s.read(1)  # time base
        s.read(1)  # speed1
        s.read(1)  # speed2
        s.read(1)  # arp
        _ = self._rf32(s)  # ticks per second
        _ = self._ru16(s)  # pattern len
        _ = self._ru16(s)  # orders len
        s.read(1); s.read(1)  # highlight A/B
        _ = self._ru16(s)  # virt tempo num
        _ = self._ru16(s)  # virt tempo den
        name = self._rstr(s)
        comment = self._rstr(s)
        if name and (not getattr(mod, 'SongName', None)):
            mod.SongName = name.replace('/', '-').replace('\\', '-')

    def _parse_SMP2(self, mod: FurnaceModule, s: io.BytesIO) -> None:
        name = self._rstr(s)
        length = self._ru32(s)
        comp_rate = self._ru32(s)
        c4_rate = self._ru32(s)
        print(f"Debug: Parsing SMP2 sample '{name}' length={length} comp_rate={comp_rate} c4_rate={c4_rate}", file=sys.stderr)
        depth = self._ru8(s)
        loop_dir = self._ru8(s)
        flags = self._ru8(s)
        flags2 = self._ru8(s)
        loop_start = self._ri32(s)
        loop_end = self._ri32(s)
        s.read(16)  # presence bitfields
        raw = s.read(length)
        idx = len(mod.Samples)
        samp = FurnaceSample(index=idx, name=self._sanitize_name(name or f'Sample{idx}'))
        samp.c4_rate = int(c4_rate) if c4_rate else None
        samp.sample_rate = int(comp_rate) if comp_rate else None
        samp.depth = int(depth or 16)
        # Interpret raw payload: depth 16/8 are PCM, depth 9 is BRR blocks
        pcm16: List[int] = []
        try:
            if samp.depth == 16:
                n = len(raw) // 2
                pcm16 = list(struct.unpack('<' + 'h' * n, raw[: n * 2]))
                samp.pcm16 = pcm16
            elif samp.depth == 8:
                # Signed 8-bit to 16-bit
                pcm16 = [int(struct.unpack('<b', bytes([b]))[0]) << 8 for b in raw]
                samp.pcm16 = pcm16
            elif samp.depth == 9:
                # BRR data (9 bytes per block). Keep raw for direct write.
                print(f"Debug: Sample '{samp.name}' is BRR data, storing raw BRR", file=sys.stderr)
                samp.brr_raw = raw
                samp.pcm16 = []
            else:
                samp.pcm16 = []
        except Exception:
            samp.pcm16 = []
        # Loop markers
        if loop_start is not None and loop_end is not None and loop_start >= 0 and loop_end > loop_start:
            samp.loop_start = int(loop_start)
            samp.loop_end = int(loop_end)
        mod.Samples.append(samp)

    def _parse_INS2(self, mod: FurnaceModule, s: io.BytesIO) -> None:
        fmt_version = self._ru16(s)
        ins_type = self._ru16(s)
        if ins_type != 29:
            print(f"Warning: INS2 instrument type {ins_type} not supported, only SNES samples allowed.")
        idx = len(mod.Instruments)
        ins = FurnaceInstrument(index=idx, name=f'Inst{idx}')
        # Parse features until EN
        while True:
            code_b = s.read(2)
            if len(code_b) < 2:
                break
            length = self._ru16(s)
            data = s.read(length)
            code = code_b.decode('ascii', errors='ignore')
            if code == 'NA':
                # instrument name as STR
                name_stream = io.BytesIO(data)
                nm = self._rstr(name_stream)
                if nm:
                    ins.name = self._sanitize_name(nm)
            elif code == 'SN':
                # SNES ADSR/gain per newIns.md
                ds = io.BytesIO(data)
                if length >= 2:
                    ad = self._ru8(ds)
                    sr = self._ru8(ds)
                    ins.sn_attack = ad & 0x0F
                    ins.sn_decay = (ad >> 4) & 0x07
                    ins.sn_sustain = (sr >> 5) & 0x07
                    ins.sn_release = sr & 0x1F
                if length >= 3:
                    ins.sn_flags = self._ru8(ds)
                if length >= 4:
                    ins.sn_gain = self._ru8(ds)
                if length >= 5:
                    ins.sn_decay2susmode = self._ru8(ds)
            elif code == 'SM':
                # Sample instrument data: initial sample, flags, waveform len, sample map
                ds = io.BytesIO(data)
                if length >= 4:
                    ins.initial_sample = self._ru16(ds)
                    flags = self._ru8(ds)
                    ins.use_sample_map = bool(flags & 0x01)
                    wav_len = self._ru8(ds)  # unused
                    # Sample map 120 entries if enabled
                    if ins.use_sample_map:
                        table: List[Tuple[int, int]] = []
                        for _ in range(120):
                            note_to_play = self._ru16(ds)
                            samp_to_play = self._ru16(ds)
                            table.append((note_to_play, samp_to_play))
                        if table:
                            ins.sample_table = table
            elif code == 'EN':
                break
            else:
                # skip unknown feature
                # usually MA (macro data) or NE (NES DPCM sample map data)
                pass
        mod.Instruments.append(ins)

        # If no SM was parsed, we assume it's using sample 0 by default
        try:
            if bool(Config.flag('diag')) and (ins.initial_sample == 0):
                print(f"[diag]   INS2 inst {idx:02d}: no SM feature present, assuming default instrument with sample 0")
        except Exception:
            pass

        # print instrument number, name, sample number, and if it uses the sample map
        try:
            if bool(Config.flag('diag')):
                sm_text = "with sample map" if ins.use_sample_map else "no sample map"
                print(f"[diag]   INS2 inst {idx:02d}: '{ins.name}', initial sample {ins.initial_sample}, {sm_text}")
        except Exception:
            pass

    def _parse_PATN(self, mod: FurnaceModule, s: io.BytesIO) -> None:
        # Decode Furnace PATN block minimally (based on fur2tad logic)
        song_index = self._ru8(s)  # ignore
        channel = self._ru8(s)
        pat_index = self._ru16(s)
        _pat_name = self._rstr(s)
        # Ensure containers
        while len(mod.PatternsByChannel) < mod.NumChannels:
            mod.PatternsByChannel.append({})
        rows = [FurnacePatternRow() for _ in range(mod.PatternLength or 64)]
        idx = 0
        def read_effect(note: FurnacePatternRow, have_type: bool, have_value: bool):
            t = self._ru8(s) if have_type else None
            v = self._ru8(s) if have_value else None
            if (t is not None) and (v is None):
                v = 0
            if have_type or have_value:
                if t is None:
                    t = 0
                if v is None:
                    v = 0
                note.Effects.append((t, v))
        while idx < len(rows):
            b = self._ru8(s)
            if b == 0xFF:
                break
            if b & 0x80:
                idx += 2 + (b & 0x7F)
                continue
            note = rows[idx]
            eff1 = None
            eff2 = None
            if b & 0x20:
                eff1 = self._ru8(s)
            if b & 0x40:
                eff2 = self._ru8(s)
            if b & 0x01:
                note.Note = self._ru8(s)
            if b & 0x02:
                note.Ins = self._ru8(s)
            if b & 0x04:
                vol = self._ru8(s)
                note.Vol = min(255, vol * 2 + (vol & 1))  # scale to 0-255 like fur2tad
            # effects in first column
            read_effect(note, bool(b & 0x08), bool(b & 0x10))
            # expanded effects masks in eff1/eff2
            def handle_mask(mask: int):
                read_effect(note, bool(mask & 0x04), bool(mask & 0x08))
                read_effect(note, bool(mask & 0x10), bool(mask & 0x20))
                read_effect(note, bool(mask & 0x40), bool(mask & 0x80))
            if eff1 is not None:
                handle_mask(eff1)
            if eff2 is not None:
                handle_mask(eff2)
            idx += 1
        # Store
        if channel < len(mod.PatternsByChannel):
            mod.PatternsByChannel[channel][pat_index] = rows

    # ------------- helpers -------------

    def _read_file_bytes(self, filename: str) -> bytes:
        with open(filename, 'rb') as f:
            return f.read()

    def _ru8(self, s: io.BytesIO) -> int:
        b = s.read(1)
        return b[0] if b else 0

    def _ru16(self, s: io.BytesIO) -> int:
        b = s.read(2)
        if len(b) < 2:
            return 0
        return int.from_bytes(b, 'little', signed=False)

    def _ru32(self, s: io.BytesIO) -> int:
        b = s.read(4)
        if len(b) < 4:
            return 0
        return int.from_bytes(b, 'little', signed=False)

    def _ri32(self, s: io.BytesIO) -> int:
        b = s.read(4)
        if len(b) < 4:
            return 0
        return int.from_bytes(b, 'little', signed=True)

    def _rf32(self, s: io.BytesIO) -> float:
        b = s.read(4)
        if len(b) < 4:
            return 0.0
        return struct.unpack('<f', b)[0]

    def _rstr(self, s: io.BytesIO) -> str:
        out = bytearray()
        while True:
            c = s.read(1)
            if not c:
                break
            if c == b'\x00':
                break
            out.extend(c)
        try:
            return out.decode('utf-8', errors='replace')
        except Exception:
            return ''

    def _sanitize_name(self, text: str) -> str:
        # Keep alnum, space, underscore, dash; replace others with underscore.
        return ''.join(ch if (ch.isalnum() or ch in ' _-') else '_' for ch in text).strip() or 'Sample'

    def _read_u32_list(self, s: io.BytesIO, n: int):
        for _ in range(int(n)):
            b = s.read(4)
            if len(b) < 4:
                yield 0
            else:
                yield int.from_bytes(b, 'little', signed=False)

    def _scan_and_parse_blocks(self, mod: FurnaceModule, data: bytes, tag: bytes, handler):
        i = 0
        n = len(data)
        while True:
            i = data.find(tag, i)
            if i < 0 or i + 8 > n:
                break
            size = int.from_bytes(data[i+4:i+8], 'little', signed=False)
            if size <= 0 or i + 8 + size > n:
                i += 1
                continue
            try:
                handler(mod, io.BytesIO(data[i+8:i+8+size]))
            except Exception:
                # skip malformed occurrence
                pass
            i += 8 + size


# --------------------------------------------------------------------------------------
# Event model (simplified, compatible shape for MML)


class EventState:
    def __init__(self) -> None:
        # Mirror keys used by it2amk where possible
        self.state_d: Dict[str, Any] = {
            '': None, 'M': None, 'S': 0x90, 'X': 0x80,
            'E': 0x00, 'H': 0x00, 'I': 0x00, 'J': 0x00,
            'Q': 0x00, 'R': 0x00, 'v': None, '@': None,
            'IV': None, 'SV': None, 'EV': None, 'EX': 32, 'EE': None,
            'eflag': False, 'pflag': False, 'H': 0x00, 'Hon': False,
            'Z1': None,
            'a': 0x00, 'b': 0x00, 'c': 0x00, 'd': 0x00, 'l': 0x00, 'r': 0x00,
            'D': 0x00, 'N': 0x00, 'P': 0x00,
        }


class Event:
    def __init__(self, tick: int, effect: str, value: Any, visible: bool = True) -> None:
        self.tick = tick
        self.effect = effect
        self.value = value
        self.visible = visible


class EventTable:
    """Build an event list from the FurnaceModule.

    MVP: produce empty tracks; later, port full row/effect traversal from fur2tad.
    """

    def __init__(self, module: FurnaceModule) -> None:
        self.events: List[List[Event]] = [[] for _ in range(8)]
        self.g_events: List[Event] = []
        self.module = module
        self.states = [EventState() for _ in range(8)]
        self.g_state_d: Dict[str, Any] = {'T': None, 'V': None}
        self.used_samples: set[int] = set()
        # map sample index -> (filename, tuning_hex)
        self.sample_dict: Dict[int, Tuple[str, str]] = {}
        self.ins_dict: Dict[int, Any] = {}
        # List of (instrument_index, sample_index) pairs to emit in #instruments
        self.ins_list: List[Tuple[int, int]] = []
        self.loop_tick: int = 0
        self.convert()

    def convert(self) -> None:
        # TODO: Traverse Orders/Patterns from FurnaceModule and emit per-channel events
        # For now: initialize T/V once at tick 0 for header completeness
        self.g_events.append(Event(0, 'T', self.module.IT, visible=False))
        self.g_events.append(Event(0, 'V', self.module.GV, visible=False))
        # Build trivial instrument/sample dictionary from module
        for s in self.module.Samples:
            self.used_samples.add(s.index)
            # Build sample filename and tuning string
            fname = f"{s.index:02d}_" + (s.name or f"Sample{s.index}").replace(' ', '_') + '.brr'
            tuning_word = 0x0100
            try:
                if s.c4_rate and s.c4_rate > 0:
                    # MAGIC NUMBERS to convert from c4_rate to AMK instrument tuning value
                    # stolen from it2amk's SampConv
                    val = int(round(float(s.c4_rate) * 768 / 12539))
                    tuning_word = max(0, min(0xFFFF, val))
            except Exception:
                tuning_word = 0x0100
            tune_str = f"${(tuning_word >> 8) & 0xFF:02X} ${(tuning_word & 0xFF):02X}"
            self.sample_dict[s.index] = (fname, tune_str)
        # For each instrument, gather unique samples referenced by its sample map
        for ins in self.module.Instruments:
            samples_for_ins: List[int] = []
            try:
                if ins.use_sample_map and ins.sample_table:
                    # Collect unique, non-zero sample indices from the 120-entry map
                    uniq: List[int] = []
                    seen = set()
                    for (_note_to_play, samp_to_play) in ins.sample_table:
                        # Furnace SM uses 0-based sample indices
                        try:
                            sidx1 = int(samp_to_play)
                        except Exception:
                            sidx1 = -1
                        if sidx1 < 0:
                            continue
                        if sidx1 not in seen:
                            seen.add(sidx1)
                            uniq.append(sidx1)
                    samples_for_ins = uniq
                else:
                    # Use initial_sample (0-based) if available
                    try:
                        if ins.initial_sample is not None and int(ins.initial_sample) >= 0:
                            sidx1 = int(ins.initial_sample)
                        else:
                            sidx1 = 0
                    except Exception:
                        sidx1 = 0
                    samples_for_ins = [sidx1]
            except Exception:
                try:
                    sidx1 = int(ins.initial_sample) if (ins.initial_sample is not None and int(ins.initial_sample) >= 0) else 0
                except Exception:
                    sidx1 = 0
                samples_for_ins = [sidx1]
            # Track used samples and populate instrument entries
            for sidx in samples_for_ins:
                self.used_samples.add(sidx)
                self.ins_list.append((ins.index, sidx))


# --------------------------------------------------------------------------------------
# MML writer (streamlined for now)


class MMLState:
    def __init__(self) -> None:
        self.state_d: Dict[str, Any] = {
            'o': None, 'h': 0, 'v': None, 'q': None,
            'tune': 0x00, 'y': (10, 0, 0), 'p': (0, 0, 0), 'trem': (0, 0, 0),
            'echo': 0x00, '@': 0, 'dgain': None, 'note': None,
            'echof': False, 'n': None, 'amp': 0x00, 'gain': None,
        }
        self.hstate_d: Dict[str, Any] = {
            '': None, 'M': None, 'S': 0x90, 'X': 0x80,
            'E': 0x00, 'H': 0x00, 'I': 0x00, 'J': 0x00,
            'Q': 0x00, 'R': 0x00, 'v': None, '@': None,
            'IV': None, 'SV': None, 'EV': None, 'EX': None, 'EE': None, 'H': 0x00,
            'Z1': None,
        }


class MML:
    def __init__(self, event_table: EventTable, module_path: str) -> None:
        self.txt: str = ''
        self.event_table = event_table
        self.module_path = module_path
        self.states = [MMLState() for _ in range(8)]
        self.g_state: Dict[str, Any] = {'evoll': 0, 'evolr': 0}
        self.echo_set = False

        self.add_amk_header()
        self.add_spc_info()
        self.add_sample_info()
        self.add_ins_info()
        self.add_init_info()
        self.convert()

    # Sections
    def add_amk_header(self) -> None:
        self.txt += '#amk 2\n\n'

    # --- helpers ---
    def _row_kind(self, row: FurnacePatternRow) -> str:
        """Classify a Furnace row for emission.

        Returns: 'note' | 'off' | 'cut' | 'empty'.
        OFF = 255, CUT = 254 as per parser comment.
        """
        n = row.Note
        if n is None:
            return 'empty'
        try:
            v = int(n)
        except Exception:
            return 'empty'
        if v == 255:
            return 'off'
        if v == 254:
            return 'cut'
        if 0 <= v <= 179:
            return 'note'
        return 'empty'

    def _divisors(self, n: int) -> List[int]:
        n = int(n)
        if n <= 0:
            return [1]
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return sorted(divs)

    def _run_to_denoms(self, run_rows: int, base_den: int) -> List[int]:
        """Decompose a run of rows into a list of AMK length denominators to tie.

        Each row is 1/base_den. We choose chunks that are divisors of base_den
        and sum to run_rows. For each chunk, the length number is base_den/chunk.
        Example: base_den=16, run=3 -> chunks [2,1] => denoms [8,16] -> c8^16.
        """
        run = max(1, int(run_rows))
        bd = max(1, int(base_den))
        divs = self._divisors(bd)
        # allowed chunks are divisors of base_den
        chunks = sorted(divs, reverse=True)
        out: List[int] = []
        rem = run
        while rem > 0:
            # pick largest chunk <= rem
            pick = None
            for c in chunks:
                if c <= rem:
                    pick = c
                    break
            if pick is None:
                # fallback to 1-row chunks (shouldn't happen since 1 divides bd)
                pick = 1
            out.append(bd // pick)
            rem -= pick
        return out

    def add_spc_info(self) -> None:
        # Emit AddmusicK readme-style #spc block with #title/#game/#author/#length
        lines = ['#spc', '{']
        mod = self.event_table.module
        title = getattr(mod, 'SongName', '') or ''
        if title:
            lines.append(f'  #title "{title}"')
        if Config.flag('game'):
            lines.append(f'  #game "{Config.flag("game")}"')
        if Config.flag('author'):
            lines.append(f'  #author "{Config.flag("author")}"')
        if Config.flag('length'):
            lines.append(f'  #length "{Config.flag("length")}"')
        # Optional comment: use first line of Message if present
        msg = str(getattr(mod, 'Message', '') or '').strip()
        if msg:
            first_line = msg.splitlines()[0]
            lines.append(f'  #comment "{first_line}"')
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_sample_info(self) -> None:
        path_name = os.path.splitext(os.path.basename(self.module_path.replace('\\', '/')))[0]
        sample_dir = os.path.join('music', path_name)
        os.makedirs(sample_dir, exist_ok=True)
        # Attempt to dump samples to BRR files (unless disabled)
        if not bool(Config.flag('nosmpl')):
            self._dump_samples_to_brr(sample_dir)
        sample_lines = [f'#path "{path_name}"', '', '#samples', '{', '  #optimized']
        add_any = False
        # Prefer listing only BRRs we actually generated to avoid missing files
        mod = self.event_table.module
        for samp in sorted(mod.Samples, key=lambda x: x.index):
            base = f"{samp.index:02d}_" + (samp.name or f'Sample{samp.index}').replace(' ', '_')
            brr_rel = f'{base}.brr'
            brr_abs = os.path.join(sample_dir, brr_rel)
            if os.path.exists(brr_abs) and os.path.getsize(brr_abs) > 0:
                # Match AMK style: list quoted filenames only
                sample_lines.append(f'  "{brr_rel}"')
                add_any = True
        sample_lines.append('}')
        # Even if no extra BRRs, we still want #samples { #optimized } for clarity
        self.txt += '\n'.join(sample_lines) + '\n\n'

    def add_ins_info(self) -> None:
        if not self.event_table.ins_list:
            return
        lines = ['#instruments', '{']
        # Assign AMK instrument numbers starting at 30 in the order we emit
        # Map of (instrument_index, sample_index) -> AMK instrument number
        self.insnum_map: Dict[Tuple[int, int], int] = {}
        next_num = 30
        for ins_idx, samp_idx in self.event_table.ins_list:
            # Resolve sample filename and tuning
            samp_entry = self.event_table.sample_dict.get(samp_idx)
            if not samp_entry:
                # Fallback to first sample
                samp_entry = next(iter(self.event_table.sample_dict.values()), ("Sample1.brr", "$01 $00"))
            samp_name, samp_tuning = samp_entry
            # ADSR/GAIN
            ins = self.event_table.module.Instruments[ins_idx]
            # Default: no envelope -> $00 $00 $7F
            da = 0x00
            sr = 0x00
            ga = 0x7F
            try:
                if ins.sn_flags is not None and (ins.sn_flags & 0x10):
                    d = int(ins.sn_decay or 0)
                    a = int(ins.sn_attack or 0)
                    ssv = int(ins.sn_sustain or 0)
                    rv = int(ins.sn_release or 0)
                    da = ((d & 0x7) | 0x8) << 4 | (a & 0xF)
                    sr = ((ssv & 0x7) << 5) | (rv & 0x1F)
                    # When ADSR is on, GA is ignored by AMK; keep placeholder $7F.
                else:
                    # ADSR off: use raw GAIN value if provided (clamped to 7-bit)
                    if ins.sn_gain is not None:
                        ga = max(0, min(0x7F, int(ins.sn_gain)))
            except Exception:
                pass
            lines.append(f'  "{samp_name}" ${da:02X} ${sr:02X} ${ga:02X} {samp_tuning} ;@{next_num}')
            self.insnum_map[(ins_idx, samp_idx)] = next_num
            next_num += 1
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_init_info(self) -> None:
        # Basic init MML stub; can be expanded to include echo/fir/master
        addmml = Config.flag('addmml')
        if addmml:
            self.txt += '\n'.join(addmml) + '\n\n'

    # Conversion
    def convert(self) -> None:
        # If we have parsed orders/patterns, emit simple note streams with basic durations per channel.
        mod = self.event_table.module
        if getattr(mod, 'OrdersPerChannel', None) and getattr(mod, 'PatternsByChannel', None) and any(mod.OrdersPerChannel):
            # Compute shared defaults once
            base_den = 4 * (mod.HighlightA or 4)
            if base_den <= 0:
                base_den = 16
            # Compute AMK tempo so that one row matches Furnace timing approximately
            tps = float(getattr(mod, 'TicksPerSecond', 0.0) or 0.0)
            spd = int(getattr(mod, 'Speed1', 0) or 0)
            if spd <= 0:
                spd = 6
            if tps > 0:
                bpm = max(1, int(round(240.0 * tps / (base_den * spd))))
            else:
                bpm = int(getattr(mod, 'IT', 125) or 125)

            amk_tempo = bpm * 8192 // 20025
            self.txt += f't{amk_tempo}\n\n'

            for c in range(mod.NumChannels):
                self.txt += f'#%d\n' % c
                # No default length: emit explicit lengths for every note/rest
                current_oct = None
                current_ins = None  # Furnace instrument index
                current_amk_ins: Optional[int] = None  # AMK @ number actually in use
                current_vol: Optional[int] = None  # 0..255
                tokens: List[str] = []
                orders = mod.OrdersPerChannel[c] if c < len(mod.OrdersPerChannel) else []
                patmap = mod.PatternsByChannel[c] if c < len(mod.PatternsByChannel) else {}
                # Flatten rows for this channel
                flat_rows: List[FurnacePatternRow] = []
                for pat in orders:
                    rows = patmap.get(pat)
                    if rows:
                        flat_rows.extend(rows)
                    else:
                        flat_rows.extend([FurnacePatternRow() for _ in range(mod.PatternLength)])

                i = 0
                N = len(flat_rows)
                while i < N:
                    row = flat_rows[i]
                    kind = self._row_kind(row)
                    # Track instrument changes (don’t emit @ yet; defer until note to choose sample variant)
                    if row.Ins is not None and row.Ins != current_ins and row.Ins != 255:
                        current_ins = int(row.Ins)
                        current_amk_ins = None  # force re-select on next note

                    # Determine if this is a note or rest
                    if kind == 'note':
                        note_idx = int(row.Note)  # type: ignore[arg-type]
                        # Ensure we have some instrument context
                        if current_ins is None or current_ins == 255:
                            current_ins = 0
                            current_amk_ins = None
                        # Determine which sample this note should use for this instrument
                        amk_num = self._resolve_amk_instrument_for_note(current_ins, note_idx)
                        if amk_num is not None and amk_num != current_amk_ins:
                            tokens.append(f'@{amk_num}')
                            current_amk_ins = amk_num
                        name, octv = self._note_name_and_octave(note_idx)
                        if current_oct != octv:
                            tokens.append(f'o{octv}')
                            current_oct = octv
                        # Apply volume if present on this row and changed
                        if row.Vol is not None:
                            v = max(0, min(255, int(row.Vol)))
                            if current_vol != v:
                                tokens.append(f'v{v}')
                                current_vol = v
                        # Count run length of same note continuing (no new note starts)
                        run = 1
                        j = i + 1
                        while j < N:
                            r2 = flat_rows[j]
                            k2 = self._row_kind(r2)
                            # stop if next row starts a new note, OFF/CUT, or instrument change
                            if (k2 in ('note','off','cut')) or (r2.Ins is not None and r2.Ins != current_ins and r2.Ins != 255):
                                break
                            run += 1
                            j += 1
                        # Emit note with duration expressed as ties if run>1
                        # Always emit explicit duration numbers and numeric ties
                        denoms = self._run_to_denoms(run, base_den)
                        # First segment includes note name
                        tokens.append(f'{name}{denoms[0]}')
                        # Subsequent segments are ties with numbers only
                        for d in denoms[1:]:
                            tokens.append(f'^{d}')
                        i = j
                        continue
                    else:
                        # Rest or OFF/CUT run
                        run = 1
                        j = i + 1
                        while j < N:
                            r2 = flat_rows[j]
                            if self._row_kind(r2) == 'note':
                                break
                            run += 1
                            j += 1
                        # Always emit explicit rest duration and numeric ties
                        denoms = self._run_to_denoms(run, base_den)
                        tokens.append(f'r{denoms[0]}')
                        for d in denoms[1:]:
                            tokens.append(f'^{d}')
                        i = j
                        continue
                self.txt += ' '.join(tokens) + '\n\n'
            return
        # Fallback: emit 8 empty channels as before
        for c in range(8):
            self.txt += f'#%d\n' % c
            self.txt += '\n'

    def _dump_samples_to_brr(self, out_dir: str) -> None:
        """Write PCM to temporary WAVs and encode to BRR using snesbrr.exe if available."""
        mod = self.event_table.module
        if not getattr(mod, 'Samples', None):
            return
        # Resolve encoder path
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.normpath(os.path.join(here, '..', 'snesbrr', 'snesbrr.exe')),
        ]
        encoder = next((p for p in candidates if os.path.exists(p)), None)
        total = 0
        with_pcm = 0
        created = 0
        for s in mod.Samples:
            total += 1
            # Target BRR path
            # Prefix with index to avoid name collisions and keep ordering stable
            fname_base = (f"{s.index:02d}_" + (f"{s.name}".strip() or f"Sample{s.index}")).replace(' ', '_')
            brr_path = os.path.join(out_dir, fname_base + '.brr')
            s.brr_path = brr_path
            # Always overwrite existing BRR: remove it first if present
            try:
                if os.path.exists(brr_path):
                    os.remove(brr_path)
                    if bool(Config.flag('diag')):
                        print(f"[diag] removed existing BRR: {os.path.basename(brr_path)}")
            except OSError:
                pass
            # If the sample already contains raw BRR data, wrap it with AMK 2-byte loop header and write
            if s.brr_raw:
                try:
                    data = s.brr_raw
                    # Ensure (len - 2) % 9 == 0 by adding header; raw data itself should be multiple of 9
                    if len(data) % 9 != 0:
                        # Truncate to the nearest lower whole block to satisfy AMK; log diagnostics
                        trunc = (len(data) // 9) * 9
                        if bool(Config.flag('diag')):
                            print(f"[diag] warning: BRR raw not block-aligned ({len(data)}); truncating to {trunc}")
                        data = data[:trunc]
                    loop_off = 0
                    if s.loop_start is not None and s.loop_start >= 0:
                        # Convert PCM loop start (samples) to BRR byte offset: floor(loop/16)*9
                        loop_off = (int(s.loop_start) // 16) * 9
                    header = bytes((loop_off & 0xFF, (loop_off >> 8) & 0xFF))
                    with open(brr_path, 'wb') as f:
                        f.write(header + data)
                    created += 1
                    if bool(Config.flag('diag')):
                        print(f"[diag] wrote BRR (raw+hdr): {os.path.basename(brr_path)} loop_off={loop_off}")
                    continue
                except Exception:
                    if bool(Config.flag('diag')):
                        print(f"[diag] failed to write raw BRR for {s.index:02d} {s.name}")
                    # fall through to try PCM encode if available
            # If we don't have PCM, skip
            if not s.pcm16:
                if bool(Config.flag('diag')):
                    print(f"[diag] skip: no PCM for {s.index:02d} {s.name}")
                continue
            with_pcm += 1
            # Write WAV
            wav_path = os.path.join(out_dir, fname_base + '.wav')
            try:
                with wave.open(wav_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    # Prefer a standard 32000 Hz output to match SNES native rate
                    rate = 32000
                    wf.setframerate(rate)
                    # Pack samples little-endian
                    frames = struct.pack('<' + 'h' * len(s.pcm16), *s.pcm16)
                    wf.writeframes(frames)
            except Exception:
                # Couldn't write WAV; skip encoding
                if bool(Config.flag('diag')):
                    print(f"[diag] failed to write WAV for {s.index:02d} {s.name}")
                continue
            # Encode to BRR (overwrite target)
            if encoder:
                cmd = [encoder, '--encode']
                # Loop start in samples (WAV frames), if available
                if s.loop_start is not None and s.loop_start >= 0:
                    # Align to BRR block boundary (16 samples)
                    ls = int(s.loop_start)
                    if ls < 0:
                        ls = 0
                    ls &= ~0xF
                    cmd += ['--loop-start', str(ls)]
                cmd += [wav_path, brr_path]
                try:
                    if bool(Config.flag('diag')):
                        print(f"[diag] encode: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    # Leave WAV; user can encode manually if needed
                    if bool(Config.flag('diag')):
                        print(f"[diag] encoder failed for {s.index:02d} {s.name}")
                    pass
            # If an encoder was used and produced a raw BRR without AMK header, prepend 2-byte loop header
            try:
                if os.path.exists(brr_path) and os.path.getsize(brr_path) > 0:
                    with open(brr_path, 'rb') as f:
                        br = f.read()
                    # If file already seems to have header (len-2 divisible by 9), keep; else add header
                    needs_header = (len(br) - 2) % 9 != 0
                    if needs_header:
                        loop_off = 0
                        if s.loop_start is not None and s.loop_start >= 0:
                            loop_off = (int(s.loop_start) // 16) * 9
                        header = bytes((loop_off & 0xFF, (loop_off >> 8) & 0xFF))
                        # Ensure payload is integral blocks
                        if len(br) % 9 != 0:
                            trunc = (len(br) // 9) * 9
                            if bool(Config.flag('diag')):
                                print(f"[diag] warning: encoder BRR not block-aligned ({len(br)}); truncating to {trunc}")
                            br = br[:trunc]
                        with open(brr_path, 'wb') as f:
                            f.write(header + br)
                        if bool(Config.flag('diag')):
                            print(f"[diag] inserted AMK header: {os.path.basename(brr_path)} loop_off={loop_off}")
            except Exception:
                pass
            # Optionally remove WAV to keep folder clean if BRR exists
            try:
                if os.path.exists(brr_path) and os.path.getsize(brr_path) > 0:
                    created += 1
                    os.remove(wav_path)
                    if bool(Config.flag('diag')):
                        print(f"[diag] created: {os.path.basename(brr_path)}")
            except OSError:
                pass
        if bool(Config.flag('diag')):
            print(f"[diag] summary: samples={total} with_pcm={with_pcm} brr_created={created}")

    def _note_name_and_octave(self, i: int) -> Tuple[str, int]:
        # Map Furnace note index (0=C-0) to AMK note name and octave using oN
        names = ['c', 'c+', 'd', 'd+', 'e', 'f', 'f+', 'g', 'g+', 'a', 'a+', 'b']
        note = i % 12
        octave = i // 12 - 5  # align with fur2tad convention
        return names[note], octave

    def _resolve_amk_instrument_for_note(self, ins_idx: int, note_idx: int) -> Optional[int]:
        """Pick the AMK instrument number for this Furnace instrument at a given note.

        Uses the instrument's sample map (INS2 'SM') when present; else the initial sample.
        """
        try:
            mod = self.event_table.module
            if ins_idx <= 0 or ins_idx > len(mod.Instruments):
                return None
            ins = mod.Instruments[ins_idx]

            # Determine sample index to use
            samp_idx: Optional[int] = None
            n = int(note_idx)
            if ins.use_sample_map and ins.sample_table:
                # Furnace provides 120 entries; clamp into range
                n120 = n
                if n120 < 0:
                    n120 = 0
                if n120 >= len(ins.sample_table):
                    n120 = n120 % len(ins.sample_table)
                _note_to_play, samp_to_play = ins.sample_table[n120]
                try:
                    sidx_raw = int(samp_to_play)
                except Exception:
                    sidx_raw = -1
                if sidx_raw >= 0:
                    samp_idx = sidx_raw + 1
            if samp_idx is None:
                try:
                    if ins.initial_sample is not None and int(ins.initial_sample) >= 0:
                        samp_idx = int(ins.initial_sample)
                    else:
                        samp_idx = 0
                except Exception:
                    samp_idx = 1
            # Map to AMK instrument number
            if hasattr(self, 'insnum_map') and isinstance(self.insnum_map, dict):
                return self.insnum_map.get((ins_idx, int(samp_idx)))
            # Fallback: sequential mapping (unlikely to be correct, but avoids crash)
            return 30 + int(ins_idx)
        except Exception:
            return None

    # Output
    def save(self, filename: str) -> None:
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.txt)


# --------------------------------------------------------------------------------------
# Main


def parse_cli(argv: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    if len(argv) < 2:
        prev = _get_previous_module()
        usage = (
            'Usage: python fur2amk.py <furnace_file.fur> <flags>\n' +
            (f'Previous: {prev}\n' if prev else '')
        )
        print(usage)
        sys.exit(1)

    module_path = argv[1]
    if not os.path.exists(module_path):
        print(f"Error: {module_path} does not exist.")
        sys.exit(1)

    if len(argv) >= 2 and len(argv) % 2 != 0:
        print('Error: Missing flag argument (flags must be in pairs).')
        sys.exit(1)

    pairs: List[Tuple[str, str]] = []
    i = 2
    while i < len(argv):
        pairs.append((argv[i], argv[i + 1]))
        i += 2
    return module_path, pairs


def main() -> None:
    module_path, flag_pairs = parse_cli(sys.argv)

    # Apply CLI flags
    for flag, arg in flag_pairs:
        name = flag.lstrip('-').strip()
        try:
            Config.set_flag(name, arg)
        except (ValueError, KeyError) as e:
            print(f"Flag error for '{flag}': {e}")
            sys.exit(1)

    # Load module (Furnace)
    parser = FurnaceParser()
    module = parser.parse(module_path)

    # Build events and MML
    evtbl = EventTable(module)
    mml = MML(evtbl, module_path)

    # Output
    song_name = os.path.splitext(os.path.basename(module_path))[0]
    out_path = os.path.join('music', f'{song_name}.txt')
    mml.save(out_path)
    print(f"Wrote {out_path}")

    # Cache last path
    _save_previous_module(module_path)


if __name__ == "__main__":
    main()
