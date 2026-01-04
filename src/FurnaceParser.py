from __future__ import annotations

import io
import struct
import zlib
from typing import List, Tuple
import logging

from .model.FurnaceData import *
from .model.FurnaceEffects import FurnaceEffectFactory


class CompileErrorException(Exception):
    pass

class FurnaceParser:
    """
    Reader for Furnace .fur (INFO/SMP2/INS2/PATN).
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse(self, filename: str) -> FurnaceModule:
        self.logger.debug(f"Parsing {filename}")
        data = self._read_file_bytes(filename)
        # Try as-is, else zlib-decompress and retry
        if data and data[0] == 0x78:  # zlib magic byte
            try:
                data = zlib.decompress(data)
            except Exception:
                pass
        if not data.startswith(b"-Furnace module-"):
            raise CompileErrorException("Not a Furnace .fur file (magic not found)")

        mod = FurnaceModule()
        if (mod.NumChannels != 8):  # default for SNES
            self.logger.warning("Unsupported channel count; defaulting to 8 channels.")
            mod.NumChannels = 8

        # Keep data around for pointer-based seeks
        self._data = data
        bio = io.BytesIO(data)
        # Header (32 bytes)
        _ = bio.read(16)  # magic
        version = self._ru16(bio)
        # patterns require version 157+
        # instruments require version 127+
        # sound chip flags require version 118+
        # samples require version 102+
        if version < 157:
            raise CompileErrorException(f"Unsupported Furnace version {version}, cannot read patterns")
        bio.read(2)  # reserved
        info_ptr = self._ru32(bio)
        bio.read(8)  # reserved

        # First, try to read INFO at info_ptr
        inst_ptrs: List[int] = []
        samp_ptrs: List[int] = []
        patn_ptrs: List[int] = []
        try:
            if 0 < info_ptr < len(data) - 8:
                tag = data[info_ptr:info_ptr+4]
                size = int.from_bytes(data[info_ptr+4:info_ptr+8], 'little')
                if tag == b'INFO' and (info_ptr+8+size) <= len(data):
                    payload = data[info_ptr+8:info_ptr+8+size]
                    inst_ptrs, samp_ptrs, patn_ptrs, chip_flags_ptrs = self._parse_INFO(mod, io.BytesIO(payload))
        except Exception:
            # Fall back to scanning for INFO in the stream
            pass

        # Pointer-driven parse of SMP2, INS2, and PATN blocks
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
        for off in patn_ptrs:
            if 0 < off+8 <= len(data):
                tag = data[off:off+4]
                size = int.from_bytes(data[off+4:off+8], 'little')
                if tag == b'PATN' and off+8+size <= len(data):
                    self._parse_PATN(mod, io.BytesIO(data[off+8:off+8+size]))
        # Associate chip type bytes with chip flag pointers; capture SNES (0x87) if present
        for chip_byte, off in chip_flags_ptrs:
            if chip_byte == 0x87 and 0 < off + 8 < len(data):
                tag = data[off:off+4] # FLAG tag
                size = int.from_bytes(data[off+4:off+8], 'little')
                if tag == b'FLAG' and off+8+size <= len(data):
                    flag_data = data[off+8:off+8+size]
                    self._parse_FLAG(mod, flag_data)

        return mod

    # ------------- block handlers -------------

    def _parse_INFO(self, mod: FurnaceModule, s: io.BytesIO) -> Tuple[List[int], List[int], List[int], List[Tuple[int, int]]]:
        # Read subset as per docs; many fields will be ignored.
        # time base, speed1, speed2, arp time
        _tb = self._ru8(s); sp1 = self._ru8(s); sp2 = self._ru8(s); _arp = self._ru8(s)
        tps = self._rf32(s)  # ticks per second
        pat_len = self._ru16(s)
        ord_len = self._ru16(s)
        hlA = self._ru8(s); hlB = self._ru8(s)
        inst_count = self._ru16(s); wavetable_count = self._ru16(s); sample_count = self._ru16(s)
        mod.PatternLength = max(1, int(pat_len) or 64)
        # store timing
        mod.HighlightA = int(hlA) or 4
        mod.HighlightB = int(hlB) or 16
        mod.TicksPerSecond = float(tps)
        mod.Speed1 = int(sp1)
        mod.Speed2 = int(sp2)
        # global pattern count
        gpat_count = self._ru32(s)
        # Sound chip IDs (32 bytes)
        chips = s.read(32)
        # Assume SNES chip present (0x87), set channels to 8
        mod.NumChannels = 8
        # Ensure containers sized by channels
        if not mod.OrdersPerChannel or len(mod.OrdersPerChannel) != mod.NumChannels:
            mod.OrdersPerChannel = [[] for _ in range(mod.NumChannels)]
        if not mod.PatternsByChannel or len(mod.PatternsByChannel) != mod.NumChannels:
            mod.PatternsByChannel = [dict() for _ in range(mod.NumChannels)]
        # legacy fields per spec
        s.read(32)   # sound chip volumes (per-chip), see format.md
        s.read(32)   # sound chip panning (per-chip), see format.md
        chip_flags_ptrs = list(self._read_u32_list(s, 32))
        # Read song name and author
        name = self._rstr(s)
        author = self._rstr(s)
        if name:
            mod.SongName = name.replace('/', '-').replace('\\', '-')
        if author:
            mod.Author = author.replace('/', '-').replace('\\', '-')
        # tuning (unused)
        _a4 = self._rf32(s)
        # Read the 20-ish 1-byte compatibility flags up to pointer tables (match fur2tad order)
        for _i in range(20):
            s.read(1)
        # Now read pointers to instruments/wavetables/samples/patterns
        inst_ptrs = list(self._read_u32_list(s, inst_count))
        _wav_ptrs = list(self._read_u32_list(s, wavetable_count))
        samp_ptrs = list(self._read_u32_list(s, sample_count))
        patn_ptrs = list(self._read_u32_list(s, int(gpat_count)))
        # Orders and channel metadata
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
            pass
        mod.Comment = self._rstr(s)   # song comment
        mod.GV = float(self._rf32(s)) # 1.0f is 100%

        # Pair each chip type byte with its corresponding flags pointer
        chip_pairs: List[Tuple[int, int]] = []
        chip_bytes = list(chips)
        for i in range(min(len(chip_bytes), len(chip_flags_ptrs))):
            chip_pairs.append((int(chip_bytes[i]), int(chip_flags_ptrs[i])))

        return inst_ptrs, samp_ptrs, patn_ptrs, chip_pairs

    def _parse_SMP2(self, mod: FurnaceModule, s: io.BytesIO) -> None:
        name = self._rstr(s)
        length = self._ru32(s)
        comp_rate = self._ru32(s)
        c4_rate = self._ru32(s)
        depth = self._ru8(s)
        _loop_dir = self._ru8(s)
        _flags = self._ru8(s)
        _flags2 = self._ru8(s)
        loop_start = self._ri32(s)
        loop_end = self._ri32(s)
        s.read(16)  # presence bitfields
        # length is in samples, not bytes - brrs have 9 bytes per 16 samples
        num_bytes = (length + 15) // 16 * 9
        raw = s.read(num_bytes)
        idx = len(mod.Samples)
        samp = FurnaceSample(index=idx, name=self._sanitize_name(name or f'Sample{idx}'))
        samp.c4_rate = int(c4_rate) if c4_rate else None
        samp.sample_rate = int(comp_rate) if comp_rate else None
        samp.depth = int(depth or 16)
        if samp.depth == 9:
            # BRR data (9 bytes per block). Keep raw for direct write.
            samp.brr_raw = raw
        else:
            self.logger.error(f"Sample '{samp.name}' has unsupported depth {samp.depth} (only BRR/depth 9 is supported). Using empty BRR data.")
            # Create minimal valid BRR block: 9 bytes with end flag set
            samp.brr_raw = bytes([0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        # Loop markers
        if loop_start is not None and loop_end is not None and loop_start >= 0 and loop_end > loop_start:
            samp.loop_start = int(loop_start)
            samp.loop_end = int(loop_end)

        # debug
        # print(f"Loaded sample {samp.index}: '{samp.name}', length={length} samples, {num_bytes} bytes, depth={samp.depth}, c4_rate={samp.c4_rate}, sample_rate={samp.sample_rate}")
        # print(f"  Loop start: {samp.loop_start}, Loop end: {samp.loop_end}")
        mod.Samples.append(samp)

    def _parse_INS2(self, mod: FurnaceModule, s: io.BytesIO) -> None:
        _fmt_version = self._ru16(s)
        ins_type = self._ru16(s)
        if ins_type != 29:
            self.logger.warning(f"Unsupported instrument type {ins_type}, SNES instruments expected. Output may be incorrect.")
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
                    sn_flags = self._ru8(ds)
                    ins.sn_envelope_on = bool(sn_flags & 0x10)
                    # only in versions <137
                    sn_sustain_effective = bool(sn_flags & 0x08)
                    ins.gain_mode = sn_flags & 0x07
                if length >= 4:
                    ins.sn_gain = self._ru8(ds)
                if length >= 5:
                    val = self._ru8(ds)
                    # bits 5-6: sustain mode, bits 0-4: decay 2
                    ins.decay2 = val & 0x1F,  # bits 0-4
                    ins.sustain_mode = (val >> 5) & 0x03  # bits 5-6
            elif code == 'SM':
                # Sample instrument data: initial sample, flags, waveform len, sample map
                ds = io.BytesIO(data)
                if length >= 4:
                    ins.initial_sample = self._ru16(ds)
                    flags = self._ru8(ds)
                    ins.use_sample_map = bool(flags & 0x01)
                    _wav_len = self._ru8(ds)  # unused
                    # Sample map 120 entries if enabled
                    if ins.use_sample_map:
                        table: List[Tuple[int, int]] = []
                        for _ in range(120):
                            note_to_play = self._ru16(ds)
                            samp_to_play = self._ru16(ds)
                            table.append((note_to_play, samp_to_play))
                        if table:
                            ins.sample_table = table
            elif code == 'MA':
                # Instrument macro data per newIns.md
                ds = io.BytesIO(data)
                header_len = self._ru16(ds)

                def _read_macro_values(word_kind: int, count: int) -> Tuple[bytes, List[int]]:
                    # word_kind: 0=8u,1=8s,2=16s,3=32s
                    if word_kind == 0:
                        size, signed = 1, False
                    elif word_kind == 1:
                        size, signed = 1, True
                    elif word_kind == 2:
                        size, signed = 2, True
                    else:
                        size, signed = 4, True
                    byte_len = size * max(0, count)
                    raw = ds.read(byte_len)
                    vals: List[int] = []
                    try:
                        if size == 1:
                            if signed:
                                vals = list(struct.unpack('<' + 'b' * count, raw[:count]))
                            else:
                                vals = list(struct.unpack('<' + 'B' * count, raw[:count]))
                        elif size == 2:
                            vals = list(struct.unpack('<' + 'h' * count, raw[: 2 * count]))
                        else:
                            vals = list(struct.unpack('<' + 'i' * count, raw[: 4 * count]))
                    except Exception:
                        vals = []
                    return vals

                # Loop until code 255 (stop)
                while True:
                    # Peek next byte to ensure there is data
                    peek = ds.read(1)
                    if not peek:
                        break
                    # The interpretation of duty, wave and extra macros depends on chip/instrument type
                    # for snes
                    #  duty = pitch frew
                    #  wave = waveform
                    #  extra1 = special
                    #  extra2 = gain
                    macro_code = peek[0]
                    if macro_code == 255:
                        # Explicit end-of-macros marker
                        break
                    macro_length = self._ru8(ds)
                    macro_loop = self._ru8(ds)
                    macro_release = self._ru8(ds)
                    macro_mode = self._ru8(ds) # no idea what this is for
                    open_type_word = self._ru8(ds)
                    macro_delay = self._ru8(ds)
                    macro_speed = self._ru8(ds)

                    word_kind = (open_type_word >> 6) & 0x03
                    macro_type = (open_type_word >> 1) & 0x03  # 0=normal,1=ADSR,2=LFO
                    macro_open = bool(open_type_word & 0x01)
                    instant_rel = bool(open_type_word & 0x08)

                    values = _read_macro_values(word_kind, macro_length)

                    # Store macro by code; last occurrence wins if duplicates
                    ins.macros[int(macro_code)] = FurnaceMacro(
                        code=int(macro_code),
                        length=int(macro_length),
                        loop=int(macro_loop),
                        release=int(macro_release),
                        type=int(macro_type),
                        instant_release=instant_rel,
                        delay=int(macro_delay),
                        speed=int(macro_speed),
                        values=values,
                    )

                    ins.parse_snes_macro_flags()  # to process macros if needed
            elif code == 'EN':
                break
            else:
                # skip unknown feature
                pass

        mod.Instruments.append(ins)

    def _parse_PATN(self, mod: FurnaceModule, s: io.BytesIO) -> None:
        # Decode Furnace PATN block minimally (based on fur2tad logic)
        _song_index = self._ru8(s)
        channel = self._ru8(s)
        pat_index = self._ru16(s)
        _pat_name = self._rstr(s)
        # Ensure containers
        while len(mod.PatternsByChannel) < mod.NumChannels:
            mod.PatternsByChannel.append({})
        rows = [FurnaceRow() for _ in range(mod.PatternLength or 64)]
        idx = 0

        def read_effect(note: FurnaceRow, have_type: bool, have_value: bool):
            t = self._ru8(s) if have_type else None
            v = self._ru8(s) if have_value else None
            if (t is not None) and (v is None):
                v = 0
            if have_type or have_value:
                if t is None:
                    t = 0
                if v is None:
                    v = 0
                if t not in FurnaceEffectFactory.effect_map:
                    # print(f"Unknown effect type: {t:02X}")
                    return
                note.Effects.append(FurnaceEffectFactory.create_effect(t, v))

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
                note.Vol = min(255, vol * 2)  # scale to 0..7F to 0..255
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

    def _parse_FLAG(self, mod: FurnaceModule, data: bytes) -> None:
        # Parse key=value pairs from text data and store in mod.SNESFlags dict
        text = data.decode('utf-8', errors='replace')
        flags = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or '=' not in line:
                continue
            key, value = line.split('=', 1)
            flags[key.strip()] = value.strip()

        mod.SNESFlags.antiClick = bool(flags.get('antiClick', '0'))
        mod.SNESFlags.echo = bool(flags.get('echo', '0'))
        mod.SNESFlags.echoDelay = int(flags.get('echoDelay', '0'))
        mod.SNESFlags.echoFeedback = int(flags.get('echoFeedback', '0'))
        mod.SNESFlags.echoFilterCoeffs = [
            int(flags.get(f'echoFilter{i}', '0')) for i in range(8)
        ]
        mod.SNESFlags.echoMask = int(flags.get('echoMask', '0'))
        mod.SNESFlags.echoVolL = int(flags.get('echoVolL', '0'))
        mod.SNESFlags.echoVolR = int(flags.get('echoVolR', '0'))
        mod.SNESFlags.volScaleL = int(flags.get('volScaleL', '0'))
        mod.SNESFlags.volScaleR = int(flags.get('volScaleR', '0'))

        # from furnace docs: "scale volumes to prevent clipping/distortion"
        if mod.SNESFlags.volScaleL != mod.SNESFlags.volScaleR:
            print(f"Looks like you have different volume scales for left and right channels ({mod.SNESFlags.volScaleL} vs {mod.SNESFlags.volScaleR}).")
            print("AMK does not support volume scaling so this may not sound right.")

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