from abc import ABC

class FurnaceEffect(ABC):
    """Base class for all Furnace effects."""
    pass


class PitchSlideEffect(FurnaceEffect):
    """Pitch slide effect (0x01, 0x02). Value is speed."""
    
    def __init__(self, raw_value: int, is_up: bool):
        if raw_value == 0:
            self.change_per_tick = None
        else:
            if is_up:
                self.change_per_tick = raw_value
            else:
                self.change_per_tick = -raw_value

class PortamentoEffect(FurnaceEffect):
    """Portamento effect (0x03). Value is speed."""
    
    def __init__(self, raw_value: int):
        self.speed = raw_value

class NoteSlideEffect(FurnaceEffect):
    """Note slide effect (0xE1/0xE2). Value encodes speed (upper nibble) and semitones (lower nibble)."""
    
    def __init__(self, raw_value: int, is_up: bool):
        self.speed = raw_value >> 4
        self.semitones = raw_value & 0x0F
        if not is_up:
            self.semitones = -self.semitones

class StereoPanEffect(FurnaceEffect):
    """Stereo pan effect (0x08). Value encodes left (upper nibble) and right (lower nibble) volumes."""
    
    def __init__(self, raw_value: int):
        self.left_volume = raw_value >> 4
        self.right_volume = raw_value & 0x0F

class PanEffect(FurnaceEffect):
    """Pan effect (0x80). Value is pan position (0-255, 0=left, 128=center, 255=right)."""
    
    def __init__(self, raw_value: int):
        self.pan_position = raw_value

class PanSlideEffect(FurnaceEffect):
    """Pan slide effect (0x83). Value encodes left (upper nibble) and right (lower nibble) slide rates."""
    
    def __init__(self, raw_value: int):
        self.change_per_tick = None
        left = raw_value >> 4
        right = raw_value & 0x0F
        if right == 0 and left == 0:
            self.change_per_tick = None
        elif right == 0:
            # halved because pan is spread across both channels in Furnace
            self.change_per_tick = -left / 2
        elif left == 0:
            self.change_per_tick = right / 2
        else:
            print(f"Warning: Invalid pan slide effect value {raw_value}.")

class LegatoEffect(FurnaceEffect):
    """Legato effect (0xEA). Value is legato on/off."""
    
    def __init__(self, raw_value: int):
        self.legato_on = bool(raw_value)

class QuickLegatoEffect(FurnaceEffect):
    """Quick legato effect (0xE6). Value encodes semitones and delay."""
    
    def __init__(self, raw_value: int, is_fami: True, is_up: True):
        x = raw_value >> 4
        semitones = raw_value & 0x0F
        if is_fami: # famitracker-style command is a bit more complex to parse
            if x < 8:
                self.delay = x
            else:
                self.delay = x - 8
                semitones = -semitones
        else:
            self.delay = x
            if not is_up:
                semitones = -semitones

        self.semitones = semitones


class NoteDelayEffect(FurnaceEffect):
    """Note delay effect (0xED). Value is delay in ticks."""
    
    def __init__(self, raw_value: int):
        self.delay_ticks = raw_value
        

class VolumeSlideEffect(FurnaceEffect):
    """Fast volume slide effect (0xFA). Value encodes up (upper nibble) and down (lower nibble) rates."""
    
    def __init__(self, raw_value: int, is_fast: bool):
        self.change_per_tick = None
        rate_divisor = 4
        if is_fast:
            # fast volume slides are 4 times faster than normal volume slides
            rate_divisor = 1

        # doubled because furnace speed internally operates on 0->7F volume range
        # but the binary normalized volume to 0->FE, so in fur2amk we work with that range instead
        up = 2 * (raw_value >> 4)
        down = 2 * (raw_value & 0x0F)
        if down == 0 and up == 0:
            self.change_per_tick = None
        elif down == 0:
            self.change_per_tick = up / rate_divisor
        elif up == 0:
            self.change_per_tick = -(down / rate_divisor)
        else:
            print("Warning: Invalid volume slide effect value.")

class FineVolumeSlideEffect(FurnaceEffect):
    """Fine volume slide effect (0xF3, 0xF4). Value is fine rate."""
    
    def __init__(self, raw_value: int, is_up: bool):
        # fine volume slides are 64 times slower than normal volume slides
        self.change_per_tick = None
        # also doubled, same reason as normal volume slides
        speed = 2 * raw_value
        if is_up:
            if raw_value > 0:
                self.change_per_tick = speed / 256
        else:
            if raw_value > 0:
                self.change_per_tick = -(speed / 256)

class VibratoEffect(FurnaceEffect):
    def __init__(self, raw_value: int):
        # speed is number of table positions to move every tick
        # table is 64 positions long, so fastest rate is 15/64 cycles per tick, or ~4 cycles per tick
        self.speed = raw_value >> 4
        # max depth is +/-1 semitone
        self.depth = raw_value & 0x0F


class JumpToOrderEffect(FurnaceEffect):
    """Jump to order effect (0x0B). Value is order number."""
    
    def __init__(self, raw_value: int):
        self.order_number = raw_value

class JumpToNextPatternEffect(FurnaceEffect):
    """Jump to next pattern effect (0x0D). Value is row number to start on."""
    
    def __init__(self, raw_value: int):
        self.row_number = raw_value

class FurnaceEffectFactory:
    effect_map = {
            0x01: lambda v: PitchSlideEffect(v, True),
            0x02: lambda v: PitchSlideEffect(v, False),
            0x03: PortamentoEffect,
            0x04: VibratoEffect,
            0x08: StereoPanEffect,
            0x0A: lambda v: VolumeSlideEffect(v, False),
            0x0B: JumpToOrderEffect,
            0x0D: JumpToNextPatternEffect,
            0x80: PanEffect,
            0x83: PanSlideEffect,
            0xE1: lambda v: NoteSlideEffect(v, True),
            0xE2: lambda v: NoteSlideEffect(v, False),
            0xE6: lambda v: QuickLegatoEffect(v, True, None),   # quick legato
            0xE8: lambda v: QuickLegatoEffect(v, False, True),  # quick legaato up
            0xE9: lambda v: QuickLegatoEffect(v, False, False), # quick legato down
            0xEA: LegatoEffect,
            0xED: NoteDelayEffect,
            0xF3: lambda v: FineVolumeSlideEffect(v, True),
            0xF4: lambda v: FineVolumeSlideEffect(v, False),
            0xFA: lambda v: VolumeSlideEffect(v, True),
        }
    
    @staticmethod
    def create_effect(effect_type: int, value: int) -> FurnaceEffect:
        effect_class = FurnaceEffectFactory.effect_map[effect_type]
        return effect_class(value)
