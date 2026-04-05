from abc import ABC

class ChiptuneCommand(ABC):
    """Base class for all Chiptune commands."""
    pass

class PitchSlideCommand(ChiptuneCommand):
    """Pitch slide Command (0x01, 0x02). Value is speed."""
    
    def __init__(self, change_per_tick):
        self.change_per_tick = change_per_tick

class NoteSlideCommand(ChiptuneCommand):
    """Note slide Command (0xE1/0xE2). Value encodes speed (upper nibble) and semitones (lower nibble)."""
    
    def __init__(self, speed: int, semitones: int):
        self.speed = speed
        self.semitones = semitones

class TuningCommand(ChiptuneCommand):
    """Sets tuning for the channel. Use decimals for in-between semitones."""

    def __init__(self, tuning: float):
        self.tuning = tuning

class StereoPanCommand(ChiptuneCommand):
    """Stereo pan Command (0x08). Value encodes left (upper nibble) and right (lower nibble) volumes."""

    def __init__(self, left_volume: int, right_volume: int):
        self.left_volume = left_volume
        self.right_volume = right_volume

class PanCommand(ChiptuneCommand):
    """Pan Command (0x80). Value is pan position (0-255, 0=left, 128=center, 255=right)."""

    def __init__(self, pan_position: int):
        self.pan_position = pan_position

class PanSlideCommand(ChiptuneCommand):
    """Pan slide Command (0x83). change_per_tick is the rate of pan change per tick."""

    def __init__(self, change_per_tick):
        self.change_per_tick = change_per_tick

class LegatoEnableCommand(ChiptuneCommand):
    """Enables or disables legato for this channel."""

    def __init__(self, legato_on: bool):
        self.legato_on = legato_on

class VolumeSlideCommand(ChiptuneCommand):
    """Volume slide Command. change_per_tick is the rate of volume change per tick."""

    def __init__(self, change_per_tick):
        self.change_per_tick = change_per_tick

class FineVolumeSlideCommand(ChiptuneCommand):
    """Fine volume slide Command. change_per_tick is the fine rate of volume change per tick."""

    def __init__(self, change_per_tick):
        self.change_per_tick = change_per_tick

class VibratoCommand(ChiptuneCommand):
    def __init__(self, speed: int, depth: int):
        self.speed = speed
        self.depth = depth

class EchoEnableCommand(ChiptuneCommand):
    """Enables or disables echo for this channel"""

    def __init__(self, echo_on: bool):
        self.echo_on = echo_on

class SetTickRateCommand(ChiptuneCommand):
    """Sets the Tick Rate. tick_rate is from 000 to 3FF in Hz."""

    def __init__(self, tick_rate: int):
        self.tick_rate = tick_rate

class SendExternalCommand(ChiptuneCommand):
    """Special command for fur2amk hints. Valid values:
        00 - Remove 1-tick gap between this note and the next note
    """

    def __init__(self, value: int):
        self.value = value