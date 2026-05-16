from abc import ABC

class ChiptuneCommand(ABC):
    """Base class for all Chiptune commands."""
    pass

class PitchSlideCommand(ChiptuneCommand):
    """Pitch slide Command. Value is speed."""
    
    def __init__(self, change_per_tick):
        self.change_per_tick = change_per_tick

class NoteSlideCommand(ChiptuneCommand):
    """Note slide Command."""
    
    def __init__(self, speed: int, semitones: int):
        self.speed = speed
        self.semitones = semitones

class TuningCommand(ChiptuneCommand):
    """Sets tuning for the channel. Use decimals for in-between semitones."""

    def __init__(self, tuning: float):
        self.tuning = tuning

class PanCommand(ChiptuneCommand):
    """Value is pan position (0-255, 0=left, 128=center, 255=right)."""

    def __init__(self, pan_position: int):
        self.pan_position = pan_position

class PanFadeCommand(ChiptuneCommand):
    """Duration in ticks and target is final pan position (0-255, 0=left, 128=center, 255=right)."""

    def __init__(self, duration: int, target: int):
        self.duration = duration
        self.target = target

class LegatoEnableCommand(ChiptuneCommand):
    """Enables or disables legato for this channel."""

    def __init__(self, legato_on: bool):
        self.legato_on = legato_on

class VolumeFadeCommand(ChiptuneCommand):
    """duration is in furnace ticks, target is 0-254."""

    def __init__(self, duration: int, target: int):
        self.duration = duration
        self.target = target

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