import logging

class FurnaceUtil:
    PITCH_STEPS_PER_OCTAVE = 384

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def fur_pitch_change_to_semitones(change: int) -> float:
        semitones = change * 12 / FurnaceUtil.PITCH_STEPS_PER_OCTAVE
        return semitones

    @staticmethod
    def ticks_from_speed(speed: int, semitones: int) -> float:
        if speed == 0:
            # don't allow zero speed
            logging.warning("Pitch shift speed is zero, setting to 1 instead.")
            speed = 1
        ticks_per_octave = FurnaceUtil.PITCH_STEPS_PER_OCTAVE / speed
        octaves_to_slide = abs(semitones) / 12
        return ticks_per_octave * octaves_to_slide

    # Convert from Furnace stereo pan format (left and right, both 0->15)
    # to Furnace unity pan format (00=left, 80=center, FF=right)
    @staticmethod
    def stereo_to_unity_pan(left: int, right: int) -> int:
        # Clamp to valid range
        left = max(0, min(15, left))
        right = max(0, min(15, right))
        
        # Handle edge cases
        if left == 0 and right == 0:
            return 0x80
        
        # Calculate linear pan based on relative balance
        total = left + right
        level = round(255 * right / total)
        
        return max(0, min(255, level))