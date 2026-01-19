# Advanced Usage and Notes

## Requirements

- Requires furnace files saved in Furnace 0.6pre5 or later
- Requires all samples to be converted to BRR format prior to use

## ARAM Optimization

Furnace projects may require optimization if AMK throws an error about ARAM. There are a few ways to do this:

1. Decrease the SNES echo delay in the chip manager
2. Reduce sample sizes by downsampling or trimming
   - Need to switch to 8 or 16 bit PCM first, edit, then back to BRR
3. Replace interpolated commands with slide commands

## Gain Handling

If the gain macro is used in Furnace, the first gain value is used as the primary gain setting for the instrument. The second gain value is handled via a remote command. Any more gain values will not be converted.

If the gain macro is unused then the gain setting in the instrument SNES tab is used.

## Jump Commands

You can use one instance of the "Jump to Order" command 0Bxx. The last instance of the command will be used to place the intro marker in the amk output.

## Note Range

AMK only supports C1 -> A6. If any notes are out of this range they will be octave-shifted until they are in range. You can resolve this by retuning samples and find/replacing notes in Furnace to get them in range.

## Limitations

- **Compatibility Flags** are not supported. Conversion assumes that all compatibility flags are disabled.
- **Wavetables** are not supported. All instruments must use samples or noise.
- **Most Macros** are not yet supported. Only 1 Noise Freq value, up to 2 Gain values, and the Special Echo and Noise flags are supported. 
- **Alternating Speeds and Grooves** are not supported.