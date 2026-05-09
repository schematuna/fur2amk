# Advanced Usage and Notes

## Requirements

- Requires furnace files saved in Furnace 0.6pre5 or later
- Must be built on Furnace's SNES system
- Requires all samples to be converted to BRR format prior to use

## ARAM Optimization

Furnace projects may require optimization if AMK throws an error about ARAM. There are a few ways to do this:

1. Decrease the SNES echo delay in the chip manager
2. Reduce sample sizes by downsampling or trimming
   - Need to switch to 8 or 16 bit PCM first, edit, then back to BRR
3. Replace interpolated commands with slide commands
4. Avoid excessive sequence macros

Additionally, fur2amk's MML loop optimization will never be as optimal or clean as what an experienced AMK porter 
can do by hand. As a final step, you may find it useful to optimize the MML output yourself.

This is an especially prudent step if you intend to submit the output to an archive like SMWCentral,
which has high moderation standards.

## Supported Features

- BBR samples
- Echo Settings (Song -> Chip Manager)
- Noise & Noise Freq (via instrument macros)
- Sample Maps
- All instrument envelope and gain types
- Vanilla Samples/Instruments

## Supported Effects

- 01: Pitch Slide Up
- 02: Pitch Slide Down
- 03: Portamento
- 04: Vibrato
- 08: Stereo Pan
- 0A: Volume Slide
- 0B: Jump to Order (1 occurrence allowed)
- 0D: Jump to Next Pattern
- 0F: Set speed
- 80: Pan
- 83: Pan Slide
- C:  Set Tick Rate (Hz)
- E1: Note Slide Up
- E2: Note Slide Down
- E5: Set Pitch
- E6: Quick Legato
- E8: Quick legato up
- E9: Quick legato down
- EA: Toggle Legato
- ED: Note Delay
- EE: Send External Command (for fur2amk hints, see below)
- F0: Set Tick Rate (BPM)
- F1: Single tick pitch up
- F2: Single tick pitch down
- F3: Fine Volume Slide Up
- F4: Fine Volume Slide Down
- F8: Single tick volume up
- F9: Single tick volume down
- FA: Fast Volume Slide

## Vanilla Samples

To use a vanilla SMW sample, put "@N" in the name of the sample in Furnace, where N is the stock SMW instrument whose sample you'd like to use. When the conversion runs, the original sample will be used rather than the Furnace sample data.

In the `templates` folder of the fur2amk download, there is a "Default Unsampled SMW" Furnace project with all of the vanilla samples already inserted and properly named. There are also instruments set up for them with stock SMW ADSRs as well. I recommend using this as a starting point if you'd like to make a vanilla port.

## Gain Handling

If the gain macro is used in Furnace, the first gain value is used as the primary gain setting for the instrument. The second gain value is handled via a remote command. Any more gain values will not be converted.

If the gain macro is unused then the gain setting in the instrument SNES tab is used.

## Jump Commands

You can use one instance of the "Jump to Order" command 0Bxx. The last instance of the command will be used to place the intro marker in the amk output.

## Note Range

AMK only supports C1 -> A6. If any notes are out of this range they will be octave-shifted until they are in range. You can resolve this by retuning samples and find/replacing notes in Furnace to get them in range.

## Limitations

- **Wavetables** are not supported. All instruments must use samples or noise.
- **Most Macros** are not yet supported. Only Volume, 1 Noise Freq value, up to 2 Gain values, and the Special Echo and Noise flags are supported. 
- **Alternating Speeds and Grooves** are not supported.
- **Compatibility Flags** are not supported. Conversion assumes that all compatibility flags are disabled.

## Removing the N-SPC 1-tick gap

By default, Super Mario World forces a 1-tick gap between successive notes, even if there is no gap in Furnace.

There is a trick to work around this using some crafty MML syntax. To enable this trick on a specific note, 
call the EE effect in Furnace with a value of 00 at the beginning of the note. 

Be careful when using this, as it can cause crackles in the AMK playback.

## Importing from other trackers

It is possible to use Furnace to convert files from other trackers like Impulse Tracker or FamiTracker, 
but some additional steps are required after import to prepare the project for fur2amk.

First:
- Import the tracker project file (xm, it, etc) to furnace (can just drag and drop onto furnace)
- Go to File->Manage Chips and Change Ensoniq ES5506 to SNES

Then for all samples:
- Create a meaningful name for the sample if name is blank
- Set their Type to BRR
- Fix the Loop Start and End points to be multiples of 16. Be mindful of changing sample tuning during this process.
- Set Mode to Forward - other modes are not supported by the SNES

For all instruments:
- Change their type to SNES
- Turn off "Use sample map" and select the correct sample in the main sample dropdown. 
	- This prevents fur2amk making a billion instruments
- Deselect any active "Special" macros, if necessary. These may get incorrectly converted from the Ensoniq chip

The project should now be convertible with fur2amk. If you find any other quirks that you are unable to work around,
feel free to reach out on discord.