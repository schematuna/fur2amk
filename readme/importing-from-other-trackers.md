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