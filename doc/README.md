fur2amk

In progress and at best can only be used for noise music at the moment.

A converter that reads Furnace (.fur) modules and emits AddmusicK-compatible MML and BRR samples.

Highlights
- Parses INFO/SMP2/INS2/PATN blocks from .fur
- Exports samples as .brr (adds AMK 2-byte loop header)
- Generates MML with #amk/#spc/#path/#samples/#instruments and per-note sample switching
- Includes a helper script to copy outputs into AddmusicK

Quick start (PowerShell)
1) Convert a .fur to MML + BRRs
	python ".\fur2amk.py" ".\modules\YourSong.fur" --diag true

2) Copy outputs into AddmusicK
	python ".\copy_to_amk.py" --amk-dir "..\AddmusicK_1.0.11" --song "YourSong"

Outputs
- MML: .\music\YourSong.txt
- Samples: .\music\YourSong\*.brr (referenced by #path "YourSong")

Notes
- Requires Python 3.9+
- Optional: snesbrr.exe at ..\snesbrr\snesbrr.exe for WAV→BRR encoding; raw BRR from Furnace is written directly
- The BRR writer ensures (filesize - 2) is a multiple of 9 and sets the loop header from loop_start when available
