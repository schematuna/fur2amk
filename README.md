fur2amk

Converts a Furnace SNES module to an AddMusicK txt file and corresponding BRR sample files.

Quick start (PowerShell)
1) get Python 3.9+

1) Convert a .fur to MML + BRRs
	python .\fur2amk.py ".\modules\Sunken Lights.fur"

2) Copy outputs into AddmusicK
	python ".\copy_to_amk.py" --amk-dir "..\AddmusicK_1.0.11" --song "Sunken Lights"

Outputs
- MML: .\music\Sunken Lights.txt
- Samples: .\music\Sunken Lights\*.brr