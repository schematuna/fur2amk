# fur2amk

Converts a Furnace SNES module to an AddMusicK txt file and corresponding BRR sample files.

Open fur2amk_gui.exe to get started.

Alternatively, you can run the python scripts manually if desired. See the README in the python directory.

See ADVANCED.md for detailed support and usage documentation.

## Quick Start

### Prerequisites

- **Furnace 0.6 or later** : https://tildearrow.org/furnace/
- **AddmusicK 1.0.11** : https://www.smwcentral.net/?p=section&a=details&id=37906

### Prepare your Furnace file

- Must use the SNES system
- Must have only BRR samples (convert all samples to BRR format in Furnace)
- Should be fairly optimized (see [ADVANCED.md](ADVANCED.md) for optimization methods)

### Using the GUI

1. Run `fur2amk_gui.exe`
2. Browse for your `.fur` file
3. Optionally set the **AMK directory** — if set, outputs are copied to AddmusicK automatically
4. Click **Convert**

Outputs are placed in the `music\` folder:

- **MML:** `music\Frost Man.txt`
- **Samples:** `music\Frost Man\*.brr`

