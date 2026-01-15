# fur2amk

Converts a Furnace SNES module to an AddMusicK txt file and corresponding BRR sample files.

## Quick Start (PowerShell)

### Prerequisites

- **Python 3.9 or higher**
  - Download from [python.org](https://www.python.org/downloads/)
  - **Important:** During installation, check "Add Python to PATH" to use the `python` command
  - Verify installation: `python --version` should show Python 3.9 or higher
- **Furnace 0.6 or later**
- **AddmusicK 1.0.11**


### Usage

1. **Prepare your Furnace file:**
   - Must be a SNES module
   - Must have only BRR samples (convert all samples to BRR format in Furnace)
   - Should be fairly optimized (see [ADVANCED.md](ADVANCED.md) for optimization methods)

2. **Convert a `.fur` file to MML + BRRs:**
   ```powershell
   python .\fur2amk.py ".\examples\Sunken Lights.fur"
   ```

You can optionally edit `fur2amk_config.json` to set the AddmusicK directory path. If `amk_dir` is set, the output will automatically be copied to AddmusicK after conversion:

```json
{
    "amk_dir": "..\\AddmusicK_1.0.11"
}
```

Leave `amk_dir` as `null` to disable automatic copying.

## Outputs

- **MML:** `.\music\Sunken Lights.txt`
- **Samples:** `.\music\Sunken Lights\*.brr`
