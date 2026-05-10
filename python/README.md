# fur2amk — Python / Command Line

## Prerequisites

- **Python 3.9 or higher**
  - Download from [python.org](https://www.python.org/downloads/)
  - **Important:** During installation, check "Add Python to PATH"
  - Verify: `python --version` should show 3.9 or higher

## Usage

```powershell
python .\fur2amk.py ".\examples\Frost Man.fur"
```

You can optionally edit `fur2amk_config.json` to set the AddmusicK directory:

```json
{
    "amk_dir": "../AddmusicK_1.0.11"
}
```

Leave `amk_dir` as `null` to disable automatic copying.
