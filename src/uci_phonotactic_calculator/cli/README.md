# cli

*Purpose* – This package handles command-line argument parsing, CLI output formatting, and execution of phonotactic calculations based on user input.

See the [parent README](../../README.md) for overall source context.

## What’s inside
- **`main.py`** – The main entry point for the CLI application. Orchestrates model training and scoring.
- **`parser.py`** – Defines the command-line argument structure using `argparse` and handles argument parsing.
- **`formatter.py`** – Manages the formatting of output data, typically into CSV.
- **`demo_data.py`** – Contains logic for loading and using built-in demo datasets.
- **`legacy.py`** – Provides compatibility layers or shims for older CLI behaviors or flags.
- **`extensions.py`** – Supports extending CLI functionality through a plugin-like mechanism for arguments.
- **`utils.py`** – Contains utility functions specific to CLI operations.
- **`__init__.py`** – Makes the `cli` directory a Python package.

## When you’d edit this folder
You would typically edit files in this folder when adding new command-line options, changing how results are displayed, or modifying the main CLI workflow.
