"""
Helper script to print the absolute paths to packaged example data files.

Provides easy access to data files for CLI usage.
"""

import argparse
import importlib.resources as pkg_resources
import sys


def list_data_files(subdir=None):
    base = "uci_phonotactic_calculator.data"
    if subdir:
        base = f"{base}.{subdir}"
    return [str(f) for f in pkg_resources.files(base).iterdir() if f.is_file()]


def main():
    parser = argparse.ArgumentParser(
        description="Print absolute paths to packaged data files."
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default=None,
        help="Subdirectory under data (e.g. sample_test_data)",
    )
    args = parser.parse_args()
    files = list_data_files(args.subdir)
    if not files:
        print("No data files found.", file=sys.stderr)
        sys.exit(1)
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
