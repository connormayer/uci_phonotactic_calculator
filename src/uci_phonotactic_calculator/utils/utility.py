"""utility.py — File validation and score‑filename helper functions."""

from os import path
from typing import Tuple


def valid_file(file_path: str) -> Tuple[bool, str]:
    """
    Check that the given file is comma‑delimited with space‑separated phonemes.

    Returns a tuple (is_valid, error_message). If valid, error_message is empty.
    """
    # First check if the file exists
    if not path.exists(file_path):
        return False, f"File not found: {file_path}"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Split into non‑empty lines
    raw_lines = [line for line in content.splitlines() if line]

    # Each line should be comma‑delimited: [phoneme_str, optional_freq]
    tokens = [line.split(",") for line in raw_lines]
    # The phoneme list is token[0] split on spaces
    tokens_no_freq = [tok[0].split(" ") for tok in tokens]

    # Disallow tab delimiters
    if any("\t" in line for line in raw_lines):
        return False, "Files must be comma-delimited."

    # Ensure at least one entry contains multiple phonemes
    if all(len(t) == 1 for t in tokens_no_freq):
        return False, "Phonemes must be separated by spaces."

    return True, ""


def get_filename(test_file: str, timestamp: float) -> str:
    """
    Given a test‑file path and a timestamp, construct an output filename
    under a timestamped folder.

    Example:
      test_file="data.csv", timestamp=1612345678.9
      → "data_scores.csv" inside folder "data_scores_1612345678_9"
    """
    base_name, ext = path.splitext(test_file)
    outfile = f"{base_name}_scores{ext}"
    ts = str(timestamp).replace(".", "_")
    folder = f"{outfile[:4]}_{ts}"
    return path.join(folder, outfile)


# End of src/utility.py
