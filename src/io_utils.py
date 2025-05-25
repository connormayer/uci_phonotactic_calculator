"""
src/io_utils.py - Utility module for I/O operations: reading tokens from a CSV file and writing results.
This module now uses a dynamically generated header via get_header() from constants.

"""

import csv
from constants import get_header

WORD_BOUNDARY = "#"


def read_tokens(dataset):
    """
    Reads in a file containing tokens and optional frequencies, and converts it to a list of token/frequency pairs.

    Parameters:
      dataset (str): The path to the dataset.

    Returns:
      list: A list of lists, where each sublist corresponds to a token and consists of:
            - A list of the individual symbols (token).
            - A frequency (float). If no frequency is provided in the CSV row, a default value of 1 is used.
    """
    with open(dataset, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        token_freqs = []
        for row in reader:
            split_token = row[0].split(" ")
            freq = (
                float(row[1]) if len(row) == 2 else 1
            )  # Default frequency is 1 if not provided
            token_freqs.append([split_token, freq])
    return token_freqs


def write_results(results, outfile):
    """
    Writes the results of scoring the test dataset to a CSV file.

    Parameters:
      results (list): The list of result rows to write.
      outfile (str): The path to the output CSV file.

    Returns:
      None
    """
    # Dynamically generate header using get_header()
    results = [get_header()] + results
    with open(outfile, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)


# End of src/io_utils.py
