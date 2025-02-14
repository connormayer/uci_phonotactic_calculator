"""
io_utils.py - Utility module for I/O operations: reading tokens from a CSV file and writing results.
Version: 1.0.1
"""

import csv
from .constants import HEADER  # Import the centralized header

WORD_BOUNDARY = '#'

def read_tokens(dataset):
    """
    Reads in a file containing tokens and optional frequencies and converts
    it to a list of tokens and a list of token/frequency pairs.

    dataset: The path to the dataset.

    returns: A list of lists, where each sublist corresponds to a token and
             consists of a list of the individual symbols, plus a frequency if provided.
    """
    with open(dataset, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        token_freqs = []
        for row in reader:
            split_token = row[0].split(' ')
            freq = float(row[1]) if len(row) == 2 else 0  # Default frequency is 0
            token_freqs.append([split_token, freq])
    return token_freqs

def write_results(results, outfile):
    """
    Writes the results of scoring the test dataset to a CSV file.

    results: The list of result rows to write.
    outfile: The path to the output CSV file.

    returns: None
    """
    results = [HEADER] + results
    with open(outfile, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)