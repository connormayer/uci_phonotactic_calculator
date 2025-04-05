"""
tests/integration/test_ngram_calculator_integration.py - Integration tests for ngram_calculator.
Verifies that the n-gram calculator correctly reads, processes, and writes results with the expected dynamic header and score formatting.
"""

import csv
import numpy as np
import pytest
from src.ngram_calculator import run_calculator
from src.constants import get_header

def test_run_calculator_integration(dummy_train_file, dummy_test_file, tmp_path):
    # Create a temporary output file path.
    output_file = tmp_path / "output.csv"
    
    # Run the n-gram calculator with the dummy files.
    run_calculator(dummy_train_file, dummy_test_file, str(output_file))
    
    # Read and verify the output CSV file.
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Verify that the header matches the dynamically generated header.
    expected_header = get_header()
    assert rows[0] == expected_header, f"Header mismatch: {rows[0]} != {expected_header}"
    
    # Verify each result row has the same number of columns as the header.
    for row in rows[1:]:
        assert len(row) == len(expected_header), f"Row length {len(row)} does not match header length {len(expected_header)}"
        # Check that any infinite score is rendered as an empty string.
        for cell in row[2:]:
            try:
                # If cell converts to a float, ensure it is finite.
                val = float(cell)
                assert np.isfinite(val)
            except ValueError:
                # If conversion fails, the cell should be empty.
                assert cell == "", f"Expected empty string for infinite score, got {cell}"

# End of tests/integration/test_ngram_calculator_integration.py