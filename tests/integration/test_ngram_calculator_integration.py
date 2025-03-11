# tests/integration/test_ngram_calculator_integration.py
"""
test_ngram_calculator_integration.py - Integration tests for the ngram_calculator module.
Uses dummy training and test CSV files to verify that the n-gram calculator correctly reads,
processes, and writes results with the expected header and score formatting.
"""

import csv
import numpy as np
import pytest
from src.ngram_calculator import run_calculator
from src.constants import HEADER

def test_run_calculator_integration(dummy_train_file, dummy_test_file, tmp_path):
    # Create a temporary output file path.
    output_file = tmp_path / "output.csv"
    
    # Run the n-gram calculator with the dummy files.
    run_calculator(dummy_train_file, dummy_test_file, str(output_file))
    
    # Read and verify the output CSV file.
    with open(output_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Verify that the header matches the centralized header.
    assert rows[0] == HEADER, f"Header mismatch: {rows[0]} != {HEADER}"
    
    # Verify each result row has the same number of columns as the header.
    for row in rows[1:]:
        assert len(row) == len(HEADER), f"Row length {len(row)} does not match header length {len(HEADER)}"
        # Check that any infinite score is rendered as an empty string.
        for cell in row[2:]:
            try:
                # If cell converts to a float, ensure it is finite.
                val = float(cell)
                assert np.isfinite(val)
            except ValueError:
                # If conversion fails, the cell should be empty.
                assert cell == "", f"Expected empty string for infinite score, got {cell}"