"""
tests/common_helpers.py - Consolidated helper functions for n-gram tests.
Provides common utility functions to compute expected probabilities.
"""

import numpy as np

def log_expected(matrix):
    """
    Given a 2D list or array of probabilities, returns the natural logarithm
    of the array as a NumPy array. Any entries that are zero or below are
    replaced with -inf explicitly, to avoid runtime warnings from log(0).

    Parameters:
      matrix (list or array): 2D list/array of probabilities.

    Returns:
      numpy.ndarray: A NumPy array of the same shape, where each cell is
                     either the log(probability) if prob > 0, or -inf if prob <= 0.
    """
    arr = np.array(matrix, dtype=float)
    # Suppress divide-by-zero warnings when computing np.log(arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(arr > 0, np.log(arr), float('-inf'))
    return result

# End of tests/common_helpers.py