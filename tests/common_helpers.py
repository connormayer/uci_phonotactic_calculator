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
    result = np.empty_like(arr)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if val <= 0:
                result[i, j] = float('-inf')
            else:
                result[i, j] = np.log(val)

    return result

# End of tests/common_helpers.py