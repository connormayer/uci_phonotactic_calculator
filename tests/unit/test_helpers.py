"""
test_helpers.py - Helper functions for n-gram tests.
Provides common utility functions to compute expected probabilities.
"""

import numpy as np

def log_expected(matrix):
    """
    Given a 2D list or array of probabilities, returns the natural logarithm
    of the array as a NumPy array.

    Parameters:
      matrix (list or array): 2D list/array of probabilities.

    Returns:
      numpy.ndarray: Logarithm of the probabilities.
    """
    return np.log(np.array(matrix))