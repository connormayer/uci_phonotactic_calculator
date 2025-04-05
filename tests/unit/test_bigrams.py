"""
tests/unit/test_bigrams.py - Unit tests for bigram models.
Verifies that each cell in the bigram probability matrices is computed correctly.
Tests both joint and conditional models with and without word boundaries.
"""

import numpy as np
import pytest
from src.io_utils import WORD_BOUNDARY
from src.ngram_models import (
    fit_bigram_conditional_nonpos_wb,
    fit_bigram_joint_nonpos_wb,
    fit_bigram_conditional_nonpos_nwb,
    fit_bigram_joint_nonpos_nwb
)
from tests.common_helpers import log_expected

def assert_matrix_close(actual, expected, atol=1e-6):
    rows, cols = expected.shape
    for i in range(rows):
        for j in range(cols):
            assert np.isclose(actual[i, j], expected[i, j], atol=atol), \
                f"Mismatch at cell ({i}, {j}): {actual[i,j]} != {expected[i,j]}"

# Existing tests for conditional models with word boundaries
def test_fit_bigrams_conditional_WB(token_freqs, unique_sounds):
    sounds = unique_sounds + [WORD_BOUNDARY]
    bigram_probs = fit_bigram_conditional_nonpos_wb(token_freqs, sounds, token_weighted=False, smoothed=False)
    expected_probs = log_expected([
        [1/9, 6/7, 2/5],
        [3/9, 1/7, 3/5],
        [5/9, 0/7, 0/5]
    ])
    assert_matrix_close(bigram_probs, expected_probs)

def test_fit_bigrams_conditional_WB_smoothed(token_freqs, unique_sounds):
    sounds = unique_sounds + [WORD_BOUNDARY]
    bigram_probs = fit_bigram_conditional_nonpos_wb(token_freqs, sounds, token_weighted=False, smoothed=True)
    expected_probs = log_expected([
        [2/12, 7/10, 3/8],
        [4/12, 2/10, 4/8],
        [6/12, 1/10, 1/8]
    ])
    assert_matrix_close(bigram_probs, expected_probs)

def test_fit_bigrams_conditional_WB_weighted(token_freqs, unique_sounds):
    sounds = unique_sounds + [WORD_BOUNDARY]
    bigram_probs = fit_bigram_conditional_nonpos_wb(token_freqs, sounds, token_weighted=True, smoothed=False)
    wb_t = np.log(10)*2 + np.log(30)
    wb_a = np.log(20)*2
    wb_wb = 0
    t_a = np.log(10)*2 + np.log(20)*2 + np.log(30)*2
    t_t = np.log(20)
    t_wb = 0
    a_t = np.log(20)*2 + np.log(30)
    a_a = np.log(10)
    a_wb = np.log(10)*2 + np.log(20)*2 + np.log(30)
    wb_c = wb_a + wb_t + wb_wb
    t_c = t_a + t_t + t_wb
    a_c = a_a + a_t + a_wb
    expected_probs = log_expected([
        [a_a/a_c, t_a/t_c, wb_a/wb_c],
        [a_t/a_c, t_t/t_c, wb_t/wb_c],
        [a_wb/a_c, 0,       0]
    ])
    assert_matrix_close(bigram_probs, expected_probs)

# New tests for non-positional models with WB and NWB

def test_fit_bigrams_non_positional_joint_WB(token_freqs, unique_sounds):
    """
    Test fit_bigram_joint_nonpos_wb() with word boundaries, no smoothing, no weighting.
    (Joint, non-positional bigram model.)

    Manual derivation based on unit test training data:
      - Count occurrences:
          #t = 3, #a = 2, ta = 6, at = 3, tt = 1, aa = 1, a# = 5.
      - Total bigrams = 21.
      Expected matrix (rows: ['a','t','#'], columns: ['a','t','#']):
          [[1/21, 6/21, 2/21],
           [3/21, 1/21, 3/21],
           [5/21, 0/21, 0/21]]
    """
    sounds = unique_sounds + [WORD_BOUNDARY]
    bigram_probs = fit_bigram_joint_nonpos_wb(token_freqs, token_weighted=False, smoothed=False)
    expected_probs = log_expected([
        [1/21, 6/21, 2/21],
        [3/21, 1/21, 3/21],
        [5/21, 0/21, 0/21]
    ])
    assert_matrix_close(bigram_probs, expected_probs)

def test_fit_bigrams_non_positional_conditional_WB(token_freqs, unique_sounds):
    """
    Test fit_bigram_conditional_nonpos_wb() with word boundaries, no smoothing, no weighting.
    (Conditional, non-positional bigram model with WB.)

    Manual derivation:
      - For first symbol 'a': counts: aa=1, at=3, a#=5 → total = 9.
      - For first symbol 't': counts: ta=6, tt=1 → total = 7.
      - For first symbol '#' (boundary): counts: #a=2, #t=3 → total = 5.
      Expected matrix (rows: ['a','t','#'], columns: ['a','t','#']):
         [[1/9,   6/7,   2/5],
          [3/9,   1/7,   3/5],
          [5/9,   0,     0]]
    """
    # Updated: add WORD_BOUNDARY to unique_sounds
    bigram_probs = fit_bigram_conditional_nonpos_wb(token_freqs, unique_sounds + [WORD_BOUNDARY])
    expected_probs = log_expected([
        [1/9,   6/7,   2/5],
        [3/9,   1/7,   3/5],
        [5/9,   0,     0]
    ])
    assert_matrix_close(bigram_probs, expected_probs)

def test_fit_bigrams_non_positional_conditional_NWB(token_freqs, unique_sounds):
    """
    Test fit_bigram_conditional_nonpos_nwb() with no word boundaries, no smoothing, no weighting.
    (Conditional, non-positional bigram model without WB.)

    For NWB tokens (only characters), the counts are:
      - For first symbol 'a': (a,t)=3, (a,a)=1 → total=4.
      - For first symbol 't': (t,a)=6, (t,t)=1 → total=7.
      Expected matrix (rows: ['a','t'], columns: ['a','t']):
         [[1/4, 6/7],
          [3/4, 1/7]]
    """
    bigram_probs = fit_bigram_conditional_nonpos_nwb(token_freqs, unique_sounds)
    expected_probs = log_expected([
        [1/4, 6/7],
        [3/4, 1/7]
    ])
    assert_matrix_close(bigram_probs, expected_probs)

def test_fit_bigrams_non_positional_joint_NWB(token_freqs, unique_sounds):
    """
    Test fit_bigram_joint_nonpos_nwb() with no word boundaries, no smoothing, no weighting.
    (Joint, non-positional bigram model without WB.)

    For NWB tokens, total bigram count = 11 and counts:
      - (a,a)=1, (a,t)=3, (t,a)=6, (t,t)=1.
      Expected matrix (rows: ['a','t'], columns: ['a','t']):
         [[1/11, 6/11],
          [3/11, 1/11]]
    """
    # Updated: remove unique_sounds parameter
    bigram_probs = fit_bigram_joint_nonpos_nwb(token_freqs)
    expected_probs = log_expected([
        [1/11, 6/11],
        [3/11, 1/11]
    ])
    assert_matrix_close(bigram_probs, expected_probs)

# End of tests/unit/test_bigrams.py