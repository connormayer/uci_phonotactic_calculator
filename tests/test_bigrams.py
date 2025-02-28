"""
test_bigrams.py - Tests for bigram models with various configurations.
This module verifies that our bigram model implementations are accurately written by comparing
their output (in log-space) to manually derived expected probabilities.
The expected values are calculated based on the training data:
    t a,10
    a t a,20
    t a t a,30
    a t t a,20
    t a a,10
For example, in the conditional non-positional bigram test with word boundaries:
  1. Each token is augmented with boundaries. For instance, "t a" becomes ['#', 't', 'a', '#'].
  2. The raw counts for each bigram are manually summed (using simple counts):
     - For previous symbol 'a': 
         • Next symbol 'a': 1 occurrence  
         • Next symbol 't': 3 occurrences  
         • Next symbol '#': 5 occurrences  
       Total = 9 → Normalized: [1/9, 3/9, 5/9].
     - For previous symbol 't': 
         • Next symbol 'a': 6 occurrences  
         • Next symbol 't': 1 occurrence  
         • Next symbol '#': 0 occurrences  
       Total = 7 → Normalized: [6/7, 1/7, 0/7].
     - For previous symbol '#' (boundary): 
         • Next symbol 'a': 2 occurrences  
         • Next symbol 't': 3 occurrences  
         • Next symbol '#': 0 occurrences  
       Total = 5 → Normalized: [2/5, 3/5, 0/5].
  3. The expected probability matrix is then:
         [[1/9, 6/7, 2/5],
          [3/9, 1/7, 3/5],
          [5/9, 0/7, 0/5]]
  4. When token weighting and smoothing are applied, each cell is initialized with 1 (smoothing)
     and weighted counts (using np.log(freq)) are added. Thus, every cell receives a nonzero value.
     The expected matrix for the weighted+smoothed case is manually derived in the code below.
"""

import unittest
import numpy as np
from src.io_utils import read_tokens, WORD_BOUNDARY
from src.ngram_models import fit_bigrams, fit_non_positional_bigrams, fit_positional_bigrams
from tests.test_helpers import log_expected

TRAINING_FILE = 'data/unit_test_data/unit_test_training_data.txt'

class TestBigrams(unittest.TestCase):
    def setUp(self):
        self.token_freqs = read_tokens(TRAINING_FILE)
        # Unique sounds from training data (letters) then appended with boundary '#'
        self.unique_sounds = sorted(list(set(
            sound for token, _ in self.token_freqs for sound in token
        ))) + [WORD_BOUNDARY]

    def testFitBigrams_Conditional_WB(self):
        """
        Test fit_bigrams() with word boundaries, no smoothing, no weighting.
        (Conditional, non-positional bigram model.)
        
        Manual derivation:
        - Augment tokens with boundaries (e.g., "t a" becomes ['#','t','a','#']).
        - For previous symbol 'a': counts → [1 (next 'a'), 3 (next 't'), 5 (next '#')]; normalized [1/9, 3/9, 5/9].
        - For previous symbol 't': counts → [6, 1, 0]; normalized [6/7, 1/7, 0/7].
        - For previous symbol '#' (boundary): counts → [2, 3, 0]; normalized [2/5, 3/5, 0/5].
        - Expected matrix (rows: next symbols ['a','t','#'], columns: previous symbols ['a','t','#']):
             [[1/9, 6/7, 2/5],
              [3/9, 1/7, 3/5],
              [5/9, 0/7, 0/5]]
        - Then convert to log-space.
        """
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds)
        expected_probs = log_expected([
            [1/9,    6/7,    2/5],
            [3/9,    1/7,    3/5],
            [5/9,    0/7,    0/5]
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitBigrams_Conditional_WB_Smoothed(self):
        """Test fit_bigrams() with word boundaries, smoothing, no weighting (conditional, non-positional)."""
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds, smoothed=True)
        # With smoothing, one count is added to every cell.
        expected_probs = log_expected([
            [2/12, 7/10, 3/8],
            [4/12, 2/10, 4/8],
            [6/12, 1/10, 1/8]
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitBigrams_Conditional_WB_Weighted(self):
        """Test fit_bigrams() with word boundaries, token weighting, no smoothing (conditional, non-positional)."""
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds, token_weighted=True)
        # Manual derivation for weighted counts (without smoothing):
        # For token weighting, each occurrence adds np.log(freq) instead of 1.
        wb_t = np.log(10)*2 + np.log(30)     # Sum for bigrams with previous '#' and next 't'
        wb_a = np.log(20)*2                  # For previous '#' and next 'a'
        wb_wb = 0                          # For previous '#' and next '#'
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
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitBigrams_Conditional_WB_Smoothed_Weighted(self):
        """
        Test fit_bigrams() with word boundaries, token weighting, and smoothing (conditional, non-positional).
        
        With smoothing, one count is added to every cell before adding weighted counts.
        Manual derivation (weighted with smoothing):
          For each cell, the total count = 1 (smoothing) + sum(log(freq)) from bigram occurrences.
          For example, for previous symbol 'a' and next symbol 'a':
             Count = 1 + (from tokens: only token 5 contributes: np.log(10))
                     = 1 + np.log(10)
          Similarly, derive contributions for each cell.
          Expected values:
            - For column 'a' (previous 'a'):
                Row 'a': a_a = 1 + np.log(10)
                Row 't': a_t = 1 + 2*np.log(20) + np.log(30)
                Row '#': a_wb = 1 + 2*np.log(10) + 2*np.log(20) + np.log(30)
                Denom a_c = a_a + a_t + a_wb.
            - For column 't' (previous 't'):
                Row 'a': t_a = 1 + 2*np.log(10) + 2*np.log(20) + 2*np.log(30)
                Row 't': t_t = 1 + np.log(20)
                Row '#': t_wb = 1   (smoothing only)
                Denom t_c = t_a + t_t + t_wb.
            - For column '#' (previous '#', denoted wb):
                Row 'a': wb_a = 1 + 2*np.log(20)
                Row 't': wb_t = 1 + 2*np.log(10) + np.log(30)
                Row '#': wb_wb = 1
                Denom wb_c = wb_a + wb_t + wb_wb.
        The expected log-probability matrix is computed by taking the log of each cell's value divided by its column sum.
        """
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds, token_weighted=True, smoothed=True)
        wb_t = np.log(10)*2 + np.log(30) + 1
        wb_a = np.log(20)*2 + 1
        wb_wb = 1
        t_a = np.log(10)*2 + np.log(20)*2 + np.log(30)*2 + 1
        t_t = np.log(20) + 1
        t_wb = 1
        a_t = np.log(20)*2 + np.log(30) + 1
        a_a = np.log(10) + 1
        a_wb = np.log(10)*2 + np.log(20)*2 + np.log(30) + 1
        # Denominators for each column:
        a_c = a_a + a_t + a_wb
        t_c = t_a + t_t + t_wb
        wb_c = wb_a + wb_t + wb_wb
        # Note: With smoothing, every cell gets a nonzero value.
        expected_probs = log_expected([
            [a_a/a_c, t_a/t_c, wb_a/wb_c],
            [a_t/a_c, t_t/t_c, wb_t/wb_c],
            [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

if __name__ == '__main__':
    unittest.main()