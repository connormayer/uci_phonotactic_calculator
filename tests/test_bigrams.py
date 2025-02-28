"""
test_bigrams.py - Tests for bigram models with various configurations.
This module verifies that our bigram model implementations are accurately written by comparing
their output (in log-space) to manually derived expected probabilities.
These expected probabilities are calculated independently (“the official calculator”) from the training data:
    t a,10
    a t a,20
    t a t a,30
    a t t a,20
    t a a,10
For example, in the conditional non-positional bigram test with word boundaries:
  1. Each token is augmented with boundaries (e.g., "t a" becomes ['#', 't', 'a', '#']).
  2. The raw counts for each bigram are manually summed:
     - For previous symbol 'a': next counts are:
         • 'a': 1 occurrence  
         • 't': 3 occurrences  
         • '#': 5 occurrences  
       (Total = 9 → Probabilities: [1/9, 3/9, 5/9])
     - For previous symbol 't': next counts are:
         • 'a': 6 occurrences  
         • 't': 1 occurrence  
         • '#': 0 occurrences  
       (Total = 7 → Probabilities: [6/7, 1/7, 0/7])
     - For previous symbol '#' (boundary): next counts are:
         • 'a': 2 occurrences  
         • 't': 3 occurrences  
         • '#': 0 occurrences  
       (Total = 5 → Probabilities: [2/5, 3/5, 0/5])
  3. These manually derived fractions are arranged as a matrix with rows as next symbols in order ['a', 't', '#']
     and columns as previous symbols in order ['a', 't', '#'], then converted to log-space via log_expected().
  4. Similar detailed derivations are used for the weighted and smoothed cases.
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
        # Unique sounds from training data are 'a' and 't'; add boundary '#' to get ['a', 't', '#'].
        self.unique_sounds = sorted(list(set(
            sound for token, _ in self.token_freqs for sound in token
        ))) + [WORD_BOUNDARY]

    def testFitBigrams_Conditional_WB(self):
        """
        Test fit_bigrams() with word boundaries, no smoothing, no weighting.
        (Conditional, non-positional bigram model.)
        
        Manual derivation:
        1. Augment each token with boundaries:
           e.g., "t a" → ['#', 't', 'a', '#'].
        2. Sum simple counts for each bigram (ignoring frequency numbers):
           - For previous symbol 'a': 
             • Next 'a': 1 occurrence  
             • Next 't': 3 occurrences  
             • Next '#': 5 occurrences  
             Total = 9 → Probabilities: [1/9, 3/9, 5/9].
           - For previous symbol 't': 
             • Next 'a': 6 occurrences  
             • Next 't': 1 occurrence  
             • Next '#': 0 occurrences  
             Total = 7 → Probabilities: [6/7, 1/7, 0/7].
           - For previous symbol '#' (boundary): 
             • Next 'a': 2 occurrences  
             • Next 't': 3 occurrences  
             • Next '#': 0 occurrences  
             Total = 5 → Probabilities: [2/5, 3/5, 0/5].
        3. Arrange expected matrix (rows: next symbols ['a','t','#'], columns: previous symbols ['a','t','#']):
             [[1/9, 6/7, 2/5],
              [3/9, 1/7, 3/5],
              [5/9, 0/7, 0/5]]
        4. Convert to log-space using log_expected().
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
        # With smoothing, each cell is initialized with a count of 1.
        expected_probs = log_expected([
            [2/12, 7/10, 3/8],
            [4/12, 2/10, 4/8],
            [6/12, 1/10, 1/8]
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitBigrams_Conditional_WB_Weighted(self):
        """Test fit_bigrams() with word boundaries, token weighting, no smoothing (conditional, non-positional)."""
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds, token_weighted=True)
        # Manual derivation using weighted counts (using np.log(freq) per occurrence):
        wb_t = np.log(10)*2 + np.log(30)      # for transitions from '#' to 't'
        wb_a = np.log(20)*2                   # from '#' to 'a'
        wb_wb = 0                           # from '#' to '#' remains 0
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
        
        With smoothing, each cell gets an initial count of 1.
        Manual derivation for each cell:
          - For previous symbol 'a':
              Row 'a': a_a = 1 + np.log(10)
              Row 't': a_t = 1 + 2*np.log(20) + np.log(30)
              Row '#': a_wb = 1 + 2*np.log(10) + 2*np.log(20) + np.log(30)
              Denom a_c = a_a + a_t + a_wb.
          - For previous symbol 't':
              Row 'a': t_a = 1 + 2*np.log(10) + 2*np.log(20) + 2*np.log(30)
              Row 't': t_t = 1 + np.log(20)
              Row '#': t_wb = 1   (smoothing only, no weighted occurrences)
              Denom t_c = t_a + t_t + t_wb.
          - For previous symbol '#' (boundary):
              Row 'a': wb_a = 1 + 2*np.log(20)
              Row 't': wb_t = 1 + 2*np.log(10) + np.log(30)
              Row '#': wb_wb = 1
              Denom wb_c = wb_a + wb_t + wb_wb.
        Expected matrix (each cell’s value divided by its column sum) is then converted to log-space.
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
        a_c = a_a + a_t + a_wb
        t_c = t_a + t_t + t_wb
        wb_c = wb_a + wb_t + wb_wb
        expected_probs = log_expected([
            [a_a/a_c, t_a/t_c, wb_a/wb_c],
            [a_t/a_c, t_t/t_c, wb_t/wb_c],
            [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

if __name__ == '__main__':
    unittest.main()