"""
test_unigrams.py - Tests for unigram models with various configurations.
This module tests non-positional unigram models (plain counts and token weighting) 
and positional unigram models (with and without smoothing, and with token weighting).
Test function names include suffixes indicating configuration (e.g., _Weighted, _Smoothed).
"""

import unittest
import numpy as np
from src.io_utils import read_tokens
from src.ngram_models import fit_non_positional_unigram, fit_positional_unigrams

TRAINING_FILE = 'data/unit_test_data/unit_test_training_data.txt'

class TestUnigrams(unittest.TestCase):
    def setUp(self):
        self.token_freqs = read_tokens(TRAINING_FILE)
        self.unique_sounds = sorted(list(set(
            sound for token, _ in self.token_freqs for sound in token
        )))
    
    def testFitNonPositionalUnigrams(self):
        """Test fit_non_positional_unigram() with no smoothing and no weighting."""
        unigram_freqs = fit_non_positional_unigram(self.token_freqs)
        # Expected: t = 7/16 and a = 9/16 (log-probabilities)
        self.assertEqual(unigram_freqs['t'], np.log(7/16))
        self.assertEqual(unigram_freqs['a'], np.log(9/16))
    
    def testFitNonPositionalUnigrams_Weighted(self):
        """Test fit_non_positional_unigram() with token weighting and no smoothing."""
        unigram_freqs = fit_non_positional_unigram(self.token_freqs, token_weighted=True)
        t_total = np.log(10) * 2 + np.log(20) * 3 + np.log(30) * 2
        a_total = np.log(10) * 3 + np.log(20) * 4 + np.log(30) * 2
        total = t_total + a_total
        expected_dict = {
            't': np.log(t_total / total),
            'a': np.log(a_total / total)
        }
        self.assertEqual(unigram_freqs, expected_dict)
    
    def testFitPositionalUnigrams(self):
        """Test fit_positional_unigrams() with no smoothing and no weighting."""
        pos_unigram_freqs = fit_positional_unigrams(self.token_freqs)
        self.assertEqual(pos_unigram_freqs[0]['t'], 3/5)
        self.assertEqual(pos_unigram_freqs[0]['a'], 2/5)
        self.assertEqual(pos_unigram_freqs[1]['t'], 2/5)
        self.assertEqual(pos_unigram_freqs[1]['a'], 3/5)
        self.assertEqual(pos_unigram_freqs[2]['t'], 2/4)
        self.assertEqual(pos_unigram_freqs[2]['a'], 2/4)
        self.assertEqual(pos_unigram_freqs[3]['t'], 0/2)
        self.assertEqual(pos_unigram_freqs[3]['a'], 2/2)
    
    def testFitPositionalUnigrams_Weighted(self):
        """Test fit_positional_unigrams() with token weighting and no smoothing."""
        pos_unigram_freqs = fit_positional_unigrams(self.token_freqs, token_weighted=True)
        t_0 = np.log(10) * 2 + np.log(30)
        a_0 = np.log(20) * 2
        t_1 = np.log(20) * 2
        a_1 = np.log(10) * 2 + np.log(30)
        t_2 = np.log(20) + np.log(30)
        a_2 = np.log(20) + np.log(10)
        t_3 = 0
        a_3 = np.log(20) + np.log(30)
        total_0 = t_0 + a_0
        total_1 = t_1 + a_1
        total_2 = t_2 + a_2
        total_3 = t_3 + a_3
        self.assertEqual(pos_unigram_freqs[0]['t'], t_0 / total_0)
        self.assertEqual(pos_unigram_freqs[0]['a'], a_0 / total_0)
        self.assertEqual(pos_unigram_freqs[1]['t'], t_1 / total_1)
        self.assertEqual(pos_unigram_freqs[1]['a'], a_1 / total_1)
        self.assertEqual(pos_unigram_freqs[2]['t'], t_2 / total_2)
        self.assertEqual(pos_unigram_freqs[2]['a'], a_2 / total_2)
        self.assertEqual(pos_unigram_freqs[3]['t'], t_3 / total_3)
        self.assertEqual(pos_unigram_freqs[3]['a'], a_3 / total_3)
    
    def testFitPositionalUnigrams_Smoothed(self):
        """Test fit_positional_unigrams() with smoothing and no weighting."""
        pos_unigram_freqs = fit_positional_unigrams(self.token_freqs, smoothed=True)
        self.assertEqual(pos_unigram_freqs[0]['t'], 4/7)
        self.assertEqual(pos_unigram_freqs[0]['a'], 3/7)
        self.assertEqual(pos_unigram_freqs[1]['t'], 3/7)
        self.assertEqual(pos_unigram_freqs[1]['a'], 4/7)
        self.assertEqual(pos_unigram_freqs[2]['t'], 3/6)
        self.assertEqual(pos_unigram_freqs[2]['a'], 3/6)
        self.assertEqual(pos_unigram_freqs[3]['t'], 1/4)
        self.assertEqual(pos_unigram_freqs[3]['a'], 3/4)
    
    def testFitPositionalUnigrams_Smoothed_Weighted(self):
        """Test fit_positional_unigrams() with smoothing and token weighting."""
        pos_unigram_freqs = fit_positional_unigrams(self.token_freqs, token_weighted=True, smoothed=True)
        t_0 = np.log(10) * 2 + np.log(30) + 1
        a_0 = np.log(20) * 2 + 1
        t_1 = np.log(20) * 2 + 1
        a_1 = np.log(10) * 2 + np.log(30) + 1
        t_2 = np.log(20) + np.log(30) + 1
        a_2 = np.log(20) + np.log(10) + 1
        t_3 = 1
        a_3 = np.log(20) + np.log(30) + 1
        total_0 = t_0 + a_0
        total_1 = t_1 + a_1
        total_2 = t_2 + a_2
        total_3 = t_3 + a_3
        self.assertEqual(pos_unigram_freqs[0]['t'], t_0 / total_0)
        self.assertEqual(pos_unigram_freqs[0]['a'], a_0 / total_0)
        self.assertEqual(pos_unigram_freqs[1]['t'], t_1 / total_1)
        self.assertEqual(pos_unigram_freqs[1]['a'], a_1 / total_1)
        self.assertEqual(pos_unigram_freqs[2]['t'], t_2 / total_2)
        self.assertEqual(pos_unigram_freqs[2]['a'], a_2 / total_2)
        self.assertEqual(pos_unigram_freqs[3]['t'], t_3 / total_3)
        self.assertEqual(pos_unigram_freqs[3]['a'], a_3 / total_3)