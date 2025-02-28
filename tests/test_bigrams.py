"""
test_bigrams.py - Tests for bigram models with various configurations.
This module tests non-positional bigram models (conditional normalization) 
with word boundaries and different options: no smoothing, smoothed, token weighted,
and their combinations. It also tests positional bigram models (joint normalization)
without word boundaries, with similar configuration suffixes.
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
        self.unique_sounds = sorted(list(set(
            sound for token, _ in self.token_freqs for sound in token
        ))) + [WORD_BOUNDARY]

    def testFitBigrams_Conditional_WB(self):
        """Test fit_bigrams() with word boundaries, no smoothing, no weighting (conditional, non-positional)."""
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds)
        t_idx = self.unique_sounds.index('t')
        a_idx = self.unique_sounds.index('a')
        wb_idx = self.unique_sounds.index(WORD_BOUNDARY)
        expected_probs = log_expected([
            #     a_      t_      #_
            [1/9,    6/7,    2/5],  # a
            [3/9,    1/7,    3/5],  # t
            [5/9,    0/7,    0/5]   # #
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitBigrams_Conditional_WB_Smoothed(self):
        """Test fit_bigrams() with word boundaries, smoothing, no weighting (conditional, non-positional)."""
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds, smoothed=True)
        expected_probs = log_expected([
            #     a_      t_      #_
            [2/12,   7/10,   3/8],  # a
            [4/12,   2/10,   4/8],  # t
            [6/12,   1/10,   1/8]   # #
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitBigrams_Conditional_WB_Weighted(self):
        """Test fit_bigrams() with word boundaries, token weighting, no smoothing (conditional, non-positional)."""
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds, token_weighted=True)
        wb_t = np.log(10) * 2 + np.log(30)
        wb_a = np.log(20) * 2
        wb_wb = 0
        t_a = np.log(10) * 2 + np.log(20) * 2 + np.log(30) * 2
        t_t = np.log(20)
        t_wb = 0
        a_t = np.log(20) * 2 + np.log(30)
        a_a = np.log(10)
        a_wb = np.log(10) * 2 + np.log(20) * 2 + np.log(30)
        wb_c = wb_t + wb_a + wb_wb
        t_c = t_a + t_t + t_wb
        a_c = a_t + a_a + a_wb
        expected_probs = log_expected([
            #         a_          t_          #_
            [a_a/a_c,    t_a/t_c,    wb_a/wb_c],  # a
            [a_t/a_c,    t_t/t_c,    wb_t/wb_c],  # t
            [a_wb/a_c,   t_wb/t_c,   wb_wb/wb_c]  # #
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitBigrams_Conditional_WB_Smoothed_Weighted(self):
        """Test fit_bigrams() with word boundaries, token weighting, and smoothing (conditional, non-positional)."""
        bigram_probs = fit_bigrams(self.token_freqs, self.unique_sounds, token_weighted=True, smoothed=True)
        wb_t = np.log(10) * 2 + np.log(30) + 1
        wb_a = np.log(20) * 2 + 1
        wb_wb = 1
        t_a = np.log(10) * 2 + np.log(20) * 2 + np.log(30) * 2 + 1
        t_t = np.log(20) + 1
        t_wb = 1
        a_t = np.log(20) * 2 + np.log(30) + 1
        a_a = np.log(10) + 1
        a_wb = np.log(10) * 2 + np.log(20) * 2 + np.log(30) + 1
        wb_c = wb_t + wb_a + wb_wb
        t_c = t_a + t_t + t_wb
        a_c = a_t + a_a + a_wb
        expected_probs = log_expected([
            #         a_          t_          #_
            [a_a/a_c,    t_a/t_c,    wb_a/wb_c],  # a
            [a_t/a_c,    t_t/t_c,    wb_t/wb_c],  # t
            [a_wb/a_c,   t_wb/t_c,   wb_wb/wb_c]  # #
        ])
        self.assertTrue(np.allclose(bigram_probs, expected_probs))
    
    def testFitPositionalBigrams_Joint_NoWB(self):
        """Test fit_positional_bigrams() with no smoothing and no weighting (joint, positional without word boundaries)."""
        pos_bigram_freqs = fit_positional_bigrams(self.token_freqs)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('t', 'a')], 3/5)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('a', 't')], 2/5)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('t', 't')], 0/5)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('a', 'a')], 0/5)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('t', 'a')], 1/4)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('a', 't')], 1/4)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('t', 't')], 1/4)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('a', 'a')], 1/4)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('t', 'a')], 2/2)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('a', 't')], 0/2)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('t', 't')], 0/2)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('a', 'a')], 0/2)
    
    def testFitPositionalBigrams_Joint_NoWB_Weighted(self):
        """Test fit_positional_bigrams() with token weighting and no smoothing (joint, positional without word boundaries)."""
        pos_bigram_freqs = fit_positional_bigrams(self.token_freqs, token_weighted=True)
        ta_01 = np.log(10) * 2 + np.log(30)
        at_01 = np.log(20) * 2
        aa_01 = 0
        tt_01 = 0
        ta_12 = np.log(20)
        at_12 = np.log(30)
        aa_12 = np.log(10)
        tt_12 = np.log(20)
        ta_23 = np.log(30) + np.log(20)
        at_23 = 0
        aa_23 = 0
        tt_23 = 0
        total_01 = ta_01 + at_01 + aa_01 + tt_01
        total_12 = ta_12 + at_12 + aa_12 + tt_12
        total_23 = ta_23 + at_23 + aa_23 + tt_23
        self.assertEqual(pos_bigram_freqs[(0, 1)][('t', 'a')], ta_01 / total_01)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('a', 't')], at_01 / total_01)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('t', 't')], tt_01 / total_01)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('a', 'a')], aa_01 / total_01)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('t', 'a')], ta_12 / total_12)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('a', 't')], at_12 / total_12)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('t', 't')], tt_12 / total_12)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('a', 'a')], aa_12 / total_12)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('t', 'a')], ta_23 / total_23)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('a', 't')], at_23 / total_23)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('t', 't')], tt_23 / total_23)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('a', 'a')], aa_23 / total_23)
    
    def testFitPositionalBigrams_Joint_NoWB_Smoothed(self):
        """Test fit_positional_bigrams() with smoothing and no weighting (joint, positional without word boundaries)."""
        pos_bigram_freqs = fit_positional_bigrams(self.token_freqs, smoothed=True)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('t', 'a')], 4/9)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('a', 't')], 3/9)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('t', 't')], 1/9)
        self.assertEqual(pos_bigram_freqs[(0, 1)][('a', 'a')], 1/9)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('t', 'a')], 2/8)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('a', 't')], 2/8)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('t', 't')], 2/8)
        self.assertEqual(pos_bigram_freqs[(1, 2)][('a', 'a')], 2/8)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('t', 'a')], 3/6)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('a', 't')], 1/6)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('t', 't')], 1/6)
        self.assertEqual(pos_bigram_freqs[(2, 3)][('a', 'a')], 1/6)
    
    def testFitPositionalBigrams_Joint_NoWB_Smoothed_Weighted(self):
        """Test fit_positional_bigrams() with smoothing and token weighting (joint, positional without word boundaries)."""
        pos_bigram_freqs = fit_positional_bigrams(self.token_freqs, token_weighted=True, smoothed=True)
        ta_01 = np.log(10) * 2 + np.log(30) + 1
        at_01 = np.log(20) * 2 + 1
        aa_01 = 1
        tt_01 = 1
        ta_12 = np.log(20) + 1
        at_12 = np.log(30) + 1
        aa_12 = np.log(10) + 1
        tt_12 = np.log(20) + 1
        ta_23 = np.log(30) + np.log(20) + 1
        at_23 = 1
        aa_23 = 1
        tt_23 = 1
        total_01 = ta_01 + at_01 + aa_01 + tt_01
        total_12 = ta_12 + at_12 + aa_12 + tt_12
        total_23 = ta_23 + at_23 + aa_23 + tt_23
        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][('t', 'a')], ta_01 / total_01)
        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][('a', 't')], at_01 / total_01)
        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][('t', 't')], tt_01 / total_01)
        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][('a', 'a')], aa_01 / total_01)
        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][('t', 'a')], ta_12 / total_12)
        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][('a', 't')], at_12 / total_12)
        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][('t', 't')], tt_12 / total_12)
        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][('a', 'a')], aa_12 / total_12)
        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][('t', 'a')], ta_23 / total_23)
        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][('a', 't')], at_23 / total_23)
        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][('t', 't')], tt_23 / total_23)
        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][('a', 'a')], aa_23 / total_23)