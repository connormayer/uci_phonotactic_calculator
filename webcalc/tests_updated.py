"""
tests_updated.py - Unit tests for verifying n-gram model fitting and scoring functions.
Version: 1.2
"""

#run command: python -m webcalc.tests_updated

import unittest
from src.io_utils import read_tokens

#unigram models
from src.ngram_models import fit_positional_unigrams, fit_non_positional_unigram_probabilities
#bigram models
from src.ngram_models import fit_bigrams, fit_positional_bigrams, fit_non_positional_bigrams

import numpy as np

TRAINING_FILE = 'data/unit_test_data/unit_test_training_data.txt'


class FitNGramsTestCase(unittest.TestCase):
    """
    Tests the suite of unigram/bigram fitting functions with smoothing and
    token weighting. This includes unigram, bigram, positional unigram,
    positional bigram, bigrams without boundaries, and non-positional
    models (unigram/bigram).
    """

    def setUp(self):
        """
        Reads the sample tokens from TRAINING_FILE. Also extracts the unique
        sounds, including '#' as a boundary symbol.
        """
        self.token_freqs = read_tokens(TRAINING_FILE)
        self.unique_sounds = sorted(list(set(
            [sound for token, _ in self.token_freqs for sound in token]
        ))) + ['#']

    ###########################################################################
    #                EXISTING TESTS: Unigrams (Boundary-Free)                 #
    ###########################################################################
    def testFitUnigrams(self):
        """
        Test fit_unigrams() with no smoothing and no weighting.
        """
        unigram_freqs = fit_non_positional_unigram_probabilities(self.token_freqs)
        self.assertEqual(unigram_freqs['t'], np.log(7/16))
        self.assertEqual(unigram_freqs['a'], np.log(9/16))

    def testFitUnigramsWeighted(self):
        """
        Test fit_unigrams() with no smoothing but token weighting.
        """
        unigram_freqs = fit_non_positional_unigram_probabilities(
            self.token_freqs, token_weighted=True
        )

        t_total = np.log(10) * 2 + np.log(20) * 3 + np.log(30) * 2
        a_total = np.log(10) * 3 + np.log(20) * 4 + np.log(30) * 2
        total = t_total + a_total

        expected_dict = {
            't': np.log(t_total / total),
            'a': np.log(a_total / total)
        }

        self.assertEqual(unigram_freqs, expected_dict)

    ###########################################################################
    #                     EXISTING TESTS: Bigrams (With Boundary)             #
    ###########################################################################
    def testFitBigrams(self):
        """
        Test fit_bigrams() with word boundaries, no smoothing, no weighting.
        (Conditional, non-positional bigram model.)
        """
        bigram_probs = fit_bigrams(
            self.token_freqs, self.unique_sounds
        )

        t_idx = self.unique_sounds.index('t')
        a_idx = self.unique_sounds.index('a')
        wb_idx = self.unique_sounds.index('#')

        expected_probs = np.log(
            np.array([
                #     a_    t_    #_
                [1/9,  6/7,  2/5],  # a
                [3/9,  1/7,  3/5],  # t
                [5/9,  0/7,  0/5]   # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsSmoothed(self):
        """
        Test fit_bigrams() with word boundaries, with smoothing, no weighting.
        (Conditional, non-positional.)
        """
        bigram_probs = fit_bigrams(
            self.token_freqs, self.unique_sounds, smoothed=True
        )

        expected_probs = np.log(
            np.array([
                #     a_    t_    #_
                [2/12, 7/10, 3/8],  # a
                [4/12, 2/10, 4/8],  # t
                [6/12, 1/10, 1/8]   # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsWeighted(self):
        """
        Test fit_bigrams() with word boundaries, token-weighted, no smoothing.
        (Conditional, non-positional.)
        """
        bigram_probs = fit_bigrams(
            self.token_freqs, self.unique_sounds, token_weighted=True
        )

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

        expected_probs = np.log(
            np.array([
                #     a_        t_       #_
                [a_a/a_c,  t_a/t_c,  wb_a/wb_c],  # a
                [a_t/a_c,  t_t/t_c,  wb_t/wb_c],  # t
                [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]   # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsSmoothedWeighted(self):
        """
        Test fit_bigrams() with word boundaries, token-weighted, and smoothing.
        (Conditional, non-positional.)
        """
        bigram_probs = fit_bigrams(
            self.token_freqs, self.unique_sounds,
            token_weighted=True, smoothed=True
        )

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

        expected_probs = np.log(
            np.array([
                #     a_        t_       #_
                [a_a/a_c,  t_a/t_c,  wb_a/wb_c],  # a
                [a_t/a_c,  t_t/t_c,  wb_t/wb_c],  # t
                [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]   # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    ###########################################################################
    #         EXISTING TESTS: Positional Unigrams & Positional Bigrams        #
    ###########################################################################
    def testFitPositionalUnigrams(self):
        """
        Test fit_positional_unigrams() with no smoothing, no weighting.
        """
        pos_unigram_freqs = fit_positional_unigrams(
            self.token_freqs
        )
        self.assertEqual(pos_unigram_freqs[0]['t'], 3/5)
        self.assertEqual(pos_unigram_freqs[0]['a'], 2/5)
        self.assertEqual(pos_unigram_freqs[1]['t'], 2/5)
        self.assertEqual(pos_unigram_freqs[1]['a'], 3/5)
        self.assertEqual(pos_unigram_freqs[2]['t'], 2/4)
        self.assertEqual(pos_unigram_freqs[2]['a'], 2/4)
        self.assertEqual(pos_unigram_freqs[3]['t'], 0/2)
        self.assertEqual(pos_unigram_freqs[3]['a'], 2/2)

    def testFitPositionalUnigramsWeighted(self):
        """
        Test fit_positional_unigrams() with no smoothing, token-weighted.
        """
        pos_unigram_freqs = fit_positional_unigrams(
            self.token_freqs, token_weighted=True
        )
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

    def testFitPositionalUnigramsSmoothed(self):
        """
        Test fit_positional_unigrams() with smoothing, no weighting.
        """
        pos_unigram_freqs = fit_positional_unigrams(
            self.token_freqs, smoothed=True
        )
        self.assertEqual(pos_unigram_freqs[0]['t'], 4/7)
        self.assertEqual(pos_unigram_freqs[0]['a'], 3/7)
        self.assertEqual(pos_unigram_freqs[1]['t'], 3/7)
        self.assertEqual(pos_unigram_freqs[1]['a'], 4/7)
        self.assertEqual(pos_unigram_freqs[2]['t'], 3/6)
        self.assertEqual(pos_unigram_freqs[2]['a'], 3/6)
        self.assertEqual(pos_unigram_freqs[3]['t'], 1/4)
        self.assertEqual(pos_unigram_freqs[3]['a'], 3/4)

    def testFitPositionalUnigramsSmoothedWeighted(self):
        """
        Test fit_positional_unigrams() with smoothing and token weighting.
        """
        pos_unigram_freqs = fit_positional_unigrams(
            self.token_freqs, token_weighted=True, smoothed=True
        )
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

    def testFitPositionalBigrams(self):
        """
        Test fit_positional_bigrams() with no smoothing and no weighting.
        (Joint, positional bigram model without word boundaries.)
        """
        pos_bigram_freqs = fit_positional_bigrams(
            self.token_freqs
        )

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

    def testFitPositionalBigramsWeighted(self):
        """
        Test fit_positional_bigrams() with no smoothing, token-weighted.
        (Joint, positional without word boundaries.)
        """
        pos_bigram_freqs = fit_positional_bigrams(
            self.token_freqs, token_weighted=True
        )

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

    def testFitPositionalBigramsSmoothed(self):
        """
        Test fit_positional_bigrams() with smoothing, no weighting.
        (Joint, positional without word boundaries.)
        """
        pos_bigram_freqs = fit_positional_bigrams(
            self.token_freqs, smoothed=True
        )

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

    def testFitPositionalBigramsSmoothedWeighted(self):
        """
        Test fit_positional_bigrams() with smoothing and token weighting.
        (Joint, positional without word boundaries.)
        """
        pos_bigram_freqs = fit_positional_bigrams(
            self.token_freqs, token_weighted=True, smoothed=True
        )

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


class TestNgramModelCombinations(unittest.TestCase):
    """
    New tests to ensure every combination is present:
      - Positional bigrams: joint vs. conditional and with vs. without word boundaries.
      - Non-positional bigrams: joint (via fit_non_positional_bigrams) vs. conditional (via fit_bigrams)
        and with vs. without word boundaries.
    """

    def setUp(self):
        self.token_freqs = read_tokens(TRAINING_FILE)
        # For non-positional bigrams conditional tests (fit_bigrams) we need the complete set including boundary.
        self.unique_sounds = sorted(list(set(
            [sound for token, _ in self.token_freqs for sound in token]
        )))
        self.boundary = '#'
        self.sounds_with_boundary = self.unique_sounds + [self.boundary]

    # --- Positional Bigrams Tests ---
    def testPositionalBigrams_Conditional_NoWB(self):
        """
        Positional bigrams with conditional normalization (grouped by preceding symbol)
        and NO word boundaries.
        For each position and each preceding symbol, the probabilities should sum to 1.
        """
        pos_bigrams = fit_positional_bigrams(
            self.token_freqs, conditional=True, use_word_boundaries=False
        )
        for pos in pos_bigrams:
            counts_by_prev = {}
            for (prev, nxt), prob in pos_bigrams[pos].items():
                counts_by_prev.setdefault(prev, 0)
                counts_by_prev[prev] += prob
            for prev, total in counts_by_prev.items():
                self.assertAlmostEqual(total, 1.0, places=6)

    def testPositionalBigrams_Joint_NoWB(self):
        """
        Positional bigrams with joint (non-conditional) normalization and NO word boundaries.
        For each positional pair, the sum over all bigrams should equal 1.
        """
        pos_bigrams = fit_positional_bigrams(
            self.token_freqs, conditional=False, use_word_boundaries=False
        )
        for pos in pos_bigrams:
            total = sum(pos_bigrams[pos].values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def testPositionalBigrams_Conditional_WB(self):
        """
        Positional bigrams with conditional normalization and WITH word boundaries.
        For each position (including boundaries) and each preceding symbol, probabilities should sum to 1.
        """
        pos_bigrams = fit_positional_bigrams(
            self.token_freqs, conditional=True, use_word_boundaries=True
        )
        for pos in pos_bigrams:
            counts_by_prev = {}
            for (prev, nxt), prob in pos_bigrams[pos].items():
                counts_by_prev.setdefault(prev, 0)
                counts_by_prev[prev] += prob
            for prev, total in counts_by_prev.items():
                self.assertAlmostEqual(total, 1.0, places=6)

    def testPositionalBigrams_Joint_WB(self):
        """
        Positional bigrams with joint (non-conditional) normalization and WITH word boundaries.
        For each positional pair, the sum over all bigrams should equal 1.
        """
        pos_bigrams = fit_positional_bigrams(
            self.token_freqs, conditional=False, use_word_boundaries=True
        )
        for pos in pos_bigrams:
            total = sum(pos_bigrams[pos].values())
            self.assertAlmostEqual(total, 1.0, places=6)

    # --- Non-Positional Bigrams Tests ---
    def testNonPositionalBigrams_Conditional_WB(self):
        """
        Non-positional bigrams with conditional normalization computed by fit_bigrams
        with word boundaries.
        For each previous symbol (column), the probabilities (after exponentiating the log)
        should sum to 1.
        """
        bp = fit_bigrams(
            self.token_freqs, self.sounds_with_boundary, use_word_boundaries=True
        )
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)

    def testNonPositionalBigrams_Joint_WB(self):
        """
        Non-positional bigrams with joint normalization computed by fit_non_positional_bigrams
        with word boundaries.
        For each previous symbol (column), the probabilities should sum to 1.
        """
        bp = fit_non_positional_bigrams(
            self.token_freqs, use_word_boundaries=True
        )
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)

    def testNonPositionalBigrams_Conditional_NoWB(self):
        """
        Non-positional bigrams with conditional normalization computed by fit_bigrams
        without word boundaries.
        For each previous symbol, the probabilities should sum to 1.
        """
        bp = fit_bigrams(
            self.token_freqs, self.unique_sounds, use_word_boundaries=False
        )
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)

    def testNonPositionalBigrams_Joint_NoWB(self):
        """
        Non-positional bigrams with joint normalization computed by fit_non_positional_bigrams
        without word boundaries.
        For each previous symbol, the probabilities should sum to 1.
        """
        bp = fit_non_positional_bigrams(
            self.token_freqs, use_word_boundaries=False
        )
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)


if __name__ == '__main__':
    unittest.main()