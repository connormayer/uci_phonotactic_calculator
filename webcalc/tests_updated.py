# FileName: tests_updated.py
# version 1.0
# Summary: [Short summary of functions contained within]
# Tags: [comma delineated tags associated with this file]

#Run command, from parent directory
#"python manage.py test"

from django.test import TestCase
from src import ngram_calculator

import numpy as np

TRAINING_FILE = 'data/unit_test_data/unit_test_training_data.txt'


class FitNGramsTestCase(TestCase):
    """
    Tests the suite of unigram/bigram fitting functions with smoothing and
    type weighting. This includes unigram, bigram, positional unigram, 
    positional bigram, bigrams without boundaries, and non-positional 
    models (unigram/bigram).
    """

    def setUp(self):
        """
        Reads the sample tokens from TRAINING_FILE. Also extracts the unique
        sounds, including '#' as a boundary symbol.
        """
        self.token_freqs = ngram_calculator.read_tokens(TRAINING_FILE)
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
        unigram_freqs = ngram_calculator.fit_unigrams(self.token_freqs)
        self.assertEqual(unigram_freqs['t'], np.log(7/16))
        self.assertEqual(unigram_freqs['a'], np.log(9/16))

    def testFitUnigramsWeighted(self):
        """
        Test fit_unigrams() with no smoothing but token weighting.
        """
        unigram_freqs = ngram_calculator.fit_unigrams(
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
        """
        bigram_probs = ngram_calculator.fit_bigrams(
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
        """
        bigram_probs = ngram_calculator.fit_bigrams(
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
        """
        bigram_probs = ngram_calculator.fit_bigrams(
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
                [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]  # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsSmoothedWeighted(self):
        """
        Test fit_bigrams() with word boundaries, token-weighted, and smoothing.
        """
        bigram_probs = ngram_calculator.fit_bigrams(
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
                [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]  # #
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
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
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
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
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
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
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
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
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
        """
        pos_bigram_freqs = ngram_calculator.fit_positional_bigrams(
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
        """
        pos_bigram_freqs = ngram_calculator.fit_positional_bigrams(
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
        """
        pos_bigram_freqs = ngram_calculator.fit_positional_bigrams(
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
        """
        pos_bigram_freqs = ngram_calculator.fit_positional_bigrams(
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

    ###########################################################################
    #           NEW TESTS: Bigrams WITHOUT WORD BOUNDARIES                    #
    ###########################################################################
    def testFitBigramsNoBoundary(self):
        """
        Bigram probabilities without word boundaries (no weighting, no smoothing).
        """
        bigram_probs = ngram_calculator.fit_bigrams(
            self.token_freqs,
            self.unique_sounds, 
            use_word_boundaries=False
        )
        # p(a|a)=1/4, p(t|a)=3/4, p(a|t)=6/7, p(t|t)=1/7
        expected_probs = np.log(
            np.array([
                [1/4,  6/7],  # row= 'a', col= 'a','t'
                [3/4,  1/7],  # row= 't', col= 'a','t'
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsNoBoundaryWeighted(self):
        """
        Bigram probabilities without word boundaries, token-weighted, no smoothing.
        """
        bigram_probs = ngram_calculator.fit_bigrams(
            self.token_freqs,
            self.unique_sounds,
            token_weighted=True,
            use_word_boundaries=False
        )
        # Weighted counts => (a->a)=10, (a->t)=70, total=80
        #                   (t->a)=120,(t->t)=20, total=140
        # => p(a|a)=10/80=1/8, p(t|a)=70/80=7/8, p(a|t)=120/140=6/7, p(t|t)=20/140=1/7
        expected_probs = np.log(
            np.array([
                [1/8,  6/7],
                [7/8,  1/7]
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsNoBoundarySmoothed(self):
        """
        Bigram probabilities without word boundaries, add-1 smoothing, no weighting.
        """
        bigram_probs = ngram_calculator.fit_bigrams(
            self.token_freqs,
            self.unique_sounds,
            smoothed=True,
            use_word_boundaries=False
        )
        # (a->a)=1+1=2, (a->t)=3+1=4 => total=6
        # (t->a)=6+1=7, (t->t)=1+1=2 => total=9
        # => p(a|a)=2/6=1/3, p(t|a)=4/6=2/3, p(a|t)=7/9, p(t|t)=2/9
        expected_probs = np.log(
            np.array([
                [1/3,  7/9],
                [2/3,  2/9]
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs, rtol=1e-7))

    def testFitBigramsNoBoundarySmoothedWeighted(self):
        """
        Bigram probabilities without word boundaries, add-1 smoothing, token-weighted.
        """
        bigram_probs = ngram_calculator.fit_bigrams(
            self.token_freqs,
            self.unique_sounds,
            smoothed=True,
            token_weighted=True,
            use_word_boundaries=False
        )
        # Weighted => (a->a)=10 => +1=11, (a->t)=70 => +1=71 => denom=82
        #             (t->a)=120 => +1=121, (t->t)=20 => +1=21 => denom=142
        expected_probs = np.log(
            np.array([
                [11/82,   121/142],
                [71/82,    21/142]
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs, rtol=1e-7))

    ###########################################################################
    #      NEW TESTS: NON-POSITIONAL UNIGRAM & BIGRAM MODELS                  #
    ###########################################################################
    def testFitNonPositionalUnigrams(self):
        """
        Non-positional unigram probabilities (no smoothing, no weighting).
        """
        unigram_freqs = ngram_calculator.fit_non_positional_unigrams(self.token_freqs)

        # Manual counts ignoring freq: t=7, a=9 => total=16
        # => p(t)=7/16, p(a)=9/16
        expected_t = np.log(7/16)
        expected_a = np.log(9/16)

        self.assertAlmostEqual(unigram_freqs['t'], expected_t, places=7)
        self.assertAlmostEqual(unigram_freqs['a'], expected_a, places=7)
        self.assertNotIn('#', unigram_freqs)

    def testFitNonPositionalUnigramsWeighted(self):
        """
        Non-positional unigram probabilities, token-weighted (no smoothing).
        """
        unigram_freqs = ngram_calculator.fit_non_positional_unigrams(
            self.token_freqs, token_weighted=True
        )

        # Weighted counts => t=140, a=170 => total=310
        expected_t = np.log(140/310)
        expected_a = np.log(170/310)

        self.assertAlmostEqual(unigram_freqs['t'], expected_t, places=7)
        self.assertAlmostEqual(unigram_freqs['a'], expected_a, places=7)

    def testFitNonPositionalUnigramsSmoothed(self):
        """
        Non-positional unigram probabilities, add-1 smoothing, not weighted.
        """
        unigram_freqs = ngram_calculator.fit_non_positional_unigrams(
            self.token_freqs, smoothed=True
        )

        # t=7+1=8, a=9+1=10, total=16+2=18
        expected_t = np.log(8/18)
        expected_a = np.log(10/18)

        self.assertAlmostEqual(unigram_freqs['t'], expected_t, places=7)
        self.assertAlmostEqual(unigram_freqs['a'], expected_a, places=7)

    def testFitNonPositionalUnigramsSmoothedWeighted(self):
        """
        Non-positional unigram probabilities, add-1 smoothing + token-weighted.
        """
        unigram_freqs = ngram_calculator.fit_non_positional_unigrams(
            self.token_freqs, smoothed=True, token_weighted=True
        )

        # Weighted => t=140+1=141, a=170+1=171, total=310+2=312
        expected_t = np.log(141/312)
        expected_a = np.log(171/312)

        self.assertAlmostEqual(unigram_freqs['t'], expected_t, places=7)
        self.assertAlmostEqual(unigram_freqs['a'], expected_a, places=7)

    def testFitNonPositionalBigrams(self):
        """
        Non-positional bigram probabilities (no smoothing, no weighting).
        """
        bigram_probs = ngram_calculator.fit_non_positional_bigrams(self.token_freqs)
        # 'a'->'a'=1, 'a'->'t'=3 => total=4
        # 't'->'a'=6, 't'->'t'=1 => total=7
        # => p(a|a)=1/4, p(t|a)=3/4, p(a|t)=6/7, p(t|t)=1/7
        expected = np.log(np.array([
            [1/4, 6/7],  # row= 'a', col= 'a','t'
            [3/4, 1/7]   # row= 't', col= 'a','t'
        ]))
        self.assertTrue(np.allclose(bigram_probs, expected, rtol=1e-7))

    def testFitNonPositionalBigramsWeighted(self):
        """
        Non-positional bigram probabilities, token-weighted, no smoothing.
        """
        bigram_probs = ngram_calculator.fit_non_positional_bigrams(
            self.token_freqs, token_weighted=True
        )
        # Weighted => (a->a)=10, (a->t)=70 => 80
        #             (t->a)=120, (t->t)=20 => 140
        # => p(a|a)=10/80=1/8, p(t|a)=70/80=7/8, p(a|t)=120/140=6/7, p(t|t)=20/140=1/7
        expected = np.log(np.array([
            [1/8,  6/7],
            [7/8,  1/7]
        ]))
        self.assertTrue(np.allclose(bigram_probs, expected, rtol=1e-7))

    def testFitNonPositionalBigramsSmoothed(self):
        """
        Non-positional bigram probabilities, add-1 smoothing, not weighted.
        """
        bigram_probs = ngram_calculator.fit_non_positional_bigrams(
            self.token_freqs, smoothed=True
        )
        # (a->a)=1+1=2, (a->t)=3+1=4 => denom=4+2=6
        # (t->a)=6+1=7, (t->t)=1+1=2 => denom=7+2=9
        expected = np.log(np.array([
            [1/3, 7/9],
            [2/3, 2/9]
        ]))
        self.assertTrue(np.allclose(bigram_probs, expected, rtol=1e-7))

    def testFitNonPositionalBigramsSmoothedWeighted(self):
        """
        Non-positional bigram probabilities, add-1 smoothing + token-weighted.
        """
        bigram_probs = ngram_calculator.fit_non_positional_bigrams(
            self.token_freqs, smoothed=True, token_weighted=True
        )
        # Weighted => (a->a)=10+1=11, (a->t)=70+1=71 => denom=80+2=82
        #             (t->a)=120+1=121, (t->t)=20+1=21 => denom=140+2=142
        expected = np.log(np.array([
            [11/82,   121/142],
            [71/82,    21/142]
        ]))
        self.assertTrue(np.allclose(bigram_probs, expected, rtol=1e-7))


class TestNGramsTestCase(TestCase):
    """
    Tests "get" functions: get_unigram_prob, get_bigram_prob, 
    get_pos_unigram_score, get_pos_bigram_score, etc.
    """
    def setUp(self):
        """
        We re-use the same training data, but pre-fit the models
        that we'll query in the following tests.
        """
        self.token_freqs = ngram_calculator.read_tokens(TRAINING_FILE)
        self.unique_sounds = sorted(list(set(
            [sound for token, _ in self.token_freqs for sound in token]
        ))) + ['#']

        self.unigram_probs = ngram_calculator.fit_unigrams(self.token_freqs)
        self.bigram_probs = ngram_calculator.fit_bigrams(
            self.token_freqs, self.unique_sounds
        )
        self.pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
            self.token_freqs
        )
        self.pos_bigram_freqs = ngram_calculator.fit_positional_bigrams(
            self.token_freqs
        )

    def testGetUnigramProbs(self):
        """
        Checks get_unigram_prob() for a known word and for an OOV scenario.
        """
        test_word = ['t', 'a', 't', 'a']
        prob = ngram_calculator.get_unigram_prob(test_word, self.unigram_probs)
        expected_prob = self.unigram_probs['t'] * 2 + self.unigram_probs['a'] * 2
        self.assertEqual(prob, expected_prob)

        test_word = ['b', 'l', 'a', 'h']
        prob = ngram_calculator.get_unigram_prob(test_word, self.unigram_probs)
        self.assertEqual(prob, float('-inf'))

    def testGetBigramProbs(self):
        """
        Checks get_bigram_prob() for a known word and OOV scenario.
        """
        test_word = ['t', 'a', 't', 'a']
        prob = ngram_calculator.get_bigram_prob(
            test_word, self.bigram_probs, self.unique_sounds
        )

        t_idx = self.unique_sounds.index('t')
        a_idx = self.unique_sounds.index('a')
        hash_idx = self.unique_sounds.index('#')

        # Because get_bigram_prob likely includes boundary as (first->#) and (#->last),
        # the expected log-prob is the sum of those transitions with our bigram_probs.
        expected_prob = 0
        expected_prob += self.bigram_probs[t_idx, hash_idx]
        expected_prob += self.bigram_probs[a_idx, t_idx]
        expected_prob += self.bigram_probs[t_idx, a_idx]
        expected_prob += self.bigram_probs[a_idx, t_idx]
        expected_prob += self.bigram_probs[hash_idx, a_idx]

        self.assertEqual(prob, expected_prob)

        test_word = ['b', 'l', 'a', 'h']
        prob = ngram_calculator.get_bigram_prob(
            test_word, self.unigram_probs, self.unique_sounds
        )
        self.assertEqual(prob, float('-inf'))

    def testGetPositionalUnigramScore(self):
        """
        Checks get_pos_unigram_score() with a known test word.
        """
        test_word = ['t', 'a', 't', 'a']
        score = ngram_calculator.get_pos_unigram_score(
            test_word, self.pos_unigram_freqs
        )
        expected_score = 1
        expected_score += self.pos_unigram_freqs[0]['t']
        expected_score += self.pos_unigram_freqs[1]['a']
        expected_score += self.pos_unigram_freqs[2]['t']
        expected_score += self.pos_unigram_freqs[3]['a']

        self.assertEqual(score, expected_score)

    def testGetPositionalBigramScore(self):
        """
        Checks get_pos_bigram_score() with a known test word.
        """
        test_word = ['t', 'a', 't', 'a']
        score = 1
        score += self.pos_bigram_freqs[(0, 1)][('t', 'a')]
        score += self.pos_bigram_freqs[(1, 2)][('a', 't')]
        score += self.pos_bigram_freqs[(2, 3)][('t', 'a')]

        self.assertEqual(
            ngram_calculator.get_pos_bigram_score(test_word, self.pos_bigram_freqs), 
            score
        )