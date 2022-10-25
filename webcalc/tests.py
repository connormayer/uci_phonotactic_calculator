from django.test import TestCase
from src import ngram_calculator

import numpy as np

TRAINING_FILE_NO_FREQS = 'data/unit_test_data/unit_test_training_data.txt'

class BaseCase(TestCase):
    def setUp(self):
        self.training_data = ngram_calculator.read_tokens(TRAINING_FILE_NO_FREQS)
        self.unique_sounds = sorted(list(set(
            [item for sublist, _ in self.training_data for item in sublist]
        ))) + ['#']

class NGramsTestCase(BaseCase):
    """
    Tests the suite of unigram/bigram fitting functions with smoothing and
    type weighting. This includes unigram, bigram, positional unigram, and 
    positional bigram scores.
    """
    def testFitUnigrams(self):
        unigram_freqs = ngram_calculator.fit_unigrams(self.training_data)
        self.assertEqual(unigram_freqs['t'], np.log(7/16))
        self.assertEqual(unigram_freqs['a'], np.log(9/16))

    def testFitUnigramsWeighted(self):
        unigram_freqs = ngram_calculator.fit_unigrams(
            self.training_data, token_weighted=True
        )

        t_total = np.log(10) * 2 + np.log(20) * 3 + np.log(30) * 2
        a_total = np.log(10) * 3 + np.log(20) * 4 + np.log(30) * 2
        total = t_total + a_total

        expected_dict = {
            't': np.log(t_total / total),
            'a': np.log(a_total / total)
        }
        
        self.assertEqual(unigram_freqs, expected_dict)

    def testFitBigrams(self):
        bigram_probs = ngram_calculator.fit_bigrams(
            self.training_data, self.unique_sounds
        )

        t_idx = self.unique_sounds.index('t')
        a_idx = self.unique_sounds.index('a')
        wb_idx = self.unique_sounds.index('#')

        expected_probs = np.log(
            np.array([
            #    a_   t_   #_
                [1/9, 6/7, 2/5], # a 
                [3/9, 1/7, 3/5], # t
                [5/9, 0/7, 0/5]  # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsSmoothed(self):
        bigram_probs = ngram_calculator.fit_bigrams(
            self.training_data, self.unique_sounds, smoothed=True
        )

        expected_probs = np.log(
            np.array([
            #    a_   t_   #_
                [2/12, 7/10, 3/8], # a 
                [4/12, 2/10, 4/8], # t
                [6/12, 1/10, 1/8]  # #
            ])
        )

    def testFitBigramsWeighted(self):
        bigram_probs = ngram_calculator.fit_bigrams(
            self.training_data, self.unique_sounds, token_weighted=True
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
            #    a_   t_   #_
                [a_a/a_c, t_a/t_c, wb_a/wb_c], # a 
                [a_t/a_c, t_t/t_c, wb_t/wb_c], # t
                [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]  # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsSmoothedWeighted(self):
        bigram_probs = ngram_calculator.fit_bigrams(
            self.training_data, self.unique_sounds, 
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
            #    a_   t_   #_
                [a_a/a_c, t_a/t_c, wb_a/wb_c], # a 
                [a_t/a_c, t_t/t_c, wb_t/wb_c], # t
                [a_wb/a_c, t_wb/t_c, wb_wb/wb_c]  # #
            ])
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitPositionalUnigrams(self):
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
            self.training_data
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
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
            self.training_data, token_weighted=True
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
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
            self.training_data, smoothed=True
        )
        import pdb; pdb.set_trace()
        self.assertEqual(pos_unigram_freqs[0]['t'], 4/7)
        self.assertEqual(pos_unigram_freqs[0]['a'], 3/7)
        self.assertEqual(pos_unigram_freqs[1]['t'], 3/7)
        self.assertEqual(pos_unigram_freqs[1]['a'], 4/7)
        self.assertEqual(pos_unigram_freqs[2]['t'], 3/6)
        self.assertEqual(pos_unigram_freqs[2]['a'], 3/6)
        self.assertEqual(pos_unigram_freqs[3]['t'], 1/4)
        self.assertEqual(pos_unigram_freqs[3]['a'], 3/4)

    def testFitPositionalBigrams(self):
        pass

    def testFitPositionalBigramsWeighted(self):
        pass

    def testFitPositionalBigramsSmoothed(self):
        pass

    def testFitPositionalBigramsSmoothedWeighted(self):
        pass

