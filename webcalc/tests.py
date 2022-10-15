from django.test import TestCase
from src import ngram_calculator

import numpy as np

TRAINING_FILE_NO_FREQS = 'data/unit_test_data/unit_test_training_data.txt'

class NonFreqTestCase(TestCase):
    def setUp(self):
        self.training_data, _ = ngram_calculator.read_tokens(TRAINING_FILE_NO_FREQS)
        self.unique_sounds = sorted(list(set(
            [item for sublist in self.training_data for item in sublist]
        ))) + ['#']

class HelperFunctionsTestCase(NonFreqTestCase):
    def testGenerateBigrams(self):
        bigrams = ngram_calculator.generate_bigrams(self.training_data)
        sorted_bigrams = sorted(bigrams)
        expected_bigrams = (
            [('#', 'a')] * 2 + 
            [('#', 't')] * 3 + 
            [('a', '#')] * 5 + 
            [('a', 'a')] * 1 + 
            [('a', 't')] * 3 +
            [('t', 'a')] * 6 +
            [('t', 't')] * 1
        )
        self.assertEqual(sorted_bigrams, expected_bigrams)

class UnsmoothedTypeNGramsTestCase(NonFreqTestCase):
    """
    Tests the suite of unigram/bigram measurements with no smoothing and
    type weighting. This includes unigram, bigram, positional unigram, and 
    positional bigram scores.
    """
    def testFitUnigrams(self):
        unigram_freqs = ngram_calculator.fit_unigrams(self.training_data)
        self.assertEqual(unigram_freqs['t'], np.log(7/16))
        self.assertEqual(unigram_freqs['a'], np.log(9/16))

    def testFitBigrams(self):
        bigram_probs = ngram_calculator.fit_bigrams(
            self.training_data, self.unique_sounds
        )

        t_idx = self.unique_sounds.index('t')
        a_idx = self.unique_sounds.index('a')
        wb_idx = self.unique_sounds.index('#')

        expected_probs = np.log(
            np.matrix([
            #    a_   t_   #_
                [1/9, 6/7, 2/5], # a 
                [3/9, 1/7, 3/5], # t
                [5/9, 0/7, 0/5]  # #
            ])
        )
        self.assertTrue(np.array_equal(bigram_probs, expected_probs))