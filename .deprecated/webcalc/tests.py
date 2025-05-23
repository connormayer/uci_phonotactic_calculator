import numpy as np
from django.test import TestCase

from uci_phonotactic_calculator import ngram_calculator

TRAINING_FILE = "data/unit_test_data/unit_test_training_data.txt"


class FitNGramsTestCase(TestCase):
    """
    Tests the suite of unigram/bigram fitting functions with smoothing and
    type weighting. This includes unigram, bigram, positional unigram, and
    positional bigram scores.
    """

    def setUp(self):
        self.token_freqs = ngram_calculator.read_tokens(TRAINING_FILE)
        self.unique_sounds = sorted(
            {sound for token, _ in self.token_freqs for sound in token}
        ) + ["#"]

    def testFitUnigrams(self):
        unigram_freqs = ngram_calculator.fit_unigrams(self.token_freqs)
        self.assertEqual(unigram_freqs["t"], np.log(7 / 16))
        self.assertEqual(unigram_freqs["a"], np.log(9 / 16))

    def testFitUnigramsWeighted(self):
        unigram_freqs = ngram_calculator.fit_unigrams(
            self.token_freqs, token_weighted=True
        )

        t_total = np.log(10) * 2 + np.log(20) * 3 + np.log(30) * 2
        a_total = np.log(10) * 3 + np.log(20) * 4 + np.log(30) * 2
        total = t_total + a_total

        expected_dict = {"t": np.log(t_total / total), "a": np.log(a_total / total)}

        self.assertEqual(unigram_freqs, expected_dict)

    def testFitBigrams(self):
        bigram_probs = ngram_calculator.fit_bigrams(
            self.token_freqs, self.unique_sounds
        )

        # Variables used in previous versions of this test

        expected_probs = np.log(
            np.array(
                [
                    #    a_   t_   #_
                    [1 / 9, 6 / 7, 2 / 5],  # a
                    [3 / 9, 1 / 7, 3 / 5],  # t
                    [5 / 9, 0 / 7, 0 / 5],  # #
                ]
            )
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsSmoothed(self):
        # Test bigram probabilities with smoothing
        _ = ngram_calculator.fit_bigrams(
            self.token_freqs, self.unique_sounds, smoothed=True
        )

        # Expected probabilities - used in previous version of this test
        _ = np.log(
            np.array(
                [
                    #    a_   t_   #_
                    [2 / 12, 7 / 10, 3 / 8],  # a
                    [4 / 12, 2 / 10, 4 / 8],  # t
                    [6 / 12, 1 / 10, 1 / 8],  # #
                ]
            )
        )

    def testFitBigramsWeighted(self):
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
            np.array(
                [
                    #    a_   t_   #_
                    [a_a / a_c, t_a / t_c, wb_a / wb_c],  # a
                    [a_t / a_c, t_t / t_c, wb_t / wb_c],  # t
                    [a_wb / a_c, t_wb / t_c, wb_wb / wb_c],  # #
                ]
            )
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitBigramsSmoothedWeighted(self):
        bigram_probs = ngram_calculator.fit_bigrams(
            self.token_freqs, self.unique_sounds, token_weighted=True, smoothed=True
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
            np.array(
                [
                    #    a_   t_   #_
                    [a_a / a_c, t_a / t_c, wb_a / wb_c],  # a
                    [a_t / a_c, t_t / t_c, wb_t / wb_c],  # t
                    [a_wb / a_c, t_wb / t_c, wb_wb / wb_c],  # #
                ]
            )
        )
        self.assertTrue(np.allclose(bigram_probs, expected_probs))

    def testFitPositionalUnigrams(self):
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(self.token_freqs)
        self.assertEqual(pos_unigram_freqs[0]["t"], 3 / 5)
        self.assertEqual(pos_unigram_freqs[0]["a"], 2 / 5)
        self.assertEqual(pos_unigram_freqs[1]["t"], 2 / 5)
        self.assertEqual(pos_unigram_freqs[1]["a"], 3 / 5)
        self.assertEqual(pos_unigram_freqs[2]["t"], 2 / 4)
        self.assertEqual(pos_unigram_freqs[2]["a"], 2 / 4)
        self.assertEqual(pos_unigram_freqs[3]["t"], 0 / 2)
        self.assertEqual(pos_unigram_freqs[3]["a"], 2 / 2)

    def testFitPositionalUnigramsWeighted(self):
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

        self.assertEqual(pos_unigram_freqs[0]["t"], t_0 / total_0)
        self.assertEqual(pos_unigram_freqs[0]["a"], a_0 / total_0)
        self.assertEqual(pos_unigram_freqs[1]["t"], t_1 / total_1)
        self.assertEqual(pos_unigram_freqs[1]["a"], a_1 / total_1)
        self.assertEqual(pos_unigram_freqs[2]["t"], t_2 / total_2)
        self.assertEqual(pos_unigram_freqs[2]["a"], a_2 / total_2)
        self.assertEqual(pos_unigram_freqs[3]["t"], t_3 / total_3)
        self.assertEqual(pos_unigram_freqs[3]["a"], a_3 / total_3)

    def testFitPositionalUnigramsSmoothed(self):
        pos_unigram_freqs = ngram_calculator.fit_positional_unigrams(
            self.token_freqs, smoothed=True
        )
        self.assertEqual(pos_unigram_freqs[0]["t"], 4 / 7)
        self.assertEqual(pos_unigram_freqs[0]["a"], 3 / 7)
        self.assertEqual(pos_unigram_freqs[1]["t"], 3 / 7)
        self.assertEqual(pos_unigram_freqs[1]["a"], 4 / 7)
        self.assertEqual(pos_unigram_freqs[2]["t"], 3 / 6)
        self.assertEqual(pos_unigram_freqs[2]["a"], 3 / 6)
        self.assertEqual(pos_unigram_freqs[3]["t"], 1 / 4)
        self.assertEqual(pos_unigram_freqs[3]["a"], 3 / 4)

    def testFitPositionalUnigramsSmoothedWeighted(self):
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

        self.assertEqual(pos_unigram_freqs[0]["t"], t_0 / total_0)
        self.assertEqual(pos_unigram_freqs[0]["a"], a_0 / total_0)
        self.assertEqual(pos_unigram_freqs[1]["t"], t_1 / total_1)
        self.assertEqual(pos_unigram_freqs[1]["a"], a_1 / total_1)
        self.assertEqual(pos_unigram_freqs[2]["t"], t_2 / total_2)
        self.assertEqual(pos_unigram_freqs[2]["a"], a_2 / total_2)
        self.assertEqual(pos_unigram_freqs[3]["t"], t_3 / total_3)
        self.assertEqual(pos_unigram_freqs[3]["a"], a_3 / total_3)

    def testFitPositionalBigrams(self):
        pos_bigram_freqs = ngram_calculator.fit_positional_bigrams(self.token_freqs)

        self.assertEqual(pos_bigram_freqs[(0, 1)][("t", "a")], 3 / 5)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("a", "t")], 2 / 5)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("t", "t")], 0 / 5)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("a", "a")], 0 / 5)

        self.assertEqual(pos_bigram_freqs[(1, 2)][("t", "a")], 1 / 4)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("a", "t")], 1 / 4)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("t", "t")], 1 / 4)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("a", "a")], 1 / 4)

        self.assertEqual(pos_bigram_freqs[(2, 3)][("t", "a")], 2 / 2)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("a", "t")], 0 / 2)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("t", "t")], 0 / 2)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("a", "a")], 0 / 2)

    def testFitPositionalBigramsWeighted(self):
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

        self.assertEqual(pos_bigram_freqs[(0, 1)][("t", "a")], ta_01 / total_01)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("a", "t")], at_01 / total_01)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("t", "t")], tt_01 / total_01)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("a", "a")], aa_01 / total_01)

        self.assertEqual(pos_bigram_freqs[(1, 2)][("t", "a")], ta_12 / total_12)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("a", "t")], at_12 / total_12)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("t", "t")], tt_12 / total_12)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("a", "a")], aa_12 / total_12)

        self.assertEqual(pos_bigram_freqs[(2, 3)][("t", "a")], ta_23 / total_23)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("a", "t")], at_23 / total_23)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("t", "t")], tt_23 / total_23)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("a", "a")], aa_23 / total_23)

    def testFitPositionalBigramsSmoothed(self):
        pos_bigram_freqs = ngram_calculator.fit_positional_bigrams(
            self.token_freqs, smoothed=True
        )

        self.assertEqual(pos_bigram_freqs[(0, 1)][("t", "a")], 4 / 9)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("a", "t")], 3 / 9)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("t", "t")], 1 / 9)
        self.assertEqual(pos_bigram_freqs[(0, 1)][("a", "a")], 1 / 9)

        self.assertEqual(pos_bigram_freqs[(1, 2)][("t", "a")], 2 / 8)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("a", "t")], 2 / 8)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("t", "t")], 2 / 8)
        self.assertEqual(pos_bigram_freqs[(1, 2)][("a", "a")], 2 / 8)

        self.assertEqual(pos_bigram_freqs[(2, 3)][("t", "a")], 3 / 6)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("a", "t")], 1 / 6)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("t", "t")], 1 / 6)
        self.assertEqual(pos_bigram_freqs[(2, 3)][("a", "a")], 1 / 6)

    def testFitPositionalBigramsSmoothedWeighted(self):
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

        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][("t", "a")], ta_01 / total_01)
        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][("a", "t")], at_01 / total_01)
        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][("t", "t")], tt_01 / total_01)
        self.assertAlmostEqual(pos_bigram_freqs[(0, 1)][("a", "a")], aa_01 / total_01)

        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][("t", "a")], ta_12 / total_12)
        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][("a", "t")], at_12 / total_12)
        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][("t", "t")], tt_12 / total_12)
        self.assertAlmostEqual(pos_bigram_freqs[(1, 2)][("a", "a")], aa_12 / total_12)

        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][("t", "a")], ta_23 / total_23)
        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][("a", "t")], at_23 / total_23)
        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][("t", "t")], tt_23 / total_23)
        self.assertAlmostEqual(pos_bigram_freqs[(2, 3)][("a", "a")], aa_23 / total_23)


class TestNGramsTestCase(TestCase):
    def setUp(self):
        self.token_freqs = ngram_calculator.read_tokens(TRAINING_FILE)
        self.unique_sounds = sorted(
            {sound for token, _ in self.token_freqs for sound in token}
        ) + ["#"]

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
        test_word = ["t", "a", "t", "a"]
        prob = ngram_calculator.get_unigram_prob(test_word, self.unigram_probs)
        expected_prob = self.unigram_probs["t"] * 2 + self.unigram_probs["a"] * 2
        self.assertEqual(prob, expected_prob)

        test_word = ["b", "l", "a", "h"]
        prob = ngram_calculator.get_unigram_prob(test_word, self.unigram_probs)
        self.assertEqual(prob, float("-inf"))

    def testGetBigramProbs(self):
        test_word = ["t", "a", "t", "a"]
        prob = ngram_calculator.get_bigram_prob(
            test_word, self.bigram_probs, self.unique_sounds
        )

        t_idx = self.unique_sounds.index("t")
        a_idx = self.unique_sounds.index("a")
        hash_idx = self.unique_sounds.index("#")

        expected_prob = 0
        expected_prob += self.bigram_probs[t_idx, hash_idx]
        expected_prob += self.bigram_probs[a_idx, t_idx]
        expected_prob += self.bigram_probs[t_idx, a_idx]
        expected_prob += self.bigram_probs[a_idx, t_idx]
        expected_prob += self.bigram_probs[hash_idx, a_idx]

        self.assertEqual(prob, expected_prob)

        test_word = ["b", "l", "a", "h"]
        prob = ngram_calculator.get_bigram_prob(
            test_word, self.unigram_probs, self.unique_sounds
        )
        self.assertEqual(prob, float("-inf"))

    def testGetPositionalUnigramScore(self):
        test_word = ["t", "a", "t", "a"]
        score = ngram_calculator.get_pos_unigram_score(
            test_word, self.pos_unigram_freqs
        )
        expected_score = 1
        expected_score += self.pos_unigram_freqs[0]["t"]
        expected_score += self.pos_unigram_freqs[1]["a"]
        expected_score += self.pos_unigram_freqs[2]["t"]
        expected_score += self.pos_unigram_freqs[3]["a"]

        self.assertEqual(score, expected_score)

    def testGetPositionalBigramScore(self):
        test_word = ["t", "a", "t", "a"]
        score = ngram_calculator.get_pos_bigram_score(test_word, self.pos_bigram_freqs)
        expected_score = 1
        expected_score += self.pos_bigram_freqs[(0, 1)][("t", "a")]
        expected_score += self.pos_bigram_freqs[(1, 2)][("a", "t")]
        expected_score += self.pos_bigram_freqs[(2, 3)][("t", "a")]

        self.assertEqual(score, expected_score)
