"""
test_ngram_model_combinations.py - Tests for the unified n-gram model interface and overall normalization checks.
This module verifies both positional and non-positional models in conditional and joint modes,
with and without word boundaries, as well as the unified interface (UnigramModel and BigramModel)
using configuration dictionaries from get_model_configs.
It ensures that all models produce properly normalized probability distributions.
"""

import unittest
import numpy as np
from src.io_utils import read_tokens
from src.ngram_models import (
    fit_positional_bigrams,
    fit_bigrams,
    fit_non_positional_bigrams,
    UnigramModel,
    BigramModel
)
from src.model_configs import get_model_configs

TRAINING_FILE = 'data/unit_test_data/unit_test_training_data.txt'

class TestNgramModelCombinations(unittest.TestCase):
    def setUp(self):
        self.token_freqs = read_tokens(TRAINING_FILE)
        self.unique_sounds = sorted(list(set(
            sound for token, _ in self.token_freqs for sound in token
        )))
        self.boundary = '#'
        self.sounds_with_boundary = self.unique_sounds + [self.boundary]

    def testPositionalBigrams_Conditional_NoWB(self):
        pos_bigrams = fit_positional_bigrams(self.token_freqs, conditional=True, use_word_boundaries=False)
        for pos in pos_bigrams:
            counts_by_prev = {}
            for (prev, nxt), prob in pos_bigrams[pos].items():
                counts_by_prev.setdefault(prev, 0)
                counts_by_prev[prev] += prob
            for prev, total in counts_by_prev.items():
                self.assertAlmostEqual(total, 1.0, places=6)

    def testPositionalBigrams_Joint_NoWB(self):
        pos_bigrams = fit_positional_bigrams(self.token_freqs, conditional=False, use_word_boundaries=False)
        for pos in pos_bigrams:
            total = sum(pos_bigrams[pos].values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def testPositionalBigrams_Conditional_WB(self):
        pos_bigrams = fit_positional_bigrams(self.token_freqs, conditional=True, use_word_boundaries=True)
        for pos in pos_bigrams:
            counts_by_prev = {}
            for (prev, nxt), prob in pos_bigrams[pos].items():
                counts_by_prev.setdefault(prev, 0)
                counts_by_prev[prev] += prob
            for prev, total in counts_by_prev.items():
                self.assertAlmostEqual(total, 1.0, places=6)

    def testPositionalBigrams_Joint_WB(self):
        pos_bigrams = fit_positional_bigrams(self.token_freqs, conditional=False, use_word_boundaries=True)
        for pos in pos_bigrams:
            total = sum(pos_bigrams[pos].values())
            self.assertAlmostEqual(total, 1.0, places=6)

    def testNonPositionalBigrams_Conditional_WB(self):
        bp = fit_bigrams(self.token_freqs, self.sounds_with_boundary, use_word_boundaries=True)
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)

    def testNonPositionalBigrams_Joint_WB(self):
        bp = fit_non_positional_bigrams(self.token_freqs, use_word_boundaries=True)
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)

    def testNonPositionalBigrams_Conditional_NoWB(self):
        bp = fit_bigrams(self.token_freqs, self.unique_sounds, use_word_boundaries=False)
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)

    def testNonPositionalBigrams_Joint_NoWB(self):
        bp = fit_non_positional_bigrams(self.token_freqs, use_word_boundaries=False)
        for col in range(bp.shape[1]):
            col_probs = np.exp(bp[:, col])
            total = np.sum(col_probs)
            self.assertAlmostEqual(total, 1.0, places=6)
    
    def testUnifiedModelInterface(self):
        """Test the unified model interface using model configurations from get_model_configs()."""
        model_configs = get_model_configs()
        models = {}
        for config in model_configs:
            if config["model"] == "unigram":
                prob_type = "joint" if config.get("joint", False) else "log"
                model = UnigramModel(
                    position=config["position"],
                    prob_type=prob_type,
                    smoothed=config["smoothed"],
                    token_weighted=config["token_weighted"],
                    aggregation=config.get("aggregation", "sum")
                ).fit(self.token_freqs, self.unique_sounds)
            elif config["model"] == "bigram":
                prob_type = "conditional" if config.get("conditional", False) else "joint"
                model = BigramModel(
                    position=config["position"],
                    prob_type=prob_type,
                    use_boundaries=config["use_boundaries"],
                    smoothed=config["smoothed"],
                    token_weighted=config["token_weighted"],
                    aggregation=config.get("aggregation", "sum")
                ).fit(self.token_freqs, self.unique_sounds)
            models[config["name"]] = model
        
        # Score a sample token using each model and verify that the score is a valid numerical value (i.e., not NaN).
        sample_token = ['t', 'a']
        for config in model_configs:
            score = models[config["name"]].score(sample_token, self.unique_sounds)
            self.assertFalse(np.isnan(score))

if __name__ == '__main__':
    unittest.main()