"""
tests/unit/test_ngram_model_combinations.py - Unit tests for the unified n-gram model interface.
This module verifies that both unigram and bigram models (across all configurations)
return correct and finite scores based on their configuration dictionaries.
"""

import numpy as np
import pytest
from src.ngram_models import UnigramModel, BigramModel
from src.model_configs import get_model_configs

def test_unified_model_interface_unigram_expected(token_freqs, unique_sounds):
    """
    Test that a non-positional unigram model in log mode returns the expected score.
    The expected score is computed by summing the log probabilities for each symbol.
    """
    model = UnigramModel(position="non_positional", prob_type="log", smoothed=False, token_weighted=False)
    model.fit(token_freqs, unique_sounds)
    token = ['t', 'a']
    expected_score = 0.0
    for symbol in token:
        expected_score += model.model_data.get(symbol, float('-inf'))
    computed_score = model.score(token, unique_sounds)
    assert np.isclose(computed_score, expected_score, atol=1e-6), \
        f"Computed score {computed_score} != expected {expected_score}"

def test_unified_model_interface_all(token_freqs, unique_sounds):
    """
    Test that all models generated from the configuration dictionaries return finite scores.
    Iterates over every model configuration to ensure no model returns NaN for a given token.
    """
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
            ).fit(token_freqs, unique_sounds)
        elif config["model"] == "bigram":
            prob_type = "conditional" if config.get("conditional", False) else "joint"
            model = BigramModel(
                position=config["position"],
                prob_type=prob_type,
                use_boundaries=config["use_boundaries"],
                smoothed=config["smoothed"],
                token_weighted=config["token_weighted"],
                aggregation=config.get("aggregation", "sum")
            ).fit(token_freqs, unique_sounds)
        models[config["name"]] = model

    token = ['t', 'a']
    for name, model in models.items():
        score = model.score(token, unique_sounds)
        assert not np.isnan(score), f"Model {name} returned NaN for token {token}"

# End of tests/unit/test_ngram_model_combinations.py