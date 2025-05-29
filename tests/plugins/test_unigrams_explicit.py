import math
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.core.corpus import Corpus
from uci_phonotactic_calculator.plugins.core import get_model


@pytest.mark.parametrize(
    "weighted, weight_mode", [(False, "none"), (True, "legacy_log")]
)
def test_fit_unigrams(training_path, weighted, weight_mode):
    """
    Fit a 1-gram model on the five-row verification corpus and check that the
    learned log-probabilities exactly match the hand-computed values from the
    original unit-tests.

    * un-weighted  → log(7/16) / log(9/16)
    * weighted     → log-frequency weighting with legacy formula (log(freq))
    """
    # ----- 1. build Config & Corpus -------------------------------------------------
    cfg: Config = Config.default(
        ngram_order=1,
        boundary_mode="none",
        prob_mode="joint",
        weight_mode=weight_mode,
        smoothing_scheme="none",
    )
    corpus = Corpus(training_path, cfg)

    # ----- 2. train the model -------------------------------------------------------
    Model = get_model("ngram")
    model = Model(cfg)
    model.fit(corpus)

    # model._logprobs is a 1-D array with indices matching corpus.sound_index
    # In this corpus sound_index == ["a", "t"]
    # Type ignore needed because _logprobs is an implementation detail not in BaseModel
    learnt = cast(NDArray[np.float64], model._logprobs)  # type: ignore
    assert corpus.sound_index == ["a", "t"]

    # ----- 3. reference values ------------------------------------------------------
    if not weighted:
        # raw symbol counts: a=9, t=7  (total 16)
        expected = np.log(np.array([9 / 16, 7 / 16]))
    else:
        # legacy weighted totals (identical arithmetic from the Django suite)
        t_total = math.log(10) * 2 + math.log(20) * 3 + math.log(30) * 2
        a_total = math.log(10) * 3 + math.log(20) * 4 + math.log(30) * 2
        total = t_total + a_total
        expected = np.log(np.array([a_total / total, t_total / total]))

    # The expectation is written out as a matrix-style literal so it's visible:
    #
    #   np.array([
    #       #   a_prob ,  t_prob
    #           [   ??? ,    ??? ]
    #   ])
    #
    # Here we keep the same visual style but with the real values substituted.
    explicit_matrix = np.array([[np.exp(expected[0]), np.exp(expected[1])]])
    # Convert log probabilities to actual probabilities for more human-readable output
    actual_probs = np.exp(learnt)

    # ----- 4. assertion -------------------------------------------------------------
    # Include the actual probabilities in the error message for easier debugging
    assert np.allclose(learnt, expected), (
        f"\nExpected log-probs:\n{expected}\nGot log-probs:\n{learnt}"
        f"\n\nExpected probs:\n{explicit_matrix}\nGot probs:\n{actual_probs}"
    )


@pytest.mark.parametrize(
    "weighted, weight_mode", [(False, "none"), (True, "legacy_log")]
)
def test_unigram_score(training_path, weighted, weight_mode):
    """Verify that `model.score()` returns Σ log P(symbol) for a test word."""
    cfg: Config = Config.default(
        ngram_order=1,
        boundary_mode="none",
        prob_mode="joint",
        weight_mode=weight_mode,
        smoothing_scheme="none",
    )
    corpus = Corpus(training_path, cfg)
    model = get_model("ngram")(cfg)
    model.fit(corpus)

    test_tok = ["t", "a", "t", "a"]
    score = model.score(test_tok)

    # manual sum of component log-probs
    a_idx, t_idx = 0, 1  # sound_index is ["a","t"]
    # Use type assertion for implementation-specific attribute
    logprobs = cast(NDArray[np.float64], model._logprobs)  # type: ignore
    expected = logprobs[t_idx] + logprobs[a_idx] + logprobs[t_idx] + logprobs[a_idx]

    assert np.isclose(score, expected)
