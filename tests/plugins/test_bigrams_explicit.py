import math
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.core.corpus import Corpus
from uci_phonotactic_calculator.plugins.core import get_model


@pytest.mark.parametrize(
    "weighted, smoothed, weight_mode, smooth_scheme",
    [
        (False, False, "none", "none"),  # plain
        (False, True, "none", "laplace"),  # +1 smoothing
        (True, False, "legacy_log", "none"),  # token-weighted
        (True, True, "legacy_log", "laplace"),  # weighted + smoothing
    ],
    ids=["plain", "smoothed", "weighted", "weighted+smoothed"],
)
def test_fit_bigrams(training_path, weighted, smoothed, weight_mode, smooth_scheme):
    """
    Verifies the learned conditional bigram matrix against the hand-computed
    values from the original test-suite.

    Rows = previous symbol  (a, t, #)   ;  Cols = next symbol (a, t, #)
    """
    # 1️⃣  Build Config and Corpus
    cfg = Config.default(
        ngram_order=2,
        boundary_mode="both",  # '#' on both sides, like the legacy suite
        prob_mode="conditional",
        weight_mode=weight_mode,
        smoothing_scheme=smooth_scheme,
    )
    corpus = Corpus(training_path, cfg)

    # 2️⃣  Train
    Model = get_model("ngram")
    model = Model(cfg)
    model.fit(corpus)

    # 3️⃣  Expected matrix (linear space) copied from the old tests
    if not weighted and not smoothed:
        expected_linear = np.array(
            [
                # a_      t_      #_
                [1 / 9, 3 / 9, 5 / 9],
                [6 / 7, 1 / 7, 0 / 7],
                [2 / 5, 3 / 5, 0 / 5],
            ]
        )
    elif not weighted and smoothed:
        expected_linear = np.array(
            [
                [2 / 12, 4 / 12, 6 / 12],
                [7 / 10, 2 / 10, 1 / 10],
                [3 / 8, 4 / 8, 1 / 8],
            ]
        )
    elif weighted and not smoothed:
        # Shorthand names from the legacy test
        wb_t = math.log(10) * 2 + math.log(30)  # (#, t)
        wb_a = math.log(20) * 2  # (#, a)
        wb_wb = 0

        t_a = math.log(10) * 2 + math.log(20) * 2 + math.log(30) * 2
        t_t = math.log(20)
        t_wb = 0

        a_t = math.log(20) * 2 + math.log(30)
        a_a = math.log(10)
        a_wb = math.log(10) * 2 + math.log(20) * 2 + math.log(30)

        wb_c = wb_t + wb_a + wb_wb
        t_c = t_a + t_t + t_wb
        a_c = a_t + a_a + a_wb

        expected_linear = np.array(
            [
                [a_a / a_c, a_t / a_c, a_wb / a_c],
                [t_a / t_c, t_t / t_c, t_wb / t_c],
                [wb_a / wb_c, wb_t / wb_c, wb_wb / wb_c],
            ]
        )
    else:  # weighted and smoothed
        wb_t = math.log(10) * 2 + math.log(30) + 1
        wb_a = math.log(20) * 2 + 1
        wb_wb = 1

        t_a = math.log(10) * 2 + math.log(20) * 2 + math.log(30) * 2 + 1
        t_t = math.log(20) + 1
        t_wb = 1

        a_t = math.log(20) * 2 + math.log(30) + 1
        a_a = math.log(10) + 1
        a_wb = math.log(10) * 2 + math.log(20) * 2 + math.log(30) + 1

        wb_c = wb_t + wb_a + wb_wb
        t_c = t_a + t_t + t_wb
        a_c = a_t + a_a + a_wb

        expected_linear = np.array(
            [
                [a_a / a_c, a_t / a_c, a_wb / a_c],
                [t_a / t_c, t_t / t_c, t_wb / t_c],
                [wb_a / wb_c, wb_t / wb_c, wb_wb / wb_c],
            ]
        )

    expected_log = np.log(expected_linear, where=(expected_linear > 0))
    expected_log[expected_linear == 0] = float("-inf")

    # 4️⃣  Compare with a sensible float tolerance
    # Use type assertion for implementation-specific attribute
    logprobs = cast(NDArray[np.float64], model._logprobs)  # type: ignore
    assert np.allclose(logprobs, expected_log, rtol=5e-3, atol=1e-8), (
        f"\nExpected:\n{expected_log}\n\n"
        f"Got:\n{logprobs}\n\n"
        f"Sound index: {corpus.sound_index}"
    )
