"""
Explicit verification of the three NEW aggregation modes
    • arithmetic mean  (registry key:  "mean")
    • geometric mean   (registry key:  "geom")
    • harmonic mean    (registry key:  "harm")

The reference corpus (freqs in parentheses):
    t a           (10)
    a t a         (20)
    t a t a       (30)
    a t t a       (20)
    t a a         (10)

Raw symbol counts:
    a = 9,   t = 7      (total 16)

P(a) = 9/16   ;   P(t) = 7/16
"""

import math
from pathlib import Path

import numpy as np
import pytest

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.core.corpus import Corpus
from uci_phonotactic_calculator.models import aggregate as ag
from uci_phonotactic_calculator.plugins.core import get_model

# ------------------------------------------------------------------ #
# 1)  Pure helper‐function tests (no model involved)
# ------------------------------------------------------------------ #


def test_linear_mean_helper() -> None:
    """Arithmetic mean  log( (p1 + p2) / 2 )"""
    comps = [math.log(0.1), math.log(0.2)]
    expected = math.log((0.1 + 0.2) / 2)
    assert math.isclose(ag.linear_mean(comps), expected)


def test_geometric_mean_helper() -> None:
    """Geometric mean  log( √(p1·p2) )  = (Σ log p) / 2"""
    comps = [math.log(0.1), math.log(0.2)]
    expected = (math.log(0.1) + math.log(0.2)) / 2
    assert math.isclose(ag.geometric_mean(comps), expected)


def test_harmonic_mean_helper() -> None:
    """
    Harmonic mean
        H =  n / Σ(1/p_i)
      log H = log n − log Σ(1/p_i)
    """
    p1, p2 = 0.1, 0.2
    comps = [math.log(p1), math.log(p2)]
    expected = math.log(2) - math.log(1 / p1 + 1 / p2)
    assert math.isclose(ag.harmonic_mean(comps), expected)


# ------------------------------------------------------------------ #
# 2)  Integration tests: the new modes inside N-gram scoring
# ------------------------------------------------------------------ #

TOKEN = ["t", "a", "t", "a"]  # the same word existing in the corpus


@pytest.mark.parametrize(
    "agg_key, expect_linear",
    [
        (
            "mean",
            # Arithmetic mean: ½·P(t) + ½·P(a)          (see doc-string above)
            0.5,
        ),
        (
            "geom",
            # Geometric mean: √(P(t)·P(a))              (4 tokens → exp(Σlog p / 4))
            math.sqrt((7 / 16) * (9 / 16)),
        ),
        (
            "harm",
            # Harmonic mean: 4 / (Σ 1/P(sym))
            4 / (2 * (1 / (7 / 16)) + 2 * (1 / (9 / 16))),
        ),
    ],
    ids=["arithmetic_mean", "geometric_mean", "harmonic_mean"],
)
def test_unigram_score_with_new_aggregators(
    training_path: Path,
    agg_key: str,
    expect_linear: float,
) -> None:
    """
    Fit a **unigram** model with the chosen aggregation mode and verify
    model.score(token) == hand-worked expectation.
    """

    # --- 1. build Config + Corpus ---------------------------------
    cfg = Config.default(
        ngram_order=1,
        boundary_mode="none",
        prob_mode="joint",
        aggregate_mode=agg_key,
        smoothing_scheme="none",
        weight_mode="none",
    )
    corpus = Corpus(training_path, cfg)

    # --- 2. train the model ---------------------------------------
    Model = get_model("ngram")
    model = Model(cfg)
    model.fit(corpus)

    # --- 3. compute expected log-score by hand --------------------
    expect_log = math.log(expect_linear)

    # --- 4. assertion ---------------------------------------------
    got = model.score(TOKEN)
    assert np.isclose(
        got, expect_log, rtol=1e-12
    ), f"\nAgg mode: {agg_key}\nExpected log {expect_log}\nGot      {got}"
