"""
Explicit verification that the built-in probability-transform classes
(`joint`, `conditional`) return the exact log-probabilities expected
from the five-row reference corpus.

Corpus lines (freqs in parentheses):
    t a           (10)
    a t a         (20)
    t a t a       (30)
    a t t a       (20)
    t a a         (10)
"""

import numpy as np
import pytest

from uci_phonotactic_calculator.core.config import Config
from uci_phonotactic_calculator.core.corpus import Corpus
from uci_phonotactic_calculator.plugins.core import (
    ProbTransformRegistry,
    get_prob_transform,
)

# ------------------------------------------------------------------ #
# Helpers – canonical order of symbols we care about
SYMS = ["a", "t", "#"]  # '#' is the boundary symbol used by boundary_scheme='word'
IDX = {s: i for i, s in enumerate(SYMS)}
# ------------------------------------------------------------------ #


def _bigram_counts_from_corpus(corpus: Corpus) -> np.ndarray:
    """
    Return a 3×3 dense count matrix in the fixed A/T/# order.
    boundary_mode='both' so every token is framed with pads.
    """
    M = np.zeros((3, 3), dtype=float)
    for tok in corpus.tokens:
        grams = Corpus.generate_ngrams(
            tok,
            n=2,
            boundary_mode="both",
            boundary="#",
        )
        for prev, nxt in grams:
            i, j = IDX[prev], IDX[nxt]
            M[i, j] += 1.0
    return M


# ------------------------------------------------------------------ #
# 1) Joint-probability explicitly checked against hand maths
# ------------------------------------------------------------------ #
def test_joint_transform_unigrams(training_path):
    # raw symbol counts: a=9, t=7  (verified by earlier unigram test)
    counts = np.array([9.0, 7.0])
    tf = get_prob_transform("joint")
    logp = tf.transform(counts)

    expected = np.log(np.array([9 / 16, 7 / 16]))
    assert np.allclose(logp, expected), f"\nExpected {expected}\nGot {logp}"


# ------------------------------------------------------------------ #
# 2) Conditional-probability bigram matrix from the corpus
# ------------------------------------------------------------------ #
def test_conditional_transform_bigrams(training_path):
    cfg = Config.default(ngram_order=2, boundary_mode="both")
    corpus = Corpus(training_path, cfg)

    counts = _bigram_counts_from_corpus(corpus)

    # Sanity-check the raw counts against the hand-worked matrix:
    #    prev ⟶  next    a   t   #
    expected_counts = np.array(
        [
            # a-row
            [1, 3, 5],
            # t-row
            [6, 1, 0],
            # #-row
            [2, 3, 0],
        ],
        dtype=float,
    )
    assert np.array_equal(counts, expected_counts), (
        f"\nRaw bigram counts differ from expectation\n"
        f"Expected:\n{expected_counts}\nGot:\n{counts}"
    )

    # Feed the counts into the Conditional transform
    tf = get_prob_transform("conditional")
    logp = tf.transform(counts)

    # Hand-worked linear probabilities, row-normalised
    expected_linear = np.array(
        [
            [1 / 9, 3 / 9, 5 / 9],  # a→
            [6 / 7, 1 / 7, 0 / 7],  # t→
            [2 / 5, 3 / 5, 0 / 5],  # #→
        ]
    )
    expected_log = np.log(expected_linear, where=(expected_linear > 0))
    expected_log[expected_linear == 0] = float("-inf")

    assert np.allclose(
        logp, expected_log, equal_nan=False
    ), f"\nExpected log-probs:\n{expected_log}\nGot:\n{logp}"

    # Extra guard: each row must sum to ≈1 in linear space
    linear = np.exp(
        logp,
        where=~np.isneginf(logp),
        out=np.zeros_like(logp),  # zero-fill masked positions
    )
    row_sums = linear.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums are not 1: {row_sums}"


# ------------------------------------------------------------------ #
# 3) Transform registry completeness – every registered transform
#    must at least round-trip a toy array without NaN.
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("name", sorted(ProbTransformRegistry))
def test_every_registered_transform_has_finite_output(name):
    tf = get_prob_transform(name)
    toy = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = tf.transform(toy)
    assert not np.isnan(out).any()
