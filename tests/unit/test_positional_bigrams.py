# tests/unit/test_bigrams_positional.py
"""
tests/unit/test_bigrams_positional.py -- Unit tests for positional bigram models.
=======================================================================
Verifies that the positional bigram probability dictionaries are computed
correctly for different configurations: conditional vs. joint, boundaries
vs. no boundaries, smoothing, and token weighting.
"""

import pytest
import numpy as np
from collections import defaultdict
from src.ngram_models import fit_bigram_positional, generate_bigrams, WORD_BOUNDARY
from src.ngram_models import normalize_positional_counts

def test_fit_bigram_positional_conditional_wb_no_smoothing_no_weighting(token_freqs):
    """
    Manually verify a core scenario:
    - conditional=True
    - use_word_boundaries=True
    - smoothed=False
    - token_weighted=False

    Checks that probabilities at the first position (0,1) match expected.
    """
    pos_bi_freqs = fit_bigram_positional(
        token_freqs,
        token_weighted=False,
        smoothed=False,
        conditional=True,
        use_word_boundaries=True
    )

    # For each token in token_freqs, we insert boundaries and get the (0,1) bigram.
    # The training data in unit_test_training_data.txt is:
    #
    #   1) t a
    #   2) a t a
    #   3) t a t a
    #   4) a t t a
    #   5) t a a
    #
    # With boundaries, each token is prefixed/suffixed by '#'.
    # So for the first bigram position (0,1), we compare (#, symbol).
    #
    # Counting how many tokens start with 't' vs. 'a':
    #   (#, 't') => 3 times (lines 1, 3, 5)
    #   (#, 'a') => 2 times (lines 2, 4)
    #
    # Because it's conditional on '#', we normalize (#, 't') + (#, 'a') = 1.0
    #
    # Expected: (#, 't') => 3/5, (#, 'a') => 2/5
    #
    # We'll verify that in pos_bi_freqs[(0,1)].

    first_position_dict = pos_bi_freqs.get((0, 1), {})
    # Check the relevant bigrams
    prob_hash_t = first_position_dict.get((WORD_BOUNDARY, 't'), 0.0)
    prob_hash_a = first_position_dict.get((WORD_BOUNDARY, 'a'), 0.0)

    assert np.isclose(prob_hash_t, 3/5, atol=1e-6), \
        f"Expected (#, 't') = 3/5, got {prob_hash_t}"
    assert np.isclose(prob_hash_a, 2/5, atol=1e-6), \
        f"Expected (#, 'a') = 2/5, got {prob_hash_a}"

    # Also check that all other bigrams with # at pos (0,1) are zero or missing
    # in that dictionary
    for (symbol1, symbol2), val in first_position_dict.items():
        if symbol1 == WORD_BOUNDARY and symbol2 not in ('t', 'a'):
            assert val == 0.0, f"Unexpected nonzero bigram freq for {symbol1, symbol2}: {val}"


@pytest.mark.parametrize("conditional, use_boundaries, smoothed, token_weighted", [
    (False, False, False, False),
    (False, False, True,  False),
    (False, True,  False, False),
    (True,  True,  True,  False),
    (True,  True,  False, True),
    (False, True,  True,  True),
    (True,  False, False, False),
    (True,  False, True,  True),
])
def test_fit_bigram_positional_all_combos(token_freqs, conditional, use_boundaries, smoothed, token_weighted):
    """
    Parametrized test that confirms the shape and valid probability sums
    for multiple combinations of positional bigram configurations.
    We do not manually verify each cell but ensure no invalid probabilities
    and correct normalization (conditional vs. joint).
    """
    pos_bi_freqs = fit_bigram_positional(
        token_freqs,
        token_weighted=token_weighted,
        smoothed=smoothed,
        conditional=conditional,
        use_word_boundaries=use_boundaries
    )

    # For joint: each position's sums should total 1.0 across all bigrams.
    # For conditional: each preceding symbol in that position set should sum to 1.0
    # across its next symbols.
    # We'll verify these constraints in linear space.

    if conditional:
        # Sums by the preceding sound
        for (pos1, pos2), bigram_dict in pos_bi_freqs.items():
            # Group the bigrams by their first symbol
            sums_by_symbol = defaultdict(float)
            for (s1, s2), prob in bigram_dict.items():
                sums_by_symbol[s1] += prob
                assert 0.0 <= prob <= 1.0, f"Invalid prob {prob} for bigram {s1, s2}"
            for s1, agg_val in sums_by_symbol.items():
                # Some positions might have zero bigrams if not present in training
                if agg_val > 0.0:
                    assert np.isclose(agg_val, 1.0, atol=1e-4), \
                        f"Conditional sums for symbol '{s1}' at position {(pos1,pos2)} must be ~1, got {agg_val}"
    else:
        # Joint sums for each position
        for (pos1, pos2), bigram_dict in pos_bi_freqs.items():
            total = 0.0
            for (s1, s2), prob in bigram_dict.items():
                total += prob
                assert 0.0 <= prob <= 1.0, f"Invalid prob {prob} for bigram {s1, s2}"
            # Some positions might not appear often, so total can be near zero. If > 0, it should be ~1.
            if total > 0.0:
                assert np.isclose(total, 1.0, atol=1e-4), \
                    f"Joint sums at position {(pos1,pos2)} must be ~1, got {total}"


def test_fit_bigram_positional_score_sum(token_freqs):
    """
    Quick test of the scoring for a positional bigram model with sum aggregation.
    This verifies we can compute a finite linear-based score.
    """
    from src.ngram_models import BigramModel

    model = BigramModel(
        position="positional",
        prob_type="conditional",  # test conditional bigrams
        use_boundaries=True,
        smoothed=False,
        token_weighted=False,
        aggregation="sum"
    ).fit(token_freqs, sound_index=['t', 'a', WORD_BOUNDARY])  # pass a minimal sound_index

    # Score a simple token
    token = ['t', 'a']
    score_val = model.score(token, sound_index=['t', 'a', WORD_BOUNDARY])

    assert not np.isnan(score_val), "Positional bigram sum-based score returned NaN."
    # Expect a numeric value >= 1.0 in linear space for 'sum' aggregator
    assert score_val >= 1.0, f"Expected score >= 1.0, got {score_val}"


def test_fit_bigram_positional_score_prod(token_freqs):
    """
    Quick test of the scoring for a positional bigram model with product aggregation.
    This verifies we get a log-prob back (finite or -inf).
    """
    from src.ngram_models import BigramModel

    model = BigramModel(
        position="positional",
        prob_type="conditional",
        use_boundaries=True,
        smoothed=False,
        token_weighted=False,
        aggregation="prod"
    ).fit(token_freqs, sound_index=['t', 'a', WORD_BOUNDARY])

    token = ['t', 'x']  # 'x' might not be in the training set
    score_val = model.score(token, sound_index=['t', 'a', WORD_BOUNDARY])

    # Score should be a log-prob. If 'x' is unknown, we might get -inf:
    assert score_val <= 0 or np.isinf(score_val), \
        f"Expected a non-positive or -inf log-prob for unknown symbol, got {score_val}"

# End of tests/unit/test_bigrams_positional.py