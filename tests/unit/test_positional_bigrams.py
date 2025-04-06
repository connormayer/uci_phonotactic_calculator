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
from src.ngram_models import fit_bigram_positional, generate_bigrams, WORD_BOUNDARY, BigramModel
from src.ngram_models import normalize_positional_counts

##################
# NONPOS COND WB #
##################
def test_fit_bigrams_positional_conditional_WB(token_freqs, sound_index):
    """
    Test fit_bigram_conditional_positional_nwb() with no word boundaries, no smoothing, no weighting.
    (Conditional, positional bigram model without WB.)
    """
    sounds = sound_index + [WORD_BOUNDARY]
    model = BigramModel("positional", prob_type="conditional", use_boundaries=True)
    model.fit(token_freqs, sounds)
    assert model.model_data[(0, 1)][('#', 'a')] == 2/5
    assert model.model_data[(0, 1)][('#', 't')] == 3/5

    assert model.model_data[(1, 2)][('t', 'a')] == 3/3
    assert model.model_data[(1, 2)][('a', 't')] == 2/2

    assert model.model_data[(2, 3)][('t', 'a')] == 1/2
    assert model.model_data[(2, 3)][('t', 't')] == 1/2
    assert model.model_data[(2, 3)][('a', 't')] == 1/3
    assert model.model_data[(2, 3)][('a', 'a')] == 1/3
    assert model.model_data[(2, 3)][('a', '#')] == 1/3

    assert model.model_data[(3, 4)][('t', 'a')] == 2/2
    assert model.model_data[(3, 4)][('a', '#')] == 2/2

    assert model.model_data[(4,5)][('a', '#')] == 2/2 

###################
# NONPOS JOINT WB #
###################
def test_fit_bigrams_positional_joint_WB(token_freqs, sound_index):
    """
    Test fit_bigrams_joint_positional_nwb() with no word boundaries, no smoothing, no weighting.
    (Joint, positional bigram model without WB.)
    """
    model = BigramModel("positional", prob_type="joint", use_boundaries=True)
    model.fit(token_freqs, sound_index)

    assert model.model_data[(0, 1)][('#', 'a')] == 2/5
    assert model.model_data[(0, 1)][('#', 't')] == 3/5

    assert model.model_data[(1, 2)][('t', 'a')] == 3/5
    assert model.model_data[(1, 2)][('a', 't')] == 2/5

    assert model.model_data[(2, 3)][('t', 'a')] == 1/5
    assert model.model_data[(2, 3)][('a', 't')] == 1/5
    assert model.model_data[(2, 3)][('t', 't')] == 1/5
    assert model.model_data[(2, 3)][('a', 'a')] == 1/5
    assert model.model_data[(2, 3)][('a', '#')] == 1/5

    assert model.model_data[(3, 4)][('t', 'a')] == 2/4
    assert model.model_data[(3, 4)][('a', '#')] == 2/4

    assert model.model_data[(4,5)][('a', '#')] == 2/2 

###################
# NONPOS COND NWB #
###################
def test_fit_bigrams_positional_conditional_NWB(token_freqs, sound_index):
    """
    Test fit_bigram_conditional_positional_nwb() with no word boundaries, no smoothing, no weighting.
    (Conditional, positional bigram model without WB.)
    """
    model = BigramModel("positional", prob_type="conditional", use_boundaries=False)
    model.fit(token_freqs, sound_index)
    assert model.model_data[(0, 1)][('t', 'a')] == 3/3
    assert model.model_data[(0, 1)][('a', 't')] == 2/2

    assert model.model_data[(1, 2)][('t', 'a')] == 1/2
    assert model.model_data[(1, 2)][('a', 't')] == 1/2
    assert model.model_data[(1, 2)][('t', 't')] == 1/2
    assert model.model_data[(1, 2)][('a', 'a')] == 1/2

    assert model.model_data[(2, 3)][('t', 'a')] == 2/2

####################
# NONPOS JOINT NWB #
####################
def test_fit_bigrams_positional_joint_NWB(token_freqs, sound_index):
    """
    Test fit_bigrams_joint_positional_nwb() with no word boundaries, no smoothing, no weighting.
    (Joint, positional bigram model without WB.)
    """
    model = BigramModel("positional", prob_type="joint", use_boundaries=False)
    model.fit(token_freqs, sound_index)
    assert model.model_data[(0, 1)][('t', 'a')] == 3/5
    assert model.model_data[(0, 1)][('a', 't')] == 2/5

    assert model.model_data[(1, 2)][('t', 'a')] == 1/4
    assert model.model_data[(1, 2)][('a', 't')] == 1/4
    assert model.model_data[(1, 2)][('t', 't')] == 1/4
    assert model.model_data[(1, 2)][('a', 'a')] == 1/4

    assert model.model_data[(2, 3)][('t', 'a')] == 2/2

###############
# AGGREGATION #
###############
def test_fit_bigram_positional_score_sum(token_freqs, sound_index):
    """
    Quick test of the scoring for a positional bigram model with sum aggregation.
    This verifies we can compute a finite linear-based score.
    """
    sounds = sound_index + [WORD_BOUNDARY]
    model = BigramModel(
        position="positional",
        prob_type="joint",  
        use_boundaries=False,
        smoothed=False,
        token_weighted=False,
        aggregation="sum"
    )
    model.fit(token_freqs, sounds) 

    # Score a simple token
    token = ['t', 'a', 't', 'a']
    score_val = model.score(token, sounds)
    
    assert score_val == (1 + 
                         model.model_data[(0, 1)][('t', 'a')] +
                         model.model_data[(1, 2)][('a', 't')] +
                         model.model_data[(2, 3)][('t', 'a')]
    )

def test_fit_bigram_positional_score_prod(token_freqs, sound_index):
    """
    Quick test of the scoring for a positional bigram model with sum aggregation.
    This verifies we can compute a finite linear-based score.
    """
    sounds = sound_index + [WORD_BOUNDARY]
    model = BigramModel(
        position="positional",
        prob_type="joint",  
        use_boundaries=False,
        smoothed=False,
        token_weighted=False,
        aggregation="prod"
    )
    model.fit(token_freqs, sounds) 

    # Score a simple token
    token = ['t', 'a', 't', 'a']
    score_val = model.score(token, sounds)

    assert score_val == (np.log(model.model_data[(0, 1)][('t', 'a')]) +
                         np.log(model.model_data[(1, 2)][('a', 't')]) +
                         np.log(model.model_data[(2, 3)][('t', 'a')])
    )
