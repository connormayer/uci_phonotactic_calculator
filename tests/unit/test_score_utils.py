"""
tests/unit/test_score_utils.py - Unit tests for scoring functions using consistent sound_index naming.
This module verifies that the generic scoring functions compute the correct values
using mock probability dictionaries.

"""

import numpy as np
import pytest
from src.score_utils import generic_unigram_score, generic_bigram_score, generic_pos_unigram_score, generic_pos_bigram_score
from src.ngram_models import WORD_BOUNDARY
import nltk

def test_generic_unigram_score_log():
    model_data = {'a': 0.0, 'b': -0.6931}
    token = ['a', 'b']
    score = generic_unigram_score(token, model_data, mode='log')
    assert np.isclose(score, -0.6931, atol=1e-3)

def test_generic_unigram_score_joint():
    model_data = {'a': 0.0, 'b': -0.6931}
    token = ['a', 'b']
    score = generic_unigram_score(token, model_data, mode='joint')
    expected = 1 + 1 + 0.5
    assert np.isclose(score, 2.5, atol=1e-3)

def test_generic_bigram_score_with_boundaries():
    sound_index = ['#', 'a', 'b']
    matrix = np.zeros((3,3))
    token = ['a', 'b']
    score = generic_bigram_score(token, matrix, sound_index, use_word_boundaries=True)
    assert np.isclose(score, 0.0, atol=1e-3)

def test_generic_pos_unigram_score_sum():
    pos_uni_freqs = {0: {'a': 0.2}, 1: {'b': 0.3}}
    token = ['a', 'b']
    score = generic_pos_unigram_score(token, pos_uni_freqs, aggregation="sum")
    assert np.isclose(score, 1.0 + 0.2 + 0.3, atol=1e-3)

def test_generic_pos_unigram_score_prod():
    pos_uni_freqs = {0: {'a': 0.2}, 1: {'b': 0.3}}
    token = ['a', 'b']
    score = generic_pos_unigram_score(token, pos_uni_freqs, aggregation="prod")
    expected = np.log(0.2 * 0.3)
    assert np.isclose(score, expected, atol=1e-3)

def test_generic_pos_bigram_score_sum():
    pos_bi_freqs = {
        (0, 1): {('#', 'a'): 0.1},
        (1, 2): {('a', 'b'): 0.2},
        (2, 3): {('b', '#'): 0.3}
    }
    token = ['a', 'b']
    score = generic_pos_bigram_score(token, pos_bi_freqs, conditional=False, use_word_boundaries=True, aggregation="sum")
    assert np.isclose(score, 1.0 + 0.1 + 0.2 + 0.3, atol=1e-3)

def test_generic_pos_bigram_score_prod():
    pos_bi_freqs = {
        (0, 1): {('#', 'a'): 0.1},
        (1, 2): {('a', 'b'): 0.2},
        (2, 3): {('b', '#'): 0.3}
    }
    token = ['a', 'b']
    score = generic_pos_bigram_score(token, pos_bi_freqs, conditional=False, use_word_boundaries=True, aggregation="prod")
    expected = np.log(0.1 * 0.2 * 0.3)
    assert np.isclose(score, expected, atol=1e-3)

# End of tests/unit/test_score_utils.py