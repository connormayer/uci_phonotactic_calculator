# tests/unit/test_unigrams.py
"""
test_unigrams.py - Unit tests for unigram models.
This module tests both non-positional and positional unigram model fitting,
including configurations with token weighting and smoothing.
"""

import numpy as np
from src.ngram_models import UnigramModel


def test_fit_non_positional_unigrams(token_freqs):
    model = UnigramModel("non_positional")
    model.fit(token_freqs)
    # Expected: 't' → log(7/16), 'a' → log(9/16)
    assert np.isclose(model.model_data["t"], np.log(7 / 16), atol=1e-6)
    assert np.isclose(model.model_data["a"], np.log(9 / 16), atol=1e-6)


def test_fit_non_positional_unigrams_smoothed(token_freqs):
    model = UnigramModel("non_positional", smoothed=True)
    model.fit(token_freqs)
    assert np.isclose(model.model_data["t"], np.log(8 / 18), atol=1e-6)
    assert np.isclose(model.model_data["a"], np.log(10 / 18), atol=1e-6)


def test_fit_non_positional_unigrams_weighted(token_freqs):
    model = UnigramModel("non_positional", token_weighted=True)
    model.fit(token_freqs)
    t_total = np.log(10) * 2 + np.log(20) * 3 + np.log(30) * 2
    a_total = np.log(10) * 3 + np.log(20) * 4 + np.log(30) * 2
    total = t_total + a_total
    expected = {"t": np.log(t_total / total), "a": np.log(a_total / total)}
    for key in expected:
        assert np.isclose(model.model_data[key], expected[key], atol=1e-6)


def test_fit_non_positional_unigrams_smoothed_weighted(token_freqs):
    model = UnigramModel("non_positional", smoothed=True, token_weighted=True)
    model.fit(token_freqs)
    t_total = np.log(10) * 2 + np.log(20) * 3 + np.log(30) * 2 + 1
    a_total = np.log(10) * 3 + np.log(20) * 4 + np.log(30) * 2 + 1
    total = t_total + a_total
    expected = {"t": np.log(t_total / total), "a": np.log(a_total / total)}
    for key in expected:
        assert np.isclose(model.model_data[key], expected[key], atol=1e-6)


def test_fit_positional_unigrams(token_freqs):
    model = UnigramModel("positional")
    model.fit(token_freqs)
    assert np.isclose(model.model_data[0]["t"], 3 / 5, atol=1e-6)
    assert np.isclose(model.model_data[0]["a"], 2 / 5, atol=1e-6)
    assert np.isclose(model.model_data[1]["t"], 2 / 5, atol=1e-6)
    assert np.isclose(model.model_data[1]["a"], 3 / 5, atol=1e-6)
    assert np.isclose(model.model_data[2]["t"], 2 / 4, atol=1e-6)
    assert np.isclose(model.model_data[2]["a"], 2 / 4, atol=1e-6)
    assert np.isclose(model.model_data[3]["t"], 0 / 2, atol=1e-6)
    assert np.isclose(model.model_data[3]["a"], 2 / 2, atol=1e-6)


def test_fit_positional_unigrams_weighted(token_freqs):
    model = UnigramModel("positional", token_weighted=True)
    model.fit(token_freqs)
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
    assert np.isclose(model.model_data[0]["t"], t_0 / total_0, atol=1e-6)
    assert np.isclose(model.model_data[0]["a"], a_0 / total_0, atol=1e-6)
    assert np.isclose(model.model_data[1]["t"], t_1 / total_1, atol=1e-6)
    assert np.isclose(model.model_data[1]["a"], a_1 / total_1, atol=1e-6)
    assert np.isclose(model.model_data[2]["t"], t_2 / total_2, atol=1e-6)
    assert np.isclose(model.model_data[2]["a"], a_2 / total_2, atol=1e-6)
    assert np.isclose(model.model_data[3]["t"], t_3 / total_3, atol=1e-6)
    assert np.isclose(model.model_data[3]["a"], a_3 / total_3, atol=1e-6)


def test_fit_positional_unigrams_smoothed(token_freqs):
    model = UnigramModel("positional", smoothed=True)
    model.fit(token_freqs)
    assert np.isclose(model.model_data[0]["t"], 4 / 7, atol=1e-6)
    assert np.isclose(model.model_data[0]["a"], 3 / 7, atol=1e-6)
    assert np.isclose(model.model_data[1]["t"], 3 / 7, atol=1e-6)
    assert np.isclose(model.model_data[1]["a"], 4 / 7, atol=1e-6)
    assert np.isclose(model.model_data[2]["t"], 3 / 6, atol=1e-6)
    assert np.isclose(model.model_data[2]["a"], 3 / 6, atol=1e-6)
    assert np.isclose(model.model_data[3]["t"], 1 / 4, atol=1e-6)
    assert np.isclose(model.model_data[3]["a"], 3 / 4, atol=1e-6)


def test_fit_positional_unigrams_smoothed_weighted(token_freqs):
    model = UnigramModel("positional", smoothed=True, token_weighted=True)
    model.fit(token_freqs)
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
    assert np.isclose(model.model_data[0]["t"], t_0 / total_0, atol=1e-6)
    assert np.isclose(model.model_data[0]["a"], a_0 / total_0, atol=1e-6)
    assert np.isclose(model.model_data[1]["t"], t_1 / total_1, atol=1e-6)
    assert np.isclose(model.model_data[1]["a"], a_1 / total_1, atol=1e-6)
    assert np.isclose(model.model_data[2]["t"], t_2 / total_2, atol=1e-6)
    assert np.isclose(model.model_data[2]["a"], a_2 / total_2, atol=1e-6)
    assert np.isclose(model.model_data[3]["t"], t_3 / total_3, atol=1e-6)
    assert np.isclose(model.model_data[3]["a"], a_3 / total_3, atol=1e-6)
