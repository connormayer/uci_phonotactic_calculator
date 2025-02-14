"""
score_utils.py - Utility module for generic scoring of tokens using n-gram models.
This module provides functions to compute scores for unigram and bigram models,
including positional models with configurable aggregation (sum or product).
Version: 1.1.0
"""

import nltk
import numpy as np

WORD_BOUNDARY = '#'  # Constant used to denote word boundaries

def generic_unigram_score(token, model_data, mode='log'):
    """
    Generic scoring for unigram models.

    Parameters:
      token: List of symbols (the word).
      model_data: Dictionary mapping symbols to log probabilities.
      mode: 'log' for summing log probabilities, 'joint' for summing probabilities in a joint model.

    Returns:
      Total score (float).
    """
    if mode == 'log':
        total = 0.0
        for sound in token:
            prob = model_data.get(sound, float('-inf'))
            total += prob
            if np.isinf(total):
                return float('-inf')
        return total
    elif mode == 'joint':
        total = 1.0
        for sound in token:
            log_p = model_data.get(sound, float('-inf'))
            if np.isinf(log_p):
                continue
            total += np.exp(log_p)
        return total
    else:
        raise ValueError("Invalid mode. Use 'log' or 'joint'.")

def generic_bigram_score(token, model_data, sound_idx, use_word_boundaries=True):
    """
    Generic scoring for non-positional bigram models.

    Parameters:
      token: List of symbols (the word).
      model_data: 2D numpy array of log probabilities.
      sound_idx: List of sounds used to index rows/columns in model_data.
      use_word_boundaries: Whether to add word boundary markers.

    Returns:
      Total log score (float).
    """
    if use_word_boundaries:
        token = [WORD_BOUNDARY] + token + [WORD_BOUNDARY]
    total_log = 0.0
    for (prev, nxt) in nltk.ngrams(token, 2):
        try:
            col = sound_idx.index(prev)
            row = sound_idx.index(nxt)
        except ValueError:
            return float('-inf')
        total_log += model_data[row, col]
        if np.isinf(total_log):
            return float('-inf')
    return total_log

def generic_pos_unigram_score(token, pos_uni_freqs, aggregation="sum"):
    """
    Generic scoring for positional unigram models.

    Parameters:
      token: List of symbols (the word).
      pos_uni_freqs: Dictionary mapping positions to dictionaries of symbol probabilities.
      aggregation: Aggregation method, "sum" to sum probabilities (default) or "prod" to multiply probabilities and take log.

    Returns:
      Total score (float) calculated based on the aggregation method.
    """
    if aggregation == "sum":
        total = 1.0
        for idx, sound in enumerate(token):
            total += pos_uni_freqs[idx].get(sound, 0.0)
        return total
    elif aggregation == "prod":
        prod = 1.0
        for idx, sound in enumerate(token):
            # Default to 1.0 if the probability isn’t found
            prod *= pos_uni_freqs[idx].get(sound, 1.0)
        return np.log(prod) if prod > 0 else float('-inf')
    else:
        raise ValueError("Invalid aggregation mode")

def generic_pos_bigram_score(token, pos_bi_freqs, conditional=False, use_word_boundaries=False, aggregation="sum"):
    """
    Generic scoring for positional bigram models.

    Parameters:
      token: List of symbols (the word).
      pos_bi_freqs: Dictionary mapping (position, position+1) tuples to dictionaries of bigram probabilities.
      conditional: If True, bigrams are conditionally normalized (not used in aggregation here).
      use_word_boundaries: Whether to add word boundary markers.
      aggregation: Aggregation method, "sum" to sum probabilities (default) or "prod" to multiply probabilities and take log.

    Returns:
      Total score (float) calculated based on the aggregation method.
    """
    if use_word_boundaries:
        token = [WORD_BOUNDARY] + token + [WORD_BOUNDARY]
    token_bigrams = list(nltk.ngrams(token, 2))
    if aggregation == "sum":
        total = 1.0
        for i, bigram in enumerate(token_bigrams):
            total += pos_bi_freqs[(i, i+1)].get(bigram, 0.0)
        return total
    elif aggregation == "prod":
        prod = 1.0
        for i, bigram in enumerate(token_bigrams):
            prod *= pos_bi_freqs[(i, i+1)].get(bigram, 1.0)
        return np.log(prod) if prod > 0 else float('-inf')
    else:
        raise ValueError("Invalid aggregation mode")