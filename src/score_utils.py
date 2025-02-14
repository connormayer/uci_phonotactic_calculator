"""
score_utils.py - Utility module for scoring tokens using n-gram models with configurable aggregation.
This module provides functions to compute scores for unigram and bigram models, including positional models,
with two aggregation methods: "sum" (linear additive) and "prod" (multiplicative log-product).
Version: 1.1.1
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

    This function computes a score for a token using positional unigram probabilities.
    Two aggregation methods are supported:
      - "sum": Computes the score as a baseline (1.0) plus the sum of the linear probabilities 
               for each symbol at its corresponding position. This yields a score on the linear scale.
      - "prod": Computes the score as the product of the linear probabilities for each symbol 
                (defaulting to 1.0 if a probability is missing), then returns the logarithm of the product.
                This yields a log-probability score.
    
    Parameters:
      token: List of symbols (the word).
      pos_uni_freqs: Dictionary mapping position indices to dictionaries of symbol probabilities (in linear space).
      aggregation: Aggregation method, either "sum" for additive scoring or "prod" for multiplicative log-product scoring.
    
    Returns:
      Total score (float) computed based on the chosen aggregation method.
    """
    if aggregation == "sum":
        total = 1.0  # Baseline score (linear)
        for idx, sound in enumerate(token):
            total += pos_uni_freqs.get(idx, {}).get(sound, 0.0)
        return total
    elif aggregation == "prod":
        prod = 1.0
        for idx, sound in enumerate(token):
            # Use default value of 1.0 if probability is not found to avoid zero product.
            prod *= pos_uni_freqs.get(idx, {}).get(sound, 1.0)
        return np.log(prod) if prod > 0 else float('-inf')
    else:
        raise ValueError("Invalid aggregation mode. Use 'sum' for additive scoring or 'prod' for multiplicative log-product scoring.")

def generic_pos_bigram_score(token, pos_bi_freqs, conditional=False, use_word_boundaries=False, aggregation="sum"):
    """
    Generic scoring for positional bigram models.

    This function computes a score for a token using positional bigram probabilities.
    Two aggregation methods are supported:
      - "sum": Computes the score as a baseline (1.0) plus the sum of the linear probabilities
               for each bigram at its corresponding position pair. This yields a score on the linear scale.
      - "prod": Computes the score as the product of the linear probabilities for each bigram 
                (defaulting to 1.0 if a probability is missing), then returns the logarithm of the product.
                This yields a log-probability score.
    
    The 'conditional' parameter indicates whether the bigram probabilities were normalized conditionally.
    Note: This parameter is informational and does not affect the aggregation computation.
    
    Parameters:
      token: List of symbols (the word).
      pos_bi_freqs: Dictionary mapping (position, position+1) tuples to dictionaries of bigram probabilities (in linear space).
      conditional: Boolean flag indicating if bigram probabilities were conditionally normalized.
                   This is for documentation purposes and does not affect aggregation.
      use_word_boundaries: Whether to add word boundary markers to the token before generating bigrams.
      aggregation: Aggregation method, either "sum" for additive scoring or "prod" for multiplicative log-product scoring.
    
    Returns:
      Total score (float) computed based on the chosen aggregation method.
    """
    if use_word_boundaries:
        token = [WORD_BOUNDARY] + token + [WORD_BOUNDARY]
    token_bigrams = list(nltk.ngrams(token, 2))
    if aggregation == "sum":
        total = 1.0  # Baseline score (linear)
        for i, bigram in enumerate(token_bigrams):
            total += pos_bi_freqs.get((i, i+1), {}).get(bigram, 0.0)
        return total
    elif aggregation == "prod":
        prod = 1.0
        for i, bigram in enumerate(token_bigrams):
            prod *= pos_bi_freqs.get((i, i+1), {}).get(bigram, 1.0)
        return np.log(prod) if prod > 0 else float('-inf')
    else:
        raise ValueError("Invalid aggregation mode. Use 'sum' for additive scoring or 'prod' for multiplicative log-product scoring.")