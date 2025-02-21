"""
ngram_models.py - Module for fitting n-gram models and providing an object-oriented model interface.
This module provides functions for fitting unigram and bigram models, both positional and non-positional,
and includes a unified model interface.
Version: 1.2.5
"""

import nltk
import numpy as np
from collections import defaultdict

from .score_utils import (
    generic_unigram_score,
    generic_bigram_score,
    generic_pos_unigram_score,
    generic_pos_bigram_score
)

# Global constants
WORD_BOUNDARY = '#'
MAX_WORD_LEN = 100

def _fit_bigram_matrix(token_freqs, sound_list, token_weighted=False, smoothed=False,
                       use_word_boundaries=True, log_boundaries_weight=False):
    """
    Builds a bigram probability matrix with column-based (conditional) normalization:
    
    Steps:
      1) If smoothed=True, initialize counts to ones; else zeros.
      2) For each token, add log(freq) if token_weighted=True and freq>0, else 1.
         Bigram indices are determined by [row=next sound, col=previous sound].
      3) Column-normalize in linear space to get conditional probabilities.
      4) Take the log of each probability, returning a matrix of log-probs.
    
    Args:
      token_freqs: List of (token, freq) pairs.
      sound_list: All sounds (potential columns/rows).   
      token_weighted: If True, increments = log(freq), else increments = 1.
      smoothed: If True, start each count cell at 1 instead of 0.
      use_word_boundaries: Whether to prepend/append # around the token.
      log_boundaries_weight: (Legacy) If True, also apply log to boundary freq.
    
    Returns:
      A 2D numpy array of shape (N, N), where each column sums to 1 in linear space,
      and each cell is the log of that probability.
    """
    local_sounds = list(sound_list)
    if not use_word_boundaries and WORD_BOUNDARY in local_sounds:
        local_sounds.remove(WORD_BOUNDARY)

    N = len(local_sounds)
    count_matrix = np.ones((N, N), dtype=float) if smoothed else np.zeros((N, N), dtype=float)

    for token, freq in token_freqs:
        if token_weighted and freq > 0:
            val = np.log(freq)
        else:
            val = 1.0
        token_bigrams = generate_bigrams(token, use_word_boundaries)
        for (prev, nxt) in token_bigrams:
            if prev in local_sounds and nxt in local_sounds:
                col = local_sounds.index(prev)
                row = local_sounds.index(nxt)
                count_matrix[row, col] += val

    col_sums = count_matrix.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_matrix = np.divide(count_matrix, col_sums,
                                out=np.zeros_like(count_matrix),
                                where=(col_sums != 0))
    with np.errstate(divide='ignore', invalid='ignore'):
        bigram_probs = np.log(prob_matrix)
    return bigram_probs

def normalize_positional_counts(counts, conditional=False):
    """
    Normalizes positional counts by total counts for each position if not conditional,
    or by the preceding sound if conditional=True, in linear space.
    
    The final dictionary entries become linear probabilities:
      counts[idx][(sound)] or counts[(idx, idx+1)][(sound1, sound2)]
    
    No additional log transform here. The scoring functions 1) read these linear
    probabilities and 2) do '1 + sum_of(probabilities)' if relevant.
    """
    if not conditional:
        for idx in counts.keys():
            total = sum(counts[idx].values())
            if total > 0:
                for gram in counts[idx].keys():
                    counts[idx][gram] /= total
            else:
                for gram in counts[idx].keys():
                    counts[idx][gram] = 0
    else:
        from collections import defaultdict
        for idx in counts.keys():
            preceding_sound_dict = defaultdict(float)
            for gram, count_val in counts[idx].items():
                preceding_sound_dict[gram[0]] += count_val
            for gram in counts[idx].keys():
                total = preceding_sound_dict[gram[0]]
                counts[idx][gram] = (counts[idx][gram] / total) if total > 0 else 0
    return counts

def generate_bigrams(token, use_word_boundaries=True):
    """
    Returns a list of bigrams for the given token.
    Adds boundaries if specified.
    """
    if use_word_boundaries:
        token = [WORD_BOUNDARY] + token + [WORD_BOUNDARY]
    return list(nltk.ngrams(token, 2))

# --------------------------
# Fitting Functions
# --------------------------

def fit_non_positional_unigram(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits non-positional unigram probabilities (returns a dict of {sound: log(prob)}).
    
    Steps:
      1. Gather all unique sounds.
      2. If smoothed=True, initialize each sound's count to 1, else 0.
      3. For each token, add either log(freq) or 1 to each sound's count.
      4. Normalize by total to get probabilities in linear space.
      5. Convert each probability to log(prob). Return as { sound: log(prob) }.
    """
    unique_sounds = set(sound for token, _ in token_freqs for sound in token)

    if smoothed:
        unigram_freqs = {sound: 1.0 for sound in unique_sounds}
    else:
        unigram_freqs = {sound: 0.0 for sound in unique_sounds}

    for token, freq in token_freqs:
        val = np.log(freq) if (token_weighted and freq > 0) else 1.0
        for sound in token:
            unigram_freqs[sound] += val

    total_sounds = sum(unigram_freqs.values())
    if total_sounds <= 0:
        return {sound: float('-inf') for sound in unigram_freqs}

    unigram_probs = {}
    for sound, count_val in unigram_freqs.items():
        prob = count_val / total_sounds
        unigram_probs[sound] = np.log(prob) if prob > 0 else float('-inf')
    return unigram_probs


def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits positional unigram probabilities, storing them in linear space.
    
    Steps:
      1) Possibly initialize every position-sound to 1 if smoothed=True.
      2) For each token, add log(freq) if token_weighted and freq>0, else 1.
      3) Call normalize_positional_counts(...) to convert each position's
         counts into linear probabilities that sum to 1.
      4) Return that nested dictionary (pos -> {sound: prob_in_linear_space}).
    
    Scoring is typically 1 + sum_of(probabilities) in linear space.
    """
    pos_unigram_freqs = defaultdict(lambda: defaultdict(float))
    if smoothed:
        unique_sounds = set(sound for token, _ in token_freqs for sound in token)
        for i in range(MAX_WORD_LEN):
            for sound in unique_sounds:
                pos_unigram_freqs[i][sound] = 1.0
    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted and freq > 0 else 1.0
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += val
    pos_unigram_freqs = normalize_positional_counts(pos_unigram_freqs)
    return pos_unigram_freqs

def fit_bigrams(token_freqs, sound_idx, token_weighted=False, smoothed=False, use_word_boundaries=True):
    """
    Fits non-positional bigram conditional probabilities.
    Returns a 2D numpy array of log probabilities.
    """
    log_boundaries = use_word_boundaries
    return _fit_bigram_matrix(token_freqs, sound_idx, token_weighted, smoothed,
                              use_word_boundaries, log_boundaries)

def fit_positional_bigrams(token_freqs, token_weighted=False, smoothed=False,
                           conditional=False, use_word_boundaries=False):
    """
    Fits positional bigram probabilities in linear space.
    
    Steps:
      1) Possibly initialize every position pair and bigram to 1 if smoothed=True.
      2) For each token, add log(freq) if token_weighted and freq>0, else 1, to the
         bigram entry at (pos, pos+1).
      3) Call normalize_positional_counts(...) with 'conditional' if needed. This
         converts raw counts to linear probabilities that sum to 1 either by row
         or for each preceding sound.
      4) Return that nested dict: pos_bigram_freqs[(i, i+1)][(s1, s2)] = linear_prob.
    
    The final probabilities remain in linear space. Scoring is 1 + sum_of(probabilities).
    """
    pos_bigram_freqs = defaultdict(lambda: defaultdict(float))
    unique_sounds = set(sound for token, _ in token_freqs for sound in token)
    if use_word_boundaries:
        unique_sounds.add(WORD_BOUNDARY)
    if smoothed:
        max_len = MAX_WORD_LEN + 2 if use_word_boundaries else MAX_WORD_LEN
        for i in range(max_len - 1):
            for s1 in unique_sounds:
                for s2 in unique_sounds:
                    pos_bigram_freqs[(i, i+1)][(s1, s2)] = 1.0
    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted and freq > 0 else 1.0
        token_bigrams = generate_bigrams(token, use_word_boundaries)
        for idx, bigram in enumerate(token_bigrams):
            pos_bigram_freqs[(idx, idx+1)][bigram] += val
    pos_bigram_freqs = normalize_positional_counts(pos_bigram_freqs, conditional=conditional)
    return pos_bigram_freqs

def fit_non_positional_bigrams(token_freqs, token_weighted=False, smoothed=False, use_word_boundaries=True):
    """
    Fits non-positional bigram joint probabilities.
    Returns a 2D numpy array of log probabilities.
    """
    unique_sounds = {sound for token, _ in token_freqs for sound in token}
    if use_word_boundaries:
        unique_sounds.add(WORD_BOUNDARY)
    else:
        unique_sounds.discard(WORD_BOUNDARY)
    sound_list = sorted(unique_sounds)
    return _fit_bigram_matrix(token_freqs, sound_list, token_weighted, smoothed,
                              use_word_boundaries, log_boundaries_weight=False)

# --------------------------
# Unified Model Interface: Base Class and Subclasses
# --------------------------

class NgramModel:
    """
    Base class for n-gram models.
    This class should not be instantiated directly.
    """
    def __init__(self, position, prob_type, smoothed=False, token_weighted=False):
        self.position = position                # "non_positional" or "positional"
        self.prob_type = prob_type              # For unigrams: "log" or "joint"; for bigrams: "conditional" or "joint"
        self.smoothed = smoothed
        self.token_weighted = token_weighted
        self.model_data = None

    def fit(self, token_freqs, sound_idx):
        raise NotImplementedError("Subclasses must implement fit method.")

    def score(self, word, sound_idx):
        raise NotImplementedError("Subclasses must implement score method.")

class UnigramModel(NgramModel):
    """
    Unigram model subclass.
    Implements .fit() and .score() for unigram models.
    """
    def __init__(self, position, prob_type, smoothed=False, token_weighted=False, aggregation="sum"):
        super().__init__(position, prob_type, smoothed, token_weighted)
        self.aggregation = aggregation

    def fit(self, token_freqs, sound_idx):
        if self.position == "non_positional":
            # Unified fitting: non-positional unigrams always use log probabilities.
            self.model_data = fit_non_positional_unigram(token_freqs, self.token_weighted, self.smoothed)
        elif self.position == "positional":
            self.model_data = fit_positional_unigrams(token_freqs, self.token_weighted, self.smoothed)
        return self

    def score(self, word, sound_idx):
        if self.position == "non_positional":
            # Unified scoring: non-positional unigrams are scored in log mode.
            return generic_unigram_score(word, self.model_data, mode='log')
        elif self.position == "positional":
            return generic_pos_unigram_score(word, self.model_data, aggregation=self.aggregation)
        return None

class BigramModel(NgramModel):
    """
    Bigram model subclass.
    Implements .fit() and .score() for bigram models.
    """
    def __init__(self, position, prob_type, use_boundaries=False, smoothed=False, token_weighted=False, aggregation="sum"):
        super().__init__(position, prob_type, smoothed, token_weighted)
        self.use_boundaries = use_boundaries
        self.aggregation = aggregation

    def fit(self, token_freqs, sound_idx):
        if self.position == "non_positional":
            if self.prob_type == "conditional":
                self.model_data = fit_bigrams(token_freqs, sound_idx,
                                              self.token_weighted,
                                              self.smoothed,
                                              self.use_boundaries)
            elif self.prob_type == "joint":
                self.model_data = fit_non_positional_bigrams(token_freqs,
                                                             self.token_weighted,
                                                             self.smoothed,
                                                             self.use_boundaries)
        elif self.position == "positional":
            self.model_data = fit_positional_bigrams(token_freqs,
                                                     self.token_weighted,
                                                     self.smoothed,
                                                     conditional=(self.prob_type=="conditional"),
                                                     use_word_boundaries=self.use_boundaries)
        return self

    def score(self, word, sound_idx):
        if self.position == "non_positional":
            return generic_bigram_score(word, self.model_data, sound_idx,
                                        use_word_boundaries=self.use_boundaries)
        elif self.position == "positional":
            return generic_pos_bigram_score(word, self.model_data, conditional=(self.prob_type=="conditional"),
                                            use_word_boundaries=self.use_boundaries, aggregation=self.aggregation)
        return None