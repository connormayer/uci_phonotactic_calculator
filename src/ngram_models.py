"""
ngram_models.py - Module for fitting n-gram models and providing a unified model interface.
Version: 1.0.0
"""

import nltk
import numpy as np
from collections import defaultdict

# Global constants
WORD_BOUNDARY = '#'
MAX_WORD_LEN = 100

def _fit_bigram_matrix(token_freqs, sound_list, token_weighted=False, smoothed=False,
                       use_word_boundaries=True, log_boundaries_weight=False):
    """
    Builds a bigram probability matrix with column-based normalization.

    - If smoothed=True, initialize counts to 1; else 0.
    - If token_weighted=True:
       * If log_boundaries_weight=True, value = log(freq) if freq>0 else 0
       * Else, value = freq.
      If token_weighted=False: value = 1 for each bigram occurrence.
    - use_word_boundaries: whether to include the boundary marker.
    - After counting, each column is normalized to sum to 1, then we return log-probs.
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
    Normalizes positional counts by total counts for each position
    if not conditional, or by the preceding sound if conditional=True.
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
    Returns a list of bigrams for the given token. Adds boundaries if specified.
    """
    if use_word_boundaries:
        token = [WORD_BOUNDARY] + token + [WORD_BOUNDARY]
    return list(nltk.ngrams(token, 2))

# --------------------------
# Fitting Functions
# --------------------------

def fit_non_positional_unigram_probabilities(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits non-positional unigram probabilities (returns a dict of {sound: log(prob)}).
    Used for log-probability unigram models.
    """
    default_func = lambda: int(smoothed)
    unigram_freqs = defaultdict(default_func)
    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted and freq > 0 else 1
        for sound in token:
            unigram_freqs[sound] += val
    total_sounds = sum(unigram_freqs.values())
    if total_sounds <= 0:
        return {k: float('-inf') for k in unigram_freqs}
    unigram_probs = {}
    for key, value in unigram_freqs.items():
        prob = value / total_sounds
        unigram_probs[key] = np.log(prob) if prob > 0 else float('-inf')
    return unigram_probs

def fit_non_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits non-positional unigram joint probabilities, returning {sound: log(prob)}.
    Used for joint unigram models.
    """
    unigram_freqs = defaultdict(float)
    if smoothed:
        unique_sounds = set(sound for token, _ in token_freqs for sound in token)
        for s in unique_sounds:
            unigram_freqs[s] = 1.0
    for token, freq in token_freqs:
        val = freq if token_weighted else 1.0
        for sound in token:
            unigram_freqs[sound] += val
    total = sum(unigram_freqs.values())
    if total > 0:
        for sound in unigram_freqs:
            p = unigram_freqs[sound] / total
            unigram_freqs[sound] = np.log(p) if p > 0 else float('-inf')
    else:
        for sound in unigram_freqs:
            unigram_freqs[sound] = float('-inf')
    return unigram_freqs

def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits positional unigram joint probabilities.
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
    Returns a 2D numpy array of log-probabilities.
    """
    log_boundaries = use_word_boundaries
    return _fit_bigram_matrix(token_freqs, sound_idx, token_weighted, smoothed, use_word_boundaries, log_boundaries)

def fit_positional_bigrams(token_freqs, token_weighted=False, smoothed=False, conditional=False, use_word_boundaries=False):
    """
    Fits positional bigram probabilities in linear space.
    If conditional=True, normalizes by the preceding sound; else by total counts.
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
    Returns a 2D numpy array of log-probabilities.
    """
    unique_sounds = {sound for token, _ in token_freqs for sound in token}
    if use_word_boundaries:
        unique_sounds.add(WORD_BOUNDARY)
    else:
        unique_sounds.discard(WORD_BOUNDARY)
    sound_list = sorted(unique_sounds)
    return _fit_bigram_matrix(token_freqs, sound_list, token_weighted, smoothed, use_word_boundaries, log_boundaries_weight=False)

# --------------------------
# Scoring Functions
# --------------------------

def get_unigram_prob(word, unigram_probs):
    """
    Calculates the total log probability of a word under a unigram model.
    """
    prob = 0.0
    for sound in word:
        sound_prob = unigram_probs.get(sound, float('-inf'))
        prob += sound_prob
        if np.isinf(prob):
            return float('-inf')
    return prob

def get_non_pos_unigram_score(word, unigram_probs):
    """
    Calculates a joint score for a word under a non-positional unigram model.
    """
    score = 1.0
    for sound in word:
        log_p = unigram_probs.get(sound, float('-inf'))
        if np.isinf(log_p):
            continue
        score += np.exp(log_p)
    return score

def get_pos_unigram_score(word, pos_uni_freqs):
    """
    Calculates the positional unigram score for a word.
    """
    score = 1.0
    for idx, sound in enumerate(word):
        score += pos_uni_freqs[idx].get(sound, 0.0)
    return score

def get_bigram_prob(word, bigram_probs, sound_idx, use_word_boundaries=True):
    """
    Calculates total log probability of a word using a non-positional bigram model.
    """
    token_bigrams = generate_bigrams(word, use_word_boundaries)
    prob = 0.0
    for (prev, nxt) in token_bigrams:
        try:
            col = sound_idx.index(prev)
            row = sound_idx.index(nxt)
        except ValueError:
            return float('-inf')
        prob += bigram_probs[row, col]
        if np.isinf(prob):
            return float('-inf')
    return prob

def get_non_pos_bigram_score(word, bigram_probs, sound_idx, use_word_boundaries=False):
    """
    Sums log-probabilities from a non-positional bigram joint model.
    """
    if use_word_boundaries:
        word = [WORD_BOUNDARY] + word + [WORD_BOUNDARY]
    total_log = 0.0
    for (prev, nxt) in nltk.ngrams(word, 2):
        try:
            col = sound_idx.index(prev)
            row = sound_idx.index(nxt)
        except ValueError:
            return float('-inf')
        total_log += bigram_probs[row, col]
        if np.isinf(total_log):
            return float('-inf')
    return total_log

def get_pos_bigram_score(word, pos_bi_freqs, conditional=False, use_word_boundaries=False):
    """
    Calculates the positional bigram score for a word.
    """
    if use_word_boundaries:
        word = [WORD_BOUNDARY] + word + [WORD_BOUNDARY]
    score = 1.0
    token_bigrams = list(nltk.ngrams(word, 2))
    for i, bigram in enumerate(token_bigrams):
        score += pos_bi_freqs[(i, i+1)].get(bigram, 0.0)
    return score

# --------------------------
# Model Factory / Unified Interface
# --------------------------

class NgramModel:
    """
    A unified n-gram model that can be configured for unigram or bigram,
    positional or non-positional, and for joint versus conditional/log-probability scoring.
    """
    def __init__(self, model_type, position, prob_type,
                 use_boundaries=False, smoothed=False, token_weighted=False):
        self.model_type = model_type            # "unigram" or "bigram"
        self.position = position                # "non_positional" or "positional"
        self.prob_type = prob_type              # For unigrams: "log" or "joint"; for bigrams: "conditional" or "joint"
        self.use_boundaries = use_boundaries
        self.smoothed = smoothed
        self.token_weighted = token_weighted
        self.model_data = None

    def fit(self, token_freqs, sound_idx):
        if self.model_type == "unigram":
            if self.position == "non_positional":
                if self.prob_type == "log":
                    self.model_data = fit_non_positional_unigram_probabilities(token_freqs,
                                                                               self.token_weighted,
                                                                               self.smoothed)
                elif self.prob_type == "joint":
                    self.model_data = fit_non_positional_unigrams(token_freqs,
                                                                  self.token_weighted,
                                                                  self.smoothed)
            elif self.position == "positional":
                self.model_data = fit_positional_unigrams(token_freqs,
                                                          self.token_weighted,
                                                          self.smoothed)
        elif self.model_type == "bigram":
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
                if self.prob_type == "conditional":
                    self.model_data = fit_positional_bigrams(token_freqs,
                                                             self.token_weighted,
                                                             self.smoothed,
                                                             conditional=True,
                                                             use_word_boundaries=self.use_boundaries)
                elif self.prob_type == "joint":
                    self.model_data = fit_positional_bigrams(token_freqs,
                                                             self.token_weighted,
                                                             self.smoothed,
                                                             conditional=False,
                                                             use_word_boundaries=self.use_boundaries)
        return self

    def score(self, word, sound_idx):
        if self.model_type == "unigram":
            if self.position == "non_positional":
                if self.prob_type == "log":
                    return get_unigram_prob(word, self.model_data)
                elif self.prob_type == "joint":
                    return get_non_pos_unigram_score(word, self.model_data)
            elif self.position == "positional":
                return get_pos_unigram_score(word, self.model_data)
        elif self.model_type == "bigram":
            if self.position == "non_positional":
                if self.prob_type == "conditional":
                    return get_bigram_prob(word, self.model_data, sound_idx,
                                           use_word_boundaries=self.use_boundaries)
                elif self.prob_type == "joint":
                    return get_non_pos_bigram_score(word, self.model_data, sound_idx,
                                                    use_word_boundaries=self.use_boundaries)
            elif self.position == "positional":
                if self.prob_type == "conditional":
                    return get_pos_bigram_score(word, self.model_data, conditional=True,
                                                use_word_boundaries=self.use_boundaries)
                elif self.prob_type == "joint":
                    return get_pos_bigram_score(word, self.model_data, conditional=False,
                                                use_word_boundaries=self.use_boundaries)
        return None