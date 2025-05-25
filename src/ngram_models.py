"""
src/ngram_models.py - Module for fitting n-gram models.
Provides functions for building unigram and bigram models with explicit conditions:
Positional vs. Non_Positional, Word Boundaries vs. No Boundaries,
Joint vs. Conditional, and Sum vs. Prod aggregation.
All non-positional bigram helpers obey MATRIX_LAYOUT.

"""

import nltk
import numpy as np
from collections import defaultdict

from score_utils import (
    generic_unigram_score,
    generic_bigram_score,
    generic_pos_unigram_score,
    generic_pos_bigram_score,
)

WORD_BOUNDARY = "#"
MAX_WORD_LEN = 100
MATRIX_LAYOUT = "row_prev_col_next"  # global convention


def _sort_sounds_boundary_last(sound_index):
    """
    Ensure that if '#' is present, it appears at the end of the list.
    This aligns with the test suite's expected row/column ordering.
    """
    if WORD_BOUNDARY in sound_index:
        without_boundary = [s for s in sound_index if s != WORD_BOUNDARY]
        return sorted(without_boundary) + [WORD_BOUNDARY]
    else:
        return sorted(sound_index)


def _fit_bigram_matrix_conditional(
    token_freqs,
    sound_index,
    token_weighted=False,
    smoothed=False,
    use_word_boundaries=True,
):
    """
    Internal helper: builds a bigram probability matrix of *conditional* probabilities,
    Row = previous sound, Column = next sound; each row sums to 1.
    """
    # Ensure '#' is last in indexing
    sound_index = _sort_sounds_boundary_last(sound_index)
    if not use_word_boundaries and WORD_BOUNDARY in sound_index:
        sound_index.remove(WORD_BOUNDARY)

    N = len(sound_index)
    count_matrix = (
        np.ones((N, N), dtype=float) if smoothed else np.zeros((N, N), dtype=float)
    )

    for token, freq in token_freqs:
        if token_weighted and freq > 0:
            val = np.log(freq)
        else:
            val = 1.0
        token_bigrams = generate_bigrams(token, use_word_boundaries)
        for prev, nxt in token_bigrams:
            if prev in sound_index and nxt in sound_index:
                row = sound_index.index(prev)  # prev sound on rows
                col = sound_index.index(nxt)  # next sound on cols
                count_matrix[row, col] += val

    row_sums = count_matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_matrix = np.divide(
            count_matrix,
            row_sums,
            out=np.zeros_like(count_matrix),
            where=(row_sums != 0),
        )
        bigram_probs = np.log(prob_matrix)

    return bigram_probs


def _fit_bigram_matrix_joint(
    token_freqs,
    sound_index,
    token_weighted=False,
    smoothed=False,
    use_word_boundaries=True,
):
    """
    Internal helper: builds a bigram probability matrix of *joint* probabilities,
    i.e., matrix sums to 1 overall in linear space (p(prev, next)).
    """
    # Ensure '#' is last in indexing
    sound_index = _sort_sounds_boundary_last(sound_index)
    if not use_word_boundaries and WORD_BOUNDARY in sound_index:
        sound_index.remove(WORD_BOUNDARY)

    N = len(sound_index)
    count_matrix = (
        np.ones((N, N), dtype=float) if smoothed else np.zeros((N, N), dtype=float)
    )

    for token, freq in token_freqs:
        if token_weighted and freq > 0:
            val = np.log(freq)
        else:
            val = 1.0
        token_bigrams = generate_bigrams(token, use_word_boundaries)
        for prev, nxt in token_bigrams:
            if prev in sound_index and nxt in sound_index:
                row = sound_index.index(prev)  # prev sound on rows
                col = sound_index.index(nxt)  # next sound on cols
                count_matrix[row, col] += val

    total_sum = count_matrix.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_matrix = np.divide(
            count_matrix,
            total_sum,
            out=np.zeros_like(count_matrix),
            where=(total_sum != 0),
        )
        bigram_probs = np.log(prob_matrix)

    return bigram_probs


def normalize_positional_counts(counts, conditional=False):
    """
    Normalize positional counts by total counts for each position if not conditional,
    or by the preceding sound if conditional=True, in linear space.
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
    Returns a list of bigrams for the given token. Optionally adds boundary symbols.
    """
    if use_word_boundaries:
        token = [WORD_BOUNDARY] + token + [WORD_BOUNDARY]
    return list(nltk.ngrams(token, 2))


# --------------------------
# Fitting Functions for Unigrams
# --------------------------


def fit_non_positional_unigram(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits non-positional unigram probabilities (returns a dict of {sound: log(prob)}).
    """
    sound_index = set(sound for token, _ in token_freqs for sound in token)

    if smoothed:
        unigram_freqs = {sound: 1.0 for sound in sound_index}
    else:
        unigram_freqs = {sound: 0.0 for sound in sound_index}

    for token, freq in token_freqs:
        val = np.log(freq) if (token_weighted and freq > 0) else 1.0
        for sound in token:
            unigram_freqs[sound] += val

    total_sounds = sum(unigram_freqs.values())
    if total_sounds <= 0:
        return {sound: float("-inf") for sound in unigram_freqs}

    unigram_probs = {}
    for sound, count_val in unigram_freqs.items():
        prob = count_val / total_sounds
        unigram_probs[sound] = np.log(prob) if prob > 0 else float("-inf")
    return unigram_probs


def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits positional unigram probabilities, storing them in linear space.
    """
    pos_unigram_freqs = defaultdict(lambda: defaultdict(float))

    if smoothed:
        sound_index = set(sound for token, _ in token_freqs for sound in token)
        for i in range(MAX_WORD_LEN):
            for sound in sound_index:
                pos_unigram_freqs[i][sound] = 1.0

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted and freq > 0 else 1.0
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += val

    pos_unigram_freqs = normalize_positional_counts(pos_unigram_freqs)
    return pos_unigram_freqs


# --------------------------
# Fitting Functions for Bigrams
# --------------------------


def fit_bigram_conditional_non_positional_wb(
    token_freqs, sound_index, token_weighted=False, smoothed=False
):
    """
    Fits a conditional bigram model (non-positional) with word boundaries.
    Returns a 2D numpy array of log probabilities: p(next|prev).
    """
    return _fit_bigram_matrix_conditional(
        token_freqs,
        sound_index,
        token_weighted=token_weighted,
        smoothed=smoothed,
        use_word_boundaries=True,
    )


def fit_bigram_conditional_non_positional_nwb(
    token_freqs, sound_index, token_weighted=False, smoothed=False
):
    """
    Fits a conditional bigram model (non-positional) without word boundaries.
    Returns a 2D numpy array of log probabilities: p(next|prev).
    """
    return _fit_bigram_matrix_conditional(
        token_freqs,
        sound_index,
        token_weighted=token_weighted,
        smoothed=smoothed,
        use_word_boundaries=False,
    )


def fit_bigram_joint_non_positional_wb(
    token_freqs, token_weighted=False, smoothed=False
):
    """
    Fits a joint bigram model (non-positional) with word boundaries.
    Returns a 2D numpy array of log probabilities: p(prev, next).
    """
    sound_index = {sound for token, _ in token_freqs for sound in token}
    sound_index.add(WORD_BOUNDARY)
    sound_index = list(sound_index)
    return _fit_bigram_matrix_joint(
        token_freqs,
        sound_index,
        token_weighted=token_weighted,
        smoothed=smoothed,
        use_word_boundaries=True,
    )


def fit_bigram_joint_non_positional_nwb(
    token_freqs, token_weighted=False, smoothed=False
):
    """
    Fits a joint bigram model (non-positional) without word boundaries.
    Returns a 2D numpy array of log probabilities: p(prev, next).
    """
    sound_index = {sound for token, _ in token_freqs for sound in token}
    sound_index.discard(WORD_BOUNDARY)
    sound_index = list(sound_index)
    return _fit_bigram_matrix_joint(
        token_freqs,
        sound_index,
        token_weighted=token_weighted,
        smoothed=smoothed,
        use_word_boundaries=False,
    )


def fit_bigram_positional(
    token_freqs,
    token_weighted=False,
    smoothed=False,
    conditional=False,
    use_word_boundaries=False,
):
    """
    Fits positional bigram probabilities.

    Parameters:
      token_freqs: List of token frequencies.
      token_weighted: Boolean, whether to use token weighting.
      smoothed: Boolean, whether to apply smoothing.
      conditional: Boolean; if True, compute conditional probabilities; otherwise, joint probabilities.
      use_word_boundaries: Boolean, whether to add word boundary markers.

    Returns:
      A dictionary mapping position tuples to dictionaries of bigram probabilities (in linear space)
      normalized appropriately and then transformed to log probabilities.
    """
    pos_bigram_freqs = defaultdict(lambda: defaultdict(float))

    sound_index = {sound for token, _ in token_freqs for sound in token}
    if use_word_boundaries:
        sound_index.add(WORD_BOUNDARY)

    if smoothed:
        max_len = MAX_WORD_LEN + 2 if use_word_boundaries else MAX_WORD_LEN
        for i in range(max_len - 1):
            for s1 in sound_index:
                for s2 in sound_index:
                    pos_bigram_freqs[(i, i + 1)][(s1, s2)] = 1.0

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted and freq > 0 else 1.0
        token_bigrams = generate_bigrams(token, use_word_boundaries)
        for idx, bigram in enumerate(token_bigrams):
            pos_bigram_freqs[(idx, idx + 1)][bigram] += val

    pos_bigram_freqs = normalize_positional_counts(
        pos_bigram_freqs, conditional=conditional
    )
    return pos_bigram_freqs


# --------------------------
# Unified Model Interface
# --------------------------


class NgramModel:
    """
    Base class for n-gram models.
    This class should not be instantiated directly.
    """

    def __init__(self, position, prob_type, smoothed=False, token_weighted=False):
        self.position = position
        self.prob_type = prob_type
        self.smoothed = smoothed
        self.token_weighted = token_weighted
        self.model_data = None

    def fit(self, token_freqs, sound_index):
        raise NotImplementedError("Subclasses must implement fit method.")

    def score(self, word, sound_index):
        raise NotImplementedError("Subclasses must implement score method.")


class UnigramModel(NgramModel):
    """
    Unigram model subclass.
    Implements .fit() and .score() for unigram models.
    """

    def __init__(
        self,
        position,
        prob_type=None,
        smoothed=False,
        token_weighted=False,
        aggregation="prod",
    ):
        super().__init__(position, prob_type, smoothed, token_weighted)
        self.aggregation = aggregation

    def fit(self, token_freqs, sound_index=None):
        if self.position == "non_positional":
            self.model_data = fit_non_positional_unigram(
                token_freqs, self.token_weighted, self.smoothed
            )
        elif self.position == "positional":
            self.model_data = fit_positional_unigrams(
                token_freqs, self.token_weighted, self.smoothed
            )
        return self

    def score(self, word, sound_index=None):
        if self.position == "non_positional":
            return generic_unigram_score(
                word, self.model_data, aggregation=self.aggregation
            )
        elif self.position == "positional":
            return generic_pos_unigram_score(
                word, self.model_data, aggregation=self.aggregation
            )
        return None


class BigramModel(NgramModel):
    """
    Bigram model subclass.
    Implements .fit() and .score() for bigram models.
    """

    def __init__(
        self,
        position,
        prob_type="conditional",
        use_boundaries=True,
        smoothed=False,
        token_weighted=False,
        aggregation="prod",
    ):
        super().__init__(position, prob_type, smoothed, token_weighted)
        self.use_boundaries = use_boundaries
        self.aggregation = aggregation

    def fit(self, token_freqs, sound_index):
        if self.position == "non_positional":
            if self.prob_type == "conditional":
                if self.use_boundaries:
                    self.model_data = fit_bigram_conditional_non_positional_wb(
                        token_freqs, sound_index, self.token_weighted, self.smoothed
                    )
                else:
                    self.model_data = fit_bigram_conditional_non_positional_nwb(
                        token_freqs, sound_index, self.token_weighted, self.smoothed
                    )
            else:  # Joint model
                if self.use_boundaries:
                    self.model_data = fit_bigram_joint_non_positional_wb(
                        token_freqs, self.token_weighted, self.smoothed
                    )
                else:
                    self.model_data = fit_bigram_joint_non_positional_nwb(
                        token_freqs, self.token_weighted, self.smoothed
                    )
        else:
            self.model_data = fit_bigram_positional(
                token_freqs,
                self.token_weighted,
                self.smoothed,
                conditional=(self.prob_type == "conditional"),
                use_word_boundaries=self.use_boundaries,
            )
        return self

    def score(self, word, sound_index):
        if self.position == "non_positional":
            # rows = prev, cols = next – see MATRIX_LAYOUT
            return generic_bigram_score(
                word,
                self.model_data,
                sound_index,
                use_word_boundaries=self.use_boundaries,
                aggregation=self.aggregation,
            )
        else:
            return generic_pos_bigram_score(
                word,
                self.model_data,
                conditional=(self.prob_type == "conditional"),
                use_word_boundaries=self.use_boundaries,
                aggregation=self.aggregation,
            )
