# FileName: ngram_calculator.py 
# version 1.0
# Summary: [Contains functions to fit and calculate n-gram probabilities (both positional & non-positional)]
# Tags: [ngram, bigram, phonotactics, csv, log-probabilities]

# Run command, from parent directory:
# "python src/ngram_calculator.py data/english.csv data/sample_test_data/english_test_data.csv src/output.csv"

import csv
import nltk
import numpy as np

from collections import defaultdict

WORD_BOUNDARY = '#'
MAX_WORD_LEN = 100

HEADER = [
    'word',
    'word_len',

    'uni_prob',
    'uni_prob_freq_weighted',
    'uni_prob_smoothed',
    'uni_prob_freq_weighted_smoothed',

    'bi_cond_pos_wb',
    'bi_cond_pos_wb_freq_weighted',
    'bi_cond_pos_wb_smoothed',
    'bi_cond_pos_wb_freq_weighted_smoothed',

    'bi_cond_pos_noWB',
    'bi_cond_pos_noWB_freq_weighted',
    'bi_cond_pos_noWB_smoothed',
    'bi_cond_pos_noWB_freq_weighted_smoothed',

    'bi_cond_nonpos_wb',
    'bi_cond_nonpos_wb_freq_weighted',
    'bi_cond_nonpos_wb_smoothed',
    'bi_cond_nonpos_wb_freq_weighted_smoothed',

    'bi_cond_nonpos_noWB',
    'bi_cond_nonpos_noWB_freq_weighted',
    'bi_cond_nonpos_noWB_smoothed',
    'bi_cond_nonpos_noWB_freq_weighted_smoothed',

    'uni_joint_pos',
    'uni_joint_pos_freq_weighted',
    'uni_joint_pos_smoothed',
    'uni_joint_pos_freq_weighted_smoothed',

    'uni_joint_nonpos',
    'uni_joint_nonpos_freq_weighted',
    'uni_joint_nonpos_smoothed',
    'uni_joint_nonpos_freq_weighted_smoothed',

    'bi_joint_pos_wb',
    'bi_joint_pos_wb_freq_weighted',
    'bi_joint_pos_wb_smoothed',
    'bi_joint_pos_wb_freq_weighted_smoothed',

    'bi_joint_pos_noWB',
    'bi_joint_pos_noWB_freq_weighted',
    'bi_joint_pos_noWB_smoothed',
    'bi_joint_pos_noWB_freq_weighted_smoothed',

    'bi_joint_nonpos_wb',
    'bi_joint_nonpos_wb_freq_weighted',
    'bi_joint_nonpos_wb_smoothed',
    'bi_joint_nonpos_wb_freq_weighted_smoothed',

    'bi_joint_nonpos_noWB',
    'bi_joint_nonpos_noWB_freq_weighted',
    'bi_joint_nonpos_noWB_smoothed',
    'bi_joint_nonpos_noWB_freq_weighted_smoothed',
]


####################
# Helper functions #
####################

def generate_bigrams(token, use_word_boundaries=True):
    """
    Returns a list of sound bigrams given a single word token.

    token: The list of symbols in the token.
    use_word_boundaries: Whether to include word boundary markers.

    returns: The list of bigrams of the token.
    """
    if use_word_boundaries:
        token = [WORD_BOUNDARY] + token + [WORD_BOUNDARY]
    return list(nltk.ngrams(token, 2))

def read_tokens(dataset):
    """
    Reads in a file containing tokens and optional frequencies and converts
    it to a list of tokens and a list of token/frequency pairs.

    dataset: The path to the dataset.

    returns: A list of lists, where each sublist corresponds to a token and
    consists of a list of the individual symbols, plus a frequency if provided.
    """
    import csv
    with open(dataset, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        token_freqs = []
        for row in reader:
            split_token = row[0].split(' ')
            freq = float(row[1]) if len(row) == 2 else 0  # Default frequency is 0
            token_freqs.append([split_token, freq])
    return token_freqs

def write_results(results, outfile):
    """
    Writes the results of scoring the test dataset to a file.

    results: The results to write.
    outfile: The path to the output file.

    returns: None
    """
    results = [HEADER] + results
    with open(outfile, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)

###########################
# Code for fitting models #
###########################

def fit_ngram_models(token_freqs, sound_idx):
    # Get unigram probabilities
    uni_models = []
    uni_models.append(fit_unigrams(token_freqs))  # uni_prob
    uni_models.append(fit_unigrams(token_freqs, token_weighted=True))  # uni_prob_freq_weighted
    uni_models.append(fit_unigrams(token_freqs, smoothed=True))  # uni_prob_smoothed
    uni_models.append(fit_unigrams(token_freqs, smoothed=True, token_weighted=True))  # uni_prob_freq_weighted_smoothed

    # Get bigram conditional, positional, with word boundaries
    bi_cond_pos_wb_models = []
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, conditional=True, use_word_boundaries=True))
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, conditional=True, use_word_boundaries=True))
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, conditional=True, use_word_boundaries=True))
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, conditional=True, use_word_boundaries=True))
    
    # Get bigram conditional, positional, without word boundaries
    bi_cond_pos_noWB_models = []
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, conditional=True, use_word_boundaries=False))
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, conditional=True, use_word_boundaries=False))
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, conditional=True, use_word_boundaries=False))
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, conditional=True, use_word_boundaries=False))

    # Get bigram conditional non-positional models with word boundaries
    bi_cond_nonpos_wb_models = []
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx))
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx, token_weighted=True))
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True))
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True, token_weighted=True))

    # Get bigram conditional non-positional models without word boundaries
    bi_cond_nonpos_noWB_models = []
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, use_word_boundaries=False))
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, token_weighted=True, use_word_boundaries=False))
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True, use_word_boundaries=False))
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True, token_weighted=True, use_word_boundaries=False))
    
    # Get positional unigram joint probabilities
    uni_joint_pos_models = []
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs))
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs, token_weighted=True))
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs, smoothed=True))
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs, smoothed=True, token_weighted=True))

    # Non-Positional Unigram Joint Probabilities
    uni_joint_nonpos_models = []
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs))
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs, token_weighted=True))
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs, smoothed=True))
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs, smoothed=True, token_weighted=True))

    # Bigram Joint Positional Models with Word Boundaries
    bi_joint_pos_wb_models = []
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, use_word_boundaries=True))
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=True))
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=True))
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=True))

    # Bigram Joint Positional Models without Word Boundaries
    bi_joint_pos_noWB_models = []
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, use_word_boundaries=False))
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=False))
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=False))
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=False))

    # Bigram Joint Non-Positional Models with Word Boundaries
    bi_joint_nonpos_wb_models = []
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, use_word_boundaries=True))
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=True))
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=True))
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=True))

    # Bigram Joint Non-Positional Models without Word Boundaries
    bi_joint_nonpos_noWB_models = []
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, use_word_boundaries=False))
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=False))
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=False))
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=False))

    return (
        uni_models,
        bi_cond_pos_wb_models,
        bi_cond_pos_noWB_models,
        bi_cond_nonpos_wb_models,
        bi_cond_nonpos_noWB_models,
        uni_joint_pos_models,
        uni_joint_nonpos_models,
        bi_joint_pos_wb_models,
        bi_joint_pos_noWB_models,
        bi_joint_nonpos_wb_models,
        bi_joint_nonpos_noWB_models,
    )


def fit_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits unigram probabilities (returns a dict of {sound: log(prob)}).

    For weighting, we add np.log(freq) if freq>0. 
    For smoothing, we initialize each sound's count with 1 in linear space
    but effectively store that as an integer offset for the "counts."
    """
    default_func = lambda: int(smoothed)
    unigram_freqs = defaultdict(default_func)

    for token, freq in token_freqs:
        if token_weighted:
            # Add the log of freq in the "count" (tests require this approach)
            val = np.log(freq) if freq > 0 else 0
        else:
            val = 1
        for sound in token:
            unigram_freqs[sound] += val

    total_sounds = sum(unigram_freqs.values())
    if total_sounds <= 0:
        # If no sounds were counted, return -inf for everything
        return {k: float('-inf') for k in unigram_freqs}

    unigram_probs = {}
    for key, value in unigram_freqs.items():
        prob = value / total_sounds
        if prob <= 0:
            unigram_probs[key] = float('-inf')
        else:
            unigram_probs[key] = np.log(prob)
    return unigram_probs


def fit_bigrams(token_freqs, sound_idx,
                token_weighted=False,
                smoothed=False,
                use_word_boundaries=True):
    """
    Fits bigram probabilities in a manner consistent with the test suite:
    bigram_probs[row, col] = log( p(row|col) ), meaning col-based normalization.

    If token_weighted:
      - If use_word_boundaries=True, we add np.log(freq).
      - If use_word_boundaries=False, we add raw freq.
    Smoothing => +1 in linear space; then do col-sum-based normalization.
    """
    # Possibly remove '#' if use_word_boundaries=False
    if not use_word_boundaries and WORD_BOUNDARY in sound_idx:
        local_sounds = [s for s in sound_idx if s != WORD_BOUNDARY]
    else:
        local_sounds = sound_idx

    N = len(local_sounds)

    if smoothed:
        count_matrix = np.ones((N, N), dtype=float)
    else:
        count_matrix = np.zeros((N, N), dtype=float)

    for token, freq in token_freqs:
        if token_weighted:
            if use_word_boundaries:
                val = np.log(freq) if freq > 0 else 0
            else:
                val = freq
        else:
            val = 1

        bigrams = generate_bigrams(token, use_word_boundaries)
        for (prev_sound, next_sound) in bigrams:
            if prev_sound in local_sounds and next_sound in local_sounds:
                col_index = local_sounds.index(prev_sound)  # "col" => prev
                row_index = local_sounds.index(next_sound)  # "row" => next
                count_matrix[row_index, col_index] += val

    # Normalize so each column sums to 1
    col_sums = count_matrix.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_matrix = np.divide(
            count_matrix,
            col_sums,
            out=np.zeros_like(count_matrix),
            where=(col_sums != 0)
        )
    with np.errstate(divide='ignore', invalid='ignore'):
        bigram_probs = np.log(prob_matrix)

    return bigram_probs


def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits positional unigram scores in linear space. 
    e.g. pos_unigram_freqs[position][sound] = fraction among that position.
    Weighted => add np.log(freq) if freq>0 (per older tests).
    """
    pos_unigram_freqs = defaultdict(lambda: defaultdict(float))

    if smoothed:
        unique_sounds = set(sound for token, _ in token_freqs for sound in token)
        for i in range(MAX_WORD_LEN):
            for sound in unique_sounds:
                pos_unigram_freqs[i][sound] = 1.0

    for token, freq in token_freqs:
        if token_weighted:
            val = np.log(freq) if freq > 0 else 0
        else:
            val = 1.0
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += val

    pos_unigram_freqs = normalize_positional_counts(pos_unigram_freqs)
    return pos_unigram_freqs


def fit_non_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits non-positional unigram probabilities, returning {sound: log(prob)}.

    Weighted => add raw freq, not logs, per the "non-positional" tests.
    Smoothing => +1 in linear space for each sound.
    """
    unigram_freqs = defaultdict(float)

    if smoothed:
        unique_sounds = set(sound for token, _ in token_freqs for sound in token)
        for s in unique_sounds:
            unigram_freqs[s] = 1.0

    for token, freq in token_freqs:
        if token_weighted:
            val = freq  # raw freq
        else:
            val = 1.0
        for sound in token:
            unigram_freqs[sound] += val

    total = sum(unigram_freqs.values())
    if total > 0:
        for sound in unigram_freqs:
            p = unigram_freqs[sound] / total
            if p <= 0:
                unigram_freqs[sound] = float('-inf')
            else:
                unigram_freqs[sound] = np.log(p)
    else:
        for sound in unigram_freqs:
            unigram_freqs[sound] = float('-inf')

    return unigram_freqs


def fit_positional_bigrams(token_freqs, token_weighted=False, smoothed=False, conditional=False, use_word_boundaries=False):
    """
    Fits positional bigram probabilities in linear space, 
    storing them in pos_bigram_freqs[(pos1,pos2)][(s1,s2)] = fraction.

    Weighted => add np.log(freq) if freq>0 (positional tests).
    Smoothing => +1 for each possible bigram in linear space.
    If conditional=True, we normalize by the first element in the bigram 
    at that position. If not, we normalize by total in that position.
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
        if token_weighted:
            val = np.log(freq) if freq > 0 else 0
        else:
            val = 1.0
        bigrams = generate_bigrams(token, use_word_boundaries)
        for idx in range(len(bigrams)):
            s1, s2 = bigrams[idx]
            pos_bigram_freqs[(idx, idx+1)][(s1, s2)] += val

    pos_bigram_freqs = normalize_positional_counts(pos_bigram_freqs, conditional=conditional)
    return pos_bigram_freqs


def fit_non_positional_bigrams(
    token_freqs,
    token_weighted=False,
    smoothed=False,
    use_word_boundaries=False
):
    """
    Fits non-positional bigram probabilities, returning a 2D array of shape (N,N):
        bigram_probs[row, col] = log( p(row | col) ).
    The tests want column-based normalization (col sums to 1).
    Weighted => add raw freq if token_weighted, else +1 per occurrence. 
    Smoothing => +1 in linear space for each cell.
    """
    unique_sounds = {sound for token, _ in token_freqs for sound in token}
    if use_word_boundaries:
        unique_sounds.add(WORD_BOUNDARY)
    else:
        if WORD_BOUNDARY in unique_sounds:
            unique_sounds.remove(WORD_BOUNDARY)

    sound_list = sorted(unique_sounds)
    N = len(sound_list)

    if smoothed:
        count_matrix = np.ones((N, N), dtype=float)
    else:
        count_matrix = np.zeros((N, N), dtype=float)

    for token, freq in token_freqs:
        val = freq if token_weighted else 1.0
        bigrams = generate_bigrams(token, use_word_boundaries)
        for (s1, s2) in bigrams:
            if s1 in sound_list and s2 in sound_list:
                col = sound_list.index(s1)  # "prev"
                row = sound_list.index(s2)  # "next"
                count_matrix[row, col] += val

    # Column-based normalization => each column sums to 1
    col_sums = count_matrix.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        prob_matrix = np.divide(
            count_matrix,
            col_sums,
            out=np.zeros_like(count_matrix),
            where=(col_sums != 0)
        )
    with np.errstate(divide='ignore', invalid='ignore'):
        bigram_probs = np.log(prob_matrix)

    return bigram_probs


def normalize_positional_counts(counts, conditional=False):
    """
    Normalizes positional counts by total counts for each position 
    if not conditional, or by the preceding sound if conditional=True.

    The final stored values remain in *linear* space (fractions), 
    e.g. pos_bigram_freqs[(pos1,pos2)][(s1,s2)] = fraction.
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
                # gram is (s1, s2), so s1 is the "preceding" item
                preceding_sound_dict[gram[0]] += count_val

            for gram in counts[idx].keys():
                total = preceding_sound_dict[gram[0]]
                if total > 0:
                    counts[idx][gram] /= total
                else:
                    counts[idx][gram] = 0
    return counts


###########################
# Code for testing models #
###########################

def score_corpus(token_freqs, fitted_models, sound_idx):
    """
    Given a dataset and a list of fitted models, returns the score for each 
    word under each model. If a log probability is -inf, we replace it with ''.
    """
    (
        uni_models,
        bi_cond_pos_wb_models,
        bi_cond_pos_noWB_models,
        bi_cond_nonpos_wb_models,
        bi_cond_nonpos_noWB_models,
        uni_joint_pos_models,
        uni_joint_nonpos_models,
        bi_joint_pos_wb_models,
        bi_joint_pos_noWB_models,
        bi_joint_nonpos_wb_models,
        bi_joint_nonpos_noWB_models,
    ) = fitted_models

    results = []

    for token, _ in token_freqs:
        row = [' '.join(token), len(token)]

        # --- Unigram probabilities ---
        for model in uni_models:
            prob = get_unigram_prob(token, model)
            row.append(prob if not np.isinf(prob) else '')

        # --- Bigram conditional positional (WB) ---
        for model in bi_cond_pos_wb_models:
            score = get_pos_bigram_score(token, model, conditional=True, use_word_boundaries=True)
            row.append(score if not np.isinf(score) else '')

        # --- Bigram conditional positional (noWB) ---
        for model in bi_cond_pos_noWB_models:
            score = get_pos_bigram_score(token, model, conditional=True, use_word_boundaries=False)
            row.append(score if not np.isinf(score) else '')

        # --- Bigram conditional non-positional (WB) ---
        for model in bi_cond_nonpos_wb_models:
            prob = get_bigram_prob(token, model, sound_idx)
            row.append(prob if not np.isinf(prob) else '')

        # --- Bigram conditional non-positional (noWB) ---
        for model in bi_cond_nonpos_noWB_models:
            prob = get_bigram_prob(token, model, sound_idx, use_word_boundaries=False)
            row.append(prob if not np.isinf(prob) else '')

        # --- Positional unigram joint probabilities ---
        for model in uni_joint_pos_models:
            score = get_pos_unigram_score(token, model)
            row.append(score)

        # --- Non-positional unigram joint probabilities ---
        for model in uni_joint_nonpos_models:
            score = get_non_pos_unigram_score(token, model)
            row.append(score)

        # --- Positional bigram joint probabilities (WB) ---
        for model in bi_joint_pos_wb_models:
            score = get_pos_bigram_score(token, model, conditional=False, use_word_boundaries=True)
            row.append(score if not np.isinf(score) else '')

        # --- Positional bigram joint probabilities (noWB) ---
        for model in bi_joint_pos_noWB_models:
            score = get_pos_bigram_score(token, model, conditional=False, use_word_boundaries=False)
            row.append(score if not np.isinf(score) else '')

        # --- Non-positional bigram joint probabilities (WB) ---
        for model in bi_joint_nonpos_wb_models:
            score = get_non_pos_bigram_score(token, model, sound_idx, use_word_boundaries=True)
            row.append(score if not np.isinf(score) else '')

        # --- Non-positional bigram joint probabilities (noWB) ---
        for model in bi_joint_nonpos_noWB_models:
            score = get_non_pos_bigram_score(token, model, sound_idx, use_word_boundaries=False)
            row.append(score if not np.isinf(score) else '')

        results.append(row)

    return results


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


def get_bigram_prob(word, bigram_probs, sound_idx, use_word_boundaries=True):
    """
    Calculates total log probability of a word using a 2D bigram matrix:
    bigram_probs[row, col] = log( p(row|col) ), col=prev, row=next.
    """
    bigrams = generate_bigrams(word, use_word_boundaries)
    prob = 0.0
    for (prev, nxt) in bigrams:
        try:
            col = sound_idx.index(prev)
            row = sound_idx.index(nxt)
        except ValueError:
            return float('-inf')
        prob_val = bigram_probs[row, col]
        prob += prob_val
        if np.isinf(prob):
            return float('-inf')
    return prob


def get_pos_unigram_score(word, pos_uni_freqs):
    """
    Positional unigram "score" = start at 1.0, then add the fraction for each position.
    """
    score = 1.0
    for idx, sound in enumerate(word):
        score += pos_uni_freqs[idx].get(sound, 0.0)
    return score


def get_non_pos_unigram_score(word, unigram_freqs):
    """
    Non-positional unigram "score" = start at 1.0, then add exp of each log-prob 
    for each symbol. (The older tests expect a linear sum of probabilities.)
    """
    score = 1.0
    for sound in word:
        log_p = unigram_freqs.get(sound, float('-inf'))
        if np.isinf(log_p):
            # Adding 0 for unknown
            continue
        score += np.exp(log_p)
    return score


def get_pos_bigram_score(word, pos_bi_freqs, conditional=False, use_word_boundaries=False):
    """
    Returns a linear sum of the positional bigram probabilities, 
    starting from 1.0. Tests expect: 
        score = 1 + sum of pos_bi_freqs[(idx, idx+1)][(s1,s2)]
    """
    if use_word_boundaries:
        word = [WORD_BOUNDARY] + word + [WORD_BOUNDARY]

    score = 1.0
    bigrams = list(nltk.ngrams(word, 2))
    for i in range(len(bigrams)):
        s1, s2 = bigrams[i]
        position = (i, i+1)
        prob = pos_bi_freqs[position].get((s1, s2), 0.0)
        score += prob
    return score


def get_non_pos_bigram_score(word, bigram_freqs, sound_idx, use_word_boundaries=False):
    """
    Sums log-probs from a 2D matrix, bigram_freqs[row, col] = log( p(row|col) ).
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
        val = bigram_freqs[row, col]
        total_log += val
        if np.isinf(total_log):
            return float('-inf')
    return total_log


##################
# Entry function #
##################

def run(train, test, out):
    train_token_freqs = read_tokens(train)
    test_token_freqs = read_tokens(test)
    unique_sounds = set(sound for token, _ in train_token_freqs for sound in token)
    # Keep boundary symbol in the sorted list
    sound_idx = sorted(list(unique_sounds)) + [WORD_BOUNDARY]

    fitted_models = fit_ngram_models(train_token_freqs, sound_idx)
    results = score_corpus(test_token_freqs, fitted_models, sound_idx)
    write_results(results, out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate a suite of unigram/bigram scores for a data set."
    )
    parser.add_argument('train_file', type=str, help='Path to the input corpus file.')
    parser.add_argument('test_file', type=str, help='Path to test data file.')
    parser.add_argument('output_file', type=str, help='Path to output file with word judgements.')
    args = parser.parse_args()
    run(args.train_file, args.test_file, args.output_file)