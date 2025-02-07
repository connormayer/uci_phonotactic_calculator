# FileName: ngram_calculator.py 
# version 1.2
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
    use_word_boundaries: Whether to include word boundary markers (default=True).

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
    """
    Fits a suite of n-gram models (both positional and non-positional)
    for use in scoring a test corpus. The models include:
    
    - Non-positional unigram probabilities (uni_prob) via
      fit_non_positional_unigram_probabilities().
    - Positional bigrams (conditional and joint, with and without word boundaries)
      via fit_positional_bigrams().
    - Non-positional bigrams (conditional) via fit_bigrams() and joint via
      fit_non_positional_bigrams().
    - Positional unigram joint probabilities via fit_positional_unigrams().
    - Non-positional unigram joint probabilities via fit_non_positional_unigrams().
    
    token_freqs: List of [token, frequency] pairs.
    sound_idx: Sorted list of sounds (including the boundary marker).

    returns: A tuple of fitted models in a fixed order.
    """
    # Get non-positional unigram probabilities (for uni_prob columns)
    uni_models = []
    uni_models.append(fit_non_positional_unigram_probabilities(token_freqs))  # uni_prob
    uni_models.append(fit_non_positional_unigram_probabilities(token_freqs, token_weighted=True))  # uni_prob_freq_weighted
    uni_models.append(fit_non_positional_unigram_probabilities(token_freqs, smoothed=True))  # uni_prob_smoothed
    uni_models.append(fit_non_positional_unigram_probabilities(token_freqs, smoothed=True, token_weighted=True))  # uni_prob_freq_weighted_smoothed

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


def fit_non_positional_unigram_probabilities(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits non-positional unigram probabilities (returns a dict of {sound: log(prob)}).

    This function is used for the uni_prob columns.
    - When token_weighted is True, each occurrence is weighted by np.log(freq)
      (if freq > 0); otherwise, each occurrence counts as 1.
    - Smoothing: if smoothed=True, each sound’s count is initialized with 1.
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


#######################################
# New helper for both bigram functions
#######################################
def _fit_bigram_matrix(
    token_freqs,
    sound_list,
    token_weighted=False,
    smoothed=False,
    use_word_boundaries=True,
    log_boundaries_weight=False,
):
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
    # Possibly remove '#' if not using boundaries
    local_sounds = list(sound_list)
    if not use_word_boundaries and WORD_BOUNDARY in local_sounds:
        local_sounds.remove(WORD_BOUNDARY)

    N = len(local_sounds)

    # Initialize counts
    if smoothed:
        count_matrix = np.ones((N, N), dtype=float)
    else:
        count_matrix = np.zeros((N, N), dtype=float)

    # Fill counts
    for token, freq in token_freqs:
        if token_weighted and freq > 0:
            val = np.log(freq)
        else:
            val = 1.0 #else 1.0

        bigrams = generate_bigrams(token, use_word_boundaries)
        for (prev, nxt) in bigrams:
            if prev in local_sounds and nxt in local_sounds:
                col = local_sounds.index(prev)
                row = local_sounds.index(nxt)
                count_matrix[row, col] += val

    # Column-based normalization: each column sums to 1
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


def fit_bigrams(token_freqs, sound_idx,
                token_weighted=False,
                smoothed=False,
                use_word_boundaries=True):
    """
    Fits non-positional bigram probabilities in a manner consistent with the test suite.
    Here, bigram_probs[row, col] = log( p(row|col) ), where normalization is done on each column.

    Parameters:
      token_freqs: List of [token, frequency] pairs.
      sound_idx: Sorted list of sounds.
      token_weighted: If True, counts are weighted by frequency (using np.log(freq)
                     when use_word_boundaries is True, else raw freq).
      smoothed: If True, each cell is initialized with 1.
      use_word_boundaries: Whether to include word boundary markers (default=True).

    returns:
      A 2D numpy array of log-probabilities.
    """
    # We pass log_boundaries_weight=True when use_word_boundaries is True.
    log_boundaries = use_word_boundaries

    bigram_probs = _fit_bigram_matrix(
        token_freqs,
        sound_idx,
        token_weighted=token_weighted,
        smoothed=smoothed,
        use_word_boundaries=use_word_boundaries,
        log_boundaries_weight=log_boundaries,
    )
    return bigram_probs


def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits positional unigram joint probabilities.
    For each position, counts the occurrences of sounds (with optional weighting and smoothing)
    and then normalizes to produce a probability distribution at that position.
    (Used for uni_joint_pos columns.)
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
    Fits non-positional unigram joint probabilities, returning {sound: log(prob)}.
    This function is used for the uni_joint_nonpos columns.
    - When token_weighted is True, raw frequency is added.
    - Smoothing: if smoothed=True, each sound’s count is initialized with 1.
    """
    unigram_freqs = defaultdict(float)

    if smoothed:
        unique_sounds = set(sound for token, _ in token_freqs for sound in token)
        for s in unique_sounds:
            unigram_freqs[s] = 1.0

    for token, freq in token_freqs:
        if token_weighted:
            val = freq  # raw frequency
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

    Note: For positional models, tests expect word boundaries NOT to be added.
    Thus the default for use_word_boundaries is set to False.
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
    use_word_boundaries=True
):
    """
    Fits non-positional bigram joint probabilities, returning a 2D array of shape (N,N):
        bigram_probs[row, col] = log( p(row | col) ).
    This function (used for the bi_joint_nonpos columns) performs column-based normalization.
    - When token_weighted is True, raw frequency is added.
    - Smoothing: if smoothed=True, each cell is initialized with 1.
    - use_word_boundaries: Whether to include the boundary marker (default=True).
    """
    # Gather the unique sounds, then call our unified bigram function
    unique_sounds = {sound for token, _ in token_freqs for sound in token}
    if use_word_boundaries:
        unique_sounds.add(WORD_BOUNDARY)
    else:
        if WORD_BOUNDARY in unique_sounds:
            unique_sounds.remove(WORD_BOUNDARY)

    sound_list = sorted(unique_sounds)

    # For the non-positional bigram joint model, we do not use log(freq) weighting even if use_word_boundaries is True.
    bigram_probs = _fit_bigram_matrix(
        token_freqs,
        sound_list,
        token_weighted=token_weighted,
        smoothed=smoothed,
        use_word_boundaries=use_word_boundaries,
        log_boundaries_weight=False,
    )
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
    word under each model. If a log probability is -inf, it is replaced with ''.
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
    bigram_probs[row, col] = log( p(row|col) ), where the bigrams (possibly including boundaries)
    are generated automatically.
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
    (Used for positional joint unigram models.)
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
    Returns a linear sum of the positional bigram probabilities, starting from 1.0.
    Tests expect: score = 1 + sum of pos_bi_freqs[(idx, idx+1)][(s1,s2)].
    
    Parameters:
      conditional: If True, the bigram probabilities have been normalized conditionally 
                   on the preceding symbol; if False, they are joint probabilities.
      use_word_boundaries: Whether the word should be augmented with boundary markers.
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
    Sums log-probs from a 2D matrix, where bigram_freqs[row, col] = log( p(row|col) ).
    Used for non-positional joint bigram models.
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