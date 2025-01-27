# FileName: ngram_calculator.py 
# version 1.0
# Summary: [Short summary of functions contained within]
# Tags: [comma delineated tags associated with this file]

#Run command, from parent directory
#"python src/ngram_calculator.py data/english.csv data/sample_test_data/english_test_data.csv src/output.csv"

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
    consists of a list of the individual symbols.
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
    # Get unigram probabilities
    uni_models = []
    uni_models.append(fit_unigrams(token_freqs))  # uni_prob
    uni_models.append(fit_unigrams(token_freqs, token_weighted=True))  # uni_prob_freq_weighted
    uni_models.append(fit_unigrams(token_freqs, smoothed=True))  # uni_prob_smoothed
    uni_models.append(fit_unigrams(token_freqs, smoothed=True, token_weighted=True))  # uni_prob_freq_weighted_smoothed

    # Get bigram conditional, positional, with word boundaries
    bi_cond_pos_wb_models = []
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, conditional=True, use_word_boundaries=True))  # bi_cond_pos_wb
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, conditional=True, use_word_boundaries=True))  # bi_cond_pos_wb_freq_weighted
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, conditional=True, use_word_boundaries=True))  # bi_cond_pos_wb_smoothed
    bi_cond_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, conditional=True, use_word_boundaries=True))  # bi_cond_pos_wb_freq_weighted_smoothed
    
    # Get bigram conditional, positional, without word boundaries
    bi_cond_pos_noWB_models = []
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, conditional=True, use_word_boundaries=False))  # bi_cond_pos_noWB
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, conditional=True, use_word_boundaries=False))  # bi_cond_pos_noWB_freq_weighted
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, conditional=True, use_word_boundaries=False))  # bi_cond_pos_noWB_smoothed
    bi_cond_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, conditional=True, use_word_boundaries=False))  # bi_cond_pos_noWB_freq_weighted_smoothed

    # Get bigram conditional non-positional models with word boundaries
    bi_cond_nonpos_wb_models = []
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx))  # bi_cond_nonpos_wb
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx, token_weighted=True))  # bi_cond_nonpos_wb_freq_weighted
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True))  # bi_cond_nonpos_wb_smoothed
    bi_cond_nonpos_wb_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True, token_weighted=True))  # bi_cond_nonpos_wb_freq_weighted_smoothed

    # Get bigram conditional non-positional models without word boundaries
    bi_cond_nonpos_noWB_models = []
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, use_word_boundaries=False))  # bi_cond_nonpos_noWB
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, token_weighted=True, use_word_boundaries=False))  # bi_cond_nonpos_noWB_freq_weighted
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True, use_word_boundaries=False))  # bi_cond_nonpos_noWB_smoothed
    bi_cond_nonpos_noWB_models.append(fit_bigrams(token_freqs, sound_idx, smoothed=True, token_weighted=True, use_word_boundaries=False))  # bi_cond_nonpos_noWB_freq_weighted_smoothed
    
    # Get positional unigram joint probabilities
    uni_joint_pos_models = []
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs))  # uni_joint_pos
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs, token_weighted=True))  # uni_joint_pos_freq_weighted
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs, smoothed=True))  # uni_joint_pos_smoothed
    uni_joint_pos_models.append(fit_positional_unigrams(token_freqs, smoothed=True, token_weighted=True))  # uni_joint_pos_freq_weighted_smoothed

    # Non-Positional Unigram Joint Probabilities
    uni_joint_nonpos_models = []
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs))  # uni_joint_nonpos
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs, token_weighted=True))  # uni_joint_nonpos_freq_weighted
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs, smoothed=True))  # uni_joint_nonpos_smoothed
    uni_joint_nonpos_models.append(fit_non_positional_unigrams(token_freqs, smoothed=True, token_weighted=True))  # uni_joint_nonpos_freq_weighted_smoothed

    # Bigram Joint Positional Models with Word Boundaries
    bi_joint_pos_wb_models = []
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, use_word_boundaries=True))  # bi_joint_pos_wb
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=True))  # bi_joint_pos_wb_freq_weighted
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=True))  # bi_joint_pos_wb_smoothed
    bi_joint_pos_wb_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=True))  # bi_joint_pos_wb_freq_weighted_smoothed

    # Bigram Joint Positional Models without Word Boundaries
    bi_joint_pos_noWB_models = []
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, use_word_boundaries=False))  # bi_joint_pos_noWB
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=False))  # bi_joint_pos_noWB_freq_weighted
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=False))  # bi_joint_pos_noWB_smoothed
    bi_joint_pos_noWB_models.append(fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=False))  # bi_joint_pos_noWB_freq_weighted_smoothed

    # Bigram Joint Non-Positional Models with Word Boundaries
    bi_joint_nonpos_wb_models = []
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, use_word_boundaries=True))  # bi_joint_nonpos_wb
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=True))  # bi_joint_nonpos_wb_freq_weighted
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=True))  # bi_joint_nonpos_wb_smoothed
    bi_joint_nonpos_wb_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=True))  # bi_joint_nonpos_wb_freq_weighted_smoothed

    # Bigram Joint Non-Positional Models without Word Boundaries
    bi_joint_nonpos_noWB_models = []
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, use_word_boundaries=False))  # bi_joint_nonpos_noWB
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, token_weighted=True, use_word_boundaries=False))  # bi_joint_nonpos_noWB_freq_weighted
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, use_word_boundaries=False))  # bi_joint_nonpos_noWB_smoothed
    bi_joint_nonpos_noWB_models.append(fit_non_positional_bigrams(token_freqs, smoothed=True, token_weighted=True, use_word_boundaries=False))  # bi_joint_nonpos_noWB_freq_weighted_smoothed

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
    Fits unigram probabilities.

    returns: A dictionary of unigram:log-probability pairs.
    """
    default_func = lambda: int(smoothed)
    unigram_freqs = defaultdict(default_func)

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for sound in token:
            unigram_freqs[sound] += val

    total_sounds = sum(unigram_freqs.values())
    unigram_probs = {}
    for key, value in unigram_freqs.items():
        prob = value / total_sounds
        unigram_probs[key] = np.log(prob)
    return unigram_probs

def fit_bigrams(token_freqs, sound_idx,
                token_weighted=False,
                smoothed=False,
                use_word_boundaries=True):
    """
    Fits bigram probabilities in a manner consistent with the test suite, i.e.
    bigram_probs[row, col] = log( count(row, col) / sum_of_column(col) ),
    so each column sums to 1, meaning p(next | prev).

    Returns:
        A 2D numpy array of shape (N, N), where N = len(sound_idx).
    """
    # 1) Possibly remove '#' if use_word_boundaries=False
    #    Otherwise keep '#' so we get a 3x3 matrix for a,t,#
    if not use_word_boundaries and '#' in sound_idx:
        filtered = [s for s in sound_idx if s != '#']
        sound_idx = filtered

    N = len(sound_idx)

    # 2) Initialize count matrix
    import numpy as np
    if smoothed:
        count_matrix = np.ones((N, N), dtype=float)
    else:
        count_matrix = np.zeros((N, N), dtype=float)

    # 3) Fill counts: col=previous, row=next
    for token, freq in token_freqs:
        # Weighted => add freq; else => add 1
        val = freq if token_weighted else 1

        # Use boundaries if asked
        bigrams = generate_bigrams(token, use_word_boundaries)

        for (prev_sound, next_sound) in bigrams:
            if prev_sound in sound_idx and next_sound in sound_idx:
                col_index = sound_idx.index(prev_sound)  # col=prev
                row_index = sound_idx.index(next_sound)  # row=next
                count_matrix[row_index, col_index] += val

    # 4) Normalize each column so columns sum to 1
    col_sums = count_matrix.sum(axis=0, keepdims=True)
    prob_matrix = np.divide(
        count_matrix,
        col_sums,
        out=np.zeros_like(count_matrix),
        where=(col_sums != 0)
    )

    # 5) Take log
    with np.errstate(divide='ignore', invalid='ignore'):
        bigram_probs = np.log(prob_matrix)
        # 0’s become -inf automatically

    return bigram_probs

def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits positional unigram scores.

    returns: A dictionary mapping positions to dictionaries of unigrams and their scores.
    """
    pos_unigram_freqs = defaultdict(lambda: defaultdict(int))

    if smoothed:
        unique_sounds = set(
            [sound for token, _ in token_freqs for sound in token]
        )
        for i in range(MAX_WORD_LEN):
            for sound in unique_sounds:
                pos_unigram_freqs[i][sound] = 1

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += val

    pos_unigram_freqs = normalize_positional_counts(pos_unigram_freqs)

    return pos_unigram_freqs

def fit_non_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    Fits non-positional unigram scores.

    returns: A dictionary of unigrams and their scores.
    """
    unigram_freqs = defaultdict(int)

    if smoothed:
        unique_sounds = set(sound for token, _ in token_freqs for sound in token)
        for sound in unique_sounds:
            unigram_freqs[sound] = 1

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for sound in token:
            unigram_freqs[sound] += val

    total = sum(unigram_freqs.values())
    for sound in unigram_freqs:
        unigram_freqs[sound] /= total

    return unigram_freqs

def fit_positional_bigrams(token_freqs, token_weighted=False, smoothed=False, conditional=False, use_word_boundaries=False):
    """
    Fits positional bigram probabilities.

    returns: A dictionary mapping position pairs to dictionaries of bigrams and their probabilities.
    """
    pos_bigram_freqs = defaultdict(lambda: defaultdict(int))

    # Define unique_sounds before using it
    unique_sounds = set(
        [sound for token, _ in token_freqs for sound in token]
    )
    if use_word_boundaries:
        unique_sounds.add(WORD_BOUNDARY)

    if smoothed:
        max_len = MAX_WORD_LEN + 2 if use_word_boundaries else MAX_WORD_LEN
        for i in range(max_len - 1):
            for s1 in unique_sounds:
                for s2 in unique_sounds:
                    pos_bigram_freqs[(i, i+1)][(s1, s2)] = 1

    for token, freq in token_freqs:
        val = freq if token_weighted else 1
        bigrams = generate_bigrams(token, use_word_boundaries)
        for idx in range(len(bigrams)):
            s1, s2 = bigrams[idx]
            pos_bigram_freqs[(idx, idx + 1)][(s1, s2)] += val

    pos_bigram_freqs = normalize_positional_counts(pos_bigram_freqs, conditional=conditional)

    return pos_bigram_freqs

def fit_non_positional_bigrams(
    token_freqs,
    token_weighted=False,
    smoothed=False,
    use_word_boundaries=False
):
    """
    Fits non-positional bigram probabilities (in log space).
    
    - Defaults to use_word_boundaries=False, because the test 
      "testFitNonPositionalBigrams" expects a 2x2 matrix for [a,t] only.
    - If you do want boundaries in other contexts, pass use_word_boundaries=True.
    
    Returns:
        bigram_probs: A 2D NumPy array of shape (N, N),
          where N = len(unique_sounds) (which excludes '#' if no boundaries).
          bigram_probs[i, j] = log( P( col_j | row_i ) ).
    """

    # 1) Identify the unique sounds. If no boundaries, remove '#'.
    unique_sounds = {sound for token, _ in token_freqs for sound in token}
    if use_word_boundaries:
        unique_sounds.add(WORD_BOUNDARY)
    else:
        if WORD_BOUNDARY in unique_sounds:
            unique_sounds.remove(WORD_BOUNDARY)

    sound_list = sorted(unique_sounds)
    N = len(sound_list)

    # 2) Initialize count matrix, using 1 if smoothed (Add-1), else 0
    if smoothed:
        count_matrix = np.ones((N, N), dtype=float)
    else:
        count_matrix = np.zeros((N, N), dtype=float)

    # 3) Count bigram occurrences, row=previous sound, col=next sound
    for token, freq in token_freqs:
        val = freq if token_weighted else 1  # for "weighted" sums use raw freq
        bigrams = generate_bigrams(token, use_word_boundaries)

        for (s1, s2) in bigrams:
            if s1 in sound_list and s2 in sound_list:
                row = sound_list.index(s1)  # "previous"
                col = sound_list.index(s2)  # "next"
                count_matrix[row, col] += val

    # 4) Normalize rows => p(next|prev)
    row_sums = count_matrix.sum(axis=1, keepdims=True)
    prob_matrix = np.divide(
        count_matrix,
        row_sums,
        out=np.zeros_like(count_matrix),
        where=(row_sums != 0)
    )

    # 5) Take log(prob)
    with np.errstate(divide='ignore', invalid='ignore'):
        bigram_probs = np.log(prob_matrix)
        # zeros become -inf automatically

    return bigram_probs

def normalize_positional_counts(counts, conditional=False):
    """
    Normalizes positional counts by total counts for each position.
    """
    if not conditional:
        for idx in counts.keys():
            total = sum(counts[idx].values())
            if total > 0:
                # Normalize counts by total count
                for gram in counts[idx].keys():
                    counts[idx][gram] /= total
            else:
                # Handle the case when total is zero
                for gram in counts[idx].keys():
                    counts[idx][gram] = 0  # Set to zero or a default value
    else:
        # Iterate over each pair of indices
        for idx in counts.keys():
            preceding_sound_dict = defaultdict(int)
            for gram, count in counts[idx].items():
                preceding_sound_dict[gram[0]] += count
            for gram in counts[idx].keys():
                total = preceding_sound_dict[gram[0]]
                if total > 0:
                    counts[idx][gram] /= total
                else:
                    # Handle the case when total is zero
                    counts[idx][gram] = 0  # Set to zero or a default value
    return counts


###########################
# Code for testing models #
###########################

def score_corpus(token_freqs, fitted_models, sound_idx):
    """
    Given a dataset and a list of fitted models, returns the score for each 
    word under each model.

    token_freqs: A list of tuples of word-frequency pairs. Frequencies 
    aren't used in this function.
    sound_idx: The list of unique sounds used to map sound identity to matrix
    dimensions.

    fitted_models: A tuple of lists of models.

    returns: A list of lists of scores. Each sublist corresponds to one word. Each
    sublist contains the word itself, its length, and its score under each of the
    ngram models.
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

        # Unigram probabilities
        for model in uni_models:
            prob = get_unigram_prob(token, model)
            row.append(prob if not np.isinf(prob) else '')

        # Bigram conditional positional with word boundaries
        for model in bi_cond_pos_wb_models:
            score = get_pos_bigram_score(token, model, conditional=True, use_word_boundaries=True)
            row.append(score if not np.isinf(score) else '')
        
        # Bigram conditional positional without word boundaries
        for model in bi_cond_pos_noWB_models:
            score = get_pos_bigram_score(token, model, conditional=True, use_word_boundaries=False)
            row.append(score if not np.isinf(score) else '')

        # Bigram conditional non-positional with word boundaries
        for model in bi_cond_nonpos_wb_models:
            prob = get_bigram_prob(token, model, sound_idx)
            row.append(prob if not np.isinf(prob) else '')

        # Bigram conditional non-positional without word boundaries
        for model in bi_cond_nonpos_noWB_models:
            prob = get_bigram_prob(token, model, sound_idx, use_word_boundaries=False)
            row.append(prob if not np.isinf(prob) else '')
        
        # Positional unigram joint probabilities
        for model in uni_joint_pos_models:
            score = get_pos_unigram_score(token, model)
            row.append(score)

        # Non-positional unigram joint probabilities
        for model in uni_joint_nonpos_models:
            score = get_non_pos_unigram_score(token, model)
            row.append(score)

        # Positional bigram joint probabilities with word boundaries
        for model in bi_joint_pos_wb_models:
            score = get_pos_bigram_score(token, model, conditional=False, use_word_boundaries=True)
            row.append(score if not np.isinf(score) else '')

        # Positional bigram joint probabilities without word boundaries
        for model in bi_joint_pos_noWB_models:
            score = get_pos_bigram_score(token, model, conditional=False)
            row.append(score if not np.isinf(score) else '')

        # Non-positional bigram joint probabilities with word boundaries
        for model in bi_joint_nonpos_wb_models:
            score = get_non_pos_bigram_score(token, model, use_word_boundaries=True)
            row.append(score)

        # Non-positional bigram joint probabilities without word boundaries
        for model in bi_joint_nonpos_noWB_models:
            score = get_non_pos_bigram_score(token, model)
            row.append(score)


        results.append(row)

    return results

def get_unigram_prob(word, unigram_probs):
    """
    Calculates the unigram probability of a word given a fitted unigram model.

    returns: The log probability of the word under the unigram model.
    """
    prob = 0
    for sound in word:
        sound_prob = unigram_probs.get(sound, float('-inf'))
        prob += sound_prob
    return prob

def get_bigram_prob(word, bigram_probs, sound_idx, use_word_boundaries=True):
    """
    Calculates the bigram probability of a word given a fitted bigram model.

    returns: The log probability of the word under the bigram model.
    """
    bigrams = generate_bigrams(word, use_word_boundaries)
    prob = 0
    for s1, s2 in bigrams:
        try:
            idx_s1 = sound_idx.index(s1)
            idx_s2 = sound_idx.index(s2)
            prob += bigram_probs[idx_s2][idx_s1]
        except ValueError:
            prob += float('-inf')
    return prob


def get_pos_unigram_score(word, pos_uni_freqs):
    """
    Calculates the positional unigram score of a word.

    returns: The score of the word under the positional unigram model.
    """
    score = 1
    for idx, sound in enumerate(word):
        score += pos_uni_freqs[idx].get(sound, 0)
    return score

def get_non_pos_unigram_score(word, unigram_freqs):
    """
    Calculates the non-positional unigram score of a word.
    """
    score = 1
    for sound in word:
        score += unigram_freqs.get(sound, 0)
    return score

def get_pos_bigram_score(word, pos_bi_freqs, conditional=False, use_word_boundaries=False):
    """
    Calculates the positional bigram score of a word.

    returns: The score of the word under the positional bigram model.
    """

    if use_word_boundaries:
        word = [WORD_BOUNDARY] + word + [WORD_BOUNDARY]
    score = 0  # Start with 0 for log probabilities
    bigrams = list(nltk.ngrams(word, 2))
    for idx in range(len(bigrams)):
        bigram = bigrams[idx]
        position = (idx, idx + 1)
        prob = pos_bi_freqs[position].get(bigram, 0)
        if prob > 0:
            score += np.log(prob)
        else:
            score += float('-inf')
    return score


def get_non_pos_bigram_score(word, bigram_freqs, use_word_boundaries=False):
    """
    Calculates the non-positional bigram score of a word.
    """
    if use_word_boundaries:
        word = [WORD_BOUNDARY] + word + [WORD_BOUNDARY]
    score = 0  # Use 0 for log probabilities
    bigrams = nltk.ngrams(word, 2)
    for bigram in bigrams:
        prob = bigram_freqs.get(bigram, 0)
        if prob > 0:
            score += np.log(prob)
        else:
            score += float('-inf')
    return score

##################
# Entry function #
##################

def run(train, test, out):
    train_token_freqs = read_tokens(train)
    test_token_freqs = read_tokens(test)
    unique_sounds = set(
        [sound for token, _ in train_token_freqs for sound in token]
    )
    sound_idx = sorted(list(unique_sounds)) + [WORD_BOUNDARY]

    fitted_models = fit_ngram_models(train_token_freqs, sound_idx)
    results = score_corpus(test_token_freqs, fitted_models, sound_idx)
    write_results(results, out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = "Calculate a suite of unigram/bigram scores for a data set."
    )
    parser.add_argument(
        'train_file', type=str, help='Path to the input corpus file.'
    )
    parser.add_argument(
        'test_file', type=str, help='Path to test data file' 
    )
    parser.add_argument(
        'output_file', type=str, help='Path to output file with word judgements' 
    )
    args = parser.parse_args()
    run(args.train_file, args.test_file, args.output_file)
