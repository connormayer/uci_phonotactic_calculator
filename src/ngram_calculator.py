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

    'bi_prob',
    'bi_prob_freq_weighted',
    'bi_prob_smoothed',
    'bi_prob_freq_weighted_smoothed',

    'pos_uni_score',
    'pos_uni_score_freq_weighted',
    'pos_uni_score_smoothed',
    'pos_uni_score_freq_weighted_smoothed',

    'pos_bi_score',
    'pos_bi_score_freq_weighted',
    'pos_bi_score_smoothed',
    'pos_bi_score_freq_weighted_smoothed'
]

####################
# Helper functions #
####################

def generate_bigrams(token):
    """
    Returns a list of sound bigrams given a single word token.

    token: The list of symbols in the token.

    returns: The list of bigrams of the token.
    """
    return nltk.ngrams([WORD_BOUNDARY] + token + [WORD_BOUNDARY], 2)

def read_tokens(dataset):
    """
    Reads in a file containing tokens and optional frequencies and converts
    it to a list of tokens and a list of token/frequency pairs.

    dataset: The path to the dataset.

    returns: A list of lists, where each sublist corresponds to a token and
    consists of a list of the individual symbols.
    """
    with open(dataset, 'r') as f:
        reader = csv.reader(f)
    
        token_freqs = []

        for row in reader:
            split_token = row[0].split(' ')
            freq = float(row[1]) if len(row) == 2 else 0
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
    with open(outfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)

###########################
# Code for fitting models #
###########################

def fit_ngram_models(token_freqs, sound_idx):
    """
    Fits all of the ngram models to the provided data and returns the fitted
    models.

    token_freqs: A list of tuples of word-frequency pairs.
    sound_idx: The list of unique sounds used to map sound identity to matrix
    dimensions.

    returns: A list of lists of models. These models are in the same order as
    defined in the HEADER file at the top of this file, and broken into sublists
    based on their type (unigram/bigram/positional unigram/positional bigram).
    """
    unigram_models = []
    # Get unigram probabilities
    unigram_models.append(fit_unigrams(token_freqs))
    unigram_models.append(fit_unigrams(token_freqs, token_weighted=True))

    # Get bigram probabilities
    bigram_models = []
    bigram_models.append(fit_bigrams(token_freqs, sound_idx))
    bigram_models.append(
        fit_bigrams(token_freqs, sound_idx, token_weighted=True)
    )
    bigram_models.append(
        fit_bigrams(token_freqs, sound_idx, smoothed=True)
    )
    bigram_models.append(
        fit_bigrams(token_freqs, sound_idx, smoothed=True, token_weighted=True)
    )

    # Get positional unigram probabilities
    pos_unigram_models = []
    pos_unigram_models.append(fit_positional_unigrams(token_freqs))
    pos_unigram_models.append(
        fit_positional_unigrams(token_freqs, token_weighted=True)
    )
    pos_unigram_models.append(
        fit_positional_unigrams(token_freqs, smoothed=True)
    )
    pos_unigram_models.append(
        fit_positional_unigrams(token_freqs, smoothed=True, token_weighted=True)
    )

    # Get positional bigram probabilities
    pos_bigram_models = []
    pos_bigram_models.append(fit_positional_bigrams(token_freqs))
    pos_bigram_models.append(
        fit_positional_bigrams(token_freqs, token_weighted=True)
    )
    pos_bigram_models.append(
        fit_positional_bigrams(token_freqs, smoothed=True)
    )
    pos_bigram_models.append(
        fit_positional_bigrams(token_freqs, smoothed=True, token_weighted=True)
    )

    return unigram_models, bigram_models, pos_unigram_models, pos_bigram_models

def fit_unigrams(token_freqs, token_weighted=False):
    """
    This function takes a set of word tokens and returns a dictionary whose
    keys are unigrams and whose values are log unigram probabilities. Smoothing 
    isn't implemented for standard unigram scores: we assume the set of sounds
    in the training data is the full set of sounds.

    token_freqs: A list of tuples of word-frequency pairs
    token_weighted: If true, counts are weighted by log frequency of token

    returns: A dictionary of unigram:probability pairs.
    """
    unigram_freqs = defaultdict(int)

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for sound in token:
            unigram_freqs[sound] += val

    total_sounds = sum(unigram_freqs.values())
    unigram_freqs = {
        key: (np.log(value/total_sounds)) 
        for key, value in unigram_freqs.items()
    }
    return unigram_freqs

def fit_bigrams(token_freqs, sound_idx, token_weighted=False, smoothed=False):
    """
    This function takes a set of word tokens and a list of sounds and
    returns a matrix of bigrams probabilities. The sound list is necessary because
    we include counts of 0 for unattested bigram combinations.

    token_freqs: A list of tuples of word-frequency pairs.

    token_weighted: if True, counts are weighted by the log frequency
    of the words they occur in.

    smoothed: if True, start with a pseudo-count of 1 for every bigram.

    returns: A matrix of bigram probabilities, where rows correspond to the second
    sound in the bigram and columns correspond to the first.
    """
    num_sounds = len(sound_idx)

    if smoothed:
        count_matrix = np.ones((num_sounds, num_sounds))
    else:
        count_matrix = np.zeros((num_sounds, num_sounds))

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        bigrams = generate_bigrams(token)
        for s1, s2 in bigrams:
            count_matrix[sound_idx.index(s2)][sound_idx.index(s1)] += val

    bigram_probs = np.log(count_matrix / np.sum(count_matrix, 0))
    return bigram_probs

def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    This function takes a set of word tokens and returns a dictionary containing
    positional unigram log scores.

    token_freqs: A list of tuples of word-frequency pairs.

    token_weighted: If True, counts are weighted by log frequency of token.

    smoothed: If True, each start with a pseudo-count of 1 for every unigram in
    every position up to MAX_WORD_LEN. Note that this smoothing does not allow
    unseen unigrams to get probabilities > 0: rather it assigns known unigrams
    in unknown positions probabilities > 0.

    returns: A dictionary of dictionaries, where the first dictionary maps position
    indices to unigrams, and the second maps unigrams to their scores in that
    position.
    """
    pos_unigram_freqs = defaultdict(lambda: defaultdict(int))

    if smoothed:
        unique_sounds = set(
            [sound for token, _ in token_freqs for sound in token]
        )
        max_idx = max([len(token) for token, _ in token_freqs])

        for i in range(MAX_WORD_LEN):
            for sound in unique_sounds:
                pos_unigram_freqs[i][sound] = 1

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += val

    pos_unigram_freqs = normalize_positional_counts(pos_unigram_freqs)

    return pos_unigram_freqs

def fit_positional_bigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    This function takes a set of word tokens and returns a dictionary containing
    positional bigram scores.

    token_freqs: A list of tuples of word-frequency pairs.

    token_weighted: If True, counts are weighted by log frequency of token.

    smoothed: If True, each start with a pseudo-count of 1 for every bigram in
    every pair of positions up to MAX_WORD_LEN.

    returns: A dictionary of dictionaries, where the first dictionary maps pairs of
    position indices to bigrams, and the second maps bigram to their scores in those
    positions.
    """
    pos_bigram_freqs = defaultdict(lambda: defaultdict(int))

    if smoothed:
        unique_sounds = set(
            [sound for token, _ in token_freqs for sound in token]
        )
        for i in range(MAX_WORD_LEN - 1):
            for s1 in unique_sounds:
                for s2 in unique_sounds:
                    pos_bigram_freqs[(i, i+1)][(s1, s2)] = 1

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for idx, sound in enumerate(token):
            if idx < len(token) - 1:
                pos_bigram_freqs[(idx, idx + 1)][(sound, token[idx + 1])] += val

    pos_bigram_freqs = normalize_positional_counts(pos_bigram_freqs)
    
    return pos_bigram_freqs

def normalize_positional_counts(counts):
    """
    Normalizes positional counts by total counts for each position.
    """
    for idx in counts.keys():
        total = sum(counts[idx].values())
        for gram in counts[idx].keys():
            counts[idx][gram] /= total

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

    fitted_models: A list of lists of models. These models are in the same order as
    defined in the HEADER file at the top of this file, and broken into sublists
    based on their type (unigram/bigram/positional unigram/positional bigram).

    returns: A list of lists of scores. Each sublist corresponds to one word. Each
    sublist contains the word itself, its length, and its score under each of the
    ngram models.
    """
    uni_models, bi_models, pos_uni_models, pos_bi_models = fitted_models

    results = []
    
    for token, _ in token_freqs:
        row = [' '.join(token), len(token)]

        for model in uni_models:
            row.append(get_unigram_prob(token, model))

        for model in bi_models:
            row.append(get_bigram_prob(token, model, sound_idx))

        for model in pos_uni_models:
            row.append(get_pos_unigram_score(token, model))

        for model in pos_bi_models:
            row.append(get_pos_bigram_score(token, model))

        results.append(row)

    return results

def get_unigram_prob(word, unigram_probs):
    """
    Calculcates the unigram probability of a word given a fitted unigram model

    word: The test word
    ungiram_probs: The fitted model

    returns: The log probability of the word under the unigram model.
    """
    prob = 0
    for sound in word:
        # Add basic smoothing for sounds that appear in test data but
        #   not training data. Use float('-inf') because we are
        #   using log probabilities.
        prob += unigram_probs.get(sound, float('-inf'))

    return prob

def get_bigram_prob(word, bigram_probs, sound_idx):
    """
    Calculcates the bigram probability of a word given a fitted bigram model

    word: The test word
    bigram_probs: The fitted model
    sound_idx: The list of unique sounds used to map sound identity to matrix
    dimensions.

    returns: The log probability of the word under the bigram model.
    """
    bigrams = generate_bigrams(word)
    prob = 0
    for s1, s2 in bigrams:
        try:
            prob += bigram_probs[sound_idx.index(s2), sound_idx.index(s1)]
        except:
            # If bigram contains symbol we haven't seen, assign 0 probability
            prob += float('-inf')

    return prob

def get_pos_unigram_score(word, pos_uni_freqs):
    """
    Calculcates the positional unigram score of a word given a fitted 
    positional unigram model.

    word: The test word
    pos_uni_freqs: The fitted positional unigram model

    returns: The score of the word under the positional unigram model. Following
    Vitevich & Luce (2004), we add 1 to these scores.
    """
    score = 1

    for idx, sound in enumerate(word):
        score += pos_uni_freqs[idx][sound]

    return score

def get_pos_bigram_score(word, pos_bi_freqs):
    """
    Calculcates the positional bigram score of a word given a fitted 
    positional bigram model.

    word: The test word
    pos_bi_freqs: The fitted positional bigram model

    returns: The score of the word under the positional unigram model. Following
    Vitevich & Luce (2004), we add 1 to these scores.
    """
    score = 1

    for idx, sound in enumerate(word):
        if idx < len(word) - 1:
            score += pos_bi_freqs[(idx, idx + 1)][sound, word[idx + 1]]

    return score

##################
# Entry function #
##################

def run(train, test, out):
    """
    Trains all of the n-gram models on the training set, evaluates them on
    the test set, and writes the evaluation results to a file.

    train: The path to the training file.
    test: The path to the test file.
    out: The path to the output file.

    returns: None
    """
    train_token_freqs = read_tokens(train)
    test_token_freqs = read_tokens(test)
    unique_sounds = set(
        [sound for token, _ in train_token_freqs for sound in token]
    )
    sound_idx = sorted(list(unique_sounds)) + ['#']

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
