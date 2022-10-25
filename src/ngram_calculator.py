from struct import calcsize

import csv
import nltk
import numpy as np
import matplotlib.pyplot as plt
import re

from collections import defaultdict

WORD_BOUNDARY = '#'

####################
# Helper functions #
####################

def generate_bigrams(token):
    """
    Returns a list of sound bigrams given a single word token
    """
    return nltk.ngrams([WORD_BOUNDARY] + token + [WORD_BOUNDARY], 2)

def read_tokens(dataset):
    """
    Reads in a file containing tokens and optional frequencies and converts
    it to a list of tokens and a list of token/frequency pairs.
    """
    with open(dataset, 'r') as f:
        reader = csv.reader(f)
    
        token_freqs = []

        for row in reader:
            split_token = row[0].split(' ')
            freq = float(row[1]) if len(row) == 2 else 0
            token_freqs.append([split_token, freq])

    return token_freqs

###########################
# Code for fitting models #
###########################

#############################
# Standard unigrams/bigrams #
#############################

def fit_unigrams(token_freqs, token_weighted=False):
    """
    This function takes a set of word tokens and returns a dictionary whose
    keys are unigrams and whose values are log unigram probabilities

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
    returns a count matrix of bigrams. The sound list is necessary because
    we include counts of 0 for unattested bigram combinations.

    If token_weighted is True, counts are weighted by the log frequency
    of the words they occur in.

    If smoothed is True, start with a pseudo-count of 1 for every bigram.
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

###############################
# Positional unigrams/bigrams #
###############################

def fit_positional_unigrams(token_freqs, token_weighted=False, smoothed=False):
    """
    TODO DOCUMENTATION
    """
    pos_unigram_freqs = defaultdict(lambda: defaultdict(int))

    if smoothed:
        unique_sounds = set(
            [sound for token, _ in token_freqs for sound in token]
        )
        max_idx = max([len(token) for token, _ in token_freqs])

        for i in range(max_idx):
            for sound in unique_sounds:
                pos_unigram_freqs[i][sound] = 1

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for idx, sound in enumerate(token):
            pos_unigram_freqs[idx][sound] += val

    pos_unigram_freqs = normalize_positional_counts(pos_unigram_freqs)

    return pos_unigram_freqs

def fit_positional_bigrams(token_freqs, token_weighted=False, smoothed=True):
    pos_bigram_freqs = defaultdict(lambda: defaultdict(int))

    for token, freq in token_freqs:
        val = np.log(freq) if token_weighted else 1
        for idx, sound in enumerate(token):
            if idx < len(token) - 1:
                pos_bigram_freqs[(idx, idx + 1)][(sound, token[idx + 1])] += val

    pos_bigram_tok_freqs = normalize_positional_counts(pos_bigram_tok_freqs)
    
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

###############
# Other stuff #
###############

def build_ngram_models(dataset):
    token_freqs = read_tokens(dataset)

    unique_sounds = set(
        [item for sublist in token_freqs for item, _ in sublist]
    )
    sound_idx = sorted(list(unique_sounds)) + ['#']

    # Get type-weighted probabilities
    unigram_probs = fit_unigrams(token_freqs)
    bigram_probs = fit_bigrams(token_freqs, sound_idx)

    # Get token-weighted probabilities
    unigram_token_freq, bigram_token_freq = get_token_freqs(token_freqs, sound_idx)
    unigram_token_probs, bigram_token_probs = fit_token_ngrams(unigram_token_freq, bigram_token_freq)

    return unigram_probs, bigram_probs, sound_idx, unigram_token_probs, bigram_token_probs

###########################
# Code for testing models #
###########################

def get_unigram_prob(word, unigram_probs):
    prob = 0
    for sound in word:
        # Add basic smoothing for sounds that appear in test data but
        #   not training data. Use float('-inf') because we are
        #   using log probabilities --> log(-inf) = 0
        prob += unigram_probs.get(sound, float('-inf'))

    return prob

def get_bigram_prob(word, bigram_probs, sound_idx):
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
    score = 1

    for idx, sound in enumerate(word):
        score += pos_uni_freqs[idx][sound]

    return score

def get_pos_bigram_score(word, pos_bi_freqs):
    score = 1

    for idx, sound in enumerate(word):
        if idx < len(word) - 1:
            score += pos_bi_freqs[(idx, idx + 1)][sound, word[idx + 1]]

    return score

def get_unigram_token_score(word, unigram_token_probs):
    if not unigram_token_probs:
        return "N/A"
    score = 0
    for sound in word:
        score += unigram_token_probs.get(sound, float('-inf'))
    return score

def get_bigram_token_score(word, bigram_token_probs):
    if not bigram_token_probs:
        return "N/A"
    score = 0
    expanded_word = ['#'] + word + ['#']
    for i in range(len(expanded_word) - 1):
        bigram = (expanded_word[i], expanded_word[i+1])
        score += bigram_token_probs.get(bigram, float('-inf'))
    return score

def get_pos_unigram_tok_score(word, pos_uni_tok_freqs):
    score = 1

    for idx, sound in enumerate(word):
        score += pos_uni_tok_freqs[idx][sound]

    return score

def get_pos_bigram_tok_score(word, pos_bi_tok_freqs):
    score = 1

    for idx, sound in enumerate(word):
        if idx < len(word) - 1:
            score += pos_bi_tok_freqs[(idx, idx + 1)][sound, word[idx + 1]]

    return score

def score_corpus(dataset, outfile, unigram_probs, bigram_probs, pos_uni_freqs, 
                 pos_bi_freqs, sound_idx, unigram_token_probs,
                 bigram_token_probs, pos_uni_tok_freqs, pos_bi_tok_freqs):
    token_freqs = read_tokens(dataset)

    uni_prob_list = []
    bi_prob_list = []
    pos_uni_score_list = []
    pos_bi_score_list = []
    uni_tok_prob_list = []
    bi_tok_prob_list = []
    pos_uni_tok_score_list = []
    pos_bi_tok_score_list = []

    with open(outfile, 'w') as f:
        f.write(','.join([
            'word',
            'word_len',
            'uni_prob',
            'bi_prob',
            'pos_uni_freq',
            'pos_bi_freq',
            'uni_tok_prob',
            'bi_tok_prob',
            'pos_uni_tok',
            'pos_bi_tok\n'
        ]))

        for token, _ in token_freqs:
            word_len = len(token)
            unigram_prob = get_unigram_prob(token, unigram_probs)
            bigram_prob = get_bigram_prob(token, bigram_probs, sound_idx)
            pos_uni_score = get_pos_unigram_score(token, pos_uni_freqs)
            pos_bi_score = get_pos_bigram_score(token, pos_bi_freqs)
            uni_tok_prob = get_unigram_prob(token, unigram_token_probs)
            bi_tok_prob = get_bigram_prob(token, bigram_token_probs, sound_idx)
            pos_uni_tok_score = get_pos_unigram_tok_score(token, pos_uni_tok_freqs)
            pos_bi_tok_score = get_pos_bigram_tok_score(token, pos_bi_tok_freqs)

            uni_prob_list.append(unigram_prob)
            bi_prob_list.append(bigram_prob)
            pos_uni_score_list.append(pos_uni_score)
            pos_bi_score_list.append(pos_bi_score)
            uni_tok_prob_list.append(uni_tok_prob)
            bi_tok_prob_list.append(bi_tok_prob)
            pos_uni_tok_score_list.append(pos_uni_tok_score)
            pos_bi_tok_score_list.append(pos_bi_tok_score)

            f.write('{}\n'.format(','.join([
                ' '.join(token), 
                str(word_len),
                str(unigram_prob), 
                str(bigram_prob), 
                str(pos_uni_score), 
                str(pos_bi_score),
                str(uni_tok_prob),
                str(bi_tok_prob),
                str(pos_uni_tok_score),
                str(pos_bi_tok_score)
            ])))

##################
# Entry function #
##################

def run(train, test, out):
    # TODO: Replace with individual calls
    unigram_probs, bigram_probs, sound_idx, unigram_token_probs, bigram_token_probs = build_ngram_models(train)
    # TODO: Replace with individual calls
    pos_uni_freqs, pos_bi_freqs, pos_uni_tok_freqs, pos_bi_tok_freqs = build_positional_models(train)

    score_corpus(
        test, out, 
        unigram_probs, bigram_probs, 
        pos_uni_freqs, pos_bi_freqs,
        sound_idx, unigram_token_probs,
        bigram_token_probs, pos_uni_tok_freqs,
        pos_bi_tok_freqs
    )
