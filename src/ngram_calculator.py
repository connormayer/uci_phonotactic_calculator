"""
ngram_calculator.py - Contains functions to fit and calculate n-gram probabilities (both positional & non-positional).
Version: 1.2.1
"""

import csv
import nltk
import numpy as np
from collections import defaultdict

from io_utils import read_tokens, write_results
from ngram_models import NgramModel, WORD_BOUNDARY

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

def fit_ngram_models(token_freqs, sound_idx):
    """
    Fits a suite of n-gram models using the NgramModel factory.
    Returns a dictionary of fitted model instances keyed by descriptive names.
    """
    models = {}

    # Unigram models (non-positional, log-probability)
    models['uni_prob'] = NgramModel("unigram", "non_positional", "log", smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['uni_prob_freq_weighted'] = NgramModel("unigram", "non_positional", "log", smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['uni_prob_smoothed'] = NgramModel("unigram", "non_positional", "log", smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['uni_prob_freq_weighted_smoothed'] = NgramModel("unigram", "non_positional", "log", smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Unigram models (non-positional, joint)
    models['uni_joint_nonpos'] = NgramModel("unigram", "non_positional", "joint", smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['uni_joint_nonpos_freq_weighted'] = NgramModel("unigram", "non_positional", "joint", smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['uni_joint_nonpos_smoothed'] = NgramModel("unigram", "non_positional", "joint", smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['uni_joint_nonpos_freq_weighted_smoothed'] = NgramModel("unigram", "non_positional", "joint", smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Unigram models (positional, joint only)
    models['uni_joint_pos'] = NgramModel("unigram", "positional", "joint", smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['uni_joint_pos_freq_weighted'] = NgramModel("unigram", "positional", "joint", smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['uni_joint_pos_smoothed'] = NgramModel("unigram", "positional", "joint", smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['uni_joint_pos_freq_weighted_smoothed'] = NgramModel("unigram", "positional", "joint", smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - conditional, positional with word boundaries
    models['bi_cond_pos_wb'] = NgramModel("bigram", "positional", "conditional", use_boundaries=True, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_pos_wb_freq_weighted'] = NgramModel("bigram", "positional", "conditional", use_boundaries=True, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_cond_pos_wb_smoothed'] = NgramModel("bigram", "positional", "conditional", use_boundaries=True, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_pos_wb_freq_weighted_smoothed'] = NgramModel("bigram", "positional", "conditional", use_boundaries=True, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - conditional, positional without word boundaries
    models['bi_cond_pos_noWB'] = NgramModel("bigram", "positional", "conditional", use_boundaries=False, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_pos_noWB_freq_weighted'] = NgramModel("bigram", "positional", "conditional", use_boundaries=False, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_cond_pos_noWB_smoothed'] = NgramModel("bigram", "positional", "conditional", use_boundaries=False, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_pos_noWB_freq_weighted_smoothed'] = NgramModel("bigram", "positional", "conditional", use_boundaries=False, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - conditional, non-positional with word boundaries
    models['bi_cond_nonpos_wb'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=True, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_nonpos_wb_freq_weighted'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=True, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_cond_nonpos_wb_smoothed'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=True, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_nonpos_wb_freq_weighted_smoothed'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=True, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - conditional, non-positional without word boundaries
    models['bi_cond_nonpos_noWB'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=False, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_nonpos_noWB_freq_weighted'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=False, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_cond_nonpos_noWB_smoothed'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=False, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_cond_nonpos_noWB_freq_weighted_smoothed'] = NgramModel("bigram", "non_positional", "conditional", use_boundaries=False, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - joint, positional with word boundaries
    models['bi_joint_pos_wb'] = NgramModel("bigram", "positional", "joint", use_boundaries=True, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_pos_wb_freq_weighted'] = NgramModel("bigram", "positional", "joint", use_boundaries=True, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_joint_pos_wb_smoothed'] = NgramModel("bigram", "positional", "joint", use_boundaries=True, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_pos_wb_freq_weighted_smoothed'] = NgramModel("bigram", "positional", "joint", use_boundaries=True, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - joint, positional without word boundaries
    models['bi_joint_pos_noWB'] = NgramModel("bigram", "positional", "joint", use_boundaries=False, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_pos_noWB_freq_weighted'] = NgramModel("bigram", "positional", "joint", use_boundaries=False, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_joint_pos_noWB_smoothed'] = NgramModel("bigram", "positional", "joint", use_boundaries=False, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_pos_noWB_freq_weighted_smoothed'] = NgramModel("bigram", "positional", "joint", use_boundaries=False, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - joint, non-positional with word boundaries
    models['bi_joint_nonpos_wb'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=True, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_nonpos_wb_freq_weighted'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=True, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_joint_nonpos_wb_smoothed'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=True, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_nonpos_wb_freq_weighted_smoothed'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=True, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    # Bigram models - joint, non-positional without word boundaries
    models['bi_joint_nonpos_noWB'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=False, smoothed=False, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_nonpos_noWB_freq_weighted'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=False, smoothed=False, token_weighted=True).fit(token_freqs, sound_idx)
    models['bi_joint_nonpos_noWB_smoothed'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=False, smoothed=True, token_weighted=False).fit(token_freqs, sound_idx)
    models['bi_joint_nonpos_noWB_freq_weighted_smoothed'] = NgramModel("bigram", "non_positional", "joint", use_boundaries=False, smoothed=True, token_weighted=True).fit(token_freqs, sound_idx)

    return models

def score_corpus(token_freqs, fitted_models, sound_idx):
    """
    Given a dataset and a dictionary of fitted models, returns the score for each
    word under each model. If a score is -inf, it is replaced with ''.
    """
    results = []
    # Order of models corresponding to the HEADER (after word and word_len)
    model_keys = [
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
    for token, _ in token_freqs:
        row = [' '.join(token), len(token)]
        for key in model_keys:
            model = fitted_models.get(key)
            if model is not None:
                score = model.score(token, sound_idx)
                row.append(score if not np.isinf(score) else '')
            else:
                row.append('')
        results.append(row)
    return results

def run(train, test, out):
    train_token_freqs = read_tokens(train)
    test_token_freqs = read_tokens(test)
    unique_sounds = set(sound for token, _ in train_token_freqs for sound in token)
    sound_idx = sorted(list(unique_sounds)) + [WORD_BOUNDARY]

    fitted_models = fit_ngram_models(train_token_freqs, sound_idx)
    results = score_corpus(test_token_freqs, fitted_models, sound_idx)
    write_results(results, out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate a suite of unigram/bigram scores for a data set.")
    parser.add_argument('train_file', type=str, help='Path to the input corpus file.')
    parser.add_argument('test_file', type=str, help='Path to test data file.')
    parser.add_argument('output_file', type=str, help='Path to output file with word judgements.')
    args = parser.parse_args()
    run(args.train_file, args.test_file, args.output_file)