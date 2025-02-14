"""
io_utils.py - Utility module for I/O operations: reading tokens from a CSV file and writing results.
Version: 1.0.0
"""

import csv

WORD_BOUNDARY = '#'

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
    results = [HEADER] + results
    with open(outfile, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)