"""
ngram_calculator.py - Orchestration entry point for calculating n-gram scores.
This file loads data using io_utils, creates models via ngram_models,
scores tokens, and writes results.
Version: 1.0.0
"""

import nltk
import numpy as np
from io_utils import read_tokens, write_results
from ngram_models import NgramModel, WORD_BOUNDARY

# Pre-defined header; the order corresponds to the model configuration order below.
HEADER = [
    'word',
    'word_len',
    'uni_prob',
    'uni_prob_freq_weighted',
    'uni_prob_smoothed',
    'uni_prob_freq_weighted_smoothed',
    'uni_joint_nonpos',
    'uni_joint_nonpos_freq_weighted',
    'uni_joint_nonpos_smoothed',
    'uni_joint_nonpos_freq_weighted_smoothed',
    'uni_joint_pos',
    'uni_joint_pos_freq_weighted',
    'uni_joint_pos_smoothed',
    'uni_joint_pos_freq_weighted_smoothed',
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

def run(train, test, out):
    # Load training and test tokens using the I/O module
    train_token_freqs = read_tokens(train)
    test_token_freqs = read_tokens(test)
    
    # Extract unique sounds and build the sound index (include boundary marker)
    unique_sounds = set(sound for token, _ in train_token_freqs for sound in token)
    sound_idx = sorted(list(unique_sounds)) + [WORD_BOUNDARY]

    # Define model configurations (order must match the header columns, starting after word and word_len)
    model_configs = [
        # Unigram non-positional (log probability)
        {"name": "uni_prob", "model_type": "unigram", "position": "non_positional", "prob_type": "log", "smoothed": False, "token_weighted": False},
        {"name": "uni_prob_freq_weighted", "model_type": "unigram", "position": "non_positional", "prob_type": "log", "smoothed": False, "token_weighted": True},
        {"name": "uni_prob_smoothed", "model_type": "unigram", "position": "non_positional", "prob_type": "log", "smoothed": True, "token_weighted": False},
        {"name": "uni_prob_freq_weighted_smoothed", "model_type": "unigram", "position": "non_positional", "prob_type": "log", "smoothed": True, "token_weighted": True},
        
        # Unigram non-positional (joint)
        {"name": "uni_joint_nonpos", "model_type": "unigram", "position": "non_positional", "prob_type": "joint", "smoothed": False, "token_weighted": False},
        {"name": "uni_joint_nonpos_freq_weighted", "model_type": "unigram", "position": "non_positional", "prob_type": "joint", "smoothed": False, "token_weighted": True},
        {"name": "uni_joint_nonpos_smoothed", "model_type": "unigram", "position": "non_positional", "prob_type": "joint", "smoothed": True, "token_weighted": False},
        {"name": "uni_joint_nonpos_freq_weighted_smoothed", "model_type": "unigram", "position": "non_positional", "prob_type": "joint", "smoothed": True, "token_weighted": True},
        
        # Unigram positional (joint)
        {"name": "uni_joint_pos", "model_type": "unigram", "position": "positional", "prob_type": "joint", "smoothed": False, "token_weighted": False},
        {"name": "uni_joint_pos_freq_weighted", "model_type": "unigram", "position": "positional", "prob_type": "joint", "smoothed": False, "token_weighted": True},
        {"name": "uni_joint_pos_smoothed", "model_type": "unigram", "position": "positional", "prob_type": "joint", "smoothed": True, "token_weighted": False},
        {"name": "uni_joint_pos_freq_weighted_smoothed", "model_type": "unigram", "position": "positional", "prob_type": "joint", "smoothed": True, "token_weighted": True},
        
        # Bigram conditional, positional with word boundaries
        {"name": "bi_cond_pos_wb", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_pos_wb_freq_weighted", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_pos_wb_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_pos_wb_freq_weighted_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": True, "token_weighted": True},
        
        # Bigram conditional, positional without word boundaries
        {"name": "bi_cond_pos_noWB", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_pos_noWB_freq_weighted", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_pos_noWB_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_pos_noWB_freq_weighted_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": True, "token_weighted": True},
        
        # Bigram conditional, non-positional with word boundaries
        {"name": "bi_cond_nonpos_wb", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_nonpos_wb_freq_weighted", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_nonpos_wb_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_nonpos_wb_freq_weighted_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": True, "smoothed": True, "token_weighted": True},
        
        # Bigram conditional, non-positional without word boundaries
        {"name": "bi_cond_nonpos_noWB", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_nonpos_noWB_freq_weighted", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_nonpos_noWB_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_nonpos_noWB_freq_weighted_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "conditional", "use_boundaries": False, "smoothed": True, "token_weighted": True},
        
        # Bigram joint, positional with word boundaries
        {"name": "bi_joint_pos_wb", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_pos_wb_freq_weighted", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_pos_wb_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_pos_wb_freq_weighted_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": True, "smoothed": True, "token_weighted": True},
        
        # Bigram joint, positional without word boundaries
        {"name": "bi_joint_pos_noWB", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_pos_noWB_freq_weighted", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_pos_noWB_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_pos_noWB_freq_weighted_smoothed", "model_type": "bigram", "position": "positional", "prob_type": "joint", "use_boundaries": False, "smoothed": True, "token_weighted": True},
        
        # Bigram joint, non-positional with word boundaries
        {"name": "bi_joint_nonpos_wb", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_nonpos_wb_freq_weighted", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_nonpos_wb_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_nonpos_wb_freq_weighted_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": True, "smoothed": True, "token_weighted": True},
        
        # Bigram joint, non-positional without word boundaries
        {"name": "bi_joint_nonpos_noWB", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_nonpos_noWB_freq_weighted", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_nonpos_noWB_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_nonpos_noWB_freq_weighted_smoothed", "model_type": "bigram", "position": "non_positional", "prob_type": "joint", "use_boundaries": False, "smoothed": True, "token_weighted": True},
    ]

    # Create and fit models using the NgramModel factory
    models = {}
    for config in model_configs:
        model = NgramModel(
            model_type=config["model_type"],
            position=config["position"],
            prob_type=config["prob_type"],
            use_boundaries=config.get("use_boundaries", False),
            smoothed=config.get("smoothed", False),
            token_weighted=config.get("token_weighted", False)
        ).fit(train_token_freqs, sound_idx)
        models[config["name"]] = model

    # Score each test token for all models and compile results
    results = []
    for token, _ in test_token_freqs:
        row = [' '.join(token), len(token)]
        for config in model_configs:
            score = models[config["name"]].score(token, sound_idx)
            row.append(score if not np.isinf(score) else '')
        results.append(row)

    # Write the results using the I/O module
    write_results([HEADER] + results, out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate a suite of n-gram scores for a dataset.")
    parser.add_argument('train_file', type=str, help='Path to the input corpus file.')
    parser.add_argument('test_file', type=str, help='Path to the test data file.')
    parser.add_argument('output_file', type=str, help='Path to output file with word judgments.')
    args = parser.parse_args()
    run(args.train_file, args.test_file, args.output_file)