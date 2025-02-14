"""
ngram_calculator.py - Orchestration module for calculating n-gram scores.
This module loads training and test data, builds n-gram models using configuration dictionaries,
scores tokens, and writes the results.
Version: 1.0.2
"""
#run command: python -m src.ngram_calculator data/english.csv data/sample_test_data/english_test_data.csv src/output.csv


import nltk
import numpy as np
import argparse
from .io_utils import read_tokens, write_results
from .ngram_models import UnigramModel, BigramModel, WORD_BOUNDARY

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

def run_calculator(train_file, test_file, output_file):
    """
    Run the n-gram calculator on the given training and test data, and write the results.

    Parameters:
      train_file (str): Path to the training corpus file.
      test_file (str): Path to the test data file.
      output_file (str): Path to the output file with word judgments.
    """
    # Load training and test tokens using the I/O module  
    train_token_freqs = read_tokens(train_file)
    test_token_freqs = read_tokens(test_file)
      
    # Build the sound index from training data (include boundary marker)  
    unique_sounds = sorted(list(set(sound for token, _ in train_token_freqs for sound in token))) + [WORD_BOUNDARY]
      
    # Define model configurations using dictionaries  
    model_configs = [
        # Unigram non-positional (log probability)
        {"name": "uni_prob", "model": "unigram", "position": "non_positional", "joint": False, "smoothed": False, "token_weighted": False},
        {"name": "uni_prob_freq_weighted", "model": "unigram", "position": "non_positional", "joint": False, "smoothed": False, "token_weighted": True},
        {"name": "uni_prob_smoothed", "model": "unigram", "position": "non_positional", "joint": False, "smoothed": True, "token_weighted": False},
        {"name": "uni_prob_freq_weighted_smoothed", "model": "unigram", "position": "non_positional", "joint": False, "smoothed": True, "token_weighted": True},
          
        # Unigram non-positional (joint)
        {"name": "uni_joint_nonpos", "model": "unigram", "position": "non_positional", "joint": True, "smoothed": False, "token_weighted": False},
        {"name": "uni_joint_nonpos_freq_weighted", "model": "unigram", "position": "non_positional", "joint": True, "smoothed": False, "token_weighted": True},
        {"name": "uni_joint_nonpos_smoothed", "model": "unigram", "position": "non_positional", "joint": True, "smoothed": True, "token_weighted": False},
        {"name": "uni_joint_nonpos_freq_weighted_smoothed", "model": "unigram", "position": "non_positional", "joint": True, "smoothed": True, "token_weighted": True},
          
        # Unigram positional (joint)
        {"name": "uni_joint_pos", "model": "unigram", "position": "positional", "joint": True, "smoothed": False, "token_weighted": False},
        {"name": "uni_joint_pos_freq_weighted", "model": "unigram", "position": "positional", "joint": True, "smoothed": False, "token_weighted": True},
        {"name": "uni_joint_pos_smoothed", "model": "unigram", "position": "positional", "joint": True, "smoothed": True, "token_weighted": False},
        {"name": "uni_joint_pos_freq_weighted_smoothed", "model": "unigram", "position": "positional", "joint": True, "smoothed": True, "token_weighted": True},
          
        # Bigram conditional, positional with word boundaries
        {"name": "bi_cond_pos_wb", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_pos_wb_freq_weighted", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_pos_wb_smoothed", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_pos_wb_freq_weighted_smoothed", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": True, "smoothed": True, "token_weighted": True},
          
        # Bigram conditional, positional without word boundaries
        {"name": "bi_cond_pos_noWB", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_pos_noWB_freq_weighted", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_pos_noWB_smoothed", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_pos_noWB_freq_weighted_smoothed", "model": "bigram", "position": "positional", "conditional": True, "use_boundaries": False, "smoothed": True, "token_weighted": True},
          
        # Bigram conditional, non-positional with word boundaries
        {"name": "bi_cond_nonpos_wb", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_nonpos_wb_freq_weighted", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_nonpos_wb_smoothed", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_nonpos_wb_freq_weighted_smoothed", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": True, "smoothed": True, "token_weighted": True},
          
        # Bigram conditional, non-positional without word boundaries
        {"name": "bi_cond_nonpos_noWB", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_cond_nonpos_noWB_freq_weighted", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_cond_nonpos_noWB_smoothed", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_cond_nonpos_noWB_freq_weighted_smoothed", "model": "bigram", "position": "non_positional", "conditional": True, "use_boundaries": False, "smoothed": True, "token_weighted": True},
          
        # Bigram joint, positional with word boundaries
        {"name": "bi_joint_pos_wb", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_pos_wb_freq_weighted", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_pos_wb_smoothed", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_pos_wb_freq_weighted_smoothed", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": True, "smoothed": True, "token_weighted": True},
          
        # Bigram joint, positional without word boundaries
        {"name": "bi_joint_pos_noWB", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_pos_noWB_freq_weighted", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_pos_noWB_smoothed", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_pos_noWB_freq_weighted_smoothed", "model": "bigram", "position": "positional", "conditional": False, "use_boundaries": False, "smoothed": True, "token_weighted": True},
          
        # Bigram joint, non-positional with word boundaries
        {"name": "bi_joint_nonpos_wb", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": True, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_nonpos_wb_freq_weighted", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": True, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_nonpos_wb_smoothed", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": True, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_nonpos_wb_freq_weighted_smoothed", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": True, "smoothed": True, "token_weighted": True},
          
        # Bigram joint, non-positional without word boundaries
        {"name": "bi_joint_nonpos_noWB", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": False, "smoothed": False, "token_weighted": False},
        {"name": "bi_joint_nonpos_noWB_freq_weighted", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": False, "smoothed": False, "token_weighted": True},
        {"name": "bi_joint_nonpos_noWB_smoothed", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": False, "smoothed": True, "token_weighted": False},
        {"name": "bi_joint_nonpos_noWB_freq_weighted_smoothed", "model": "bigram", "position": "non_positional", "conditional": False, "use_boundaries": False, "smoothed": True, "token_weighted": True},
    ]
      
    # Create and fit models using the model classes based on the configuration dictionaries  
    models = {}
    for config in model_configs:
        if config["model"] == "unigram":
            prob_type = "joint" if config.get("joint", False) else "log"
            model = UnigramModel(
                position=config["position"],
                prob_type=prob_type,
                smoothed=config["smoothed"],
                token_weighted=config["token_weighted"]
            ).fit(train_token_freqs, unique_sounds)
        elif config["model"] == "bigram":
            prob_type = "conditional" if config.get("conditional", False) else "joint"
            model = BigramModel(
                position=config["position"],
                prob_type=prob_type,
                use_boundaries=config["use_boundaries"],
                smoothed=config["smoothed"],
                token_weighted=config["token_weighted"]
            ).fit(train_token_freqs, unique_sounds)
        models[config["name"]] = model
      
    # Score each test token using the fitted models in the order defined by model_configs  
    results = []
    for token, _ in test_token_freqs:
        row = [' '.join(token), len(token)]
        for config in model_configs:
            score = models[config["name"]].score(token, unique_sounds)
            row.append(score if not np.isinf(score) else '')
        results.append(row)
      
    # Write the results using the I/O module  
    write_results(results, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate n-gram scores for a dataset.")
    parser.add_argument('train_file', type=str, help='Path to the input corpus file.')
    parser.add_argument('test_file', type=str, help='Path to the test data file.')
    parser.add_argument('output_file', type=str, help='Path to output file with word judgments.')
    args = parser.parse_args()
    run_calculator(args.train_file, args.test_file, args.output_file)