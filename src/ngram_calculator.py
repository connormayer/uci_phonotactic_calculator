"""
src/ngram_calculator.py - Orchestration module for calculating n-gram scores.
This module loads training and test data, builds n-gram models using various configurations,
scores test tokens, and writes the results using a dynamically generated header.

"""

import argparse
from .io_utils import read_tokens, write_results
from .ngram_models import UnigramModel, BigramModel, WORD_BOUNDARY
from .model_configs import get_model_configs


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
    sound_index = sorted(
        list(set(sound for token, _ in train_token_freqs for sound in token))
    ) + [WORD_BOUNDARY]

    # Get model configurations from the model_configs module
    model_configs = get_model_configs()

    # Create and fit models using the model classes based on the configuration dictionaries
    models = {}
    for config in model_configs:
        # breakpoint()
        if config["model"] == "unigram":
            model = UnigramModel(
                position=config["position"],
                smoothed=config["smoothed"],
                token_weighted=config["token_weighted"],
                aggregation=config.get("aggregation", "sum"),
            ).fit(train_token_freqs)
        elif config["model"] == "bigram":
            prob_type = "conditional" if config.get("conditional", False) else "joint"
            model = BigramModel(
                position=config["position"],
                prob_type=prob_type,
                use_boundaries=config["use_boundaries"],
                smoothed=config["smoothed"],
                token_weighted=config["token_weighted"],
                aggregation=config.get("aggregation", "sum"),
            ).fit(train_token_freqs, sound_index)
        models[config["name"]] = model

    # Score each test token using the fitted models in the order defined by model_configs
    results = []
    for token, _ in test_token_freqs:
        row = [" ".join(token), len(token)]
        for config in model_configs:
            score = models[config["name"]].score(token, sound_index)
            row.append(score)
        results.append(row)

    # Write the results using the I/O module (dynamic header is generated internally)
    write_results(results, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate n-gram scores for a dataset."
    )
    parser.add_argument("train_file", type=str, help="Path to the input corpus file.")
    parser.add_argument("test_file", type=str, help="Path to the test data file.")
    parser.add_argument(
        "output_file", type=str, help="Path to output file with word judgments."
    )
    args = parser.parse_args()
    run_calculator(args.train_file, args.test_file, args.output_file)
